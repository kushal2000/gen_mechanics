"""Black-box sanity checks on the joint transformer's train step.

Deliberately does NOT trust the repo's implementation. Each check has an
external reference -- a from-scratch network, a hardware roofline, or a
scaling law -- so a bug shows up as a disagreement with something that was
derived independently.

1. INDEPENDENT REIMPLEMENTATION. A 22-token encoder written from scratch in
   this file, same shapes and same parameter count, nothing imported from
   coevolution. If the repo network is materially slower than a naive
   from-scratch equivalent, the cost is a bug in the repo network. If they
   match, the cost is intrinsic to the shape.

2. ROOFLINE. What the hardware says this shape must cost, from FLOPs and from
   activation bytes. Being far ABOVE both ceilings means time is going
   somewhere neither compute nor bandwidth explains.

3. HIDDEN SYNCHRONIZATION. torch.cuda.set_sync_debug_mode catches any
   host-device sync (a .item(), .cpu(), a data-dependent shape) hiding in the
   forward or backward -- the classic silent throughput killer in a 24k-env
   rollout.

4. BATCH SCALING. ms/sample must flatten once the GPU is saturated. A curve
   that keeps falling means the batch is too small; one that RISES means
   something superlinear (recompilation, allocator thrash).

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/smoke_transformer_sanity.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_policy_train_step import (  # noqa: E402
    make_transformer, stub_kit_free_packages,
)

stub_kit_free_packages()

# RTX 6000 Ada. Used only as a yardstick; a run on other hardware should pass
# --peak_tflops / --peak_bw to match.
PEAK_TFLOPS = 91.1      # fp16 with fp32 accumulate
PEAK_BW_GBS = 960.0


class ReferenceNet(nn.Module):
    """A 22-token, 1-layer encoder written here, from nothing.

    Same tensor shapes as JointTransformerNet L1 so the comparison is fair:
    gather 22x32 tokens out of a 778-wide observation, project to d_model,
    append one global token, one pre-LN block, then the same three heads.
    """

    def __init__(self, obs_dim=778, n_tokens=22, token_dim=32, global_dim=74,
                 d_model=128, n_heads=4, ff_mult=4, n_arm=7):
        super().__init__()
        self.n_heads = n_heads
        self.register_buffer(
            "gather",
            torch.arange(n_tokens * token_dim).reshape(n_tokens, token_dim),
        )
        self.register_buffer("gidx", torch.arange(global_dim) + n_tokens * token_dim)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                                nn.Linear(ff_mult * d_model, d_model))
        self.ln_out = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, 1)
        self.arm_head = nn.Sequential(
            nn.Linear(d_model + global_dim, 1024), nn.ELU(),
            nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, n_arm))
        self.value_head = nn.Sequential(
            nn.Linear(2 * d_model + global_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 1))

    def forward(self, obs):
        tokens = self.token_proj(obs[:, self.gather])
        glob = obs[:, self.gidx]
        x = torch.cat([tokens, self.global_proj(glob).unsqueeze(1)], dim=1)
        b, t, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v))
        attn = ((q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)).softmax(-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        x = x + self.ff(self.ln2(x))
        x = self.ln_out(x)
        joints, g = x[:, :-1], x[:, -1]
        mu_hand = self.mu_head(joints).squeeze(-1)
        mu_arm = self.arm_head(torch.cat([g, glob], dim=-1))
        value = self.value_head(torch.cat([joints.mean(1), g, glob], dim=-1))
        return torch.cat([mu_arm, mu_hand], dim=-1), value


def train_step_ms(step, iters, warmup, device):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize(device)
    return (1e3 * (time.perf_counter() - start) / iters,
            torch.cuda.max_memory_allocated(device) / 2**20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--peak_tflops", type=float, default=PEAK_TFLOPS)
    parser.add_argument("--peak_bw", type=float, default=PEAK_BW_GBS)
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    b = args.batch_size
    obs = torch.randn(b, 778, device=device)

    repo = make_transformer(1).to(device).train()
    ref = ReferenceNet().to(device).train()
    repo_params = sum(p.numel() for p in repo.parameters())
    ref_params = sum(p.numel() for p in ref.parameters())

    def make_step(net, call):
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)

        def step():
            opt.zero_grad(set_to_none=True)
            with autocast:
                out = call(net)
                loss = sum(t.square().mean() for t in out
                           if isinstance(t, torch.Tensor))
            loss.backward()
            opt.step()
        return step

    repo_step = make_step(repo, lambda n: n({"obs": obs}))
    ref_step = make_step(ref, lambda n: n(obs))

    print("=== 1. repo network vs an independent from-scratch equivalent ===",
          flush=True)
    repo_ms, repo_mem = train_step_ms(repo_step, args.iters, args.warmup, device)
    ref_ms, ref_mem = train_step_ms(ref_step, args.iters, args.warmup, device)
    print(f"  repo  JointTransformerNet L1  {repo_ms:8.2f} ms  "
          f"{repo_params:>9,} params  {repo_mem:6.0f} MiB", flush=True)
    print(f"  ref   written here, no imports {ref_ms:8.2f} ms  "
          f"{ref_params:>9,} params  {ref_mem:6.0f} MiB", flush=True)
    ratio = repo_ms / max(ref_ms, 1e-9)
    verdict = "no repo-side overhead" if ratio < 1.25 else "REPO IS SLOWER"
    print(f"  -> repo / reference = {ratio:.2f}x  ({verdict})", flush=True)

    print("\n=== 2. roofline for this shape ===", flush=True)
    macs_per_env = 5_659_392          # measured by profile_arch_efficiency
    act_bytes_per_env = 190_282       # measured, fp16 activations
    flop_ms = (macs_per_env * b * 2 * 3) / (args.peak_tflops * 1e12) * 1e3
    # fwd writes activations, bwd reads them and writes grads: ~3 passes.
    bw_ms = (act_bytes_per_env * b * 3) / (args.peak_bw * 1e9) * 1e3
    print(f"  compute-bound floor  {flop_ms:8.2f} ms  "
          f"(at {args.peak_tflops:.0f} TFLOP/s)", flush=True)
    print(f"  bandwidth-bound floor{bw_ms:8.2f} ms  "
          f"(at {args.peak_bw:.0f} GB/s)", flush=True)
    print(f"  measured             {repo_ms:8.2f} ms  "
          f"= {repo_ms / max(flop_ms, 1e-9):.1f}x compute floor, "
          f"{repo_ms / max(bw_ms, 1e-9):.1f}x bandwidth floor", flush=True)
    print("  a shape this narrow is expected to sit near the bandwidth floor;\n"
          "  far above BOTH would mean time is going somewhere else entirely.",
          flush=True)

    print("\n=== 3. hidden host-device synchronization in the train step ===",
          flush=True)
    torch.cuda.synchronize(device)
    caught = []
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            repo_step()
            torch.cuda.synchronize(device)
        caught = [str(w.message)[:110] for w in records]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    if caught:
        print(f"  {len(caught)} synchronizing call(s) inside one train step:",
              flush=True)
        for line in caught[:8]:
            print(f"    {line}", flush=True)
    else:
        print("  none -- the train step never blocks on the host", flush=True)

    print("\n=== 4. batch scaling (ms per 1k samples must flatten) ===",
          flush=True)
    for size in (2048, 4096, 8192, 16384, 32768):
        o = torch.randn(size, 778, device=device)
        net = make_transformer(1).to(device).train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-4)

        def step(net=net, opt=opt, o=o):
            opt.zero_grad(set_to_none=True)
            with autocast:
                out = net({"obs": o})
                loss = sum(t.square().mean() for t in out
                           if isinstance(t, torch.Tensor))
            loss.backward()
            opt.step()

        ms, _ = train_step_ms(step, max(6, args.iters // 2), args.warmup, device)
        print(f"  batch {size:>6,}  {ms:8.2f} ms  "
              f"{1e3 * ms / size:7.3f} ms/1k samples", flush=True)
        del net, opt, o
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
