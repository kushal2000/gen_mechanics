"""Toy benchmark: peak GPU memory and time of ONE PPO-style forward+backward, joint transformer vs MLP, and the
standard ways to cut a transformer's activation memory.

Shapes are the real training ones: a 114688-sample minibatch, 21 tokens per sample (20 Wuji joints + 1 global),
the production block (coevolution/networks/joint_transformer.py _EncoderLayer: d_model 64, 1 head, ff x2, 4 layers,
explicit attention), fp16 autocast + GradScaler as rl_games' mixed_precision does. The MLP is SAPG's
[1024, 1024, 512, 512] on the same observation flattened. The loss is a stand-in (Gaussian log-likelihood + value
MSE): the backward pass is what costs memory, not which loss feeds it.

Variants (each run in a fresh state; peak = torch.cuda.max_memory_allocated during one optimizer step):
  eager                    the block as written, no compile
  compile                  Module.compile, as production (compile_net=true)
  ckpt                     activation checkpointing per block (torch.utils.checkpoint, use_reentrant=False)
  micro<k>                 the minibatch split into k micro-batches, gradients accumulated (same update)
  compile+micro<k>, compile+ckpt, bf16, sdpa (F.scaled_dot_product_attention)
Also checks that micro-batching gives the same gradients as the full batch.

    .venv_isaacsim/bin/python experiments/08oct_multi_embodiment_evolution/analysis/toy_transformer_memory.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from coevolution.networks.joint_transformer import _EncoderLayer  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results/toy_transformer_memory.json"


class SDPALayer(_EncoderLayer):
    """The production block with F.scaled_dot_product_attention in place of the explicit softmax."""

    def forward(self, x, key_mask=None):
        b, t, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.ln_ff(x))


class ToyJointTransformer(nn.Module):
    """Token embedding -> L blocks -> per-joint action head + pooled value head (production widths)."""

    def __init__(self, token_dim, global_dim, d=64, layers=4, heads=1, ff=2, n_act=20, sdpa=False, ckpt=False):
        super().__init__()
        Layer = SDPALayer if sdpa else _EncoderLayer
        self.embed = nn.Linear(token_dim, d)
        self.gembed = nn.Linear(global_dim, d)
        self.blocks = nn.ModuleList(Layer(d, heads, ff, 0.0) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.mu = nn.Sequential(nn.Linear(d, 64), nn.ELU(), nn.Linear(64, 1))
        self.value = nn.Sequential(nn.Linear(2 * d + global_dim, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                   nn.Linear(256, 1))
        self.ckpt, self.n_act = ckpt, n_act

    def forward(self, tok, glob):
        x = torch.cat([self.embed(tok), self.gembed(glob).unsqueeze(1)], dim=1)
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False) if self.ckpt else blk(x)
        x = self.norm(x)
        mu = self.mu(x[:, : self.n_act]).squeeze(-1)
        v = self.value(torch.cat([x[:, : self.n_act].mean(1), x[:, -1], glob], dim=-1))
        return mu, v


class ToyMLP(nn.Module):
    def __init__(self, in_dim, n_act=20):
        super().__init__()
        units = [1024, 1024, 512, 512]
        layers, d = [], in_dim
        for u in units:
            layers += [nn.Linear(d, u), nn.ELU()]
            d = u
        self.trunk = nn.Sequential(*layers)
        self.mu, self.value = nn.Linear(d, n_act), nn.Linear(d, 1)

    def forward(self, tok, glob):
        h = self.trunk(torch.cat([tok.flatten(1), glob], dim=-1))
        return self.mu(h), self.value(h)


def loss_fn(mu, v, act, ret):
    return 0.5 * ((mu.float() - act) ** 2).sum(-1).mean() + 0.5 * ((v.float().squeeze(-1) - ret) ** 2).mean()


def step(model, opt, scaler, data, micro, dtype):
    tok, glob, act, ret = data
    opt.zero_grad(set_to_none=True)
    for idx in torch.arange(tok.shape[0], device=tok.device).chunk(micro):
        with torch.autocast("cuda", dtype=dtype):
            mu, v = model(tok[idx], glob[idx])
            loss = loss_fn(mu, v, act[idx], ret[idx]) * (idx.numel() / tok.shape[0])
        scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


def bench(name, make, data, micro=1, dtype=torch.float16, compile_=False, iters=8):
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    model = make().cuda()
    if compile_:
        model.compile()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    for _ in range(3):                                   # warm-up (and compile)
        step(model, opt, scaler, data, micro, dtype)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for _ in range(iters):
        step(model, opt, scaler, data, micro, dtype)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated()
    res = {"variant": name, "peak_gb": peak / 2**30, "activation_gb": (peak - base) / 2**30, "ms_per_step": ms,
           "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    print(f"{name:24s} peak {res['peak_gb']:6.2f} GB  (activations {res['activation_gb']:6.2f} GB)  "
          f"{ms:8.1f} ms/step  {res['params_m']:.2f} M params", flush=True)
    del model, opt
    return res


def grad_check(make, data):
    """Micro-batching must give the full batch's gradient (up to rounding)."""
    grads = []
    for micro in (1, 4):
        torch.manual_seed(0)
        m = make().cuda()
        tok, glob, act, ret = data
        m.zero_grad()
        for idx in torch.arange(tok.shape[0], device=tok.device).chunk(micro):
            mu, v = m(tok[idx], glob[idx])                 # fp32 for an exact comparison
            (loss_fn(mu, v, act[idx], ret[idx]) * (idx.numel() / tok.shape[0])).backward()
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    return float((grads[0] - grads[1]).abs().max() / grads[0].abs().max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=114688)
    ap.add_argument("--tokens", type=int, default=20, help="joint tokens (+1 global)")
    ap.add_argument("--token-dim", type=int, default=24)
    ap.add_argument("--global-dim", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="fewer variants")
    args = ap.parse_args()
    dev = "cuda"
    B, J = args.batch, args.tokens
    data = (torch.randn(B, J, args.token_dim, device=dev), torch.randn(B, args.global_dim, device=dev),
            torch.randn(B, J, device=dev), torch.randn(B, device=dev))
    print(f"[toy] {torch.cuda.get_device_name()}  batch {B}  tokens {J}+1  torch {torch.__version__}", flush=True)

    T = lambda **kw: (lambda: ToyJointTransformer(args.token_dim, args.global_dim, n_act=J, **kw))
    M = lambda: ToyMLP(J * args.token_dim + args.global_dim, n_act=J)
    runs = [("mlp", M, {}), ("mlp compile", M, {"compile_": True}),
            ("tf eager", T(), {}), ("tf compile (prod)", T(), {"compile_": True}),
            ("tf ckpt", T(ckpt=True), {}), ("tf compile+ckpt", T(ckpt=True), {"compile_": True}),
            ("tf micro2", T(), {"micro": 2}), ("tf micro4", T(), {"micro": 4}), ("tf micro8", T(), {"micro": 8}),
            ("tf compile+micro4", T(), {"compile_": True, "micro": 4}),
            ("tf bf16 compile", T(), {"compile_": True, "dtype": torch.bfloat16}),
            ("tf sdpa", T(sdpa=True), {}), ("tf sdpa compile", T(sdpa=True), {"compile_": True})]
    if args.quick:
        runs = [r for r in runs if r[0] in ("mlp", "tf eager", "tf compile (prod)", "tf ckpt", "tf micro4")]
    results = []
    for name, make, kw in runs:
        try:
            results.append(bench(name, make, data, **kw))
        except torch.cuda.OutOfMemoryError:
            print(f"{name:24s} OUT OF MEMORY", flush=True)
            results.append({"variant": name, "oom": True})
            torch.cuda.empty_cache()
    small = tuple(t[: B // 8] for t in data)
    err = grad_check(T(), small)
    print(f"[toy] micro-batching (4) vs full batch: max relative gradient difference {err:.2e} (fp32)", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"device": torch.cuda.get_device_name(), "batch": B, "tokens": J + 1,
                               "results": results, "micro_grad_rel_err": err}, indent=1))
    print(f"[toy] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
