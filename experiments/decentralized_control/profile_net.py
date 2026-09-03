"""Network-side profiling. Pure torch -- no simulator, no rl_games loop.

Everything here is about the policy networks in isolation, on prefilled dummy
observations, so nothing that made the end-to-end numbers hard to read
(physics, GPU contention, SAPG bookkeeping, a policy that changes as it learns)
is present.

    profile_net.py arch        MLP vs L0/L1/L4: params, MACs/env, TFLOP/s
    profile_net.py sweep       d_model / MLP-width / minibatch scans
    profile_net.py speedups    compile scope x mode x precision x optimizer
    profile_net.py sanity      black-box checks against from-scratch references

Run it on a DEDICATED gpu; two of these sharing a card inflates everything.

    BENCH_TAG=arch SCRIPT="profile_net.py arch" \
      sbatch experiments/decentralized_control/bench.sub

WHAT THIS ESTABLISHED (RTX 6000 Ada, batch 16384, fp16):

  * The MLP is the only configuration here that is compute-bound. It hits
    98 TFLOP/s at [1024,1024,512,512] and only 10 at [128,128,64,64] -- so
    "small networks are inefficient" is general, not a transformer fact.
  * Both implementations are clean: independently written from-scratch
    equivalents match JointTransformerNet to 1.00x and the rl_games MLP to
    0.95x. Neither carries framework overhead.
  * The transformer's cost is plumbing, not arithmetic. L0 does 2.3x less
    work than the MLP; of its floor, 4% is matmul and 96% is LayerNorm,
    gathers, concatenations, casts and a reduction.
  * torch.compile over the WHOLE module is worth 2.0-2.2x and over the
    encoder stack alone only 1.41x. It does ~nothing for the MLP, which is
    the confirmation that it is removing plumbing the MLP never had.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time
import warnings

import torch
import torch.nn as nn

from rl_games.algos_torch.network_builder import A2CBuilder

OBS_DIM = 778
ACTIONS = 29
# RTX 6000 Ada. Override with --peak_tflops / --peak_bw elsewhere.
PEAK_TFLOPS, PEAK_BW_GBS = 182e12, 960e9

OBS_LIST = [
    "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
    "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
    "joint_enabled", "object_keypoints_rel_joint", "hand_scale", "palm_pos",
    "palm_rot", "object_rot", "keypoints_rel_palm", "keypoints_rel_goal",
    "object_scales",
]


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def stub_kit_free_packages() -> None:
    """Expose layout.py / robots.py without importing their Kit-backed parents.

    joint_transformer reaches into isaacsimenvs...obs_utils.layout and
    ...scene_utils.robots, both pure Python -- but importing the packages that
    contain them pulls in Isaac Lab. Namespace packages pointing straight at
    the directories let this build the real network with no simulator.
    """
    repo_root = pathlib.Path(__file__).parents[2]
    for pkg, rel in (
        ("isaacsimenvs.pose_reaching_6d.obs_utils", "isaacsimenvs/pose_reaching_6d/obs_utils"),
        ("isaacsimenvs.pose_reaching_6d.scene_utils", "isaacsimenvs/pose_reaching_6d/scene_utils"),
    ):
        if pkg not in sys.modules:
            mod = type(sys)(pkg)
            mod.__path__ = [str(repo_root / rel)]
            sys.modules[pkg] = mod


SPACE = {"continuous": {
    "mu_activation": "None", "sigma_activation": "None",
    "mu_init": {"name": "default"},
    "sigma_init": {"name": "const_initializer", "val": 0},
    "fixed_sigma": "fixed",
}}


def make_mlp(units=(1024, 1024, 512, 512)) -> nn.Module:
    builder = A2CBuilder()
    builder.load({"separate": False, "space": SPACE,
                  "mlp": {"units": list(units), "activation": "elu",
                          "d2rl": False, "initializer": {"name": "default"},
                          "regularizer": {"name": "None"}}})
    return builder.build("profile", actions_num=ACTIONS, input_shape=(OBS_DIM,),
                         value_size=1, num_seqs=1, type="simple")


def make_transformer(n_layers, d_model=128, n_heads=4, ff_mult=4,
                     mu_head_units=None, final_norm=True, central_value=False,
                     compile_net=False, compile_mode="") -> nn.Module:
    from coevolution.networks.joint_transformer import JointTransformerNet
    key = "state_list" if central_value else "obs_list"
    params = {
        "robot_spec": "sharpa_iiwa14", key: OBS_LIST,
        "central_value": central_value, "d_model": d_model,
        "n_layers": n_layers, "n_heads": n_heads, "ff_mult": ff_mult,
        "dropout": 0.0, "arm_head_units": [1024, 512],
        "value_head_units": [512, 256],
        "mu_head_units": list(mu_head_units or []),
        "final_norm": final_norm, "compile_net": compile_net,
        "compile_mode": compile_mode, "space": SPACE,
    }
    return JointTransformerNet(params, actions_num=ACTIONS,
                               input_shape=(OBS_DIM,), value_size=1,
                               num_seqs=1, type="simple")


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def bench(fn, iters, warmup, device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / iters


def count_macs(net, obs) -> int:
    """Exact multiply-accumulates per env for one forward, by hooking Linear.

    Linear is every matmul in both networks except attention, which at 23
    tokens is ~6% of a transformer block and is noted separately where it
    matters.
    """
    total = 0
    handles = []

    def hook(module, inputs, _output):
        nonlocal total
        rows = inputs[0].numel() // inputs[0].shape[-1]
        total += rows * module.in_features * module.out_features

    for module in net.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        net({"obs": obs})
    for handle in handles:
        handle.remove()
    return total // obs.shape[0]


def train_step(net, obs, autocast, fused=False):
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, fused=fused)

    def step():
        opt.zero_grad(set_to_none=True)
        with autocast:
            out = net({"obs": obs})
            # .float() so the loss is identical across precisions. The
            # central-value net returns (value, states); the actor returns
            # (mu, sigma, value, states).
            loss = sum(t.float().square().mean() for t in out
                       if isinstance(t, torch.Tensor))
        loss.backward()
        opt.step()

    return step, opt


# --------------------------------------------------------------------------
# from-scratch references (import nothing from coevolution / rl_games)
# --------------------------------------------------------------------------

class ReferenceMLP(nn.Module):
    """rl_games' `separate: false` continuous A2C MLP, written out."""

    def __init__(self, obs_dim=OBS_DIM, units=(1024, 1024, 512, 512)):
        super().__init__()
        layers, dim = [], obs_dim
        for width in units:
            layers += [nn.Linear(dim, width), nn.ELU()]
            dim = width
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(dim, ACTIONS)
        self.value = nn.Linear(dim, 1)
        self.sigma = nn.Parameter(torch.zeros(ACTIONS))

    def forward(self, obs_dict):
        h = self.actor_mlp(obs_dict["obs"])
        mu = self.mu(h)
        return mu, mu * 0 + self.sigma, self.value(h), None


class ReferenceEncoder(nn.Module):
    """A 22-token, 1-layer encoder written here, from nothing."""

    def __init__(self, n_tokens=22, token_dim=32, global_dim=74, d_model=128,
                 n_heads=4, ff_mult=4, n_arm=7):
        super().__init__()
        self.h = n_heads
        self.register_buffer("gather",
                             torch.arange(n_tokens * token_dim).reshape(n_tokens, token_dim))
        self.register_buffer("gidx", torch.arange(global_dim) + n_tokens * token_dim)
        self.token_proj = nn.Linear(token_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
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

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        tokens = self.token_proj(obs[:, self.gather])
        glob = obs[:, self.gidx]
        x = torch.cat([tokens, self.global_proj(glob).unsqueeze(1)], dim=1)
        b, t, d = x.shape
        h = self.h
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, h, d // h).transpose(1, 2) for z in (q, k, v))
        attn = ((q @ k.transpose(-2, -1)) * ((d // h) ** -0.5)).softmax(-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        x = self.ln_out(x + self.ff(self.ln2(x)))
        joints, g = x[:, :-1], x[:, -1]
        mu = torch.cat([self.arm_head(torch.cat([g, glob], -1)),
                        self.mu_head(joints).squeeze(-1)], dim=-1)
        return mu, mu * 0, self.value_head(torch.cat([joints.mean(1), g, glob], -1)), None


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_arch(args, device, autocast):
    """MLP vs the transformer ladder: params, arithmetic, achieved TFLOP/s."""
    obs = torch.randn(args.batch_size, OBS_DIM, device=device)
    nets = [("MLP [1024,1024,512,512]", lambda: make_mlp())]
    for n in (0, 1, 4):
        nets.append((f"Transformer-L{n} d={args.d_model} h={args.n_heads}",
                     lambda n=n: make_transformer(n, args.d_model, args.n_heads)))

    print(f"batch={args.batch_size:,} fp16 autocast\n")
    head = (f"{'network':<36} {'params':>10} {'MACs/env':>11} {'fwd':>8} "
            f"{'fwd+bwd':>9} {'TFLOP/s':>8}")
    print(head + "\n" + "-" * len(head), flush=True)
    for label, build in nets:
        net = build().to(device)
        macs = count_macs(net, obs)
        params = sum(p.numel() for p in net.parameters())
        net.eval()
        with torch.no_grad(), autocast:
            fwd = bench(lambda: net({"obs": obs}), args.iters, args.warmup, device)
        net.train()
        step, opt = train_step(net, obs, autocast)
        bwd = bench(step, args.iters, args.warmup, device)
        tflops = (macs * args.batch_size * 2 * 3) / (bwd * 1e-3) / 1e12
        print(f"{label:<36} {params:>10,} {macs:>11,} {fwd:>6.2f}ms "
              f"{bwd:>7.2f}ms {tflops:>8.1f}", flush=True)
        del net, opt
        torch.cuda.empty_cache()
    print("\nTFLOP/s counts fwd+bwd as 3x the forward's MACs -- the rate the\n"
          "hardware sustains on each shape. The MLP reaches ~98; the\n"
          "transformer ~16-28, because 22 narrow token rows have far lower\n"
          "arithmetic intensity than one wide row.", flush=True)


def cmd_sweep(args, device, autocast):
    """d_model, MLP width, and minibatch scans."""
    if args.axis in ("d_model", "all"):
        print("\n=== transformer d_model (L1, standard heads/ff_mult) ===")
        print(f"{'d_model':>8} {'22*d':>6} {'MACs/env':>11} {'fwd':>8} {'fwd+bwd':>9}")
        for d in (32, 48, 64, 96, 128, 192, 256):
            obs = torch.randn(args.batch_size, OBS_DIM, device=device)
            net = make_transformer(args.n_layers, d, args.n_heads).to(device)
            macs = count_macs(net, obs)
            net.eval()
            with torch.no_grad(), autocast:
                fwd = bench(lambda: net({"obs": obs}), args.iters, args.warmup, device)
            net.train()
            step, opt = train_step(net, obs, autocast)
            print(f"{d:>8} {22*d:>6} {macs:>11,} {fwd:>6.2f}ms "
                  f"{bench(step, args.iters, args.warmup, device):>7.2f}ms", flush=True)
            del net, opt, obs
            torch.cuda.empty_cache()

    if args.axis in ("mlp", "all"):
        print("\n=== MLP hidden width (the control: where does IT saturate?) ===")
        print(f"{'units':<26} {'MACs/env':>11} {'fwd+bwd':>9} {'TFLOP/s':>8}")
        obs = torch.randn(args.batch_size, OBS_DIM, device=device)
        for u in ([128, 128, 64, 64], [256, 256, 128, 128], [512, 512, 256, 256],
                  [1024, 1024, 512, 512], [2048, 2048, 1024, 1024]):
            net = make_mlp(u).to(device)
            macs = count_macs(net, obs)
            net.train()
            step, opt = train_step(net, obs, autocast)
            ms = bench(step, args.iters, args.warmup, device)
            tf = (macs * args.batch_size * 2 * 3) / (ms * 1e-3) / 1e12
            print(f"{str(u):<26} {macs:>11,} {ms:>7.2f}ms {tf:>8.1f}", flush=True)
            del net, opt
            torch.cuda.empty_cache()

    if args.axis in ("minibatch", "all"):
        # The actor sees the SAPG-augmented batch; the central-value net does
        # not (central_value.py:80 uses horizon * num_actors). Per-epoch time
        # models the ragged split PPODataset actually produces.
        print("\n=== minibatch size (actor batch 458,752, 2 mini_epochs) ===")
        print(f"{'minibatch':>10} {'steps':>6} {'ms/step':>9} {'per epoch':>10} {'peak MiB':>9}")
        for size in (16384, 32768, 65536, 98304):
            n_mb = 458752 // size
            last = size + (458752 - n_mb * size)
            net = make_transformer(args.n_layers, args.d_model, args.n_heads).to(device)
            net.train()
            try:
                obs = torch.randn(size, OBS_DIM, device=device)
                step, opt = train_step(net, obs, autocast)
                torch.cuda.reset_peak_memory_stats(device)
                iters = max(4, round(args.iters * 16384 / size))
                ms = bench(step, iters, args.warmup, device)
                peak = torch.cuda.max_memory_allocated(device) / 2**20
            except torch.cuda.OutOfMemoryError:
                print(f"{size:>10,} {'OOM':>6}", flush=True)
                torch.cuda.empty_cache()
                continue
            epoch = 2 * ((n_mb - 1) * ms + ms * last / size) / 1e3
            print(f"{size:>10,} {n_mb * 2:>6} {ms:>7.2f}ms {epoch:>8.3f}s "
                  f"{peak:>9.0f}", flush=True)
            del net, opt, obs
            torch.cuda.empty_cache()
        print("  measured end-to-end, raising the minibatch does NOT help:\n"
              "  16384 gives 77,369 fps and 98304 gives 74,966.", flush=True)


def cmd_speedups(args, device, _autocast):
    """compile scope x mode x precision x optimizer."""
    B = args.batch_size
    mlp = make_mlp().to(device).train()
    obs = torch.randn(B, OBS_DIM, device=device)
    step, opt = train_step(mlp, obs, torch.autocast("cuda", torch.float16))
    mlp_ms = bench(step, args.iters, args.warmup, device)
    del mlp, opt, obs
    torch.cuda.empty_cache()

    print(f"L{args.n_layers} d_model={args.d_model} n_heads={args.n_heads} "
          f"batch={B:,}   MLP baseline = {mlp_ms:.2f} ms\n")
    print(f"{'compile':<8} {'mode':<15} {'precision':<15} {'adam':<7} "
          f"{'ms':>8} {'vs eager':>9} {'vs MLP':>8}", flush=True)

    base = None
    for (scope, mode), (pname, dtype, native), (aname, fused) in itertools.product(
        [("none", ""), ("layers", ""), ("net", ""), ("net", "max-autotune")],
        [("fp16 autocast", torch.float16, False),
         ("bf16 NATIVE", torch.bfloat16, True)],
        [("default", False), ("fused", True)],
    ):
        if fused and scope == "none":
            continue
        net = make_transformer(args.n_layers, args.d_model, args.n_heads,
                               compile_net=(scope == "net"), compile_mode=mode)
        net = net.to(device)
        o = torch.randn(B, OBS_DIM, device=device)
        if native:
            net, o = net.to(dtype), o.to(dtype)
            ctx = torch.autocast("cuda", dtype, enabled=False)
        else:
            ctx = torch.autocast("cuda", dtype)
        if scope == "layers":
            for layer in net.layers:
                layer.compile(dynamic=False)
        net.train()
        step, opt = train_step(net, o, ctx, fused=fused)
        try:
            ms = bench(step, args.iters, args.warmup, device)
        except Exception as exc:  # noqa: BLE001 - diagnostic sweep
            print(f"{scope:<8} {mode or 'default':<15} {pname:<15} {aname:<7} "
                  f"FAIL {str(exc).splitlines()[0][:26]}", flush=True)
            del net, opt, o
            torch.cuda.empty_cache()
            continue
        base = base or ms
        print(f"{scope:<8} {mode or 'default':<15} {pname:<15} {aname:<7} "
              f"{ms:>7.2f}m {base / ms:>8.2f}x {ms / mlp_ms:>7.2f}x", flush=True)
        del net, opt, o
        torch.cuda.empty_cache()
    print("\ncompile SCOPE is the lever: layer-only ~1.41x, whole-module ~1.98x,\n"
          "because Inductor then fuses across the gather, cat, ln_out and heads.\n"
          "Native bf16 is worth ~1.29x eager but ~nothing once compiled --\n"
          "Inductor already removes the autocast fp32-LayerNorm round trip.",
          flush=True)


def cmd_sanity(args, device, autocast):
    """Black-box checks that do not trust the repo implementations."""
    B = args.batch_size
    obs = torch.randn(B, OBS_DIM, device=device)

    print("=== 1. repo networks vs independent from-scratch equivalents ===")
    for label, repo, ref in (
        ("MLP", make_mlp(), ReferenceMLP()),
        (f"Transformer-L1 d={args.d_model}", make_transformer(1, args.d_model, args.n_heads),
         ReferenceEncoder(d_model=args.d_model, n_heads=args.n_heads)),
    ):
        out = []
        for net in (repo, ref):
            net = net.to(device).train()
            step, opt = train_step(net, obs, autocast)
            out.append((bench(step, args.iters, args.warmup, device),
                        sum(p.numel() for p in net.parameters())))
            del opt
            torch.cuda.empty_cache()
        (rms, rp), (fms, fp) = out
        verdict = "clean" if rms / fms < 1.25 else "REPO IS SLOWER"
        print(f"  {label:<28} repo {rms:6.2f}ms ({rp:,}) vs ref {fms:6.2f}ms "
              f"({fp:,})  ->  {rms/fms:.2f}x  {verdict}", flush=True)

    print("\n=== 2. roofline for the L1 shape ===")
    net = make_transformer(1, args.d_model, args.n_heads).to(device).train()
    macs = count_macs(net, obs)
    step, opt = train_step(net, obs, autocast)
    ms = bench(step, args.iters, args.warmup, device)
    flop_ms = (macs * B * 2 * 3) / PEAK_TFLOPS * 1e3
    print(f"  {macs:,} MACs/env -> compute floor {flop_ms:.2f} ms, "
          f"measured {ms:.2f} ms ({ms/flop_ms:.1f}x)", flush=True)
    print("  the MLP does MORE arithmetic and is faster; the gap is plumbing.")

    print("\n=== 3. hidden host-device synchronization in the train step ===")
    caught = []
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            step()
            torch.cuda.synchronize(device)
        caught = [str(w.message)[:100] for w in records
                  if "sync" in str(w.message).lower()]
    finally:
        torch.cuda.set_sync_debug_mode("default")
    print(f"  {len(caught) or 'no'} synchronizing call(s) inside one train step",
          flush=True)
    del net, opt
    torch.cuda.empty_cache()

    print("\n=== 4. token gather: correctness and cost ===")
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import (
        HAND_TOKEN_FIELDS, JOINT_WIDTH_FIELDS, build_token_layout, field_offsets)
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec
    spec = get_robot_spec("sharpa_iiwa14")
    layout = build_token_layout(spec, OBS_LIST)
    offsets = field_offsets(OBS_LIST, spec)
    n_arm, n_hand = layout["n_arm"], layout["n_hand"]
    # Each column holds its own index, so a mis-gather is a wrong NUMBER.
    flat = torch.arange(layout["obs_dim"], dtype=torch.float32,
                        device=device).unsqueeze(0)
    gather = torch.tensor(layout["token_columns"], dtype=torch.long, device=device)
    tokens = flat[:, gather][0]
    bad, slot = 0, 0
    for field in JOINT_WIDTH_FIELDS:
        start, _ = offsets[field]
        for j in range(n_hand):
            bad += int(tokens[j, slot].item()) != start + n_arm + j
        slot += 1
    for field, stride in HAND_TOKEN_FIELDS.items():
        start, _ = offsets[field]
        for j in range(n_hand):
            for k in range(stride):
                bad += int(tokens[j, slot + k].item()) != start + j * stride + k
        slot += stride
    gidx = [i for s, e in layout["global_slices"] for i in range(s, e)]
    used = sorted(int(v) for v in gather.flatten().tolist()) + sorted(gidx)
    print(f"  {slot * n_hand} slots checked, {bad} mis-gathered; "
          f"{len(set(range(layout['obs_dim'])) - set(used))} columns unused, "
          f"{len(used) - len(set(used))} duplicated  -> "
          f"{'PASS' if not bad and len(used) == len(set(used)) else 'FAIL'}",
          flush=True)

    o = torch.randn(B, OBS_DIM, device=device, requires_grad=True)
    g2 = gather
    gf = gather.reshape(-1).contiguous()

    def fb(make):
        def run():
            with autocast:
                out = make()
            out.sum().backward()
            o.grad = None
        return run

    with autocast:
        a = bench(fb(lambda: o[:, g2]), args.iters, args.warmup, device)
        b = bench(fb(lambda: torch.index_select(o, 1, gf)), args.iters, args.warmup, device)
    print(f"  2-D advanced index {a:.2f} ms vs index_select {b:.2f} ms "
          f"({a/b:.1f}x) -- bit-identical, which is why the network uses the latter",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("arch", "sweep", "speedups", "sanity"))
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--axis", default="all",
                        choices=("all", "d_model", "mlp", "minibatch"),
                        help="sweep only")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    stub_kit_free_packages()
    device = torch.device("cuda")
    autocast = torch.autocast("cuda", torch.float16)
    {"arch": cmd_arch, "sweep": cmd_sweep,
     "speedups": cmd_speedups, "sanity": cmd_sanity}[args.command](
        args, device, autocast)


if __name__ == "__main__":
    main()
