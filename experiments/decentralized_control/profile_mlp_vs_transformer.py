"""MLP vs joint-transformer, as architectures. No env, no data, no checkpoint.

    python experiments/decentralized_control/profile_mlp_vs_transformer.py
    python experiments/decentralized_control/profile_mlp_vs_transformer.py --batch 16384 --sweep

Both networks are handed the SAME input and asked for the SAME output:

    in :  22 hand joints x 32 features  +  74 global (arm state + task)  =  778
    out:  22 hand torques              +   7 arm torques                =   29

The only difference is what they are allowed to assume. The MLP assumes
nothing: every input element may influence every output element, so it wires
778 -> 1024 -> 1024 -> 512 -> 512 -> 29 densely. The transformer assumes the
input is 22 interchangeable joints plus a shared context, so it embeds each
joint separately, lets them exchange information through attention, and reads
each joint's torque off its own token with a head shared by all 22.

THE CLAIM THIS FILE TESTS. "Dense is strictly more expensive than modular."
On parameters and FLOPs that is simply true, and by a wide margin. On
activation memory and on wall-clock it is false. Measured at batch 16384,
bf16 + torch.compile, forward+backward (historical measurements):

  dense MLP 778-1024-1024-512-512-29   2.650M params  86.72 GFLOP   2.43 ms
  JT d32 h1 ff1 mu=[32]                0.072M params   9.31 GFLOP   3.18 ms

The transformer uses 2.7% of the parameters and 10.7% of the arithmetic and
is still slower. Why: the MLP's matmuls are (16384 x 778) @ (778 x 1024),
430 FLOP/byte, above this card's 190 balance -- compute bound, 180-213
TFLOP/s. The transformer's are (376832 x 32) @ (32 x 32), 16 FLOP/byte --
memory bound, 2-7% of peak. Arithmetic intensity is harmonic_mean(M,K,N)/3
for bf16, so it is set by the SMALLEST dimension; weight sharing across 22
joints is exactly what makes K and N small. The bias you are buying costs
throughput; it does not save it.

MEASURE IN THE TRAINING CONFIGURATION. Three separate conclusions in this
file's history were reversed by measuring the wrong thing:

  eager vs compiled   compile is 2.1x for the transformer and 1.08x for the
                      MLP (which is already ~90% GEMM with nothing to fuse).
                      Eager makes the transformer look 2x worse than it is.
  fp32 vs bf16        the MLP gains 4.10x from bf16, the transformer only
                      2.52x -- fat matmuls saturate tensor cores, skinny ones
                      do not. In fp32 the transformer appears to WIN; in bf16
                      it loses. These are historical BF16 measurements;
                      the launchers actually use FP16 autocast.
  attention's share   at 23 tokens it is not the bottleneck in eager, where
                      LayerNorm's dgamma/dbeta reduction is 40%. Compiled,
                      Inductor fuses that away and attention becomes 19%.

Use --compile 1 and --precision fp16 (the defaults) to match training's
forward precision. The historical BF16 timings above must be remeasured in
FP16 before drawing conclusions about training throughput. This benchmark
does not include training's GradScaler or optimizer step.

WHAT THIS COMPARISON DOES NOT CONTROL FOR. It measures cost, not quality, and
the two networks are not matched on either of the things that would make a
learning comparison fair:

  capacity        2.650M parameters against 0.072M, a 37x gap. Any claim that
                  one architecture "does the same job cheaper" is unsupported
                  by this file -- it shows only what each costs to run.
  global context  the MLP wires all 74 global dimensions (palm pose, object
                  rotation, keypoints_rel_goal -- the actual task target)
                  directly into every hidden unit. The transformer projects
                  them to d_model first, so at d_model 32 a hand joint reaches
                  the task through a 3.3x bottleneck, and only via attention
                  with one global token. Per-joint object geometry
                  (object_keypoints_rel_joint) IS in each token, so this is not
                  a total loss of task information -- but the goal is global.
                  Note arm_head takes a RAW skip of the global vector for
                  exactly this reason (a version without it "scored ~0 on the
                  post-lift keypoint term"); the hand path has no such skip and
                  may be paying the same cost silently.

FLOPs are counted analytically (2 per multiply-accumulate) rather than with a
profiler, so every number here can be checked by hand against the shapes.
"""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
import torch.nn as nn

# The layout both architectures are handed. These are the real widths from
# isaacsimenvs/pose_reaching_6d: 22 hand joints, 32 features each, 74 shared.
N_HAND = 22
TOKEN_DIM = 32
GLOBAL_DIM = 74
N_ARM = 7
OBS_DIM = N_HAND * TOKEN_DIM + GLOBAL_DIM      # 778
ACT_DIM = N_HAND + N_ARM                       # 29


# --------------------------------------------------------------------------
# architectures
# --------------------------------------------------------------------------

def _mlp(in_dim: int, units, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for width in units:
        layers += [nn.Linear(dim, width), nn.ELU()]
        dim = width
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class DenseMLP(nn.Module):
    """Assumes nothing. Every input may reach every output."""

    def __init__(self, units):
        super().__init__()
        self.net = _mlp(OBS_DIM, units, ACT_DIM)
        self.units = list(units)

    def forward(self, obs):                      # (B, 778) -> (B, 29)
        return self.net(obs)

    def flops(self, batch: int) -> int:
        dims = [OBS_DIM] + self.units + [ACT_DIM]
        return batch * sum(2 * a * b for a, b in zip(dims, dims[1:]))


class EncoderLayer(nn.Module):
    """Pre-LN block, explicit attention (no fused kernel shape limits)."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = (
            nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
                          nn.Linear(ff_mult * d_model, d_model))
            if ff_mult else None
        )

    def forward(self, x):
        b, t, dim = x.shape
        h = self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, h, dim // h).transpose(1, 2) for z in (q, k, v))
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = scores.softmax(dim=-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, dim))
        return x if self.ff is None else x + self.ff(self.ln_ff(x))


class JointTransformer(nn.Module):
    """Assumes the 22 hand joints are interchangeable parts of one hand.

    Nothing here has a size that depends on the joint COUNT: token_proj is per
    joint, the encoder is length-agnostic, mu_head is shared. That is the
    inductive bias, and it is also why the same weights run on a 4-fingered
    hand.
    """

    def __init__(self, d_model, n_heads, ff_mult, n_layers, mu_units, arm_units):
        super().__init__()
        self.d_model, self.n_heads, self.ff_mult = d_model, n_heads, ff_mult
        self.n_layers, self.mu_units, self.arm_units = n_layers, list(mu_units), list(arm_units)
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.global_proj = nn.Linear(GLOBAL_DIM, d_model)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, ff_mult) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.mu_head = _mlp(d_model, mu_units, 1)
        # The arm reads the global token PLUS the raw global vector: a first
        # version without the skip lifted fine but scored ~0 on the post-lift
        # keypoint term, every task number squeezed through one token.
        self.arm_head = _mlp(d_model + GLOBAL_DIM, arm_units, N_ARM)

    def forward(self, obs):                      # (B, 778) -> (B, 29)
        b = obs.shape[0]
        hand = obs[:, : N_HAND * TOKEN_DIM].view(b, N_HAND, TOKEN_DIM)
        glob = obs[:, N_HAND * TOKEN_DIM:]
        x = torch.cat([self.token_proj(hand), self.global_proj(glob).unsqueeze(1)], 1)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_out(x)
        mu_hand = self.mu_head(x[:, :N_HAND]).squeeze(-1)
        mu_arm = self.arm_head(torch.cat([x[:, N_HAND], glob], dim=-1))
        return torch.cat([mu_hand, mu_arm], dim=-1)

    def flops(self, batch: int) -> int:
        d, t = self.d_model, N_HAND + 1
        f = 2 * N_HAND * TOKEN_DIM * d + 2 * GLOBAL_DIM * d          # embeddings
        for _ in range(self.n_layers):
            f += 2 * t * d * 3 * d                                   # qkv
            f += 2 * t * t * d                                       # scores
            f += 2 * t * t * d                                       # weights @ v
            f += 2 * t * d * d                                       # proj
            if self.ff_mult:
                f += 2 * 2 * t * d * self.ff_mult * d                # ff up + down
        dims = [d] + self.mu_units + [1]
        f += N_HAND * sum(2 * a * b for a, b in zip(dims, dims[1:]))  # per-joint head
        dims = [d + GLOBAL_DIM] + self.arm_units + [N_ARM]
        f += sum(2 * a * b for a, b in zip(dims, dims[1:]))           # arm head
        return batch * f


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def precision_context(device, precision):
    torch.backends.cuda.matmul.allow_tf32 = (precision == "tf32")
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")
    if precision in ("fp16", "bf16") and torch.device(device).type == "cuda":
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def measure(net, batch, device, iters=20, compile_net=False, precision="fp16"):
    """Peak activation memory and wall-clock, forward and forward+backward."""
    net = net.to(device)
    if compile_net:
        net = torch.compile(net, dynamic=False)
    obs = torch.randn(batch, OBS_DIM, device=device)

    # TF32 is a global flag, autocast is a context. Set both explicitly rather
    # than inheriting whatever the process happened to leave them at.
    amp = precision_context(device, precision)

    def fwd():
        with amp:
            return net(obs)

    def fwd_bwd():
        net.zero_grad(set_to_none=True)
        # The loss is computed in fp32 outside autocast, as rl_games does.
        fwd().float().pow(2).mean().backward()

    for _ in range(12 if compile_net else 5):            # compile needs more warmup
        fwd_bwd()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Peak with the graph alive is the number that decides whether a run fits.
    before = torch.cuda.memory_allocated() if device == "cuda" else 0
    fwd_bwd()
    if device == "cuda":
        torch.cuda.synchronize()
    act = (torch.cuda.max_memory_allocated() - before) / 2**20 if device == "cuda" else float("nan")

    def timeit(fn):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3           # ms

    # The no_grad forward is a DIFFERENT torch.compile guard from the one the
    # fwd+bwd warmup built, so it recompiles on first call. Without this warmup
    # the "forward" time is compilation time -- it read 83 ms against a 10 ms
    # fwd+bwd, which is how the bug announced itself.
    with torch.no_grad():
        for _ in range(12 if compile_net else 3):
            fwd()
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd = timeit(fwd)
    t_both = timeit(fwd_bwd)
    del obs
    if device == "cuda":
        torch.cuda.empty_cache()
    return act, t_fwd, t_both


def row(name, net, batch, device, compile_net=False, precision="fp16"):
    params = sum(p.numel() for p in net.parameters())
    fl = net.flops(batch)
    act, t_fwd, t_both = measure(net, batch, device, compile_net=compile_net,
                                 precision=precision)
    # Forward is ~1/3 of fwd+bwd FLOPs; report achieved rate on the fwd pass.
    tflops = fl / (t_fwd * 1e-3) / 1e12
    return dict(name=name, params=params, flops=fl, act=act,
                t_fwd=t_fwd, t_both=t_both, tflops=tflops)


def report(rows):
    b = rows[0]
    hdr = (f"{'architecture':<34}{'params':>10}{'GFLOP/step':>12}{'act MiB':>10}"
           f"{'fwd ms':>9}{'fwd+bwd ms':>12}{'TFLOP/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<34}{r['params']/1e6:>9.3f}M{r['flops']/1e9:>12.2f}"
              f"{r['act']:>10.0f}{r['t_fwd']:>9.2f}{r['t_both']:>12.2f}{r['tflops']:>9.1f}")
    print()
    print(f"{'':<34}{'relative to the dense MLP':>10}")
    print("-" * len(hdr))
    for r in rows[1:]:
        print(f"{r['name']:<34}{r['params']/b['params']:>9.3f}x{r['flops']/b['flops']:>11.3f}x"
              f"{r['act']/b['act']:>9.2f}x{r['t_fwd']/b['t_fwd']:>8.2f}x"
              f"{r['t_both']/b['t_both']:>11.2f}x")


CATS = [("GEMM", ("sgemm", "cutlass", "gemm", "Kernel2")),
        ("LayerNorm", ("GammaBeta", "layer_norm", "LayerNorm")),
        ("elementwise", ("elementwise",)), ("cat/copy", ("CatArray", "copy", "Copy")),
        ("reduce", ("reduce",)), ("softmax", ("softmax", "Softmax"))]


def breakdown(nets, batch, device, compile_net, precision="fp16"):
    """Where the CUDA time goes, using the selected forward precision."""
    from torch.profiler import ProfilerActivity, profile as tprofile

    def bucket(key):
        for name, pats in CATS:
            if any(pat in key for pat in pats):
                return name
        return "other"

    cols = [n for n, _ in CATS] + ["other"]
    print(f"{'config':<34}{'ms/step':>9}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (43 + 12 * len(cols)))
    for label, mk in nets:
        net = mk().to(device)
        if compile_net:
            net = torch.compile(net, dynamic=False)
        x = torch.randn(batch, OBS_DIM, device=device)
        amp = precision_context(device, precision)

        def step():
            net.zero_grad(set_to_none=True)
            with amp:
                output = net(x)
            output.float().pow(2).mean().backward()

        for _ in range(12):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1e3
        with tprofile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(10):
                step()
            torch.cuda.synchronize()
        evs = [e for e in pr.key_averages() if e.self_device_time_total > 0]
        tot = sum(e.self_device_time_total for e in evs)
        agg = {}
        for e in evs:
            agg[bucket(e.key)] = agg.get(bucket(e.key), 0) + e.self_device_time_total
        print(f"{label:<34}{ms:>9.2f}"
              + "".join(f"{agg.get(c, 0) / tot * 100:>11.0f}%" for c in cols))
        del net, x
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mlp_units", type=int, nargs="+", default=[1024, 1024, 512, 512])
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=1)
    p.add_argument("--ff_mult", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=1)
    p.add_argument("--mu_head_units", type=int, nargs="*", default=[])
    p.add_argument("--arm_head_units", type=int, nargs="*", default=[256, 128])
    p.add_argument("--precision", default="fp16",
                   choices=["fp16", "bf16", "tf32", "fp32"],
                   help="fp16 (default) uses CUDA autocast to match training's "
                        "mixed_precision: True. bf16 is available for comparison; "
                        "tf32 and fp32 disable autocast.")
    p.add_argument("--compile", type=int, default=1,
                   help="1 wraps each net in torch.compile(dynamic=False), which "
                        "is what the launchers actually run. Eager numbers "
                        "flatter the MLP: it is already ~88%% GEMM with nothing "
                        "to fuse, while the transformer's LayerNorm and "
                        "elementwise ops are exactly what Inductor removes.")
    p.add_argument("--breakdown", action="store_true",
                   help="per-kernel CUDA time by category, instead of the table.")
    p.add_argument("--sweep", action="store_true",
                   help="also run a few transformer configs worth comparing")
    args = p.parse_args()

    torch.manual_seed(0)
    print(f"\ninput {OBS_DIM} ({N_HAND} joints x {TOKEN_DIM} + {GLOBAL_DIM} global)"
          f"  ->  output {ACT_DIM} ({N_HAND} hand + {N_ARM} arm)")
    print(f"batch {args.batch}, {args.device}, {args.precision}, "
          f"{'torch.compile' if args.compile else 'EAGER (understates the transformer ~2x)'}"
          + (f", {torch.cuda.get_device_name(0)}" if args.device == "cuda" else "") + "\n")

    def jt(**kw):
        cfg = dict(d_model=args.d_model, n_heads=args.n_heads, ff_mult=args.ff_mult,
                   n_layers=args.n_layers, mu_units=args.mu_head_units,
                   arm_units=args.arm_head_units)
        cfg.update(kw)
        return JointTransformer(**cfg)

    c = bool(args.compile)
    if args.breakdown:
        breakdown([("DENSE MLP", lambda: DenseMLP(args.mlp_units)),
                   (f"JT d{args.d_model} h{args.n_heads} ff{args.ff_mult}", jt),
                   ("JT d32 h1 ff1 mu=[32]",
                    lambda: jt(d_model=32, n_heads=1, ff_mult=1, mu_units=[32]))],
                  args.batch, args.device, c, args.precision)
        return

    rows = [row(f"DENSE MLP {'-'.join(map(str, args.mlp_units))}",
                DenseMLP(args.mlp_units), args.batch, args.device, c, args.precision),
            row(f"JT d{args.d_model} h{args.n_heads} ff{args.ff_mult} L{args.n_layers}",
                jt(), args.batch, args.device, c, args.precision)]
    if args.sweep:
        rows += [
            row("JT d32 h4 ff1 mu=[32]", jt(d_model=32, n_heads=4, ff_mult=1, mu_units=[32]),
                args.batch, args.device, c, args.precision),
            row("JT d32 h1 ff1 mu=[32]", jt(d_model=32, n_heads=1, ff_mult=1,
                                            mu_units=[32]), args.batch, args.device, c, args.precision),
            row("JT d64 h1 ff0 (no FF)", jt(d_model=64, n_heads=1, ff_mult=0),
                args.batch, args.device, c, args.precision),
        ]
    report(rows)

    print("\nRead it this way:")
    print("  params / GFLOP   modularity wins -- weights and arithmetic are shared")
    print("                   across 22 joints instead of unrolled into dense layers.")
    print("  act MiB          modularity loses -- every intermediate carries a token")
    print("                   axis, and attention saves (B, heads, T, T) on top.")
    print("  ms               compare the measured times for the selected precision")
    print("                   and compile setting; historical BF16 results may differ.")
    print("  TFLOP/s          how far each is from the hardware's peak. A low number")
    print("                   means the shape, not the arithmetic, is the limit.")


if __name__ == "__main__":
    main()
