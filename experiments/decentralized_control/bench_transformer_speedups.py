"""How fast can the joint transformer's train step actually be made?

The architecture arguments are settled: L1 at d_model 64 does FEWER MACs/env
than the MLP baseline (2,175,744 vs 2,647,040) and its own bandwidth roofline
is ~3.3 ms, yet it measures ~21 ms eager. That gap is implementation, so this
sweeps the implementation axes rather than the model:

  compile scope   none / the encoder layers only / the whole network
  compile mode    default / max-autotune / reduce-overhead (CUDA graphs)
  precision       autocast fp16 / autocast bf16 / NATIVE bf16
  optimizer       Adam default (foreach) / fused

Native precision matters more than it looks: under autocast, LayerNorm is on
the fp32 list, so every norm upcasts its input, works at double the bytes, and
downcasts the result. A natively-bf16 module skips all three.

Meant for a DEDICATED gpu -- two of these sharing a card inflates every number.

    BENCH_TAG=speedups SCRIPT=bench_transformer_speedups.py \
      sbatch experiments/decentralized_control/bench_torch_only.sub
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_policy_train_step import (  # noqa: E402
    OBS_LIST, make_mlp, stub_kit_free_packages,
)

stub_kit_free_packages()
from coevolution.networks.joint_transformer import JointTransformerNet  # noqa: E402


def build(d_model, n_layers, n_heads, device):
    params = {
        "robot_spec": "sharpa_iiwa14", "obs_list": OBS_LIST,
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "ff_mult": 4, "dropout": 0.0, "arm_head_units": [1024, 512],
        "value_head_units": [512, 256], "mu_head_units": [],
        "final_norm": True, "compile_layers": False,
        "space": {"continuous": {
            "mu_activation": "None", "sigma_activation": "None",
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": "fixed"}},
    }
    return JointTransformerNet(
        params, actions_num=29, input_shape=(778,), value_size=1,
        num_seqs=1, type="simple").to(device)


def timed(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / iters


def run(net, batch, device, scope, mode, dtype, native, fused, iters, warmup):
    obs = torch.randn(batch, 778, device=device)
    if native:
        net = net.to(dtype)
        obs = obs.to(dtype)
        ctx = torch.autocast("cuda", dtype, enabled=False)
    else:
        ctx = torch.autocast("cuda", dtype)
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-4, fused=fused)

    kw = {"dynamic": False}
    if mode:
        kw["mode"] = mode
    if scope == "layers":
        for layer in net.layers:
            layer.compile(**kw)
        stepped = net
    elif scope == "net":
        stepped = torch.compile(net, **kw)
    else:
        stepped = net

    def one():
        opt.zero_grad(set_to_none=True)
        with ctx:
            out = stepped({"obs": obs})
            # .float() so the loss is identical across precisions.
            loss = sum(t.float().square().mean() for t in out
                       if isinstance(t, torch.Tensor))
        loss.backward()
        opt.step()

    return timed(one, iters, warmup, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda")
    B = args.batch_size

    mlp = make_mlp().to(device).train()
    obs = torch.randn(B, 778, device=device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    def mlp_step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16):
            out = mlp({"obs": obs})
            loss = sum(t.float().square().mean() for t in out
                       if isinstance(t, torch.Tensor))
        loss.backward()
        opt.step()

    mlp_ms = timed(mlp_step, args.iters, args.warmup, device)
    del mlp, opt, obs
    torch.cuda.empty_cache()

    macs = 2_175_744 if (args.d_model, args.n_layers) == (64, 1) else None
    print(f"L{args.n_layers} d_model={args.d_model} n_heads={args.n_heads} "
          f"batch={B:,}   MLP baseline = {mlp_ms:.2f} ms"
          + (f"   (L1 does {macs:,} MACs/env vs the MLP's 2,647,040)"
             if macs else ""), flush=True)
    print(f"\n{'compile':<8} {'mode':<16} {'precision':<16} {'adam':<7} "
          f"{'ms':>8} {'vs eager':>9} {'vs MLP':>8}", flush=True)

    grid = list(itertools.product(
        [("none", None), ("layers", None), ("net", None),
         ("net", "max-autotune"), ("net", "reduce-overhead")],
        [("fp16 autocast", torch.float16, False),
         ("bf16 autocast", torch.bfloat16, False),
         ("bf16 NATIVE", torch.bfloat16, True)],
        [("default", False), ("fused", True)],
    ))
    base = None
    for (scope, mode), (pname, dtype, native), (aname, fused) in grid:
        # fused Adam moved nothing on its own; only pair it with the winners.
        if fused and scope == "none":
            continue
        net = build(args.d_model, args.n_layers, args.n_heads, device)
        try:
            ms = run(net, B, device, scope, mode, dtype, native, fused,
                     args.iters, args.warmup)
        except Exception as exc:  # noqa: BLE001 - diagnostic sweep
            print(f"{scope:<8} {str(mode):<16} {pname:<16} {aname:<7} "
                  f"{'FAIL: ' + str(exc).splitlines()[0][:28]}", flush=True)
            del net
            torch.cuda.empty_cache()
            continue
        if base is None:
            base = ms
        print(f"{scope:<8} {str(mode):<16} {pname:<16} {aname:<7} "
              f"{ms:>7.2f}m {base / ms:>8.2f}x {ms / mlp_ms:>7.2f}x", flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
