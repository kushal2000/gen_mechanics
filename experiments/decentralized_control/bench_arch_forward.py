"""Architecture-only throughput, on prefilled dummy inputs.

No simulator, no rl_games loop, no optimizer, no autograd graph beyond what is
asked for: one tensor of random observations, held constant, pushed through
each policy. Everything that made the end-to-end numbers hard to compare --
physics, contention, SAPG bookkeeping, memory pressure, a policy that changes
as it learns -- is absent by construction, so what is left is the architecture.

Reports forward-only (the rollout cost, once per env per step) and
forward+backward (the gradient-step cost), plus the env-steps/second each
implies, so the two phases of a real epoch can be reasoned about separately.

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/bench_arch_forward.py
    ... --batch_sizes 24576        # the rollout width
    ... --batch_sizes 16384        # the minibatch width
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from profile_policy_train_step import (  # noqa: E402
    make_mlp, make_transformer, stub_kit_free_packages,
)

stub_kit_free_packages()


def build_variants(mu_units):
    """(label, factory). Each factory takes no arguments and returns a module."""
    variants = [("MLP [1024,1024,512,512]", make_mlp)]
    for layers in (0, 1, 4):
        variants.append((f"Transformer-L{layers}",
                         lambda n=layers: make_transformer(n)))
    if mu_units:
        for layers in (0, 1):
            variants.append(
                (f"Transformer-L{layers} mu_head{mu_units}",
                 lambda n=layers: make_transformer(n, mu_head_units=mu_units)))
    return variants


def count_macs(net, obs) -> int:
    total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal total
        rows = inputs[0].numel() // inputs[0].shape[-1]
        total += rows * module.in_features * module.out_features

    for module in net.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        net({"obs": obs})
    for handle in handles:
        handle.remove()
    return total // obs.shape[0]


def timed(fn, iters, warmup, device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(device)
    return 1e3 * (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", default="16384,24576")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--mu_head_units", default="512,256",
        help="Also benchmark the per-joint head widened to this MLP. Empty "
             "to skip. The bare Linear(d_model, 1) it replaces is an N=1 GEMM "
             "running at its launch floor, so this is expected to be ~free.",
    )
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16,
                              enabled=args.amp)
    mu_units = [int(v) for v in args.mu_head_units.split(",") if v.strip()]

    for batch in (int(v) for v in args.batch_sizes.split(",") if v.strip()):
        # Prefilled once and reused: the input is never the thing being timed.
        obs = torch.randn(batch, 778, device=device)
        print(f"\n=== batch {batch:,}  fp16={args.amp} "
              f"(dummy inputs, no env, no optimizer) ===", flush=True)
        header = (f"{'architecture':<34} {'MACs/env':>10} {'forward':>9} "
                  f"{'fwd+bwd':>9} {'fwd env/s':>12}")
        print(header, flush=True)
        print("-" * len(header), flush=True)

        for label, build in build_variants(mu_units):
            net = build().to(device)
            macs = count_macs(net, obs)

            net.eval()
            with torch.no_grad():
                fwd = timed(lambda: net({"obs": obs}), args.iters, args.warmup,
                            device) if not args.amp else None
            if args.amp:
                with torch.no_grad(), autocast:
                    fwd = timed(lambda: net({"obs": obs}), args.iters,
                                args.warmup, device)

            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=1e-4)

            def step():
                opt.zero_grad(set_to_none=True)
                with autocast:
                    out = net({"obs": obs})
                    loss = sum(t.square().mean() for t in out
                               if isinstance(t, torch.Tensor))
                loss.backward()
                opt.step()

            bwd = timed(step, args.iters, args.warmup, device)
            print(f"{label:<34} {macs:>10,} {fwd:>7.2f}ms {bwd:>7.2f}ms "
                  f"{batch / (fwd * 1e-3):>12,.0f}", flush=True)
            del net, opt
            torch.cuda.empty_cache()

    print("\nforward = the rollout cost (once per policy step, both networks).\n"
          "fwd+bwd = one gradient step. A SAPG epoch is 16 rollout forwards\n"
          "plus 56 gradient steps, for the actor AND the central-value net.",
          flush=True)


if __name__ == "__main__":
    main()
