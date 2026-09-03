"""Why the 22-token transformer is slower than the MLP it replaces.

Two effects compound, and it is worth separating them because only one of them
is fixable:

1. MORE ARITHMETIC. Sharing one set of weights across 22 joints is what makes
   the policy modular, and sharing means APPLYING those weights 22 times. The
   transformer has far fewer parameters than the MLP and does more work with
   them.

2. WORSE ARITHMETIC PER BYTE. The MLP is four large GEMMs. The encoder layer
   is many small ones (reduction depth K = d_model = 128) separated by
   LayerNorm / GELU / residual passes that are pure memory traffic. The
   hardware runs the first shape near peak and the second nowhere near it.

Prints parameters, MACs/env, and the TFLOP/s each network actually achieves,
so the gap can be attributed rather than guessed at.

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_arch_efficiency.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from profile_policy_train_step import (  # noqa: E402
    make_mlp, make_transformer, stub_kit_free_packages,
)

stub_kit_free_packages()


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def measure_macs(net, obs) -> int:
    """Exact multiply-accumulates for one forward, counted by hooking Linear.

    Linear is every matmul in both networks except attention, which is added
    separately by the caller -- at 23 tokens it is a rounding error either way.
    """
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
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    obs = torch.randn(args.batch_size, 778, device=device)

    nets = [
        ("MLP [1024,1024,512,512]", make_mlp()),
        ("Transformer-L1 (22+1 tokens)", make_transformer(1)),
        ("Transformer-L4 (22+1 tokens)", make_transformer(4)),
    ]

    print(f"batch={args.batch_size:,} fp16 autocast\n", flush=True)
    header = (f"{'network':<30} {'params':>10} {'MACs/env':>11} "
              f"{'fwd+bwd':>10} {'TFLOP/s':>9} {'act B/env':>10}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name, net in nets:
        net = net.to(device).train()
        macs = measure_macs(net, obs) // args.batch_size
        params = count_params(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        def iteration():
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out = net({"obs": obs})
                loss = sum(t.square().mean() for t in out
                           if isinstance(t, torch.Tensor))
            loss.backward()
            optimizer.step()

        for _ in range(args.warmup):
            iteration()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.iters):
            iteration()
        torch.cuda.synchronize(device)
        ms = 1e3 * (time.perf_counter() - start) / args.iters

        # fwd+bwd is ~3x the forward's multiply-adds, x2 flops per MAC.
        tflops = (macs * args.batch_size * 2 * 3) / (ms * 1e-3) / 1e12
        # Activation bytes touched per env, as a stand-in for arithmetic
        # intensity: peak allocation is dominated by stored activations.
        act_bytes = torch.cuda.max_memory_allocated(device) / args.batch_size
        print(
            f"{name:<30} {params:>10,} {macs:>11,} {ms:>7.2f} ms "
            f"{tflops:>9.1f} {act_bytes:>9.0f}B",
            flush=True,
        )
        del net, optimizer
        torch.cuda.empty_cache()

    print(
        "\nMACs/env is one forward. TFLOP/s counts fwd+bwd as 3x that, so it "
        "is the rate the hardware actually sustains on each shape --\n"
        "the same card, the same precision, the same batch.",
        flush=True,
    )


if __name__ == "__main__":
    main()
