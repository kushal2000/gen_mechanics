"""What actually raises the joint transformer's achieved TFLOP/s.

The L1 network sustains ~16 TFLOP/s where the MLP it replaces sustains ~109 on
the same card. The cause is not the amount of arithmetic -- it is that at
d_model 128 with 23 tokens, the encoder layer is a chain of low-intensity
passes (LayerNorm, GELU, softmax, residual adds) around GEMMs whose reduction
depth K is only 128. One encoder layer is ~80% of the L1 network's time.

So this sweeps the things that change that ratio, and reports TFLOP/s rather
than milliseconds, because the interesting question is not "what is faster"
but "what uses the hardware better" -- a variant that costs the same
wall-clock while doing 4x the arithmetic is a win, not a wash.

Variants split into two kinds, and they should not be confused:

  NUMERICALLY EQUIVALENT (free): torch.compile.
  MODEL CHANGES (a design decision, not an optimization): head count,
  RMSNorm, and d_model.

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_transformer_variants.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
from profile_policy_train_step import (  # noqa: E402
    make_transformer, stub_kit_free_packages,
)

stub_kit_free_packages()

from profile_arch_efficiency import measure_macs  # noqa: E402


def swap_layernorm_for_rmsnorm(net: nn.Module) -> nn.Module:
    """LayerNorm does two reduction passes (mean, then variance); RMSNorm does
    one and drops the centering. Same shape, strictly less memory traffic."""
    for module in net.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.LayerNorm):
                replacement = nn.RMSNorm(
                    child.normalized_shape, eps=child.eps
                ).to(next(child.parameters()).device)
                setattr(module, name, replacement)
    return net


def build(d_model: int, n_heads: int, rmsnorm: bool, device) -> nn.Module:
    from coevolution.networks.joint_transformer import JointTransformerNet
    from profile_policy_train_step import OBS_LIST

    params = {
        "robot_spec": "sharpa_iiwa14", "obs_list": OBS_LIST,
        "d_model": d_model, "n_layers": 1, "n_heads": n_heads,
        "ff_mult": 4, "dropout": 0.0,
        "arm_head_units": [1024, 512], "value_head_units": [512, 256],
        "space": {"continuous": {
            "mu_activation": "None", "sigma_activation": "None",
            "sigma_init": {"name": "const_initializer", "val": 0},
            "fixed_sigma": "fixed",
        }},
    }
    net = JointTransformerNet(
        params, actions_num=29, input_shape=(778,), value_size=1,
        num_seqs=1, type="simple",
    ).to(device).train()
    return swap_layernorm_for_rmsnorm(net) if rmsnorm else net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    obs = torch.randn(args.batch_size, 778, device=device)

    #                 label                    d_model heads rms  compile
    variants = [
        ("baseline (d128, 4 heads)",               128,   4, False, False),
        ("+ torch.compile",                        128,   4, False, True),
        ("1 head",                                 128,   1, False, False),
        ("RMSNorm",                                128,   4, True,  False),
        ("1 head + RMSNorm + compile",             128,   1, True,  True),
        ("d_model 256 (4 heads)",                  256,   4, False, False),
        ("d_model 256 + compile",                  256,   4, False, True),
        ("d_model 384 + compile",                  384,   4, False, True),
        ("d_model 512 + compile",                  512,   4, False, True),
    ]

    print(f"batch={args.batch_size:,} fp16 autocast, 1 encoder layer, "
          f"fwd+bwd incl. optimizer\n", flush=True)
    header = (f"{'variant':<30} {'params':>10} {'MACs/env':>11} "
              f"{'fwd+bwd':>10} {'TFLOP/s':>9} {'vs base':>8}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    baseline_ms = None
    for label, d_model, n_heads, rmsnorm, compile_it in variants:
        net = build(d_model, n_heads, rmsnorm, device)
        macs = measure_macs(net, obs) // args.batch_size
        params = sum(p.numel() for p in net.parameters())
        step_net = torch.compile(net) if compile_it else net
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        def iteration():
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out = step_net({"obs": obs})
                loss = sum(t.square().mean() for t in out
                           if isinstance(t, torch.Tensor))
            loss.backward()
            optimizer.step()

        try:
            for _ in range(args.warmup):
                iteration()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.iters):
                iteration()
            torch.cuda.synchronize(device)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            print(f"{label:<30} FAIL: {str(exc).splitlines()[0][:40]}",
                  flush=True)
            del net, optimizer
            torch.cuda.empty_cache()
            continue
        ms = 1e3 * (time.perf_counter() - start) / args.iters
        tflops = (macs * args.batch_size * 2 * 3) / (ms * 1e-3) / 1e12
        if baseline_ms is None:
            baseline_ms = ms
        print(
            f"{label:<30} {params:>10,} {macs:>11,} {ms:>7.2f} ms "
            f"{tflops:>9.1f} {baseline_ms / ms:>7.2f}x",
            flush=True,
        )
        del net, step_net, optimizer
        torch.cuda.empty_cache()

    print(
        "\n'vs base' is wall-clock speedup. Read it together with MACs/env: a "
        "wider d_model that holds wall-clock while multiplying the\n"
        "arithmetic is buying capacity for free, which is a different and "
        "often better deal than going faster at the same capacity.",
        flush=True,
    )


if __name__ == "__main__":
    main()
