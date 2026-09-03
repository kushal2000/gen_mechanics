"""A from-scratch MLP policy, to check the rl_games baseline against.

The transformer was cleared this way -- an independently written 22-token
encoder matched ``JointTransformerNet`` to 1.00x, so its cost was the shape and
not the code. The MLP baseline deserves the same test before any L0-vs-MLP
number is trusted: if rl_games' builder carries overhead of its own, then the
comparison has been unfair in the *other* direction.

Nothing here imports rl_games. Same widths, same activation, same heads, same
parameter count as ``A2CBuilder`` produces for::

    mlp: {units: [1024, 1024, 512, 512], activation: elu}

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/reference_mlp.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch
import torch.nn as nn


class ReferenceMLP(nn.Module):
    """rl_games' `separate: false` continuous A2C MLP, written out.

    trunk -> (mu, value), with a state-independent log-sigma parameter, which
    is what ``fixed_sigma: fixed`` builds.
    """

    def __init__(self, obs_dim=778, units=(1024, 1024, 512, 512), actions=29,
                 value_size=1):
        super().__init__()
        layers: list[nn.Module] = []
        dim = obs_dim
        for width in units:
            layers += [nn.Linear(dim, width), nn.ELU()]
            dim = width
        self.actor_mlp = nn.Sequential(*layers)
        self.mu = nn.Linear(dim, actions)
        self.value = nn.Linear(dim, value_size)
        self.sigma = nn.Parameter(torch.zeros(actions))

    def forward(self, obs_dict):
        h = self.actor_mlp(obs_dict["obs"])
        mu = self.mu(h)
        return mu, mu * 0 + self.sigma, self.value(h), None


def bench(fn, iters, warmup, device) -> float:
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
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=12)
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from profile_policy_train_step import make_mlp, stub_kit_free_packages
    stub_kit_free_packages()
    from torch.profiler import profile, ProfilerActivity

    device = torch.device("cuda")
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)
    obs = torch.randn(args.batch_size, 778, device=device)

    print(f"batch={args.batch_size:,} fp16 autocast\n")
    print(f"{'implementation':<34} {'params':>10} {'kernels':>8} "
          f"{'forward':>9} {'fwd+bwd':>9}")
    print("-" * 74)

    results = {}
    for label, net in (("rl_games A2CBuilder", make_mlp()),
                       ("ReferenceMLP (this file)", ReferenceMLP())):
        net = net.to(device)
        params = sum(p.numel() for p in net.parameters())

        net.eval()
        with torch.no_grad(), autocast:
            fwd = bench(lambda: net({"obs": obs}), args.iters, args.warmup, device)
            for _ in range(3):
                net({"obs": obs})
            torch.cuda.synchronize(device)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                net({"obs": obs})
                torch.cuda.synchronize(device)
        n_kernels = sum(e.count for e in prof.key_averages()
                        if e.device_time_total > 0)

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

        bwd = bench(step, args.iters, args.warmup, device)
        results[label] = (fwd, bwd)
        print(f"{label:<34} {params:>10,} {n_kernels:>8} {fwd:>7.2f}ms {bwd:>7.2f}ms",
              flush=True)
        del net, opt
        torch.cuda.empty_cache()

    (rf, rb), (mf, mb) = results["rl_games A2CBuilder"], results["ReferenceMLP (this file)"]
    print(f"\n  rl_games / reference:  forward {rf / mf:.2f}x   fwd+bwd {rb / mb:.2f}x")
    print("  >1.10x would mean the BASELINE carries builder overhead too, and every\n"
          "  L0-vs-MLP number so far has been unfair in the other direction.")


if __name__ == "__main__":
    main()
