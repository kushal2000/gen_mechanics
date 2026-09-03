"""Attribute the SAPG update phase, which per-network compute does not explain.

The L1 transformer run spends 12.82 s/epoch in ``update_time``. The actor's
56 gradient steps and the critic's 48 account for ~7.1 s of that in isolated
benchmarks. This finds the other ~5.7 s.

``update_time`` in ``a2c_common.train_epoch`` (line 1386-1455) spans exactly
four things, and this times each of them plus the interesting sub-parts:

    prepare_dataset -> algo_observer.after_steps -> train_central_value
      -> mini_epochs x minibatches of train_actor_critic

Every boundary synchronizes CUDA, so the attribution is unambiguous and the
total will run slightly above an uninstrumented epoch. Compare shares, not
absolute seconds, against a normal run.

Run it exactly like train.py -- it wraps the classes and then hands off:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_update_breakdown.py \
      --task GenMech-PoseReach-Direct-v0 \
      --agent rl_games_joint_transformer_cfg_entry_point --headless \
      agent.params.config.max_epochs=12 ...
"""

from __future__ import annotations

import collections
import functools
import pathlib
import sys
import time


TOTALS: dict[str, float] = collections.defaultdict(float)
COUNTS: dict[str, int] = collections.defaultdict(int)
EPOCH = {"n": 0}


def _install_instrumentation() -> None:
    import torch
    from rl_games.algos_torch import a2c_continuous, central_value
    from rl_games.common import a2c_common, datasets

    device = torch.device("cuda")

    def timed(label: str, unwrapped):
        @functools.wraps(unwrapped)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            try:
                return unwrapped(*args, **kwargs)
            finally:
                torch.cuda.synchronize(device)
                TOTALS[label] += time.perf_counter() - start
                COUNTS[label] += 1
        return wrapper

    # --- the four things update_time is made of --------------------------
    a2c_continuous.A2CAgent.prepare_dataset = timed(
        "prepare_dataset", a2c_continuous.A2CAgent.prepare_dataset)
    central_value.CentralValueTrain.train_net = timed(
        "central_value.train_net (total)", central_value.CentralValueTrain.train_net)
    central_value.CentralValueTrain.update_dataset = timed(
        "  central_value.update_dataset", central_value.CentralValueTrain.update_dataset)
    central_value.CentralValueTrain.train_critic = timed(
        "  central_value.train_critic", central_value.CentralValueTrain.train_critic)
    a2c_continuous.A2CAgent.train_actor_critic = timed(
        "actor train_actor_critic (total)",
        a2c_continuous.A2CAgent.train_actor_critic)
    a2c_continuous.A2CAgent.calc_gradients = timed(
        "  actor calc_gradients", a2c_continuous.A2CAgent.calc_gradients)

    # --- the usual suspects inside a gradient step -----------------------
    a2c_common.A2CBase._preproc_obs = timed(
        "  _preproc_obs (actor)", a2c_common.A2CBase._preproc_obs)
    central_value.CentralValueTrain._preproc_obs = timed(
        "  _preproc_obs (critic)", central_value.CentralValueTrain._preproc_obs)
    datasets.PPODataset.__getitem__ = timed(
        "  dataset __getitem__", datasets.PPODataset.__getitem__)
    torch.nn.utils.clip_grad_norm_ = timed(
        "  clip_grad_norm_", torch.nn.utils.clip_grad_norm_)

    # --- play side, for context ------------------------------------------
    a2c_common.A2CBase.augment_batch_for_mixed_expl = timed(
        "augment_batch_for_mixed_expl (play)",
        a2c_common.A2CBase.augment_batch_for_mixed_expl)

    # --- print and reset once per epoch ----------------------------------
    original_train_epoch = a2c_continuous.A2CAgent.train_epoch

    def train_epoch(self, *args, **kwargs):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        result = original_train_epoch(self, *args, **kwargs)
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
        EPOCH["n"] += 1
        # Skip the first few: allocator growth and any lazy init land there.
        if EPOCH["n"] >= 4:
            _report(wall, torch)
        TOTALS.clear()
        COUNTS.clear()
        return result

    a2c_continuous.A2CAgent.train_epoch = train_epoch


def _report(wall: float, torch) -> None:
    order = [
        "augment_batch_for_mixed_expl (play)",
        "prepare_dataset",
        "central_value.train_net (total)",
        "  central_value.update_dataset",
        "  central_value.train_critic",
        "  _preproc_obs (critic)",
        "actor train_actor_critic (total)",
        "  actor calc_gradients",
        "  _preproc_obs (actor)",
        "  dataset __getitem__",
        "  clip_grad_norm_",
    ]
    print(f"\n=== epoch {EPOCH['n']}  wall {wall:.3f} s "
          f"(peak {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB "
          f"reserved {torch.cuda.max_memory_reserved() / 2**20:,.0f} MiB)",
          flush=True)
    # Indented rows are inside the row above them; only top-level rows sum.
    top_level = sum(v for k, v in TOTALS.items() if not k.startswith("  ")
                    and "(play)" not in k)
    for key in order:
        if key not in TOTALS:
            continue
        seconds, calls = TOTALS[key], COUNTS[key]
        share = 100.0 * seconds / wall
        print(f"  {key:<38} {seconds:7.3f} s  {share:5.1f}% wall  "
              f"n={calls:<4} {1e3 * seconds / max(calls, 1):7.2f} ms/call",
              flush=True)
    print(f"  {'-> update accounted for':<38} {top_level:7.3f} s", flush=True)
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

    # Patching is on the CLASSES, so it only has to happen before Runner.run
    # constructs the agent -- and rl_games is pure torch, so importing it ahead
    # of AppLauncher is safe (unlike anything under isaaclab.*).
    _install_instrumentation()

    import coevolution.train as train_module

    train_module.main()  # parses sys.argv exactly as a normal training run


if __name__ == "__main__":
    main()
