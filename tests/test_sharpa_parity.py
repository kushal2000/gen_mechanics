"""Port-correctness gate: genmech reproduces simtoolreal's SHARPA rollout.

Replays the deterministic rollout in ``golden_rollout.py`` and compares against
``tests/data/sharpa_golden.npz``, captured from simtoolreal's Isaac Sim backend
before any refactor landed. The comparison covers observations (policy and
critic), joint targets, every individual reward term, the total reward, and
both termination flags.

**This must stay green through the RobotSpec refactor (M2) and everything after
it.** The reward is ported unchanged by design (docs/methodology.md §2) and the
action-order fix is a no-op for SHARPA — Isaac Lab places the SHARPA arm at
joint indices 0..6, which ``test_action_pipeline.py`` asserts — so any drift
here is a port bug, not an expected consequence of a change.

The baseline was bitwise identical when captured, so the default tolerance is
exact. CUDA reductions are not guaranteed bit-reproducible across driver or
hardware changes, so ``--rtol/--atol`` allow a graded check; a run that passes
only under loosened tolerance prints a warning rather than passing silently.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python tests/test_sharpa_parity.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "data" / "sharpa_golden.npz"


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--rtol", type=float, default=0.0,
                        help="0.0 demands exact equality (the captured baseline was "
                             "bitwise identical); raise only to diagnose.")
    parser.add_argument("--atol", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    import numpy as np

    golden_path = Path(args.golden)
    if not golden_path.exists():
        raise SystemExit(
            f"golden baseline not found: {golden_path}\n"
            f"It is committed with the repo; regenerate from simtoolreal with:\n"
            f"  cd /share/portal/kk837/simtoolreal && OMNI_KIT_ACCEPT_EULA=YES \\\n"
            f"    .venv_isaacsim/bin/python {REPO_ROOT}/tests/capture_golden.py \\\n"
            f"    --backend simtoolreal --out {golden_path}"
        )
    golden = np.load(golden_path)

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import genmech.tasks  # noqa: F401  registers GenMech-PoseReach-Direct-v0
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

    from golden_rollout import build_cfg, run_rollout

    num_envs = int(golden["meta_num_envs"])
    steps = int(golden["meta_steps"])
    seed = int(golden["meta_seed"])
    n_assets = int(golden["meta_num_assets_per_type"])
    action_scale = float(golden["meta_action_scale"])
    print(f"[parity] baseline: num_envs={num_envs} steps={steps} seed={seed} "
          f"assets_per_type={n_assets} action_scale={action_scale}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cfg = build_cfg(PoseReachEnvCfg, num_envs=num_envs, num_assets_per_type=n_assets)
    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped

    for name, got in (("obs_dim", inner.cfg.observation_space),
                      ("state_dim", inner.cfg.state_space),
                      ("act_dim", int(inner.cfg.action_space))):
        want = int(golden[f"meta_{name}"])
        assert got == want, f"{name}: got {got}, baseline {want}"
    print(f"[parity] dims match: obs={inner.cfg.observation_space} "
          f"state={inner.cfg.state_space} act={int(inner.cfg.action_space)}")

    got = run_rollout(env, num_envs=num_envs, steps=steps, seed=seed,
                      action_scale=action_scale)

    compare = [k for k in golden.files if not k.startswith("meta_")]
    failures, loose = [], []
    print(f"\n{'array':26s} {'max|Δ|':>12s} {'mean|Δ|':>12s}  status")
    print("-" * 68)
    for k in sorted(compare):
        want, have = golden[k], got[k]
        assert want.shape == have.shape, f"{k}: shape {have.shape} vs {want.shape}"
        d = np.abs(want.astype(np.float64) - have.astype(np.float64))
        exact = np.array_equal(want, have)
        ok = exact or np.allclose(want, have, rtol=args.rtol, atol=args.atol)
        status = "exact" if exact else ("close" if ok else "FAIL")
        if not ok:
            failures.append(k)
        elif not exact:
            loose.append(k)
        print(f"{k:26s} {d.max():12.3e} {d.mean():12.3e}  {status}")
    print("-" * 68)

    if failures:
        raise AssertionError(
            f"parity FAILED on {len(failures)} array(s): {failures}\n"
            f"genmech no longer reproduces simtoolreal's SHARPA rollout."
        )
    if loose:
        print(
            f"\n[parity] WARNING: {len(loose)} array(s) matched only within tolerance, "
            f"not exactly: {loose}\n"
            f"[parity] The baseline was captured bitwise-identical. Non-exact matches "
            f"mean either a real behavioral change or a CUDA/driver difference — "
            f"confirm which before treating this as a pass."
        )
    else:
        print("\n[parity] all arrays bitwise identical to the simtoolreal baseline")

    print("[parity] SHARPA parity test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
