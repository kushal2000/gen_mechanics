"""Capture a deterministic rollout trace, from either this repo or simtoolreal.

This writes the port-correctness baseline. ``genmech`` is a rename-and-
restructure of simtoolreal's Isaac Sim backend, so before any behavioral
refactor lands we need proof the physics, observations, and reward are
unchanged. The same script runs against both backends;
``test_sharpa_parity.py`` then replays the rollout and compares.

The rollout itself — every pinned config field, the action sequence, the
recorded arrays — lives in ``golden_rollout.py``, shared with the test so the
two cannot drift.

Usage (note the CWD: simtoolreal resolves asset paths against it, genmech
against REPO_ROOT):

    # baseline, from simtoolreal
    cd /share/portal/kk837/simtoolreal
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        /share/portal/kk837/gen_mechanics/tests/capture_golden.py \\
        --backend simtoolreal \\
        --out /share/portal/kk837/gen_mechanics/tests/data/sharpa_golden.npz

    # port, from gen_mechanics
    cd /share/portal/kk837/gen_mechanics
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python tests/capture_golden.py \\
        --backend genmech --out /tmp/sharpa_port.npz
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


BACKENDS = {
    "genmech": {
        "register_module": "genmech.tasks",
        "cfg_module": "genmech.tasks.pose_reach.env_cfg",
        "cfg_class": "PoseReachEnvCfg",
        "task_id": "GenMech-PoseReach-Direct-v0",
    },
    "simtoolreal": {
        "register_module": "isaacsimenvs",
        "cfg_module": "isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg",
        "cfg_class": "SimToolRealEnvCfg",
        "task_id": "Isaacsimenvs-SimToolReal-Direct-v0",
    },
}


def main() -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from golden_rollout import DEFAULTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num_envs", type=int, default=DEFAULTS["num_envs"])
    parser.add_argument("--num_assets_per_type", type=int,
                        default=DEFAULTS["num_assets_per_type"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--action_scale", type=float, default=DEFAULTS["action_scale"])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import importlib

    import gymnasium as gym
    import numpy as np
    import torch

    from golden_rollout import build_cfg, run_rollout

    backend = BACKENDS[args.backend]
    importlib.import_module(backend["register_module"])  # gym.register side effect
    cfg_cls = getattr(importlib.import_module(backend["cfg_module"]), backend["cfg_class"])

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = build_cfg(cfg_cls, num_envs=args.num_envs,
                    num_assets_per_type=args.num_assets_per_type)
    env = gym.make(backend["task_id"], cfg=cfg)
    inner = env.unwrapped

    n_obs, n_state = inner.cfg.observation_space, inner.cfg.state_space
    n_act = int(inner.cfg.action_space)
    print(f"[golden] backend={args.backend} obs={n_obs} state={n_state} act={n_act}")

    arrays = run_rollout(env, num_envs=args.num_envs, steps=args.steps,
                         seed=args.seed, action_scale=args.action_scale)

    # Recorded so the test can refuse to compare traces from different rollouts.
    for key, val in dict(num_envs=args.num_envs, steps=args.steps, seed=args.seed,
                         num_assets_per_type=args.num_assets_per_type,
                         obs_dim=n_obs, state_dim=n_state, act_dim=n_act).items():
        arrays[f"meta_{key}"] = np.asarray(val)
    arrays["meta_action_scale"] = np.asarray(args.action_scale, dtype=np.float64)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"[golden] wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(
        f"[golden] summary: mean_reward={arrays['reward'].mean():.4f} "
        f"terminated={int(arrays['terminated'].sum())} "
        f"truncated={int(arrays['truncated'].sum())}"
    )

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
