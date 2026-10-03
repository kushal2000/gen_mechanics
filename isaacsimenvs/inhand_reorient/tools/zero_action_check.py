r"""Roll out zero actions (HORA: the cached grasp's PD targets held) for N
steps and report each design's episodes and mean time to terminate.
Compares a hand alone with the same hand inside a population: with the
same grasps and no policy, its holding time should not depend on what else
is in the scene.

    OMNI_KIT_ACCEPT_EULA=YES timeout -k 30 900 .venv_isaacsim/bin/python -m \
        isaacsimenvs.inhand_reorient.tools.zero_action_check --steps 400 --out OUT.json \
        env.task_profile=hora env.assets.hand_population=POP.json env.scene.num_envs=4096 \
        env.anyrotate.grasp_cache=CACHE.npz env.anyrotate.grasp_cache_generate=false
"""

from __future__ import annotations

import json
import sys


def summarise(snapshot: dict) -> dict:
    out = {}
    for d in snapshot["designs"].values():
        n = int(d["episodes"])
        out[d["source"]] = {
            "episodes": n, "envs": int(d.get("envs_per_design", 0)),
            "ttt_mean_s": (float(d["time_held_sum_s"]) / n) if n else None,
            "rotations_mean": (float(d["rotation_progress_sum_rad"]) / n / 6.283185307179586) if n else None,
        }
    return out


def main() -> None:
    import argparse
    import os

    from isaaclab.app import AppLauncher

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="GenMech-InHandReorient-Direct-v0")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    AppLauncher.add_app_launcher_args(ap)
    args, hydra_args = ap.parse_known_args()
    args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    from isaacsimenvs.inhand_reorient import design_scoring

    @hydra_task_config_with_yaml(args.task, "rl_games_anyrotate_ppo_cfg_entry_point")
    def run(env_cfg, agent_cfg) -> None:
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()
        zero = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
        for _ in range(int(args.steps)):
            env.step(zero)
        rows = summarise(design_scoring.read_snapshot(inner))
        with open(args.out, "w") as f:
            json.dump({"steps": int(args.steps), "num_envs": int(inner.num_envs), "designs": rows}, f, indent=1)
        print(f"[zero_action_check] {json.dumps(rows)}", flush=True)

    run()
    sys.stdout.flush()
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
