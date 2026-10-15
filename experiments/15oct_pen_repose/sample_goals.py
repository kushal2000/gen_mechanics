"""Sample pen_repose goals with the task's own code (tasks/pen_repose.py) and save them for viser_goals.py: many first
goals (one per reset) and a few random-walk episodes (a first goal, then next_goal again and again).

The task only needs its env for the config, the goal buffers and a few heights, so a stand-in env is used: no scene,
no physics, so it runs in seconds once Kit has booted (Isaac Lab's math utilities import only inside Kit).

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/15oct_pen_repose/sample_goals.py \\
        [--first 2000 --walks 24 --steps 60 --seed 0 --out <npz>] [--set pen_repose.max_height=0.10 ...]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "debug_outputs/train_logs/15oct_pen_repose/goal_samples.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2000, help="first goals to sample (one per reset)")
    ap.add_argument("--walks", type=int, default=24, help="random-walk episodes")
    ap.add_argument("--steps", type=int, default=60, help="goals per walk after the first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pen-shape", default="box", choices=("box", "capsule"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. pen_repose.max_height=0.10")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--hands", default="", help="comma-separated robot specs to build wrist URDFs for (default: Wuji "
                                                   "full and thumb + index, and every built primitive hand)")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import math

    import numpy as np
    import torch

    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.tasks.pen_repose import PenRepose

    cfg = InHandOlympicsEnvCfg()
    cfg.pen.shape = args.pen_shape
    for kv in args.set:
        key, val = kv.split("=", 1)
        obj = cfg
        *path, last = key.split(".")
        for p in path:
            obj = getattr(obj, p)
        old = getattr(obj, last)
        setattr(obj, last, type(old)(json.loads(val)) if not isinstance(old, tuple) else tuple(json.loads(val)))
    torch.manual_seed(args.seed)

    class StubEnv:
        """Just what PenRepose reads and writes."""

        def __init__(self, n: int):
            self.cfg, self.num_envs, self.device = cfg, n, "cpu"
            self.pen_rest_height = cfg.table.top_height + cfg.pen.diameter / 2
            self.goal_pos = torch.zeros(n, 3)
            self.goal_quat = torch.zeros(n, 4)
            self.goal_quat[:, 0] = 1.0

        def set_goal(self, ids, pos, quat):
            self.goal_pos[ids], self.goal_quat[ids] = pos, quat

        def place_pen(self, ids, pos, quat):
            pass

        def start_pen_yaw(self, n):
            return (torch.rand(n) * 2 - 1) * math.pi

    env = StubEnv(args.first)
    task = PenRepose(env)
    task.reset(torch.arange(args.first))
    first_pos, first_quat = env.goal_pos.clone(), env.goal_quat.clone()

    env = StubEnv(args.walks)
    task = PenRepose(env)
    ids = torch.arange(args.walks)
    task.reset(ids)
    walk_pos, walk_quat = [env.goal_pos.clone()], [env.goal_quat.clone()]
    for _ in range(args.steps):
        task.next_goal(ids)
        walk_pos.append(env.goal_pos.clone())
        walk_quat.append(env.goal_quat.clone())

    # Each hand on the env's dummy wrist (wrist.build_wrist_urdf: the URDF the env converts), for the viewer.
    from isaacsimenvs.inhand_olympics.wrist import build_wrist_urdf
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [h for h in args.hands.split(",") if h] or (
        ["wuji2_left_uniform_handonly", "wuji2_left_uniform_handonly_only_thumb_index"]
        + sorted(str(p.relative_to(REPO)) for p in (REPO / "assets/urdf/generated/14oct_design_search").glob("*.spec.json")))
    hands = []
    for name in names:
        label = pathlib.Path(name).name.removesuffix(".spec.json")
        urdf = build_wrist_urdf(get_robot_spec(name), cfg.wrist, out.parent / "hand_urdfs" / f"{label}.urdf")
        hands.append({"name": label, "urdf": str(urdf)})

    c = cfg.pen_repose
    np.savez(out, first_pos=first_pos.numpy(), first_quat=first_quat.numpy(),
             walk_pos=torch.stack(walk_pos, 1).numpy(), walk_quat=torch.stack(walk_quat, 1).numpy(),
             meta=json.dumps({
                 "table_top": cfg.table.top_height, "table_size": list(cfg.table.size),
                 "pen": [cfg.pen.length, cfg.pen.diameter, cfg.pen.shape],
                 "pen_rest_height": env.pen_rest_height, "centre_xy": list(c.centre_xy),
                 "half_width_xy": c.half_width_xy, "min_clearance": c.min_clearance, "max_height": c.max_height,
                 "step_pos": c.step_pos, "step_deg": list(c.step_deg), "random_first_goal": c.random_first_goal,
                 "seed": args.seed, "overrides": args.set, "hands": hands,
                 "wrist_default": dict(zip(("wrist_x", "wrist_y", "wrist_z", "wrist_roll", "wrist_pitch", "wrist_yaw"),
                                           map(float, cfg.wrist.default_pose)))}))
    print(f"[sample_goals] wrote {out}: {args.first} first goals, {args.walks} walks x {args.steps + 1} goals", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
