"""Can every finger joint reach the ends of its range in sim, and is anything that cannot blocked by self-collision?

One env per pose, all stepped together for 2 s with the wrist held still, then each driven joint's reached position
is compared with its target. Poses (other finger joints at 0 unless said):
    single   each hand joint to its lower limit, and to its upper limit
    thumb    the thumb's 2^4 corners (every thumb joint at its lower or upper limit), other fingers straight
    thumb+fist  the same 16 corners with the other four fingers at their upper (flexed) limits
Run with the robot's self-collision on (as trained) and off (--no-self-collision): a shortfall only with it on is a
collision. --collider convex_decomposition swaps each link's single convex hull for several closer-fitting ones.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python isaacsimenvs/inhand_olympics/tests/joint_range_check.py \\
        [--no-self-collision]
"""
from __future__ import annotations

import argparse
import itertools
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-self-collision", action="store_true")
    ap.add_argument("--collider", default="convex_hull", choices=("convex_hull", "convex_decomposition"))
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=0.05, help="rad; a joint short of its target by more is reported")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import torch

    import isaacsimenvs.inhand_olympics.scene as scene
    if args.no_self_collision:
        _flatten = scene.flatten_robot_usd

        def flatten_without_self_collision(*fa, **fk):
            path, root = _flatten(*fa, **fk)
            from pxr import Usd, UsdPhysics
            stage = Usd.Stage.Open(path)
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.GetAttribute("physxArticulation:enabledSelfCollisions").Set(False)
            stage.GetRootLayer().Save()
            return path, root
        scene.flatten_robot_usd = flatten_without_self_collision

    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    # Poses as {joint name: "lower" | "upper"}.
    probe = InHandOlympicsEnvCfg()
    from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec
    hand = list(get_robot_spec(probe.hand.robot_spec).hand_joint_names)
    thumb = [j for j in hand if "thumb" in j]
    others = [j for j in hand if "thumb" not in j]
    poses, labels = [], []
    for j in hand:
        for end in ("lower", "upper"):
            poses.append({j: end})
            labels.append(f"single {j} -> {end}")
    for corner in itertools.product(("lower", "upper"), repeat=len(thumb)):
        tag = "".join(e[0] for e in corner)
        poses.append(dict(zip(thumb, corner)))
        labels.append(f"thumb {tag}")
        poses.append({**dict(zip(thumb, corner)), **{j: "upper" for j in others}})
        labels.append(f"thumb+fist {tag}")

    cfg = InHandOlympicsEnvCfg()
    cfg.scene.num_envs = len(poses)
    cfg.hand.collider_type = args.collider
    cfg.goal.seconds_per_goal = 1e6
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.finger_pos_fraction = 0.0
    cfg.reset.finger_vel = 0.0
    cfg.action.use_action_delay = False
    cfg.termination.max_goal_distance = 1e3
    cfg.termination.fall_below_table = 1e3
    cfg.wrist.default_pose = (0.0, 0.0, 0.20, 0.0, 0.0, 0.0)        # high above the pen
    env = InHandOlympicsEnv(cfg)
    env.reset()
    names = list(env.robot.joint_names)
    target = env.robot.data.default_joint_pos.clone()
    driven = torch.zeros_like(target, dtype=torch.bool)
    for i, pose in enumerate(poses):
        for j, end in pose.items():
            k = names.index(j)
            target[i, k] = env.joint_lower[i, k] if end == "lower" else env.joint_upper[i, k]
            driven[i, k] = True
    for _ in range(int(args.seconds / env.step_dt)):
        env.step(env.action_toward(target))
    short = ((target - env.robot.data.joint_pos).abs() * driven).cpu()

    mode = ("self-collision OFF" if args.no_self_collision else "self-collision ON") + f", colliders {args.collider}"
    print(f"[range] {mode}: {len(poses)} poses, joints short of target by > {args.tol} rad after {args.seconds:g} s:",
          flush=True)
    bad = 0
    for i, label in enumerate(labels):
        miss = [(names[k], float(short[i, k])) for k in range(len(names)) if short[i, k] > args.tol]
        if miss:
            bad += 1
            print(f"[range]   {label:42s} " + ", ".join(f"{n} {d:.2f}" for n, d in miss), flush=True)
    print(f"[range] {bad} of {len(poses)} poses fall short ({mode})", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
