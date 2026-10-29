"""Does the template scene simulate the same hand as the URDF scene? Drive one hand through both with the same action
sequence and compare joint positions, fingertips and the pen.

    python experiments/29oct_template_scene/equivalence.py record --spec <spec.json> --out a.npz
    python experiments/29oct_template_scene/equivalence.py record --spec multi:template:<spec.json> --out b.npz
    python experiments/29oct_template_scene/equivalence.py compare a.npz b.npz

Recording boots Kit (headless). The hand starts at its default pose (finger_pos_fraction 0), the pen at its init
pose, no action delay, palm rotation locked; actions are sinusoids per hand joint, applied in the hand's own joint
order on both paths (the template path scatters them to the hand's slots).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OLD_OBS = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets", "joint_link_bbox",
           "joint_lower", "joint_upper", "joint_enabled", "keypoints_rel_ee", "keypoints_rel_goal", "object_vel",
           "palm_extents", "task_onehot"]


def record(spec: str, out: str, envs: int, steps: int) -> None:
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    import torch
    import isaacsimenvs  # noqa: F401
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.wrist import WRIST_JOINTS

    cfg = InHandOlympicsEnvCfg()
    cfg.hand.robot_spec = spec
    cfg.scene.num_envs = envs
    cfg.task.names = ("pen_repose",)
    cfg.pen.shape = "box"
    cfg.obs.obs_list = tuple(OLD_OBS)
    cfg.obs.state_list = tuple(OLD_OBS)
    cfg.action.finger_target = "home"
    cfg.action.use_action_delay = False
    cfg.action.wrist_max_speed_ang = 0.0
    cfg.reset.finger_pos_fraction = 0.0
    cfg.reset.stagger_initial_episodes = False
    cfg.seed = 0
    env = InHandOlympicsEnv(cfg)
    env.reset()
    n_w = len(WRIST_JOINTS)
    template = env.hand_set is not None
    if template:
        hand_joints = list(env.hand_set.specs[0].hand_joint_names)
        slots = list(env.hand_set.joint_slots[0])                      # hand joint j -> slot (canonical column n_w + slot)
        tip_slots = list(env.hand_set.finger_slots[0])
        real_cols = env.robot.real_slots(0)                            # lab columns of wrist + hand joints
    else:
        hand_joints = list(env.robot_spec.hand_joint_names)
        slots = list(range(len(hand_joints)))
        tip_slots = list(range(env.robot_spec.num_fingertips))
        canon = list(WRIST_JOINTS) + hand_joints
        real_cols = [list(env.robot.joint_names).index(n) for n in canon]
    n_act = env.action_space.shape[-1]
    q, tips, pen, palm = [], [], [], []
    for t in range(steps):
        act = torch.zeros(envs, n_act, device=env.device)
        for j in range(len(hand_joints)):
            act[:, n_w + slots[j]] = 0.8 * math.sin(0.15 * t + 0.7 * j)
        env.step(act)
        q.append(env.robot.data.joint_pos[0, real_cols].cpu().numpy().copy())
        tips.append(env.fingertips()[0, tip_slots].cpu().numpy().copy())
        p, r = env.pen_pose()
        pen.append(np.concatenate([p[0].cpu().numpy(), r[0].cpu().numpy()]))
        pp, pr = env.palm_pose()
        palm.append(np.concatenate([pp[0].cpu().numpy(), pr[0].cpu().numpy()]))
    lim = env.joint_lower_hand[0].cpu().numpy(), env.joint_upper_hand[0].cpu().numpy()
    np.savez(out, q=np.array(q), tips=np.array(tips), pen=np.array(pen), palm=np.array(palm),
             lower=lim[0][slots], upper=lim[1][slots], joints=np.array(hand_joints), spec=spec)
    print(f"wrote {out}: {steps} steps, {len(hand_joints)} hand joints, {len(tip_slots)} tips", flush=True)
    app.close()


def compare(a: str, b: str) -> None:
    A, B = np.load(a), np.load(b)
    assert list(A["joints"]) == list(B["joints"]), (A["joints"], B["joints"])
    for k, unit in (("q", "rad"), ("tips", "m"), ("pen", "m / quat"), ("palm", "m / quat"), ("lower", "rad"),
                    ("upper", "rad")):
        d = np.abs(A[k] - B[k])
        print(f"{k:6s} max |diff| {d.max():.2e} {unit}   (mean {d.mean():.2e})")
    dq = np.abs(A["q"] - B["q"])
    worst = np.unravel_index(dq.argmax(), dq.shape)
    print(f"worst joint diff at step {worst[0]}, column {worst[1]}: {A['q'][worst]:.4f} vs {B['q'][worst]:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--spec", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--envs", type=int, default=8)
    r.add_argument("--steps", type=int, default=90)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.spec, args.out, args.envs, args.steps)
    else:
        compare(args.a, args.b)
