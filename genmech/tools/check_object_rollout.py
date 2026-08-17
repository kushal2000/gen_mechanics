"""What do authored objects DO in the real env, with the policy taking no action?

The isolated drop test (compare_object_physics) passed to 2e-7 m -- but it drops
objects onto a bare ground plane. The task does something different: it resets
the object onto a table, inside a scene with a robot, using the env's own reset
logic. And the one measured anomaly that survives every asset comparison is that
the robot's joint_vel observation is 37% noisier under authored objects WITH
ZERO ACTIONS. With no action and no contact, the robot should not move at all,
so something is touching it, or something is moving that should be still.

This records, per step, with zero actions:

  * object position and orientation
  * object linear and angular speed
  * robot joint velocity magnitude
  * object height above the table surface

Run once per path and diff. A settling transient in one and not the other means
the object spawns interpenetrating; a persistent velocity means it never comes
to rest; a height offset means the collider sits differently than the reset
logic assumes.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.check_object_rollout --out conv.npz
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.check_object_rollout --author --out auth.npz \\
        --compare conv.npz
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--author", action="store_true")
    parser.add_argument("--robot_spec", default="sharpa_iiwa14")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import genmech.tasks  # noqa: F401
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.author_object_usds = args.author
    cfg.assets.robot_spec = args.robot_spec

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    tag = "authored" if args.author else "converted"

    # Seed everything the same way in both runs so the object poses, goals and
    # asset draw are identical -- otherwise a difference in the rollout says
    # nothing about the asset path.
    torch.manual_seed(0)
    np.random.seed(0)
    obs, _ = env.reset(seed=0)

    # Materials as PhysX actually holds them, after the env's own setup ran.
    try:
        mats = inner.object.root_physx_view.get_material_properties().cpu().numpy()
        print(f"[roll:{tag}] object materials shape {mats.shape} "
              f"static[min,max]=({mats[..., 0].min():.4f},{mats[..., 0].max():.4f}) "
              f"dynamic=({mats[..., 1].min():.4f},{mats[..., 1].max():.4f}) "
              f"restitution=({mats[..., 2].min():.4f},{mats[..., 2].max():.4f})")
    except Exception as exc:                                   # noqa: BLE001
        print(f"[roll:{tag}] material readback failed: {exc}")
        mats = np.zeros((1, 1, 3))

    act = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)

    pos, quat, lin, ang, jvel = [], [], [], [], []
    for _ in range(args.steps):
        env.step(act)
        pos.append(inner.object.data.root_pos_w.clone().cpu().numpy())
        quat.append(inner.object.data.root_quat_w.clone().cpu().numpy())
        lin.append(inner.object.data.root_lin_vel_w.norm(dim=-1).cpu().numpy())
        ang.append(inner.object.data.root_ang_vel_w.norm(dim=-1).cpu().numpy())
        jvel.append(inner.robot.data.joint_vel.abs().max(dim=-1).values.cpu().numpy())

    pos = np.stack(pos)       # (T, N, 3)
    quat = np.stack(quat)
    lin = np.stack(lin)       # (T, N)
    ang = np.stack(ang)
    jvel = np.stack(jvel)

    # Height relative to each env's origin, so table offset cancels.
    origins = inner.scene.env_origins.cpu().numpy()
    z_rel = pos[:, :, 2] - origins[None, :, 2]

    print(f"[roll:{tag}] step   z_rel(mean)   |v|(mean)   |w|(mean)   "
          f"robot|qd|(mean)")
    for t in (0, 1, 2, 5, 10, 20, 40, 80, args.steps - 1):
        if t >= args.steps:
            continue
        print(f"[roll:{tag}] {t:>4}   {z_rel[t].mean():>10.5f}   "
              f"{lin[t].mean():>9.5f}   {ang[t].mean():>9.5f}   "
              f"{jvel[t].mean():>13.6f}")

    # A spawn interpenetration shows up as a large FIRST-step speed that decays.
    print(f"[roll:{tag}] peak |v| over all steps/envs: {lin.max():.4f} m/s "
          f"(step {int(np.unravel_index(lin.argmax(), lin.shape)[0])})")
    print(f"[roll:{tag}] peak |w|: {ang.max():.4f} rad/s")
    print(f"[roll:{tag}] settled |v| (last step, max over envs): "
          f"{lin[-1].max():.6f} m/s")
    print(f"[roll:{tag}] robot joint_vel std over rollout: {jvel.std():.6f}")

    np.savez(args.out, pos=pos, quat=quat, lin=lin, ang=ang, jvel=jvel,
             z_rel=z_rel, mats=mats, tag=tag)
    print(f"[roll:{tag}] wrote {args.out}")

    if args.compare and os.path.exists(args.compare):
        o = np.load(args.compare)
        otag = str(o["tag"])
        print(f"\n[roll] vs {otag}:")
        dz = np.abs(z_rel - o["z_rel"])
        print(f"[roll]   z_rel  max diff {dz.max():.6f} m "
              f"(step 0: {dz[0].max():.6f}, final: {dz[-1].max():.6f})")
        print(f"[roll]   |v|    max diff {np.abs(lin - o['lin']).max():.6f} m/s")
        print(f"[roll]   robot |qd| mean {jvel.mean():.6f} vs {o['jvel'].mean():.6f}")
        print(f"[roll]   peak |v| {lin.max():.4f} vs {float(o['lin'].max()):.4f}")
        om = o["mats"]
        if om.shape == mats.shape:
            print(f"[roll]   material max diff {np.abs(mats - om).max():.6f}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
