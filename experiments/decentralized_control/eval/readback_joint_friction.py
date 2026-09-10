"""Boot the env and read the hand's joint friction back out of PhysX.

The two eval arms differed by 0.3%, which is inside noise -- that is also what
a flag that silently did nothing would look like. Read the number instead.
"""
import argparse, os, pathlib, sys

parser = argparse.ArgumentParser()
parser.add_argument("--friction", type=int, default=0)
args = parser.parse_args()
sys.path.insert(0, "/share/portal/kk837/gen_mechanics")

from isaaclab.app import AppLauncher
app_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(app_parser)
app_args = app_parser.parse_args([])
app_args.headless = True
app = AppLauncher(app_args).app

import gymnasium as gym
import torch
import isaacsimenvs  # noqa: F401
from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg
from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec

cfg = PoseReachEnvCfg()
cfg.scene.num_envs = 16
cfg.assets.num_assets_per_type = 4
cfg.assets.robot_spec = "sharpa_iiwa14"
cfg.assets.apply_hand_joint_friction = bool(args.friction)
env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg).unwrapped
env.reset()

spec = get_robot_spec("sharpa_iiwa14")
names = list(env.robot.joint_names)
view = env.robot.root_physx_view
print("\nphysx view friction accessors:", flush=True)
for attr in sorted(a for a in dir(view) if "friction" in a.lower()):
    print(f"   view.{attr}")
print("isaaclab data friction attrs:", flush=True)
for attr in sorted(a for a in dir(env.robot.data) if "friction" in a.lower()):
    print(f"   data.{attr}")

readback = None
for attr in ("get_dof_friction_properties", "get_dof_friction_coefficients"):
    if hasattr(view, attr):
        try:
            readback = (attr, getattr(view, attr)())
            break
        except Exception as exc:
            print(f"   {attr} raised {exc!r}")

print(f"\n===== apply_hand_joint_friction={bool(args.friction)} =====", flush=True)
print(f"physx accessor: {readback[0] if readback else 'NONE FOUND'}")
if readback is not None:
    vals = readback[1]
    vals = vals.cpu().numpy() if hasattr(vals, "cpu") else vals
    row = vals[0]
    # (num_dofs, 3) in Isaac Sim 5.x: static, dynamic, viscous. Column 0 is the
    # one ImplicitActuatorCfg.friction feeds.
    print(f"  raw shape {tuple(row.shape)} -> per-dof {row.reshape(len(names), -1).shape[1]} column(s)")
    row = row.reshape(len(names), -1)[:, 0]
    total = 0.0
    for n, v in zip(names, row):
        want = spec.hand_friction.get(n)
        if want is not None:
            total += float(v)
            print(f"  {n:<28} physx={float(v):<12.6f} spec={want:<12.6f} "
                  f"{'MATCH' if abs(float(v) - want) < 1e-6 else 'DIFFER'}")
    print(f"  sum over hand joints = {total:.8f}")
print(f"isaaclab data.joint_friction[0][:3] = "
      f"{getattr(env.robot.data, 'joint_friction', getattr(env.robot.data, 'joint_friction_coeff', None))}"[:200])
# app.close() hangs under Kit 5.1 headless; os._exit skips the teardown.
os._exit(0)
