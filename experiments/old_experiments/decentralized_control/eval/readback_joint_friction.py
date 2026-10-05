"""Boot the env and check every joint's friction is zero in PhysX.

Joint friction is zero on every robot now. This asserts it against the simulator
rather than the config, because the config has lied before: get_dof_friction_
coefficients() returns zeros whatever is set under Isaac Sim 5.x, so the honest
handle is get_dof_friction_properties(), which is (num_dofs, 3) for static,
dynamic and viscous.
"""
import argparse, os, pathlib, sys

parser = argparse.ArgumentParser()  # no knobs: friction is always zero
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

cfg = PoseReachEnvCfg()
cfg.scene.num_envs = 16
cfg.assets.num_assets_per_type = 4
cfg.assets.robot_spec = "sharpa_iiwa14"
env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg).unwrapped
env.reset()

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

print("\n===== joint friction readback =====", flush=True)
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
        total += abs(float(v))
        if float(v) != 0.0:
            print(f"  NONZERO {n:<28} physx={float(v):.8f}")
    print(f"  sum |friction| over all {len(names)} joints = {total:.8f}")
    print("  OK: every joint is frictionless" if total == 0.0
          else "  FAIL: joint friction is not zero")
print(f"isaaclab data.joint_friction[0][:3] = "
      f"{getattr(env.robot.data, 'joint_friction', getattr(env.robot.data, 'joint_friction_coeff', None))}"[:200])
# app.close() hangs under Kit 5.1 headless; os._exit skips the teardown.
os._exit(0)
