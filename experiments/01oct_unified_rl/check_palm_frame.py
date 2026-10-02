"""Is obs.canonical_palm_frame the same frame on every hand? Per hand, after reset, with the flag on:

  - the canonical axes in WORLD (every hand is mounted palm-up at the same tilt, so x = grasp normal, y,
    z should agree across hands to a few degrees);
  - where the cube's keypoint centroid and the fingertips sit in the canonical frame (cube on +x above
    the palm; fingertips out along +z);
  - palm_extents, and palm_keypoints (axis-aligned, centred at 0).

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/01oct_unified_rl/check_palm_frame.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="multi:uniform")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_apply
    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.pose_reaching_6d.obs_utils.layout import field_offsets

    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_cfg.assets.robot_spec = args.spec
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = 0
        env_cfg.obs.canonical_palm_frame = True
        fields = ("keypoints_rel_ee", "fingertip_pos_rel_ee", "palm_keypoints", "palm_extents", "ee_rot")
        env_cfg.obs.state_list = tuple(env_cfg.obs.state_list) + ("palm_extents", "fingertip_pos_rel_ee")
        env_cfg.domain_randomization.use_obs_delay = False
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()
        for _ in range(args.steps):
            obs, *_ = env.step(torch.zeros(inner.num_envs, inner.action_space.shape[-1], device=inner.device))
        st = obs["critic"]
        off = field_offsets(inner.cfg.obs.state_list, inner.scene_record.robot_spec)
        get = {f: st[:, off[f][0]:off[f][1]] for f in fields}
        q = torch.roll(get["ee_rot"], 1, dims=-1)                      # xyzw -> wxyz
        eye = torch.eye(3, device=inner.device)
        axes_w = torch.stack([quat_apply(q, eye[i].expand(len(q), 3)) for i in range(3)], 1)  # (N, 3, 3)
        hs = inner.scene_record.hand_set
        idx = inner.scene_record.robot_design_index
        specs = hs.specs if hs is not None else (inner.scene_record.robot_spec,)
        ref = None
        print(f"{'hand':10s} {'x_w (normal)':>22s} {'z_w (fingers)':>22s} {'deg vs 1st':>10s} "
              f"{'cube(mm)':>20s} {'tips mean(mm)':>20s} {'extents(mm)':>18s} kp-centroid")
        for h, s in enumerate(specs):
            ids = (idx == h).nonzero(as_tuple=True)[0] if hs is not None else torch.arange(inner.num_envs)
            ax = axes_w[ids].mean(0)
            ax = ax / ax.norm(dim=-1, keepdim=True)
            if ref is None:
                ref = ax
            ang = torch.rad2deg(torch.acos((ax * ref).sum(-1).clamp(-1, 1))).max()
            cube = get["keypoints_rel_ee"][ids].reshape(len(ids), -1, 3).mean((0, 1)) * 1000
            ntip = s.num_fingertips
            tips = get["fingertip_pos_rel_ee"][ids].reshape(len(ids), -1, 3)[:, :ntip].mean((0, 1)) * 1000
            ext = get["palm_extents"][ids[0]] * 1000
            kpc = get["palm_keypoints"][ids[0]].reshape(4, 3)
            kpc = (kpc[0] + 0.5 * (kpc[1:] - kpc[0]).sum(0)).abs().max() * 1000
            f = lambda v: "[" + ", ".join(f"{x:+.2f}" if abs(x) < 10 else f"{x:+.0f}" for x in v.tolist()) + "]"
            print(f"{s.hand_name:10s} {f(ax[0]):>22s} {f(ax[2]):>22s} {float(ang):10.1f} {f(cube):>20s} "
                  f"{f(tips):>20s} {f(ext):>18s} {float(kpc):.2f}mm")
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
