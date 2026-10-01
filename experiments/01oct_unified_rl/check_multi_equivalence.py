"""Does a hand behave the same inside a multi-hand scene as alone? Same actions, compare statistics.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/01oct_unified_rl/check_multi_equivalence.py \
        --spec allegro_left_uniform_handonly --num-envs 1024 --out /tmp/single.json
    ... --spec multi:allegro_left_uniform_handonly+dex3_left_uniform_handonly --num-envs 2048 --out /tmp/multi.json

Joint-reset noise is off; every env gets the SAME open-loop action sequence (a slow sine per joint slot,
phase by slot), so a hand's envs differ only in the cube's random start orientation. Recorded per hand:
the mean and std over envs of each real joint's position at fixed steps, the mean cube height above the
palm and the fraction of envs that dropped it. A wrong row or joint mapping in the multi-hand wrapper
shows up as a gross mismatch; the cube statistics agree within sampling noise.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import gymnasium as gym
    import torch
    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    sys.argv = [sys.argv[0]]
    out = {}

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_cfg.assets.robot_spec = args.spec
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = 0
        env_cfg.obs.obs_list = tuple(env_cfg.obs.state_list)
        env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        env_cfg.reset.reset_dof_vel_random_interval = 0.0
        env_cfg.action.hand_moving_average = 1.0
        env_cfg.domain_randomization.use_action_delay = False
        env_cfg.domain_randomization.use_obs_delay = False
        for k in ("robot_friction", "finger_tip_friction", "object_friction"):
            setattr(env_cfg.assets, k, 0.5)
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        env.reset()
        J = inner.action_space.shape[-1]
        hs = getattr(inner.scene_record, "hand_set", None)
        if hs is None:
            groups = {inner.scene_record.robot_spec.hand_name: (torch.arange(inner.num_envs, device=inner.device),
                                                               inner.scene_record.robot_spec.num_hand_joints)}
        else:
            idx = inner.scene_record.robot_design_index
            groups = {s.hand_name: ((idx == h).nonzero(as_tuple=True)[0], s.num_hand_joints)
                      for h, s in enumerate(hs.specs)}
        slots = torch.arange(J, device=inner.device, dtype=torch.float32)
        rec = {name: {"q": {}, "cube_h": [], "dropped": torch.zeros(len(ids), device=inner.device)}
               for name, (ids, n) in groups.items()}
        for t in range(args.steps):
            act = 0.6 * torch.sin(0.05 * t + 0.7 * slots).unsqueeze(0).expand(inner.num_envs, J)
            env.step(act)
            q = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
            h = (inner.object.data.root_pos_w - inner._palm_center_pos_w).norm(dim=-1)
            fall = inner._termination_reasons["fall"]
            for name, (ids, n) in groups.items():
                rec[name]["dropped"] = torch.maximum(rec[name]["dropped"], fall[ids].float())
                rec[name]["cube_h"].append(float(h[ids].mean()))
                if t in (19, 99, args.steps - 1):
                    rec[name]["q"][t] = {"mean": q[ids, :n].mean(0).tolist(), "std": q[ids, :n].std(0).tolist()}
        for name, r in rec.items():
            out[name] = {"q": r["q"], "cube_dist_mean": r["cube_h"][::20],
                         "drop_frac": float(r["dropped"].mean()), "n_envs": int(groups[name][0].numel())}
        env.close()

    _run()
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print("[equiv] wrote", args.out, flush=True)
    app.close()


if __name__ == "__main__":
    main()
