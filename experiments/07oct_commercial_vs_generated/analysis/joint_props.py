"""What the simulator actually gives each joint and link of a hand, read back from Isaac Lab after the env is built
(the 52-hand policy's config, as the zero-shot evaluation and the live viewer run it): limits, velocity and effort
caps, stiffness, damping, armature, friction, and every link's mass. One process per hand; writes JSON.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/analysis/joint_props.py \
        --spec wuji2_left_uniform_handonly --name wuji2_commercial
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent.parent / "results/joint_props"
sys.path.insert(0, str(REPO))
TASK = "GenMech-InHandReorient-Direct-v0"
CKPT = ("debug_outputs/train_logs/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
        "/rank_0/*/nn/last_*_ep_5000_*.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    spec = args.spec
    if spec.startswith("handonly:") and not spec[9:].startswith("/"):
        spec = "handonly:" + str(REPO / spec[9:])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app

    import gymnasium as gym
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    ckpt = pathlib.Path(sorted(glob.glob(str(REPO / CKPT)))[0])
    saved = OmegaConf.load(ckpt.parents[3] / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = 1
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        env.reset()
        rob = env.robot
        D = rob.data
        view = rob.root_physx_view
        names = list(rob.joint_names)
        hid = [int(i) for i in env._hand_joint_ids]
        get = lambda t: t[0].detach().cpu().tolist()
        stiff, damp = get(view.get_dof_stiffnesses()), get(view.get_dof_dampings())
        arm, fric = get(view.get_dof_armatures()), get(view.get_dof_friction_coefficients())
        vmax, emax = get(view.get_dof_max_velocities()), get(view.get_dof_max_forces())
        lim = get(D.joint_pos_limits)
        joints = {names[i]: {"limits": lim[i], "velocity": vmax[i], "effort": emax[i], "stiffness": stiff[i],
                             "damping": damp[i], "armature": arm[i], "friction": fric[i]} for i in hid}
        masses = get(view.get_masses())
        links = dict(zip(rob.body_names, masses))
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.name}.json").write_text(json.dumps({"spec": args.spec, "joints": joints, "links": links,
                                                           "total_mass": sum(masses)}, indent=1))
        print(f"[props] {args.name}: {len(joints)} hand joints, total mass {sum(masses):.3f} kg", flush=True)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
