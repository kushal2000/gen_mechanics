"""Scene-equivalence audit: one hand in multi:template: (grammar only) or multi:finger:, built as the evals build it.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/audit/scene_dump.py --scene template|finger \
        --spec <spec> --train-run <run dir> --checkpoint <pth> --out <file.npz> [--mode probe|eval] [--envs 128]

probe (33 envs): every hand joint at 0, the table moved 2 m down (no contacts), the pen parked; then
  * the zero-pose policy observation (env 0), all fields;
  * the policy's deterministic action and value on it;
  * envs 1..31 hold action +0.6 on hand slot k-1 only (env 0 holds 0) for 40 control steps, env 32 a sinusoid on
    every slot: per env every URDF link's position in the palm body frame (physical, by URDF link name), the
    scene-reported canonical joint_pos, and the full observation along env 32's trajectory.
eval: closed loop, stochastic actions, SAPG leader (coef 0), every env's first episode: goals per episode.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=("template", "finger"), required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("probe", "eval"), default="probe")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--table-safe", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=30000)
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841

    import numpy as np
    import torch
    import yaml
    from isaaclab.utils.math import quat_apply_inverse
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.observations import field_sizes
    from isaacsimenvs.inhand_olympics.tasks import NUM_TASK_SLOTS

    full = OmegaConf.to_container(OmegaConf.load(next(Path(args.train_run).rglob(".hydra/config.yaml"))), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    if args.scene == "template":
        mdir = Path(tempfile.mkdtemp())
        (mdir / "manifest.json").write_text(json.dumps({"generation": 0, "grid": GRID,
                                                        "hands": [{"id": "h", "spec": args.spec}]}))
        ref = f"multi:template:{mdir / 'manifest.json'}"
    else:
        ref = f"multi:finger:{args.spec}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = 33 if args.mode == "probe" else args.envs
    cfg.reset.stagger_initial_episodes = False
    cfg.reset.table_safe = bool(args.table_safe)
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    n, dev = env.num_envs, env.device

    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"train": acfg}, f)
    player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, args.checkpoint,
                      device=dev, sapg_expl_coef=0.0, num_envs=n)

    if args.mode == "eval":
        open_ = torch.ones(n, dtype=torch.bool, device=dev)
        goals = torch.zeros(n, device=dev)
        steps = 0
        with torch.no_grad():
            while steps < args.max_steps:
                act, _ = player.get_action_and_value(obs["policy"], deterministic_actions=False)
                obs, rew, term, trunc, extras = env.step(act)
                steps += 1
                done = (term | trunc) & open_
                if done.any():
                    goals = torch.where(done, extras["episode_final"]["successes"].float(), goals)
                    open_ &= ~done
                if steps % 60 == 0 and not bool(open_.any()):
                    break
        goals = torch.where(open_, env.successes.float(), goals)
        res = {"scene": args.scene, "spec": args.spec, "envs": n, "table_safe": args.table_safe, "seed": args.seed,
               "goals_per_episode": float(goals.mean()), "goals_std": float(goals.std()), "steps": steps,
               "goals": goals.tolist()}
        Path(args.out).write_text(json.dumps(res, indent=1))
        print(f"[audit] {args.scene} goals/episode {res['goals_per_episode']:.2f} +- {res['goals_std']:.2f}", flush=True)
        import os
        os._exit(0)

    # ---- probe -------------------------------------------------------------------------------------------
    hs = env.hand_set
    spec = hs.specs[0]
    # physical bodies, by URDF link name: (articulation data, body index) per URDF link
    if args.scene == "template":
        from isaacsimenvs.common.robots.template_robot import template_robot
        from isaacsimenvs.inhand_olympics.wrist import palm_down_rotation, palm_frame
        rob = template_robot(spec, hs.layouts[0], cfg.wrist, palm_down_rotation(palm_frame(spec)))
        art = env.robot
        link_names = sorted(rob.body_of_link)
        link_idx = [art.body_names.index(rob.body_of_link[l]) for l in link_names]
        palm_idx = art.body_names.index(rob.body_of_link[spec.palm_body_name])
    else:
        art = env.robot.arts[0]
        link_names = sorted(b for b in art.body_names if not b.startswith("wrist_"))
        link_idx = [art.body_names.index(l) for l in link_names]
        palm_idx = art.body_names.index(spec.palm_body_name)

    def links_in_palm():
        bs = art.data.body_state_w
        p, q = bs[:, palm_idx, 0:3], bs[:, palm_idx, 3:7]
        rel = bs[:, link_idx, 0:3] - p.unsqueeze(1)
        return quat_apply_inverse(q.unsqueeze(1).expand(-1, len(link_idx), -1).reshape(-1, 4),
                                  rel.reshape(-1, 3)).view(n, len(link_idx), 3)

    # table away, pen parked, every hand joint at 0, wrist at its default, targets there
    top = cfg.table.top_height
    tbl = env.table.data.default_root_state.clone()
    tbl[:, :3] += env.scene.env_origins
    tbl[:, 2] -= 2.0
    env.table.write_root_pose_to_sim(tbl[:, :7])
    ids = torch.arange(n, device=dev)
    pen_pos = torch.tensor([0.0, 0.0, top + cfg.pen.diameter / 2], device=dev).expand(n, 3)
    pen_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
    env.place_pen(ids, pen_pos.clone(), pen_q.clone())
    env.set_goal(ids, torch.tensor([0.0, 0.02, top + 0.08], device=dev).expand(n, 3).clone(), pen_q.clone())
    q = env.robot.data.default_joint_pos.clone()
    q[:, env.hand_ids] = 0.0
    env.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.targets[:] = q
    env.wrist_command[:] = q[:, env.wrist_ids]
    env.robot.set_joint_position_target(q)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.physics_dt)
    env.prev_joint_pos_canon[:] = env.robot.data.joint_pos[:, env.perm_lab_to_canon]
    env.prev_joint_vel_canon[:] = 0.0
    env.closest_goal_dist[:] = math.nan
    env.closest_tip_dist[:] = math.nan
    o0 = env._get_observations()["policy"]
    with torch.no_grad():
        mu0, v0 = player.get_action_and_value(o0, deterministic_actions=True)
    links0 = links_in_palm()
    tips0 = env.fingertips()[0].cpu().numpy()

    sizes = field_sizes(env.cfg.action_space, hs.template.num_hand_joints, hs.template.num_fingertips, NUM_TASK_SLOTS)
    slices, s = {}, 0
    for fld in env.cfg.obs.obs_list:
        slices[fld] = (s, s + sizes[fld])
        s += sizes[fld]

    def raw_step(actions):
        env._pre_physics_step(actions)
        for _ in range(cfg.decimation):
            env._apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)

    T = 40
    nw = 6
    traj_obs, traj_links, traj_q = [], [], []
    for t in range(T):
        act = torch.zeros(n, env.action_space.shape[-1], device=dev)
        for k in range(31):
            act[1 + k, nw + k] = 0.6
        ph = torch.arange(31, device=dev).float() * 0.7
        act[32, nw:] = 0.8 * torch.sin(2 * math.pi * t / 20.0 + ph)
        raw_step(act)
        traj_obs.append(env._get_observations()["policy"][32].cpu().numpy())
        traj_links.append(links_in_palm()[32].cpu().numpy())
        traj_q.append(env.robot.data.joint_pos[32, env.perm_lab_to_canon].cpu().numpy())
    q_end = env.robot.data.joint_pos[:, env.perm_lab_to_canon].cpu().numpy()
    links_end = links_in_palm().cpu().numpy()
    slot_joint = {}
    for j, sl in enumerate(hs.joint_slots[0]):
        slot_joint[int(sl)] = spec.hand_joint_names[j]
    np.savez(args.out, obs0=o0[0].cpu().numpy(), mu0=mu0[0].cpu().numpy(), v0=float(v0[0]),
             links0=links0[0].cpu().numpy(), links_end=links_end, q_end=q_end,
             traj_obs=np.stack(traj_obs), traj_links=np.stack(traj_links), traj_q=np.stack(traj_q),
             link_names=np.array(link_names), slices=json.dumps(slices), slot_joint=json.dumps(slot_joint),
             joint_lower=env.joint_lower_canon[0].cpu().numpy(), joint_upper=env.joint_upper_canon[0].cpu().numpy(),
             joint_home=env.joint_home[0, env.perm_lab_to_canon].cpu().numpy(),
             tips0=tips0)
    print(f"[audit] {args.scene} probe -> {args.out}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
