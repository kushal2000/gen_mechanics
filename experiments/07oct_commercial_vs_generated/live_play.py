"""LIVE: a policy acting on one hand in Isaac Sim, drawn in viser as it runs.

The zero-shot evaluation's setup (the policy's run config, the hand swapped in via env.assets.robot_spec, the
network built for its joints), stepped in real time with viser updated every step. Controls: play / pause, speed,
single step, reset, which env to watch, greedy or sampled actions. The hand is drawn in its own canonical palm frame
(palm centre at the origin) with its cube and the goal; the panel counts goals and drops live.

Isaac Sim builds ONE hand per process, so to compare two hands run two of these on two ports:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/live_play.py \
        --spec wuji2_left_uniform_handonly --name wuji2_commercial --port 8086
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/live_play.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica_clone.json \
        --name wuji2_replica_clone --port 8087
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
TASK = "GenMech-InHandReorient-Direct-v0"
DEFAULT_CKPT = ("debug_outputs/train_logs/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
                "/rank_0/*/nn/last_*_ep_5000_*.pth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="registry name, or handonly:<population .json>")
    ap.add_argument("--name", required=True, help="for the replica/vendor URDF lookup, e.g. wuji2_replica_clone")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--port", type=int, default=8086)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    spec = args.spec
    if spec.startswith("handonly:") and not spec[9:].startswith("/"):
        spec = "handonly:" + str(REPO / spec[9:])
    ckpt = pathlib.Path(args.checkpoint or sorted(glob.glob(str(REPO / DEFAULT_CKPT)))[0])

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import viser
    import yaml
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation as Rot
    from viser.extras import ViserUrdf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.eval._play_worker import _resize_sigma
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaaclab.utils.math import quat_mul
    from viser_spawn import urdf_of

    saved = OmegaConf.load(ckpt.parents[3] / "rank_0/.hydra/config.yaml")
    saved.env.assets.robot_spec = spec
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name,
                          _resize_sigma(str(ckpt), int(inner.action_space.shape[-1])),
                          device=args.device, sapg_expl_coef=0.0, num_envs=inner.num_envs)
        dt = float(inner.step_dt)
        R = inner.robot.data
        org = inner.scene.env_origins
        jn = list(inner.robot.joint_names)

        server = viser.ViserServer(port=args.port)
        server.scene.set_up_direction("+z")
        st = {"playing": True, "step_once": False, "reset": False, "env": 0,
              "goals": np.zeros(inner.num_envs, int), "drops": np.zeros(inner.num_envs, int),
              "ep_goals": np.zeros(inner.num_envs, int), "t": 0}
        with server.gui.add_folder(f"live: {args.name}"):
            b_play = server.gui.add_button("play / pause")
            b_step = server.gui.add_button("step once")
            b_reset = server.gui.add_button("reset all envs")
            g_speed = server.gui.add_slider("speed (x real time)", 0.1, 2.0, 0.1, 1.0)
            g_env = server.gui.add_slider("env to watch", 0, max(inner.num_envs - 1, 1), 1, 0)
            g_greedy = server.gui.add_checkbox("greedy actions (as evaluated)", True)
            md = server.gui.add_markdown("")
        b_play.on_click(lambda _: st.update(playing=not st["playing"]))
        b_step.on_click(lambda _: st.update(step_once=True))
        b_reset.on_click(lambda _: st.update(reset=True))

        nodes: dict = {}
        robot = {"r": None}

        def place_root(env):
            ps = R.body_state_w[env, inner._palm_body_id, :]
            canon = quat_mul(ps[None, 3:7], inner._palm_frame_quat[env:env + 1])[0].cpu().numpy()
            w, x, y, z = canon
            Rc = Rot.from_quat([x, y, z, w]).as_matrix()
            c = (ps[0:3] - org[env]).cpu().numpy()
            q = Rot.from_matrix(Rc.T).as_quat()
            if "root" in nodes:
                nodes["root"].remove()
            nodes["root"] = server.scene.add_frame("/hand", position=tuple(-Rc.T @ c), wxyz=(q[3], q[0], q[1], q[2]),
                                                   show_axes=False)
            nodes["base"] = server.scene.add_frame("/hand/base", show_axes=False)
            urdf = urdf_of(args.name)
            robot["r"] = (ViserUrdf(server, urdf, root_node_name="/hand/base/robot",
                                    mesh_color_override=(0.85, 0.55, 0.2, 0.9) if "commercial" in args.name
                                    else (0.2, 0.45, 0.85, 0.9)) if urdf is not None else None)
            size = tuple(inner.cfg.reward.fixed_size)
            nodes["obj"] = server.scene.add_box("/hand/object", dimensions=size, color=(0.9, 0.6, 0.2), opacity=0.8)
            nodes["goal"] = server.scene.add_box("/hand/goal", dimensions=size, color=(0.3, 0.8, 0.3), opacity=0.3)
            st["env"] = env

        def draw():
            e = st["env"]
            rs = R.root_state_w[e]
            p = (rs[0:3] - org[e]).cpu().numpy()
            nodes["base"].position, nodes["base"].wxyz = tuple(p), tuple(rs[3:7].cpu().numpy())
            r = robot["r"]
            if r is not None:
                q = dict(zip(jn, R.joint_pos[e].cpu().numpy()))
                r.update_cfg(np.array([q.get(j, 0.0) for j in r.get_actuated_joint_names()]))
            for k, obj in (("obj", inner.object.data), ("goal", inner.goal_viz.data)):
                nodes[k].position = tuple((obj.root_pos_w[e] - org[e]).cpu().numpy())
                nodes[k].wxyz = tuple(obj.root_quat_w[e].cpu().numpy())
            md.content = (f"`{args.name}` env {e}, t = {st['t'] * dt:.1f} s\n\n"
                          f"this episode **{st['ep_goals'][e]}** goals | total {st['goals'][e]} goals, "
                          f"{st['drops'][e]} drops\n\nall envs: {st['goals'].sum()} goals, {st['drops'].sum()} drops")

        place_root(0)
        draw()
        print(f"[live] {args.name} on port {args.port}", flush=True)
        while app.is_running():
            if int(g_env.value) != st["env"]:
                place_root(int(g_env.value))
            if st["reset"]:
                obs, _ = env.reset()
                st.update(reset=False, t=0)
                st["goals"][:] = 0; st["drops"][:] = 0; st["ep_goals"][:] = 0
            if st["playing"] or st["step_once"]:
                t0 = time.time()
                with torch.no_grad():
                    act = player.get_normalized_action(obs["policy"], deterministic_actions=g_greedy.value)
                obs, _, _, _, _ = env.step(act)
                succ = inner._is_success.cpu().numpy().astype(int)
                r = inner._termination_reasons
                fall = r["fall"].cpu().numpy().astype(int)
                end = (r["fall"] | r["timeout"] | r["hand_far"] | r["max_successes"]).cpu().numpy()
                st["goals"] += succ; st["drops"] += fall; st["ep_goals"] += succ
                st["ep_goals"][end] = 0
                st["t"] += 1
                st["step_once"] = False
                draw()
                time.sleep(max(0.0, dt / g_speed.value - (time.time() - t0)))
            else:
                time.sleep(0.05)
        env.close()

    _run()
    app.close()


if __name__ == "__main__":
    main()
