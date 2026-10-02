"""Live viewer: several policies on the SAME hand, side by side, in viser (for screen recording).

Default: Wuji v2 driven by the unified policy (left) and by the Wuji-only policy (right), each policy's
newest checkpoint. Both policies observe the canonical palm frame + palm_extents, so one env config serves
both; the unified network is rebuilt for the hand's joints (every weight is shared across joint tokens;
sigma, the one per-joint tensor, is resized -- it is never read by the greedy action).

    .venv_isaacsim/bin/python experiments/01oct_unified_rl/viser_compare.py --port 8081 \
        [--hand wuji2] [--policy "Unified=<.pth>" --policy "Wuji only=<.pth>"]

The scene is the hand's single-hand env (<hand>_left_uniform_handonly), one env per policy, under the env
config of the LAST policy's run; every policy's observation list must match it.
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Client, Listener

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from viser_unified import NAMES, TASK, _face_cube  # noqa: E402

LOGS = REPO / "debug_outputs/train_logs"


def newest(pattern: str) -> str:
    pths = glob.glob(pattern)
    if not pths:
        raise SystemExit(f"no checkpoints match {pattern}")
    return max(pths, key=os.path.getmtime)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="wuji2")
    ap.add_argument("--policy", action="append", default=[], help="LABEL=checkpoint.pth (repeatable)")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--spacing", type=float, default=0.32)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--host", default="")
    ap.add_argument("--wport", type=int, default=0)
    ap.add_argument("--authkey", default="")
    a = ap.parse_known_args()[0]
    if not a.policy:
        a.policy = [
            "Unified (8 hands)=" + newest(f"{LOGS}/01oct_unified_rl/0_scale_train_left_multi_uniform_canon_c01_*"
                                          "/rank_0/*/nn/last_*.pth"),
            f"{NAMES.get(a.hand, a.hand)} only=" + newest(
                f"{LOGS}/01oct_uniform_dynamics/0_scale_train_left_{a.hand}_uniform_canon_c01_*/rank_0/*/nn/last_*.pth"),
        ]
    return a


def worker(args):
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _loop(args, conn)
    except Exception:
        import traceback
        conn.send({"kind": "error", "text": traceback.format_exc()})
    conn.close()
    app.close()
    os._exit(0)


def _loop(args, conn):
    import gymnasium as gym
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval._play_worker import _resize_sigma
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    spec_name = f"{args.hand}_left_uniform_handonly"
    pols = []
    for p in args.policy:
        label, ck = p.split("=", 1)
        saved = OmegaConf.load(pathlib.Path(ck).parents[3] / "rank_0/.hydra/config.yaml")
        saved.env.assets.robot_spec = spec_name           # the network is built for THIS hand's joints
        pols.append((label, ck, saved))
    obs_lists = {tuple(s.env.obs.obs_list) for _, _, s in pols}
    frames = {bool(s.env.obs.canonical_palm_frame) for _, _, s in pols}
    if len(obs_lists) != 1 or len(frames) != 1:
        raise RuntimeError(f"policies observe different things: {obs_lists} / canonical {frames}")
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(pols[-1][2].env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = len(pols)
        env_cfg.seed = 0
        env_cfg.termination.max_consecutive_successes = 0
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        n_act = env.action_space.shape[-1]
        players = []
        for label, ck, saved in pols:
            acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
            acfg["params"]["config"]["multi_gpu"] = False
            f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
            yaml.safe_dump({"train": acfg}, f)
            f.close()
            players.append(RlPlayer(obs["policy"].shape[-1], n_act, f.name, _resize_sigma(ck, n_act),
                                    device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs))
        spec = env.scene_record.robot_spec
        dims = (env.scene_record.object_scale * env._object_scale_multiplier
                * float(env.cfg.reward.object_base_size))
        conn.send({"kind": "ready", "urdf": str(REPO / spec.urdf_path), "joints": list(spec.joint_names_canonical),
                   "base_pos": list(map(float, spec.base_pos)), "base_rot": list(map(float, spec.base_rot)),
                   "cube": dims[0].tolist(), "hand": NAMES.get(args.hand, args.hand),
                   "labels": [f"{l}  (ep {pathlib.Path(c).name.split('_ep_')[-1].split('_')[0]})" for l, c, _ in pols]})

        origins = env.scene.env_origins
        goals = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        running, speed, t_next = True, 1.0, time.time()
        while True:
            while conn.poll():
                m = conn.recv()
                if m["cmd"] == "quit":
                    return
                if m["cmd"] == "toggle":
                    running = not running
                elif m["cmd"] == "speed":
                    speed = float(m["value"])
                elif m["cmd"] == "reset":
                    obs, _ = env.reset()
                    goals.zero_()
            if not running:
                time.sleep(0.05)
                continue
            with torch.no_grad():
                acts = [pl.get_normalized_action(obs["policy"], deterministic_actions=True) for pl in players]
            act = torch.stack([acts[i][i] for i in range(len(players))])     # env i <- policy i
            obs, *_ = env.step(act)
            goals += env._is_success.long()
            q = env.robot.data.joint_pos[:, env._perm_lab_to_canon]
            conn.send({"kind": "frame", "q": q.tolist(),
                       "obj_pos": (env.object.data.root_pos_w - origins).tolist(),
                       "obj_quat": env.object.data.root_quat_w.tolist(),
                       "goal_pos": (env.goal_viz.data.root_pos_w - origins).tolist(),
                       "goal_quat": env.goal_viz.data.root_quat_w.tolist(),
                       "goals": goals.tolist()})
            t_next += float(env.step_dt) / max(speed, 1e-3)
            time.sleep(max(0.0, t_next - time.time()))
            t_next = max(t_next, time.time() - 0.1)

    _run()


def main(args):
    import viser
    from viser.extras import ViserUrdf

    for p in args.policy:
        print(f"[compare] {p}", flush=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    cmd = [sys.executable, __file__, "--worker", "--hand", args.hand, "--host", host, "--wport", str(port),
           "--authkey", authkey.hex()]
    for p in args.policy:
        cmd += ["--policy", p]
    child = subprocess.Popen(cmd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    print("[compare] Kit is booting in the child (about a minute)", flush=True)
    conn = listener.accept()
    server = viser.ViserServer(port=args.port)
    print(f"\n[compare] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)
    outbox: "queue.Queue[dict]" = queue.Queue()
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_reset = server.gui.add_button("reset all")
        s_speed = server.gui.add_slider("speed (x real time)", 0.1, 2.0, 0.05, 1.0)
        g_labels = server.gui.add_checkbox("show labels", True)
    md = server.gui.add_markdown("waiting for Kit...")
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    s_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": float(s_speed.value)}))

    info, urdfs, objs, goals_, labels = None, [], [], [], []
    try:
        while child.poll() is None:
            while True:
                try:
                    conn.send(outbox.get_nowait())
                except queue.Empty:
                    break
            if not conn.poll(timeout=0.05):
                continue
            batch = [conn.recv()]
            while conn.poll():
                batch.append(conn.recv())
            frame = None
            for m in batch:
                if m["kind"] == "frame":
                    frame = m
                elif m["kind"] == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                    raise SystemExit(1)
                elif m["kind"] == "ready":
                    info = m
                    n = len(m["labels"])
                    for i in range(n):
                        off = (args.spacing * (i - (n - 1) / 2), 0.0, 0.0)
                        server.scene.add_frame(f"/p{i}", position=off, show_axes=False)
                        server.scene.add_frame(f"/p{i}/base", position=tuple(m["base_pos"]),
                                               wxyz=tuple(m["base_rot"]), show_axes=False)
                        urdfs.append(ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name=f"/p{i}/base/robot"))
                        objs.append(_face_cube(server, f"/p{i}/obj", m["cube"], opacity=1.0))
                        goals_.append(_face_cube(server, f"/p{i}/goal", m["cube"], opacity=0.35))
                        labels.append(server.scene.add_label(f"/p{i}/label", m["labels"][i]))
                    md.content = f"{m['hand']}: " + " vs ".join(m["labels"])
            if frame is None or info is None:
                continue
            for i in range(len(urdfs)):
                urdfs[i].update_cfg(dict(zip(info["joints"], frame["q"][i])))
                objs[i].position, objs[i].wxyz = tuple(frame["obj_pos"][i]), tuple(frame["obj_quat"][i])
                goals_[i].position, goals_[i].wxyz = tuple(frame["goal_pos"][i]), tuple(frame["goal_quat"][i])
                gp = frame["goal_pos"][i]
                labels[i].position = (gp[0], gp[1], gp[2] + 0.09)
                labels[i].text = f"{info['labels'][i]}  ·  {frame['goals'][i]} goals"
                labels[i].visible = bool(g_labels.value)
    except (EOFError, ConnectionResetError):
        print("[compare] worker closed the connection", flush=True)
    finally:
        try:
            conn.send({"cmd": "quit"})
        except Exception:
            pass
        time.sleep(1.0)
        if child.poll() is None:
            child.kill()


if __name__ == "__main__":
    a = parse()
    worker(a) if a.worker else main(a)
