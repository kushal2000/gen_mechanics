"""Live viewer: the unified policy driving all 8 hands side by side, in viser (for screen recording).

One env per hand (multi:uniform deals env i to hand i mod 8), the run's own env config, the policy acting
greedily (exploration block 0). Each hand stands in its own slot of a grid with its cube, a translucent
goal cube and a goal counter. Physics runs in real time by default; the GUI can pause, reset, slow down.

    .venv_isaacsim/bin/python experiments/01oct_unified_rl/viser_unified.py --port 8080 \
        [--checkpoint <.pth>]      # default: the newest checkpoint of the canonical-frame unified run

Kit and viser cannot share an interpreter (coevolution/eval/viser_play.py), so the simulation runs in a
child process (this file with --worker) and streams poses over a local connection.
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
TASK = "GenMech-InHandReorient-Direct-v0"
RUN_GLOB = str(REPO / "debug_outputs/train_logs/01oct_unified_rl/0_scale_train_left_multi_uniform_canon_c01_*")
NAMES = {"sharpa": "SHARPA", "allegro": "Allegro", "leap": "LEAP", "shadow": "Shadow", "dex3": "Dex3",
         "tesollo": "Tesollo", "wuji2": "Wuji v2", "xhand": "XHAND"}


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=0.32, help="metres between hands in the grid")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--host", default="")
    ap.add_argument("--wport", type=int, default=0)
    ap.add_argument("--authkey", default="")
    return ap.parse_known_args()[0]


def newest_checkpoint() -> str:
    pths = glob.glob(f"{RUN_GLOB}/rank_0/*/nn/*.pth")
    if not pths:
        raise SystemExit(f"no checkpoints under {RUN_GLOB}")
    return max(pths, key=os.path.getmtime)


# ----------------------------------------------------------------------------------------- worker (Kit)
def worker(args):
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _worker_loop(args, conn)
    except Exception:
        import traceback
        conn.send({"kind": "error", "text": traceback.format_exc()})
    conn.close()
    app.close()
    os._exit(0)


def _worker_loop(args, conn):
    import gymnasium as gym
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    ckpt = pathlib.Path(args.checkpoint)
    run_dir = ckpt.parents[3]
    saved = OmegaConf.load(run_dir / "rank_0/.hydra/config.yaml")
    sys.argv = [sys.argv[0]]

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = 8
        env_cfg.seed = 0
        env_cfg.termination.max_consecutive_successes = 0
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        obs, _ = env.reset()
        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, str(ckpt),
                          device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)

        hs = env.scene_record.hand_set
        hand_of_env = env.scene_record.robot_design_index.tolist()
        envs = [hand_of_env.index(h) for h in range(hs.n_hands)]
        dims = (env.scene_record.object_scale * env._object_scale_multiplier
                * float(env.cfg.reward.object_base_size))
        hands = []
        for h, (spec, e) in enumerate(zip(hs.specs, envs)):
            hands.append({"name": NAMES.get(spec.hand_name, spec.hand_name),
                          "urdf": str(REPO / spec.urdf_path), "joints": list(spec.hand_joint_names),
                          "base_pos": list(map(float, spec.base_pos)), "base_rot": list(map(float, spec.base_rot)),
                          "cube": dims[e].tolist()})
        epoch = ckpt.name.split("_ep_")[-1].split("_")[0] if "_ep_" in ckpt.name else "best"
        conn.send({"kind": "ready", "hands": hands, "epoch": epoch, "dt": float(env.step_dt)})

        origins = env.scene.env_origins
        goals = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        running, speed = True, 1.0
        t_next = time.time()
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
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, *_ = env.step(act)
            goals += env._is_success.long()
            q = env.robot.data.joint_pos[:, env._perm_lab_to_canon]
            conn.send({"kind": "frame",
                       "q": [q[e, : hs.specs[h].num_hand_joints].tolist() for h, e in enumerate(envs)],
                       "obj_pos": (env.object.data.root_pos_w - origins)[envs].tolist(),
                       "obj_quat": env.object.data.root_quat_w[envs].tolist(),
                       "goal_pos": (env.goal_viz.data.root_pos_w - origins)[envs].tolist(),
                       "goal_quat": env.goal_viz.data.root_quat_w[envs].tolist(),
                       "goals": goals[envs].tolist()})
            t_next += float(env.step_dt) / max(speed, 1e-3)       # real time x speed
            time.sleep(max(0.0, t_next - time.time()))
            t_next = max(t_next, time.time() - 0.1)

    _run()


# ----------------------------------------------------------------------------------------- parent (viser)
# One colour per face (+x, -x, +y, -y, +z, -z), as on a Rubik's cube, so the cube's orientation -- the
# whole task -- reads at a glance, and the goal in the same colours shows which way it must turn.
FACE_RGB = ((200, 30, 30), (255, 140, 0), (0, 90, 200), (0, 160, 70), (245, 245, 245), (250, 210, 0))


def _face_cube(server, name: str, dims, opacity: float):
    """A frame holding a dark core and six thin coloured plates on its faces; returns the frame."""
    frame = server.scene.add_frame(name, show_axes=False)
    d = [float(v) for v in dims]
    t = 0.0015                                          # plate thickness
    server.scene.add_box(f"{name}/core", dimensions=tuple(v - 2 * t for v in d), color=(40, 40, 40),
                         opacity=opacity)
    k = 0
    for ax in range(3):
        for sign in (1.0, -1.0):
            size = list(d)
            size[ax] = t
            pos = [0.0, 0.0, 0.0]
            pos[ax] = sign * (d[ax] / 2 - t / 2)
            # Inset each plate a hair on its other axes so neighbouring faces do not z-fight at the edges.
            size = [v if i == ax else v - 0.002 for i, v in enumerate(size)]
            server.scene.add_box(f"{name}/f{k}", dimensions=tuple(size), position=tuple(pos),
                                 color=FACE_RGB[k], opacity=opacity)
            k += 1
    return frame


def main(args):
    import viser
    from viser.extras import ViserUrdf

    ckpt = args.checkpoint or newest_checkpoint()
    print(f"[unified] policy {ckpt}", flush=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    child = subprocess.Popen([sys.executable, __file__, "--worker", "--checkpoint", ckpt, "--host", host,
                              "--wport", str(port), "--authkey", authkey.hex()],
                             env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    print("[unified] Kit is booting in the child (about a minute)", flush=True)
    conn = listener.accept()

    server = viser.ViserServer(port=args.port)
    print(f"\n[unified] viser on http://{socket.gethostname()}:{args.port}\n", flush=True)
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

    hands, urdfs, objs, goals_, labels = [], [], [], [], []
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
                    hands = m["hands"]
                    for h, hd in enumerate(hands):
                        r, c = divmod(h, args.cols)
                        # Hands point their fingers along world -y, palm up: lay the grid out in x
                        # (columns) and y (rows), centred on the origin.
                        off = (args.spacing * (c - (args.cols - 1) / 2), -1.25 * args.spacing * r, 0.0)
                        slot = server.scene.add_frame(f"/h{h}", position=off, show_axes=False)
                        base = server.scene.add_frame(f"/h{h}/base", position=tuple(hd["base_pos"]),
                                                      wxyz=tuple(hd["base_rot"]), show_axes=False)
                        urdfs.append(ViserUrdf(server, pathlib.Path(hd["urdf"]), root_node_name=f"/h{h}/base/robot"))
                        objs.append(_face_cube(server, f"/h{h}/obj", hd["cube"], opacity=1.0))
                        goals_.append(_face_cube(server, f"/h{h}/goal", hd["cube"], opacity=0.35))
                        labels.append(server.scene.add_label(f"/h{h}/label", hd["name"],
                                                             position=(0.0, 0.0, 0.0)))
                        del slot, base
                    md.content = f"policy epoch {m['epoch']} · {len(hands)} hands"
            if frame is None or not hands:
                continue
            for h, hd in enumerate(hands):
                urdfs[h].update_cfg(dict(zip(hd["joints"], frame["q"][h])))
                objs[h].position, objs[h].wxyz = tuple(frame["obj_pos"][h]), tuple(frame["obj_quat"][h])
                goals_[h].position, goals_[h].wxyz = tuple(frame["goal_pos"][h]), tuple(frame["goal_quat"][h])
                gp = frame["goal_pos"][h]
                labels[h].position = (gp[0], gp[1], gp[2] + 0.09)
                labels[h].text = f"{hd['name']}  ·  {frame['goals'][h]} goals"
                labels[h].visible = bool(g_labels.value)
    except (EOFError, ConnectionResetError):
        print("[unified] worker closed the connection", flush=True)
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
