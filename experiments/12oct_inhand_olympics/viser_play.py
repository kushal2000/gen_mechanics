"""Watch any in-hand olympics policy in viser: every run under debug_outputs/train_logs/12oct_inhand_olympics that
trained on --task (default pen_lift) and has a checkpoint is in a dropdown; pick a run and a checkpoint and its
policy plays in the run's own env (its hand(s), pen, pushes, rewards) over --envs envs. The page draws one env --
pick which -- and keeps per-hand statistics of every finished episode since the policy was loaded.

    .venv_isaacsim/bin/python experiments/12oct_inhand_olympics/viser_play.py [--port 8091] [--run 257159]
    # then open http://<node>:8091
    ... viser_play.py --check 257159          # no viewer: load the run, play 600 steps, print the statistics

Actions are sampled (the SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. Envs start with whole episodes (no staggered clocks), so the first episodes
are full ones. The push and twist scales can be changed live (pen_lift's pushes start after the episode's second
goal, the pickup and one hold); a red line from the pen shows the push while one acts.

Kit and viser cannot share an interpreter (coevolution/eval/viser_play.py), so the sim runs in a child worker that
streams joint positions and poses back; switching to another run restarts the worker (about a minute), switching
checkpoints within a run does not.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import Client, Listener

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUNS_DIR = REPO / "debug_outputs/train_logs/12oct_inhand_olympics"
FULL = "wuji2_left_uniform_handonly"
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "0.5x": 0.5, "2x": 2.0, "as fast as possible": 0.0}
FRAME_PERIOD_S = 1.0 / 30.0


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--task", default="pen_lift", help="list the runs that trained on this task")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--envs", type=int, default=24, help="envs the policy plays in (statistics over all of them)")
    ap.add_argument("--run", default="", help="job id of the run to load first (default: the newest)")
    ap.add_argument("--check", default="", help="no viewer: load this job id's run, play, print statistics")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--checkpoint", default="", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


# --- the runs (parent side: no Kit) -------------------------------------------------------------------------

def checkpoints(run_dir: pathlib.Path) -> list[tuple[str, str]]:
    """(label, path), newest first: every last_*_ep_<N> by epoch, then the best-reward one."""
    import re
    out = []
    for p in run_dir.glob("0_*/nn/last_*_ep_*.pth"):
        m = re.search(r"_ep_(\d+)_", p.name)
        if m:
            out.append((int(m.group(1)), p))
    rows = [(f"epoch {e}", str(p)) for e, p in sorted(out, key=lambda x: -x[0])]
    best = [p for p in run_dir.glob("0_*/nn/0_*.pth")]
    rows += [("best reward", str(best[0]))] if best else []
    return rows


def _hand_short(spec: str) -> str:
    if spec.startswith("multi:"):
        names = spec[len("multi:"):].split("+")
        return "mix " + "+".join(_hand_short(n) for n in names)
    if spec == FULL:
        return "full"
    return spec.replace(FULL + "_only_", "").replace(FULL + "_", "").replace("thumb_index", "ti")


def discover(runs_dir: pathlib.Path, task: str) -> list[dict]:
    """Every run that trained on ``task`` and has a checkpoint, newest job first, with a one-line label."""
    import yaml
    dirs = [d for d in runs_dir.glob("0_*") if (d / ".hydra/config.yaml").is_file()]
    state = {}                                      # job id -> Slurm state (RUNNING, COMPLETED, CANCELLED, ...)
    try:
        jobs = ",".join(d.name.rsplit("_", 1)[-1] for d in dirs if d.name.rsplit("_", 1)[-1].isdigit())
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", jobs],
                             capture_output=True, text=True).stdout
        state = {j: s.split()[0] for j, s in (line.split("|") for line in out.splitlines() if "|" in line)}
    except (FileNotFoundError, ValueError):
        pass
    runs = []
    for d in dirs:
        cfg_path = d / ".hydra/config.yaml"
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            env, agent = cfg["env"], cfg["agent"]["params"]
            if task not in env["task"]["names"]:
                continue
        except Exception:
            continue
        ckpts = checkpoints(d)
        if not ckpts:
            continue
        job = d.name.rsplit("_", 1)[-1]
        r, rw, pert = env.get("reset", {}), env.get("reward", {}), env.get("perturb", {})
        bits = [job, _hand_short(env["hand"]["robot_spec"]),
                "JT" if agent["network"].get("name") == "joint_transformer" else
                ("LSTM" if agent["network"].get("rnn") else "MLP"),
                f"s{agent.get('seed')}", f"f{float(pert.get('force_scale', 20.0)):g}"]
        if r.get("stagger_initial_episodes"):
            bits.append("stagger")
        if len(env["task"]["names"]) > 1:
            bits.append("+".join(t.replace("pen_", "") for t in env["task"]["names"]))
        if env.get("pen", {}).get("shape", "capsule") != "box":
            bits.append("capsule")
        if not rw.get("approach_rew_scale", 1.0) and not rw.get("lost_pen_penalty", 1.0):
            bits.append("no approach/lost")
        if r.get("randomize_pen_yaw") is False:
            bits.append(f"yaw {r.get('pen_yaw_deg', 0):g}")
        if "_ft" in d.name or "_from_" in d.name:
            bits.append("fine-tune")
        st = state.get(job, "")
        if st in ("RUNNING", "PENDING"):
            bits.append("(training)")
        elif st and st != "COMPLETED":
            bits.append(f"({st.lower()})")
        runs.append({"job": job, "dir": str(d), "label": " · ".join(bits), "ckpts": ckpts})
    return sorted(runs, key=lambda r: -int(r["job"]) if r["job"].isdigit() else 0)


# --- the worker (Kit) -----------------------------------------------------------------------------------

def worker(args):
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _work(args, conn)
    except Exception:
        import traceback
        try:
            conn.send({"kind": "error", "text": traceback.format_exc()})
        except Exception:
            pass
    del app
    os._exit(0)


def _work(args, conn):
    import tempfile

    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    run = pathlib.Path(args.run_dir)
    saved = OmegaConf.load(run / ".hydra/config.yaml")
    env_d = OmegaConf.to_container(saved.env, resolve=True)
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.scene.num_envs = args.envs
    cfg.reset.stagger_initial_episodes = False      # whole first episodes
    cfg.seed = 0
    conn.send({"kind": "status", "text": "building the scene"})
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()

    def make_player(ckpt):
        return RlPlayer(obs["policy"].shape[-1], env.action_space.shape[-1], f.name, ckpt, device=env.device,
                        sapg_expl_coef=0.0, num_envs=env.num_envs)

    player = make_player(args.checkpoint)
    n, dev = env.num_envs, env.device
    multi = env.hand_set is not None
    labels = list(env.hand_set.labels) if multi else [_hand_short(cfg.hand.robot_spec)]
    hand_of_env = env.hand_of_env.tolist()
    # Each hand's joint names, in the order its slice of joint_pos is sent (the env's joint order).
    if multi:
        first = [hand_of_env.index(h) for h in range(len(labels))]
        hand_joints = [env.robot.real_joint_names(i) for i in first]
    else:
        hand_joints = [list(env.robot.joint_names)]
    conn.send({"kind": "ready", "urdfs": [str(p) for p in env.hand_urdf_paths], "hand_joints": hand_joints,
               "hand_labels": labels, "hand_of_env": hand_of_env, "num_envs": n,
               "table_size": list(cfg.table.size), "table_top": cfg.table.top_height,
               "pen": [cfg.pen.length, cfg.pen.diameter], "pen_shape": cfg.pen.shape,
               "force_scale": cfg.perturb.force_scale, "torque_scale": cfg.perturb.torque_scale,
               "tasks": [t.name for t in env.tasks]})

    running, pending, det, watch, speed = True, 0, False, 0, 1.0
    finished = [deque(maxlen=1000) for _ in labels]          # goals of each finished episode, per hand
    ends = [dict() for _ in labels]                          # how they ended, per hand
    last_end = ""
    t_next = time.time()
    t_frame = 0.0
    steps = 0
    while True:
        while conn.poll():
            m = conn.recv()
            c = m["cmd"]
            if c == "quit":
                return
            if c == "toggle":
                running = not running
            elif c == "step":
                running, pending = False, pending + 1
            elif c == "det":
                det = bool(m["value"])
            elif c == "watch":
                watch = max(0, min(n - 1, int(m["value"])))
            elif c == "speed":
                speed = float(m["value"])
            elif c == "force":
                env.cfg.perturb.force_scale = float(m["value"])
            elif c == "torque":
                env.cfg.perturb.torque_scale = float(m["value"])
            elif c in ("reset", "ckpt"):
                if c == "ckpt":
                    conn.send({"kind": "status", "text": "loading the checkpoint"})
                    player = make_player(m["path"])
                obs, _ = env.reset()
                for q in finished:
                    q.clear()
                for e in ends:
                    e.clear()
                last_end, steps = "", 0
        if running or pending:
            pending = max(0, pending - 1)
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=det)
            before = env.successes.clone()
            obs, _, term, trunc, _ = env.step(act)
            steps += 1
            done = term | trunc
            if bool(done.any()):
                goals = (before + env.is_success.float())[done].tolist()
                for i, g in zip(done.nonzero().flatten().tolist(), goals):
                    h = hand_of_env[i]
                    finished[h].append(g)
                    why = next((r for r, v in env.termination_reasons.items() if bool(v[i])), "?")
                    ends[h][why] = ends[h].get(why, 0) + 1
                    if i == watch:
                        last_end = f"{why} after {g:.0f} goals"
            if speed > 0:
                t_next += float(env.step_dt) / speed
                time.sleep(max(0.0, t_next - time.time()))
                t_next = max(t_next, time.time() - 0.1)
            else:
                t_next = time.time()
        else:
            time.sleep(0.02)
            t_next = time.time()
        now = time.time()
        if now - t_frame < FRAME_PERIOD_S:
            continue
        t_frame = now
        h = hand_of_env[watch]
        k = len(hand_joints[h])
        pen_pos, pen_quat = env.pen_pose()
        lift = float(env.pen_lowest_height()[watch])
        force = env.pen_forces[watch, 0]
        stats = []
        for j, lab in enumerate(labels):
            g = list(finished[j])
            stats.append({"hand": lab, "episodes": len(g), "mean": sum(g) / len(g) if g else float("nan"),
                          "at_least_1": sum(x >= 1 for x in g) / len(g) if g else float("nan"),
                          "ends": dict(ends[j])})
        conn.send({"kind": "frame", "hand": h, "q": env.robot.data.joint_pos[watch, :k].tolist(),
                   "pen_pos": pen_pos[watch].tolist(), "pen_quat": pen_quat[watch].tolist(),
                   "goal_pos": env.goal_pos[watch].tolist(), "goal_quat": env.goal_quat[watch].tolist(),
                   "goal_dist": float(env.goal_dist[watch]), "goals": int(env.successes[watch]),
                   "task": env.tasks[int(env.task_id[watch])].name, "lift": lift, "force": force.tolist(),
                   "time": float(env.episode_length_buf[watch]) * float(env.step_dt), "last_end": last_end,
                   "running": running, "steps": steps, "stats": stats})


# --- the viewer -------------------------------------------------------------------------------------------

class Worker:
    """One Kit child for one run; connects back on ``listener`` (accepted on a thread, so a child that dies while
    booting cannot hang the page)."""

    def __init__(self, listener, authkey: bytes, run: dict, ckpt: str, envs: int):
        host, port = listener.address
        self.run, self.ckpt = run, ckpt
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--run-dir", run["dir"], "--checkpoint", ckpt,
             "--envs", str(envs), "--host", host, "--wport", str(port), "--authkey", authkey.hex()],
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        self.conn = None
        self._accepted: "queue.Queue" = queue.Queue()
        threading.Thread(target=lambda: self._accepted.put(listener.accept()), daemon=True).start()

    def connected(self) -> bool:
        if self.conn is None:
            try:
                self.conn = self._accepted.get_nowait()
            except queue.Empty:
                return False
        return True

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        try:
            if self.conn is not None:
                self.conn.send({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _add_pen(server, name: str, length: float, diameter: float, shape: str, opacity: float = 1.0):
    """The pen as drawn in the olympics viewer: a box (or capsule) along its x, its tip (+x end) red."""
    import numpy as np
    import trimesh
    if shape == "box":
        mesh = trimesh.creation.box(extents=(length, diameter, diameter))
    else:
        mesh = trimesh.creation.capsule(height=length - diameter, radius=diameter / 2, count=[24, 24])
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    node = server.scene.add_mesh_simple(name, vertices=mesh.vertices, faces=mesh.faces, color=(38, 89, 204),
                                        opacity=opacity)
    server.scene.add_icosphere(f"{name}/tip", radius=0.4 * diameter, color=(220, 40, 40),
                               position=(length / 2 - 0.3 * diameter, 0.0, 0.0), opacity=opacity)
    return node


def _stats_md(stats) -> str:
    rows = ["| hand | finished episodes | goals / episode | ≥ 1 goal | how they ended |", "|---|---|---|---|---|"]
    for s in stats:
        ends = ", ".join(f"{k} {v}" for k, v in sorted(s["ends"].items(), key=lambda kv: -kv[1]))
        mean = f"{s['mean']:.1f}" if s["episodes"] else "–"
        one = f"{100 * s['at_least_1']:.0f}%" if s["episodes"] else "–"
        rows.append(f"| {s['hand']} | {s['episodes']} | {mean} | {one} | {ends or '–'} |")
    return "\n".join(rows)


def main(args):
    import numpy as np
    import viser
    from viser.extras import ViserUrdf

    runs = discover(pathlib.Path(args.runs_dir), args.task)
    if not runs:
        raise SystemExit(f"no {args.task} runs with checkpoints under {args.runs_dir}")
    by_label = {r["label"]: r for r in runs}
    first = next((r for r in runs if r["job"] == args.run), runs[0])
    print(f"[play] {len(runs)} {args.task} runs with checkpoints", flush=True)

    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    server = viser.ViserServer(port=args.port)
    print(f"\n[play] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    server.scene.add_grid("/ground", width=2.0, height=2.0)

    outbox: "queue.Queue[dict]" = queue.Queue()
    requests: "queue.Queue[tuple[dict, str]]" = queue.Queue()
    with server.gui.add_folder("policy"):
        g_run = server.gui.add_dropdown("run", tuple(by_label), initial_value=first["label"])
        g_ckpt = server.gui.add_dropdown("checkpoint", tuple(c[0] for c in first["ckpts"]))
        b_load = server.gui.add_button("load")
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_step = server.gui.add_button("single step")
        b_reset = server.gui.add_button("reset every env (clears the statistics)")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_watch = server.gui.add_slider("watch env", 0, max(args.envs - 1, 1), 1, 0)
    with server.gui.add_folder("pushes (live)"):
        g_force = server.gui.add_number("force_scale", 20.0, step=1.0)
        g_torque = server.gui.add_number("torque_scale", 2.0, step=0.1)
    md_load = server.gui.add_markdown("")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")

    @g_run.on_update
    def _(_):
        g_ckpt.options = tuple(c[0] for c in by_label[g_run.value]["ckpts"])
        g_ckpt.value = g_ckpt.options[0]

    @b_load.on_click
    def _(_):
        r = by_label[g_run.value]
        requests.put((r, dict(r["ckpts"])[g_ckpt.value]))

    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_step.on_click(lambda _: outbox.put({"cmd": "step"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    g_det.on_update(lambda _: outbox.put({"cmd": "det", "value": bool(g_det.value)}))
    g_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": SPEEDS[g_speed.value]}))
    g_watch.on_update(lambda _: outbox.put({"cmd": "watch", "value": int(g_watch.value)}))
    g_force.on_update(lambda _: outbox.put({"cmd": "force", "value": float(g_force.value)}))
    g_torque.on_update(lambda _: outbox.put({"cmd": "torque", "value": float(g_torque.value)}))

    requests.put((first, first["ckpts"][0][1]))
    w: Worker | None = None
    info = None
    scene: dict = {}
    load_id = 0
    t_boot = 0.0

    def clear_scene():
        for k in ("urdfs",):
            for u in scene.get(k, []):
                u.remove()
        for k in ("frames", "pen", "goal", "table", "push"):
            v = scene.get(k)
            for node in (v if isinstance(v, list) else [v] if v is not None else []):
                node.remove()
        scene.clear()

    try:
        while True:
            try:
                run, ckpt = requests.get_nowait()
                if w is not None and w.alive() and w.connected() and run["dir"] == w.run["dir"]:
                    outbox.put({"cmd": "ckpt", "path": ckpt})        # same run: no restart
                    w.ckpt = ckpt
                    md_load.content = f"**{run['label']}** · {pathlib.Path(ckpt).name}"
                else:
                    if w is not None:
                        w.stop()
                    while not outbox.empty():
                        outbox.get_nowait()
                    clear_scene()
                    info = None
                    load_id += 1
                    w = Worker(listener, authkey, run, ckpt, args.envs)
                    t_boot = time.time()
                    md_load.content = f"**loading {run['label']}** -- Kit boots in a child (about a minute)"
                    md.content = md_stats.content = ""
            except queue.Empty:
                pass
            if w is None or not w.alive():
                if w is not None and info is not None:
                    md.content = "**the worker exited**"
                time.sleep(0.1)
                continue
            if not w.connected():
                md.content = f"Kit booting... {time.time() - t_boot:.0f} s"
                time.sleep(0.1)
                continue
            conn = w.conn
            try:
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
            except (EOFError, ConnectionResetError, BrokenPipeError):
                md.content = "**the worker closed the connection**"
                w.stop()
                w = None
                continue
            frame = None
            for m in batch:
                if m["kind"] == "error":
                    md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif m["kind"] == "status":
                    md.content = m["text"]
                elif m["kind"] == "frame":
                    frame = m
                elif m["kind"] == "ready":
                    info = m
                    sx, sy, sz = m["table_size"]
                    scene["table"] = server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                                          position=(0.0, 0.0, m["table_top"] - sz / 2))
                    scene["frames"], scene["urdfs"] = [], []
                    for h, urdf in enumerate(m["urdfs"]):
                        fr = server.scene.add_frame(f"/hand{load_id}_{h}", position=(0.0, 0.0, m["table_top"]),
                                                    show_axes=False)
                        scene["frames"].append(fr)
                        scene["urdfs"].append(ViserUrdf(server, pathlib.Path(urdf),
                                                        root_node_name=f"/hand{load_id}_{h}/robot"))
                    scene["pen"] = _add_pen(server, f"/pen{load_id}", *m["pen"], m["pen_shape"])
                    scene["goal"] = _add_pen(server, f"/goal{load_id}", *m["pen"], m["pen_shape"], opacity=0.3)
                    g_watch.max = max(m["num_envs"] - 1, 1)
                    g_force.value, g_torque.value = float(m["force_scale"]), float(m["torque_scale"])
                    for c in ({"cmd": "det", "value": bool(g_det.value)},
                              {"cmd": "speed", "value": SPEEDS[g_speed.value]},
                              {"cmd": "watch", "value": int(g_watch.value)}):
                        outbox.put(c)
                    md_load.content = (f"**{w.run['label']}** · {pathlib.Path(w.ckpt).name}  \n"
                                       f"{m['num_envs']} envs · tasks {', '.join(m['tasks'])} · hands "
                                       f"{', '.join(m['hand_labels'])}")
            if frame is None or info is None:
                continue
            h = frame["hand"]
            for k, fr in enumerate(scene["frames"]):
                fr.visible = k == h
            scene["urdfs"][h].update_cfg(dict(zip(info["hand_joints"][h], frame["q"])))
            scene["pen"].position, scene["pen"].wxyz = tuple(frame["pen_pos"]), tuple(frame["pen_quat"])
            scene["goal"].position, scene["goal"].wxyz = tuple(frame["goal_pos"]), tuple(frame["goal_quat"])
            f = np.asarray(frame["force"], dtype=np.float32)
            if "push" in scene:
                scene.pop("push").remove()
            fn = float(np.linalg.norm(f))
            if fn > 1e-6:
                p = np.asarray(frame["pen_pos"], dtype=np.float32)
                scene["push"] = server.scene.add_line_segments(
                    "/push", points=np.stack([p, p + 0.1 * f / fn])[None], colors=(230, 30, 30), line_width=4.0)
            ended = f"  \nlast episode: {frame['last_end']}" if frame["last_end"] else ""
            md.content = (f"env {int(g_watch.value)} ({info['hand_labels'][h]}) · **{frame['task']}** · "
                          f"goals {frame['goals']} · {frame['time']:.1f} s into this goal  \n"
                          f"goal distance {frame['goal_dist'] * 1000:.0f} mm · pen {frame['lift'] * 1000:.0f} mm "
                          f"above the table · push {fn:.2f} N{ended}  \n"
                          f"{'running' if frame['running'] else 'paused'} · {frame['steps']} steps")
            md_stats.content = _stats_md(frame["stats"])
    except KeyboardInterrupt:
        pass
    finally:
        if w is not None:
            w.stop()


def check(args):
    """No viewer: load one run's newest checkpoint, play 600 steps as fast as possible, print what the page would."""
    runs = {r["job"]: r for r in discover(pathlib.Path(args.runs_dir), args.task)}
    if args.check not in runs:
        raise SystemExit(f"job {args.check} is not a {args.task} run with checkpoints")
    run = runs[args.check]
    print(f"[check] {run['label']} · {run['ckpts'][0][0]}", flush=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    w = Worker(listener, authkey, run, run["ckpts"][0][1], args.envs)
    while not w.connected():
        if not w.alive():
            raise SystemExit("[check] the worker died before connecting")
        time.sleep(0.5)
    w.conn.send({"cmd": "speed", "value": 0.0})
    ready, last, switched = None, None, False
    while w.alive():
        if not w.conn.poll(timeout=1.0):
            continue
        m = w.conn.recv()
        if m["kind"] == "error":
            print(m["text"], flush=True)
            break
        if m["kind"] == "ready":
            ready = m
            print(f"[check] ready: hands {m['hand_labels']}, {len(m['hand_joints'][0])} joints drawn for hand 0, "
                  f"urdfs {m['urdfs']}", flush=True)
        if m["kind"] == "frame":
            last = m
            if m["steps"] >= 300 and not switched and len(run["ckpts"]) > 1:
                print(f"[check] 300 steps: {m['stats']}\n[check] switching to {run['ckpts'][1][0]} "
                      f"(the same worker)", flush=True)
                w.conn.send({"cmd": "ckpt", "path": run["ckpts"][1][1]})
                w.conn.send({"cmd": "watch", "value": 1})
                switched = True
            elif m["steps"] >= 600 and (switched or len(run["ckpts"]) == 1):
                break
    if last is not None:
        print(f"[check] watched env: hand {last['hand']}, {len(last['q'])} joint positions, goals {last['goals']}, "
              f"goal distance {last['goal_dist']:.3f} m, push {last['force']}", flush=True)
        print(f"[check] statistics: {last['stats']}", flush=True)
    w.stop()
    print("[check] ok" if ready is not None and last is not None else "[check] FAILED", flush=True)


if __name__ == "__main__":
    a = parse()
    if a.worker:
        worker(a)
    elif a.check:
        check(a)
    else:
        main(a)
