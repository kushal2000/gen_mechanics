"""Play the pen_repose policies in viser -- on repose goals and, with no retraining, on other tasks' goals.

Pick a run (every pen_repose run under debug_outputs/train_logs/15oct_pen_repose with a checkpoint; v2 -- wrist
rotation locked, uniform goals -- first) and a checkpoint. Its policy plays in the run's own env (built from its saved
config, --envs envs, 64 by default); the page draws one env -- the hand, the pen (its tip red) and the goal
(translucent) -- and keeps statistics of every episode finished since the last change.

Goals: the env stays pen_repose (so the policy sees the task one-hot it trained with) and only its goal generator
changes, live:
  repose: uniform        every goal a fresh pose from the whole goal space (what v2 trained on)
  repose: random walk    each goal the last turned 15-30 deg and moved up to 5 cm (v1's)
  spin in the air        pen_spin_air's goals: picked up flat, then each goal step_deg further round the vertical
  flip in the air        pen_flip_air's goals: picked up flat, then each goal step_deg further end over end
The pen's start: flat on the table, its yaw random or fixed (the angle of its tip from the fingers' direction). The
repose modes' first goal: a random pose or the flat pickup. Pushes on the lifted pen: on (as trained) or off. Any
change resets the envs and the statistics.

    .venv_isaacsim/bin/python experiments/15oct_pen_repose/viser_repose_play.py [--port 8096] [--run 292508]

Actions are sampled (the SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. The sim runs in a child (this file with --worker); switching runs restarts it
(about a minute), switching checkpoints within a run does not.
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import queue
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing.connection import Client, Listener

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "debug_outputs/train_logs/15oct_pen_repose"
MODES = ("repose: uniform", "repose: random walk", "spin in the air", "flip in the air")
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
FRAME_PERIOD_S = 1.0 / 30.0


def runs() -> dict[str, pathlib.Path]:
    """label -> run directory: v2 (uni_fw) runs first, then the rest; newest job first within each."""
    out = []
    for d in RUNS.iterdir():
        if not (d.is_dir() and (d / ".hydra/config.yaml").exists() and checkpoints(d)):
            continue
        m = re.search(r"prim_g(\d+)_(\d+)_(\d+)_(\d+)_(\w+?)_(\d+)$", d.name)
        if m:
            gaps = "/".join(str(int(g)) for g in m.groups()[:4])
            tag, job = m.group(5), int(m.group(6))
            label = f"{'v2' if tag == 'uni_fw' else 'v1'} · gaps {gaps} · {job}"
        else:
            tag, job, label = "", 0, d.name
        out.append((tag != "uni_fw", -job, label, d))
    return {label: d for *_, label, d in sorted(out)}


def checkpoints(run: pathlib.Path) -> dict[str, str]:
    """label -> path, newest first: every last_*_ep_<N> by epoch, then the best-reward one."""
    nn = run / run.name / "nn"
    eps = []
    for p in nn.glob("last_*_ep_*.pth"):
        m = re.search(r"_ep_(\d+)_rew_(-?\d+(?:\.\d+)?)", p.name)
        if m:
            eps.append((int(m.group(1)), float(m.group(2)), p))
    rows = {f"epoch {e} (reward {r:,.0f})": str(p) for e, r, p in sorted(eps, key=lambda x: -x[0])}
    best = sorted(nn.glob("0_*.pth"))
    if best:
        rows["best reward"] = str(best[0])
    return rows


# --- the Kit child ------------------------------------------------------------------------------------------------
def worker(args) -> None:
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    try:
        _work(args, conn)
    except Exception:
        import traceback
        try:
            conn.send({"kind": "error", "text": traceback.format_exc()})
        except Exception:
            pass
    os._exit(0)


def _pose7(pos, quat) -> list[float]:
    return [*pos.tolist(), *quat.tolist()]


def _work(args, conn) -> None:
    import tempfile

    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.tasks.pen_flip import PenFlipAir
    from isaacsimenvs.inhand_olympics.tasks.pen_spin import PenSpinAir

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
    repose = env.tasks[0]                            # pen_repose: its one-hot stays; only its goals are swapped
    delegates = {"spin in the air": PenSpinAir(env), "flip in the air": PenFlipAir(env)}
    trained_perturbed = env.task_perturbed.clone()

    def set_mode(m: dict) -> None:
        mode = m["mode"]
        for k in ("reset", "next_goal"):              # back to pen_repose's own methods
            repose.__dict__.pop(k, None)
        if mode in delegates:
            d = delegates[mode]
            d.step = math.radians(float(m["step_deg"]))
            repose.reset, repose.next_goal = d.reset, d.next_goal
        else:
            env.cfg.pen_repose.goal_sampling = "uniform" if mode == "repose: uniform" else "walk"
            env.cfg.pen_repose.random_first_goal = bool(m["random_first_goal"])
        env.cfg.reset.randomize_pen_yaw = bool(m["random_yaw"])
        env.cfg.reset.pen_yaw_deg = float(m["yaw_deg"])
        env.task_perturbed[:] = trained_perturbed & bool(m["pushes"])

    sx, sy, sz = cfg.table.size
    conn.send({"kind": "ready", "urdf": str(env.wrist_urdf_path), "joints": list(env.robot.joint_names),
               "table": [sx, sy, sz, cfg.table.top_height], "pen": [cfg.pen.length, cfg.pen.diameter],
               "num_envs": env.num_envs, "trained": {
                   "goal_sampling": cfg.pen_repose.goal_sampling, "random_first_goal": cfg.pen_repose.random_first_goal,
                   "wrist_ang": cfg.action.wrist_max_speed_ang, "first_tol": cfg.goal.first_goal_tolerance}})
    running, det, watch, speed = True, False, 0, 1.0
    finished: deque = deque(maxlen=5000)
    ends: dict[str, int] = {}
    last_end, steps, t_next, t_frame = "", 0, time.time(), 0.0

    def restart():
        nonlocal obs, last_end, steps
        obs, _ = env.reset()
        finished.clear()
        ends.clear()
        last_end, steps = "", 0

    while True:
        while conn.poll():
            m = conn.recv()
            c = m["cmd"]
            if c == "quit":
                return
            if c == "toggle":
                running = not running
            elif c == "det":
                det = bool(m["value"])
                restart()
            elif c == "watch":
                watch = max(0, min(env.num_envs - 1, int(m["value"])))
            elif c == "speed":
                speed = float(m["value"])
            elif c == "mode":
                set_mode(m)
                restart()
            elif c == "reset":
                restart()
            elif c == "ckpt":
                conn.send({"kind": "status", "text": "loading the checkpoint"})
                player = make_player(m["path"])
                restart()
        if running:
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=det)
            before = env.successes.clone()
            obs, _, term, trunc, _ = env.step(act)
            steps += 1
            done = term | trunc
            if bool(done.any()):
                goals = (before + env.is_success.float())[done].tolist()
                for i, g in zip(done.nonzero().flatten().tolist(), goals):
                    finished.append(g)
                    why = next((r for r, v in env.termination_reasons.items() if bool(v[i])), "?")
                    ends[why] = ends.get(why, 0) + 1
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
        pos, quat = env.pen_pose()
        g = list(finished)
        stats = ({"episodes": len(g), "goals": sum(g) / len(g), "one": sum(x >= 1 for x in g) / len(g),
                  "ends": dict(ends)} if g else {"episodes": 0})
        conn.send({"kind": "frame", "q": env.robot.data.joint_pos[watch].tolist(),
                   "pen": _pose7(pos[watch], quat[watch]),
                   "goal": _pose7(env.goal_pos[watch], env.goal_quat[watch]),
                   "status": (f"env {watch} · goals **{int(env.successes[watch])}** · goal distance "
                              f"{float(env.goal_dist[watch]) * 1000:.0f} mm · "
                              f"{float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                              + (f"  \nlast episode: {last_end}" if last_end else "")
                              + f"  \n{'running' if running else 'paused'} · {steps} steps since the last change"),
                   "stats": stats})


# --- the page -----------------------------------------------------------------------------------------------------
class Worker:
    def __init__(self, listener, authkey: bytes, run_dir: pathlib.Path, ckpt: str, envs: int):
        host, port = listener.address
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--run-dir", str(run_dir), "--checkpoint", ckpt, "--envs",
             str(envs), "--host", host, "--wport", str(port), "--authkey", authkey.hex()],
            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        self.conn = None
        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=lambda: self._q.put(listener.accept()), daemon=True).start()

    def connected(self) -> bool:
        if self.conn is None:
            try:
                self.conn = self._q.get_nowait()
            except queue.Empty:
                return False
        return True

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


def _pen(server, name: str, length: float, diameter: float, opacity: float = 1.0):
    """The box pen along +x, its tip (+x end) red."""
    import trimesh
    tip = 0.15 * diameter
    body = trimesh.creation.box(extents=(length - tip, diameter, diameter))
    body.apply_translation((-tip / 2, 0.0, 0.0))
    node = server.scene.add_mesh_simple(name, vertices=body.vertices, faces=body.faces, color=(38, 89, 204),
                                        opacity=opacity)
    t = trimesh.creation.box(extents=(tip, diameter, diameter))
    t.apply_translation((length / 2 - tip / 2, 0.0, 0.0))
    server.scene.add_mesh_simple(f"{name}/tip", vertices=t.vertices, faces=t.faces, color=(220, 40, 40),
                                 opacity=opacity)
    return node


def main(args) -> None:
    import signal

    import viser
    from viser.extras import ViserUrdf

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[repose-play] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    rs = runs()
    if not rs:
        raise SystemExit(f"no runs with checkpoints under {RUNS}")
    first = next((k for k, d in rs.items() if args.run and args.run in d.name), next(iter(rs)))
    outbox: "queue.Queue[dict]" = queue.Queue()

    with server.gui.add_folder("policy"):
        g_run = server.gui.add_dropdown("run", tuple(rs), initial_value=first)
        g_ckpt = server.gui.add_dropdown("checkpoint", tuple(checkpoints(rs[first])))
        g_envs = server.gui.add_number("envs (for the statistics)", args.envs, min=1, max=4096, step=1)
        b_load = server.gui.add_button("load (a new run restarts the sim, about a minute)")
        md_run = server.gui.add_markdown("")
    with server.gui.add_folder("goals and start"):
        g_mode = server.gui.add_dropdown("goals", MODES, initial_value=MODES[0])
        g_step = server.gui.add_slider("spin / flip step (deg)", 5.0, 180.0, 5.0, 45.0)
        g_first = server.gui.add_checkbox("repose: first goal a random pose (off: the flat pickup)", True)
        g_rand = server.gui.add_checkbox("pen's start yaw random", True)
        g_yaw = server.gui.add_slider("pen's start yaw (deg, tip from the fingers' direction)", -180.0, 180.0, 5.0,
                                      -90.0)
        g_push = server.gui.add_checkbox("pushes on the lifted pen (as trained)", True)
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_watch = server.gui.add_slider("watch env", 0, max(args.envs - 1, 1), 1, 0)
        b_reset = server.gui.add_button("reset the envs and the statistics")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")

    def mode_msg() -> dict:
        return {"cmd": "mode", "mode": str(g_mode.value), "step_deg": float(g_step.value),
                "random_first_goal": bool(g_first.value), "random_yaw": bool(g_rand.value),
                "yaw_deg": float(g_yaw.value), "pushes": bool(g_push.value)}

    for h in (g_mode, g_step, g_first, g_rand, g_yaw, g_push):
        h.on_update(lambda _: outbox.put(mode_msg()))
    state = {"worker": None, "run": None, "view": {}, "load": True, "t_boot": 0.0}
    g_run.on_update(lambda _: setattr(g_ckpt, "options", tuple(checkpoints(rs[str(g_run.value)]))))
    b_load.on_click(lambda _: state.update(load=True))
    b_run.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    g_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: outbox.put({"cmd": "det", "value": bool(g_det.value)}))
    g_watch.on_update(lambda _: outbox.put({"cmd": "watch", "value": int(g_watch.value)}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))

    def clear_view() -> None:
        v = state["view"]
        if "urdf" in v:
            v["urdf"].remove()
        for h in v.get("nodes", []):
            try:
                h.remove()
            except Exception:
                pass
        state["view"] = {}

    def build_view(m: dict) -> None:
        sx, sy, sz, top = m["table"]
        nodes = [server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                      position=(0.0, 0.0, top - sz / 2)),
                 server.scene.add_frame("/base", position=(0.0, 0.0, top), show_axes=False)]
        urdf = ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name="/base/robot")
        pen, goal = _pen(server, "/pen", *m["pen"]), _pen(server, "/goal", *m["pen"], opacity=0.35)
        state["view"] = {"urdf": urdf, "pen": pen, "goal": goal, "nodes": nodes + [pen, goal], "joints": m["joints"]}
        g_watch.max = max(m["num_envs"] - 1, 1)
        t = m["trained"]
        md_run.content = (f"Trained on repose goals **{t['goal_sampling']}**, first goal "
                          f"{'random' if t['random_first_goal'] else 'the flat pickup'} (tolerance "
                          f"{(t['first_tol'] or 0.01) * 100:.0f} cm), wrist rotation "
                          f"{'locked' if t['wrist_ang'] == 0 else 'free'}.")
        outbox.put(mode_msg())

    def stats_md(st: dict) -> str:
        if not st.get("episodes"):
            return f"**{g_mode.value}:** no finished episodes yet."
        ends = ", ".join(f"{k} {v}" for k, v in sorted(st["ends"].items(), key=lambda kv: -kv[1]))
        return (f"**{g_mode.value}**, over {st['episodes']} finished episodes:  \n**{st['goals']:.1f} goals / "
                f"episode** · {100 * st['one']:.0f}% reach ≥ 1 goal  \nended: {ends}")

    t_scan = time.time()
    try:
        while True:
            if time.time() - t_scan > 10.0:             # new checkpoints join the dropdown as they are saved
                t_scan = time.time()
                if str(g_run.value) in rs:
                    cks = tuple(checkpoints(rs[str(g_run.value)]))
                    if cks != tuple(g_ckpt.options):
                        cur = str(g_ckpt.value)
                        g_ckpt.options = cks
                        if cur in cks:
                            g_ckpt.value = cur
            if state.pop("load", False):
                label, ckpt = str(g_run.value), checkpoints(rs[str(g_run.value)])[str(g_ckpt.value)]
                if state["worker"] is not None and state["run"] == label and state["worker"].proc.poll() is None:
                    outbox.put({"cmd": "ckpt", "path": ckpt})
                else:
                    if state["worker"] is not None:
                        state["worker"].stop()
                    clear_view()
                    while not outbox.empty():
                        outbox.get_nowait()
                    state["worker"] = Worker(listener, authkey, rs[label], ckpt, int(g_envs.value))
                    state["t_boot"] = time.time()
                    md.content = f"loading **{label}** -- Kit boots in a child (about a minute)"
                state["run"] = label
            w = state["worker"]
            if w is None or w.proc.poll() is not None:
                time.sleep(0.1)
                continue
            if not w.connected():
                md.content = f"Kit booting... {time.time() - state['t_boot']:.0f} s"
                time.sleep(0.1)
                continue
            try:
                while True:
                    try:
                        w.conn.send(outbox.get_nowait())
                    except queue.Empty:
                        break
                if not w.conn.poll(timeout=0.05):
                    continue
                batch = [w.conn.recv()]
                while w.conn.poll():
                    batch.append(w.conn.recv())
            except (EOFError, ConnectionResetError, BrokenPipeError):
                md.content = "**the sim closed the connection**"
                state["worker"] = None
                continue
            frame = None
            for m in batch:
                if m["kind"] == "error":
                    md.content = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
                elif m["kind"] == "status":
                    md.content = m["text"]
                elif m["kind"] == "ready":
                    build_view(m)
                elif m["kind"] == "frame":
                    frame = m
            v = state["view"]
            if frame is None or "urdf" not in v:
                continue
            v["urdf"].update_cfg(dict(zip(v["joints"], frame["q"])))
            for k in ("pen", "goal"):
                v[k].position, v[k].wxyz = tuple(frame[k][:3]), tuple(frame[k][3:])
            md.content = frame["status"]
            md_stats.content = stats_md(frame["stats"])
    except KeyboardInterrupt:
        pass
    finally:
        if state["worker"] is not None:
            state["worker"].stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--run", default="", help="the run to select first (a job id or part of its name)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--checkpoint", default="", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a)
    else:
        main(a)
