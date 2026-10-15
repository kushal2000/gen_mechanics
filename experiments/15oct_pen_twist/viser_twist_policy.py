"""Watch a trained pen_twist policy in viser and set how tight the cap is.

Pick a run (every run under debug_outputs/train_logs/15oct_pen_twist; the lists rescan every 10 s) and a checkpoint; its policy plays in the run's
own env, built from its saved config, in --envs envs (64 by default). The page draws one env -- the hand, the barrel,
the cap (red, a white stripe so its turning shows) and the cap's goal (translucent, a quarter turn ahead) -- and
keeps statistics of every episode finished since the last change: goals (quarter turns) and full turns per episode,
the share reaching at least one turn, and how episodes ended.

The cap's friction torque is a live slider: changing it resets the envs and the statistics, so they always describe
the torque shown. The run's own curriculum is off here (the slider decides); a curriculum run's checkpoint was
trained at whatever torque the curriculum had reached by then (twist/friction_torque on wandb).

    .venv_isaacsim/bin/python experiments/15oct_pen_twist/viser_twist_policy.py [--port 8095] [--run 293321]
    ... viser_twist_policy.py --check 293321 [--torque 0.02] [--seconds 60]    # no viewer: play and print

Actions are sampled (the SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. Kit and viser cannot share an interpreter: the sim runs in a child (this file
with --worker). Switching runs restarts it (about a minute); switching checkpoints within a run does not.
"""
from __future__ import annotations

import argparse
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
RUNS = REPO / "debug_outputs/train_logs/15oct_pen_twist"
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
FRAME_PERIOD_S = 1.0 / 30.0


def runs() -> dict[str, pathlib.Path]:
    """label -> run directory, newest job first: every run with a saved config and at least one checkpoint."""
    out = []
    for d in RUNS.iterdir():
        if d.is_dir() and (d / ".hydra/config.yaml").exists() and checkpoints(d):
            m = re.search(r"_(\d+)$", d.name)
            tag = re.sub(r"^0_pen_twist_sapg_mlp_e\d+_s\d+_g[\d.]+_fr[\d.]+_kp\w+?_box_", "", d.name)
            out.append((int(m.group(1)) if m else 0, f"{tag}", d))
    return {label: d for _, label, d in sorted(out, key=lambda x: -x[0])}


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


def find_run(key: str) -> tuple[str, pathlib.Path]:
    for label, d in runs().items():
        if key in d.name:
            return label, d
    raise SystemExit(f"no run matching {key!r} under {RUNS}")


# --- the Kit child ------------------------------------------------------------------------------------------------
def _boot():
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    return AppLauncher(a).app


class Session:
    """The run's env with its policy, the cap torque, and the finished episodes' statistics."""

    def __init__(self, run_dir: pathlib.Path, ckpt: str, envs: int, torque: float | None):
        import tempfile

        import yaml
        from omegaconf import OmegaConf

        import isaacsimenvs  # noqa: F401
        from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
        from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

        saved = OmegaConf.load(run_dir / ".hydra/config.yaml")
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        cfg = InHandOlympicsEnvCfg()
        cfg.from_dict(env_d)
        cfg.scene.num_envs = envs
        cfg.reset.stagger_initial_episodes = False        # whole first episodes
        cfg.pen_twist.curriculum = False                  # the slider sets the torque
        self.trained_torque = float(cfg.pen.twist_friction_torque)
        if torque is not None:
            cfg.pen.twist_friction_torque = torque
        cfg.seed = 0
        self.env = InHandOlympicsEnv(cfg)
        self.obs, _ = self.env.reset()
        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        self.agent_yaml = f.name
        self.load(ckpt)
        self.torque = float(cfg.pen.twist_friction_torque)
        self.step_deg = float(cfg.pen_twist.step_deg)
        self.clear()

    def load(self, ckpt: str) -> None:
        from coevolution.eval.rl_player import RlPlayer
        env = self.env
        self.player = RlPlayer(self.obs["policy"].shape[-1], env.action_space.shape[-1], self.agent_yaml, ckpt,
                               device=env.device, sapg_expl_coef=0.0, num_envs=env.num_envs)

    def clear(self) -> None:
        self.finished: deque = deque(maxlen=5000)       # goals of each finished episode
        self.ends: dict[str, int] = {}
        self.steps = 0
        self.last_end = ""

    def set_torque(self, torque: float) -> None:
        import torch
        env = self.env
        self.torque = float(torque)
        env.cfg.pen.twist_friction_torque = self.torque
        env.pen.write_joint_effort_limit_to_sim(torch.full((env.num_envs, 1), self.torque, device=env.device),
                                                joint_ids=[0])
        self.restart()

    def restart(self) -> None:
        self.obs, _ = self.env.reset()
        self.clear()

    def step(self, deterministic: bool, watch: int) -> None:
        import torch
        env = self.env
        with torch.no_grad():
            act = self.player.get_normalized_action(self.obs["policy"], deterministic_actions=deterministic)
        before = env.successes.clone()
        self.obs, _, term, trunc, _ = env.step(act)
        self.steps += 1
        done = term | trunc
        if bool(done.any()):
            goals = (before + env.is_success.float())[done].tolist()
            for i, g in zip(done.nonzero().flatten().tolist(), goals):
                self.finished.append(g)
                why = next((r for r, v in env.termination_reasons.items() if bool(v[i])), "?")
                self.ends[why] = self.ends.get(why, 0) + 1
                if i == watch:
                    self.last_end = f"{why} after {g:.0f} goals ({g * self.step_deg / 360:.2f} turns)"

    def stats(self) -> dict:
        g = list(self.finished)
        per_turn = 360.0 / self.step_deg
        if not g:
            return {"episodes": 0}
        return {"episodes": len(g), "goals": sum(g) / len(g), "turns": sum(g) / len(g) / per_turn,
                "one_turn": sum(x >= per_turn for x in g) / len(g), "ends": dict(self.ends)}


def worker(args) -> None:
    app = _boot()  # noqa: F841
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


def _curriculum_run(run_dir: pathlib.Path) -> bool:
    f = run_dir / ".hydra/overrides.yaml"
    return f.exists() and "pen_twist.curriculum=True" in f.read_text()


def _pose7(pos, quat) -> list[float]:
    return [*pos.tolist(), *quat.tolist()]


def _work(args, conn) -> None:
    from isaaclab.utils.math import quat_apply

    from isaacsimenvs.inhand_olympics.twist_pen import link_extents

    conn.send({"kind": "status", "text": "building the scene"})
    s = Session(pathlib.Path(args.run_dir), args.checkpoint, args.envs, args.torque if args.torque > 0 else None)
    env, cfg = s.env, s.env.cfg
    sx, sy, sz = cfg.table.size
    conn.send({"kind": "ready", "urdf": str(env.wrist_urdf_path), "joints": list(env.robot.joint_names),
               "table": [sx, sy, sz, cfg.table.top_height],
               "pen": {"diameter": cfg.pen.diameter, "links": link_extents(cfg.pen.length, cfg.pen.cap_length, 0.001)},
               "torque": s.torque, "trained_torque": s.trained_torque, "num_envs": env.num_envs,
               "curriculum_run": _curriculum_run(pathlib.Path(args.run_dir))})
    running, det, watch, speed = True, False, 0, 1.0
    t_next, t_frame = time.time(), 0.0
    b, c = env.barrel_body, env.cap_body
    while True:
        while conn.poll():
            m = conn.recv()
            k = m["cmd"]
            if k == "quit":
                return
            if k == "toggle":
                running = not running
            elif k == "det":
                det = bool(m["value"])
                s.restart()
            elif k == "watch":
                watch = max(0, min(env.num_envs - 1, int(m["value"])))
            elif k == "speed":
                speed = float(m["value"])
            elif k == "torque":
                s.set_torque(float(m["value"]))
            elif k == "reset":
                s.restart()
            elif k == "ckpt":
                conn.send({"kind": "status", "text": "loading the checkpoint"})
                s.load(m["path"])
                s.restart()
        if running:
            s.step(det, watch)
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
        o = env.scene.env_origins[watch]
        d = env.pen.data
        gq = env.cap_goal_quat[watch]
        goal_link = env.cap_goal_pos[watch] - quat_apply(gq.unsqueeze(0), env.cap_centre.unsqueeze(0))[0]
        turned = float(env.tasks[0].turned[watch]) / (2 * 3.141592653589793)
        conn.send({"kind": "frame", "q": env.robot.data.joint_pos[watch].tolist(),
                   "barrel": _pose7(d.body_pos_w[watch, b] - o, d.body_quat_w[watch, b]),
                   "cap": _pose7(d.body_pos_w[watch, c] - o, d.body_quat_w[watch, c]),
                   "cap_goal": _pose7(goal_link, gq),
                   "status": (f"env {watch} · goals **{int(env.successes[watch])}** · cap turned **{turned:+.2f}** "
                              f"turns · {float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                              + (f"  \nlast episode: {s.last_end}" if s.last_end else "")
                              + f"  \n{'running' if running else 'paused'} · {s.steps} steps since the last change"),
                   "stats": s.stats(), "torque": s.torque, "running": running})


# --- the page -----------------------------------------------------------------------------------------------------
class Worker:
    def __init__(self, listener, authkey: bytes, run_dir: pathlib.Path, ckpt: str, envs: int, torque: float):
        host, port = listener.address
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--run-dir", str(run_dir), "--checkpoint", ckpt, "--envs",
             str(envs), "--torque", str(torque), "--host", host, "--wport", str(port), "--authkey", authkey.hex()],
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


def _part(server, name: str, x0: float, x1: float, diameter: float, color, opacity: float = 1.0):
    """A pen part as drawn: a box from x0 to x1 along the pen (its frame at the pen's centre, as the sim's link frames
    are), with a white stripe along one face."""
    import trimesh
    box = trimesh.creation.box(extents=(x1 - x0, diameter, diameter))
    box.apply_translation(((x0 + x1) / 2, 0.0, 0.0))
    node = server.scene.add_mesh_simple(name, vertices=box.vertices, faces=box.faces, color=color, opacity=opacity)
    stripe = trimesh.creation.box(extents=(x1 - x0, diameter * 0.25, diameter * 0.08))
    stripe.apply_translation(((x0 + x1) / 2, 0.0, diameter / 2 + diameter * 0.04))
    server.scene.add_mesh_simple(f"{name}/stripe", vertices=stripe.vertices, faces=stripe.faces,
                                 color=(250, 250, 250), opacity=opacity)
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
    print(f"\n[twist-policy] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    rs = runs()
    if not rs:
        raise SystemExit(f"no runs with checkpoints under {RUNS}")
    first = find_run(args.run)[0] if args.run else next(iter(rs))
    outbox: "queue.Queue[dict]" = queue.Queue()

    with server.gui.add_folder("policy"):
        g_run = server.gui.add_dropdown("run", tuple(rs), initial_value=first)
        g_ckpt = server.gui.add_dropdown("checkpoint", tuple(checkpoints(rs[first])))
        g_envs = server.gui.add_number("envs (for the statistics)", args.envs, min=1, max=4096, step=1)
        b_load = server.gui.add_button("load (a new run restarts the sim, about a minute)")
    with server.gui.add_folder("cap"):
        g_torque = server.gui.add_slider("friction torque (N m)", 0.001, 0.25, 0.001, args.torque or 0.01)
        md_torque = server.gui.add_markdown("")
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_watch = server.gui.add_slider("watch env", 0, max(args.envs - 1, 1), 1, 0)
        b_reset = server.gui.add_button("reset the envs and the statistics")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")

    state = {"worker": None, "run": None, "ckpt": None, "view": {}, "load": True, "t_boot": 0.0}

    @g_run.on_update
    def _(_) -> None:
        g_ckpt.options = tuple(checkpoints(rs[str(g_run.value)]))

    b_load.on_click(lambda _: state.update(load=True))
    b_run.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    g_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: outbox.put({"cmd": "det", "value": bool(g_det.value)}))
    g_watch.on_update(lambda _: outbox.put({"cmd": "watch", "value": int(g_watch.value)}))
    g_torque.on_update(lambda _: outbox.put({"cmd": "torque", "value": float(g_torque.value)}))
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
        p = m["pen"]
        parts = {"barrel": _part(server, "/barrel", *p["links"]["barrel"], p["diameter"], (38, 89, 204)),
                 "cap": _part(server, "/cap", *p["links"]["cap"], p["diameter"], (220, 60, 40)),
                 "cap_goal": _part(server, "/cap_goal", *p["links"]["cap"], p["diameter"], (220, 60, 40), 0.35)}
        state["view"] = {"urdf": urdf, "parts": parts, "nodes": nodes + list(parts.values()), "joints": m["joints"]}
        g_watch.max = max(m["num_envs"] - 1, 1)
        if abs(float(g_torque.value) - m["torque"]) > 1e-9:
            g_torque.value = m["torque"]
        cur = " It's a curriculum run: the checkpoint was trained at whatever torque the curriculum had reached " \
              "by then (twist/friction_torque on wandb)." if m.get("curriculum_run") else ""
        md_torque.content = (f"The run trained at **{m['trained_torque']:.3f} N m**.{cur} Changing the torque resets "
                             f"the envs and the statistics.")

    def stats_md(st: dict, torque: float) -> str:
        if not st.get("episodes"):
            return f"**At {torque:.3f} N m:** no finished episodes yet."
        ends = ", ".join(f"{k} {v}" for k, v in sorted(st["ends"].items(), key=lambda kv: -kv[1]))
        return (f"**At {torque:.3f} N m**, over {st['episodes']} finished episodes:  \n"
                f"**{st['turns']:.2f} turns / episode** ({st['goals']:.1f} quarter-turn goals) · "
                f"{100 * st['one_turn']:.0f}% reach ≥ 1 turn  \nended: {ends}")

    t_scan = time.time()
    try:
        while True:
            if time.time() - t_scan > 10.0:        # new runs and checkpoints join the dropdowns as they are saved
                t_scan = time.time()
                now = runs()
                if now:
                    run_now, ck_now = str(g_run.value), str(g_ckpt.value)
                    rs.clear()
                    rs.update(now)
                    if tuple(rs) != tuple(g_run.options):
                        g_run.options = tuple(rs)
                        if run_now in rs:
                            g_run.value = run_now
                    if str(g_run.value) in rs:
                        cks = tuple(checkpoints(rs[str(g_run.value)]))
                        if cks != tuple(g_ckpt.options):
                            g_ckpt.options = cks
                            if ck_now in cks:
                                g_ckpt.value = ck_now
            if state.pop("load", False):
                run_label, ckpt = str(g_run.value), checkpoints(rs[str(g_run.value)])[str(g_ckpt.value)]
                if state["worker"] is not None and state["run"] == run_label and state["worker"].proc.poll() is None:
                    outbox.put({"cmd": "ckpt", "path": ckpt})
                else:
                    if state["worker"] is not None:
                        state["worker"].stop()
                    clear_view()
                    while not outbox.empty():
                        outbox.get_nowait()
                    # A new run starts at its own torque (or --torque on the first load); then the slider decides.
                    torque = args.torque if state["run"] is None else 0.0
                    state["worker"] = Worker(listener, authkey, rs[run_label], ckpt, int(g_envs.value), torque)
                    state["t_boot"] = time.time()
                    md.content = f"loading **{run_label}** -- Kit boots in a child (about a minute)"
                state["run"], state["ckpt"] = run_label, ckpt
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
            for k in ("barrel", "cap", "cap_goal"):
                pose = frame[k]
                v["parts"][k].position, v["parts"][k].wxyz = tuple(pose[:3]), tuple(pose[3:])
            md.content = frame["status"]
            md_stats.content = stats_md(frame["stats"], frame["torque"])
    except KeyboardInterrupt:
        pass
    finally:
        if state["worker"] is not None:
            state["worker"].stop()


def check(args) -> None:
    """No viewer: play the run's newest checkpoint (or --checkpoint) for --seconds of sim time at --torque, print."""
    label, d = find_run(args.check)
    ck = checkpoints(d)
    ckpt = args.checkpoint or next(iter(ck.values()))
    _boot()
    s = Session(d, ckpt, args.envs, args.torque if args.torque > 0 else None)
    steps = int(args.seconds / s.env.step_dt)
    for _ in range(steps):
        s.step(False, 0)
    st = s.stats()
    print(f"[check] {label} · {pathlib.Path(ckpt).name} · {args.envs} envs · {args.seconds:.0f} s at "
          f"{s.torque:.3f} N m (trained at {s.trained_torque:.3f})", flush=True)
    if st.get("episodes"):
        print(f"[check] {st['episodes']} episodes: {st['turns']:.2f} turns/episode ({st['goals']:.1f} goals), "
              f"{100 * st['one_turn']:.0f}% reach 1 turn, ended {st['ends']}", flush=True)
    else:
        print("[check] no episode finished", flush=True)
    os._exit(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--run", default="", help="the run to select first (a job id or part of its name)")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--torque", type=float, default=0.0, help="cap friction torque (N m); 0 = the run's own")
    ap.add_argument("--check", default="", help="no viewer: a run (job id or part of its name) to play and print")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a)
    elif a.check:
        check(a)
    else:
        main(a)
