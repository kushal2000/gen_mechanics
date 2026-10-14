"""Watch trained policies in viser: every in-hand olympics run under debug_outputs/train_logs/12oct_inhand_olympics
that trained on --task (default pen_lift), and task_curriculum's Play runs (play_scratch: Kuka + SHARPA, Flexiv +
SHARPA, Flexiv + Wuji, Flexiv + Wuji human-like), each with its checkpoints. Pick a run and a checkpoint and its
policy plays in the run's own env, built from its saved config; the page draws one env -- pick which -- and keeps
statistics of every finished episode since the policy was loaded (per hand for olympics, per object type for Play).

Filters (task, hand / robot, network, push force, variant, seed) narrow the run list, and each run's training curve
(read from its logs) is shown with it and its checkpoints. The success tolerance starts at the checkpoint's own:
olympics' is fixed (the run's goal.success_tolerance); Play's shrinks with its curriculum, so it is the run's logged
current_success_tolerance at the frame the checkpoint was saved. It can be changed live. The start state can be set
for every env: olympics -- the pen's yaw (random or fixed) and how random the fingers start; Play -- the object's
orientation (random or a fixed roll/pitch/yaw). Olympics' push and twist scales can be changed live too.

    .venv_isaacsim/bin/python experiments/12oct_inhand_olympics/viser_play.py [--port 8091] [--run 257159]
    # then open http://<node>:8091
    ... viser_play.py --check 257159          # no viewer: load a run (olympics job id or Play run name), play, print

Actions are sampled (the SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. Olympics envs start with whole episodes (no staggered clocks). Play runs use
task_curriculum's eval protocol (its randomization off) with one env per object, as its own viewer
(isaacsimenvs/eval/viser_play.py there) does; that repo is only read.

Kit and viser cannot share an interpreter (coevolution/eval/viser_play.py), so the sim runs in a child worker that
streams joint positions and poses back -- this file for olympics runs, play_worker_task_curriculum.py (under
task_curriculum's venv) for Play runs. Switching to another run restarts the worker (about a minute); switching
checkpoints within a run does not.
"""
from __future__ import annotations

import argparse
import glob
import json
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

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUNS_DIR = REPO / "debug_outputs/train_logs/12oct_inhand_olympics"
TC = pathlib.Path("/share/portal/kk837/task_curriculum")
PLAY_WORKER = pathlib.Path(__file__).resolve().parent / "play_worker_task_curriculum.py"
FULL = "wuji2_left_uniform_handonly"
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "0.5x": 0.5, "2x": 2.0, "as fast as possible": 0.0}
FRAME_PERIOD_S = 1.0 / 30.0
OLYMPICS, PLAY = "olympics", "Play (task_curriculum)"


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--task", default="pen_lift", help="list the olympics runs that trained on this task")
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--no-play", action="store_true", help="leave task_curriculum's Play runs out")
    ap.add_argument("--envs", type=int, default=24, help="olympics envs the policy plays in (Play: one per object)")
    ap.add_argument("--mix-envs-per-hand", type=int, default=1,
                    help="a multi-hand olympics run plays in this many envs per hand (env i has hand i mod hands)")
    ap.add_argument("--run", default="", help="job id (or Play run name) of the run to load first; default newest")
    ap.add_argument("--check", default="", help="no viewer: load this run, play, print statistics")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--checkpoint", default="", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    return ap.parse_args()


# --- the runs (parent side: no Kit) -------------------------------------------------------------------------

def checkpoints(nn_dir: pathlib.Path) -> list[tuple[str, str]]:
    """(label, path), newest first: every last_*_ep_<N> by epoch, then the best-reward one."""
    out = []
    for p in nn_dir.glob("last_*_ep_*.pth"):
        m = re.search(r"_ep_(\d+)_", p.name)
        if m:
            out.append((int(m.group(1)), p))
    rows = [(f"epoch {e}", str(p)) for e, p in sorted(out, key=lambda x: -x[0])]
    best = sorted(nn_dir.glob("0_*.pth"))
    return rows + ([("best reward", str(best[0]))] if best else [])


def _hand_short(spec: str) -> str:
    if spec.startswith("multi:"):
        return "mix " + "+".join(_hand_short(n) for n in spec[len("multi:"):].split("+"))
    if spec == FULL:
        return "full"
    return spec.replace(FULL + "_only_", "").replace(FULL + "_", "").replace("thumb_index", "ti")


def _slurm_states(jobs) -> dict[str, str]:
    jobs = [j for j in jobs if j.isdigit()]
    if not jobs:
        return {}
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "JobID,State", "-j", ",".join(jobs)],
                             capture_output=True, text=True).stdout
        return {j: s.split()[0] for j, s in (line.split("|") for line in out.splitlines() if "|" in line)}
    except (FileNotFoundError, ValueError):
        return {}


def _status(state: dict, job: str) -> str:
    """done / training / time limit (finished by Slurm's wall clock: a whole run) / cancelled / failed / ..."""
    st = state.get(job, "")
    if st in ("RUNNING", "PENDING"):
        return "training"
    if st == "TIMEOUT":
        return "time limit"
    return st.lower() if st and st != "COMPLETED" else "done"


SHOWN_STATUSES = ("done", "training", "time limit")      # the rest only with "include cancelled runs"


def discover_olympics(runs_dir: pathlib.Path, task: str) -> list[dict]:
    import yaml
    dirs = [d for d in runs_dir.glob("0_*") if (d / ".hydra/config.yaml").is_file()]
    state = _slurm_states([d.name.rsplit("_", 1)[-1] for d in dirs])
    runs = []
    for d in dirs:
        try:
            cfg = yaml.safe_load((d / ".hydra/config.yaml").read_text())
            env, agent = cfg["env"], cfg["agent"]["params"]
            if task not in env["task"]["names"]:
                continue
        except Exception:
            continue
        nn = next(iter(d.glob("0_*/nn")), None)
        ckpts = checkpoints(nn) if nn else []
        if not ckpts:
            continue
        job = d.name.rsplit("_", 1)[-1]
        r, rw, pert = env.get("reset", {}), env.get("reward", {}), env.get("perturb", {})
        variant = []
        if r.get("stagger_initial_episodes"):
            variant.append("stagger")
        if len(env["task"]["names"]) > 1:
            variant.append("+".join(t.replace("pen_", "") for t in env["task"]["names"]))
        if env.get("pen", {}).get("shape", "capsule") != "box":
            variant.append("capsule")
        if not rw.get("approach_rew_scale", 1.0) and not rw.get("lost_pen_penalty", 1.0):
            variant.append("no approach/lost")
        if r.get("randomize_pen_yaw") is False:
            variant.append(f"yaw {r.get('pen_yaw_deg', 0):g}")
        if "_ft" in d.name or "_from_" in d.name:
            variant.append("fine-tune")
        runs.append({"kind": OLYMPICS, "job": job, "dir": str(d), "ckpts": ckpts, "status": _status(state, job),
                     "task": f"{OLYMPICS} {task}", "hand": _hand_short(env["hand"]["robot_spec"]),
                     "network": "JT" if agent["network"].get("name") == "joint_transformer" else
                                ("LSTM" if agent["network"].get("rnn") else "MLP"),
                     "seed": f"s{agent.get('seed')}", "force": f"f{float(pert.get('force_scale', 20.0)):g}",
                     "variant": " · ".join(variant) or "plain",
                     "events": f"{d}/0_*/summaries/events*",
                     "tolerance": float(env.get("goal", {}).get("success_tolerance", 0.01))})
    return runs


_PLAY_LIST = """
import json
from isaacsimenvs.eval.viser_play import find_runs
out = []
for r in find_runs():
    if r.task and r.name.startswith("play_scratch"):
        a = r.config["agent"]["params"]
        dr = r.config["env"].get("domain_randomization", {})
        out.append({"name": r.name, "dir": str(r.dir), "robot": r.robot, "task": r.task, "seed": a.get("seed"),
                    "rnn": bool(a["network"].get("rnn")), "force": dr.get("force_scale", 0.0)})
print("PLAYRUNS " + json.dumps(out))
"""


def discover_play() -> list[dict]:
    """task_curriculum's play_scratch runs, listed by that repo's own code under its venv (its package shares our
    package's name, so it cannot be imported here)."""
    py = TC / ".venv_isaacsim/bin/python"
    if not py.is_file():
        return []
    try:
        out = subprocess.run([str(py), "-c", _PLAY_LIST], cwd=TC, capture_output=True, text=True, timeout=120).stdout
        listed = json.loads(next(line for line in out.splitlines() if line.startswith("PLAYRUNS "))[9:])
    except Exception as e:
        print(f"[play] could not list task_curriculum's runs: {e}", flush=True)
        return []
    state = _slurm_states([r["name"].rsplit("_", 1)[-1] for r in listed])
    runs = []
    for p in listed:
        ckpts = checkpoints(pathlib.Path(p["dir"]) / "nn")
        if not ckpts:
            continue
        job = p["name"].rsplit("_", 1)[-1]
        tag = p["name"][len("play_scratch_"):].removeprefix(p["robot"]).strip("_").removesuffix(job).strip("_")
        runs.append({"kind": PLAY, "job": job if job.isdigit() else p["name"], "dir": p["dir"], "ckpts": ckpts,
                     "status": _status(state, job), "task": PLAY, "hand": p["robot"],
                     "network": "LSTM" if p["rnn"] else "MLP", "seed": f"s{p['seed']}",
                     "force": f"f{float(p['force'] or 0):g}", "variant": tag.replace("_", " ") or "plain",
                     "events": f"{p['dir']}/summaries/events*", "name": p["name"], "tolerance": None})
    return runs


def envs_for(run: dict, args) -> int:
    """Olympics envs to play a run in: --envs, or for a multi-hand run --mix-envs-per-hand per hand (env 0 the
    first hand, env 1 the second, ...)."""
    if run["kind"] == OLYMPICS and run["hand"].startswith("mix "):
        return args.mix_envs_per_hand * (run["hand"].count("+") + 1)
    return args.envs


def discover(args) -> list[dict]:
    runs = discover_olympics(pathlib.Path(args.runs_dir), args.task)
    if not args.no_play:
        runs += discover_play()
    for r in runs:
        r["label"] = " · ".join([r["job"], r["hand"], r["network"], r["seed"], r["force"], r["variant"]]
                                + ([f"({r['status']})"] if r["status"] != "done" else []))
    key = lambda r: (r["kind"] != OLYMPICS, -int(r["job"]) if r["job"].isdigit() else 0)  # noqa: E731
    return sorted(runs, key=key)


FILTERS = ("task", "hand", "network", "force", "variant", "seed")
FILTER_TITLES = {"task": "task", "hand": "hand / robot", "network": "network", "force": "push force",
                 "variant": "variant", "seed": "seed"}
ALL = "all"


def _scalars(events_glob: str, tags: tuple[str, ...]):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ev = glob.glob(events_glob)
    if not ev:
        return {}
    ea = EventAccumulator(ev[0], size_guidance={"scalars": 0, "histograms": 1, "images": 1, "tensors": 1})
    ea.Reload()
    have = ea.Tags()["scalars"]
    out = {t: sorted(((e.step, e.value) for e in ea.Scalars(t)), key=lambda x: x[0]) for t in tags if t in have}
    out["_per_hand"] = {t.split("/", 1)[1]: sorted(((e.step, e.value) for e in ea.Scalars(t)), key=lambda x: x[0])
                        for t in have if t.startswith("per_hand_successes/")}
    return out


def _smooth(vals, k=20):
    return [sum(vals[max(0, i - k + 1):i + 1]) / len(vals[max(0, i - k + 1):i + 1]) for i in range(len(vals))]


def _ckpt_frame_epoch(path: str) -> tuple[int, int]:
    import torch
    c = torch.load(path, map_location="cpu", weights_only=False)
    c = c[0] if isinstance(c, dict) and 0 in c else c
    return int(c.get("frame", 0)), int(c.get("epoch", 0))


def _at(series, step):
    """The last value logged at or before ``step``."""
    v = None
    for s, x in series:
        if s > step:
            break
        v = x
    return v


def training_curve(run: dict) -> dict:
    """The run's own training-time goals per episode (leader block, 20-epoch mean): epochs, the value at the end,
    best, the first epoch at >= 1 goal/ep, per hand (multi-hand), and per checkpoint the goals and, for Play, the
    curriculum's success tolerance when it was saved."""
    lead = "episode_final/successes_per_block/block_5"
    sc = _scalars(run["events"], (lead, "current_success_tolerance"))
    if lead not in sc:
        return {}
    steps = [s for s, _ in sc[lead]]
    g = _smooth([v for _, v in sc[lead]])
    out = {"epochs": len(g), "end": g[-1], "best": max(g),
           "first_one": next((i + 1 for i, x in enumerate(g) if x >= 1), None),
           "per_hand": {h: _smooth([v for _, v in s])[-1] for h, s in sc["_per_hand"].items() if s},
           "at_ckpt": {}, "tol_at_ckpt": {}}
    smoothed = list(zip(steps, g))
    for label, path in run["ckpts"]:
        if run["kind"] == PLAY:
            frame, epoch = _ckpt_frame_epoch(path)        # Play: by frame (runs resume and fine-tune)
            gv = _at(smoothed, frame)
            if "current_success_tolerance" in sc:
                tv = _at(sc["current_success_tolerance"], frame)
                if tv is not None:
                    out["tol_at_ckpt"][label] = tv
        else:
            m = re.match(r"epoch (\d+)", label)            # olympics: one point per epoch
            gv = g[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(g) else None
        if gv is not None:
            out["at_ckpt"][label] = gv
    return out


# --- the olympics worker (Kit) ------------------------------------------------------------------------------

def worker(args):
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


def _pose7(pos, quat):
    return [*pos.tolist(), *quat.tolist()]


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
    n = env.num_envs
    multi = env.hand_set is not None
    labels = list(env.hand_set.labels) if multi else [_hand_short(cfg.hand.robot_spec)]
    hand_of_env = env.hand_of_env.tolist()
    if multi:   # each hand's joint names, in the order its slice of joint_pos is sent
        hand_joints = [env.robot.real_joint_names(hand_of_env.index(h)) for h in range(len(labels))]
    else:
        hand_joints = [list(env.robot.joint_names)]
    sx, sy, sz = cfg.table.size
    conn.send({"kind": "ready", "controls": "olympics",
               "robots": [{"urdf": str(p), "joints": j} for p, j in zip(env.hand_urdf_paths, hand_joints)],
               "table": {"box": [sx, sy, sz], "pos": [0.0, 0.0, cfg.table.top_height - sz / 2]},
               "objects": {"pen": [cfg.pen.length, cfg.pen.diameter, cfg.pen.shape]},
               "env_names": [labels[h] for h in hand_of_env], "num_envs": n,
               "tolerance": float(cfg.goal.success_tolerance),
               "tolerance_note": "the run's own (fixed in this task)",
               "force_scale": cfg.perturb.force_scale, "torque_scale": cfg.perturb.torque_scale,
               "start": {"random_yaw": bool(cfg.reset.randomize_pen_yaw), "yaw_deg": float(cfg.reset.pen_yaw_deg),
                         "finger_fraction": float(cfg.reset.finger_pos_fraction)},
               "summary": f"{n} envs · tasks {', '.join(t.name for t in env.tasks)} · hands {', '.join(labels)}"})

    running, pending, det, watch, speed = True, 0, False, 0, 1.0
    finished = [deque(maxlen=1000) for _ in labels]          # goals of each finished episode, per hand
    ends = [dict() for _ in labels]                          # how they ended, per hand
    last_end, t_next, t_frame, steps = "", time.time(), 0.0, 0
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
            elif c == "tolerance":
                env.cfg.goal.success_tolerance = float(m["value"])
            elif c in ("reset", "ckpt", "init"):
                if c == "init":
                    # The start state, read by every reset from here on (env.start_pen_yaw, _reset_idx).
                    env.cfg.reset.randomize_pen_yaw = bool(m["random_yaw"])
                    env.cfg.reset.pen_yaw_deg = float(m["yaw_deg"])
                    env.cfg.reset.finger_pos_fraction = float(m["finger_fraction"])
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
        pen_pos, pen_quat = env.pen_pose()
        root = env.robot.data
        rows = ["| hand | finished episodes | goals / episode | ≥ 1 goal | how they ended |", "|---|---|---|---|---|"]
        for j, lab in enumerate(labels):
            g = list(finished[j])
            e = ", ".join(f"{k} {v}" for k, v in sorted(ends[j].items(), key=lambda kv: -kv[1])) or "–"
            rows.append(f"| {lab} | {len(g)} | {sum(g) / len(g):.1f} | {100 * sum(x >= 1 for x in g) / len(g):.0f}% "
                        f"| {e} |" if g else f"| {lab} | 0 | – | – | – |")
        force = env.pen_forces[watch, 0]
        fn = float(force.norm())
        conn.send({"kind": "frame", "robot": h, "q": root.joint_pos[watch, :len(hand_joints[h])].tolist(),
                   "robot_pose": _pose7(root.root_pos_w[watch] - env.scene.env_origins[watch], root.root_quat_w[watch]),
                   "table_pose": None, "object": None,
                   "object_pose": _pose7(pen_pos[watch], pen_quat[watch]),
                   "goal_pose": _pose7(env.goal_pos[watch], env.goal_quat[watch]),
                   "push": force.tolist() if fn > 1e-6 else None,
                   "status": (f"env {watch} ({labels[h]}) · **{env.tasks[int(env.task_id[watch])].name}** · goals "
                              f"**{int(env.successes[watch])}** · {float(env.episode_length_buf[watch]) * env.step_dt:.1f}"
                              f" s into this goal  \ngoal distance {float(env.goal_dist[watch]) * 1000:.0f} mm · pen "
                              f"{float(env.pen_lowest_height()[watch]) * 1000:.0f} mm above the table · push {fn:.2f} N"
                              f" · tolerance {env.cfg.goal.success_tolerance * 100:.1f} cm"
                              + (f"  \nlast episode: {last_end}" if last_end else "")
                              + f"  \n{'running' if running else 'paused'} · {steps} steps"),
                   "stats": "\n".join(rows), "steps": steps, "running": running})


# --- the viewer -------------------------------------------------------------------------------------------

class Worker:
    """One Kit child for one run; connects back on ``listener`` (accepted on a thread, so a child that dies while
    booting cannot hang the page). Olympics runs: this file under our venv; Play runs: the task_curriculum worker
    under that repo's venv."""

    def __init__(self, listener, authkey: bytes, run: dict, ckpt: str, envs: int, tolerance: float = 0.0,
                 tolerance_note: str = ""):
        host, port = listener.address
        self.run, self.ckpt = run, ckpt
        conn_args = ["--host", host, "--wport", str(port), "--authkey", authkey.hex()]
        if run["kind"] == PLAY:
            cmd = [str(TC / ".venv_isaacsim/bin/python"), str(PLAY_WORKER), "--run-dir", run["dir"],
                   "--checkpoint", ckpt, "--tolerance", str(tolerance), "--tolerance-note", tolerance_note,
                   *conn_args]
            cwd = str(TC)
        else:
            cmd = [sys.executable, __file__, "--worker", "--run-dir", run["dir"], "--checkpoint", ckpt,
                   "--envs", str(envs), *conn_args]
            cwd = str(REPO)
        self.proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
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


def checkpoint_tolerance(run: dict, label: str, curve: dict | None) -> tuple[float, str]:
    """The success tolerance to evaluate a checkpoint at, and where it comes from."""
    if run["kind"] == OLYMPICS:
        return run["tolerance"], "the run's own (fixed in this task)"
    tol = (curve or {}).get("tol_at_ckpt", {}).get(label)
    if tol is None:                     # not read yet: read this one now
        try:
            path = dict(run["ckpts"])[label]
            frame, _ = _ckpt_frame_epoch(path)
            sc = _scalars(run["events"], ("current_success_tolerance",))
            tol = _at(sc.get("current_success_tolerance", []), frame)
        except Exception:
            tol = None
    if tol is None:
        return 0.0, ""
    return float(tol), f"the curriculum's when this checkpoint was saved ({tol * 100:.2f} cm)"


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


def _set_pose(node, pose7):
    node.position, node.wxyz = tuple(pose7[:3]), tuple(pose7[3:])


def main(args):
    import numpy as np
    import viser
    from viser.extras import ViserUrdf

    import signal
    # A kill (SIGTERM) unwinds like Ctrl-C, so the finally below stops the Kit child instead of orphaning it.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    runs = discover(args)
    if not runs:
        raise SystemExit("no runs with checkpoints")
    first = next((r for r in runs if args.run in (r["job"], r.get("name"))), runs[0])
    print(f"[play] {sum(r['kind'] == OLYMPICS for r in runs)} olympics {args.task} runs, "
          f"{sum(r['kind'] == PLAY for r in runs)} Play runs", flush=True)

    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[play] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    server.scene.add_grid("/ground", width=3.0, height=3.0, cell_size=0.1)

    outbox: "queue.Queue[dict]" = queue.Queue()
    requests: "queue.Queue[tuple[dict, str]]" = queue.Queue()

    def _order(k, v):
        num = v[1:] if k in ("seed", "force") else ""
        return (0, float(num), "") if num.replace(".", "", 1).isdigit() else (1, 0.0, v)

    with server.gui.add_folder("choose a policy"):
        g_filter = {k: server.gui.add_dropdown(FILTER_TITLES[k], (ALL, *sorted({r[k] for r in runs},
                                                                                key=lambda v, k=k: _order(k, v))),
                                               initial_value=ALL) for k in FILTERS}
        g_cancelled = server.gui.add_checkbox("include cancelled runs", first["status"] not in SHOWN_STATUSES)
        g_run = server.gui.add_dropdown("run", ("-",))
        g_ckpt = server.gui.add_dropdown("checkpoint", ("-",))
        md_run = server.gui.add_markdown("")
        b_load = server.gui.add_button("load")
    with server.gui.add_folder("playback"):
        b_pause = server.gui.add_button("pause / resume")
        b_step = server.gui.add_button("single step")
        b_reset = server.gui.add_button("reset every env (clears the statistics)")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_watch = server.gui.add_dropdown("watch", ("env 0",))
        watch_index: dict[str, int] = {"env 0": 0}             # dropdown label -> env id
        g_tol = server.gui.add_number("success tolerance (m)", 0.01, step=0.001)
        md_tol = server.gui.add_markdown("")
    f_olym = server.gui.add_folder("start state (olympics)")
    with f_olym:
        g_yaw_mode = server.gui.add_dropdown("pen yaw", ("random (±180°)", "fixed"), initial_value="random (±180°)")
        # The pen lies along +x at yaw 0; -90 points its tip (red) to -y, the thumb side of the left hand.
        g_yaw = server.gui.add_slider("fixed yaw (deg; -90 = tip to the thumb side)", -180.0, 180.0, 5.0, -90.0)
        g_fingers = server.gui.add_slider("finger start: 0 default pose, 1 uniform over range", 0.0, 1.0, 0.05, 1.0)
        b_init = server.gui.add_button("apply (resets every env)")
    f_play = server.gui.add_folder("start state (Play)")
    with f_play:
        g_orient = server.gui.add_dropdown("object orientation", ("random (as in training)", "fixed"),
                                           initial_value="random (as in training)")
        g_rpy = [server.gui.add_slider(f"{a} (deg)", -180.0, 180.0, 5.0, 0.0) for a in ("roll", "pitch", "yaw")]
        b_init_play = server.gui.add_button("apply (resets every env) ")
    f_push = server.gui.add_folder("pushes (live, olympics)")
    with f_push:
        g_force = server.gui.add_number("force_scale", 20.0, step=1.0)
        g_torque = server.gui.add_number("torque_scale", 2.0, step=0.1)
    md_load = server.gui.add_markdown("")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")

    # The run list follows the filters; a run's name carries only what the filters leave open.
    shown: dict[str, dict] = {}          # run dropdown label -> run
    ckpt_paths: dict[str, str] = {}      # checkpoint dropdown label -> path
    ckpt_base: dict[str, str] = {}       # checkpoint dropdown label -> its plain label ("epoch N", "best reward")
    curves: dict[str, dict] = {}         # run key -> training_curve (a thread fills it)
    lock = threading.RLock()
    rkey = lambda r: r["dir"]            # noqa: E731

    def run_label(r) -> str:
        bits = [r["job"]] + [r[k] for k in ("hand", "network", "seed", "force", "variant")
                             if g_filter[k].value == ALL]
        return " · ".join(bits + ([r["status"]] if r["status"] != "done" else []))

    def selected():
        return shown.get(g_run.value)

    def refresh_info():
        r = selected()
        if r is None:
            md_run.content = "*no run matches these filters*"
            return
        c = curves.get(rkey(r))
        lines = [f"**{r['label']}**  \n{r['task']}"]
        if c is None:
            lines.append("training curve: reading the run's logs...")
        elif not c:
            lines.append("training curve: no episodes logged yet")
        else:
            one = f"≥ 1 goal/ep from epoch {c['first_one']}" if c["first_one"] else "never reached 1 goal/ep"
            lines.append(f"in training (leader block, 20-epoch mean): **{c['end']:.1f}** goals/ep at epoch "
                         f"{c['epochs']} · best {c['best']:.1f} · {one}")
            if c["per_hand"]:
                lines.append("per hand at the end: " + ", ".join(f"{h} {v:.1f}" for h, v in c["per_hand"].items()))
        md_run.content = "  \n".join(lines)

    def refresh_ckpts():
        with lock:
            r = selected()
            old = ckpt_paths.get(g_ckpt.value)
            ckpt_paths.clear()
            ckpt_base.clear()
            if r is not None:
                c = curves.get(rkey(r)) or {}
                at, tol = c.get("at_ckpt", {}), c.get("tol_at_ckpt", {})
                for lab, path in r["ckpts"]:
                    full = lab + (f" · {at[lab]:.1f} goals/ep" if lab in at else "") + \
                        (f" · tol {tol[lab] * 100:.1f} cm" if lab in tol else "")
                    ckpt_paths[full], ckpt_base[full] = path, lab
            if ckpt_paths:
                g_ckpt.options = tuple(ckpt_paths)
                keep = next((lab for lab, p in ckpt_paths.items() if p == old), None)
                g_ckpt.value = keep or g_ckpt.options[0]
            else:
                g_ckpt.options = ("-",)
            refresh_info()

    def refresh_runs(prefer: dict | None = None):
        with lock:
            cur = prefer or selected()
            match = [r for r in runs if all(g_filter[k].value in (ALL, r[k]) for k in FILTERS)
                     and (g_cancelled.value or r["status"] in SHOWN_STATUSES)]
            shown.clear()
            shown.update({run_label(r): r for r in match})
            g_run.options = tuple(shown) or ("no matching runs",)
            keep = next((lab for lab, r in shown.items() if cur is not None and rkey(r) == rkey(cur)), None)
            g_run.value = keep or g_run.options[0]
            refresh_ckpts()

    for g in (*g_filter.values(), g_cancelled):
        g.on_update(lambda _: refresh_runs())
    g_run.on_update(lambda _: refresh_ckpts())
    g_ckpt.on_update(lambda _: refresh_info())
    refresh_runs(prefer=first)

    @b_load.on_click
    def _(_):
        r = selected()
        if r is not None and g_ckpt.value in ckpt_paths:
            requests.put((r, ckpt_base[g_ckpt.value]))

    def read_curves():
        """Every run's training curve, the selected run first; a finished run's is cached by its log's size."""
        cache_path = pathlib.Path(args.runs_dir) / ".viser_play_curves.json"
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
        for r in sorted(runs, key=lambda r: r is not selected()):
            ev = glob.glob(r["events"])
            key = f"v2:{ev[0]}:{os.path.getsize(ev[0])}" if ev else ""
            hit = cache.get(rkey(r))
            if hit and hit.get("key") == key:
                c = hit["curve"]
            else:
                try:
                    c = training_curve(r)
                except Exception as e:                       # a half-written log: show nothing for it
                    print(f"[play] {r['job']}: no training curve ({e})", flush=True)
                    c = {}
                if r["status"] != "training":
                    cache[rkey(r)] = {"key": key, "curve": c}
                    try:
                        cache_path.write_text(json.dumps(cache))
                    except OSError:
                        pass
            curves[rkey(r)] = c
            if selected() is r:
                refresh_ckpts()
        print(f"[play] read {len(curves)} training curves", flush=True)

    threading.Thread(target=read_curves, daemon=True).start()

    b_init.on_click(lambda _: outbox.put({"cmd": "init", "random_yaw": g_yaw_mode.value != "fixed",
                                          "yaw_deg": float(g_yaw.value), "finger_fraction": float(g_fingers.value)}))
    b_init_play.on_click(lambda _: outbox.put({"cmd": "init", "random": g_orient.value != "fixed",
                                               "rpy": [float(g.value) for g in g_rpy]}))
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_step.on_click(lambda _: outbox.put({"cmd": "step"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    g_det.on_update(lambda _: outbox.put({"cmd": "det", "value": bool(g_det.value)}))
    g_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": SPEEDS[g_speed.value]}))
    g_watch.on_update(lambda _: outbox.put({"cmd": "watch", "value": watch_index.get(g_watch.value, 0)}))
    g_force.on_update(lambda _: outbox.put({"cmd": "force", "value": float(g_force.value)}))
    g_torque.on_update(lambda _: outbox.put({"cmd": "torque", "value": float(g_torque.value)}))
    g_tol.on_update(lambda _: outbox.put({"cmd": "tolerance", "value": float(g_tol.value)})
                    if g_tol.value > 0 else None)

    requests.put((first, first["ckpts"][0][0]))
    w: Worker | None = None
    info = None
    scene: dict = {}
    load_id = 0
    t_boot = 0.0

    def clear_scene():
        for u in scene.pop("urdfs", []):
            u.remove()
        for k in ("obj_urdfs",):
            for u in scene.pop(k, []):
                u.remove()
        for k in ("frames", "pen", "goal", "table", "table_urdf", "push", "obj_frames"):
            v = scene.pop(k, None)
            for node in (v if isinstance(v, list) else [v] if v is not None else []):
                try:
                    node.remove()
                except Exception:
                    pass
        scene.clear()

    def show_object(idx: int):
        """A Play env's object and its goal, drawn from the object's URDF (rebuilt when the watched env changes)."""
        for u in scene.pop("obj_urdfs", []):
            u.remove()
        for fr in scene.pop("obj_frames", []):
            fr.remove()
        path = pathlib.Path(info["objects"]["urdfs"][idx])
        frames, urdfs = [], []
        for name, color in (("object", (0.95, 0.6, 0.2, 1.0)), ("goal", (0.3, 0.85, 0.35, 0.35))):
            fr = server.scene.add_frame(f"/{name}{load_id}_{idx}", show_axes=False)
            frames.append(fr)
            urdfs.append(ViserUrdf(server, path, root_node_name=f"/{name}{load_id}_{idx}", mesh_color_override=color))
        scene["obj_frames"], scene["obj_urdfs"], scene["obj_idx"] = frames, urdfs, idx

    try:
        while True:
            try:
                run, label = requests.get_nowait()
                ckpt = dict(run["ckpts"])[label]
                tol, note = checkpoint_tolerance(run, label, curves.get(rkey(run)))
                if w is not None and w.alive() and w.connected() and run["dir"] == w.run["dir"]:
                    outbox.put({"cmd": "ckpt", "path": ckpt, "tolerance": tol})     # same run: no restart
                    if tol > 0:
                        g_tol.value = tol
                        md_tol.content = f"tolerance: {note}"
                    w.ckpt = ckpt
                    md_load.content = f"**{run['label']}** · {label}"
                else:
                    if w is not None:
                        w.stop()
                    while not outbox.empty():
                        outbox.get_nowait()
                    clear_scene()
                    info = None
                    load_id += 1
                    w = Worker(listener, authkey, run, ckpt, envs_for(run, args), tol, note)
                    t_boot = time.time()
                    md_load.content = f"**loading {run['label']}** · {label} -- Kit boots in a child (about a minute)"
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
                    play = m["controls"] == "play"
                    for f_, vis in ((f_olym, not play), (f_push, not play), (f_play, play)):
                        try:
                            f_.visible = vis
                        except Exception:
                            pass
                    t = m["table"]
                    if "box" in t:
                        scene["table"] = server.scene.add_box(f"/table{load_id}", dimensions=tuple(t["box"]),
                                                              color=(140, 107, 77), position=tuple(t["pos"]))
                    else:
                        scene["table"] = server.scene.add_frame(f"/table{load_id}", show_axes=False)
                        scene["table_urdf"] = ViserUrdf(server, pathlib.Path(t["urdf"]), root_node_name=f"/table{load_id}",
                                                        mesh_color_override=(0.82, 0.56, 0.35))
                    scene["frames"], scene["urdfs"] = [], []
                    for h, rob in enumerate(m["robots"]):
                        fr = server.scene.add_frame(f"/robot{load_id}_{h}", show_axes=False)
                        scene["frames"].append(fr)
                        scene["urdfs"].append(ViserUrdf(server, pathlib.Path(rob["urdf"]),
                                                        root_node_name=f"/robot{load_id}_{h}"))
                    if "pen" in m["objects"]:
                        scene["pen"] = _add_pen(server, f"/pen{load_id}", *m["objects"]["pen"])
                        scene["goal"] = _add_pen(server, f"/goal{load_id}", *m["objects"]["pen"], opacity=0.3)
                    labels = [f"env {i} · {name}" for i, name in enumerate(m["env_names"])]
                    watch_index.clear()
                    watch_index.update({lab: i for i, lab in enumerate(labels)})
                    g_watch.options = tuple(labels)
                    g_watch.value = labels[0]
                    g_tol.value = float(m["tolerance"])
                    md_tol.content = f"tolerance: {m['tolerance_note']}"
                    if play:
                        g_orient.value = "random (as in training)" if m["start"]["random"] else "fixed"
                    else:
                        g_force.value, g_torque.value = float(m["force_scale"]), float(m["torque_scale"])
                        g_yaw_mode.value = "random (±180°)" if m["start"]["random_yaw"] else "fixed"
                        g_yaw.value, g_fingers.value = m["start"]["yaw_deg"], m["start"]["finger_fraction"]
                    for c in ({"cmd": "det", "value": bool(g_det.value)},
                              {"cmd": "speed", "value": SPEEDS[g_speed.value]},
                              {"cmd": "watch", "value": 0}):
                        outbox.put(c)
                    md_load.content = f"**{w.run['label']}** · {pathlib.Path(w.ckpt).name}  \n{m['summary']}"
            if frame is None or info is None:
                continue
            h = frame["robot"]
            for k, fr in enumerate(scene["frames"]):
                fr.visible = k == h
            _set_pose(scene["frames"][h], frame["robot_pose"])
            scene["urdfs"][h].update_cfg(dict(zip(info["robots"][h]["joints"], frame["q"])))
            if frame["table_pose"] is not None and "table_urdf" in scene:
                _set_pose(scene["table"], frame["table_pose"])
            if frame["object"] is not None:
                if scene.get("obj_idx") != frame["object"]:
                    show_object(frame["object"])
                _set_pose(scene["obj_frames"][0], frame["object_pose"])
                _set_pose(scene["obj_frames"][1], frame["goal_pose"])
            else:
                _set_pose(scene["pen"], frame["object_pose"])
                _set_pose(scene["goal"], frame["goal_pose"])
            if "push" in scene:
                scene.pop("push").remove()
            if frame["push"] is not None:
                f = np.asarray(frame["push"], dtype=np.float32)
                p = np.asarray(frame["object_pose"][:3], dtype=np.float32)
                scene["push"] = server.scene.add_line_segments(
                    "/push", points=np.stack([p, p + 0.1 * f / np.linalg.norm(f)])[None], colors=(230, 30, 30),
                    line_width=4.0)
            md.content = frame["status"]
            md_stats.content = frame["stats"]
    except KeyboardInterrupt:
        pass
    finally:
        if w is not None:
            w.stop()


def check(args):
    """No viewer: load one run's newest checkpoint, set a fixed start, play as fast as possible, switch checkpoint in
    the same worker, and print what the page would show."""
    runs = discover(args)
    run = next((r for r in runs if args.check in (r["job"], r.get("name"))), None)
    if run is None:
        raise SystemExit(f"{args.check} is not a run with checkpoints")
    label = run["ckpts"][0][0]
    tol, note = checkpoint_tolerance(run, label, None)
    print(f"[check] {run['label']} · {label} · tolerance {tol} ({note})", flush=True)
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    w = Worker(listener, authkey, run, run["ckpts"][0][1], envs_for(run, args), tol, note)
    while not w.connected():
        if not w.alive():
            raise SystemExit("[check] the worker died before connecting")
        time.sleep(0.5)
    w.conn.send({"cmd": "speed", "value": 0.0})
    ready, last, switched, reset_seen = None, None, False, False
    while w.alive():
        if not w.conn.poll(timeout=1.0):
            continue
        m = w.conn.recv()
        if m["kind"] == "error":
            print(m["text"], flush=True)
            break
        if m["kind"] == "ready":
            ready = m
            print(f"[check] ready: {m['summary']}; robots {[r['urdf'] for r in m['robots']]}; tolerance "
                  f"{m['tolerance']} ({m['tolerance_note']})", flush=True)
            init = ({"cmd": "init", "random": False, "rpy": [0.0, 0.0, 90.0]} if m["controls"] == "play" else
                    {"cmd": "init", "random_yaw": False, "yaw_deg": -90.0, "finger_fraction": 0.0})
            w.conn.send(init)
            print(f"[check] start state: {init}", flush=True)
        if m["kind"] == "frame":
            last = m
            # Frames already in flight when the switch is sent are from before it; it has landed once steps restart.
            reset_seen = reset_seen or (switched and m["steps"] < 300)
            if m["steps"] >= 300 and not switched and len(run["ckpts"]) > 1:
                lab2 = run["ckpts"][1][0]
                tol2, _ = checkpoint_tolerance(run, lab2, None)
                print(f"[check] 300 steps:\n{m['stats']}\n[check] switching to {lab2} (tolerance {tol2})", flush=True)
                w.conn.send({"cmd": "ckpt", "path": run["ckpts"][1][1], "tolerance": tol2})
                w.conn.send({"cmd": "watch", "value": 1})
                switched = True
            elif m["steps"] >= 600 and (reset_seen or len(run["ckpts"]) == 1):
                break
    if last is not None:
        print(f"[check] watched: robot {last['robot']}, {len(last['q'])} joint positions, object {last['object']}"
              f"\n{last['status']}\n{last['stats']}", flush=True)
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
