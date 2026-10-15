"""Play every v2 pen_repose policy at once in viser -- on repose goals and, with no retraining, on other tasks' goals.

One sim (one Kit) per hand layout holds every v2 run's hand of that layout in one scene (a multi-hand scene:
hand.robot_spec "multi:a+b+...", the envs dealt round-robin, --envs per hand): all the primitive hands (22 joints,
4 tips) in one, Wuji (26, 5) in another -- a scene pads every hand to its largest, which would change the other
hands' observations, so layouts don't mix. Each hand's envs are driven by that run's own policy. The page draws one
env of the hand picked -- the hand, the pen (its tip red) and the goal (translucent) -- and keeps statistics of every
episode finished since the last change, for every hand side by side. Every sim boots at the start (about a
minute, together); then switching the hand shown, or any hand's checkpoint (its weights), is instant. Only the sim of
the hand shown runs; the other waits, paused.

v2 runs: RUN_TAG uni_fw under debug_outputs/train_logs/15oct_pen_repose (wrist rotation locked, every goal a fresh
random pose); the same recipe in every run but the hand, so the scene is built from the first one's saved config.

Goals: the env stays pen_repose (so the policies see the task one-hot they trained with) and only its goal
generator changes, live:
  repose: uniform        every goal a fresh pose from the whole goal space (what v2 trained on)
  repose: random walk    each goal the last turned 15-30 deg and moved up to 5 cm (v1's)
  spin in the air        pen_spin_air's goals: picked up flat, then each goal step_deg further round the vertical
  flip in the air        pen_flip_air's goals: picked up flat, then each goal step_deg further end over end
The pen's start: flat on the table, its yaw random or fixed (the angle of its tip from the fingers' direction). The
repose modes' first goal: a random pose or the flat pickup. Pushes on the lifted pen: on (as trained) or off. Any
change resets the envs and the statistics.

    .venv_isaacsim/bin/python experiments/15oct_pen_repose/viser_repose_play.py [--port 8096] [--envs 16]

Actions are sampled (each policy's SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. Kit and viser cannot share an interpreter: the sim runs in a child (this file
with --worker).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import queue
import re
import secrets
import socket
import subprocess
import sys
import tempfile
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
    """label -> run directory: the v2 runs (RUN_TAG uni_fw) with a checkpoint, by job (a primitive hand by its gaps,
    the full Wuji hand -- no hand tag in the name -- as Wuji)."""
    out = []
    for d in RUNS.iterdir():
        if not ((d / ".hydra/config.yaml").exists() and checkpoints(d)):
            continue
        m = re.search(r"prim_g(\d+)_(\d+)_(\d+)_(\d+)_uni_fw_(\d+)$", d.name)
        if m:
            out.append((int(m.group(5)), f"gaps {'/'.join(str(int(g)) for g in m.groups()[:4])} · {m.group(5)}", d))
        elif (w := re.search(r"_box_uni_fw_(\d+)$", d.name)):
            out.append((int(w.group(1)), f"Wuji full · {w.group(1)}", d))
    return {label: d for _, label, d in sorted(out)}


def layout(spec: str) -> str:
    """Which scene a hand goes in: every generated primitive hand together, any other hand on its own."""
    return "primitive" if spec.endswith(".spec.json") else spec


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


def hand_spec(run: pathlib.Path) -> str:
    m = re.search(r"robot_spec: (\S+)", (run / ".hydra/config.yaml").read_text())
    return m.group(1)


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
    conn.send({"kind": "hello", "first": json.loads(pathlib.Path(args.plan).read_text())[0]["label"]})
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
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.tasks.pen_flip import PenFlipAir
    from isaacsimenvs.inhand_olympics.tasks.pen_spin import PenSpinAir

    plan = json.loads(pathlib.Path(args.plan).read_text())       # [{label, run_dir, ckpt, spec}], one per hand
    saved = OmegaConf.load(pathlib.Path(plan[0]["run_dir"]) / ".hydra/config.yaml")
    env_d = OmegaConf.to_container(saved.env, resolve=True)
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = "multi:" + "+".join(h["spec"] for h in plan)
    cfg.scene.num_envs = args.envs * len(plan)
    cfg.reset.stagger_initial_episodes = False      # whole first episodes
    cfg.seed = 0
    conn.send({"kind": "status", "text": f"building one scene with {len(plan)} hands"})
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    n_act = env.action_space.shape[-1]
    hands = list(range(len(plan)))
    envs_of = [(env.hand_of_env == h).nonzero().flatten() for h in hands]
    yamls = []
    for h in plan:
        acfg = OmegaConf.to_container(OmegaConf.load(pathlib.Path(h["run_dir"]) / ".hydra/config.yaml"),
                                      resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        yamls.append(f.name)

    def make_player(h: int, ckpt: str):
        return RlPlayer(obs["policy"].shape[-1], n_act, yamls[h], ckpt, device=env.device, sapg_expl_coef=0.0,
                        num_envs=len(envs_of[h]))

    players = [make_player(h, plan[h]["ckpt"]) for h in hands]
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
    conn.send({"kind": "ready", "table": [sx, sy, sz, cfg.table.top_height], "pen": [cfg.pen.length, cfg.pen.diameter],
               "hands": [{"label": plan[h]["label"], "urdf": str(env.hand_urdf_paths[h]),
                          "joints": env.robot.real_joint_names(int(envs_of[h][0])), "envs": len(envs_of[h])}
                         for h in hands],
               "trained": {"goal_sampling": cfg.pen_repose.goal_sampling,
                           "random_first_goal": cfg.pen_repose.random_first_goal,
                           "wrist_ang": cfg.action.wrist_max_speed_ang, "first_tol": cfg.goal.first_goal_tolerance}})
    running, det, shown, offset, speed = False, False, 0, 0, 1.0         # paused until the page shows it
    finished = [deque(maxlen=5000) for _ in hands]
    ends = [dict() for _ in hands]
    last_end, steps, t_next, t_frame = "", 0, time.time(), 0.0

    def restart():
        nonlocal obs, last_end, steps
        obs, _ = env.reset()
        for q in finished:
            q.clear()
        for e in ends:
            e.clear()
        last_end, steps = "", 0

    while True:
        while conn.poll():
            m = conn.recv()
            c = m["cmd"]
            if c == "quit":
                return
            if c == "run":
                running = bool(m["value"])
            elif c == "det":
                det = bool(m["value"])
                restart()
            elif c == "show":
                shown, offset = int(m["hand"]), int(m.get("offset", 0))
            elif c == "speed":
                speed = float(m["value"])
            elif c == "mode":
                set_mode(m)
                restart()
            elif c == "reset":
                restart()
            elif c == "ckpt":
                conn.send({"kind": "status", "text": f"loading {plan[int(m['hand'])]['label']}'s checkpoint"})
                players[int(m["hand"])] = make_player(int(m["hand"]), m["path"])
                restart()
        watch = int(envs_of[shown][min(offset, len(envs_of[shown]) - 1)])
        if running:
            act = torch.zeros(env.num_envs, n_act, device=env.device)
            with torch.no_grad():
                for h in hands:
                    act[envs_of[h]] = players[h].get_normalized_action(obs["policy"][envs_of[h]],
                                                                       deterministic_actions=det)
            before = env.successes.clone()
            obs, _, term, trunc, _ = env.step(act)
            steps += 1
            done = term | trunc
            if bool(done.any()):
                goals = (before + env.is_success.float())[done].tolist()
                for i, g in zip(done.nonzero().flatten().tolist(), goals):
                    h = int(env.hand_of_env[i])
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
        pos, quat = env.pen_pose()
        stats = []
        for h in hands:
            g = list(finished[h])
            stats.append({"episodes": len(g), "goals": sum(g) / len(g), "one": sum(x >= 1 for x in g) / len(g),
                          "ends": dict(ends[h])} if g else {"episodes": 0})
        conn.send({"kind": "frame", "hand": shown,
                   "q": env.robot.data.joint_pos[watch, :len(env.robot.real_joint_names(watch))].tolist(),
                   "pen": _pose7(pos[watch], quat[watch]),
                   "goal": _pose7(env.goal_pos[watch], env.goal_quat[watch]),
                   "status": (f"{plan[shown]['label']}, env {watch} · goals **{int(env.successes[watch])}** · goal "
                              f"distance {float(env.goal_dist[watch]) * 1000:.0f} mm · "
                              f"{float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                              + (f"  \nlast episode: {last_end}" if last_end else "")
                              + f"  \n{'running' if running else 'paused'} · {steps} steps since the last change"),
                   "stats": stats})


# --- the page -----------------------------------------------------------------------------------------------------
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
    rs = runs()
    if not rs:
        raise SystemExit(f"no v2 runs with checkpoints under {RUNS}")
    labels = list(rs)
    specs = {lab: hand_spec(d) for lab, d in rs.items()}
    groups: dict[str, list[str]] = {}                  # layout -> its hands' labels, in the order its sim deals them
    for lab in labels:
        groups.setdefault(layout(specs[lab]), []).append(lab)
    where = {lab: (g, k) for g, labs in groups.items() for k, lab in enumerate(labs)}
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    sims: dict[str, dict] = {}
    for g, labs in groups.items():
        plan = [{"label": lab, "run_dir": str(rs[lab]), "ckpt": next(iter(checkpoints(rs[lab]).values())),
                 "spec": specs[lab]} for lab in labs]
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(plan, f)
        f.close()
        proc = subprocess.Popen(
            [sys.executable, __file__, "--worker", "--plan", f.name, "--envs", str(args.envs), "--host", host,
             "--wport", str(port), "--authkey", authkey.hex()],
            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        sims[g] = {"proc": proc, "conn": None, "ready": None, "status": "Kit booting", "frame": None, "stats": None,
                   "t_boot": time.time()}

    def acceptor() -> None:
        """Each sim's first message names its first hand; route its connection to its group."""
        while True:
            try:
                conn = listener.accept()
            except Exception:
                continue
            m = conn.recv()
            first_label = m.get("first")
            g = where.get(first_label, (None,))[0]
            if g in sims:
                sims[g]["conn"] = conn
                sims[g]["pending"] = [m]

    threading.Thread(target=acceptor, daemon=True).start()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[repose-play] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    first = next((lab for lab in labels if args.run and args.run in lab), labels[0])
    with server.gui.add_folder("hand and policy"):
        g_hand = server.gui.add_dropdown("hand shown", tuple(labels), initial_value=first)
        g_ckpt = server.gui.add_dropdown("its checkpoint", tuple(checkpoints(rs[first])))
        b_ckpt = server.gui.add_button("load this checkpoint for this hand")
        g_offset = server.gui.add_slider("which of its envs", 0, max(args.envs - 1, 1), 1, 0)
        md_run = server.gui.add_markdown("")
        md_boot = server.gui.add_markdown("")
    with server.gui.add_folder("goals and start (every hand)"):
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
        b_reset = server.gui.add_button("reset the envs and the statistics")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")

    state = {"view": {}, "shown": None, "active": None, "running": True,
             "loaded": {lab: next(iter(checkpoints(rs[lab]).values())) for lab in labels}}

    def send(g: str, msg: dict) -> None:
        c = sims[g]["conn"]
        if c is not None:
            try:
                c.send(msg)
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass

    def to_active(msg: dict) -> None:
        if state["active"] is not None:
            send(state["active"], msg)

    def to_all(msg: dict) -> None:
        for g in sims:
            send(g, msg)

    def mode_msg() -> dict:
        return {"cmd": "mode", "mode": str(g_mode.value), "step_deg": float(g_step.value),
                "random_first_goal": bool(g_first.value), "random_yaw": bool(g_rand.value),
                "yaw_deg": float(g_yaw.value), "pushes": bool(g_push.value)}

    def show_msg() -> dict:
        return {"cmd": "show", "hand": where[str(g_hand.value)][1], "offset": int(g_offset.value)}

    for h in (g_mode, g_step, g_first, g_rand, g_yaw, g_push):
        h.on_update(lambda _: to_all(mode_msg()))
    g_speed.on_update(lambda _: to_all({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: to_all({"cmd": "det", "value": bool(g_det.value)}))
    g_offset.on_update(lambda _: to_active(show_msg()))
    b_reset.on_click(lambda _: to_active({"cmd": "reset"}))

    @g_hand.on_update
    def _(_) -> None:
        lab = str(g_hand.value)
        cks = checkpoints(rs[lab])
        g_ckpt.options = tuple(cks)
        cur = next((k for k, p in cks.items() if p == state["loaded"][lab]), None)
        if cur:
            g_ckpt.value = cur
        state["shown"] = None                          # redraw (and switch sims if its layout differs)

    @b_ckpt.on_click
    def _(_) -> None:
        lab = str(g_hand.value)
        path = checkpoints(rs[lab])[str(g_ckpt.value)]
        state["loaded"][lab] = path
        g, k = where[lab]
        send(g, {"cmd": "ckpt", "hand": k, "path": path})

    @b_run.on_click
    def _(_) -> None:
        state["running"] = not state["running"]
        to_active({"cmd": "run", "value": state["running"]})

    def build_scene(m: dict) -> None:
        if "pen" in state["view"]:
            return
        sx, sy, sz, top = m["table"]
        server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77), position=(0.0, 0.0, top - sz / 2))
        state["view"]["base"] = server.scene.add_frame("/base", position=(0.0, 0.0, top), show_axes=False)
        state["view"]["pen"] = _pen(server, "/pen", *m["pen"])
        state["view"]["goal"] = _pen(server, "/goal", *m["pen"], opacity=0.35)

    def draw_hand(lab: str) -> None:
        """Show this hand: its sim runs (the other waits, paused), its URDF swaps in."""
        g, k = where[lab]
        if state["active"] != g:
            to_active({"cmd": "run", "value": False})
            state["active"] = g
            send(g, {"cmd": "run", "value": state["running"]})
        send(g, show_msg())
        sims[g]["frame"] = None
        v, m = state["view"], sims[g]["ready"]
        if "urdf" in v:
            v["urdf"].remove()
        info = m["hands"][k]
        v["urdf"] = ViserUrdf(server, pathlib.Path(info["urdf"]), root_node_name="/base/robot")
        v["joints"] = info["joints"]
        g_offset.max = max(info["envs"] - 1, 1)
        state["shown"] = lab
        t = m["trained"]
        md_run.content = (f"Every hand trained on repose goals **{t['goal_sampling']}**, first goal "
                          f"{'random' if t['random_first_goal'] else 'the flat pickup'} (tolerance "
                          f"{(t['first_tol'] or 0.01) * 100:.0f} cm), wrist rotation "
                          f"{'locked' if t['wrist_ang'] == 0 else 'free'}. {info['envs']} envs per hand.")

    def stats_md() -> str:
        rows = [f"**{g_mode.value}** -- every hand, since the last change (a paused sim's hands stand still)", "",
                "| hand | episodes | goals / episode | ≥ 1 goal | how they ended |", "|---|---|---|---|---|"]
        for lab in labels:
            g, k = where[lab]
            st = (sims[g]["stats"] or [{}] * len(groups[g]))[k]
            if st.get("episodes"):
                e = ", ".join(f"{a} {b}" for a, b in sorted(st["ends"].items(), key=lambda kv: -kv[1]))
                rows.append(f"| {lab} | {st['episodes']} | **{st['goals']:.1f}** | {100 * st['one']:.0f}% | {e} |")
            else:
                rows.append(f"| {lab} | 0 | – | – | – |")
        return "\n".join(rows)

    t_scan = time.time()
    try:
        while True:
            if time.time() - t_scan > 10.0:             # new checkpoints join the dropdown as they are saved
                t_scan = time.time()
                cks = tuple(checkpoints(rs[str(g_hand.value)]))
                if cks != tuple(g_ckpt.options):
                    cur = str(g_ckpt.value)
                    g_ckpt.options = cks
                    if cur in cks:
                        g_ckpt.value = cur
            for g, sim in sims.items():                    # read everything each sim sent
                c = sim["conn"]
                if c is None:
                    continue
                batch = sim.pop("pending", [])
                try:
                    while c.poll():
                        batch.append(c.recv())
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    sim["status"], sim["conn"] = "**the sim closed the connection**", None
                for m in batch:
                    k = m["kind"]
                    if k == "ready":
                        sim["ready"] = m
                        build_scene(m)
                        for msg in (mode_msg(), {"cmd": "speed", "value": SPEEDS[str(g_speed.value)]},
                                    {"cmd": "det", "value": bool(g_det.value)}):
                            send(g, msg)
                    elif k == "frame":
                        sim["frame"], sim["stats"] = m, m["stats"]
                    elif k == "status":
                        sim["status"] = m["text"]
                    elif k == "error":
                        sim["status"] = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                        print(m["text"], flush=True)
            md_boot.content = "  \n".join(
                f"{'▶ ' if g == state['active'] else ''}sim for {', '.join(l.split(' ·')[0] for l in groups[g])}: "
                + ("ready" if s_["ready"] else (s_["status"] if s_["conn"] else
                                                 f"Kit booting... {time.time() - s_['t_boot']:.0f} s"))
                for g, s_ in sims.items())
            lab = str(g_hand.value)
            g = where[lab][0]
            if sims[g]["ready"] is None:
                md.content = f"waiting for the sim with **{lab}** to boot"
                time.sleep(0.05)
                continue
            if state["shown"] != lab:
                draw_hand(lab)
            frame, v = sims[g]["frame"], state["view"]
            if frame is None or frame["hand"] != where[lab][1]:
                time.sleep(0.01)
                continue
            sims[g]["frame"] = None
            v["urdf"].update_cfg(dict(zip(v["joints"], frame["q"])))
            for k in ("pen", "goal"):
                v[k].position, v[k].wxyz = tuple(frame[k][:3]), tuple(frame[k][3:])
            md.content = frame["status"]
            md_stats.content = stats_md()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        to_all({"cmd": "quit"})
        for sim in sims.values():
            try:
                sim["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                sim["proc"].kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--run", default="", help="the hand to show first (a job id or its gaps, e.g. 292508)")
    ap.add_argument("--envs", type=int, default=16, help="envs per hand")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--plan", default="", help=argparse.SUPPRESS)
    ap.add_argument("--host", default="", help=argparse.SUPPRESS)
    ap.add_argument("--wport", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--authkey", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        worker(a)
    else:
        main(a)
