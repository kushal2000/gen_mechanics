"""Play every fixed-palm policy in viser, and switch the task it is told and the goals it gets.

Policies: every run under debug_outputs/train_logs/{15oct_pen_repose,15oct_spin_flip_ft,15oct_pen_twist} trained with
the palm's rotation locked (its saved config's action.wrist_max_speed_ang == 0) and with a checkpoint: the v2 repose
hands and Wuji, the spin / flip fine-tunes, the locked-wrist twist runs.

Sims: one Kit per kind of scene, all booted at the start (about a minute, together) and kept up:
  primitive hands, rigid pen   one multi-hand scene of every generated primitive hand (the envs dealt round-robin,
                               --envs per hand); each hand's envs run the policy picked for that hand
  each other hand, rigid pen   e.g. Wuji
  each twist pen               the twist runs' scene (its hand, square or round cap, barrel fixed): pen_twist only
Picking a policy loads its weights onto its hand's envs (instant) and shows that hand; only the shown sim runs.

Task: the one-hot (the task the policy is told) and the goals (whose goal generator runs) are picked separately,
linked by default. Every env is pinned to the one-hot task (env.forced_task); a different goal task lends it its
reset / next_goal (and, for spin / flip, its success gate and off-course ending). The rest follows the one-hot task
(carried, pushes, success steps). Live: repose goal sampling (uniform / walk) and first goal, spin / flip success
(forward-only; off-course limits), the pen's start yaw, pushes, the twist cap's friction torque. Any change resets
the shown sim's envs and statistics.

    .venv_isaacsim/bin/python experiments/16oct_viser_evals/viser_play_all.py [--port 8097] [--envs 16]

Actions are sampled (the SAPG leader block, exploration coefficient 0), as every eval since 12 Oct; tick
"deterministic" for the mean action. Kit and viser cannot share an interpreter: each sim is a child (this file with
--worker).
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
LOGS = REPO / "debug_outputs/train_logs"
GROUPS_DIRS = ("15oct_pen_repose", "15oct_spin_flip_ft", "15oct_pen_twist")
RIGID_TASKS = ("pen_repose", "pen_spin_air", "pen_flip_air", "pen_lift", "pen_spin")
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
FRAME_PERIOD_S = 1.0 / 30.0


# --- discovering the policies ---------------------------------------------------------------------------------------
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


def _short_hand(spec: str) -> str:
    m = re.search(r"g(\d+)_(\d+)_(\d+)_(\d+)\.spec\.json$", spec)
    if m:
        return "gaps " + "/".join(str(int(g)) for g in m.groups())
    return {"wuji2_left_uniform_handonly": "Wuji full"}.get(spec, spec)


def policies() -> list[dict]:
    """Every fixed-palm run with a checkpoint: label, run_dir, spec, task, scene key, twist settings."""
    from omegaconf import OmegaConf
    out = []
    for g in GROUPS_DIRS:
        for d in sorted((LOGS / g).iterdir()):
            f = d / ".hydra/config.yaml"
            if not (d.is_dir() and f.exists() and checkpoints(d)):
                continue
            env = OmegaConf.load(f).env
            if float(env.action.wrist_max_speed_ang) != 0.0:
                continue
            spec, task = str(env.hand.robot_spec), str(list(env.task.names)[0])
            if spec.startswith("multi:"):
                continue
            twist = bool(env.pen.get("twist", False))
            job = re.search(r"_(\d+)$", d.name).group(1)
            what = {"pen_repose": "repose", "pen_spin_air": "spin ft", "pen_flip_air": "flip ft"}.get(task, task)
            if twist:
                what = f"twist ({env.pen.shape} cap)"
            scene = (f"twist:{env.pen.shape}:{spec}" if twist else
                     ("rigid:primitive" if spec.endswith(".spec.json") else f"rigid:{spec}"))
            out.append({"label": f"{what} · {_short_hand(spec)} · {job}", "run_dir": str(d), "spec": spec,
                        "task": task, "scene": scene, "twist": twist, "job": int(job)})
    order = {"repose": 0, "spin ft": 1, "flip ft": 2}
    return sorted(out, key=lambda p: (p["twist"], order.get(p["label"].split(" ·")[0], 3), p["label"]))


# --- the Kit child ----------------------------------------------------------------------------------------------------
def worker(args) -> None:
    sys.path.insert(0, str(REPO))
    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    a, _ = lp.parse_known_args([])
    a.headless = True
    app = AppLauncher(a).app  # noqa: F841
    plan = json.loads(pathlib.Path(args.plan).read_text())
    conn = Client((args.host, args.wport), authkey=bytes.fromhex(args.authkey))
    conn.send({"kind": "hello", "scene": plan["scene"]})
    try:
        _work(args, plan, conn)
    except Exception:
        import traceback
        try:
            conn.send({"kind": "error", "text": traceback.format_exc()})
        except Exception:
            pass
    os._exit(0)


def _pose7(pos, quat) -> list[float]:
    return [*pos.tolist(), *quat.tolist()]


def _work(args, plan: dict, conn) -> None:
    import torch
    import yaml
    from isaaclab.utils.math import quat_apply
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    hands = plan["hands"]                               # [{"spec", "label"}], one slot each
    twist = plan["twist"]
    env_d = OmegaConf.to_container(OmegaConf.load(pathlib.Path(plan["base_run"]) / ".hydra/config.yaml").env,
                                   resolve=True)
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = hands[0]["spec"] if len(hands) == 1 else "multi:" + "+".join(h["spec"] for h in hands)
    cfg.scene.num_envs = args.envs * len(hands)
    cfg.task.names = ("pen_twist",) if twist else RIGID_TASKS
    cfg.reset.stagger_initial_episodes = False      # whole first episodes
    cfg.pen_twist.curriculum = False                # the page's slider sets the cap's torque
    cfg.seed = 0
    conn.send({"kind": "status", "text": f"building the scene ({len(hands)} hand{'s' * (len(hands) > 1)})"})
    env = InHandOlympicsEnv(cfg)
    names = [t.name for t in env.tasks]
    env.forced_task = names.index("pen_twist" if twist else "pen_repose")
    obs, _ = env.reset()
    n_act = env.action_space.shape[-1]
    slots = list(range(len(hands)))
    envs_of = [(env.hand_of_env == h).nonzero().flatten() for h in slots]

    def make_player(slot: int, run_dir: str, ckpt: str):
        acfg = OmegaConf.to_container(OmegaConf.load(pathlib.Path(run_dir) / ".hydra/config.yaml"),
                                      resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        return RlPlayer(obs["policy"].shape[-1], n_act, f.name, ckpt, device=env.device, sapg_expl_coef=0.0,
                        num_envs=len(envs_of[slot]))

    players = [make_player(h, hands[h]["run_dir"], hands[h]["ckpt"]) for h in slots]
    loaded = [hands[h]["policy"] for h in slots]
    trained_perturbed = env.task_perturbed.clone()

    def set_mode(m: dict) -> None:
        for t in env.tasks:                               # each task's own methods back
            for k in ("reset", "next_goal", "gate", "failed"):
                t.__dict__.pop(k, None)
        if twist:
            env.forced_task = names.index("pen_twist")
        else:
            hot, goals = env.tasks[names.index(m["onehot"])], env.tasks[names.index(m["goals"])]
            env.forced_task = names.index(m["onehot"])
            if goals is not hot:                          # the told task borrows the goal task's goals (and course)
                hot.reset, hot.next_goal = goals.reset, goals.next_goal
                # Its own success gate / ending would read state its own reset no longer keeps: the goal task's
                # instead, or none.
                hot.gate = goals.gate if hasattr(type(goals), "gate") else (lambda near: near)
                hot.failed = (goals.failed if hasattr(type(goals), "failed") else
                              (lambda: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
            env.cfg.pen_repose.goal_sampling = m["repose_sampling"]
            env.cfg.pen_repose.random_first_goal = bool(m["random_first_goal"])
            for t in ("pen_spin_air", "pen_flip_air"):
                c = getattr(env.cfg, t)
                c.forward_only = bool(m["forward_only"])
                c.max_off_axis_deg = float(m["max_off_axis_deg"])
                c.max_backward_deg = float(m["max_backward_deg"])
            env.cfg.reset.randomize_pen_yaw = bool(m["random_yaw"])
            env.cfg.reset.pen_yaw_deg = float(m["yaw_deg"])
            env.task_perturbed[:] = trained_perturbed & bool(m["pushes"])

    def set_torque(t: float) -> None:
        env.cfg.pen.twist_friction_torque = float(t)
        env.pen.write_joint_effort_limit_to_sim(torch.full((env.num_envs, 1), float(t), device=env.device),
                                                joint_ids=[0])

    sx, sy, sz = cfg.table.size
    from isaacsimenvs.inhand_olympics.twist_pen import link_extents
    conn.send({"kind": "ready", "table": [sx, sy, sz, cfg.table.top_height],
               "pen": {"length": cfg.pen.length, "diameter": cfg.pen.diameter, "shape": cfg.pen.shape, "twist": twist,
                       "links": link_extents(cfg.pen.length, cfg.pen.cap_length, 0.001) if twist else None},
               "slots": [{"label": h["label"], "urdf": str(env.hand_urdf_paths[k]),
                          "joints": (env.robot.real_joint_names(int(envs_of[k][0])) if env.hand_set is not None
                                     else list(env.robot.joint_names)), "envs": len(envs_of[k])}
                         for k, h in enumerate(hands)],
               "torque": float(cfg.pen.twist_friction_torque) if twist else None})
    running, det, shown, offset, speed = False, False, 0, 0, 1.0
    finished = [deque(maxlen=5000) for _ in slots]
    ends = [dict() for _ in slots]
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
                shown, offset = int(m["slot"]), int(m.get("offset", 0))
            elif c == "speed":
                speed = float(m["value"])
            elif c == "mode":
                set_mode(m)
                restart()
            elif c == "torque" and twist:
                set_torque(m["value"])
                restart()
            elif c == "reset":
                restart()
            elif c == "policy":
                k = int(m["slot"])
                conn.send({"kind": "status", "text": f"loading {m['label']}"})
                players[k] = make_player(k, m["run_dir"], m["ckpt"])
                loaded[k] = m["label"]
                restart()
        watch = int(envs_of[shown][min(offset, len(envs_of[shown]) - 1)])
        if running:
            act = torch.zeros(env.num_envs, n_act, device=env.device)
            with torch.no_grad():
                for h in slots:
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
        o = env.scene.env_origins[watch]
        stats = []
        for h in slots:
            g = list(finished[h])
            stats.append({"policy": loaded[h], "episodes": len(g), "goals": sum(g) / len(g) if g else 0.0,
                          "one": sum(x >= 1 for x in g) / len(g) if g else 0.0, "ends": dict(ends[h])})
        cols = env.robot.real_slots(watch) if env.hand_set is not None else slice(None)
        frame = {"kind": "frame", "slot": shown, "q": env.robot.data.joint_pos[watch, cols].tolist(), "stats": stats,
                 "status": (f"{loaded[shown]}, env {watch} · task **{env.tasks[int(env.task_id[watch])].name}** · goals "
                            f"**{int(env.successes[watch])}** · goal distance {float(env.goal_dist[watch]) * 1000:.0f} mm"
                            f" · {float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                            + (f"  \nlast episode: {last_end}" if last_end else "")
                            + f"  \n{'running' if running else 'paused'} · {steps} steps since the last change")}
        if twist:
            d = env.pen.data
            b, cb = env.barrel_body, env.cap_body
            gq = env.cap_goal_quat[watch]
            goal_link = env.cap_goal_pos[watch] - quat_apply(gq.unsqueeze(0), env.cap_centre.unsqueeze(0))[0]
            frame.update(barrel=_pose7(d.body_pos_w[watch, b] - o, d.body_quat_w[watch, b]),
                         cap=_pose7(d.body_pos_w[watch, cb] - o, d.body_quat_w[watch, cb]),
                         cap_goal=_pose7(goal_link, gq))
        else:
            pos, quat = env.pen_pose()
            frame.update(pen=_pose7(pos[watch], quat[watch]), goal=_pose7(env.goal_pos[watch], env.goal_quat[watch]))
        conn.send(frame)


# --- the page ---------------------------------------------------------------------------------------------------------
def _box(server, name, x0, x1, diameter, color, opacity=1.0, round_=False, tip=False):
    """A pen part from x0 to x1 along +x: a box or a 24-gon prism, a white stripe along its top (a red end slab for a
    one-piece pen's tip instead)."""
    import numpy as np
    import trimesh
    if round_:
        m = trimesh.creation.cylinder(radius=diameter / 2, height=x1 - x0, sections=24)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    else:
        m = trimesh.creation.box(extents=(x1 - x0, diameter, diameter))
    m.apply_translation(((x0 + x1) / 2, 0.0, 0.0))
    node = server.scene.add_mesh_simple(name, vertices=m.vertices, faces=m.faces, color=color, opacity=opacity)
    if tip:
        t = trimesh.creation.box(extents=(0.15 * diameter, diameter * 1.01, diameter * 1.01))
        t.apply_translation((x1 - 0.075 * diameter, 0.0, 0.0))
        server.scene.add_mesh_simple(f"{name}/tip", vertices=t.vertices, faces=t.faces, color=(220, 40, 40),
                                     opacity=opacity)
    else:
        s = trimesh.creation.box(extents=(x1 - x0, diameter * 0.25, diameter * 0.08))
        s.apply_translation(((x0 + x1) / 2, 0.0, diameter / 2 + diameter * 0.04))
        server.scene.add_mesh_simple(f"{name}/stripe", vertices=s.vertices, faces=s.faces, color=(250, 250, 250),
                                     opacity=opacity)
    return node


def main(args) -> None:
    import signal

    import viser
    from viser.extras import ViserUrdf

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    pols = policies()
    if not pols:
        raise SystemExit("no fixed-palm policies found")
    by_label = {p["label"]: p for p in pols}
    # Scenes, and in each the hand slots (one per hand; its starting policy: its repose run if any, else its first).
    scenes: dict[str, dict] = {}
    for p in pols:
        s = scenes.setdefault(p["scene"], {"twist": p["twist"], "slots": [], "base_run": None})
        slot = next((sl for sl in s["slots"] if sl["spec"] == p["spec"]), None)
        if slot is None:
            slot = {"spec": p["spec"], "label": _short_hand(p["spec"]), "policy": p["label"]}
            s["slots"].append(slot)
        elif p["task"] == "pen_repose" and by_label[slot["policy"]]["task"] != "pen_repose":
            slot["policy"] = p["label"]
        p["slot"] = s["slots"].index(slot)
    for key, s in scenes.items():
        firsts = [by_label[sl["policy"]] for sl in s["slots"]]
        s["base_run"] = next((f["run_dir"] for f in firsts if f["task"] == "pen_repose"), firsts[0]["run_dir"])
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, port = listener.address
    for key, s in scenes.items():
        plan = {"scene": key, "twist": s["twist"], "base_run": s["base_run"],
                "hands": [{"spec": sl["spec"], "label": sl["label"], "policy": sl["policy"],
                           "run_dir": by_label[sl["policy"]]["run_dir"],
                           "ckpt": next(iter(checkpoints(pathlib.Path(by_label[sl["policy"]]["run_dir"])).values()))}
                          for sl in s["slots"]]}
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(plan, f)
        f.close()
        s.update(proc=subprocess.Popen(
            [sys.executable, __file__, "--worker", "--plan", f.name, "--envs", str(args.envs), "--host", host,
             "--wport", str(port), "--authkey", authkey.hex()],
            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}),
            conn=None, ready=None, frame=None, stats=None, status="Kit booting", t_boot=time.time(), pending=[],
            loaded=[sl["policy"] for sl in s["slots"]])

    def acceptor() -> None:
        while True:
            try:
                c = listener.accept()
                hello = c.recv()
            except Exception:
                continue
            if hello.get("scene") in scenes:
                scenes[hello["scene"]]["conn"] = c

    threading.Thread(target=acceptor, daemon=True).start()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[play-all] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    labels = [p["label"] for p in pols]
    first = next((lab for lab in labels if args.policy and args.policy in lab), labels[0])
    with server.gui.add_folder("policy"):
        g_pol = server.gui.add_dropdown("policy", tuple(labels), initial_value=first)
        g_ckpt = server.gui.add_dropdown("checkpoint", tuple(checkpoints(pathlib.Path(by_label[first]["run_dir"]))))
        b_load = server.gui.add_button("load and show")
        g_offset = server.gui.add_slider("which of its envs", 0, max(args.envs - 1, 1), 1, 0)
        md_boot = server.gui.add_markdown("")
    with server.gui.add_folder("task"):
        g_hot = server.gui.add_dropdown("one-hot (the task the policy is told)", RIGID_TASKS, initial_value="pen_repose")
        g_goals = server.gui.add_dropdown("goals (whose goal generator)", RIGID_TASKS, initial_value="pen_repose")
        g_link = server.gui.add_checkbox("link: the goals follow the one-hot", True)
        g_samp = server.gui.add_dropdown("repose goals", ("uniform", "walk"), initial_value="uniform")
        g_first = server.gui.add_checkbox("repose: first goal a random pose (off: the flat pickup)", True)
        g_fwd = server.gui.add_checkbox("spin / flip: forward-only success", True)
        g_off = server.gui.add_slider("spin / flip: end when out of the plane by (deg, 0 = never)", 0.0, 90.0, 5.0,
                                      30.0)
        g_back = server.gui.add_slider("spin / flip: end when turned back by (deg, 0 = never)", 0.0, 180.0, 5.0, 30.0)
        g_rand = server.gui.add_checkbox("pen's start yaw random", True)
        g_yaw = server.gui.add_slider("pen's start yaw (deg)", -180.0, 180.0, 5.0, -90.0)
        g_push = server.gui.add_checkbox("pushes (where the one-hot task has them)", True)
        g_torque = server.gui.add_slider("twist: cap friction torque (N m)", 0.001, 1.5, 0.001, 0.1)
        md_task = server.gui.add_markdown("")
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        b_reset = server.gui.add_button("reset the envs and the statistics")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")
    state = {"active": None, "shown": None, "running": True, "view": {}, "want": first}

    def send(key: str, msg: dict) -> None:
        c = scenes[key]["conn"]
        if c is not None:
            try:
                c.send(msg)
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass

    def to_active(msg: dict) -> None:
        if state["active"]:
            send(state["active"], msg)

    def mode_msg() -> dict:
        return {"cmd": "mode", "onehot": str(g_hot.value), "goals": str(g_goals.value),
                "repose_sampling": str(g_samp.value), "random_first_goal": bool(g_first.value),
                "forward_only": bool(g_fwd.value), "max_off_axis_deg": float(g_off.value),
                "max_backward_deg": float(g_back.value), "random_yaw": bool(g_rand.value),
                "yaw_deg": float(g_yaw.value), "pushes": bool(g_push.value)}

    @g_hot.on_update
    def _(_) -> None:
        if g_link.value and g_goals.value != g_hot.value:
            g_goals.value = g_hot.value                    # fires the goals' update, which sends the mode
        else:
            to_active(mode_msg())

    for h in (g_goals, g_samp, g_first, g_fwd, g_off, g_back, g_rand, g_yaw, g_push):
        h.on_update(lambda _: to_active(mode_msg()))
    g_link.on_update(lambda _: setattr(g_goals, "value", g_hot.value) if g_link.value else None)
    g_torque.on_update(lambda _: to_active({"cmd": "torque", "value": float(g_torque.value)}))
    g_speed.on_update(lambda _: to_active({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: to_active({"cmd": "det", "value": bool(g_det.value)}))
    g_offset.on_update(lambda _: to_active({"cmd": "show", "slot": by_label[state["want"]]["slot"],
                                            "offset": int(g_offset.value)}))
    b_reset.on_click(lambda _: to_active({"cmd": "reset"}))
    g_pol.on_update(lambda _: setattr(g_ckpt, "options",
                                      tuple(checkpoints(pathlib.Path(by_label[str(g_pol.value)]["run_dir"])))))

    @b_run.on_click
    def _(_) -> None:
        state["running"] = not state["running"]
        to_active({"cmd": "run", "value": state["running"]})

    @b_load.on_click
    def _(_) -> None:
        lab = str(g_pol.value)
        p = by_label[lab]
        s = scenes[p["scene"]]
        ckpt = checkpoints(pathlib.Path(p["run_dir"]))[str(g_ckpt.value)]
        send(p["scene"], {"cmd": "policy", "slot": p["slot"], "label": f"{lab} ({g_ckpt.value})",
                          "run_dir": p["run_dir"], "ckpt": ckpt})
        s["loaded"][p["slot"]] = lab
        if not p["twist"] and g_link.value:                # show it on its own task first
            g_hot.value = p["task"]
        state["want"], state["shown"] = lab, None

    def draw(key: str, slot: int) -> None:
        """Show this scene's slot: its sim runs (any other waits, paused), its hand and pen drawn."""
        s = scenes[key]
        if state["active"] != key:
            to_active({"cmd": "run", "value": False})
            state["active"] = key
            for msg in ({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]},
                        {"cmd": "det", "value": bool(g_det.value)}, mode_msg(),
                        {"cmd": "run", "value": state["running"]}):
                send(key, msg)
            if s["twist"]:
                send(key, {"cmd": "torque", "value": float(g_torque.value)})
        send(key, {"cmd": "show", "slot": slot, "offset": int(g_offset.value)})
        s["frame"] = None
        v, m = state["view"], s["ready"]
        for k in list(v):
            try:
                v.pop(k).remove()
            except Exception:
                pass
        sx, sy, sz, top = m["table"]
        v["table"] = server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77),
                                          position=(0.0, 0.0, top - sz / 2))
        v["base"] = server.scene.add_frame("/base", position=(0.0, 0.0, top), show_axes=False)
        info = m["slots"][slot]
        v["urdf"] = ViserUrdf(server, pathlib.Path(info["urdf"]), root_node_name="/base/robot")
        v["joints"] = info["joints"]
        pen = m["pen"]
        rnd = pen["shape"] == "round"
        if pen["twist"]:
            L = pen["links"]
            v["barrel"] = _box(server, "/barrel", *L["barrel"], pen["diameter"], (38, 89, 204), round_=rnd)
            v["cap"] = _box(server, "/cap", *L["cap"], pen["diameter"], (220, 60, 40), round_=rnd)
            v["cap_goal"] = _box(server, "/cap_goal", *L["cap"], pen["diameter"], (220, 60, 40), 0.35, round_=rnd)
        else:
            h = pen["length"] / 2
            v["pen"] = _box(server, "/pen", -h, h, pen["diameter"], (38, 89, 204), tip=True)
            v["goal"] = _box(server, "/goal", -h, h, pen["diameter"], (38, 89, 204), 0.35, tip=True)
        g_offset.max = max(info["envs"] - 1, 1)
        md_task.content = ("This scene is a twist pen: the task is pen_twist; the one-hot and goal controls don't "
                           "apply; the torque slider does." if s["twist"] else
                           "Rigid pen: every env is pinned to the one-hot task; the goals come from the goals task.")
        state["shown"] = state["want"]

    def stats_md(key: str) -> str:
        s = scenes[key]
        rows = ["| hand | its policy | episodes | goals / episode | ≥ 1 goal | how they ended |",
                "|---|---|---|---|---|---|"]
        for sl, st in zip(s["slots"], s["stats"] or [{}] * len(s["slots"])):
            if st.get("episodes"):
                e = ", ".join(f"{a} {b}" for a, b in sorted(st["ends"].items(), key=lambda kv: -kv[1]))
                rows.append(f"| {sl['label']} | {st['policy']} | {st['episodes']} | **{st['goals']:.1f}** | "
                            f"{100 * st['one']:.0f}% | {e} |")
            else:
                rows.append(f"| {sl['label']} | {(st or {}).get('policy', sl['policy'])} | 0 | – | – | – |")
        return "\n".join(rows)

    try:
        while True:
            for key, s in scenes.items():                   # read everything each sim sent
                c = s["conn"]
                if c is None:
                    continue
                try:
                    while c.poll():
                        m = c.recv()
                        k = m["kind"]
                        if k == "ready":
                            s["ready"] = m
                        elif k == "frame":
                            s["frame"], s["stats"] = m, m["stats"]
                        elif k == "status":
                            s["status"] = m["text"]
                        elif k == "error":
                            s["status"] = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                            print(m["text"], flush=True)
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    s["status"], s["conn"] = "**the sim closed the connection**", None
            md_boot.content = "  \n".join(
                f"{'▶ ' if key == state['active'] else ''}sim **{key}**: " +
                ("ready" if s["ready"] else (s["status"] if s["conn"] else
                                             f"Kit booting... {time.time() - s['t_boot']:.0f} s"))
                for key, s in scenes.items())
            p = by_label[state["want"]]
            key = p["scene"]
            s = scenes[key]
            if s["ready"] is None:
                md.content = f"waiting for the sim with **{state['want']}** to boot"
                time.sleep(0.05)
                continue
            if state["shown"] != state["want"]:
                draw(key, p["slot"])
            frame, v = s["frame"], state["view"]
            if frame is None or frame["slot"] != p["slot"]:
                time.sleep(0.01)
                continue
            s["frame"] = None
            v["urdf"].update_cfg(dict(zip(v["joints"], frame["q"])))
            for k in ("pen", "goal", "barrel", "cap", "cap_goal"):
                if k in frame and k in v:
                    v[k].position, v[k].wxyz = tuple(frame[k][:3]), tuple(frame[k][3:])
            md.content = frame["status"]
            md_stats.content = stats_md(key)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        for key in scenes:
            send(key, {"cmd": "quit"})
        for s in scenes.values():
            try:
                s["proc"].wait(timeout=10)
            except subprocess.TimeoutExpired:
                s["proc"].kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--envs", type=int, default=16, help="envs per hand")
    ap.add_argument("--policy", default="", help="the policy to show first (part of its label, e.g. 292508)")
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
