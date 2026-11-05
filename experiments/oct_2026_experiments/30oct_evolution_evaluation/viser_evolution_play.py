"""Play any evolution policy on any evolved (or seed) hand, on the top-down pen-repose task, in viser.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/viser_evolution_play.py [--port 8097] [--envs 4]
        [--sources seed map_elites baseline] [--hands map_elites:g0018_c00 baseline:g0017_c01 <spec.json> ...]

One sim (one Kit, a child process) holds every hand of the chosen sources in ONE template scene (``multi:template:``,
the evolution runs' pinned 5 + 2 x 5 slot grid), ``--envs`` envs per hand:
  seed         the 30 seed hands (assets/urdf/29oct_niche_seed), one per niche: what every run started from
  map_elites   MAP-Elites' current map: the elite of each niche
  baseline     the baseline's population at its latest finished iteration
  rl_only      (same hands as seed)
plus any ``--hands`` (``<run>:<hand id>`` from that run's archive, or a .spec.json path). A hand in several sources
appears once.

Every policy the evolution experiment produced has the same network, observations and slot grid, so any of them can
drive any of these hands: the page's "policy" picks a source (map_elites, baseline, rl_only: one policy per
iteration; the long single runs: their kept checkpoints and the rolling latest) and a checkpoint; loading one swaps the
weights for every env (the scene stays; players are cached, so switching back is instant). The env is the runs'
training env exactly (the latest MAP-Elites iteration's saved config: pen repose v2, uniform goals, wrist rotation
locked, the pre-27-Oct tokens), actions sampled from the SAPG leader block (exploration 0) unless "deterministic".

The page draws one env of the hand picked (hand on its wrist, the box pen with its red tip, the goal translucent) and
keeps statistics of every episode finished under the current policy since the last change, for every hand.
Run it on a GPU node (the interactive allocation): open http://<node>:<port>. Kit and viser cannot share an
interpreter, so the sim runs in a child (this file with --worker).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
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
EVO = REPO / "debug_outputs/evolution"
EVO_LOGS = REPO / "debug_outputs/train_logs/evolution"
LONG = REPO / "debug_outputs/train_logs/29oct_evolution_from_scratch/long_runs"
SEED = REPO / "assets/urdf/29oct_niche_seed/manifest.json"
LOOP_RUNS = ("map_elites", "baseline", "rl_only")
LONG_RUNS = ("claude_runner_up_long", "claude_pick_long", "random_pick_long", "rl_only_long")
LENGTH = {"j0": "short", "j1": "medium", "j2": "long"}
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
REASONS = ("max_successes", "timeout", "dropped", "fell", "away", "off_course")
FRAME_PERIOD_S = 1.0 / 30.0


# --- what to load -------------------------------------------------------------------------------------------------
def _niche_words(n: str) -> str:
    f, cut, j = n.split("_")
    return f"{f[1:]} fingers, {'cut' if cut == 'cut' else 'no cut'}, {LENGTH[j]}"


def _latest(run: str) -> int | None:
    """The run's latest iteration with a policy (finished)."""
    gens = sorted(int(p.parent.name[4:]) for p in (EVO / run).glob("gen_*/policy.pth"))
    return gens[-1] if gens else None


def _score(rec: dict, g: int) -> float | None:
    e = [x for x in rec["fitness"] if x["generation"] == g and not x["screen"]]
    return e[-1]["goals_per_episode"] if e else None


def hands_for(sources: list[str], extra: list[str]) -> list[dict]:
    """[{spec, label}] in display order, one per spec (a hand in several sources keeps its first label)."""
    out: dict[str, str] = {}

    def add(spec: str, label: str) -> None:
        out.setdefault(spec, label)

    for src in sources:
        if src in ("seed", "rl_only"):
            for h in json.loads(SEED.read_text())["hands"]:
                add(h["spec"], f"seed · {_niche_words(h['niche'])} · {h['id']}")
        elif src in ("map_elites", "baseline"):
            st = json.loads((EVO / src / "state.json").read_text())
            arch = json.loads((EVO / src / "archive.json").read_text())["hands"]
            recs = {r["id"]: r for r in (arch if isinstance(arch, list) else arch.values())}
            g = _latest(src)
            if src == "map_elites":
                ids = [st["map"][n] for n in sorted(st["map"])]
            else:
                ids = [h["id"] for h in json.loads((EVO / src / f"gen_{g:04d}/manifest.json").read_text())["hands"]]
                ids.sort(key=lambda i: -(_score(recs[i], g) or 0.0))
            for i in ids:
                r = recs[i]
                s = _score(r, g)
                n = _niche_of(r)
                add(r["spec"], f"{src} it{g} · {_niche_words(n)} · {i}" + (f" · {s:g} goals/ep" if s is not None else ""))
        else:
            raise SystemExit(f"unknown source {src!r}")
    for x in extra:
        if x.endswith(".spec.json"):
            add(x, f"extra · {pathlib.Path(x).name.replace('.spec.json', '')}")
            continue
        run, hid = x.split(":", 1)
        arch = json.loads((EVO / run / "archive.json").read_text())["hands"]
        recs = {r["id"]: r for r in (arch if isinstance(arch, list) else arch.values())}
        add(recs[hid]["spec"], f"extra · {run} · {_niche_words(_niche_of(recs[hid]))} · {hid}")
    return [{"spec": s, "label": lab} for s, lab in out.items()]


def _niche_of(rec: dict) -> str:
    from coevolution.evolution.descriptors import niche
    d = rec["design"]
    if isinstance(d, str):
        import ast
        d = ast.literal_eval(d)
    return niche(d)


def policies() -> dict[str, dict[str, str]]:
    """source -> {checkpoint label: path}, newest first."""
    out: dict[str, dict[str, str]] = {}
    for run in LOOP_RUNS:
        gens = sorted(((int(p.parent.name[4:]), p) for p in (EVO / run).glob("gen_*/policy.pth")), reverse=True)
        if gens:
            out[run] = {f"iteration {g}" + (" (latest)" if k == 0 else ""): str(p) for k, (g, p) in enumerate(gens)}
    for name in LONG_RUNS:
        d = next(iter(sorted(LONG.glob(f"*_evo_{name}_*"))), None)
        if d is None:
            continue
        cks: dict[str, str] = {}
        roll = d / d.name / "last" / "model.pth"
        if roll.exists():
            cks["latest (rolling, every 200 epochs)"] = str(roll)
        for ep, p in sorted(((int(re.search(r"_ep_(\d+)_", p.name).group(1)), p)
                             for p in (d / d.name / "nn").glob("last_*_ep_*.pth")), reverse=True):
            cks[f"epoch {ep} (= end of iteration {ep // 636 - 1})"] = str(p)
        best = d / d.name / "nn" / f"{d.name}.pth"
        if best.exists():
            cks["best training reward"] = str(best)
        if cks:
            out[name] = cks
    return out


def env_run() -> pathlib.Path:
    """A training run whose saved config is the env and agent: the latest finished MAP-Elites iteration."""
    g = _latest("map_elites")
    run = next((EVO_LOGS / "map_elites" / f"gen_{g:04d}").glob("*/.hydra/config.yaml"), None)
    if run is None:
        raise SystemExit("no saved MAP-Elites training config found")
    return run.parent.parent


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
    import torch
    import yaml
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg

    plan = json.loads(pathlib.Path(args.plan).read_text())   # {"run": train run dir, "manifest": path, "hands": [...]}
    full = OmegaConf.to_container(OmegaConf.load(pathlib.Path(plan["run"]) / ".hydra/config.yaml"), resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    ref = f"multi:template:{plan['manifest']}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs * len(plan["hands"])
    cfg.reset.stagger_initial_episodes = False
    cfg.seed = 0
    conn.send({"kind": "status", "text": f"building one scene: {len(plan['hands'])} hands x {args.envs} envs"})
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    if [s.name for s in env.hand_set.specs] != [h["spec"] for h in plan["hands"]]:
        raise RuntimeError("the scene's hands are not the plan's, in order")
    n_act = env.action_space.shape[-1]
    nh = len(plan["hands"])
    envs_of = [(env.hand_of_env == h).nonzero().flatten() for h in range(nh)]
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()
    players: dict[str, object] = {}

    def player(path: str):
        if path not in players:
            conn.send({"kind": "status", "text": f"loading {pathlib.Path(path).name}"})
            players[path] = RlPlayer(obs["policy"].shape[-1], n_act, f.name, path, device=env.device,
                                     sapg_expl_coef=0.0, num_envs=env.num_envs)
        return players[path]

    trained_perturbed = env.task_perturbed.clone()
    from isaacsimenvs.inhand_olympics.pose_viewer import _cylinders_as_capsules

    def drawn_urdf(h: int) -> str:
        """The hand on its wrist as PhysX simulates it: where the spec swaps cylinders for capsules (the grammar
        hands' links), every visual cylinder gets a sphere at each end (the wandb pose viewer's rule)."""
        src = pathlib.Path(env.hand_urdf_paths[h])
        if not getattr(env.hand_set.specs[h], "replace_cylinders_with_capsules", False):
            return str(src)
        out = src.with_name("hand_on_wrist_capsules.urdf")          # mesh paths in it are absolute
        out.write_text(_cylinders_as_capsules(src.read_text()))
        return str(out)

    sx, sy, sz = cfg.table.size
    conn.send({"kind": "ready", "table": [sx, sy, sz, cfg.table.top_height], "pen": [cfg.pen.length, cfg.pen.diameter],
               "hands": [{"urdf": drawn_urdf(h), "joints": env.robot.real_joint_names(int(envs_of[h][0])),
                          "envs": len(envs_of[h])} for h in range(nh)]})
    active, running, det, shown, offset, speed = None, False, False, 0, 0, 1.0
    finished = [deque(maxlen=2000) for _ in range(nh)]
    ends = [dict() for _ in range(nh)]
    last_end, steps, t_next, t_frame = "", 0, time.time(), 0.0
    hand = env.hand_of_env.long()

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
            elif c == "pushes":
                env.task_perturbed[:] = trained_perturbed & bool(m["value"])
                restart()
            elif c == "reset":
                restart()
            elif c == "policy":
                try:
                    active = player(m["path"])
                    conn.send({"kind": "status", "text": f"policy: {m['label']}"})
                except Exception as exc:                  # e.g. the rolling checkpoint caught mid-write
                    conn.send({"kind": "status", "text": f"**could not load** {m['label']}: {exc}"})
                restart()
        watch = int(envs_of[shown][min(offset, len(envs_of[shown]) - 1)])
        if running and active is not None:
            with torch.no_grad():
                act = active.get_normalized_action(obs["policy"], deterministic_actions=det)
            obs, _, term, trunc, extras = env.step(act)
            steps += 1
            done = term | trunc
            if bool(done.any()):
                final = extras["episode_final"]
                for i in done.nonzero().flatten().tolist():
                    h, g = int(hand[i]), float(final["successes"][i])
                    finished[h].append(g)
                    why = next((r for r in REASONS if bool(final[f"done_{r}"][i])), "?")
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
        for h in range(nh):
            g = list(finished[h])
            stats.append({"episodes": len(g), "goals": sum(g) / len(g), "one": sum(x >= 1 for x in g) / len(g),
                          "ends": dict(ends[h])} if g else {"episodes": 0})
        q = env.robot.data.joint_pos[watch][env.robot.real_slots(watch)]
        conn.send({"kind": "frame", "hand": shown, "q": q.tolist(),
                   "pen": _pose7(pos[watch], quat[watch]),
                   "goal": _pose7(env.goal_pos[watch], env.goal_quat[watch]),
                   "status": (f"env {watch} · goals **{int(env.successes[watch])}** · goal distance "
                              f"{float(env.goal_dist[watch]) * 1000:.0f} mm · "
                              f"{float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                              + (f"  \nlast episode here: {last_end}" if last_end else "")
                              + f"  \n{'running' if running and active is not None else 'paused'} · {steps} steps "
                                f"since the last change"),
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
    sys.path.insert(0, str(REPO))
    hands = hands_for(args.sources, args.hands)
    pols = policies()
    if not pols:
        raise SystemExit("no evolution policies found")
    grid = json.loads(SEED.read_text())["grid"]
    mf = tempfile.NamedTemporaryFile("w", suffix="_manifest.json", delete=False)   # template refs end in manifest.json
    json.dump({"generation": 0, "grid": grid, "hands": [{"id": h["label"], "spec": h["spec"]} for h in hands]}, mf)
    mf.close()
    plan = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"run": str(env_run()), "manifest": mf.name, "hands": hands}, plan)
    plan.close()
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    host, wport = listener.address
    proc = subprocess.Popen([sys.executable, __file__, "--worker", "--plan", plan.name, "--envs", str(args.envs),
                             "--host", host, "--wport", str(wport), "--authkey", authkey.hex()],
                            cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
    sim = {"conn": None, "ready": None, "status": "Kit booting", "frame": None, "stats": None, "t_boot": time.time()}

    def acceptor() -> None:
        sim["conn"] = listener.accept()

    threading.Thread(target=acceptor, daemon=True).start()

    labels = [h["label"] for h in hands]
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[evolution-play] {len(hands)} hands x {args.envs} envs; open http://{socket.gethostname()}:{args.port}\n",
          flush=True)
    first_src = next((s for s in ("map_elites", "baseline") if s in pols), next(iter(pols)))
    with server.gui.add_folder("policy (drives every hand)"):
        g_src = server.gui.add_dropdown("source", tuple(pols), initial_value=first_src)
        g_ck = server.gui.add_dropdown("checkpoint", tuple(pols[first_src]))
        b_load = server.gui.add_button("load this policy")
        md_pol = server.gui.add_markdown("")
    with server.gui.add_folder("hand"):
        g_hand = server.gui.add_dropdown("hand shown", tuple(labels), initial_value=labels[0])
        g_offset = server.gui.add_slider("which of its envs", 0, max(args.envs - 1, 1), 1, 0)
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_push = server.gui.add_checkbox("pushes on the lifted pen (as trained)", True)
        b_reset = server.gui.add_button("reset the envs and the statistics")
        md_boot = server.gui.add_markdown("")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")
    state = {"view": {}, "shown": None, "running": True, "policy": None}

    def send(msg: dict) -> None:
        c = sim["conn"]
        if c is not None:
            try:
                c.send(msg)
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass

    def show_msg() -> dict:
        return {"cmd": "show", "hand": labels.index(str(g_hand.value)), "offset": int(g_offset.value)}

    def load_policy() -> None:
        src, ck = str(g_src.value), str(g_ck.value)
        state["policy"] = f"{src} · {ck}"
        md_pol.content = f"loading **{src} · {ck}** ..."
        send({"cmd": "policy", "path": pols[src][ck], "label": state["policy"]})

    @g_src.on_update
    def _(_) -> None:
        g_ck.options = tuple(pols[str(g_src.value)])

    b_load.on_click(lambda _: load_policy())
    g_offset.on_update(lambda _: send(show_msg()))
    g_speed.on_update(lambda _: send({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: send({"cmd": "det", "value": bool(g_det.value)}))
    g_push.on_update(lambda _: send({"cmd": "pushes", "value": bool(g_push.value)}))
    b_reset.on_click(lambda _: send({"cmd": "reset"}))

    @g_hand.on_update
    def _(_) -> None:
        state["shown"] = None

    @b_run.on_click
    def _(_) -> None:
        state["running"] = not state["running"]
        send({"cmd": "run", "value": state["running"]})

    def build_scene(m: dict) -> None:
        sx, sy, sz, top = m["table"]
        server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77), position=(0.0, 0.0, top - sz / 2))
        server.scene.add_frame("/base", position=(0.0, 0.0, top), show_axes=False)
        state["view"]["pen"] = _pen(server, "/pen", *m["pen"])
        state["view"]["goal"] = _pen(server, "/goal", *m["pen"], opacity=0.35)

    def draw_hand(lab: str) -> None:
        k = labels.index(lab)
        send(show_msg())
        sim["frame"] = None
        v, info = state["view"], sim["ready"]["hands"][k]
        if "urdf" in v:
            v["urdf"].remove()
        v["urdf"] = ViserUrdf(server, pathlib.Path(info["urdf"]), root_node_name="/base/robot")
        v["joints"] = info["joints"]
        g_offset.max = max(info["envs"] - 1, 1)
        state["shown"] = lab

    def stats_md(stats: list) -> str:
        rows = [f"**{state['policy'] or 'no policy loaded'}**: every hand, episodes finished since the last change",
                "", "| hand | episodes | goals / episode | ≥ 1 goal | how they ended |", "|---|---|---|---|---|"]
        order = sorted(range(len(labels)), key=lambda h: -(stats[h].get("goals", -1)))
        for h in order:
            st = stats[h]
            if st.get("episodes"):
                e = ", ".join(f"{a} {b}" for a, b in sorted(st["ends"].items(), key=lambda kv: -kv[1]))
                rows.append(f"| {labels[h]} | {st['episodes']} | **{st['goals']:.1f}** | {100 * st['one']:.0f}% | {e} |")
            else:
                rows.append(f"| {labels[h]} | 0 | – | – | – |")
        return "\n".join(rows)

    t_scan = time.time()
    try:
        while True:
            if time.time() - t_scan > 30.0:               # new iterations / checkpoints join the dropdowns
                t_scan = time.time()
                fresh = policies()
                if fresh != pols:
                    pols.clear()
                    pols.update(fresh)
                    cur = str(g_ck.value)
                    g_ck.options = tuple(pols[str(g_src.value)])
                    if cur in g_ck.options:
                        g_ck.value = cur
            c = sim["conn"]
            if c is not None:
                batch = []
                try:
                    while c.poll():
                        batch.append(c.recv())
                except (EOFError, ConnectionResetError, BrokenPipeError):
                    sim["status"], sim["conn"] = "**the sim closed the connection**", None
                for m in batch:
                    k = m["kind"]
                    if k == "ready":
                        sim["ready"] = m
                        print(f"[evolution-play] sim ready after {time.time() - sim['t_boot']:.0f} s", flush=True)
                        build_scene(m)
                        for msg in ({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]},
                                    {"cmd": "det", "value": bool(g_det.value)},
                                    {"cmd": "run", "value": state["running"]}):
                            send(msg)
                        load_policy()
                    elif k == "frame":
                        sim["frame"], sim["stats"] = m, m["stats"]
                    elif k == "status":
                        sim["status"] = m["text"]
                        print(f"[evolution-play] {m['text']}", flush=True)
                        if args.selftest and m["text"].startswith("policy:"):
                            send({"cmd": "speed", "value": 0.0})
                            state["t_test"] = time.time()
                        if m["text"].startswith(("policy:", "**could not")):
                            md_pol.content = m["text"]
                    elif k == "error":
                        sim["status"] = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                        print(m["text"], flush=True)
            md_boot.content = ("sim: ready" if sim["ready"] else
                               f"sim: {sim['status']} ({time.time() - sim['t_boot']:.0f} s)")
            if sim["ready"] is None:
                md.content = "waiting for the sim to boot (a minute or two)"
                time.sleep(0.05)
                continue
            lab = str(g_hand.value)
            if state["shown"] != lab:
                draw_hand(lab)
            frame, v = sim["frame"], state["view"]
            if frame is None or frame["hand"] != labels.index(lab):
                time.sleep(0.01)
                continue
            sim["frame"] = None
            v["urdf"].update_cfg(dict(zip(v["joints"], frame["q"])))
            for k in ("pen", "goal"):
                v[k].position, v[k].wxyz = tuple(frame[k][:3]), tuple(frame[k][3:])
            md.content = f"**{lab}**  \n" + frame["status"]
            md_stats.content = stats_md(frame["stats"])
            if args.selftest and state.get("t_test") and time.time() - state["t_test"] > args.selftest:
                print(frame["status"], "\n" + md_stats.content, flush=True)
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        send({"cmd": "quit"})
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--envs", type=int, default=4, help="envs per hand")
    ap.add_argument("--sources", nargs="+", default=["map_elites", "baseline", "seed"],
                    help="hand sets: seed, map_elites (its current map), baseline (its latest population)")
    ap.add_argument("--hands", nargs="*", default=[], help="more hands: <run>:<hand id> or a .spec.json path")
    ap.add_argument("--selftest", type=float, default=0.0,
                    help="boot, load the default policy, run as fast as possible this many seconds, print the stats, exit")
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
