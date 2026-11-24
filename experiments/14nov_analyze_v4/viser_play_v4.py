"""Play a v4 policy on any design or commercial hand, in viser (14 Nov 2026): one env (or --envs), one Isaac Sim.

    .venv_isaacsim/bin/python experiments/14nov_analyze_v4/viser_play_v4.py [--port 8099] [--envs 1]

    then open http://<this node>:<port> (or ssh -L <port>:<node>:<port> <login host> -> http://localhost:<port>)

Hand: a source -- a v4 run's archive (MAP-Elites k5_none, Greedy k5_global, House of Dextra hod: the designs trained in
a chosen generation, by cell, best training-log goals / episode first) or the commercial hands (the unchanged vendor
URDFs, <hand>_left_uniform_handonly, no adapters) -- then "load this hand": the current sim is shut down and a fresh one
(a Kit child process, this file with --worker) builds a scene of only that hand, with the env exactly as the chosen
POLICY's generation trained (its saved .hydra config: v4's fixed -30..90 deg joint range, no limits in the tokens).
Policy: any run and generation with a kept policy (MAP-Elites' newest by default); switching it swaps the weights in
place, keeping the hand.

Actions are sampled from the SAPG leader (exploration coefficient 0), as the evals do, unless "deterministic". Links
are coloured by joint kind for grammar hands (flexion blue, abduction red, palm grey), vendor colours otherwise; the box
pen (red tip) and the goal (translucent); the page counts the episodes finished since the last change: goals per
episode and how they ended.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
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
CAMPAIGN = REPO / "debug_outputs/coevolve/v4"
RUNS = {"MAP-Elites": "k5_none", "Greedy": "k5_global", "House of Dextra": "hod"}
COMMERCIAL = "Commercial (vendor URDFs)"
VENDORS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
REASONS = ("max_successes", "timeout", "dropped", "fell", "away", "off_course")
FRAME_PERIOD_S = 1.0 / 30.0
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}


# --- what there is to load ----------------------------------------------------------------------------------------
def policies(run: str) -> dict[str, dict]:
    """"gen N" -> {policy, training run dir}, every generation of `run` with a kept policy, newest first."""
    rd = CAMPAIGN / run
    tr = json.loads((rd / "state.json").read_text()).get("train_runs", {})
    out = {}
    for g in sorted(rd.glob("gen_[0-9][0-9][0-9][0-9]"), reverse=True):
        k = str(int(g.name[4:]))
        if (g / "policy.pth").exists() and k in tr:
            out[f"gen {k}"] = {"policy": g / "policy.pth", "train_run": REPO / tr[k]}
    return out


def generations(run: str) -> list[str]:
    """"gen N" for every generation of `run` with training labels, newest first."""
    return [f"gen {int(p.parent.name[4:])}" for p in sorted((CAMPAIGN / run).glob("gen_*/labels.json"), reverse=True)]


_REG: dict[str, dict] = {}


def population(source: str, gen: str) -> dict[str, list[dict]]:
    """cell -> its trained designs {id, spec, label, episodes}, best first; or the commercial hands."""
    if source == COMMERCIAL:
        return {"commercial hands": [{"id": h, "spec": f"{h}_left_uniform_handonly", "label": None, "episodes": None}
                                     for h in VENDORS]}
    run = RUNS[source]
    if run not in _REG:
        _REG[run] = json.loads((CAMPAIGN / run / "designs.json").read_text())
    reg = _REG[run]
    gd = CAMPAIGN / run / f"gen_{int(gen.split()[1]):04d}"
    lab = json.loads((gd / "labels.json").read_text())
    by: dict[str, list[dict]] = {}
    for i, (v, n, *_) in lab.items():
        by.setdefault(reg[i]["niche"], []).append({"id": i, "spec": reg[i]["spec"], "label": v, "episodes": n})
    for hs in by.values():
        hs.sort(key=lambda h: -h["label"])
    return dict(sorted(by.items(), key=lambda kv: -kv[1][0]["label"]))    # cells by their best design


def label(h: dict) -> str:
    return h["id"] if h["label"] is None else f"{h['id']} · {h['label']:.1f} goals/ep ({h['episodes']} ep)"


def describe(spec_rel: str) -> str:
    if not spec_rel.endswith(".spec.json"):
        return f"vendor robot `{spec_rel}` (its own URDF, unchanged; no adapters)"
    sys.path.insert(0, str(REPO))
    from hand_sampler.grammar import v3
    v3.configure("v4")
    spec = json.loads((REPO / spec_rel).read_text())
    hand = v3.hand_from_dict(spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"]))
    lines = [f"{len(hand.fingers)} fingers, {sum(len(f.joints) for f in hand.fingers)} motors, "
             f"facing gap {v3.design_gap_deg(hand):.0f} deg, max reach {v3.design_reach_mm(hand):.0f} mm"]
    for i, f in enumerate(hand.fingers):
        lines.append(f"- finger {i}: base ({f.y}, {f.z}) facing {f.facing} tilt {f.tilt} roll {v3.roll_of(f)}: "
                     + ", ".join(f"{v3.kind_of(f, j)} {j.length}" for j in f.joints))
    return "  \n".join(lines)


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
    from hand_sampler.grammar.visual_colours import colour_urdf_text
    from isaacsimenvs.inhand_olympics.env import InHandOlympicsEnv
    from isaacsimenvs.inhand_olympics.env_cfg import InHandOlympicsEnvCfg
    from isaacsimenvs.inhand_olympics.pose_viewer import _cylinders_as_capsules

    plan = json.loads(pathlib.Path(args.plan).read_text())    # {"train_run", "manifest", "spec", "policy", "label"}
    full = OmegaConf.to_container(OmegaConf.load(pathlib.Path(plan["train_run"]) / ".hydra/config.yaml"),
                                  resolve=True)
    env_d = full["env"]
    for k in ("observation_space", "state_space", "action_space"):
        env_d.pop(k, None)
    # a registered robot (vendor URDF, mesh colliders) cannot go on the template: same 6 x 5 finger grid, own articulation
    grammar = plan["spec"].endswith(".spec.json") and "design" in json.loads((REPO / plan["spec"]).read_text())
    ref = f"multi:template:{plan['manifest']}" if grammar else f"multi:finger:{plan['spec']}"
    cfg = InHandOlympicsEnvCfg()
    cfg.from_dict(env_d)
    cfg.hand.robot_spec = ref
    cfg.scene.num_envs = args.envs
    cfg.reset.stagger_initial_episodes = False
    cfg.log.group_stats = False
    cfg.log.design_stats = False
    cfg.seed = 0
    shown = plan["spec"].split("/")[-2] if plan["spec"].endswith(".spec.json") else plan["spec"]
    conn.send({"kind": "status", "text": f"building the scene: {shown} x {args.envs} env(s)"})
    env = InHandOlympicsEnv(cfg)
    obs, _ = env.reset()
    n_act = env.action_space.shape[-1]
    acfg = full["agent"]
    acfg["params"]["config"]["multi_gpu"] = False
    acfg["params"]["network"]["robot_spec"] = ref
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"train": acfg}, f)
    f.close()
    players: dict[str, object] = {}

    def player(path: str):
        if path not in players:
            conn.send({"kind": "status", "text": f"loading {pathlib.Path(path).parent.name}/{pathlib.Path(path).name}"})
            players[path] = RlPlayer(obs["policy"].shape[-1], n_act, f.name, path, device=env.device,
                                     sapg_expl_coef=0.0, num_envs=env.num_envs)
        return players[path]

    src = pathlib.Path(env.hand_urdf_paths[0])
    if plan["spec"].endswith(".spec.json") and "design" in json.loads((REPO / plan["spec"]).read_text()):
        spec = json.loads((REPO / plan["spec"]).read_text())
        design = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
        text = colour_urdf_text(src.read_text(), design)
    else:                                                # registered robot (vendor URDF): its own colours
        text = src.read_text()
    if getattr(env.hand_set.specs[0], "replace_cylinders_with_capsules", False):
        text = _cylinders_as_capsules(text)
    drawn = src.with_name("hand_on_wrist_viser.urdf")             # mesh paths in it are absolute
    drawn.write_text(text)
    trained_perturbed = env.task_perturbed.clone()
    sx, sy, sz = cfg.table.size
    conn.send({"kind": "ready", "table": [sx, sy, sz, cfg.table.top_height], "pen": [cfg.pen.length, cfg.pen.diameter],
               "urdf": str(drawn), "joints": env.robot.real_joint_names(0)})
    active, running, det, offset, speed = None, True, False, 0, 1.0
    finished: deque = deque(maxlen=5000)
    ends: dict[str, int] = {}
    last_end, steps, t_next, t_frame = "", 0, time.time(), 0.0
    policy_label = ""

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
            if c == "run":
                running = bool(m["value"])
            elif c == "det":
                det = bool(m["value"])
                restart()
            elif c == "show":
                offset = min(int(m["offset"]), env.num_envs - 1)
            elif c == "speed":
                speed = float(m["value"])
            elif c == "pushes":
                env.task_perturbed[:] = trained_perturbed & bool(m["value"])
                restart()
            elif c == "reset":
                restart()
            elif c == "policy":
                try:
                    active, policy_label = player(m["path"]), m["label"]
                    conn.send({"kind": "status", "text": f"policy: {m['label']}"})
                except Exception as exc:
                    conn.send({"kind": "status", "text": f"**could not load** {m['label']}: {exc}"})
                restart()
        watch = offset
        if running and active is not None:
            with torch.no_grad():
                act = active.get_normalized_action(obs["policy"], deterministic_actions=det)
            obs, _, term, trunc, extras = env.step(act)
            steps += 1
            done = term | trunc
            if bool(done.any()):
                final = extras["episode_final"]
                for i in done.nonzero().flatten().tolist():
                    g = float(final["successes"][i])
                    finished.append(g)
                    why = next((r for r in REASONS if bool(final[f"done_{r}"][i])), "?")
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
        stats = {"episodes": len(g), "goals": sum(g) / len(g) if g else None,
                 "one": sum(x >= 1 for x in g) / len(g) if g else None, "ends": dict(ends)}
        q = env.robot.data.joint_pos[watch][env.robot.real_slots(watch)]
        conn.send({"kind": "frame", "q": q.tolist(), "pen": _pose7(pos[watch], quat[watch]),
                   "goal": _pose7(env.goal_pos[watch], env.goal_quat[watch]), "stats": stats,
                   "status": (f"env {watch} · goals **{int(env.successes[watch])}** · goal distance "
                              f"{float(env.goal_dist[watch]) * 1000:.0f} mm · "
                              f"{float(env.episode_length_buf[watch]) * env.step_dt:.1f} s into this goal"
                              + (f"  \nlast episode: {last_end}" if last_end else "")
                              + f"  \n{'running' if running and active is not None else 'paused'} · {steps} steps "
                                f"under {policy_label or 'no policy'}")})


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


class Sim:
    """One Kit child holding one design; replaced whenever another design is loaded."""

    def __init__(self, envs: int, plan: dict):
        self.ready, self.frame, self.status, self.t_boot, self.conn = None, None, "Kit booting", time.time(), None
        self.plan_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(plan, self.plan_file)
        self.plan_file.close()
        authkey = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=authkey)
        host, wport = self.listener.address
        self.proc = subprocess.Popen([sys.executable, __file__, "--worker", "--plan", self.plan_file.name,
                                      "--envs", str(envs), "--host", host, "--wport", str(wport),
                                      "--authkey", authkey.hex()],
                                     cwd=REPO, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        try:
            self.conn = self.listener.accept()
        except OSError:
            pass

    def send(self, msg: dict) -> None:
        if self.conn is not None:
            try:
                self.conn.send(msg)
            except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
                pass

    def messages(self) -> list[dict]:
        out = []
        if self.conn is None:
            return out
        try:
            while self.conn.poll():
                out.append(self.conn.recv())
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
            self.status, self.conn = "**the sim closed the connection**", None
        return out

    def close(self) -> None:
        self.send({"cmd": "quit"})
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        try:
            self.listener.close()
        except OSError:
            pass


def main(args) -> None:
    import signal

    import viser
    from viser.extras import ViserUrdf

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    sys.path.insert(0, str(REPO))
    pols = {name: policies(run) for name, run in RUNS.items()}
    src0 = "MAP-Elites"
    gens0 = generations(RUNS[src0])
    pop = population(src0, gens0[0])
    cells = list(pop)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[play-v4] open http://{socket.gethostname()}:{args.port}\n", flush=True)
    with server.gui.add_folder("hand"):
        g_src = server.gui.add_dropdown("source", (*RUNS, COMMERCIAL), initial_value=src0)
        g_gen = server.gui.add_dropdown("designs trained in", tuple(gens0), initial_value=gens0[0])
        g_cell = server.gui.add_dropdown("cell (best first)", tuple(cells), initial_value=cells[0])
        g_hand = server.gui.add_dropdown("design", tuple(label(h) for h in pop[cells[0]]))
        b_load = server.gui.add_button("load this hand (new sim)")
        md_design = server.gui.add_markdown("")
    with server.gui.add_folder("policy"):
        g_prun = server.gui.add_dropdown("policy of", tuple(RUNS), initial_value="MAP-Elites")
        g_pgen = server.gui.add_dropdown("after", tuple(pols["MAP-Elites"]))
        b_pol = server.gui.add_button("use this policy")
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_push = server.gui.add_checkbox("pushes on the lifted pen (as trained)", True)
        g_env = server.gui.add_slider("which env", 0, max(args.envs - 1, 1), 1, 0)
        b_reset = server.gui.add_button("reset the env(s) and the statistics")
        md_boot = server.gui.add_markdown("")
    server.gui.add_markdown("grammar hands: flexion **blue**, abduction **red**, palm grey; vendor hands: own colours")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")
    st = {"sim": None, "view": {}, "running": True, "design": None, "pop": pop, "policy_label": None,
          "drawn": False}

    def chosen() -> dict:
        hs = st["pop"][str(g_cell.value)]
        labs = [label(h) for h in hs]
        return hs[labs.index(str(g_hand.value))] if str(g_hand.value) in labs else hs[0]

    def chosen_policy() -> tuple[str, dict]:
        name, gen = str(g_prun.value), str(g_pgen.value)
        return f"{name} {gen}", pols[name][gen]

    def send(msg: dict) -> None:
        if st["sim"] is not None:
            st["sim"].send(msg)

    def use_policy() -> None:
        lab, p = chosen_policy()
        st["policy_label"] = f"the {lab} policy"
        send({"cmd": "policy", "path": str(p["policy"]), "label": st["policy_label"]})

    def load_design() -> None:
        h = chosen()
        if st["sim"] is not None:
            md_boot.content = "closing the previous sim ..."
            st["sim"].close()
        for k in ("urdf",):
            if k in st["view"]:
                st["view"].pop(k).remove()
        st["drawn"] = False
        mf = tempfile.NamedTemporaryFile("w", suffix="_manifest.json", delete=False)  # template refs: *manifest.json
        json.dump({"generation": 0, "grid": GRID, "hands": [{"id": h["id"], "spec": h["spec"]}]}, mf)
        mf.close()
        st["design"] = h
        md_design.content = f"**{label(h)}**  \n" + describe(h["spec"])
        _, p = chosen_policy()
        st["sim"] = Sim(args.envs, {"train_run": str(p["train_run"]), "manifest": mf.name, "spec": h["spec"]})
        print(f"[play-v4] new sim for {h['id']}", flush=True)

    def refresh_cells() -> None:
        src = str(g_src.value)
        st["pop"] = population(src, str(g_gen.value))
        g_cell.options = tuple(st["pop"])
        g_hand.options = tuple(label(h) for h in st["pop"][str(g_cell.value)])

    @g_src.on_update
    def _(_) -> None:
        src = str(g_src.value)
        gens = ["(vendor URDFs)"] if src == COMMERCIAL else generations(RUNS[src])
        g_gen.options = tuple(gens)
        g_gen.value = gens[0]
        refresh_cells()

    @g_gen.on_update
    def _(_) -> None:
        if str(g_gen.value) != "(vendor URDFs)" or str(g_src.value) == COMMERCIAL:
            refresh_cells()

    @g_cell.on_update
    def _(_) -> None:
        g_hand.options = tuple(label(h) for h in st["pop"][str(g_cell.value)])

    @g_hand.on_update
    def _(_) -> None:
        h = chosen()
        cur = st["design"]
        md_design.content = (f"**{label(h)}**  \n" + describe(h["spec"])
                             + ("" if cur and cur["id"] == h["id"] else "  \n_press **load this hand** to play it_"))

    @g_prun.on_update
    def _(_) -> None:
        g_pgen.options = tuple(pols[str(g_prun.value)])

    b_load.on_click(lambda _: load_design())
    b_pol.on_click(lambda _: use_policy())
    g_speed.on_update(lambda _: send({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]}))
    g_det.on_update(lambda _: send({"cmd": "det", "value": bool(g_det.value)}))
    g_push.on_update(lambda _: send({"cmd": "pushes", "value": bool(g_push.value)}))
    g_env.on_update(lambda _: send({"cmd": "show", "offset": int(g_env.value)}))
    b_reset.on_click(lambda _: send({"cmd": "reset"}))

    @b_run.on_click
    def _(_) -> None:
        st["running"] = not st["running"]
        send({"cmd": "run", "value": st["running"]})

    def build_static(m: dict) -> None:
        if "pen" in st["view"]:
            return
        sx, sy, sz, top = m["table"]
        server.scene.add_box("/table", dimensions=(sx, sy, sz), color=(140, 107, 77), position=(0.0, 0.0, top - sz / 2))
        server.scene.add_frame("/base", position=(0.0, 0.0, top), show_axes=False)
        st["view"]["pen"] = _pen(server, "/pen", *m["pen"])
        st["view"]["goal"] = _pen(server, "/goal", *m["pen"], opacity=0.35)

    def stats_md(s: dict) -> str:
        if not s["episodes"]:
            return f"**{st['policy_label'] or 'no policy'}**: no episode finished yet"
        e = ", ".join(f"{a} {b}" for a, b in sorted(s["ends"].items(), key=lambda kv: -kv[1]))
        return (f"**{st['policy_label']}**, since the last change: {s['episodes']} episodes, "
                f"**{s['goals']:.2f} goals / episode**, {100 * s['one']:.0f}% with at least one goal  \nended: {e}")

    if args.hand:                                       # start on a given hand (a vendor name or a design id)
        if args.hand in VENDORS:
            g_src.value = COMMERCIAL
        else:
            for c, hs in st["pop"].items():
                if any(h["id"] == args.hand for h in hs):
                    g_cell.value = c
                    g_hand.value = next(label(h) for h in hs if h["id"] == args.hand)
                    break
        if args.hand in VENDORS:
            g_hand.value = args.hand
    else:
        g_hand.value = g_hand.options[0]
    load_design()
    try:
        while True:
            sim = st["sim"]
            for m in sim.messages():
                k = m["kind"]
                if k == "ready":
                    sim.ready = m
                    print(f"[play-v4] sim ready after {time.time() - sim.t_boot:.0f} s", flush=True)
                    build_static(m)
                    v = st["view"]
                    v["urdf"] = ViserUrdf(server, pathlib.Path(m["urdf"]), root_node_name="/base/robot")
                    v["joints"] = m["joints"]
                    st["drawn"] = True
                    for msg in ({"cmd": "speed", "value": SPEEDS[str(g_speed.value)]},
                                {"cmd": "det", "value": bool(g_det.value)}, {"cmd": "pushes", "value": bool(g_push.value)},
                                {"cmd": "show", "offset": int(g_env.value)}, {"cmd": "run", "value": st["running"]}):
                        sim.send(msg)
                    use_policy()
                elif k == "frame":
                    sim.frame = m
                elif k == "status":
                    sim.status = m["text"]
                    print(f"[play-v4] {m['text']}", flush=True)
                elif k == "error":
                    sim.status = "**sim failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                    print(m["text"], flush=True)
            md_boot.content = (f"sim: ready ({st['design']['id']})" if sim.ready else
                               f"sim: {sim.status} ({time.time() - sim.t_boot:.0f} s)")
            if sim.ready is None or not st["drawn"]:
                md.content = "waiting for the sim (about a minute)" if sim.ready is None else ""
                time.sleep(0.05)
                continue
            frame, v = sim.frame, st["view"]
            if frame is None:
                time.sleep(0.01)
                continue
            sim.frame = None
            v["urdf"].update_cfg(dict(zip(v["joints"], frame["q"])))
            for k in ("pen", "goal"):
                v[k].position, v[k].wxyz = tuple(frame[k][:3]), tuple(frame[k][3:])
            md.content = f"**{st['design']['id']}**  \n" + frame["status"]
            md_stats.content = stats_md(frame["stats"])
            if args.selftest and time.time() - sim.t_boot > args.selftest:
                print(frame["status"], "\n" + md_stats.content, flush=True)
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if st["sim"] is not None:
            st["sim"].close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--envs", type=int, default=1, help="envs of the chosen hand")
    ap.add_argument("--hand", default="", help="start on this hand: a vendor name (wuji2, leap, ...) or a MAP-Elites "
                                                "design id in the newest generation (default: the best design)")
    ap.add_argument("--selftest", type=float, default=0.0,
                    help="boot the first hand, play this many seconds after launch, print the status, exit")
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
