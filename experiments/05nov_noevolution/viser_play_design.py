"""Play a no-evolution iteration's policy on one chosen design of the population, in viser: one env, one Isaac Sim.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/viser_play_design.py [--port 8099] [--envs 1]
        [--run debug_outputs/evolution/noevolution_4096]

    then open http://<this node>:<port> (or ssh -L <port>:<node>:<port> <login host> -> http://localhost:<port>)

Pick an iteration (its gen_NNNN/policy.pth), a niche, and a design (sorted by that iteration's own 4-env score; the
100-env score of study 1 shown where there is one), then "load this design": the current sim is shut down and a fresh
one (a Kit child process, this file with --worker) builds a scene of only that design, --envs envs (1 by default), with
the env exactly as that iteration trained (its saved .hydra config). Switching the policy's iteration on the same
design swaps the weights in place; switching the design starts a new sim (~1 min).

Actions are sampled from the SAPG leader (exploration coefficient 0), as the evals do, unless "deterministic". The page
draws the hand on its wrist with every link coloured by its joint's kind (flexion blue, abduction red, roll green,
roll tips striped, palm grey), the box pen (red tip) and the goal (translucent), and counts the episodes finished
since the last change: goals per episode and how they ended.
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
STUDIES = REPO / "debug_outputs/noevolution/studies"
SPEEDS = {"real time": 1.0, "0.25x": 0.25, "2x": 2.0, "as fast as possible": 0.0}
REASONS = ("max_successes", "timeout", "dropped", "fell", "away", "off_course")
KIND = {(0, 0): "flex", (90, 0): "abd", (0, 90): "roll"}
FRAME_PERIOD_S = 1.0 / 30.0


# --- what there is to load ----------------------------------------------------------------------------------------
def iterations(run: pathlib.Path) -> dict[str, dict]:
    """"iteration N" -> {gen dir, policy, training run dir}, for every iteration with a kept policy, newest first."""
    out = {}
    for g in sorted(run.glob("gen_[0-9][0-9][0-9][0-9]"), reverse=True):
        logs = sorted((REPO / "debug_outputs/train_logs/evolution" / run.name / g.name).glob("*/.hydra/config.yaml"))
        if (g / "policy.pth").exists() and logs:
            out[f"iteration {int(g.name[4:])}"] = {"gen": g, "policy": g / "policy.pth", "train_run": logs[0].parent.parent}
    return out


def population(gen: pathlib.Path) -> tuple[dict[str, list[dict]], dict]:
    """niche -> its hands (id, spec, 4-env score, 100-env score if study 1 scored it), best first; and the grid."""
    man = json.loads((gen / "manifest.json").read_text())
    fit = json.loads((gen / "fitness.json").read_text())["hands"] if (gen / "fitness.json").exists() else {}
    it = int(gen.name[4:])
    noise = STUDIES / f"it{it}_noise" / "scores.json"
    s100 = {r["id"]: r["env100"] for r in json.loads(noise.read_text())} if noise.exists() else {}
    by: dict[str, list[dict]] = {}
    for h in man["hands"]:
        f = fit.get(h["spec"], {})
        by.setdefault(h.get("niche", "all"), []).append(
            {"id": h["id"], "spec": h["spec"], "eval4": f.get("goals_per_episode"), "env100": s100.get(h["id"])})
    for hs in by.values():
        hs.sort(key=lambda h: (-(h["env100"] if h["env100"] is not None else -1), -(h["eval4"] or 0.0), h["id"]))
    return by, man["grid"]


def label(h: dict) -> str:
    s = f"{h['id'].rsplit('_', 1)[-1]} · 4-env {h['eval4']:.2f}" if h["eval4"] is not None else h["id"]
    return s + (f" · 100-env {h['env100']:.2f}" if h["env100"] is not None else "")


def describe(spec_rel: str) -> str:
    if not (spec_rel.endswith(".spec.json") and "design" in json.loads((REPO / spec_rel).read_text())):
        return f"registered robot `{spec_rel}` (its own URDF: no grammar design)"
    spec = json.loads((REPO / spec_rel).read_text())
    d = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
    lines = [f"{len(d['fingers'])} fingers, {sum(len(f['joints']) for f in d['fingers'])} motors"]
    for i, f in enumerate(d["fingers"]):
        lines.append(f"- finger {i}: base ({f['y']}, {f['z']}) facing {f['facing']} tilt {f['tilt']}: "
                     + ", ".join(f"{KIND.get(tuple(j['axis']), j['axis'])} {j['length']}" for j in f["joints"]))
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
    if not (plan["spec"].endswith(".spec.json") and "design" in json.loads((REPO / plan["spec"]).read_text())):
        cfg.reset.table_safe = False
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
    run = (REPO / args.run).resolve()
    its = iterations(run)
    if not its:
        raise SystemExit(f"no iteration with a kept policy under {run}")
    first_it = f"iteration {args.iteration}" if f"iteration {args.iteration}" in its else next(iter(its))
    pop, grid = population(its[first_it]["gen"])
    niches = list(pop)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    print(f"\n[play-design] {run.name}: {len(its)} iterations; open http://{socket.gethostname()}:{args.port}\n",
          flush=True)
    with server.gui.add_folder("design (from the population)"):
        g_pop = server.gui.add_dropdown("population of", tuple(its), initial_value=first_it)
        g_niche = server.gui.add_dropdown("niche", tuple(niches), initial_value=niches[0])
        g_hand = server.gui.add_dropdown("design", tuple(label(h) for h in pop[niches[0]]))
        b_load = server.gui.add_button("load this design (new sim)")
        md_design = server.gui.add_markdown("")
    with server.gui.add_folder("policy"):
        g_pol = server.gui.add_dropdown("policy after", tuple(its), initial_value=first_it)
        b_pol = server.gui.add_button("use this policy")
    with server.gui.add_folder("play"):
        b_run = server.gui.add_button("pause / run")
        g_speed = server.gui.add_dropdown("speed", tuple(SPEEDS), initial_value="real time")
        g_det = server.gui.add_checkbox("deterministic (mean action)", False)
        g_push = server.gui.add_checkbox("pushes on the lifted pen (as trained)", True)
        g_env = server.gui.add_slider("which env", 0, max(args.envs - 1, 1), 1, 0)
        b_reset = server.gui.add_button("reset the env(s) and the statistics")
        md_boot = server.gui.add_markdown("")
    server.gui.add_markdown("flexion **blue**, abduction **red**, roll **green** (roll tips striped), palm grey")
    md = server.gui.add_markdown("")
    md_stats = server.gui.add_markdown("")
    st = {"sim": None, "view": {}, "running": True, "design": None, "pop": pop, "policy_label": None,
          "drawn": False}

    def chosen() -> dict:
        hs = st["pop"][str(g_niche.value)]
        labs = [label(h) for h in hs]
        return hs[labs.index(str(g_hand.value))] if str(g_hand.value) in labs else hs[0]

    def send(msg: dict) -> None:
        if st["sim"] is not None:
            st["sim"].send(msg)

    def use_policy() -> None:
        it = str(g_pol.value)
        st["policy_label"] = f"the policy after {it}"
        send({"cmd": "policy", "path": str(its[it]["policy"]), "label": st["policy_label"]})

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
        json.dump({"generation": 0, "grid": grid, "hands": [{"id": h["id"], "spec": h["spec"]}]}, mf)
        mf.close()
        it = str(g_pol.value)
        st["design"] = h
        md_design.content = f"**{h['id']}** ({label(h)})  \n" + describe(h["spec"])
        st["sim"] = Sim(args.envs, {"train_run": str(its[it]["train_run"]), "manifest": mf.name, "spec": h["spec"]})
        print(f"[play-design] new sim for {h['id']}", flush=True)

    @g_pop.on_update
    def _(_) -> None:
        st["pop"], _grid = population(its[str(g_pop.value)]["gen"])
        g_niche.options = tuple(st["pop"])
        g_hand.options = tuple(label(h) for h in st["pop"][str(g_niche.value)])

    @g_niche.on_update
    def _(_) -> None:
        g_hand.options = tuple(label(h) for h in st["pop"][str(g_niche.value)])

    @g_hand.on_update
    def _(_) -> None:
        h = chosen()
        cur = st["design"]
        md_design.content = (f"**{h['id']}** ({label(h)})  \n" + describe(h["spec"])
                             + ("" if cur and cur["id"] == h["id"] else "  \n_press **load this design** to play it_"))

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

    g_hand.value = g_hand.options[0]
    load_design()
    try:
        while True:
            sim = st["sim"]
            for m in sim.messages():
                k = m["kind"]
                if k == "ready":
                    sim.ready = m
                    print(f"[play-design] sim ready after {time.time() - sim.t_boot:.0f} s", flush=True)
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
                    print(f"[play-design] {m['text']}", flush=True)
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
    ap.add_argument("--envs", type=int, default=1, help="envs of the chosen design")
    ap.add_argument("--run", default="debug_outputs/evolution/noevolution_4096")
    ap.add_argument("--iteration", type=int, default=1, help="the iteration shown first")
    ap.add_argument("--selftest", type=float, default=0.0,
                    help="boot the first design, play this many seconds after launch, print the status, exit")
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
