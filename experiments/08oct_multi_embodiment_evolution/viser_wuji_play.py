"""Live viewer: any Wuji v2 hand -- the 52 finger-drop hands and the scaled-link variants -- driven by any Wuji
policy, joint transformer or MLP, in viser.

Pick the hand with a left/right dropdown, a checkbox per finger (two or more fingers: the 52 of
multi:wuji2_lr_2plus) and a "links" dropdown (as built, or a make_scaled_links.py variant of the full left hand:
len50, len75, wid120, ...), pick a policy, press "load". The policies are the 6 Oct 52-hand run (31826) and every
run in 08oct_multi_embodiment_evolution and 08oct_repro_wuji_full, each at its newest checkpoint WHEN LOADED, so a
training run can be re-loaded as it improves. The status line says whether the hand was in the policy's training set
or is held out (e.g. a four-finger hand under the 2-3-finger policy).

MLP policies ([MLP] in the dropdown) read a flat observation whose width is their training hand's, so they drive
only that hand: picking one sets the hand selectors to it, and loading it on any other hand is refused. The joint
transformer drives any of them (one token per joint, shared weights).

Each panel's simulation is a child worker (experiments/01oct_unified_rl/viser_compare.py --worker: Kit and viser
cannot share an interpreter), under the env config of the chosen policy's own run, except that every policy runs
with action delay on (0-3 steps) and the hand-target moving average 0.1 -- today's training defaults -- so
policies are compared under the same actuation (31826 trained without delay; --as-trained runs each as its run
trained). Loading restarts the worker (about a minute). Every hand has Wuji's palm, so palm_extents is the
policy's own; no override.

Per panel, live: action delay on/off, hand friction (every hand link, fingertips included), object friction, and push force scale --
the env's own pushes (randn(3) x object mass x scale, as training DR's force_scale), one every 0.3 s, 0 = off.
Object mass (grams; set through object_density_scale = grams / the trained mass) is applied on load: PhysX masses cannot be changed at
runtime in this build (env_cfg.py).

    .venv_isaacsim/bin/python experiments/08oct_multi_embodiment_evolution/viser_wuji_play.py --port 8085 [--panels 2]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import pathlib
import queue
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from multiprocessing.connection import Listener

REPO = pathlib.Path(__file__).resolve().parents[2]
WORKER = REPO / "experiments/01oct_unified_rl/viser_compare.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(WORKER.parent))
LOGS = REPO / "debug_outputs/train_logs"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# The same actuation for every policy (worker --set, applied to the run's saved config).
CONSISTENT = ("env.domain_randomization.use_action_delay=True", "env.domain_randomization.action_delay_max=3",
              "env.action.hand_moving_average=0.1")
DELAY = ("on (0-3 steps)", "off")       # live; default on, as every policy runs (CONSISTENT)
RUN_DIRS = ("08oct_multi_embodiment_evolution", "08oct_repro_wuji_full")
MAX_GOALS = 10               # goals after which an episode ends (training used 50); --max-goals
PUSH_PERIOD = 18             # control steps between pushes: 0.3 s
CONTROL_DT = 1.0 / 60.0      # every Wuji run: sim dt 1/120 s x decimation 2 -- for the measured-rate readout


def _multi_hand():
    """robots/multi_hand.py without its package __init__ (which needs Kit's pxr)."""
    path = REPO / "isaacsimenvs/pose_reaching_6d/scene_utils/robots/multi_hand.py"
    spec = importlib.util.spec_from_file_location("_multi_hand", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_multi_hand"] = mod
    spec.loader.exec_module(mod)
    return mod


MH = _multi_hand()
WUJI = REPO / "assets/urdf/unified_dynamics_commercial_hands/wuji2"


def scaled_tags() -> tuple[str, ...]:
    """make_scaled_links.py variants on disk (their spec JSON carries a scaled_links entry)."""
    import json
    tags = []
    for p in sorted(WUJI.glob("wuji2_left_*.spec.json")):
        if "scaled_links" in json.loads(p.read_text()):
            tags.append(p.name[len("wuji2_left_"):-len(".spec.json")])
    return tuple(tags)


LINKS = ("as built", *scaled_tags())
HANDS = set(MH.WUJI2_LR_2PLUS) | {f"wuji2_left_uniform_handonly_{t}" for t in LINKS[1:]}


def hand_spec(side: str, kept: list[str], links: str = "as built") -> str:
    """Registered spec for a side + finger set (make_missing_fingers.py's tags) or a scaled-link variant (full left
    hand only)."""
    if links != "as built":
        return (f"wuji2_left_uniform_handonly_{links}" if side == "left" and len(kept) == len(FINGERS)
                else "")
    if len(kept) == len(FINGERS):
        tag = ""
    elif len(kept) == len(FINGERS) - 1:
        tag = "_no_" + next(f for f in FINGERS if f not in kept)
    else:
        tag = "_only_" + "_".join(kept)
    return f"wuji2_{side}_uniform_handonly{tag}"


_ARCH: dict[str, bool] = {}


def is_mlp(run_dir: str) -> bool:
    """The run's network is rl_games' MLP actor_critic (not the joint transformer); cached per run."""
    if run_dir not in _ARCH:
        from omegaconf import OmegaConf
        cfg = pathlib.Path(run_dir) / "rank_0/.hydra/config.yaml"
        _ARCH[run_dir] = cfg.exists() and str(OmegaConf.load(cfg).agent.params.network.name) == "actor_critic"
    return _ARCH[run_dir]


def policies() -> dict[str, str]:
    """Dropdown label -> checkpoint glob (newest epoch taken at load time)."""
    out = {"Multi 52 hands (31826, 6 Oct, no action delay)":
           f"{LOGS}/06oct_wuji2_multi_embodiment/0_scale_train_wuji2_lr_2plus_ema0.1_c01_*_31826_0_*"
           "/rank_0/*/nn/last_*_ep_*.pth"}
    for d in RUN_DIRS:
        for run in sorted(glob.glob(f"{LOGS}/{d}/0_scale_train_*_c01_*")):
            m = re.search(r"0_scale_train_(.+?)_c01_.*_(\d+)_0_[^/]+$", run)
            if m:
                out[f"{'[MLP] ' if is_mlp(run) else ''}{m.group(1)} ({m.group(2)})"] = \
                    f"{run}/rank_0/*/nn/last_*_ep_*.pth"
    return out


def run_dir_of(pattern: str) -> str:
    return pattern.split("/rank_0/")[0]


def trained_spec(pattern: str) -> str | None:
    """The robot_spec a policy trained on, from its run config (before any checkpoint exists)."""
    from omegaconf import OmegaConf
    runs = glob.glob(run_dir_of(pattern))
    cfg = pathlib.Path(runs[0]) / "rank_0/.hydra/config.yaml" if runs else None
    return str(OmegaConf.load(cfg).env.assets.robot_spec) if cfg and cfg.exists() else None


def selectors_for(spec: str):
    """(side, fingers kept, links) that select a registered single-hand spec, or None."""
    m = re.fullmatch(r"wuji2_(left|right)_uniform_handonly(?:_(.+))?", spec or "")
    if not m:
        return None
    side, tag = m.group(1), m.group(2) or ""
    if tag in LINKS[1:]:
        return side, list(FINGERS), tag
    if not tag:
        return side, list(FINGERS), "as built"
    if tag.startswith("no_"):
        return side, [f for f in FINGERS if f != tag[3:]], "as built"
    if tag.startswith("only_"):
        return side, [f for f in FINGERS if f in tag[5:].split("_")], "as built"
    return None


def newest(pattern: str) -> str | None:
    pths = glob.glob(pattern)
    return max(pths, key=lambda p: int(re.search(r"_ep_(\d+)_", p).group(1))) if pths else None


def trained_on(ckpt: str) -> set[str]:
    from omegaconf import OmegaConf
    ref = str(OmegaConf.load(pathlib.Path(ckpt).parents[3] / "rank_0/.hydra/config.yaml").env.assets.robot_spec)
    return set(MH.spec_names(ref)) if MH.is_multi_ref(ref) else {ref}


def trained_delay(ckpt: str) -> bool:
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(pathlib.Path(ckpt).parents[3] / "rank_0/.hydra/config.yaml")
    return bool(cfg.env.domain_randomization.use_action_delay)


def short(spec: str) -> str:
    side = "R" if "_right_" in spec else "L"
    tag = spec.split("uniform_handonly")[1].lstrip("_") or "full"
    return f"{side} {tag}"


class Panel:
    """One hand + one policy: its selectors, its worker, its drawing under /p<i>."""

    def __init__(self, server, i: int, x: float, pols: dict, side: str, kept: tuple, pol: str, sets: tuple):
        self.server, self.i, self.x, self.pols, self.sets = server, i, x, pols, sets
        self.authkey = secrets.token_bytes(16)
        self.listener = Listener(("127.0.0.1", 0), authkey=self.authkey)
        with server.gui.add_folder(f"panel {i + 1}"):
            self.dd_side = server.gui.add_dropdown("side", ("left", "right"), initial_value=side)
            self.cb = {f: server.gui.add_checkbox(f, f in kept) for f in FINGERS}
            self.dd_links = server.gui.add_dropdown("links", LINKS, initial_value=LINKS[0])
            self.dd_pol = server.gui.add_dropdown("policy", tuple(pols), initial_value=pol)
            self.dd_delay = server.gui.add_dropdown("action delay", DELAY, initial_value=DELAY[0])
            self.s_mass = server.gui.add_number("object mass g (on load)", 0.0, min=0.0, step=1.0)
            self.s_hand_mu = server.gui.add_slider("hand friction", 0.05, 1.5, 0.05, 0.5)
            self.s_obj_mu = server.gui.add_slider("object friction", 0.05, 1.5, 0.05, 0.5)
            self.s_push = server.gui.add_slider("push force scale (1 / 0.3 s)", 0.0, 60.0, 1.0, 0.0)
            self.b_load = server.gui.add_button("load")
            self.md = server.gui.add_markdown("")
            self.md_count = server.gui.add_markdown("")
        self.cmds: list[dict] = []
        self.base_mass_g: float | None = None   # the trained object's mass, learned from the first loaded scene
        self.density = 1.0                      # object_density_scale of the scene being loaded
        self.s_hand_mu.on_update(lambda _: self._send_physics())
        self.s_obj_mu.on_update(lambda _: self._send_physics())
        self.s_push.on_update(lambda _: self._send_physics())
        self.dd_delay.on_update(lambda _: self._send_physics())
        self.load_requested = True
        self.b_load.on_click(lambda _: setattr(self, "load_requested", True))
        for w in (self.dd_side, self.dd_links, *self.cb.values()):
            w.on_update(lambda _: self._preview())
        self.dd_pol.on_update(lambda _: self._policy_changed())
        self.child = self.conn = self.info = None
        self.urdf = self.obj = self.goal = self.label = None
        self.title, self.dead = "", False
        self.rate, self.n_frames, self.t_rate = 0.0, 0, time.time()
        self.t_count = 0.0

    def refresh_policies(self, pols: dict):
        """New runs appear as training jobs start; keep the selection."""
        if tuple(pols) != tuple(self.pols):
            self.pols = pols
            cur = str(self.dd_pol.value)
            self.dd_pol.options = tuple(pols)
            if cur in pols:
                self.dd_pol.value = cur

    def _physics_cmds(self) -> list[dict]:
        mu_h, mu_o = float(self.s_hand_mu.value), float(self.s_obj_mu.value)
        return [{"cmd": "friction", "value": {"robot_friction": mu_h, "finger_tip_friction": mu_h,
                                              "object_friction": mu_o}},
                {"cmd": "push_period", "value": PUSH_PERIOD},
                {"cmd": "action_delay", "value": self.dd_delay.value == DELAY[0]},
                {"cmd": "force_scale", "value": float(self.s_push.value)}]

    def _send_physics(self):
        if self.info is not None:
            self.cmds += self._physics_cmds()
            self._show_status()

    def _show_status(self):
        if self.info is None:
            return
        push = float(self.s_push.value)
        self.md.content = (f"{self.status}  ·  action delay **{'on' if self.dd_delay.value == DELAY[0] else 'off'}**"
                           f"\n\nobject {self.info['object_mass'] * 1000:.0f} g (trained "
                           f"{self.base_mass_g:.0f} g)  ·  friction hand "
                           f"{float(self.s_hand_mu.value):.2f}, object {float(self.s_obj_mu.value):.2f}  ·  "
                           + (f"push scale {push:.0f} every 0.3 s (~{push / 9.81:.1f} g, {push * self.info['object_mass']:.2f} N "
                              "rms per axis)"
                              if push > 0 else "no pushes"))

    def _choice(self):
        kept = [f for f in FINGERS if self.cb[f].value]
        return hand_spec(str(self.dd_side.value), kept, str(self.dd_links.value)), str(self.dd_pol.value)

    def _mlp_hand(self, pol: str) -> str | None:
        """The one hand an MLP policy can drive (None for a joint transformer)."""
        pat = self.pols.get(pol)
        return trained_spec(pat) if pat and is_mlp(run_dir_of(glob.glob(run_dir_of(pat))[0])
                                                    if glob.glob(run_dir_of(pat)) else "") else None

    def _policy_changed(self):
        """An MLP drives only its own hand: move the hand selectors to it."""
        own = self._mlp_hand(str(self.dd_pol.value))
        sel = selectors_for(own) if own else None
        if sel:
            side, kept, links = sel
            self.dd_side.value, self.dd_links.value = side, links
            for f in FINGERS:
                self.cb[f].value = f in kept
        self._preview()

    def _preview(self):
        spec, pol = self._choice()
        own = self._mlp_hand(pol)
        if spec not in HANDS:
            self.md.content = ("**pick two or more fingers** (the 52 hands); a scaled-link variant needs the full "
                               "left hand")
        elif own and spec != own:
            self.md.content = f"**{pol} is an MLP: it drives only {short(own)}**"
        else:
            self.md.content = f"selected **{short(spec)}** -- press load"

    def stop(self):
        if self.conn is not None:
            try:
                self.conn.send({"cmd": "quit"})
            except Exception:
                pass
        if self.child is not None:
            time.sleep(0.5)
            if self.child.poll() is None:
                self.child.kill()
            self.child.wait()
        try:
            self.server.scene.remove_by_name(f"/p{self.i}")
        except Exception:
            pass
        self.child = self.conn = self.info = None
        self.urdf = self.obj = self.goal = self.label = None
        self.md_count.content = ""

    def start(self):
        self.load_requested = False
        spec, pol = self._choice()
        own = self._mlp_hand(pol)
        if spec not in HANDS or (own and spec != own):
            self._preview()
            return
        ckpt = newest(self.pols[pol])
        if ckpt is None:
            self.md.content = f"**{pol}: no checkpoint yet**"
            return
        self.stop()
        ep = re.search(r"_ep_(\d+)_", ckpt).group(1)
        held = "" if spec in trained_on(ckpt) else "  ·  **held out** (not in its training set)"
        held += "  ·  MLP" if own else "  ·  joint transformer"
        held += f"  ·  max {MAX_GOALS} goals / episode" if MAX_GOALS else "  ·  no goal cap"
        held += "  ·  EMA 0.1" if self.sets else "  ·  EMA as trained"
        if not self.sets:                           # --as-trained: the dropdown starts at the run's own setting
            self.dd_delay.value = DELAY[0] if trained_delay(ckpt) else DELAY[1]
        self.title = f"{short(spec)}  ·  {pol} ep {ep}"
        self.dead = False
        self.md.content = f"**loading {short(spec)}** with {pol} ep {ep}{held} (Kit boots in about a minute)"
        self.status = f"**{short(spec)}** with {pol} ep {ep}{held}"
        host, port = self.listener.address
        cmd = [sys.executable, str(WORKER), "--worker", "--hand", spec, "--host", host, "--wport", str(port),
               "--authkey", self.authkey.hex(), "--policy", f"{pol}={ckpt}", "--max-goals", str(MAX_GOALS)]
        want = float(self.s_mass.value)
        # Every run trained at object_density_scale 1; until the first scene reports its mass, load that.
        self.density = want / self.base_mass_g if self.base_mass_g and want > 0 else 1.0
        for kv in (*self.sets, f"env.assets.object_density_scale={self.density}"):
            cmd += ["--set", kv]
        self.child = subprocess.Popen(cmd, env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"})
        self.conn = self.listener.accept()

    def on_ready(self, m):
        from viser.extras import ViserUrdf
        from viser_unified import _face_cube
        s = self.server
        s.scene.add_frame(f"/p{self.i}", position=(self.x, 0.0, 0.0), show_axes=False)
        s.scene.add_frame(f"/p{self.i}/base", position=tuple(m["base_pos"]), wxyz=tuple(m["base_rot"]),
                          show_axes=False)
        self.urdf = ViserUrdf(s, pathlib.Path(m["urdf"]), root_node_name=f"/p{self.i}/base/robot")
        self.obj = _face_cube(s, f"/p{self.i}/obj", m["cube"], opacity=1.0)
        self.goal = _face_cube(s, f"/p{self.i}/goal", m["cube"], opacity=0.35)
        self.label = s.scene.add_label(f"/p{self.i}/label", self.title)
        self.info = m
        if self.base_mass_g is None:
            self.base_mass_g = m["object_mass"] * 1000.0 / self.density
            self.s_mass.value = round(self.base_mass_g, 1)
        self.cmds += self._physics_cmds()           # the sliders' current values, onto the fresh scene
        self._show_status()
        print(f"[wuji-play] panel {self.i + 1} ready: {self.title}", flush=True)

    def pump(self, outbox: list) -> bool:
        if self.load_requested:
            self.start()
        if self.dead or self.conn is None:
            return False
        for m in (*outbox, *self.cmds):
            self.conn.send(m)
        self.cmds = []
        if not self.conn.poll(timeout=0.01):
            if self.child.poll() is not None:
                self.md.content = "**worker exited** -- press load to restart"
                self.dead = True
            return False
        batch = [self.conn.recv()]
        while self.conn.poll():
            batch.append(self.conn.recv())
        frame = None
        for m in batch:
            if m["kind"] == "frame":
                frame = m
            elif m["kind"] == "ready":
                self.on_ready(m)
            elif m["kind"] == "error":
                self.md.content = "**worker failed**\n\n```\n" + m["text"][-1500:] + "\n```"
                print(m["text"], flush=True)
                self.dead = True
        if frame is None or self.info is None:
            return True
        self.urdf.update_cfg(dict(zip(self.info["joints"], frame["q"][0])))
        self.obj.position, self.obj.wxyz = tuple(frame["obj_pos"][0]), tuple(frame["obj_quat"][0])
        self.goal.position, self.goal.wxyz = tuple(frame["goal_pos"][0]), tuple(frame["goal_quat"][0])
        self.n_frames += len([m for m in batch if m["kind"] == "frame"])
        now = time.time()
        if now - self.t_rate > 2.0:                  # sim seconds per wall second, over the last ~2 s
            self.rate = self.n_frames * CONTROL_DT / (now - self.t_rate)
            self.n_frames, self.t_rate = 0, now
            print(f"[wuji-play] panel {self.i + 1}: {self.rate:.2f}x real time", flush=True)
        if now - self.t_count > 0.25:                # counters, a few times a second
            self.t_count = now
            n_ep = frame["episodes"][0]
            per_ep = f"{frame['done_goals'][0] / n_ep:.1f}" if n_ep else "-"
            self.md_count.content = (
                f"**episodes** {n_ep}  ·  **goals** {frame['goals'][0]}  ·  **drops** {frame['drops'][0]}  ·  "
                f"**goals / episode** {per_ep}  ·  this episode {frame['ep_goals'][0]}  ·  "
                f"{self.rate:.2f}x real time")
        gp = frame["goal_pos"][0]
        self.label.position = (gp[0], gp[1], gp[2] + 0.09)
        self.label.text = (f"{self.title}  ·  {frame['goals'][0]} goals  ·  {frame['drops'][0]} drops"
                           f"  ·  {self.rate:.2f}x real time")
        return True


def main():
    global MAX_GOALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8085)
    ap.add_argument("--panels", type=int, default=1, help="side-by-side panels, each with its own hand and policy")
    ap.add_argument("--spacing", type=float, default=0.32)
    ap.add_argument("--max-goals", type=int, default=MAX_GOALS, help="goals per episode before it ends; 0 = no cap")
    ap.add_argument("--policy", default="", help="initial policy: a substring of its dropdown label (default: 31826)")
    ap.add_argument("--as-trained", action="store_true",
                    help="each policy's own run's action delay / moving average (default: delay on + EMA 0.1 for all)")
    args = ap.parse_args()
    sets = () if args.as_trained else CONSISTENT
    MAX_GOALS = args.max_goals

    import viser

    pols = policies()
    server = viser.ViserServer(port=args.port)
    print(f"\n[wuji-play] viser on http://{socket.gethostname()}:{args.port}  ({len(pols)} policies)\n", flush=True)
    first = next((k for k in pols if args.policy and args.policy in k), next(iter(pols)))
    panels = [Panel(server, i, (i - (args.panels - 1) / 2) * args.spacing, pols, "left", FINGERS, first, sets)
              for i in range(args.panels)]
    for pn in panels:
        pn._policy_changed()                         # an MLP first: its own hand
    with server.gui.add_folder("playback (all panels)"):
        b_pause = server.gui.add_button("pause / resume")
        b_reset = server.gui.add_button("reset")
        s_speed = server.gui.add_slider("speed (x real time)", 0.1, 2.0, 0.05, 1.0)
    outbox: "queue.Queue[dict]" = queue.Queue()
    b_pause.on_click(lambda _: outbox.put({"cmd": "toggle"}))
    b_reset.on_click(lambda _: outbox.put({"cmd": "reset"}))
    s_speed.on_update(lambda _: outbox.put({"cmd": "speed", "value": float(s_speed.value)}))
    t_scan = time.time()
    try:
        while True:
            if time.time() - t_scan > 60.0:          # pick up runs that started after the viewer
                t_scan = time.time()
                for pn in panels:
                    pn.refresh_policies(policies())
            msgs = []
            while True:
                try:
                    msgs.append(outbox.get_nowait())
                except queue.Empty:
                    break
            busy = [p.pump(msgs) for p in panels]
            if not any(busy):
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for p in panels:
            p.stop()


if __name__ == "__main__":
    main()
