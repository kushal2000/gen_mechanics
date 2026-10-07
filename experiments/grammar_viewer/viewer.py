"""The grammar viewer (viser): one compact panel for the locked hand grammar.

    .venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1

Grammar (Random, "found after N tries"), Rules (Evolution Rules / No Rules /
Custom Rules, the limits in plain words), Viability (C1 and C2, one line
each), Commercial hand (the conformed hand over its URDF meshes, with its
fit), Mutation (Coarse 10 mm / 30 deg, Fine 1 mm / 5 deg; only the operators
that can act are shown) and Pose (curl, re-centre). CPU only; Isaac is never
imported.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import viser  # noqa: E402

from hand_sampler.grammar import build_conformed  # noqa: E402
from hand_sampler.grammar import commercial as gcom  # noqa: E402
from hand_sampler.grammar import conform as gcf  # noqa: E402
from hand_sampler.grammar import derive as gdv  # noqa: E402
from hand_sampler.grammar import operators as gops  # noqa: E402
from hand_sampler.grammar import viability as gvb  # noqa: E402
from hand_sampler.grammar.fk import forward_kinematics  # noqa: E402
from hand_sampler.grammar.hand import (  # noqa: E402
    BASE_DISTANCE_MM, EVOLUTION_RULES, NO_RULES, STEPS, Hand, Rules, check, parameter_count)

from gviewer import draw  # noqa: E402
from gviewer import meshes as gmesh  # noqa: E402

EVOLUTION, NONE, CUSTOM = "Evolution Rules", "No Rules", "Custom Rules"
RULE_SETS: Dict[str, Optional[Rules]] = {EVOLUTION: EVOLUTION_RULES, NONE: NO_RULES, CUSTOM: None}
STAGE_LABELS = {"Coarse (10 mm / 30°)": "coarse", "Fine (1 mm / 5°)": "fine"}
NO_HAND = "(none)"
MAX_TRIES = 2000
MESH_OPACITY = 0.35

C1_HINT = ("No two links (rounded boxes; the palm plate counts as one) overlap by more than 3 mm, with every joint "
           "at 0 and at the start pose (bending joints at 0.35 of their range). Deeper starting overlaps made the "
           "physics engine push links apart at over 100 rad/s.")
C2_HINT = ("At least one pair of fingers whose fingertips can meet above the palm: each fingertip is swept over its "
           "own joints' ranges; only positions above the plate and over the palm count; two fingers pass if their "
           "swept tips come within 20 mm. Every commercial hand passes.")


def _verdict(ok: Optional[bool]) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


class GrammarViewer:
    """State and callbacks. GUI callbacks call these public methods; tests can
    call them directly (`wait=True` runs a job synchronously)."""

    def __init__(self, server: viser.ViserServer, *, seed: int = 20261007, build_initial: bool = True):
        self.server = server
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(seed)
        self.hand: Optional[Hand] = None
        self.history: List[Hand] = []
        self.commercial: Optional[dict] = None       # the conformed record shown, if any
        self.real: Optional[gcom.RealHand] = None
        self.jmap: Dict[str, Any] = {}
        self.viability: Optional[gvb.Viability] = None
        self.tries = 0
        self.last_change = ""
        self._job_running = False
        self._gate = threading.Lock()
        self._suppress = False
        self.records = build_conformed.load()

        server.scene.set_up_direction("+z")
        server.scene.add_grid("/grid", width=0.6, height=0.6, cell_size=0.02, plane="xy",
                              position=(0.0, 0.0, -gdv.PALM_THICKNESS_MM / 2 * 1e-3 - 0.002))
        self.drawing = draw.HandDrawing(server, "/hand")
        self.overlay = draw.MeshOverlay(server, "/hand/urdf")
        self._build_gui()
        if build_initial:
            self.random(wait=True)

    # ------------------------------------------------------------------ jobs
    def busy(self) -> bool:
        return self._job_running

    def run_job(self, name: str, fn: Callable[[], None], wait: bool = False) -> bool:
        with self._gate:
            if self._job_running:
                self.md_status.content = f"busy ({name} ignored)"
                return False
            self._job_running = True

        def body():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.md_status.content = f"error in {name}: {type(exc).__name__}: {exc}"
            finally:
                self._job_running = False

        t = threading.Thread(target=body, daemon=True)
        t.start()
        if wait:
            t.join()
        return True

    def wait_idle(self, timeout: float = 120.0) -> bool:
        t0 = time.time()
        while self.busy() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.busy()

    # ------------------------------------------------------------------ rules
    def rules(self) -> Rules:
        """The rules the panel shows (always inside the grammar's limits)."""
        types = ("hinge",) + (("coupled",) if self.gui_coupled.value else ()) + (
            ("sliding",) if self.gui_sliding.value else ())
        return Rules(joint_types=types, max_fingers=int(self.gui_fingers.value), max_joints=int(self.gui_joints.value),
                     max_palm_joints=int(self.gui_palm.value), max_finger_length_mm=int(self.gui_length.value),
                     min_spacing_mm=int(self.gui_spacing.value),
                     base_distance_mm=(BASE_DISTANCE_MM[0], int(self.gui_base.value))).clamp()

    def _show_rules(self, r: Rules) -> None:
        self._suppress = True
        try:
            self.gui_fingers.value = r.max_fingers
            self.gui_joints.value = r.max_joints
            self.gui_palm.value = r.max_palm_joints
            self.gui_coupled.value = "coupled" in r.joint_types
            self.gui_sliding.value = "sliding" in r.joint_types
            self.gui_length.value = r.max_finger_length_mm
            self.gui_spacing.value = r.min_spacing_mm
            self.gui_base.value = r.base_distance_mm[1]
        finally:
            self._suppress = False

    def choose_rules(self, name: str) -> None:
        if RULE_SETS.get(name) is not None:
            self._show_rules(RULE_SETS[name])
        self._suppress = True
        self.gui_rules.value = name
        self._suppress = False
        self._after_rules()

    def _on_field(self) -> None:
        if self._suppress:
            return
        r = self.rules()
        name = next((n for n, v in RULE_SETS.items() if v is not None and v == r), CUSTOM)
        self._suppress = True
        self.gui_rules.value = name
        self._suppress = False
        self._after_rules()

    def _after_rules(self) -> None:
        with self.lock:
            self._update_status()
            self._update_operator_buttons()

    # ------------------------------------------------------------------ actions
    def random(self, wait: bool = False, max_tries: int = MAX_TRIES) -> bool:
        def job():
            rules, stage = self.rules(), self.stage()
            for n in range(1, max_tries + 1):
                h = gops.random_hand(self.rng, rules, stage)
                v = gvb.viability(h)
                if self._required_ok(v):
                    self.tries = n
                    self._set_hand(h, history="reset", commercial=None, viability=v, frame=True)
                    self.md_random.content = f"found after {n} {'try' if n == 1 else 'tries'}"
                    return
            self.md_random.content = f"no hand passed the checks in {max_tries} tries"
        return self.run_job("random", job, wait)

    def _required_ok(self, v: gvb.Viability) -> bool:
        return (not self.gui_c1.value or v.c1.ok) and (not self.gui_c2.value or v.c2.ok)

    def load_commercial(self, hand_id: str, wait: bool = False) -> bool:
        def job():
            rec = self.records[hand_id]
            real = gcom.load_real_hand(hand_id)
            self.jmap = gcf.urdf_joint_map(rec["hand"], rec["palm_T"], rec["q_off"], rec["name_map"], real)
            self.real = real
            path = gcom.mesh_path(gcom.entry(hand_id))
            ms = gmesh.load_link_meshes(path)
            self.overlay.build(ms, MESH_OPACITY)
            self.overlay.set_visible(bool(self.gui_meshes.value))
            self._set_hand(rec["hand"], history="reset", commercial=rec, viability=gvb.viability(rec["hand"]),
                           frame=True)
            n = parameter_count(rec["hand"])
            fit = (f"{rec['max_joint_mm']:.1f} mm joints, {rec['max_axis_deg']:.1f}° axes, {rec['max_tip_mm']:.1f} mm "
                   f"tips ({'within' if rec['within_target'] else 'outside'} 5 mm / 10°)")
            rules_note = "follows the rules" if not check(rec["hand"], self.rules()) else "outside these rules"
            self.md_fit.content = (f"{len(rec['hand'].fingers)} fingers, {n['joints']} joints; fit: {fit}; "
                                   f"{rules_note}. Meshes: {ms.summary()}")
        return self.run_job("commercial", job, wait)

    def stage(self) -> str:
        return STAGE_LABELS[self.gui_stage.value]

    def set_stage(self, stage: str) -> None:
        self.gui_stage.value = next(k for k, v in STAGE_LABELS.items() if v == stage)
        with self.lock:
            self._update_operator_buttons()

    def mutate(self, operator: Optional[str] = None, wait: bool = False) -> bool:
        def job():
            if self.hand is None:
                return
            rules, stage = self.rules(), self.stage()
            if operator is None:
                child, name, mv = gops.mutate(self.hand, self.rng, rules, stage)
            else:
                res = gops.apply_operator(self.hand, operator, self.rng, rules, stage)
                if res is None:
                    self.md_mut.content = f"'{gops.OPERATOR_BY_NAME[operator].label}' cannot act on this hand"
                    return
                (child, mv), name = res, operator
            self.last_change = f"{gops.OPERATOR_BY_NAME[name].label} ({stage}): {mv}"
            self._set_hand(child, history="push", commercial=None,
                           viability=gvb.viability(child), frame=False)
            self.md_mut.content = self.last_change
        return self.run_job("mutate", job, wait)

    def back(self, wait: bool = False) -> bool:
        def job():
            if len(self.history) < 2:
                self.md_mut.content = "nothing to undo"
                return
            self.history.pop()
            h = self.history[-1]
            self._set_hand(h, history="keep", commercial=None, viability=gvb.viability(h), frame=False)
            self.md_mut.content = "back one step"
        return self.run_job("back", job, wait)

    def set_curl(self, frac: float) -> None:
        self.gui_curl.value = float(frac)

    def recentre(self) -> None:
        self._frame_camera()

    # ------------------------------------------------------------------ state
    def _set_hand(self, hand: Hand, history: str, commercial: Optional[dict], viability: gvb.Viability,
                  frame: bool) -> None:
        with self.lock:
            self.hand = hand
            if history == "reset":
                self.history = [hand]
            elif history == "push":
                self.history.append(hand)
            if commercial is None:
                self.commercial, self.real, self.jmap = None, None, {}
                self.overlay.clear()
                self.md_fit.content = ""
                self._suppress = True
                self.gui_hand.value = NO_HAND
                self._suppress = False
            else:
                self.commercial = commercial
            self.viability = viability
            bad = [(a, b) for a, b, _ in viability.c1.pairs] if self.gui_c1.value else []
            self.drawing.build(hand, highlight=bad)
            self._render_pose()
            self._update_status()
            self._update_operator_buttons()
            if frame:
                self._frame_camera()

    def q(self) -> np.ndarray:
        """The joint vector on screen: flexion joints at curl x their upper
        limit (a commercial hand: added to its zero-pose difference, so curl 0
        is the real hand's zero pose)."""
        q = gdv.curl_q(self.hand, float(self.gui_curl.value))
        if self.commercial is not None:
            q = gdv.tie(self.hand, q + self.commercial["q_off"])
        return q

    def _render_pose(self) -> None:
        if self.hand is None:
            return
        q = self.q()
        self.drawing.pose(q)
        if self.commercial is not None and self.real is not None:
            vals = gcf.urdf_joint_values(self.jmap, q, self.commercial["q_off"])
            model = self.real.model
            for c in model.couplings:                       # the URDF's own mimic joints
                if c.dependent not in vals and c.source in vals:
                    vals[c.dependent] = c.multiplier * vals[c.source] + c.offset
            W = forward_kinematics(model, vals)
            Pinv = np.linalg.inv(self.commercial["palm_T"])
            self.overlay.update({link: Pinv @ T for link, T in W.items()})

    def _update_status(self) -> None:
        if self.hand is None:
            return
        v = self.viability
        self.gui_c1.label = f"C1 no overlap (zero, start): {_verdict(v.c1.ok)} {v.c1.worst_mm:.1f} mm"
        best = "none" if not np.isfinite(v.c2.best_mm) else f"{v.c2.best_mm:.1f} mm"
        self.gui_c2.label = f"C2 fingertips meet above palm: {_verdict(v.c2.ok)} {best}"
        problems = check(self.hand, self.rules())
        n = parameter_count(self.hand)
        follows = "follows the rules" if not problems else f"outside the rules: {problems[0]}"
        self.md_rules.content = follows
        self.md_status.content = (f"{len(self.hand.fingers)} fingers, {n['joints']} joints, "
                                  f"{len(self.hand.palm_joints)} palm joints; {n['numbers']} numbers; {follows}")

    def _update_operator_buttons(self) -> None:
        if self.hand is None:
            return
        rules, stage = self.rules(), self.stage()
        for name, b in self.op_buttons.items():
            op = gops.OPERATOR_BY_NAME[name]
            b.visible = gops.can_act(self.hand, op, rules, stage)

    def _frame_camera(self, clients=None) -> None:
        if self.hand is None:
            return
        c, r = self.drawing.bounds(self.q())
        look = tuple(float(v) for v in c)
        pos = tuple(float(v) for v in c + np.array([-1.2, -1.6, 1.4]) * r * 1.6)
        try:                                   # clients that connect later start here
            self.server.initial_camera.look_at = look
            self.server.initial_camera.position = pos
        except AttributeError:
            pass
        for client in (clients if clients is not None else self.server.get_clients().values()):
            client.camera.look_at = look
            client.camera.position = pos

    # ------------------------------------------------------------------ GUI
    def _build_gui(self) -> None:
        g = self.server.gui
        g.configure_theme(control_width="medium")
        self.md_status = g.add_markdown("starting ...")

        with g.add_folder("Grammar"):
            btn_random = g.add_button("Random", hint="Draw hands under the rules until the ticked viability checks pass.")
            self.md_random = g.add_markdown("")

        with g.add_folder("Rules"):
            self.gui_rules = g.add_dropdown("rules", list(RULE_SETS), initial_value=EVOLUTION,
                                            hint="Random hands and every mutation follow these rules. Evolution Rules: "
                                                 "what the simulator builds and evolution uses (hinge and coupled "
                                                 "joints). No Rules: the whole grammar (sliding joints too). Editing a "
                                                 "field makes them Custom Rules; fields can only tighten the grammar.")
            r = EVOLUTION_RULES
            self.gui_fingers = g.add_slider("max fingers", 2, 6, 1, r.max_fingers, hint="Fingers per hand: 2-6.")
            self.gui_joints = g.add_slider("max joints per finger", 1, 5, 1, r.max_joints,
                                           hint="Joints per finger: 1-5.")
            self.gui_palm = g.add_slider("max palm joints", 0, 6, 1, r.max_palm_joints,
                                         hint="Hinged palm sections: at most 6, one per finger at most, never one "
                                              "without a finger.")
            self.gui_coupled = g.add_checkbox("coupled joints", True, hint="A coupled joint follows the joint before "
                                                                           "it in its finger at 1.1 x its angle.")
            self.gui_sliding = g.add_checkbox("sliding joints", False, hint="Sliding (prismatic) joints, like Dex1's "
                                                                            "jaws: in the grammar, not in evolution.")
            self.gui_length = g.add_number("max finger length (mm)", r.max_finger_length_mm, min=10, max=250, step=1,
                                           hint="Sum of a finger's link lengths: at most 250 mm (DClaw's 221 x 1.1).")
            self.gui_spacing = g.add_number("min finger spacing (mm)", r.min_spacing_mm, min=19, max=100, step=1,
                                            hint="Neighbouring finger bases at least one link width (19 mm) apart.")
            self.gui_base = g.add_number("max base distance from wrist (mm)", r.base_distance_mm[1],
                                         min=BASE_DISTANCE_MM[0], max=BASE_DISTANCE_MM[1], step=1,
                                         hint=f"Finger bases sit {BASE_DISTANCE_MM[0]}-{BASE_DISTANCE_MM[1]} mm from "
                                              "the wrist centre on the plate (commercial hands x0.9 / x1.1).")
            g.add_markdown("links: 0 or 15-90 mm, fingertip ≥ 10 mm")
            self.md_rules = g.add_markdown("")

        with g.add_folder("Viability"):
            self.gui_c1 = g.add_checkbox("C1 no overlap", True, hint=C1_HINT + " Untick: Random stops requiring it.")
            self.gui_c2 = g.add_checkbox("C2 fingertips meet above palm", True,
                                         hint=C2_HINT + " Untick: Random stops requiring it.")

        with g.add_folder("Commercial hand"):
            ids = sorted(self.records)
            self.gui_hand = g.add_dropdown("hand", [NO_HAND] + ids, initial_value=NO_HAND,
                                           hint="A commercial hand conformed onto the grammar (fine grid), over its "
                                                "own URDF meshes.")
            self.gui_meshes = g.add_checkbox("show real meshes", True)
            self.md_fit = g.add_markdown("")

        with g.add_folder("Mutation"):
            self.gui_stage = g.add_dropdown("steps", list(STAGE_LABELS), initial_value=list(STAGE_LABELS)[0],
                                            hint="Coarse: every value moves 10 mm or 30°, and fingers, joints and "
                                                 "palm joints can be added or removed. Fine: 1 mm or 5°, values only.")
            btns = g.add_button_group("mutate", ["Random mutation", "Back"])
            self.md_mut = g.add_markdown("")
            self.op_buttons: Dict[str, Any] = {}
            for op in gops.OPERATORS:
                self.op_buttons[op.name] = g.add_button(op.label, hint=op.hover)

        with g.add_folder("Pose"):
            self.gui_curl = g.add_slider("curl", 0.0, 1.0, 0.01, gdv.START_CURL,
                                         hint="Bending joints at this fraction of their range (0.35: the pose every "
                                              "episode starts from); other joints at 0.")
            btn_center = g.add_button("re-centre view", hint="Point the camera at the whole hand.")

        btn_random.on_click(lambda _: self.random())
        self.gui_rules.on_update(lambda _: None if self._suppress else self.choose_rules(self.gui_rules.value))
        for h in (self.gui_fingers, self.gui_joints, self.gui_palm, self.gui_coupled, self.gui_sliding,
                  self.gui_length, self.gui_spacing, self.gui_base):
            h.on_update(lambda _: self._on_field())

        @self.gui_hand.on_update
        def _(_):
            if not self._suppress and self.gui_hand.value != NO_HAND:
                self.load_commercial(self.gui_hand.value)

        self.gui_meshes.on_update(lambda _: self.overlay.set_visible(bool(self.gui_meshes.value)))
        self.gui_stage.on_update(lambda _: self._after_rules())

        def _checks(_):
            with self.lock:
                if self.hand is not None:
                    self._set_hand(self.hand, history="keep", commercial=self.commercial, viability=self.viability,
                                   frame=False)

        self.gui_c1.on_update(_checks)
        self.gui_c2.on_update(_checks)

        @btns.on_click
        def _(event):
            (self.mutate if event.target.value == "Random mutation" else self.back)()

        for name, b in self.op_buttons.items():
            b.on_click(lambda _, name=name: self.mutate(name))

        @self.gui_curl.on_update
        def _(_):
            with self.lock:
                self._render_pose()

        btn_center.on_click(lambda _: self.recentre())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--hand", default="", help="start with this commercial hand (a manifest id) instead of a random one")
    args = ap.parse_args(argv)
    server = viser.ViserServer(host=args.host, port=args.port, label="grammar viewer")
    app = GrammarViewer(server, build_initial=not args.hand)
    if args.hand:
        app.load_commercial(args.hand, wait=True)
    print(f"grammar viewer at http://{args.host}:{args.port}", flush=True)
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
