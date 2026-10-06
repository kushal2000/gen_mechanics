"""Essential viser viewer for the hand-kinematics grammar.

    .venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1

One compact panel, one line per item (longer explanations are hover text):
Grammar (variant, Random), Limits (the generation limits sampling and
mutation obey by construction: a preset or custom values), Viability (the
four physical checks generation cannot guarantee, each switchable),
Commercial hand, Mutation (operator buttons, greyed out where the limits or
the hand rule them out) and Pose (curl). `viewer_full.py` keeps the full tool.
CPU only: the simulator's envelope oracle is loaded by file path
(gviewer/envload.py), so Isaac is never imported.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import viser  # noqa: E402

from hand_sampler.grammar import limits as glim  # noqa: E402
from hand_sampler.grammar.derive import EVOLUTION_OPERATORS, Derivation, VariationImpossible, derive, vary  # noqa: E402
from hand_sampler.grammar.kinematics import KinematicModel  # noqa: E402

from gviewer import analysis as an  # noqa: E402
from gviewer import checks as ck  # noqa: E402
from gviewer import commercial as com  # noqa: E402
from gviewer import history as hist  # noqa: E402
from gviewer import limitsui as lui  # noqa: E402
from gviewer import meshes as gmesh  # noqa: E402
from gviewer import model as gm  # noqa: E402
from gviewer import sources as src  # noqa: E402
from gviewer.envload import load_env_modules  # noqa: E402
from gviewer.scene import SPAWN_RGB, TIP_MISS_RGB, TIP_REACH_RGB, HandRenderer, MeshOverlay, RenderOptions  # noqa: E402

GHOST_RGB = (105, 105, 125)
MESH_OPACITY = 0.35
RESET_CURL = an.CURL_FRAC          # palm_up's default_q is this fraction of every joint's range
MAX_TRIES = 5000
NO_HAND = "(none)"

# What each variant adds relative to the one it is built on
# (hand_sampler/grammar/variants.py); shown as the dropdown's hover text.
VARIANT_NOTES: Dict[str, str] = {
    "G_FULL": "The default grammar: 1-6 digits of 1-6 segments, 0-3 palm bodies, hinge, continuous, "
              "sliding and coupled joints, branching digits.",
    "G_SERIAL": "G_FULL without extra palm bodies, branches or palm joints (the old-sampler-like baseline).",
    "G_V1": "G_FULL restricted to what the simulator can build: hinges only, no branches, at most 2 palm "
            "bodies, 1-5 digits.",
    "G_V2": "V1 + digit mounts on one host spaced at least 29 mm apart.",
    "G_V3": "V2 + curl prior (hinge-like axes, segments bend toward the palm) and a thumb-like opposing last "
            "digit (first version; V3s fixes its bugs).",
    "G_V1S": "V1 + digits mount on the host's surface at a random angle around it.",
    "G_V2S": "V1s + mounts spread at least 29 mm apart across all hosts in 3-D.",
    "G_V3S": "V1s + the fixed curl prior (first segment unbent) and a thumb-like opposing last digit.",
    "G_NOPALMJOINT": "G_FULL with every extra palm body rigid.",
    "G_NOBRANCH": "G_FULL without branching digits.",
    "G_NOCOUPLE": "G_FULL without coupled joints.",
    "G_FULL_INS": "G_FULL; growth mutations insert small pieces (1-3 segments, no branches).",
    "G_NOBRANCH_INS": "G_NOBRANCH; growth mutations insert small pieces (1-3 segments).",
    "G_BEND": "G_FULL + a random rest bend (up to 30 deg, 5 mm) on 30% of segments.",
    "G_CONT": "G_FULL with joint limits drawn anywhere in +/-180 deg instead of a fixed menu.",
}

# Button label and hover text for every operator in the evolution driver's
# pool (derive.EVOLUTION_OPERATORS).
OPERATOR_INFO: Dict[str, tuple] = {
    "add_minimal_digit": ("add a digit", "New one-segment digit (one hinge) on the root palm or a palm body."),
    "remove_digit_minimal": ("remove a short digit", "Removes a digit with 1-2 segments and no branches."),
    "insert_phalanx": ("add a segment", "Inserts a new segment (link + joint) somewhere in one digit."),
    "delete_phalanx": ("remove a segment", "Deletes one segment of a digit that has at least 2; branches on it "
                                           "re-attach."),
    "add_palm_body": ("add a palm body", "Adds a palm piece on the root or another palm body, jointed with the "
                                         "variant's palm-joint probability (55% in G_FULL) where the limits allow."),
    "remove_palm_body_empty": ("remove an empty palm body", "Removes a palm body that carries nothing (the exact "
                                                            "undo of 'add a palm body')."),
    "toggle_palm_joint": ("joint/unjoint a palm body", "Gives a rigid palm body a joint (new axis and limits), or "
                                                       "makes a jointed one rigid."),
    "add_branch_digit": ("add a branch digit", "Adds a one-segment digit growing off an existing segment."),
    "remove_branch_digit": ("remove a branch digit", "Removes a one-segment branch digit."),
    "step_axis": ("tilt one joint axis", "Moves one joint's axis by one 15 deg step in elevation or azimuth."),
    "step_limits": ("change one joint's limits", "Moves one joint's limits to the neighbouring choice on the "
                                                 "variant's menu."),
    "step_mount": ("move one mount", "Moves where one digit or palm body attaches: one step along its host, or "
                                     "one 15 deg step of its mount angle."),
    "step_coupling": ("change one coupling", "Steps a coupled joint's ratio or offset to the neighbouring "
                                             "choice (needs a coupled joint)."),
    "step_root_length": ("lengthen/shorten the palm", "Changes the root palm's length by one 5 mm step "
                                                      "(within 20-80 mm)."),
    "step_radius": ("thicken/thin every link", "Moves the hand's single capsule radius to the neighbouring "
                                               "choice (8, 10 or 12 mm)."),
    "step_bend_rpy": ("change one rest bend angle", "Steps one angle of one segment's rest bend (needs a variant "
                                                    "with a bend menu: G_BEND, V3, V3s)."),
    "step_bend_offset": ("change one rest bend offset", "Steps one component of one segment's rest bend offset "
                                                        "(needs a variant with an offset menu: G_BEND)."),
}
assert set(OPERATOR_INFO) == set(EVOLUTION_OPERATORS), "OPERATOR_INFO must cover EVOLUTION_OPERATORS exactly"


def op_label(op: str) -> str:
    return OPERATOR_INFO.get(op, (op, ""))[0]


def fidelity_line(ch: com.CommercialHand, limits: glim.GenerationLimits) -> str:
    """One line: projection error against the URDF, and whether the projection
    is within the current generation limits (which ones it breaks)."""
    parts = []
    f = ch.fidelity
    if f:
        pos = max(f["max_pos_mm"], f["max_tip_mm"] or 0.0)
        parts.append(f"URDF error {pos:.2g} mm / {f['max_axis_deg']:.2g} deg")
    parts.append("within limits: " + glim.check(ch.projection.derivation, limits).summary())
    return "; ".join(parts)


def change_line(d: hist.DiffSummary) -> str:
    """The parent -> child diff in one line."""
    parts = []
    for name, (a, b) in (("digits", d.digits), ("joints", d.joints), ("palm bodies", d.palm_bodies)):
        if a != b:
            parts.append(f"{name} {a} -> {b}")
    params = [c for c in d.changed if "(renumbered)" not in c]
    if params and not parts:
        more = f" (+{len(params) - 1} more)" if len(params) > 1 else ""
        parts.append(params[0] + more)
    return "; ".join(parts) or "no visible change"


def check_label(key: str, r: Optional[ck.CheckResult]) -> str:
    c = ck.CHECK_BY_KEY[key]
    star = "*" if c.provisional else ""
    if r is None:
        return f"{c.short}{star}"
    word = {"pass": "PASS", "fail": "FAIL", "n/a": "n/a", "skipped": "-"}.get(r.status, r.status)
    value = "" if r.status == ck.NA else f" {r.value}"
    return f"{c.short}{star}: {word}{value}"


@dataclass
class Shown:
    derivation: Derivation
    model: KinematicModel
    label: str
    kind: str                                   # "sampled" | "commercial" | "mutant"
    commercial: Optional[com.CommercialHand] = None


@dataclass
class Prepared:
    shown: Shown
    ev: ck.Evaluation
    reach: List[an.FingerReach]
    view: gm.ModelView
    cells: Dict[str, gm.CellMesh]


class EssentialViewer:
    """State and callbacks. GUI callbacks call the public methods, which tests
    can also call directly (`wait=True` runs the job synchronously)."""

    def __init__(self, server: viser.ViserServer, *, variant: str = "G_V3S", start_seed: Optional[int] = None,
                 build_initial: bool = True):
        self.server = server
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(20261006)
        self.history = hist.History()
        self.prep: Optional[Prepared] = None
        self.parent_view: Optional[gm.ModelView] = None
        self.variant = variant
        self.last_search: Optional[ck.SearchResult] = None
        self.commercial_cache: Dict[str, com.CommercialHand] = {}
        self.mesh_cache: Dict[str, gmesh.MeshSet] = {}
        self.spawn_handles: List[Any] = []
        self.op_status: Dict[str, str] = {}
        self._job_gate = threading.Lock()
        self._job_running = False
        self._job: Optional[threading.Thread] = None
        self._suppress = False

        load_env_modules()
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.05, plane="xy", position=(0.0, 0.0, -0.002))
        self.renderer = HandRenderer(server, "/hand")
        self.ghost = HandRenderer(server, "/ghost")
        self.overlay = MeshOverlay(server, "/hand/urdf")
        self._build_gui()
        if build_initial:
            self.random(variant, start_seed=start_seed, wait=True)

    # ------------------------------------------------------------------
    # Jobs (one at a time, on a worker thread)
    # ------------------------------------------------------------------

    def busy(self) -> bool:
        return self._job_running

    def run_job(self, name: str, fn: Callable[[], None], wait: bool = False) -> bool:
        with self._job_gate:
            if self._job_running:
                self.md_status.content = f"busy: wait for the current job ({name} ignored)"
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

        self._job = threading.Thread(target=body, daemon=True)
        self._job.start()
        if wait:
            self._job.join()
        return True

    def wait_idle(self, timeout: float = 120.0) -> bool:
        t0 = time.time()
        while self.busy() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.busy()

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def limits(self) -> glim.GenerationLimits:
        """The generation limits the panel currently shows."""
        return lui.build_limits(
            {k: h.value for k, h in self.limit_ints.items()},
            {k: bool(h.value) for k, h in self.limit_bools.items()},
            self.gui_joint_types.value, bool(self.gui_coupled.value),
        )

    def _set_fields(self, limits: glim.GenerationLimits) -> None:
        self._suppress = True
        try:
            for k, h in self.limit_ints.items():
                h.value = lui.int_to_option(getattr(limits, k))
            for k, h in self.limit_bools.items():
                h.value = bool(getattr(limits, k))
            self.gui_joint_types.value = lui.joint_types_option(limits)
            self.gui_coupled.value = "Coupled" in limits.allowed_modules
        finally:
            self._suppress = False

    def set_preset(self, name: str) -> None:
        self.gui_preset.value = name          # fires on_update

    def set_limit(self, key: str, value) -> None:
        """Set one limit field (as the panel would), which switches the preset to Custom."""
        if key in self.limit_ints:
            self.limit_ints[key].value = lui.int_to_option(value) if not isinstance(value, str) else value
        elif key in self.limit_bools:
            self.limit_bools[key].value = bool(value)
        elif key == "joint_types":
            self.gui_joint_types.value = value
        elif key == "coupled":
            self.gui_coupled.value = bool(value)
        else:
            raise KeyError(key)

    def _on_preset(self) -> None:
        if self._suppress:
            return
        name = self.gui_preset.value
        if name in glim.PRESETS:
            self._set_fields(glim.PRESETS[name])
        self._limits_changed()

    def _on_field(self) -> None:
        if self._suppress:
            return
        try:
            lim = self.limits()
        except ValueError as exc:
            self.md_limits.content = f"invalid: {exc}"
            return
        self._suppress = True
        try:
            self.gui_preset.value = lui.preset_of(lim)
        finally:
            self._suppress = False
        self._limits_changed()

    def _limits_changed(self) -> None:
        with self.lock:
            self._update_limits_line()
            self._update_operator_buttons()
            if self.prep is not None and self.prep.shown.commercial is not None:
                self.md_fidelity.content = fidelity_line(self.prep.shown.commercial, self.limits())

    def _update_limits_line(self) -> None:
        if self.prep is None:
            return
        rep = glim.check(self.prep.shown.derivation, self.limits())
        self.md_limits.content = "this hand: " + ("within limits" if rep.ok else "outside: " + "; ".join(
            rep.line(k) for k in rep.failing))

    # ------------------------------------------------------------------
    # Viability checks
    # ------------------------------------------------------------------

    def enabled(self) -> set:
        return {k for k, cb in self.check_boxes.items() if cb.value}

    def set_check(self, key: str, on: bool) -> None:
        self.check_boxes[key].value = bool(on)   # fires on_update, which refreshes the readout

    def set_all_checks(self, on: bool) -> None:
        for k in ck.CHECK_KEYS:
            self.set_check(k, on)

    def _update_checks_panel(self) -> None:
        if self.prep is None:
            return
        ev = self.prep.ev
        for k in ck.CHECK_KEYS:
            self.check_boxes[k].label = check_label(k, ev.results.get(k))
        self._update_status()

    def _update_status(self) -> None:
        if self.prep is None:
            return
        n_mut = self.history.cursor
        head = f"**{self.history.entries[0].label}**" + (f" + {n_mut} mutation(s)" if n_mut else "")
        ev = self.prep.ev
        failing = ev.failing(self.enabled())
        if not ev.buildable:
            verdict = "viability n/a (outside the simulator)"
        elif failing:
            verdict = "fails " + ", ".join(ck.CHECK_BY_KEY[k].short for k in failing)
        else:
            verdict = "viable"
        self.md_status.content = f"{head}: {verdict}"

    # ------------------------------------------------------------------
    # Showing a design
    # ------------------------------------------------------------------

    def _prepare(self, shown: Shown) -> Prepared:
        ev = ck.evaluate(shown.model)
        reach = an.finger_reach(ev.design, ev.pu) if ev.design is not None else []
        normal = np.asarray(ev.pu.normal, dtype=float) if ev.pu is not None else None
        view = gm.ModelView.build(shown.model, palm_normal=normal)
        return Prepared(shown=shown, ev=ev, reach=reach, view=view, cells=gm.palm_cells(shown.model))

    def show(self, shown: Shown, history_mode: str = "reset", operator: Optional[str] = None,
             frame: bool = True) -> Prepared:
        prep = self._prepare(shown)
        with self.lock:
            self.prep = prep
            if history_mode == "reset":
                self.history.reset(shown.derivation, shown.label)
                self.history.current.payload = shown
            elif history_mode == "push":
                self.history.push(shown.derivation, operator or "?").payload = shown
            self.renderer.build(prep.view, prep.cells, RenderOptions())
            self._build_ghost()
            self._build_spawn()
            self._build_overlay()
            self._render_pose()
            self._update_checks_panel()
            self._update_limits_line()
            self._update_operator_buttons()
            if frame:
                self._frame_camera()
        return prep

    @property
    def shown(self) -> Optional[Shown]:
        return None if self.prep is None else self.prep.shown

    def _curl(self) -> float:
        return float(self.gui_curl.value)

    def _highlight(self) -> Dict[str, tuple]:
        """Bodies in a pair deeper than the 3 mm gate, for each failing overlap check."""
        ev = self.prep.ev
        gate = load_env_modules().grammar_envelope.MAX_REST_PENETRATION_M
        out: Dict[str, tuple] = {}
        for key, pairs in (("overlap_zero", ev.pairs_q0), ("overlap_reset", ev.pairs_reset)):
            if ev.results[key].status != ck.FAIL:
                continue
            for b1, b2, pen in an.named_pairs(ev.design, pairs):
                if pen > gate:
                    out[b1] = out[b2] = gm.OVERLAP_RGB
        return out

    def _render_pose(self) -> None:
        if self.prep is None:
            return
        view = self.prep.view
        u = gm.curl_u(view.ranges, self._curl())
        tips = {}
        if self.gui_show_reach.value:
            tips = {r.tip_body: (TIP_REACH_RGB if r.reaches else TIP_MISS_RGB) for r in self.prep.reach}
        with self.server.atomic():
            self.renderer.update(view.primitives(u), self._highlight(), tips)
            if self.parent_view is not None:
                self.ghost.update(self.parent_view.primitives(gm.curl_u(self.parent_view.ranges, self._curl())))
            ch = self.prep.shown.commercial
            if ch is not None and self.overlay.handles:
                self.overlay.update(ch.orig_link_poses(u))

    def _build_ghost(self) -> None:
        self.ghost.clear()
        self.parent_view = None
        parent = self.history.parent
        if parent is None:
            return
        pm = derive(parent.derivation)
        self.parent_view = gm.ModelView.build(pm, palm_normal=self.prep.view.palm_normal)
        self.ghost.build(self.parent_view, gm.palm_cells(pm), RenderOptions(
            tint=GHOST_RGB, opacity=0.35, cell_opacity=0.12, joint_axes=False, tips=False, root_frame=False))

    def _build_spawn(self) -> None:
        for h in self.spawn_handles:
            h.remove()
        self.spawn_handles = []
        pu = self.prep.ev.pu
        if pu is None:
            return
        sp = np.asarray(pu.spawn_offset, dtype=float)
        vis = bool(self.gui_show_reach.value)
        s = self.server.scene
        self.spawn_handles.append(s.add_icosphere("/hand/spawn/object", radius=an.OBJECT_HALF_SIZE_M,
                                                  color=SPAWN_RGB, opacity=0.35, position=sp, visible=vis))
        self.spawn_handles.append(s.add_icosphere("/hand/spawn/reach", radius=an.REACH_TOL_M, color=SPAWN_RGB,
                                                  wireframe=True, opacity=0.25, position=sp, visible=vis,
                                                  subdivisions=2))

    def _build_overlay(self) -> None:
        self.overlay.clear()
        ch = self.prep.shown.commercial
        if ch is None:
            return
        ms = self.mesh_cache.get(ch.entry.id)
        if ms is not None:
            self.overlay.build(ms, opacity=MESH_OPACITY, visible=bool(self.gui_meshes.value))

    def _frame_camera(self) -> None:
        lo, hi = self.prep.view.primitives(gm.curl_u(self.prep.view.ranges, self._curl())).bounds()
        c = (np.asarray(lo) + np.asarray(hi)) / 2
        d = 1.25 * max(float(np.linalg.norm(np.asarray(hi) - np.asarray(lo))), 0.12)
        pos = c + np.array([0.75 * d, -0.75 * d, 0.55 * d])
        self.server.initial_camera.position = tuple(pos)
        self.server.initial_camera.look_at = tuple(c)
        for client in self.server.get_clients().values():
            client.camera.position = tuple(pos)
            client.camera.look_at = tuple(c)

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def random(self, variant: Optional[str] = None, start_seed: Optional[int] = None, max_tries: int = MAX_TRIES,
               wait: bool = False) -> bool:
        """Sample designs from `variant` under the current limits until every
        enabled viability check passes."""
        variant = variant or self.variant
        start = int(self.rng.integers(0, 1_000_000)) if start_seed is None else int(start_seed)
        enabled = self.enabled()
        limits = self.limits()

        def job():
            self.variant = variant
            self.md_random.content = f"searching {variant} ..."

            def progress(k):
                self.md_random.content = f"searching {variant}: {k} tries ..."

            res = ck.search(src.distribution(variant), enabled, start, max_tries=max_tries, limits=limits,
                            progress=progress)
            self.last_search = res
            if res.derivation is None:
                self.md_random.content = f"none viable in {res.tries} tries; relax a check or the limits"
                return
            self.md_random.content = f"found after {res.tries} {'try' if res.tries == 1 else 'tries'} (seed {res.seed})"
            self._clear_commercial_choice()
            self.show(Shown(res.derivation, res.model, f"{variant} seed {res.seed}", "sampled"))

        return self.run_job(f"random {variant}", job, wait=wait)

    def load_commercial(self, hand_id: str, wait: bool = False) -> bool:
        def job():
            ch = self.commercial_cache.get(hand_id)
            if ch is None:
                ch = com.load_commercial(hand_id)
                self.commercial_cache[hand_id] = ch
            if not ch.ok:
                self.md_fidelity.content = f"{hand_id}: {ch.error}"
                return
            if hand_id not in self.mesh_cache:
                links = [b.name for b in ch.imported.model.bodies]
                self.mesh_cache[hand_id] = gmesh.load_link_meshes(ch.entry.mesh_path, links,
                                                                  fallback_dirs=self._mesh_fallbacks(ch))
            self.md_fidelity.content = fidelity_line(ch, self.limits())
            self.show(Shown(ch.projection.derivation, ch.derived, f"{hand_id} (projection)", "commercial",
                            commercial=ch))

        return self.run_job(f"load {hand_id}", job, wait=wait)

    @staticmethod
    def _mesh_fallbacks(ch: com.CommercialHand) -> List[Path]:
        """Download folders for hands whose URDF mesh paths do not resolve (as in viewer_full)."""
        root = Path(com.load_manifest().get("source_root") or "/")
        table = {"svh_right": [root / "SVH"], "shadow_right_local": [root / "Shadow"]}
        return [p for p in table.get(ch.entry.id, []) if p.is_dir()]

    def _clear_commercial_choice(self) -> None:
        self._suppress = True
        try:
            self.gui_hand.value = NO_HAND
        finally:
            self._suppress = False
        self.md_fidelity.content = ""

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _update_operator_buttons(self) -> None:
        shown = self.shown
        if shown is None:
            return
        self.op_status = lui.operator_status(shown.derivation, src.distribution(self.variant),
                                             EVOLUTION_OPERATORS, self.limits())
        for op, b in self.op_buttons.items():
            st = self.op_status[op]
            b.disabled = st != lui.OK
            note = "" if st == lui.OK else f" Now: {st}."
            b.hint = f"{op}: {OPERATOR_INFO[op][1]}{note}"

    def mutate(self, operator: Optional[str] = None, wait: bool = False) -> bool:
        """Apply `operator` under the current limits, or with None a random one
        from EVOLUTION_OPERATORS (drawing again, without replacement, while the
        drawn one cannot apply)."""
        def job():
            shown = self.shown
            if shown is None:
                return
            dist = src.distribution(self.variant)
            limits = self.limits()
            ops = [operator] if operator else [EVOLUTION_OPERATORS[i]
                                               for i in self.rng.permutation(len(EVOLUTION_OPERATORS))]
            child, used, why = None, None, ""
            for op in ops:
                try:
                    child = vary(shown.derivation, self.rng, dist, operator=op, limits=limits)
                    used = op
                    break
                except VariationImpossible:
                    why = lui.NOT_ALLOWED if self.op_status.get(op) == lui.NOT_ALLOWED else lui.NOTHING
                except Exception as exc:  # noqa: BLE001 - e.g. a projected hand's value is off the variant's grid
                    why = f"{type(exc).__name__}: {exc}"
            if child is None:
                self.md_mut.content = (f"'{op_label(operator)}': {why}" if operator
                                       else "no operator can apply to this hand")
                return
            d = hist.diff(shown.derivation, child)
            model = derive(child)
            base = self.history.entries[0].label
            self.show(Shown(child, model, base, "mutant", commercial=None), "push", operator=used, frame=False)
            prefix = "random: " if not operator else ""
            self.md_mut.content = f"{prefix}{op_label(used)}: {change_line(d)}"

        return self.run_job(f"mutate {operator or 'random'}", job, wait=wait)

    def back(self, wait: bool = False) -> bool:
        def job():
            if not self.history.can_back():
                self.md_mut.content = "nothing to undo"
                return
            undone = self.history.current.operator
            e = self.history.back()
            self.show(e.payload, "keep", frame=False)
            self.md_mut.content = f"undid '{op_label(undone)}'"

        return self.run_job("back", job, wait=wait)

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def set_curl(self, frac: float) -> None:
        self.gui_curl.value = float(frac)     # fires on_update -> _render_pose

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_gui(self) -> None:
        g = self.server.gui
        g.configure_theme(control_width="medium")
        self.md_status = g.add_markdown("starting ...")

        with g.add_folder("Grammar"):
            names = src.variant_names()
            self._variant_by_label = {src.variant_label(n): n for n in names}
            self.gui_variant = g.add_dropdown("Variant", list(self._variant_by_label),
                                              initial_value=src.variant_label(self.variant),
                                              hint=VARIANT_NOTES.get(self.variant, ""))
            btn_random = g.add_button("Random", hint="Sample designs under the limits until every enabled "
                                                     "viability check passes.")
            self.md_random = g.add_markdown("")

        with g.add_folder("Limits"):
            self.gui_preset = g.add_dropdown("Preset", list(lui.PRESET_NAMES), initial_value="Simulator",
                                             hint="Hard rules sampling and mutation obey by construction. "
                                                  "Simulator: exactly what the simulator's 32-slot hand can build. "
                                                  "Unlimited: the whole grammar. Editing a field makes it Custom.")
            self.limit_ints: Dict[str, Any] = {}
            for f in lui.INT_FIELDS:
                self.limit_ints[f.key] = g.add_dropdown(f.label, list(f.options),
                                                        initial_value=lui.int_to_option(getattr(glim.SIMULATOR, f.key)),
                                                        hint=f.hint)
            self.gui_joint_types = g.add_dropdown("Joint types", list(lui.JOINT_TYPE_OPTIONS),
                                                  initial_value=lui.joint_types_option(glim.SIMULATOR),
                                                  hint=lui.JOINT_TYPES_HINT)
            self.gui_coupled = g.add_checkbox("Coupled joints", "Coupled" in glim.SIMULATOR.allowed_modules,
                                              hint=lui.COUPLED_HINT)
            self.limit_bools: Dict[str, Any] = {}
            for key, label, hint in lui.BOOL_FIELDS:
                self.limit_bools[key] = g.add_checkbox(label, bool(getattr(glim.SIMULATOR, key)), hint=hint)
            self.md_limits = g.add_markdown("")

        with g.add_folder("Viability"):
            self.check_boxes: Dict[str, Any] = {}
            for c in ck.CHECKS:
                self.check_boxes[c.key] = g.add_checkbox(check_label(c.key, None), True,
                                                         hint=c.why + " Untick to stop Random requiring it.")
                if c.key == "reach":
                    self.gui_show_reach = g.add_checkbox("show cube and reach", True,
                                                         hint="Blue sphere: the cube's spawn point and size; wire "
                                                              "sphere: 5 cm reach; tips green if they reach it.")

        with g.add_folder("Commercial hand"):
            hands = com.list_hands()
            self._hand_ids = {h.label: h.id for h in hands}
            self.gui_hand = g.add_dropdown("Hand", [NO_HAND] + [h.label for h in hands], initial_value=NO_HAND,
                                           hint="A real hand's grammar projection, over its URDF meshes.")
            self.gui_meshes = g.add_checkbox("Show real meshes", True)
            self.md_fidelity = g.add_markdown("")

        with g.add_folder("Mutation"):
            btns = g.add_button_group("Mutate", ["Random mutation", "Back"])
            self.md_mut = g.add_markdown("")
            self.op_buttons: Dict[str, Any] = {}
            for op in EVOLUTION_OPERATORS:
                self.op_buttons[op] = g.add_button(OPERATOR_INFO[op][0], hint=f"{op}: {OPERATOR_INFO[op][1]}")

        with g.add_folder("Pose"):
            self.gui_curl = g.add_slider("Curl", 0.0, 1.0, 0.01, RESET_CURL,
                                         hint="Fraction of every joint's range: 0 lower limits, 1 upper limits, "
                                              f"{RESET_CURL} the pose every episode starts from.")

        # ---- callbacks ---------------------------------------------------
        btn_random.on_click(lambda _: self.random(self._variant_by_label[self.gui_variant.value]))

        @self.gui_variant.on_update
        def _(_):
            v = self._variant_by_label[self.gui_variant.value]
            self.gui_variant.hint = VARIANT_NOTES.get(v, "")
            self.random(v)

        self.gui_preset.on_update(lambda _: self._on_preset())
        for h in list(self.limit_ints.values()) + list(self.limit_bools.values()) + [self.gui_joint_types,
                                                                                      self.gui_coupled]:
            h.on_update(lambda _: self._on_field())

        @self.gui_hand.on_update
        def _(_):
            if not self._suppress and self.gui_hand.value != NO_HAND:
                self.load_commercial(self._hand_ids[self.gui_hand.value])

        self.gui_meshes.on_update(lambda _: self.overlay.set_visible(bool(self.gui_meshes.value)))

        for cb in self.check_boxes.values():
            cb.on_update(lambda _: self._update_status())

        @self.gui_show_reach.on_update
        def _(_):
            for h in self.spawn_handles:
                h.visible = bool(self.gui_show_reach.value)
            with self.lock:
                self._render_pose()

        @btns.on_click
        def _(event):
            (self.mutate if event.target.value == "Random mutation" else self.back)()

        for op, b in self.op_buttons.items():
            b.on_click(lambda _, op=op: self.mutate(op))

        @self.gui_curl.on_update
        def _(_):
            with self.lock:
                self._render_pose()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--variant", default="G_V3S", choices=src.variant_names())
    args = ap.parse_args(argv)
    server = viser.ViserServer(host=args.host, port=args.port, label="grammar viewer")
    EssentialViewer(server, variant=args.variant)
    print(f"grammar viewer at http://{args.host}:{args.port}", flush=True)
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
