"""Interactive viser viewer for the hand-kinematics grammar.

    .venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1

Then open http://127.0.0.1:8080. See README.md in this folder for the panels.
CPU only: no Isaac, Kit or GPU is touched (the simulator's envelope oracle is
loaded by file path, see gviewer/envload.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(HERE), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import viser  # noqa: E402

from hand_sampler.grammar.derive import Derivation, derivation_to_dict, derive  # noqa: E402
from hand_sampler.grammar.kinematics import KinematicModel  # noqa: E402

from gviewer import analysis as an  # noqa: E402
from gviewer import commercial as com  # noqa: E402
from gviewer import export as gexp  # noqa: E402
from gviewer import history as hist  # noqa: E402
from gviewer import meshes as gmesh  # noqa: E402
from gviewer import model as gm  # noqa: E402
from gviewer import sources as src  # noqa: E402
from gviewer.envload import load_env_modules  # noqa: E402
from gviewer.scene import (  # noqa: E402
    SPAWN_RGB,
    TIP_MISS_RGB,
    TIP_REACH_RGB,
    HandRenderer,
    MeshOverlay,
    RenderOptions,
)

VIABLE_RGB = (60, 180, 75)
ADMITTED_NO_REACH_RGB = (240, 170, 40)
PHYS_REJECT_RGB = (220, 60, 60)
STRUCT_REJECT_RGB = (150, 150, 150)
GHOST_RGB = (170, 170, 185)

ORIENT_ROOT = "root frame"
ORIENT_PALM_UP = "palm-up (env base_rot)"
OVERLAP_CURRENT = "at the current pose"
OVERLAP_ORACLE = "oracle poses (q=0 + reset)"
OVERLAP_OFF = "off"


@dataclass
class Current:
    derivation: Derivation
    model: KinematicModel
    label: str
    kind: str                       # "sampled" | "commercial" | "file" | "mutant"
    variant: Optional[str] = None
    seed: Optional[int] = None
    commercial: Optional[com.CommercialHand] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prepared:
    cur: Current
    analysis: an.Analysis
    view: gm.ModelView
    cells: Dict[str, gm.CellMesh]


def viability_colour(report: dict) -> Tuple[int, int, int]:
    if src.is_viable_report(report):
        return VIABLE_RGB
    if report.get("admitted"):
        return ADMITTED_NO_REACH_RGB
    if report.get("digit_count") is None:
        return STRUCT_REJECT_RGB
    return PHYS_REJECT_RGB


def viability_word(report: dict) -> str:
    if src.is_viable_report(report):
        return f"viable ({report['fingertips_reachable']} tips)"
    if report.get("admitted"):
        return f"admitted, {report.get('fingertips_reachable')} tip(s) reach"
    if report.get("digit_count") is None:
        return "structural reject"
    return "rejected (overlap/spawn)"


def shortest_rotation_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return np.array([0.0, *perp])
    q = np.array([1.0 + c, *v])
    return q / np.linalg.norm(q)


def quat_to_mat(wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fmt_mm(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{nd}f} mm"


class GrammarViewer:
    """All state and callbacks. GUI callbacks call the public methods below,
    which tests and the smoke check can also call directly."""

    def __init__(self, server: viser.ViserServer, *, variant: str = "G_V3S", seed: int = 0,
                 until_viable: bool = True, out_dir: Path = gexp.DEFAULT_OUT_DIR, build_initial: bool = True):
        self.server = server
        self.out_dir = Path(out_dir)
        self.lock = threading.RLock()
        self.rng = np.random.default_rng(20261005)
        self.history = hist.History()
        self.cur: Optional[Current] = None
        self.prep: Optional[Prepared] = None
        self.u: Dict[str, float] = {}
        self.parent_view: Optional[gm.ModelView] = None
        self.commercial_cache: Dict[str, com.CommercialHand] = {}
        self.mesh_cache: Dict[str, gmesh.MeshSet] = {}
        self.refs: Optional[Dict[str, KinematicModel]] = None
        self.refs_lock = threading.Lock()
        self.loaded_file: Optional[src.LoadedFile] = None
        self.sliders: Dict[str, Any] = {}
        self._suppress = False
        self._job: Optional[threading.Thread] = None
        self._job_gate = threading.Lock()
        self._job_running = False
        self._job_name = ""
        self._status_text = ""
        self._cancel = threading.Event()
        self._anim_token = 0
        self._anim_thread: Optional[threading.Thread] = None
        self._nearest_token = 0
        self.gallery: List[HandRenderer] = []
        self.gallery_labels: List[Any] = []
        self.spread: List[HandRenderer] = []
        self.spread_labels: List[Any] = []
        self.spread_children: List[Tuple[Derivation, str, dict]] = []
        self.gallery_items: List[Tuple[Derivation, KinematicModel, str, Optional[int]]] = []
        self.mut_variant = variant
        self.last_messages: List[str] = []

        load_env_modules()
        server.scene.set_up_direction("+z")
        server.scene.add_grid("/grid", width=1.0, height=1.0, cell_size=0.05, plane="xy",
                              position=(0.0, 0.0, -0.002))
        self.renderer = HandRenderer(server, "/hand")
        self.ghost = HandRenderer(server, "/ghost")
        self.overlay = MeshOverlay(server, "/hand/urdf")
        self.spawn_handles: List[Any] = []
        self._build_gui(variant, seed, until_viable)
        threading.Thread(target=self._load_refs, daemon=True).start()
        if build_initial:
            self.sample(variant, seed, until_viable=until_viable, wait=True)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _status(self, text: Optional[str] = None) -> None:
        if text is not None:
            self._status_text = text
        cur = f"**{self.cur.label}**" if self.cur else "no design"
        busy = f"  \n_running: {self._job_name}_" if self._job_running else ""
        self.md_status.content = f"{cur}{busy}  \n{self._status_text}"

    def _notify(self, title: str, body: str, color: str = "yellow") -> None:
        self.last_messages.append(f"{title}: {body}")
        for client in self.server.get_clients().values():
            try:
                client.add_notification(title, body, auto_close_seconds=6.0, color=color)
            except Exception:  # noqa: BLE001
                pass

    def busy(self) -> bool:
        return self._job_running

    def run_job(self, name: str, fn: Callable[[], Any], wait: bool = False) -> bool:
        """Run `fn` on a worker thread (one job at a time)."""
        with self._job_gate:
            if self._job_running:
                self._notify("Busy", f"'{self._job_name}' is still running; cancel it first.")
                return False
            self._job_running = True
        self._cancel.clear()
        self._job_name = name
        start_text = f"{name} ..."
        self._status(start_text)

        def body():
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._notify("Error", f"{name}: {type(exc).__name__}: {exc}", color="red")
                self._status_text = f"error in {name}: {type(exc).__name__}: {exc}"
            finally:
                self._job_running = False
                if self._status_text == start_text:
                    self._status_text = f"done: {name}"
                self._status()

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

    def cancel(self) -> None:
        self._cancel.set()

    # ------------------------------------------------------------------
    # Preparing and showing a design
    # ------------------------------------------------------------------

    def _prepare(self, cur: Current) -> Prepared:
        a = an.analyze(cur.model, with_fit_without_overlap=cur.kind == "commercial")
        normal = np.asarray(a.pu.normal, dtype=float) if a.pu is not None else None
        view = gm.ModelView.build(cur.model, palm_normal=normal)
        cells = gm.palm_cells(cur.model)
        return Prepared(cur=cur, analysis=a, view=view, cells=cells)

    def _render_opts(self, **over) -> RenderOptions:
        o = RenderOptions(
            capsules=self.gui_show_caps.value, palm_cells=self.gui_show_cells.value,
            palm_spines=self.gui_show_spines.value, joint_axes=self.gui_show_axes.value,
            tips=self.gui_show_tips.value, body_labels=self.gui_lbl_bodies.value,
            joint_labels=self.gui_lbl_joints.value, root_frame=self.gui_show_root.value,
            capsule_mode=self.gui_capsule_mode.value,
        )
        for k, v in over.items():
            setattr(o, k, v)
        return o

    def _apply(self, prep: Prepared, history_mode: str, operator: Optional[str] = None, note: str = "") -> None:
        with self.lock:
            self.prep, self.cur = prep, prep.cur
            if history_mode == "reset":
                self.history.reset(prep.cur.derivation, prep.cur.label)
                self.history.current.payload = prep.cur
            elif history_mode == "push":
                e = self.history.push(prep.cur.derivation, operator or "?", note)
                e.payload = prep.cur
            self.u = gm.clamp_u(prep.view.ranges, {})
            self._clear_spread()
            if self.gallery:
                self._clear_gallery()
            self.renderer.set_visible(True)
            self.ghost.set_visible(True)
            self.renderer.build(prep.view, prep.cells, self._render_opts())
            self._rebuild_sliders()
            self._build_ghost()
            self._build_spawn()
            self._build_overlay()
            self._update_orientation()
            self._render_pose()
            self._update_panels()
        self._start_nearest()

    def show(self, cur: Current, history_mode: str = "reset", operator: Optional[str] = None, note: str = "") -> Prepared:
        prep = self._prepare(cur)
        self._apply(prep, history_mode, operator, note)
        return prep

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def sample(self, variant: str, seed: int, until_viable: bool = False, step: int = 1,
               max_tries: Optional[int] = None, wait: bool = False) -> bool:
        max_tries = int(max_tries if max_tries is not None else self.gui_max_tries.value)

        def job():
            if until_viable:
                def progress(tries, s):
                    if tries % 10 == 0:
                        self.md_sample.content = f"searching {variant}: {tries} tries, at seed {s} ..."
                res = src.sample_until_viable(variant, seed, step=step, max_tries=max_tries,
                                              cancel=self._cancel.is_set, progress=progress)
                if res.derivation is None:
                    msg = ("cancelled" if res.cancelled else
                           f"no viable design in {res.tries} tries (seeds {res.seeds_tried[0]}..{res.seeds_tried[1]})")
                    self.md_sample.content = msg
                    self._notify("Sample until viable", msg)
                    return
                d, m, s = res.derivation, res.model, res.seed
                self.md_sample.content = (f"seed **{s}** is viable after **{res.tries}** tries "
                                          f"(seeds {res.seeds_tried[0]}..{res.seeds_tried[1]})")
            else:
                d, m = src.sample(variant, seed)
                s = seed
                self.md_sample.content = f"seed **{s}** (1 draw)"
            self._set_seed_field(s)
            cur = Current(derivation=d, model=m, label=f"{variant} seed {s}", kind="sampled", variant=variant, seed=s)
            self.mut_variant = variant
            self._set_dropdown(self.gui_mut_dist, src.variant_label(variant))
            self.show(cur, "reset")

        return self.run_job(f"sample {variant}", job, wait=wait)

    def load_commercial(self, hand_id: str, wait: bool = False) -> bool:
        def job():
            ch = self.commercial_cache.get(hand_id)
            if ch is None:
                ch = com.load_commercial(hand_id)
                self.commercial_cache[hand_id] = ch
            if not ch.ok:
                self.md_commercial.content = f"**{hand_id}**: {ch.error}"
                self._notify("Commercial hand", f"{hand_id}: {ch.error}")
                return
            if hand_id not in self.mesh_cache:
                links = [b.name for b in ch.imported.model.bodies]
                self.mesh_cache[hand_id] = gmesh.load_link_meshes(ch.entry.mesh_path, links,
                                                                  fallback_dirs=self._mesh_fallbacks(ch))
            cur = Current(derivation=ch.projection.derivation, model=ch.derived, label=f"projected {hand_id}",
                          kind="commercial", commercial=ch)
            self.show(cur, "reset")

        return self.run_job(f"load {hand_id}", job, wait=wait)

    def _mesh_fallbacks(self, ch: com.CommercialHand) -> List[Path]:
        """Download folders where a hand's meshes live when its URDF's own
        relative paths do not resolve (karma-hand-metric copies of SVH and
        Shadow reference meshes that are not next to them)."""
        manifest = com.load_manifest()
        root = Path(manifest.get("source_root") or "/")
        table = {"svh_right": [root / "SVH"], "shadow_right_local": [root / "Shadow"]}
        return [p for p in table.get(ch.entry.id, []) if p.is_dir()]

    def load_file(self, path: str, wait: bool = False) -> bool:
        def job():
            lf = src.load_file(path)
            self._set_loaded_file(lf)

        return self.run_job("load file", job, wait=wait)

    def load_document(self, doc: Any, name: str) -> None:
        self._set_loaded_file(src.parse_document(doc, name))

    def _set_loaded_file(self, lf: src.LoadedFile) -> None:
        self.loaded_file = lf
        n = len(lf.entries)
        err = f"  \n{len(lf.errors)} entr(ies) failed: {lf.errors[0]}" if lf.errors else ""
        self.md_file.content = f"**{Path(lf.path).name}**: {lf.kind}, {n} design(s){err}"
        if lf.variant and lf.variant in src.NAMED_DISTRIBUTIONS:
            self.mut_variant = lf.variant
            self._set_dropdown(self.gui_mut_dist, src.variant_label(lf.variant))
        if n == 0:
            return
        self._suppress = True
        try:
            self.gui_file_index.max = max(n - 1, 1)
            self.gui_file_index.value = 0
        finally:
            self._suppress = False
        self.show_file_entry(0)

    def show_file_entry(self, index: int) -> None:
        lf = self.loaded_file
        if lf is None or not lf.entries:
            return
        index = int(max(0, min(index, len(lf.entries) - 1)))
        e = lf.entries[index]
        model = derive(e.derivation)
        meta = " ".join(f"{k}={v}" for k, v in e.meta.items() if v is not None)
        self.md_file.content = (f"**{Path(lf.path).name}**: {lf.kind}  \n"
                                f"entry **{index}** / {len(lf.entries) - 1}: {e.label}  \n{meta}")
        cur = Current(derivation=e.derivation, model=model, label=f"{Path(lf.path).name}[{index}]", kind="file",
                      variant=lf.variant, meta=dict(e.meta))
        self.show(cur, "reset")

    # ------------------------------------------------------------------
    # Mutation and history
    # ------------------------------------------------------------------

    def mutate(self, operator: Optional[str] = None, wait: bool = False) -> bool:
        def job():
            cur = self.cur
            if cur is None:
                return
            res = src.mutate(cur.derivation, src.distribution(self.mut_variant), self.rng, operator)
            if res.child is None:
                self.md_mut_status.content = res.message
                self._notify("Mutation", res.message)
                return
            self._show_child(cur, res.child, res.operator, "")
            self.md_mut_status.content = f"{res.message} ({self.mut_variant})"

        return self.run_job(f"mutate {operator or 'random'}", job, wait=wait)

    def mutate_until_viable(self, max_tries: Optional[int] = None, wait: bool = False) -> bool:
        max_tries = int(max_tries if max_tries is not None else self.gui_mut_tries.value)

        def job():
            cur = self.cur
            if cur is None:
                return
            res = src.mutate_until_viable(cur.derivation, src.distribution(self.mut_variant), self.rng,
                                          max_tries=max_tries, cancel=self._cancel.is_set)
            self.md_mut_status.content = res.message
            if res.child is None:
                self._notify("Mutate until viable", res.message)
                return
            self._show_child(cur, res.child, res.operator, f"viable after {res.tries} tries")

        return self.run_job("mutate until viable", job, wait=wait)

    def _show_child(self, parent: Current, child: Derivation, operator: str, note: str) -> None:
        model = derive(child)
        depth = len(child.lineage)
        base = parent.label.split(" +")[0]
        cur = Current(derivation=child, model=model, label=f"{base} +{depth} ({operator})", kind="mutant",
                      variant=self.mut_variant, seed=parent.seed)
        self.show(cur, "push", operator=operator, note=note)

    def back(self, wait: bool = False) -> bool:
        return self._navigate(self.history.back, wait)

    def forward(self, wait: bool = False) -> bool:
        return self._navigate(self.history.forward, wait)

    def jump(self, index: int, wait: bool = False) -> bool:
        return self._navigate(lambda: self.history.jump(index), wait)

    def _navigate(self, move: Callable, wait: bool) -> bool:
        def job():
            before = self.history.cursor
            e = move()
            if e is None or self.history.cursor == before:
                return
            cur = getattr(e, "payload", None)
            if cur is None:
                d = e.derivation
                cur = Current(derivation=d, model=derive(d), label=e.label, kind="mutant", variant=self.mut_variant)
            self.show(cur, "keep")

        return self.run_job("history", job, wait=wait)

    # ------------------------------------------------------------------
    # Pose
    # ------------------------------------------------------------------

    def set_u(self, u: Dict[str, float]) -> None:
        with self.lock:
            if self.prep is None:
                return
            self.u = gm.clamp_u(self.prep.view.ranges, u)
            self._sync_sliders()
            self._render_pose()

    def pose_zero(self) -> None:
        self.set_u({})

    def pose_reset(self) -> None:
        if self.prep is None:
            return
        ru = self.prep.analysis.reset_u()
        if not ru:
            self._notify("Reset pose", "no envelope design: the env reset pose (palm_up default_q) is undefined; "
                         "showing 35% curl instead")
            ru = gm.curl_u(self.prep.view.ranges, an.CURL_FRAC)
        self.set_u(ru)

    def pose_random(self) -> None:
        if self.prep is not None:
            self.set_u(gm.random_u(self.prep.view.ranges, self.rng))

    def pose_curl(self, frac: float) -> None:
        if self.prep is not None:
            self.set_u(gm.curl_u(self.prep.view.ranges, float(frac)))

    def set_animating(self, on: bool) -> None:
        self._anim_token += 1
        if not on:
            with self.lock:
                self._sync_sliders()
            return
        token = self._anim_token
        base = dict(self.u)

        def loop():
            t0 = time.time()
            while self._anim_token == token:
                period = max(float(self.gui_anim_period.value), 0.2)
                t = time.time() - t0
                with self.lock:
                    if self.prep is None:
                        break
                    ranges = self.prep.view.ranges
                    if self.gui_anim_mode.value == "one joint at a time" and ranges:
                        k = int(t / period) % len(ranges)
                        f = 0.5 - 0.5 * math.cos(2 * math.pi * ((t % period) / period))
                        u = gm.clamp_u(ranges, base)
                        u[ranges[k].name] = ranges[k].at_fraction(f)
                        self.md_anim.content = f"sweeping `{ranges[k].name}`"
                    else:
                        f = 0.5 - 0.5 * math.cos(2 * math.pi * t / period)
                        u = gm.curl_u(ranges, f)
                        self.md_anim.content = f"curl fraction {f:.2f}"
                    self.u = u
                    self._render_pose()
                time.sleep(1.0 / 30.0)

        self._anim_thread = threading.Thread(target=loop, daemon=True)
        self._anim_thread.start()

    # ------------------------------------------------------------------
    # Rendering helpers (call with self.lock held)
    # ------------------------------------------------------------------

    def _q_full(self) -> Dict[str, float]:
        return gm.expand_q(self.prep.view.model, self.u)

    def _highlight(self) -> Tuple[set, List[Tuple[str, str, float]]]:
        a = self.prep.analysis
        mode = self.gui_overlap_mode.value
        if a.design is None or mode == OVERLAP_OFF:
            return set(), []
        if mode == OVERLAP_CURRENT:
            pairs = an.overlaps_at(a.design, self._q_full())
        else:
            pairs = a.pairs_q0 + a.pairs_reset
        return {b for p in pairs for b in p[:2]}, pairs

    def _render_pose(self) -> None:
        if self.prep is None:
            return
        view = self.prep.view
        prims = view.primitives(self.u)
        hl, pairs = self._highlight()
        tip_cols = None
        if self.prep.analysis.finger_reach:
            tip_cols = {r.tip_body: (TIP_REACH_RGB if r.reaches else TIP_MISS_RGB)
                        for r in self.prep.analysis.finger_reach}
        with self.server.atomic():
            self.renderer.update(prims, hl, tip_cols)
            if self.parent_view is not None and self.gui_ghost.value:
                pu = gm.clamp_u(self.parent_view.ranges, self.u)
                self.ghost.update(self.parent_view.primitives(pu))
            if self.cur is not None and self.cur.commercial is not None and self.overlay.handles:
                self.overlay.update(self.cur.commercial.orig_link_poses(self.u))
        if self.gui_overlap_mode.value == OVERLAP_CURRENT and self.prep.analysis.design is not None:
            worst = max((p[2] for p in pairs), default=0.0)
            self.md_live.content = (f"current pose: {len(pairs)} overlapping capsule pair(s), worst "
                                    f"{worst * 1000:.2f} mm" + ("" if not pairs else "  \n" + ", ".join(
                                        f"{a}/{b} {pen * 1000:.1f}" for a, b, pen in sorted(pairs, key=lambda p: -p[2])[:6])))
        elif self.prep.analysis.design is None:
            self.md_live.content = "no envelope design: overlaps not checked"
        else:
            self.md_live.content = ""

    def _rebuild_sliders(self) -> None:
        for h, _scale in self.sliders.values():
            h.remove()
        self.sliders = {}
        view = self.prep.view
        self.md_joints.content = (f"{len(view.ranges)} independent joint(s); kind is the rest-pose "
                                  f"class (fle/abd/twi/pal)" if view.ranges else "no movable joints")
        with self.joint_folder:
            for r in view.ranges:
                kind = view.joint_classes[r.name].kind[:3]
                if r.type == "prismatic":
                    lo, hi, scale, step = r.lo * 1000, r.hi * 1000, 1000.0, 0.5
                    label = f"{r.name} [{kind}] mm"
                else:
                    lo, hi, scale, step = math.degrees(r.lo), math.degrees(r.hi), 180.0 / math.pi, 0.5
                    label = f"{r.name} [{kind}]"
                disabled = hi - lo < 1e-6
                if disabled:
                    hi = lo + 1.0
                init = float(np.clip(self.u.get(r.name, 0.0) * scale, lo, hi))
                h = self.server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=init,
                                               disabled=disabled)

                def on_change(_, name=r.name, sc=scale, handle=h):
                    if self._suppress:
                        return
                    with self.lock:
                        self.u[name] = float(handle.value) / sc
                        self._render_pose()

                h.on_update(on_change)
                self.sliders[r.name] = (h, scale)

    def _sync_sliders(self) -> None:
        self._suppress = True
        try:
            for name, (h, scale) in self.sliders.items():
                h.value = float(np.clip(self.u.get(name, 0.0) * scale, h.min, h.max))
        finally:
            self._suppress = False

    def _build_ghost(self) -> None:
        self.ghost.clear()
        self.parent_view = None
        parent = self.history.parent
        if parent is None or not self.gui_ghost.value:
            return
        pm = derive(parent.derivation)
        self.parent_view = gm.ModelView.build(pm, palm_normal=self.prep.view.palm_normal)
        cells = gm.palm_cells(pm) if self.gui_show_cells.value else None
        self.ghost.build(self.parent_view, cells, self._render_opts(
            tint=GHOST_RGB, opacity=0.3, cell_opacity=0.12, joint_axes=False, tips=False, body_labels=False,
            joint_labels=False, root_frame=False))

    def _build_spawn(self) -> None:
        for h in self.spawn_handles:
            h.remove()
        self.spawn_handles = []
        a = self.prep.analysis
        if a.pu is None:
            return
        sp = a.spawn_point
        vis = bool(self.gui_show_spawn.value)
        s = self.server.scene
        self.spawn_handles.append(s.add_icosphere("/hand/spawn/object", radius=an.OBJECT_HALF_SIZE_M,
                                                  color=SPAWN_RGB, opacity=0.35, position=sp, visible=vis))
        self.spawn_handles.append(s.add_icosphere("/hand/spawn/reach", radius=an.REACH_TOL_M, color=SPAWN_RGB,
                                                  wireframe=True, opacity=0.25, position=sp, visible=vis,
                                                  subdivisions=2))
        self.spawn_handles.append(s.add_label("/hand/spawn/label", "spawn", position=sp, visible=vis,
                                              font_screen_scale=0.7))

    def _build_overlay(self) -> None:
        self.overlay.clear()
        cur = self.cur
        if cur is None or cur.commercial is None:
            self.md_meshes.content = ""
            return
        ms = self.mesh_cache.get(cur.commercial.entry.id)
        if ms is None:
            return
        self.overlay.build(ms, opacity=float(self.gui_mesh_opacity.value), visible=bool(self.gui_meshes.value))
        self.md_meshes.content = "meshes: " + ms.summary()

    def _palm_up_wxyz(self) -> np.ndarray:
        a = self.prep.analysis
        if a.pu is not None:
            return np.asarray(a.pu.base_rot_wxyz, dtype=float)
        return shortest_rotation_wxyz(self.prep.view.palm_normal, np.array([0.0, 0.0, 1.0]))

    def _orientation(self) -> np.ndarray:
        if self.prep is None or self.gui_orient.value == ORIENT_ROOT:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return self._palm_up_wxyz()

    def _update_orientation(self) -> None:
        wxyz = self._orientation()
        self.renderer.set_frame_pose(wxyz)
        self.ghost.set_frame_pose(wxyz)

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def analysis_markdown(self) -> str:
        a = self.prep.analysis
        r = a.report
        st = a.structure
        L = []
        if r["admitted"]:
            head = "ADMITTED"
        else:
            head = "NOT ADMITTED"
        L.append(f"**Viability** (`viability_report`): **{head}**; viable (>= 2 tips reach): "
                 f"**{'yes' if a.viable else 'no'}**")
        for reason in r["reasons"]:
            L.append(f"- {reason}")
        if r["digit_count"] is not None:
            ge = load_env_modules().grammar_envelope
            L.append(f"- max rest overlap {r['max_rest_overlap_mm']:.2f} mm (gate "
                     f"{ge.MAX_REST_PENETRATION_M * 1000:.1f} mm, worse of q=0 and reset pose)")
            L.append(f"- fingertips reaching the spawn: **{r['fingertips_reachable']}** of {len(a.finger_reach)}")
            for fr in a.finger_reach:
                L.append(f"  - f{fr.slot} digit {fr.digit_id} ({fr.tip_body}): {fr.hits}/{an.N_SWEEP} sweep samples "
                         f"within {an.REACH_TOL_M * 1000:.0f} mm, closest {fr.min_dist_m * 1000:.0f} mm")
            mn = load_env_modules().palm_calibration.MIN_SPAWN_HEIGHT_ABOVE_PALM_M
            L.append(f"- spawn height above palm {r['spawn_height_mm']:.1f} mm (min {mn * 1000:.0f} mm)")
            for title, pairs in (("q=0", a.pairs_q0), ("reset pose", a.pairs_reset)):
                if pairs:
                    L.append(f"- overlaps at {title}: " + ", ".join(
                        f"{x}/{y} {p * 1000:.1f} mm" for x, y, p in sorted(pairs, key=lambda t: -t[2])[:8]))
                else:
                    L.append(f"- overlaps at {title}: none")
        ar = load_env_modules().archive
        d_bin, j_bin = a.descriptor_cell
        lo, hi = ar.JOINT_BIN_EDGES[j_bin]
        note = f" ({a.descriptor_note})" if a.descriptor_note else ""
        L.append(f"**MAP-Elites cell** `{a.descriptor_label}`: digits bin {ar.DIGIT_BINS[d_bin]}, joints bin "
                 f"{lo}-{hi}{note}")
        if a.design is not None:
            L.append("**Envelope slots**: " + "; ".join(an.slot_table(a.design)))
        rad = st.capsule_radius_m
        L.append(f"**Counts**: {len(st.top_level_digits)} digit(s) + {len(st.branch_digits)} branch; phalanges "
                 f"{st.phalanges_per_digit()}; palm bodies {len(st.palm_bodies)} "
                 f"({len(st.jointed_palm_bodies)} jointed); joints {st.n_movable} movable, {st.n_independent} "
                 f"motors, {st.n_couplings} coupling(s); capsule radius "
                 f"{'n/a' if rad is None else f'{rad * 1000:.1f} mm'}")
        return "  \n".join(L)

    def commercial_markdown(self, ch: com.CommercialHand) -> str:
        e = ch.entry
        L = [f"**{e.id}** ({e.split}, {e.family})"]
        sha = {True: "sha256 matches manifest", False: "sha256 MISMATCH", None: "no sha256 in manifest"}[ch.sha_ok]
        L.append(f"URDF: `{e.kin_path}` ({sha})")
        f = ch.fidelity
        if f:
            tip = "n/a (no fingertip links)" if f["max_tip_mm"] is None else f"{f['max_tip_mm']:.2e} mm"
            L.append(f"**Fidelity** (E13 replay, {f['n_configs']} configs, seed {f['seed']}): joint position "
                     f"max {f['max_pos_mm']:.2e} mm, axis max {f['max_axis_deg']:.2e} deg, fingertip {tip}; "
                     f"movable joints {f['n_movable_orig']} -> {f['n_movable_der']}; motors "
                     f"{f['n_independent_orig']} -> {f['n_independent_der']}; "
                     f"**{'PASS' if f['passed'] else 'FAIL'}** at 5 mm / 10 deg. Pinocchio export check not run here.")
        ok, reasons = ch.fit
        if ok:
            L.append("**32-slot simulator envelope**: fits (admit with check_overlap=False, the projected-hand rule)")
        else:
            L.append("**32-slot simulator envelope**: does NOT fit: " + "; ".join(reasons))
        L.append("Rest overlaps of projected hands are exempt in the simulator (filtered pairs); the Analysis "
                 "tab shows the full oracle, which does count them.")
        r = ch.capsule_radius_m
        L.append(f"**Capsule radius**: {r * 1000:.1f} mm" if r is not None else "**Capsule radius**: n/a")
        L.append("**Dropped or approximated**:")
        L.extend(f"- {line}" for line in ch.approximations())
        if e.notes:
            L.append(f"_manifest note_: {e.notes}")
        return "  \n".join(L)

    def _update_panels(self) -> None:
        cur = self.cur
        self.md_analysis.content = self.analysis_markdown()
        if cur.commercial is not None:
            self.md_commercial.content = self.commercial_markdown(cur.commercial)
        self.md_lineage.content = "**History** (back/forward; mutating from an earlier entry drops later ones)  \n" + \
            "  \n".join(self.history.lineage_lines())
        parent = self.history.parent
        if parent is not None:
            d = hist.diff(parent.derivation, self.history.current.derivation)
            self.md_diff.content = "**Parent -> current**  \n" + "  \n".join(d.lines())
        else:
            self.md_diff.content = "**Parent -> current**: (root of the history)"
        self.md_nearest.content = "_nearest commercial: computing ..._"
        self._status(f"{cur.kind}; mutation distribution {self.mut_variant}")

    # ------------------------------------------------------------------
    # Nearest commercial hand
    # ------------------------------------------------------------------

    def _load_refs(self) -> None:
        with self.refs_lock:
            if self.refs is not None:
                return
            refs = {}
            for h in com.list_hands():
                if h.availability != "available":
                    continue
                try:
                    ch = com.load_commercial(h.id, with_fidelity=False)
                except Exception:  # noqa: BLE001
                    continue
                if ch.ok:
                    refs[h.id] = ch.derived
            self.refs = refs

    def _start_nearest(self) -> None:
        self._nearest_token += 1
        token = self._nearest_token
        model = self.cur.model
        own = self.cur.commercial.entry.id if self.cur.commercial is not None else None

        def run():
            self._load_refs()
            refs = {k: v for k, v in (self.refs or {}).items() if k != own}
            res = an.nearest_commercial(model, refs, n_configs=16)
            if token != self._nearest_token:
                return
            rows = [f"{hid} {d * 1000:.0f} mm" for hid, d, _ in res[:5]]
            self.md_nearest.content = ("**Nearest commercial** (phenodist tip displacement, 16 configs of this "
                                       "design, joints aligned by grammar name): " + "; ".join(rows))

        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------------
    # Gallery and mutation spread
    # ------------------------------------------------------------------

    def _clear_gallery(self) -> None:
        for r in self.gallery:
            r.remove()
        for h in self.gallery_labels:
            h.remove()
        self.gallery, self.gallery_labels, self.gallery_items = [], [], []

    def _clear_spread(self) -> None:
        for r in self.spread:
            r.remove()
        for h in self.spread_labels:
            h.remove()
        self.spread, self.spread_labels, self.spread_children = [], [], []

    def _mini_hand(self, root: str, model: KinematicModel, report: dict, colour_by_viability: bool,
                   with_cells: bool, pose: str, position, on_click) -> HandRenderer:
        ge = load_env_modules().grammar_envelope
        pu = None
        if report.get("digit_count") is not None:
            design = ge.canonicalize(model)
            pu = ge.palm_up(design, n_sweep=0)
        view = gm.ModelView.build(model, palm_normal=None if pu is None else pu.normal)
        cells = gm.palm_cells(model) if with_cells else None
        r = HandRenderer(self.server, root)
        tint = viability_colour(report) if colour_by_viability else None
        r.build(view, cells, RenderOptions(tint=tint, joint_axes=False, tips=False, root_frame=False,
                                           capsule_mode=self.gui_capsule_mode.value, cell_opacity=0.3),
                on_click=on_click)
        u = {}
        if pose == "reset" and pu is not None:
            u = an.slot_q_to_model(design, pu.default_q)
        r.update(view.primitives(u))
        if self.gui_orient.value == ORIENT_PALM_UP:
            wxyz = pu.base_rot_wxyz if pu is not None else shortest_rotation_wxyz(view.palm_normal, np.array([0, 0, 1.0]))
        else:
            wxyz = (1.0, 0.0, 0.0, 0.0)
        r.set_frame_pose(wxyz, position)
        return r

    def build_gallery(self, variant: str, compare: Optional[str], rows: int, cols: int, start_seed: int,
                      spacing: float, wait: bool = False) -> bool:
        def job():
            ge = load_env_modules().grammar_envelope
            self._clear_gallery()
            self._clear_spread()
            self.renderer.set_visible(False)
            self.ghost.set_visible(False)
            plan = []
            if compare:
                for ri, v in enumerate((variant, compare)):
                    for c in range(cols):
                        plan.append((ri, c, v, start_seed + c))
            else:
                for i in range(rows * cols):
                    plan.append((i // cols, i % cols, variant, start_seed + i))
            counts = {}
            for k, (ri, c, v, seed) in enumerate(plan):
                if self._cancel.is_set():
                    break
                d, m = src.sample(v, seed)
                rep = ge.viability_report(m)
                word = viability_word(rep)
                counts.setdefault(v, []).append(src.is_viable_report(rep))
                pos = (c * spacing, -ri * spacing, 0.0)
                idx = len(self.gallery_items)
                self.gallery_items.append((d, m, v, seed))

                def on_click(_event, i=idx):
                    self.adopt_gallery(i)

                r = self._mini_hand(f"/gallery/{k}", m, rep, bool(self.gui_gal_colour.value),
                                    bool(self.gui_gal_cells.value), self.gui_gal_pose.value, pos, on_click)
                self.gallery.append(r)
                self.gallery_labels.append(self.server.scene.add_label(
                    f"/gallery_labels/{k}", f"{v} s{seed}: {word}", position=(pos[0], pos[1] - 0.06, 0.0),
                    font_screen_scale=0.6))
                self.md_gallery.content = f"built {k + 1}/{len(plan)}"
            summary = "; ".join(f"{v}: {sum(f)}/{len(f)} viable" for v, f in counts.items())
            self.md_gallery.content = (f"{summary}. Green viable, amber admitted but < 2 tips reach, red rejected "
                                       f"(overlap/spawn), grey structural reject. Click a hand to open it.")
            self._look_at_points(np.array([[0, 0, 0], [(cols - 1) * spacing, -(max(rows, 2 if compare else rows) - 1) * spacing, 0]]),
                                 "top")

        return self.run_job("gallery", job, wait=wait)

    def clear_gallery(self) -> None:
        with self.lock:
            self._clear_gallery()
            self.renderer.set_visible(True)
            self.ghost.set_visible(True)

    def adopt_gallery(self, index: int) -> None:
        if not (0 <= index < len(self.gallery_items)):
            return
        d, m, v, seed = self.gallery_items[index]

        def job():
            self.clear_gallery()
            self.mut_variant = v
            self._set_seed_field(seed)
            self.show(Current(derivation=d, model=m, label=f"{v} seed {seed}", kind="sampled", variant=v, seed=seed))

        self.run_job("open gallery hand", job)

    def build_spread(self, k: int, radius: float, wait: bool = False) -> bool:
        def job():
            ge = load_env_modules().grammar_envelope
            cur = self.cur
            if cur is None:
                return
            self._clear_spread()
            dist = src.distribution(self.mut_variant)
            n_imp, n_viable, n_adm = 0, 0, 0
            wxyz_main = self._orientation()
            for i in range(k):
                if self._cancel.is_set():
                    break
                res = src.mutate(cur.derivation, dist, self.rng)
                ang = 2 * math.pi * i / k
                pos = np.array([radius * math.cos(ang), radius * math.sin(ang), 0.0])
                if res.child is None:
                    n_imp += 1
                    self.spread_labels.append(self.server.scene.add_label(
                        f"/spread_labels/{i}", f"{res.operator}: inapplicable", position=pos, font_screen_scale=0.6))
                    continue
                m = derive(res.child)
                rep = ge.viability_report(m)
                n_viable += src.is_viable_report(rep)
                n_adm += bool(rep["admitted"])
                idx = len(self.spread_children)
                self.spread_children.append((res.child, res.operator, rep))

                def on_click(_event, j=idx):
                    self.adopt_spread(j)

                r = self._mini_hand(f"/spread/{i}", m, rep, True, False, "zero", pos, on_click)
                # keep the main hand's orientation so children read against the parent
                r.set_frame_pose(wxyz_main, pos)
                self.spread.append(r)
                self.spread_labels.append(self.server.scene.add_label(
                    f"/spread_labels/{i}", f"{res.operator}: {viability_word(rep)}", position=pos - [0, 0, 0.05],
                    font_screen_scale=0.6))
            n_child = len(self.spread_children)
            self.md_spread.content = (f"{k} random `{self.mut_variant}` mutations: {n_child} children, "
                                      f"{n_viable} viable, {n_adm} admitted, {n_imp} inapplicable. "
                                      f"Click a child to adopt it.")

        return self.run_job("mutation spread", job, wait=wait)

    def adopt_spread(self, index: int) -> None:
        if not (0 <= index < len(self.spread_children)):
            return
        child, op, _rep = self.spread_children[index]

        def job():
            cur = self.cur
            self._show_child(cur, child, op, "adopted from spread")

        self.run_job("adopt child", job)

    # ------------------------------------------------------------------
    # Camera and export
    # ------------------------------------------------------------------

    def _world_points(self) -> np.ndarray:
        if self.prep is None:
            return np.zeros((1, 3))
        prims = self.prep.view.primitives(self.u)
        lo, hi = prims.bounds()
        pts = np.array([lo, hi])
        R = quat_to_mat(self._orientation())
        return pts @ R.T

    def _look_at_points(self, pts: np.ndarray, preset: str, target: Optional[np.ndarray] = None) -> None:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        c = (lo + hi) / 2 if target is None else target
        size = max(float(np.linalg.norm(hi - lo)), 0.12)
        d = 1.6 * size
        if preset == "top":
            pos = c + np.array([0.0, -0.02 * d, d])
        elif preset == "side":
            pos = c + np.array([d, 0.0, 0.12 * d])
        else:
            pos = c + np.array([0.45 * d, -0.55 * d, 0.75 * d])
        for client in self.server.get_clients().values():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = tuple(pos)
            client.camera.look_at = tuple(c)

    def camera(self, preset: str) -> None:
        with self.lock:
            if preset == "palm-up" and self.gui_orient.value != ORIENT_PALM_UP:
                self.gui_orient.value = ORIENT_PALM_UP  # on_update re-orients
                self._update_orientation()
            target = None
            if preset == "palm-up" and self.prep is not None and self.prep.analysis.pu is not None:
                target = quat_to_mat(self._orientation()) @ self.prep.analysis.spawn_point
            self._look_at_points(self._world_points(), preset, target)

    def export_json(self) -> Path:
        cur = self.cur
        meta = {"label": cur.label, "kind": cur.kind, "variant": cur.variant, "seed": cur.seed,
                "mutation_distribution": self.mut_variant, "report": self.prep.analysis.report,
                "descriptor": self.prep.analysis.descriptor_label,
                "history": [e.label for e in self.history.entries[: self.history.cursor + 1]]}
        path = gexp.export_derivation(cur.derivation, cur.label, self.out_dir, extra=meta)
        self.md_export.content = f"wrote `{path}` (+ `.meta.json`)"
        self._maybe_download(path)
        return path

    def export_urdf(self) -> Dict[str, Any]:
        out = gexp.export_urdf(self.cur.model, self.cur.label, self.out_dir)
        extra = f", {out['n_meshes']} palm-cell mesh(es)" if "n_meshes" in out else ""
        self.md_export.content = f"wrote `{out['urdf']}`{extra}"
        self._maybe_download(out["urdf"])
        return out

    def export_history(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{gexp.safe_stem(self.cur.label)}.history.json"
        path.write_text(json.dumps(self.history.to_dict(), indent=1))
        self.md_export.content = f"wrote `{path}`"
        self._maybe_download(path)
        return path

    def _maybe_download(self, path: Path) -> None:
        if not self.gui_download.value:
            return
        data = Path(path).read_bytes()
        for client in self.server.get_clients().values():
            client.send_file_download(Path(path).name, data)

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------

    def _set_seed_field(self, seed: int) -> None:
        self._suppress = True
        try:
            self.gui_seed.value = int(seed)
        finally:
            self._suppress = False

    def _set_dropdown(self, handle, value: str) -> None:
        self._suppress = True
        try:
            handle.value = value
        finally:
            self._suppress = False

    def _rebuild_view(self) -> None:
        """Re-create the scene nodes after a display option changed."""
        with self.lock:
            if self.prep is None:
                return
            self.renderer.build(self.prep.view, self.prep.cells, self._render_opts())
            self._build_ghost()
            self._update_orientation()
            self._render_pose()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _build_gui(self, variant: str, seed: int, until_viable: bool) -> None:
        g = self.server.gui
        g.configure_theme(control_width="large")
        self.md_status = g.add_markdown("starting ...")
        btn_cancel = g.add_button("Cancel running job", color="red")
        btn_cancel.on_click(lambda _: self.cancel())
        labels = [src.variant_label(n) for n in src.variant_names()]
        tabs = g.add_tab_group()

        # ---- Source -----------------------------------------------------
        with tabs.add_tab("Source"):
            with g.add_folder("(a) Sample a grammar variant"):
                self.gui_variant = g.add_dropdown("Variant", labels, initial_value=src.variant_label(variant))
                self.gui_seed = g.add_number("Seed", int(seed), min=0, step=1)
                seed_btns = g.add_button_group("Seed step", ["prev", "next", "random"])
                self.gui_until = g.add_checkbox("Sample until viable", bool(until_viable),
                                                hint="draw seeds until viability_report admits a design with >= 2 "
                                                     "fingertips reaching the spawn")
                self.gui_max_tries = g.add_number("Max tries", 500, min=1, max=20000, step=1)
                btn_sample = g.add_button("Sample")
                self.md_sample = g.add_markdown("")
            with g.add_folder("(b) Commercial hand", expand_by_default=False):
                hands = com.list_hands()
                self._hand_ids = {h.label: h.id for h in hands}
                self.gui_hand = g.add_dropdown("Hand", [h.label for h in hands])
                btn_hand = g.add_button("Load hand")
                self.gui_meshes = g.add_checkbox("Show URDF meshes", True)
                self.gui_mesh_opacity = g.add_slider("Mesh opacity", 0.0, 1.0, 0.05, 0.35)
                self.md_meshes = g.add_markdown("")
                self.md_commercial = g.add_markdown("")
            with g.add_folder("(c) Load a file", expand_by_default=False):
                default = REPO_ROOT / "outputs" / "viable" / "mixed32" / "population.json"
                self.gui_path = g.add_text("Path", str(default))
                btn_file = g.add_button("Load file")
                upload = g.add_upload_button("Upload JSON", mime_type="application/json")
                self.gui_file_index = g.add_slider("Index", 0, 1, 1, 0)
                file_btns = g.add_button_group("Entry", ["prev", "next"])
                self.md_file = g.add_markdown("derivation JSON, population.json, driver state.json or archive")

        # ---- Pose ---------------------------------------------------------
        with tabs.add_tab("Pose"):
            self.gui_curl = g.add_slider("Curl all (fraction of range)", 0.0, 1.0, 0.01, 0.0)
            pose_btns = g.add_button_group("Pose", ["zero", "env reset", "random"])
            self.gui_anim = g.add_checkbox("Animate sweep (play/pause)", False)
            self.gui_anim_mode = g.add_dropdown("Sweep", ["all joints together", "one joint at a time"])
            self.gui_anim_period = g.add_slider("Sweep period (s)", 0.5, 10.0, 0.5, 3.0)
            self.md_anim = g.add_markdown("")
            self.joint_folder = g.add_folder("Joints (deg)")
            with self.joint_folder:
                self.md_joints = g.add_markdown("")

        # ---- Mutate -------------------------------------------------------
        with tabs.add_tab("Mutate"):
            self.gui_mut_dist = g.add_dropdown("Mutation distribution", labels, initial_value=src.variant_label(variant))
            btn_rand = g.add_button("Random mutation (EVOLUTION_OPERATORS)")
            self.gui_mut_tries = g.add_number("Max tries (until viable)", 64, min=1, max=2000, step=1)
            btn_until = g.add_button("Mutate until viable")
            self.md_mut_status = g.add_markdown("")
            hist_btns = g.add_button_group("History", ["back", "forward"])
            self.gui_ghost = g.add_checkbox("Show parent ghost", True)
            self.md_diff = g.add_markdown("")
            self.md_lineage = g.add_markdown("")
            with g.add_folder("Growth / shrink operators"):
                op_btns_a = [(op, g.add_button(op)) for op in src.STRUCTURAL_OPERATORS]
            with g.add_folder("Small-step operators"):
                op_btns_b = [(op, g.add_button(op)) for op in src.SMALL_STEP_OPS]
            with g.add_folder("Mutation spread", expand_by_default=False):
                self.gui_spread_k = g.add_slider("Children K", 2, 24, 1, 8)
                self.gui_spread_r = g.add_slider("Ring radius (m)", 0.15, 1.2, 0.05, 0.4)
                spread_btns = g.add_button_group("Spread", ["build", "clear"])
                self.md_spread = g.add_markdown("")

        # ---- Analysis -----------------------------------------------------
        with tabs.add_tab("Analysis"):
            self.gui_overlap_mode = g.add_dropdown("Red overlap highlight",
                                                   [OVERLAP_CURRENT, OVERLAP_ORACLE, OVERLAP_OFF])
            self.gui_show_spawn = g.add_checkbox("Show spawn point (object size)", True)
            self.md_live = g.add_markdown("")
            self.md_analysis = g.add_markdown("")
            self.md_nearest = g.add_markdown("")

        # ---- View ---------------------------------------------------------
        with tabs.add_tab("View"):
            cam_btns = g.add_button_group("Camera", ["top", "side", "palm-up"])
            self.gui_orient = g.add_dropdown("Orientation", [ORIENT_ROOT, ORIENT_PALM_UP])
            self.gui_capsule_mode = g.add_dropdown("Capsule convention", list(gm.CAPSULE_MODES),
                                                   hint="simulator: PhysX capsule spanning [0, L] (what the overlap "
                                                        "oracle checks); grammar: geometry.Capsule core [0, L] "
                                                        "(to_urdf collision)")
            self.gui_show_caps = g.add_checkbox("Capsules", True)
            self.gui_show_cells = g.add_checkbox("Palm cells", True)
            self.gui_show_spines = g.add_checkbox("Palm-body capsules (spines)", False)
            self.gui_show_axes = g.add_checkbox("Joint axes", True)
            self.gui_show_tips = g.add_checkbox("Fingertip markers", True)
            self.gui_show_root = g.add_checkbox("Root frame", True)
            self.gui_lbl_bodies = g.add_checkbox("Body labels", False)
            self.gui_lbl_joints = g.add_checkbox("Joint labels", False)
            g.add_markdown("Joint axes: blue flexion -> orange abduction (blend by angle), purple twist "
                           "(axis within 35 deg of its link), teal palm joint. Tips: green reach the spawn, "
                           "orange do not. The envelope's ghost/padding slots are never drawn: the scene is the "
                           "grammar model itself.")

        # ---- Gallery -------------------------------------------------------
        with tabs.add_tab("Gallery"):
            self.gui_gal_variant = g.add_dropdown("Variant", labels, initial_value=src.variant_label(variant))
            self.gui_gal_compare = g.add_dropdown("Compare with (one row each, same seeds)", ["(none)"] + labels)
            self.gui_gal_rows = g.add_number("Rows", 4, min=1, max=10, step=1)
            self.gui_gal_cols = g.add_number("Columns", 4, min=1, max=12, step=1)
            self.gui_gal_seed = g.add_number("Start seed", 0, min=0, step=1)
            self.gui_gal_spacing = g.add_slider("Spacing (m)", 0.15, 0.8, 0.05, 0.35)
            self.gui_gal_pose = g.add_dropdown("Pose", ["zero", "reset"])
            self.gui_gal_colour = g.add_checkbox("Colour by viability", True)
            self.gui_gal_cells = g.add_checkbox("Palm cells (slower)", False)
            gal_btns = g.add_button_group("Gallery", ["build", "clear"])
            self.md_gallery = g.add_markdown("")

        # ---- Export --------------------------------------------------------
        with tabs.add_tab("Export"):
            g.add_markdown(f"Writes to `{self.out_dir}` (git-ignored).")
            exp_btns = g.add_button_group("Export", ["derivation JSON", "URDF", "history JSON"])
            self.gui_download = g.add_checkbox("Also download in the browser", False)
            self.md_export = g.add_markdown("")

        # ---- callbacks ---------------------------------------------------
        def var_name(handle) -> str:
            return src.variant_from_label(handle.value)

        @btn_sample.on_click
        def _(_):
            self.sample(var_name(self.gui_variant), int(self.gui_seed.value), self.gui_until.value)

        @self.gui_variant.on_update
        def _(_):
            if not self._suppress:
                self.sample(var_name(self.gui_variant), int(self.gui_seed.value), self.gui_until.value)

        @seed_btns.on_click
        def _(event):
            v = event.target.value
            s = int(self.gui_seed.value)
            if v == "prev":
                s, step = max(s - 1, 0), -1
            elif v == "next":
                s, step = s + 1, 1
            else:
                s, step = int(self.rng.integers(0, 1_000_000)), 1
            self._set_seed_field(s)
            self.sample(var_name(self.gui_variant), s, self.gui_until.value, step=step)

        def load_selected_hand():
            self.load_commercial(self._hand_ids[self.gui_hand.value])

        btn_hand.on_click(lambda _: load_selected_hand())
        self.gui_hand.on_update(lambda _: None if self._suppress else load_selected_hand())

        @self.gui_meshes.on_update
        def _(_):
            self.overlay.set_visible(bool(self.gui_meshes.value))

        @self.gui_mesh_opacity.on_update
        def _(_):
            self.overlay.set_opacity(float(self.gui_mesh_opacity.value))

        btn_file.on_click(lambda _: self.load_file(self.gui_path.value))

        @upload.on_upload
        def _(_):
            f = upload.value
            try:
                doc = json.loads(f.content.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.md_file.content = f"could not parse {f.name}: {exc}"
                return
            self.run_job("load upload", lambda: self.load_document(doc, f.name))

        @self.gui_file_index.on_update
        def _(_):
            if not self._suppress:
                self.run_job("file entry", lambda: self.show_file_entry(int(self.gui_file_index.value)))

        @file_btns.on_click
        def _(event):
            if self.loaded_file is None:
                return
            i = int(self.gui_file_index.value) + (1 if event.target.value == "next" else -1)
            i = max(0, min(i, len(self.loaded_file.entries) - 1))
            self.gui_file_index.value = i  # triggers on_update

        @self.gui_curl.on_update
        def _(_):
            if not self._suppress:
                self.pose_curl(self.gui_curl.value)

        @pose_btns.on_click
        def _(event):
            {"zero": self.pose_zero, "env reset": self.pose_reset, "random": self.pose_random}[event.target.value]()

        self.gui_anim.on_update(lambda _: self.set_animating(bool(self.gui_anim.value)))

        @self.gui_mut_dist.on_update
        def _(_):
            if not self._suppress:
                self.mut_variant = var_name(self.gui_mut_dist)
                self._status(f"mutation distribution {self.mut_variant}")

        btn_rand.on_click(lambda _: self.mutate(None))
        btn_until.on_click(lambda _: self.mutate_until_viable())
        for op, b in op_btns_a + op_btns_b:
            b.on_click(lambda _, op=op: self.mutate(op))

        @hist_btns.on_click
        def _(event):
            (self.back if event.target.value == "back" else self.forward)()

        @self.gui_ghost.on_update
        def _(_):
            with self.lock:
                if self.prep is not None:
                    self._build_ghost()
                    self._update_orientation()
                    self._render_pose()

        @spread_btns.on_click
        def _(event):
            if event.target.value == "build":
                self.build_spread(int(self.gui_spread_k.value), float(self.gui_spread_r.value))
            else:
                with self.lock:
                    self._clear_spread()

        @self.gui_overlap_mode.on_update
        def _(_):
            with self.lock:
                self._render_pose()

        @self.gui_show_spawn.on_update
        def _(_):
            for h in self.spawn_handles:
                h.visible = bool(self.gui_show_spawn.value)

        @cam_btns.on_click
        def _(event):
            self.camera(event.target.value)

        @self.gui_orient.on_update
        def _(_):
            with self.lock:
                if self.prep is not None:
                    self._update_orientation()

        for h in (self.gui_capsule_mode, self.gui_show_spines, self.gui_show_cells, self.gui_show_axes,
                  self.gui_show_tips, self.gui_show_root):
            h.on_update(lambda _: self._rebuild_view())

        @self.gui_show_caps.on_update
        def _(_):
            self.renderer.set_layer_visibility(capsules=bool(self.gui_show_caps.value))

        def relabel(_):
            with self.lock:
                if self.prep is not None:
                    self.renderer.set_labels(self.gui_lbl_bodies.value, self.gui_lbl_joints.value,
                                             self.prep.view.primitives(self.u))

        self.gui_lbl_bodies.on_update(relabel)
        self.gui_lbl_joints.on_update(relabel)

        @gal_btns.on_click
        def _(event):
            if event.target.value == "build":
                cmp = self.gui_gal_compare.value
                self.build_gallery(var_name(self.gui_gal_variant), None if cmp == "(none)" else src.variant_from_label(cmp),
                                   int(self.gui_gal_rows.value), int(self.gui_gal_cols.value),
                                   int(self.gui_gal_seed.value), float(self.gui_gal_spacing.value))
            else:
                self.clear_gallery()

        @exp_btns.on_click
        def _(event):
            v = event.target.value
            fn = {"derivation JSON": self.export_json, "URDF": self.export_urdf, "history JSON": self.export_history}[v]
            self.run_job(f"export {v}", fn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--variant", default="G_V3S", choices=src.variant_names())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-until-viable", action="store_true", help="start with a plain draw of --seed")
    ap.add_argument("--out-dir", default=str(gexp.DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)
    server = viser.ViserServer(host=args.host, port=args.port, label="grammar viewer")
    GrammarViewer(server, variant=args.variant, seed=args.seed, until_viable=not args.no_until_viable,
                  out_dir=Path(args.out_dir))
    print(f"grammar viewer at http://{args.host}:{args.port}", flush=True)
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
