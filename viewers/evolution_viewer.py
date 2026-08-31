"""Browse any design from any iteration of the training-free design loop, in viser.

    .venv_isaacsim/bin/python -m viewers.evolution_viewer --port 8090

Pick an arm (nominal / wrench), an iteration (0..10), and a design; the arm and
hand are drawn from that design's own parameters. Designs can be ordered BY
FITNESS rather than by slot index, which is usually what you want -- "show me
the best hand in wrench iteration 7" is a question the index cannot answer.

MUTANTS HAVE NO URDF ON DISK. mutate.py writes parameters, not files, and
_resolve_robot_population stopped emitting a URDF per design on the authored
path (nothing reads them there, and concurrent seed-jobs raced to write the same
paths). So this viewer builds the URDF from the manifest's params on demand.

THE PREVIEW URDF GOES IN assets/urdf/generated/, NOT A TEMP DIR. build_hand_urdf
emits arm mesh references as `../kuka_sharpa_description/`, which resolves only
from that directory; writing to /tmp yields an arm with no meshes and no
colliders -- the failure that made every authored-robot run train an arm that
could not touch the table (docs/multi_embodiment.md 4). One fixed filename,
rewritten per selection.

MANIFESTS ARE 254 MB EACH, so switching iteration costs a few seconds of JSON
parse. Two are held at a time; the third eviction is LRU. Loading every
iteration up front would be ~5 GB.

Entry shapes differ by iteration and slot, so nothing is assumed:
    iter00        name, urdf, params            (a copy of the seed-3 population)
    elite slot    name, fitness, source         (slot 6j of an evolved generation)
    child slot    name, source, parent_name, rank, mutation_failed, displacement
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from collections import OrderedDict
from pathlib import Path

ARMS = ("nominal", "wrench")
PREVIEW_STEM = "_evolution_preview"
LOOP_DIR = "assets/urdf/generated/population/loop"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--page_size", type=int, default=64)
    p.add_argument("--cache", type=int, default=2, help="manifests held in memory")
    p.add_argument("--hz", type=float, default=30.0)
    return p


class EvolutionViewer:
    def __init__(self, args) -> None:
        import viser

        from hand_sampler.paths import resolve as resolve_repo_path

        self.args = args
        self.root = resolve_repo_path(LOOP_DIR)
        self.iters = {a: sorted(int(d.name[4:]) for d in (self.root / a).glob("iter*"))
                      for a in ARMS}
        print(f"[evo] iterations: " +
              ", ".join(f"{a} {self.iters[a][0]}-{self.iters[a][-1]}" for a in ARMS))

        self._cache: OrderedDict[tuple, tuple] = OrderedDict()
        self._urdf = None
        self._joint_sliders: list = []
        self._joint_folder = None
        self._pending = None
        self._pending_pose = False
        self._page = 0
        self._order: list[int] = []          # slot indices, in display order
        self._hands: list[dict] = []
        self._fit: list[float] | None = None

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.9, -0.9, 1.15)
            client.camera.look_at = (0.0, 0.15, 0.62)

        self._draw_scene()
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)
        self._build_gui()
        self._pending = ("select",)

    # --- scene ------------------------------------------------------------
    def _draw_scene(self) -> None:
        from viewers.reachability_viewer import TABLE_Z, table_extents

        tx, ty, tz = table_extents()
        self.server.scene.add_box("/scene/table", dimensions=(tx, ty, tz),
                                  position=(0.0, 0.0, TABLE_Z - tz / 2.0),
                                  color=(160, 130, 100))
        self.server.scene.add_grid("/scene/grid", width=2.4, height=2.4)

    # --- data -------------------------------------------------------------
    def _load(self, arm: str, it: int) -> tuple[list[dict], list[float] | None]:
        """Manifest entries plus per-slot fitness, LRU-cached."""
        from hand_sampler.paths import resolve as resolve_repo_path

        key = (arm, it)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self.md_info.content = f"_loading {arm} iter{it:02d} (254 MB)…_"
        t0 = time.perf_counter()
        hands = json.loads((self.root / arm / f"iter{it:02d}" / "manifest.json")
                           .read_text())["hands"]
        fit = None
        f = resolve_repo_path(f"results/loop/{arm}/iter{it:02d}/fitness.csv")
        if f.is_file():
            rows = list(csv.DictReader(open(f)))
            rows.sort(key=lambda r: int(r["design_index"]))
            fit = [float(r["mean_goals"]) for r in rows]
            if len(fit) != len(hands):
                print(f"[evo] {arm} iter{it}: fitness has {len(fit)} rows for "
                      f"{len(hands)} designs; ignoring", flush=True)
                fit = None
        self._cache[key] = (hands, fit)
        while len(self._cache) > self.args.cache:
            self._cache.popitem(last=False)
        print(f"[evo] loaded {arm} iter{it:02d}: {len(hands)} designs, "
              f"fitness {'yes' if fit else 'no'}, {time.perf_counter()-t0:.1f}s",
              flush=True)
        return self._cache[key]

    # --- gui --------------------------------------------------------------
    def _build_gui(self) -> None:
        g = self.server.gui
        with g.add_folder("Population", expand_by_default=True):
            self.dd_arm = g.add_dropdown("arm", ARMS, initial_value=ARMS[0])
            self.dd_arm.on_update(lambda _: setattr(self, "_pending", ("reload",)))
            self.sl_iter = g.add_slider("iteration", min=self.iters[ARMS[0]][0],
                                        max=self.iters[ARMS[0]][-1], step=1,
                                        initial_value=0)
            self.sl_iter.on_update(lambda _: setattr(self, "_pending", ("reload",)))
            self.dd_sort = g.add_dropdown(
                "order", ("fitness (best first)", "fitness (worst first)", "slot index"),
                initial_value="fitness (best first)")
            self.dd_sort.on_update(lambda _: setattr(self, "_pending", ("reorder",)))

        with g.add_folder("Design", expand_by_default=True):
            self.dd_design = g.add_dropdown("design", ("—",))
            self.dd_design.on_update(lambda _: setattr(self, "_pending", ("select",)))
            self.sl_page = g.add_slider("page", min=0, max=0, step=1, initial_value=0)
            self.sl_page.on_update(lambda _: setattr(self, "_pending", ("page",)))
            self.md_page = g.add_markdown("")
            self.btn_parent = g.add_button("Go to parent (previous iteration)")
            self.btn_parent.on_click(lambda _: setattr(self, "_pending", ("parent",)))

        with g.add_folder("View", expand_by_default=True):
            self.dd_geom = g.add_dropdown("geometry", ("visual", "collision"),
                                          initial_value="visual")
            self.dd_geom.on_update(lambda _: self._apply_geometry())
            self.btn_home = g.add_button("Home pose")
            self.btn_home.on_click(lambda _: self._reset_joints())

        self.md_info = g.add_markdown("_loading…_")

    # --- ordering / paging -------------------------------------------------
    def _reorder(self) -> None:
        mode = self.dd_sort.value
        n = len(self._hands)
        if mode == "slot index" or self._fit is None:
            self._order = list(range(n))
            if mode != "slot index":
                print("[evo] no fitness for this iteration; ordering by slot", flush=True)
        else:
            rev = mode.startswith("fitness (best")
            self._order = sorted(range(n), key=lambda i: self._fit[i], reverse=rev)
        pages = max(1, -(-n // self.args.page_size))
        self._page = min(self._page, pages - 1)
        self.sl_page.max = pages - 1
        self._refresh_page()

    def _page_slots(self) -> list[int]:
        lo = self._page * self.args.page_size
        return self._order[lo:lo + self.args.page_size]

    def _label(self, slot: int) -> str:
        h = self._hands[slot]
        f = "" if self._fit is None else f"  {self._fit[slot]:.1f}"
        src = h.get("source", "seed3")
        return f"[{slot}] {src}{f}"

    def _refresh_page(self) -> None:
        slots = self._page_slots()
        self._labels = {self._label(s): s for s in slots}
        self.dd_design.options = tuple(self._labels)
        lo = self._page * self.args.page_size
        self.md_page.content = (f"page **{self._page+1}/{self.sl_page.max+1}** — "
                                f"rank `{lo}`–`{lo+len(slots)-1}` of {len(self._order)}")
        if self.dd_design.value not in self._labels:
            self.dd_design.value = next(iter(self._labels))

    # --- model -------------------------------------------------------------
    def _show(self, slot: int) -> None:
        import numpy as np
        import yourdfpy
        from viser.extras import ViserUrdf

        from hand_sampler.population import hand_from_json
        from hand_sampler.synth_spec import synth_spec
        from hand_sampler.iiwa14_arm import BASE_POS, BASE_ROT
        from hand_sampler.urdf import OUT_DIR, write_urdf
        from viewers.reachability_viewer import _hull_collision_scene
        from hand_sampler.paths import resolve as resolve_repo_path

        entry = self._hands[slot]
        hand = hand_from_json(entry["params"])
        path = resolve_repo_path(OUT_DIR) / f"{PREVIEW_STEM}.urdf"
        write_urdf(hand, path)
        urdf = yourdfpy.URDF.load(str(path), load_meshes=True,
                                  load_collision_meshes=True,
                                  build_scene_graph=True,
                                  build_collision_scene_graph=True)
        _hull_collision_scene(urdf)

        self._clear()
        self.robot_frame.position = tuple(float(v) for v in BASE_POS)
        self.robot_frame.wxyz = tuple(float(v) for v in BASE_ROT)
        self._urdf = ViserUrdf(self.server, urdf, root_node_name="/robot",
                               load_meshes=True, load_collision_meshes=True)
        limits = self._urdf.get_actuated_joint_limits()
        spec = synth_spec(hand)
        home = {}
        home.update(spec.arm_default_joint_pos_resolved(start_arm_higher=False))
        home.update(spec.hand_default_joint_pos)
        self._home = np.array([
            float(np.clip(home.get(j, 0.0),
                          -np.pi if lo is None else lo,
                          np.pi if hi is None else hi))
            for j, (lo, hi) in limits.items()])
        self._urdf.update_cfg(self._home)
        self._build_joint_sliders(limits)
        self._apply_geometry()
        self.md_info.content = self._describe(slot, hand)
        self._cur_slot = slot

    def _describe(self, slot: int, hand) -> str:
        from hand_sampler.mutate import phenotype

        e = self._hands[slot]
        ph = phenotype(hand)
        b = [f"### `{e['name']}`",
             f"{self.dd_arm.value} · iteration {int(self.sl_iter.value)} · slot {slot}"]
        if self._fit is not None:
            rank = self._order.index(slot)
            b.append(f"**fitness {self._fit[slot]:.2f}** goals/6k steps "
                     f"({self._fit[slot]/10:.2f} per 10 s) — rank {rank} of {len(self._order)}")
        else:
            b.append("_no fitness for this iteration (never evaluated)_")
        src = e.get("source", "seed-3 original")
        b.append(f"origin: **{src}**" + (f" of `{e['parent_name']}`"
                                         if e.get("parent_name") else ""))
        if e.get("mutation_failed"):
            b.append("_mutation FAILED — this slot holds an unchanged copy of the parent_")
        b.append(f"{ph['n_active_fingers']} fingers, {ph['n_active_joints']} joints · "
                 f"finger length {100*ph['mean_finger_length']:.1f} cm · "
                 f"min separation {100*ph['min_separation']:.1f} cm")
        d = e.get("displacement")
        if d:
            b.append("moved from parent: "
                     f"Δfinger {1000*d.get('d_mean_finger_length',0):+.1f} mm · "
                     f"Δsep {1000*d.get('d_min_separation',0):+.1f} mm · "
                     f"Δjoints {d.get('d_n_active_joints',0):+d}")
        return "\n\n".join(b)

    def _goto_parent(self) -> None:
        it = int(self.sl_iter.value)
        e = self._hands[getattr(self, "_cur_slot", 0)]
        pname = e.get("parent_name")
        if not pname or it == 0:
            self.md_info.content += "\n\n_no parent recorded for this design_"
            return
        arm = self.dd_arm.value
        hands, fit = self._load(arm, it - 1)
        idx = next((i for i, h in enumerate(hands) if h["name"] == pname), None)
        if idx is None:
            self.md_info.content += f"\n\n_parent `{pname}` not found in iter{it-1:02d}_"
            return
        self.sl_iter.value = it - 1
        self._hands, self._fit = hands, fit
        self._reorder()
        self._page = self._order.index(idx) // self.args.page_size
        self.sl_page.value = self._page
        self._refresh_page()
        self.dd_design.value = self._label(idx)
        self._show(idx)

    # --- joints / geometry --------------------------------------------------
    def _clear(self) -> None:
        if self._urdf is not None:
            self._urdf.remove()
            self._urdf = None
        for h in self._joint_sliders:
            h.remove()
        self._joint_sliders = []
        if self._joint_folder is not None:
            self._joint_folder.remove()
            self._joint_folder = None

    def _build_joint_sliders(self, limits) -> None:
        import numpy as np

        GHOST = 1e-6
        self._joint_folder = self.server.gui.add_folder("Joints",
                                                        expand_by_default=False)
        with self._joint_folder:
            for i, (jname, (lo, hi)) in enumerate(limits.items()):
                lo = -np.pi if lo is None else float(lo)
                hi = np.pi if hi is None else float(hi)
                ghost = (hi - lo) < GHOST
                s = self.server.gui.add_slider(
                    ("🔒 " if ghost else "") + jname,
                    min=float(np.degrees(lo)),
                    max=float(np.degrees(max(hi, lo + 1e-6))),
                    step=0.5, initial_value=float(np.degrees(self._home[i])),
                    disabled=ghost)
                s.on_update(lambda _: setattr(self, "_pending_pose", True))
                self._joint_sliders.append(s)

    def _push_pose(self) -> None:
        import numpy as np
        if self._urdf is None or not self._joint_sliders:
            return
        self._urdf.update_cfg(np.array([np.radians(s.value)
                                        for s in self._joint_sliders]))

    def _reset_joints(self) -> None:
        import numpy as np
        for i, s in enumerate(self._joint_sliders):
            if not s.disabled:
                s.value = float(np.degrees(self._home[i]))
        self._pending_pose = True

    def _apply_geometry(self) -> None:
        if self._urdf is None:
            return
        collision = self.dd_geom.value == "collision"
        self._urdf.show_collision = collision
        self._urdf.show_visual = not collision

    # --- loop ---------------------------------------------------------------
    def run(self) -> None:
        print(f"\n  evolution viewer  http://localhost:{self.args.port}\n")
        period = 1.0 / max(self.args.hz, 1e-3)
        while True:
            t0 = time.time()
            try:
                # viser callbacks run on their own thread and must not rebuild
                # the scene graph the draw path is walking; they set _pending.
                if self._pending is not None:
                    act, self._pending = self._pending, None
                    if act[0] == "reload":
                        arm = self.dd_arm.value
                        self.sl_iter.min = self.iters[arm][0]
                        self.sl_iter.max = self.iters[arm][-1]
                        self._hands, self._fit = self._load(arm, int(self.sl_iter.value))
                        self._page = 0
                        self._reorder()
                        self._show(self._labels[self.dd_design.value])
                    elif act[0] == "reorder":
                        self._page = 0
                        self._reorder()
                    elif act[0] == "page":
                        self._page = int(self.sl_page.value)
                        self._refresh_page()
                    elif act[0] == "select":
                        if not self._hands:
                            self._hands, self._fit = self._load(
                                self.dd_arm.value, int(self.sl_iter.value))
                            self._reorder()
                        self._show(self._labels[self.dd_design.value])
                    elif act[0] == "parent":
                        self._goto_parent()
                if self._pending_pose:
                    self._pending_pose = False
                    self._push_pose()
            except Exception as exc:  # noqa: BLE001
                self.md_info.content = f"**error** — {exc}"
                print(f"[evo] {traceback.format_exc()}", flush=True)
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)


def main() -> None:
    try:
        EvolutionViewer(build_parser().parse_args()).run()
    except KeyboardInterrupt:
        print("\n[evo] stopped")


if __name__ == "__main__":
    main()
