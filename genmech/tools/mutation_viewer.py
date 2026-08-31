"""See what a mutation operator actually does, in viser, with no simulator.

    .venv_isaacsim/bin/python -m genmech.tools.mutation_viewer --port 8089

Pick one of the 24,576 seed-3 designs, pick an operator, hit **Mutate**. The
parent stays where it is and the mutant is stood up at its OWN station -- its own
table, its own arm, its own hand, one metre to the side -- and both are posed by
the same slider set, so the only thing that can separate them visually is
geometry. The whole point is to answer "is alpha big enough to matter" with your
eyes before spending a 30-minute eval job on it: the measured displacements at
alpha = 0.1 are 2-3 mm of reach and separation, which is very hard to have an
intuition about from a table.

Kinematics only: `yourdfpy` for the model, `viser` for display, and
`mutate.mutate_one` for the child -- the SAME function the population build
calls, including validate(), the analytic self-collision gate, and the flexion
re-alignment. So a design shown here is a design that would be simulated; if the
operator cannot produce a valid child it says so rather than drawing something
the gate would have rejected.

THE PREVIEW URDF IS WRITTEN INTO assets/urdf/generated/, NOT A TEMP DIR, and it
has to be. build_hand_urdf emits arm mesh references as `../kuka_sharpa_
description/`, which resolves only from that directory. Writing the preview to
/tmp gives an arm with no meshes and no colliders -- the exact failure that made
every authored-robot run train an arm that could not touch the table
(docs/multi_embodiment.md 4). One fixed filename, overwritten per draw.
"""

from __future__ import annotations

import argparse
import random
import time
import traceback
from pathlib import Path

PREVIEW_STEM = "_mutation_preview"

# Parent and mutant get a station each: their own table, their own arm, their own
# hand. Separated in X because the scene runs along Y -- the table sits at the
# origin and the arm base at y = 0.8 (iiwa14_arm.BASE_POS), so a Y offset would
# stack one station's arm on the other's table. The table is 0.475 m wide in x,
# so 1.0 m clears both tables and both arms.
STATION_DX = 1.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--population_seed", type=int, default=3)
    p.add_argument("--port", type=int, default=8089)
    p.add_argument("--page_size", type=int, default=64)
    p.add_argument("--initial_design", default=None)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--hz", type=float, default=30.0)
    return p


class MutationViewer:
    def __init__(self, args) -> None:
        import viser

        from genmech.robots.generated.mutate import OPERATORS
        from genmech.robots.generated.population import load_population

        self.args = args
        self.hands = load_population(args.population_seed)
        self.by_name = {h.name: h for h in self.hands}
        self.operators = OPERATORS
        print(f"[mut] seed {args.population_seed}: {len(self.hands)} designs")

        self.parent = None
        self.child = None
        self.urdfs = {"parent": None, "child": None}
        self._draw = 0
        self._page = 0
        self._n_pages = max(1, -(-len(self.hands) // args.page_size))
        self._joint_sliders = []
        self._joint_folder = None
        self._pending = None
        self._pending_pose = False

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.9, -0.9, 1.15)
            client.camera.look_at = (0.0, 0.15, 0.62)

        self._draw_scene()
        self.frames = {
            "parent": self.server.scene.add_frame("/parent", show_axes=False),
            "child": self.server.scene.add_frame("/child", show_axes=False,
                                                 visible=False),
        }
        self._build_gui()
        first = args.initial_design or self._page_names()[0]
        self.dd_design.value = first
        self._pending = ("parent", first)

    # --- scene ------------------------------------------------------------
    def _draw_scene(self) -> None:
        """One full station per hand: table, arm and hand, side by side.

        The mutant is not drawn into the parent's scene, it gets its own -- so a
        change in palm size or finger placement is read against a table at the
        same height and an arm at the same base pose, which is the only way a
        3 mm difference is legible.
        """
        from genmech.tools.reachability_viewer import TABLE_Z, table_extents

        tx, ty, tz = table_extents()
        for which, dx in (("parent", 0.0), ("child", STATION_DX)):
            self.server.scene.add_box(
                f"/scene/{which}/table", dimensions=(tx, ty, tz),
                position=(dx, 0.0, TABLE_Z - tz / 2.0),
                color=(160, 130, 100) if which == "parent" else (150, 120, 140))
            self.server.scene.add_label(f"/scene/{which}/label", which,
                                        position=(dx, 0.0, TABLE_Z + 0.28))
        # The child station starts hidden: there is no mutant until you make one,
        # and an empty table reads as a bug.
        self._set_station_visible("child", False)
        self.server.scene.add_grid("/scene/grid", width=3.6, height=2.8,
                                   position=(STATION_DX / 2.0, 0.0, 0.0))

    def _set_station_visible(self, which: str, visible: bool) -> None:
        for node in (f"/scene/{which}/table", f"/scene/{which}/label"):
            try:
                self.server.scene[node].visible = visible
            except Exception:  # noqa: BLE001 - node not created yet
                pass

    def _page_names(self) -> list[str]:
        lo = self._page * self.args.page_size
        return [h.name for h in self.hands[lo:lo + self.args.page_size]]

    def _page_label(self) -> str:
        lo = self._page * self.args.page_size
        hi = min(lo + self.args.page_size, len(self.hands))
        return (f"page **{self._page + 1}/{self._n_pages}** — designs "
                f"`{lo}`–`{hi - 1}` of {len(self.hands)}")

    # --- gui --------------------------------------------------------------
    def _build_gui(self) -> None:
        g = self.server.gui
        with g.add_folder("Parent design", expand_by_default=True):
            self.dd_design = g.add_dropdown("design", tuple(self._page_names()))
            self.dd_design.on_update(
                lambda _: setattr(self, "_pending", ("parent", self.dd_design.value)))
            self.sl_page = g.add_slider("page", min=0, max=self._n_pages - 1,
                                        step=1, initial_value=0)
            self.sl_page.on_update(lambda _: self._on_page())
            self.md_page = g.add_markdown(self._page_label())

        with g.add_folder("Mutation", expand_by_default=True):
            self.dd_op = g.add_dropdown("operator", tuple(self.operators),
                                        initial_value=self.operators[0])
            self.sl_alpha = g.add_slider("alpha", min=0.01, max=1.0, step=0.01,
                                         initial_value=float(self.args.alpha))
            self.btn_mut = g.add_button("Mutate  (new sample)")
            self.btn_mut.on_click(lambda _: setattr(self, "_pending", ("mutate",)))
            self.btn_clear = g.add_button("Clear mutant")
            self.btn_clear.on_click(lambda _: setattr(self, "_pending", ("clear",)))

        with g.add_folder("View", expand_by_default=True):
            self.dd_geom = g.add_dropdown("geometry", ("visual", "collision"),
                                          initial_value="visual")
            self.dd_geom.on_update(lambda _: self._apply_geometry())
            self.btn_home = g.add_button("Home pose")
            self.btn_home.on_click(lambda _: self._reset_joints())

        self.md_info = g.add_markdown("_loading…_")

    def _on_page(self) -> None:
        self._page = int(self.sl_page.value)
        names = self._page_names()
        self.dd_design.options = tuple(names)
        self.md_page.content = self._page_label()
        if self.dd_design.value not in names:
            self.dd_design.value = names[0]
            self._pending = ("parent", names[0])

    # --- model ------------------------------------------------------------
    def _preview_path(self, stem: str) -> Path:
        from genmech.tools.build_hand_urdf import OUT_DIR
        from genmech.utils.paths import resolve as resolve_repo_path

        return resolve_repo_path(OUT_DIR) / f"{stem}.urdf"

    def _show_hand(self, hand, which: str) -> None:
        """Write this hand's URDF and stand it up at its own station."""
        import numpy as np
        import yourdfpy
        from viser.extras import ViserUrdf

        from genmech.robots.generated.synth_spec import synth_spec
        from genmech.robots.iiwa14_arm import BASE_POS, BASE_ROT
        from genmech.tools.build_hand_urdf import write_urdf
        from genmech.tools.reachability_viewer import _hull_collision_scene

        path = self._preview_path(f"{PREVIEW_STEM}_{which}")
        write_urdf(hand, path)
        urdf = yourdfpy.URDF.load(str(path), load_meshes=True,
                                  load_collision_meshes=True,
                                  build_scene_graph=True,
                                  build_collision_scene_graph=True)
        _hull_collision_scene(urdf)

        dx = 0.0 if which == "parent" else STATION_DX
        frame = self.frames[which]
        frame.position = (float(BASE_POS[0]) + dx, float(BASE_POS[1]),
                          float(BASE_POS[2]))
        frame.wxyz = tuple(float(v) for v in BASE_ROT)
        frame.visible = True
        self._set_station_visible(which, True)

        handle = ViserUrdf(self.server, urdf, root_node_name=f"/{which}",
                           load_meshes=True, load_collision_meshes=True)
        limits = handle.get_actuated_joint_limits()
        spec = synth_spec(hand)
        home = {}
        home.update(spec.arm_default_joint_pos_resolved(start_arm_higher=False))
        home.update(spec.hand_default_joint_pos)
        pose = np.array([float(np.clip(home.get(j, 0.0),
                                       -np.pi if lo is None else lo,
                                       np.pi if hi is None else hi))
                         for j, (lo, hi) in limits.items()])
        handle.update_cfg(pose)

        self.urdfs[which] = handle
        if which == "parent":
            # The sliders belong to the parent: both stations carry the same
            # 37-joint template (ghosting locks joints, it does not remove them),
            # so one slider set poses both and any visible difference between the
            # stations is geometry rather than pose.
            self._home, self._limits = pose, limits
            self._build_joint_sliders(limits)
        self._apply_geometry()
        self._push_pose()

    def _clear(self, which: str, *, sliders: bool = False) -> None:
        if self.urdfs.get(which) is not None:
            self.urdfs[which].remove()
            self.urdfs[which] = None
        self.frames[which].visible = False
        self._set_station_visible(which, False)
        if not sliders:
            return
        for h in self._joint_sliders:
            h.remove()
        self._joint_sliders = []
        if self._joint_folder is not None:
            self._joint_folder.remove()
            self._joint_folder = None

    def _load_parent(self, name: str) -> None:
        """A new parent invalidates the mutant: it was a child of the old one."""
        self.parent = self.by_name[name]
        self.child = None
        self._clear("child")
        self._clear("parent", sliders=True)
        self._show_hand(self.parent, "parent")
        self.md_info.content = self._describe(self.parent, "parent")

    def _do_mutate(self) -> None:
        from genmech.robots.generated import mutate as M

        if self.parent is None:
            return
        self._draw += 1
        op = self.dd_op.value
        alpha = float(self.sl_alpha.value)
        rng = random.Random(f"{self.args.seed}:{op}:{self.parent.name}:{self._draw}")
        child, failed, tries = M.mutate_one(
            self.parent, op, alpha, rng, name=f"{self.parent.name}_{op}",
            tmpdir=self._preview_path("x").parent)
        if failed:
            self.md_info.content = (
                f"### mutation failed\n\n`{op}` could not produce a valid child of "
                f"`{self.parent.name}` in {tries} attempt(s) at alpha={alpha:.2f}.\n\n"
                "That is the operator telling you something: `num_joints_up` fails "
                "when every finger is already at 6 joints, `num_joints_down` fails by "
                "self-collision once the metacarpal goes to zero.")
            return
        self.child = child
        self._clear("child")
        self._show_hand(child, "child")
        self.md_info.content = self._describe(child, "child", tries=tries,
                                              op=op, alpha=alpha)

    def _describe(self, hand, which: str, *, tries: int = 0, op: str = "",
                  alpha: float = 0.0) -> str:
        from genmech.robots.generated import mutate as M

        ph = M.phenotype(hand)
        bits = [f"### {which}: `{hand.name}`",
                f"{ph['n_active_fingers']} fingers, {ph['n_active_joints']} joints",
                f"reach {100 * ph['mean_reach']:.1f} cm · "
                f"min separation {100 * ph['min_separation']:.1f} cm · "
                f"palm {1e6 * ph['palm_volume']:.0f} cm³"]
        if which == "child" and self.parent is not None:
            d = M._displacement(self.parent, hand)
            bits.append(f"**{op}** at alpha **{alpha:.2f}** — {tries} attempt(s)")
            bits.append(
                f"Δreach **{1000 * d['d_mean_reach']:+.1f} mm** · "
                f"Δmin-sep **{1000 * d['d_min_separation']:+.1f} mm** · "
                f"Δmean-sep **{1000 * d['d_mean_separation']:+.1f} mm** · "
                f"Δjoints **{d['d_n_active_joints']:+d}**")
            k = M.keepout_clamped(hand)
            if k:
                bits.append(f"_{k} mount(s) sitting on the wrist keep-out plane — "
                            "part of the step was absorbed there_")
        return "\n\n".join(bits)

    # --- joints / geometry -------------------------------------------------
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
        if not self._joint_sliders:
            return
        cfg = np.array([np.radians(s.value) for s in self._joint_sliders])
        for handle in self.urdfs.values():
            if handle is None:
                continue
            try:
                handle.update_cfg(cfg)
            except Exception:  # noqa: BLE001 - a differing joint set still draws
                pass

    def _reset_joints(self) -> None:
        import numpy as np
        for i, s in enumerate(self._joint_sliders):
            if not s.disabled:
                s.value = float(np.degrees(self._home[i]))
        self._pending_pose = True

    def _apply_geometry(self) -> None:
        collision = self.dd_geom.value == "collision"
        for h in self.urdfs.values():
            if h is not None:
                h.show_collision = collision
                h.show_visual = not collision

    # --- loop --------------------------------------------------------------
    def run(self) -> None:
        print(f"\n  mutation viewer  http://localhost:{self.args.port}\n")
        period = 1.0 / max(self.args.hz, 1e-3)
        try:
            while True:
                t0 = time.time()
                try:
                    # viser callbacks run on their own thread and must not
                    # rebuild the scene graph the draw path is walking. They set
                    # _pending; this loop acts on it.
                    if self._pending is not None:
                        action, self._pending = self._pending, None
                        if action[0] == "parent":
                            self._load_parent(action[1])
                        elif action[0] == "mutate":
                            self._do_mutate()
                        elif action[0] == "clear":
                            self.child = None
                            self._clear("child")
                            if self.parent is not None:
                                self.md_info.content = self._describe(
                                    self.parent, "parent")
                    if self._pending_pose:
                        self._pending_pose = False
                        self._push_pose()
                except Exception as exc:  # noqa: BLE001
                    self.md_info.content = f"**error** — {exc}"
                    print(f"[mut] {traceback.format_exc()}", flush=True)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n[mut] stopped")


def main() -> None:
    MutationViewer(build_parser().parse_args()).run()


if __name__ == "__main__":
    main()
