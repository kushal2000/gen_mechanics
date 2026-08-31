"""Browse the population's robot URDFs in viser, one dropdown, no simulator.

    .venv_isaacsim/bin/python -m coevolution.population.urdf_viewer --port 8088

Pure kinematics: ``yourdfpy`` for the model, ``viser`` for display. It boots in
seconds and holds no GPU, which is the point -- looking at a design's geometry
should not cost a Kit boot, and ``eval_interactive.py`` charges about a minute
per switch because it also has to simulate one.

Use this to answer "what does design 4,211 look like"; use ``eval_interactive``
to answer "what does it do".

**Picking a design.** A manifest holds 24,576 of them, which no dropdown can
usefully list, so there are two controls: a dropdown over a page of designs, and
a page control to move the window. ``--designs`` names specific ones to pin at
the top of every page, which is how you reach designs an offline sweep flagged.

**Geometry.** ``visual`` is the URDF's render geometry; ``collision`` is what
PhysX would resolve contacts against -- and it is drawn as the CONVEX HULL of
each collision mesh, because that is what Isaac Lab stamps on every mesh
collider. A generated hand's capsule is one collision cylinder but a cylinder
plus two spheres in visual, so the two views genuinely differ.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

NAMED_SPECS = ("sharpa_iiwa14", "allegro_iiwa14", "gen_sharpa_like")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--population_seed", type=int, default=3)
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--page_size", type=int, default=64,
                   help="Designs listed in the dropdown at once (default 64)")
    p.add_argument("--designs", default="",
                   help="Comma-separated design names pinned to every page")
    p.add_argument("--initial_design", default=None)
    p.add_argument("--hz", type=float, default=30.0)
    return p


class UrdfBrowser:
    def __init__(self, args) -> None:
        import viser

        from hand_sampler.population import load_population

        self.args = args
        self.pinned = [d.strip() for d in args.designs.split(",") if d.strip()]

        self.hands = load_population(args.population_seed)
        self.by_name = {h.name: h for h in self.hands}
        print(f"[urdf] seed {args.population_seed}: {len(self.hands)} designs")

        self._urdf = None
        self._joint_sliders = []
        self._joint_folder = None
        self._pending = None
        self._page = 0
        self._n_pages = max(1, -(-len(self.hands) // args.page_size))

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)

        @self.server.on_client_connect
        def _(client) -> None:
            client.camera.position = (0.9, -0.9, 1.15)
            client.camera.look_at = (0.0, 0.15, 0.62)

        self._draw_scene()
        self.robot_frame = self.server.scene.add_frame("/robot", show_axes=False)

        self._build_gui()
        first = args.initial_design or (self.pinned[0] if self.pinned
                                        else self._page_names()[0])
        self.dd_design.value = first
        self._pending = first

    def _draw_scene(self) -> None:
        """Table, grid and world frame -- deliberately no goal volume.

        reachability_viewer draws the goal box because its question is "can this
        hand reach the workspace". The question here is what the mechanism looks
        like, and a translucent box spanning the whole scene sits directly in
        front of the hand and washes out its geometry.
        """
        from hand_sampler.workspace import TABLE_Z, table_extents

        tx, ty, tz = table_extents()
        self.server.scene.add_box("/scene/table", dimensions=(tx, ty, tz),
                                  position=(0.0, 0.0, TABLE_Z - tz / 2.0),
                                  color=(160, 130, 100))
        self.server.scene.add_frame("/scene/origin", axes_length=0.15,
                                    axes_radius=0.005, position=(0.0, 0.0, 0.0))
        self.server.scene.add_grid("/scene/grid", width=2.4, height=2.4,
                                   position=(0.0, 0.0, 0.0))

    # --- design listing ---------------------------------------------------
    def _page_names(self) -> list[str]:
        lo = self._page * self.args.page_size
        hi = lo + self.args.page_size
        names = list(self.pinned)
        if self._page == 0:
            names += list(NAMED_SPECS)
        names += [h.name for h in self.hands[lo:hi]]
        # dict.fromkeys keeps first-seen order while removing repeats, so a
        # pinned design does not appear twice on its own page.
        return list(dict.fromkeys(names))

    def _build_gui(self) -> None:
        g = self.server.gui
        with g.add_folder("Design", expand_by_default=True):
            self.dd_design = g.add_dropdown("design", tuple(self._page_names()))
            self.dd_design.on_update(
                lambda _: setattr(self, "_pending", self.dd_design.value))
            self.sl_page = g.add_slider("page", min=0, max=self._n_pages - 1,
                                        step=1, initial_value=0)
            self.sl_page.on_update(lambda _: self._on_page())
            self.md_page = g.add_markdown(self._page_label())
            self.md_info = g.add_markdown("_loading…_")

        with g.add_folder("View", expand_by_default=True):
            self.dd_geom = g.add_dropdown("geometry", ("visual", "collision"),
                                          initial_value="visual")
            self.dd_geom.on_update(lambda _: self._apply_geometry())
            self.btn_home = g.add_button("Home pose")
            self.btn_home.on_click(lambda _: self._reset_joints())

    def _page_label(self) -> str:
        lo = self._page * self.args.page_size
        hi = min(lo + self.args.page_size, len(self.hands))
        return (f"page **{self._page + 1}/{self._n_pages}** — designs "
                f"`{lo}`–`{hi - 1}` of {len(self.hands)}")

    def _on_page(self) -> None:
        # Repopulating options is a GUI-only change, safe from this thread; the
        # URDF rebuild it may trigger goes through _pending like everything else.
        self._page = int(self.sl_page.value)
        names = self._page_names()
        self.dd_design.options = tuple(names)
        self.md_page.content = self._page_label()
        if self.dd_design.value not in names:
            self.dd_design.value = names[0]
            self._pending = names[0]

    # --- model ------------------------------------------------------------
    def _urdf_path_for(self, name: str) -> Path:
        from isaacsimenvs.robots import REGISTRY
        from hand_sampler.urdf import urdf_path_for
        from hand_sampler.paths import resolve as resolve_repo_path

        if name in REGISTRY:
            return resolve_repo_path(REGISTRY[name].urdf_path)
        if name == "gen_sharpa_like":
            from hand_sampler.synth_spec import get_generated_spec
            return resolve_repo_path(get_generated_spec(name).urdf_path)
        if name in self.by_name:
            return urdf_path_for(self.by_name[name])
        raise KeyError(f"unknown design {name!r}")

    def _load(self, name: str) -> None:
        import numpy as np
        import yourdfpy
        from viser.extras import ViserUrdf

        from hand_sampler.iiwa14_arm import BASE_POS, BASE_ROT
        from hand_sampler.workspace import _hull_collision_scene

        path = self._urdf_path_for(name)
        if not path.is_file():
            self.md_info.content = f"**{name}** — URDF missing at `{path}`"
            return

        if self._urdf is not None:
            self._urdf.remove()
            self._urdf = None
        for handle in self._joint_sliders:
            handle.remove()
        self._joint_sliders = []
        if self._joint_folder is not None:
            self._joint_folder.remove()
            self._joint_folder = None

        urdf = yourdfpy.URDF.load(
            str(path), load_meshes=True, load_collision_meshes=True,
            build_scene_graph=True, build_collision_scene_graph=True)
        # Hull the collision scene BEFORE ViserUrdf reads it, so the collision
        # view is what PhysX would actually simulate rather than the URDF's
        # declared mesh.
        n_hulled = _hull_collision_scene(urdf)

        self.robot_frame.position = tuple(float(v) for v in BASE_POS)
        self.robot_frame.wxyz = tuple(float(v) for v in BASE_ROT)
        self._urdf = ViserUrdf(self.server, urdf, root_node_name="/robot",
                               load_meshes=True, load_collision_meshes=True)
        self._apply_geometry()

        limits = self._urdf.get_actuated_joint_limits()
        self._home = self._home_pose(name, limits)
        self._urdf.update_cfg(self._home)
        self._build_joint_sliders(limits)

        hand = self.by_name.get(name)
        bits = [f"### `{name}`", f"`{path.name}`"]
        if hand is not None:
            px, py, pz = hand.palm_extents
            bits.append(f"{hand.n_active_fingers} active fingers, "
                        f"{hand.n_active_joints} active joints")
            bits.append(f"palm {100 * px:.1f} x {100 * py:.1f} x {100 * pz:.1f} cm")
        bits.append(f"{len(limits)} actuated joints, {n_hulled} collision meshes hulled")
        self.md_info.content = "\n\n".join(bits)
        print(f"[urdf] {name}: {len(limits)} joints, {path.name}", flush=True)

    def _home_pose(self, name: str, limits) -> "np.ndarray":
        """The spec's home pose, in ViserUrdf's actuated-joint order.

        Zero is not the home pose: SHARPA's arm sits at joint_1 = -90 deg and
        joint_2 = +90 deg, so a zero vector draws the arm straight up through
        the table instead of over it. The spec carries the real thing as
        name-keyed dicts, and ViserUrdf wants a vector in ITS order, so map by
        name -- the two orders are not the same and assuming otherwise poses the
        wrong joints.
        """
        import numpy as np

        home = {}
        try:
            spec = self._spec_for(name)
            home.update(spec.arm_default_joint_pos_resolved(start_arm_higher=False))
            home.update(spec.hand_default_joint_pos)
        except Exception as exc:  # noqa: BLE001 - a URDF with no spec still draws
            print(f"[urdf] no spec home pose for {name}: {exc}", flush=True)

        out = []
        for jname, (lo, hi) in limits.items():
            value = float(home.get(jname, 0.0))
            if lo is not None and hi is not None:
                value = float(np.clip(value, lo, hi))
            out.append(value)
        return np.array(out)

    def _spec_for(self, name: str):
        from isaacsimenvs.robots import REGISTRY

        if name in REGISTRY:
            return REGISTRY[name]
        if name == "gen_sharpa_like":
            from hand_sampler.synth_spec import get_generated_spec
            return get_generated_spec(name)
        from hand_sampler.synth_spec import synth_spec
        return synth_spec(self.by_name[name])

    def _build_joint_sliders(self, limits) -> None:
        import numpy as np

        # A ghosted joint has a limit range of ~0 and cannot move; showing it as
        # a live slider invites the conclusion that the viewer is broken.
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
                    min=float(np.degrees(lo)), max=float(np.degrees(max(hi, lo + 1e-6))),
                    step=0.5, initial_value=float(np.degrees(self._home[i])),
                    disabled=ghost)
                s.on_update(lambda _: setattr(self, "_pending_pose", True))
                self._joint_sliders.append(s)
        self._pending_pose = False

    def _push_pose(self) -> None:
        import numpy as np
        if self._urdf is None or not self._joint_sliders:
            return
        self._urdf.update_cfg(
            np.array([np.radians(s.value) for s in self._joint_sliders]))

    def _reset_joints(self) -> None:
        for i, s in enumerate(self._joint_sliders):
            if not s.disabled:
                import numpy as np
                s.value = float(np.degrees(self._home[i]))
        self._pending_pose = True

    def _apply_geometry(self) -> None:
        if self._urdf is None:
            return
        collision = self.dd_geom.value == "collision"
        self._urdf.show_collision = collision
        self._urdf.show_visual = not collision

    # --- loop -------------------------------------------------------------
    def run(self) -> None:
        print(f"\n  URDF browser  http://localhost:{self.args.port}\n")
        period = 1.0 / max(self.args.hz, 1e-3)
        self._pending_pose = False
        try:
            while True:
                t0 = time.time()
                try:
                    # Same rule as the other viewers: viser callbacks run on
                    # their own thread and must not rebuild the scene graph the
                    # draw path is using. They set a flag; this loop acts.
                    if self._pending is not None:
                        name, self._pending = self._pending, None
                        self._load(name)
                    if self._pending_pose:
                        self._pending_pose = False
                        self._push_pose()
                except Exception as exc:  # noqa: BLE001
                    self.md_info.content = f"**error** — {exc}"
                    print(f"[urdf] {traceback.format_exc()}", flush=True)
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        except KeyboardInterrupt:
            print("\n[urdf] stopped")


def main() -> int:
    args = build_parser().parse_args()
    UrdfBrowser(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
