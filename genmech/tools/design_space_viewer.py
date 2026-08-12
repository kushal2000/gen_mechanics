"""Explore the generated hand design space by hand, with sliders.

The numbers in ``params.py`` describe a 61-dimensional space; this makes it
something you can look at. Drag a finger's mount, stretch a phalanx, add joints,
and watch the hand rebuild -- with a live self-collision readout, because the
sampler's validity gates are geometric proxies (mount separation, reach) and the
widened space showed they do not hold: every hand in a six-sample batch
overlapped itself at the home pose.

    .venv_isaacsim/bin/python -m genmech.tools.design_space_viewer
    .venv_isaacsim/bin/python -m genmech.tools.design_space_viewer --port 8081
    .venv_isaacsim/bin/python -m genmech.tools.design_space_viewer --params sharpa_like

**The arm and table are drawn once.** They are identical across every design by
construction (docs/methodology.md §1), so their meshes load at startup and are
never rebuilt -- reloading the arm's STLs on every slider drag is what would
make this unusable. Only the hand rebuilds. The fingers are still checked
against the arm, though, or a finger curling back into the wrist would read as
clean.

Geometry comes from the same path the physics does: the parameters are written
to a real URDF via ``build_hand_urdf``, loaded, and its *collision* geometry
rendered. Nothing here re-implements forward kinematics, so what you see cannot
drift from what gets simulated.

Overlapping links are drawn in red. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import math
import random
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from genmech.robots.generated import params as P
from genmech.robots.iiwa14_arm import ARM_DEFAULT_JOINT_POS, BASE_POS
from genmech.tools.build_hand_urdf import OUT_DIR, link_name, write_urdf
from genmech.utils.paths import resolve as resolve_repo_path
from genmech.tools.check_self_collision import (
    EPS_M, filtered_pairs, jointed_pairs, link_collision_meshes, merge_map,
    penetration,
)


D = math.degrees
R = math.radians

OK_COLOR = (0.62, 0.66, 0.70)
HIT_COLOR = (0.85, 0.25, 0.20)
PALM_COLOR = (0.45, 0.50, 0.55)
ARM_COLOR = (0.30, 0.33, 0.36)
TABLE_COLOR = (0.55, 0.47, 0.38)

# Imported, not restated: the reachability viewer already carries the task's
# scene geometry, and two copies would drift.
from genmech.tools.reachability_viewer import (          # noqa: E402
    GOAL_VOLUME_MAXS, GOAL_VOLUME_MINS, TABLE_SIZE, TABLE_Z,
)

# Finger slots keep SHARPA's role names purely as labels; a sampled finger has
# no anatomy, only a slot index.
SLOT_LABELS = ("f0 (thumb slot)", "f1", "f2", "f3", "f4")

# Everything the robot owns hangs off this frame, which sits at BASE_POS. The
# table and goal volume are then drawn in plain world coordinates, matching the
# reachability viewer -- mixing the two frames is what put the arm through the
# table.
ROBOT_ROOT = "/robot"

# Post-merge palm body: merge_fixed_joints folds gen_palm into the arm's last
# link, so the palm's geometry arrives under this name, not "gen_palm".
PALM_BODY = "iiwa14_link_7"

# Mesh filenames inside a generated URDF are written relative to the real asset
# directory (the arm is reached as "../kuka_sharpa_description/..."), so they
# resolve against that, not against the temp dir the live URDF is written to.
ASSET_BASE = resolve_repo_path(OUT_DIR)


class HandView:
    """Renders one HandParams and reports what is wrong with it."""

    def __init__(self, server, workdir: Path):
        self.server = server
        self.workdir = workdir
        self.handles: list = []
        self.readout = None
        # The arm never changes -- it is identical across every design by
        # construction -- so its meshes are loaded once. Reloading its STLs on
        # every slider drag is what would make this unusable.
        self.arm_meshes: dict = {}

    def _cfg(self, urdf) -> np.ndarray:
        """Arm at its home pose, hand at zero -- the pose episodes start from."""
        pose = dict(ARM_DEFAULT_JOINT_POS)
        return np.array([pose.get(n, 0.0) for n in urdf.actuated_joint_names])

    def load_arm(self, hand: P.HandParams) -> dict:
        import yourdfpy

        path = write_urdf(hand, self.workdir / "arm_ref.urdf")
        urdf = yourdfpy.URDF.load(
            str(path), load_meshes=False, load_collision_meshes=False,
            build_scene_graph=True, build_collision_scene_graph=True,
        )
        urdf.update_cfg(self._cfg(urdf))
        self.arm_meshes = link_collision_meshes(
            urdf, ASSET_BASE, merge_map(path), only_prefix="iiwa14_")
        # iiwa14_link_7 is where the palm merges, so it would be rebuilt with
        # every hand. Keep only the arm's own links here.
        self.arm_meshes.pop("iiwa14_link_7", None)
        return self.arm_meshes

    def rebuild(self, hand: P.HandParams) -> dict:
        import yourdfpy

        for h in self.handles:
            h.remove()
        self.handles = []

        path = write_urdf(hand, self.workdir / "live.urdf")
        urdf = yourdfpy.URDF.load(
            str(path), load_meshes=False, load_collision_meshes=False,
            build_scene_graph=True, build_collision_scene_graph=True,
        )
        urdf.update_cfg(self._cfg(urdf))

        merged = merge_map(path)
        meshes = link_collision_meshes(urdf, ASSET_BASE, merged,
                                       only_prefix="gen_")
        # Check the fingers against the arm too. The standalone tool does, and
        # without it a finger curling back into the wrist reads as clean here.
        meshes = {**self.arm_meshes, **meshes}

        # Same exclusions the simulator applies: directly-jointed pairs are
        # auto-filtered by PhysX, and the template's adjacency map is filtered
        # explicitly before baking.
        from genmech.robots.generated.synth_spec import template_adjacent_links
        skip = jointed_pairs(path, merged)
        for a, others in template_adjacent_links().items():
            for b in others:
                skip.add(frozenset((a, b)))

        names = sorted(meshes)
        hits: list[tuple[str, str, float]] = []
        bad: set[str] = set()
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in skip:
                    continue
                d = penetration(meshes[a], meshes[b])
                if d > EPS_M:
                    hits.append((a, b, d))
                    bad.add(a)
                    bad.add(b)

        for name, mesh in meshes.items():
            # Skip only the STATIC arm links; draw_static_scene already drew
            # them. iiwa14_link_7 is NOT static -- merge_fixed_joints folds the
            # palm into it, so it is rebuilt with every hand.
            if name in self.arm_meshes and name not in bad:
                continue
            colour = HIT_COLOR if name in bad else (
                PALM_COLOR if name == PALM_BODY else OK_COLOR)
            m = mesh
            self.handles.append(
                self.server.scene.add_mesh_simple(
                    f"{ROBOT_ROOT}/hand/{name}",
                    vertices=m.vertices, faces=m.faces,
                    color=tuple(int(c * 255) for c in colour),
                )
            )

        hits.sort(key=lambda h: -h[2])
        return {"hits": hits, "n_bodies": len(meshes)}


def draw_static_scene(server, view: HandView, hand: P.HandParams) -> None:
    """Arm, table and goal volume: the context a hand has to be sensible in.

    All of it is fixed across designs, so it is drawn once. The whole robot sits
    at ``BASE_POS`` and the table and goal volume are in world coordinates, so
    what you see is the task's actual geometry rather than a sketch.
    """
    import trimesh

    server.scene.add_frame(ROBOT_ROOT, show_axes=False,
                           position=tuple(float(v) for v in BASE_POS))
    for name, mesh in view.load_arm(hand).items():
        server.scene.add_mesh_simple(
            f"{ROBOT_ROOT}/arm/{name}",
            vertices=mesh.vertices, faces=mesh.faces,
            color=tuple(int(c * 255) for c in ARM_COLOR),
        )

    mins = np.array(GOAL_VOLUME_MINS)
    maxs = np.array(GOAL_VOLUME_MAXS)
    corners = np.array([[x, y, z] for x in (mins[0], maxs[0])
                        for y in (mins[1], maxs[1]) for z in (mins[2], maxs[2])])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    server.scene.add_line_segments(
        "/scene/goal_volume",
        points=np.array([[corners[a], corners[b]] for a, b in edges]),
        colors=(120, 160, 210), line_width=2.0,
    )

    server.scene.add_box(
        "/scene/table", dimensions=(TABLE_SIZE[0], TABLE_SIZE[1], 0.02),
        position=(0.0, 0.0, TABLE_Z - 0.01),
        color=tuple(int(c * 255) for c in TABLE_COLOR),
    )
    server.scene.add_grid("/scene/grid", width=2.4, height=2.4,
                          position=(0.0, 0.0, 0.0))


def _summary(hand: P.HandParams, info: dict) -> str:
    per = [f.n_active_joints for f in hand.fingers if f.active]
    lines = [
        f"**{hand.n_active_fingers} fingers**, "
        f"**{hand.n_active_joints}** active joints "
        f"of {P.N_FINGER_SLOTS * P.N_JOINT_SLOTS}",
        f"per finger: `{per}`",
        "",
    ]
    try:
        P.validate(hand)
        lines.append("sampler gates: **pass**")
    except P.InvalidHand as exc:
        lines.append(f"sampler gates: **FAIL** — {exc}")

    hits = info["hits"]
    if not hits:
        lines.append("self-collision: **none**")
    else:
        lines.append(f"self-collision: **{len(hits)} pair(s)**, "
                     f"worst {hits[0][2] * 1000:.1f} mm")
        for a, b, d in hits[:6]:
            lines.append(f"- `{a}` ↔ `{b}` — {d * 1000:.1f} mm")
        if len(hits) > 6:
            lines.append(f"- … {len(hits) - 6} more")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--params", default=None,
                        help="'sharpa_like' to start from the reference hand")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import viser

    workdir = Path(tempfile.mkdtemp(prefix="genmech_designviz_"))
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/origin", axes_length=0.06, show_axes=True)

    view = HandView(server, workdir)

    state: dict = {"hand": P.SHARPA_LIKE if args.params == "sharpa_like"
                   else P.sample_valid(random.Random(args.seed), name="live")}

    print("[design_viz] loading arm + table (once) ...")
    draw_static_scene(server, view, state["hand"])

    # --- sampling controls -------------------------------------------------
    with server.gui.add_folder("Sample"):
        g_seed = server.gui.add_number("seed", initial_value=args.seed, step=1)
        g_nf = server.gui.add_dropdown(
            "fingers", ("random", "2", "3", "4", "5"), initial_value="random")
        g_jmin = server.gui.add_slider("joints/finger min", min=2, max=6, step=1,
                                       initial_value=P.JOINTS_PER_FINGER_RANGE[0])
        g_jmax = server.gui.add_slider("joints/finger max", min=2, max=6, step=1,
                                       initial_value=P.JOINTS_PER_FINGER_RANGE[1])
        b_sample = server.gui.add_button("Resample")
        b_sharpa = server.gui.add_button("Reset to SHARPA_LIKE")

    # Palm size gates which layouts exist at all -- a thin palm has no room on
    # its +-x faces for an opposed gripper -- so it belongs next to the sampling
    # controls rather than buried per-finger.
    with server.gui.add_folder("Palm"):
        palm_sliders = {
            "x": server.gui.add_slider("thickness x (mm)", min=25, max=75,
                                       step=0.5, initial_value=49.9),
            "y": server.gui.add_slider("width y (mm)", min=45, max=125,
                                       step=0.5, initial_value=85.2),
            "z": server.gui.add_slider("height z (mm)", min=45, max=125,
                                       step=0.5, initial_value=86.4),
        }

    for _h in palm_sliders.values():
        _h.on_update(lambda _=None: apply_sliders())

    readout = server.gui.add_markdown("")

    # --- per-finger sliders -------------------------------------------------
    sliders: dict[str, dict] = {}
    updating = {"lock": False}

    def apply_sliders(_=None) -> None:
        """Slider -> params. Guarded so programmatic writes do not recurse."""
        if updating["lock"]:
            return
        hand = state["hand"]
        palm = (palm_sliders["x"].value / 1000.0,
                palm_sliders["y"].value / 1000.0,
                palm_sliders["z"].value / 1000.0)
        fingers = []
        for i, f in enumerate(hand.fingers):
            s = sliders[f"f{i}"]
            n_joints = int(s["n_joints"].value)
            active = bool(s["active"].value)
            mc = s["mc"].value / 1000.0
            if n_joints < P.MIN_JOINTS_FOR_METACARPAL:
                mc = 0.0
            mount = P.mount_on_face(
                face=s["face"].value,
                u_frac=s["u"].value, v_frac=s["v"].value,
                roll=R(s["roll"].value), tilt=R(s["tilt"].value),
                tilt_azimuth=R(s["tilt_az"].value),
                extents=palm,
            )
            aa = R(s["aa"].value)
            limits = dict(zip(P.JOINT_SLOTS, f.limits))
            limits["MCP_AA"] = (-aa, aa)
            enabled = P.enabled_for(n_joints)
            for slot, live in zip(P.JOINT_SLOTS, enabled):
                if live and (limits[slot][1] - limits[slot][0]) < 1e-6:
                    limits[slot] = (P._LIM_THUMB[slot] if slot.startswith("CMC")
                                    else P._LIM_FINGER[slot])
            fingers.append(replace(
                f, active=active, enabled=enabled,
                mount=mount,
                mount_params=(s["face"].value, s["u"].value, s["v"].value,
                              R(s["roll"].value), R(s["tilt"].value),
                              R(s["tilt_az"].value)),
                mc=P.Segment(xyz=(mc, 0.0, 0.0), rpy=f.mc.rpy),
                pp_length=s["pp"].value / 1000.0,
                mp_length=s["mp_len"].value / 1000.0,
                dp_length=s["dp"].value / 1000.0,
                radius_scale=s["rscale"].value,
                limits=tuple(limits[k] for k in P.JOINT_SLOTS),
            ))
        # Bypass HandParams' >=2-active-fingers check while the user is mid-edit;
        # the readout reports it instead of the GUI throwing.
        try:
            new = P.HandParams(name="live", fingers=tuple(fingers),
                               palm_extents=palm)
        except ValueError as exc:
            readout.content = f"**invalid**: {exc}"
            return
        state["hand"] = new
        render()

    def render() -> None:
        info = view.rebuild(state["hand"])
        readout.content = _summary(state["hand"], info)

    def push_sliders() -> None:
        """Params -> sliders, after a resample."""
        updating["lock"] = True
        pe = state["hand"].palm_extents
        put(palm_sliders["x"], round(pe[0] * 1000, 1))
        put(palm_sliders["y"], round(pe[1] * 1000, 1))
        put(palm_sliders["z"], round(pe[2] * 1000, 1))
        for i, f in enumerate(state["hand"].fingers):
            s = sliders[f"f{i}"]
            s["active"].value = f.active
            put(s["n_joints"], f.n_active_joints if f.active else 2)
            # SHARPA_LIKE's mounts are authored, not generated on a face (its
            # thumb sits inside the palm volume), so there is nothing to invert.
            # Show neutral face controls; touching one moves the mount onto a
            # face, which is a real edit and should look like one.
            fm = f.mount_params or ("+z", 0.0, 0.0, 0.0, 0.0, 0.0)
            s["face"].value = fm[0]
            put(s["u"], round(fm[1], 2))
            put(s["v"], round(fm[2], 2))
            put(s["roll"], round(D(fm[3]) % 360.0, 1))
            put(s["tilt"], round(D(fm[4]), 1))
            put(s["tilt_az"], round(D(fm[5]) % 360.0, 1))
            put(s["mc"], round(f.mc.length * 1000, 1))
            put(s["pp"], round(f.pp_length * 1000, 1))
            put(s["mp_len"], round(f.mp_length * 1000, 1))
            put(s["dp"], round(f.dp_length * 1000, 1))
            put(s["rscale"], round(f.radius_scale, 2))
            aa = f.limits[P.JOINT_SLOTS.index("MCP_AA")]
            put(s["aa"], round(D(max(abs(aa[0]), abs(aa[1]))), 1))
        updating["lock"] = False

    # Slider bounds sit OUTSIDE the sampler's ranges. A sampled value can land
    # exactly on a range endpoint, and mount rpy is jittered +-25 deg about role
    # defaults that already include +-180 -- so the pinky's roll reaches -205,
    # which viser rejects as out of bounds. Bounds are a display concern; the
    # sampler's ranges are the real constraint.
    def mk_slider(label, lo, hi, step, val):
        s = server.gui.add_slider(label, min=lo, max=hi, step=step,
                                  initial_value=min(max(val, lo), hi))
        s.on_update(apply_sliders)
        return s

    def mk_dropdown(label, options, val):
        d = server.gui.add_dropdown(label, tuple(options), initial_value=val)
        d.on_update(apply_sliders)
        return d

    def put(handle, value) -> None:
        """Write a value to a slider, clamped to its bounds."""
        lo = getattr(handle, "min", None)
        hi = getattr(handle, "max", None)
        if lo is not None and hi is not None:
            value = min(max(value, lo), hi)
        handle.value = value

    for i, label in enumerate(SLOT_LABELS):
        f = state["hand"].fingers[i]
        with server.gui.add_folder(label, expand_by_default=(i == 0)):
            act = server.gui.add_checkbox("active", initial_value=f.active)
            act.on_update(apply_sliders)
            sliders[f"f{i}"] = {
                "active": act,
                "n_joints": mk_slider("joints", 2, 6, 1,
                                      f.n_active_joints if f.active else 2),
                "face": mk_dropdown("palm face", P.PALM_FACES, "+z"),
                "u": mk_slider("pos along face u", -1.0, 1.0, 0.02, 0.0),
                "v": mk_slider("pos along face v", -1.0, 1.0, 0.02, 0.0),
                "roll": mk_slider("roll about finger axis (deg)",
                                  0, 360, 1.0, 0.0),
                "tilt": mk_slider("tilt off normal (deg)", 0, 60, 1.0, 0.0),
                "tilt_az": mk_slider("tilt direction (deg)", 0, 360, 1.0, 0.0),
                "mc": mk_slider("metacarpal (mm)", 0, 110, 0.5,
                                f.mc.length * 1000),
                "pp": mk_slider("proximal (mm)", 28, 66, 0.5,
                                f.pp_length * 1000),
                "mp_len": mk_slider("middle (mm)", 0, 44, 0.5,
                                    f.mp_length * 1000),
                "dp": mk_slider("distal (mm)", 16, 36, 0.5,
                                f.dp_length * 1000),
                "rscale": mk_slider("radius scale", 0.5, 1.4, 0.01,
                                    f.radius_scale),
                "aa": mk_slider("abduction +- (deg)", 0, 30, 0.5, 2.0),
            }

    def do_sample(_=None) -> None:
        nf = None if g_nf.value == "random" else int(g_nf.value)
        lo, hi = int(g_jmin.value), int(g_jmax.value)
        if lo > hi:
            lo, hi = hi, lo
        rng = random.Random(int(g_seed.value))
        try:
            state["hand"] = P.sample_valid(rng, name="live", n_fingers=nf,
                                           joints_per_finger=(lo, hi))
        except P.InvalidHand as exc:
            readout.content = f"**sampling failed**: {exc}"
            return
        push_sliders()
        render()
        g_seed.value = int(g_seed.value) + 1   # next click gives a new hand

    def do_sharpa(_=None) -> None:
        state["hand"] = P.SHARPA_LIKE
        push_sliders()
        render()

    b_sample.on_click(do_sample)
    b_sharpa.on_click(do_sharpa)

    push_sliders()
    render()

    print(f"\n[design_viz] serving on http://localhost:{args.port}")
    print(f"[design_viz] arm + table drawn once; only the hand rebuilds")
    print(f"[design_viz] red links are self-colliding at the home pose")
    print(f"[design_viz] 'Resample' draws a new hand and advances the seed")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[design_viz] stopped")


if __name__ == "__main__":
    main()
