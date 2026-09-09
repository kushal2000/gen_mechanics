"""Look at a hand: interactively in a browser, or as a PNG.

    python -m hand_sampler.viewer --seed 0            # viser, localhost:8080
    python -m hand_sampler.viewer png --seeds 6 --out seeds.png
    python -m hand_sampler.viewer png --lineage --seed 0 --out lineage.png

The point is to make the GRAMMAR inspectable. A mutation operator set is a claim
about which designs are adjacent to which, and that claim is far easier to check
by eye than by reading enumerations. ``--lineage`` renders a seed and each
successive mutation, which is the view that makes an operator's effect obvious:
everything holds still except the one thing it touched.

Links are coloured by their joint's ``theta``, which replaced the old FE/AA
enum: blue is pure flexion, orange pure abduction, everything between is a
design the old space could not name. Each joint is drawn as a short cylinder
lying along its hinge axis, so which way it turns is readable directly.

Both renderers share one kinematics path, so the PNG shows what the browser
would. The PNG half needs no browser and no server -- use it over a remote
shell, or to put a population in a figure.
"""

from __future__ import annotations

import argparse
import sys
import math
import random

import numpy as np
import trimesh
import viser

from hand_sampler import design_space
from hand_sampler import mutate_design
from hand_sampler import gen_init_pop
from hand_sampler.design_space import face_frame

FLEXION_RGB = (0.25, 0.45, 0.95)
ABDUCTION_RGB = (0.98, 0.60, 0.10)

# a joint marker: half the link's radius, so it reads as a hinge pin rather than a bulge, and...
JOINT_MARKER_RADIUS = 0.5 * design_space.CAPSULE_RADIUS
JOINT_MARKER_LENGTH = 2.0 * design_space.CAPSULE_RADIUS


def _align_z(v: np.ndarray) -> np.ndarray:
    """Rotation carrying +z onto ``v``.

    Both trimesh primitives used here are built along +z and centred on the
    origin. The spin about ``v`` is left unconstrained -- neither a capsule nor
    a cylinder shows it.
    """
    d = np.asarray(v, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    s, c = float(np.linalg.norm(axis)), float(z @ d)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    a = axis / s
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


def capsule_mesh(p0: np.ndarray, p1: np.ndarray, radius: float) -> trimesh.Trimesh:
    """A capsule whose TOTAL tip-to-tip extent spans p0 -> p1 exactly.

    Two things to get right, both of which were wrong in an earlier version:
      * ``trimesh.creation.capsule`` is CENTRED on the origin, so the transform
        has to put the segment MIDPOINT there, not p0. Placing p0 there shifts
        every link back by half its length -- which is what put joints in the
        middle of capsules and pushed base links through the palm.
      * the cylinder section must be shortened by 2r so the hemispherical caps
        land ON the joints rather than overhanging them. Then the tip of one link
        meets the base of the next exactly at the shared joint centre.
    """
    d = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(d))
    h = max(L - 2.0 * radius, 1e-6)
    mesh = trimesh.creation.capsule(height=h, radius=radius, count=(16, 16))

    T = np.eye(4)
    T[:3, :3] = _align_z(d)
    T[:3, 3] = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
    mesh.apply_transform(T)
    return mesh


def axis_mesh(centre: np.ndarray, axis: np.ndarray) -> trimesh.Trimesh:
    """A stub cylinder centred on a joint, lying along its hinge axis.

    The axis is a direction, not a ray -- the joint turns both ways about it --
    so the cylinder is centred on the joint and symmetric, rather than an arrow.
    """
    mesh = trimesh.creation.cylinder(radius=JOINT_MARKER_RADIUS,
                                     height=JOINT_MARKER_LENGTH, sections=16)
    T = np.eye(4)
    T[:3, :3] = _align_z(axis)
    T[:3, 3] = np.asarray(centre, float)
    mesh.apply_transform(T)
    return mesh


def theta_colour(theta: float) -> tuple[int, int, int]:
    """Blue at pure flexion, orange at pure abduction, blended between."""
    t = (theta % math.pi) / (math.pi / 2)
    t = t if t <= 1.0 else 2.0 - t          # fold: theta and pi-theta look alike
    rgb = [(1 - t) * a + t * b for a, b in zip(FLEXION_RGB, ABDUCTION_RGB)]
    return tuple(int(255 * c) for c in rgb)


def describe(hand: design_space.Hand, last_op: str | None) -> str:
    seps = design_space.mount_separations(hand)
    lines = [
        f"**{hand.n_fingers} fingers, {hand.n_joints} joints, "
        f"{hand.n_motors} motors**",
        "",
        f"palm  {hand.palm.thickness*1000:.0f} x {hand.palm.width*1000:.0f} x "
        f"{hand.palm.length*1000:.0f} mm",
        f"mount separation  {', '.join(f'{d*1000:.0f}' for d in seps)} mm"
        + ("   *(optimum measured at 40-50 mm)*" if seps else ""),
        "",
    ]
    for i, f in enumerate(hand.fingers):
        off = "" if all(abs(s.joint.phi - math.pi / 2) < 1e-9 for s in f.segments) \
              else "  **phi off-perpendicular**"
        lines.append(
            f"`{i}` {f.mount.face} u={f.mount.u:.2f} v={f.mount.v:.2f} | "
            f"{f.n_joints} joints, reach {f.reach*1000:.0f} mm{off}")
        lines.append("   " + "  ".join(
            f"[{math.degrees(s.joint.theta):.0f}d"
            + (f"{math.degrees(s.joint.offset):+.0f}o" if s.joint.offset else "")
            + f"/{s.length*1000:.0f}mm]"
            for s in f.segments))
    if last_op:
        lines += ["", f"last operator: **{last_op}**"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    rng = random.Random(args.seed)

    state: dict = {
        "hand": gen_init_pop.seed_population(args.seed, 1)[0],
        "lineage": [],          # (operator, rng_state) -- replayed, not snapshotted
        "seed": args.seed,
        "last_op": None,
        "angles": {},
    }

    with server.gui.add_folder("design"):
        info = server.gui.add_markdown("")
        op_dropdown = server.gui.add_dropdown("operator", ("(random)",) + mutate_design.OPERATORS)
        btn_mutate = server.gui.add_button("mutate")
        btn_undo = server.gui.add_button("undo")
        btn_reseed = server.gui.add_button("new seed")
        status = server.gui.add_markdown("")

    joint_folder = server.gui.add_folder("joints")
    sliders: list = []

    def rebuild_scene() -> None:
        hand = state["hand"]
        server.scene.reset()
        server.scene.add_grid("/grid", width=0.4, height=0.4, cell_size=0.02)

        t, w, l = hand.palm.extents
        palm = trimesh.creation.box(extents=(t, w, l))
        palm.apply_translation((0.0, 0.0, l / 2))
        server.scene.add_mesh_simple("/palm", palm.vertices, palm.faces,
                                     color=(120, 125, 135), opacity=0.55)

        for fi, finger in enumerate(hand.fingers):
            angles = state["angles"].get(fi, {})
            joints, capsules = design_space.forward_kinematics(finger, hand.palm, angles)
            axes = design_space.joint_axes(finger, hand.palm, angles)

            # the capsule carries the segment it belongs to; indexing by the capsule's own position would...
            for n, (p0, p1, r, si) in enumerate(capsules):
                mesh = capsule_mesh(p0, p1, r)
                server.scene.add_mesh_simple(
                    f"/f{fi}/link{n}", mesh.vertices, mesh.faces,
                    color=theta_colour(finger.segments[si].joint.theta))

            for si, (p, a) in enumerate(zip(joints[:-1], axes)):
                perp = abs(finger.segments[si].joint.phi - math.pi / 2) < 1e-9
                mesh = axis_mesh(p, a)
                server.scene.add_mesh_simple(
                    f"/f{fi}/j{si}", mesh.vertices, mesh.faces,
                    color=(30, 30, 35) if perp else (220, 40, 40))
            server.scene.add_icosphere(f"/f{fi}/tip", radius=design_space.CAPSULE_RADIUS * 0.45,
                                       position=tuple(joints[-1]),
                                       color=(250, 220, 60))

            # the face normal, so a finger offset away from it reads as offset
            p0 = design_space.mount_position(finger.mount, hand.palm)
            d = design_space.mount_direction(finger.mount, hand.palm)
            server.scene.add_spline_catmull_rom(
                f"/f{fi}/dir", np.stack([p0, p0 + 0.02 * d]),
                color=(200, 200, 210), line_width=2.0)

    def rebuild_sliders() -> None:
        for s in sliders:
            s.remove()
        sliders.clear()
        lo, hi = design_space.JOINT_LIMIT
        with joint_folder:
            for fi, finger in enumerate(state["hand"].fingers):
                for si, seg in enumerate(finger.segments):
                    label = (f"f{fi}.j{si}  "
                             f"{math.degrees(seg.joint.theta):.0f}d")
                    s = server.gui.add_slider(label, min=math.degrees(lo),
                                              max=math.degrees(hi), step=1.0,
                                              initial_value=0.0)

                    def on_change(_, fi=fi, si=si, handle=None) -> None:
                        state["angles"].setdefault(fi, {})[si] = math.radians(
                            handle.value)
                        rebuild_scene()

                    s.on_update(lambda ev, fi=fi, si=si, h=s: on_change(ev, fi, si, h))
                    sliders.append(s)

    def refresh(msg: str = "") -> None:
        info.content = describe(state["hand"], state["last_op"])
        status.content = msg
        rebuild_scene()
        rebuild_sliders()

    @btn_mutate.on_click
    def _(_) -> None:
        op = None if op_dropdown.value == "(random)" else op_dropdown.value
        before = state["hand"]
        child = mutate_design.mutate(rng, before, op)
        if child is None:
            state["last_op"] = None
            refresh(f"`{op or 'random'}` could not act on this hand "
                    f"(MutationImpossible) -- the hand is unchanged.")
            return
        state["lineage"].append((before, state["last_op"]))
        state["hand"] = child
        state["last_op"] = op or "(random)"
        state["angles"] = {}
        refresh(f"applied **{state['last_op']}**: {before.n_joints} -> "
                f"{child.n_joints} joints")

    @btn_undo.on_click
    def _(_) -> None:
        if not state["lineage"]:
            refresh("nothing to undo")
            return
        state["hand"], state["last_op"] = state["lineage"].pop()
        state["angles"] = {}
        refresh("undone")

    @btn_reseed.on_click
    def _(_) -> None:
        state["seed"] += 1
        state["hand"] = gen_init_pop.seed_population(state["seed"], 1)[0]
        state["lineage"].clear()
        state["last_op"] = None
        state["angles"] = {}
        refresh(f"seed {state['seed']}")

    refresh(f"seed {args.seed}")
    print(f"viewer on http://localhost:{args.port}")
    while True:
        import time
        time.sleep(1.0)


# --- PNG renderer: same kinematics, no browser, no server ------------------

import argparse
import math
import random

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection            # noqa: E402

from hand_sampler import design_space                             # noqa: E402
from hand_sampler import design_space                           # noqa: E402
from hand_sampler import mutate_design                               # noqa: E402
from hand_sampler import gen_init_pop                               # noqa: E402
from hand_sampler.viewer import (                                  # noqa: E402
    JOINT_MARKER_LENGTH,
    theta_colour,
)


def _palm_faces(palm: design_space.Palm) -> list[np.ndarray]:
    t, w, l = palm.extents
    x, y = t / 2, w / 2
    c = np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0],
                  [-x, -y, l], [x, -y, l], [x, y, l], [-x, y, l]], float)
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
           (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [c[list(f)] for f in idx]


def draw(ax, hand: design_space.Hand, title: str = "", flex: float = 0.0) -> None:
    ax.add_collection3d(Poly3DCollection(
        _palm_faces(hand.palm), facecolor=(0.47, 0.49, 0.53), alpha=0.30,
        edgecolor=(0.3, 0.3, 0.34), linewidths=0.5))

    for finger in hand.fingers:
        angles = {i: flex for i in range(finger.n_joints)}
        joints, capsules = design_space.forward_kinematics(finger, hand.palm, angles)
        # colour by the segment the capsule BELONGS to, which is carried on the capsule -- capsule and...
        for p0, p1, _, si in capsules:
            col = np.array(theta_colour(finger.segments[si].joint.theta)) / 255.0
            ax.plot(*zip(p0, p1), color=col, linewidth=6.0, solid_capstyle="round")

        # a joint is a stub along its own hinge axis, so which way it turns is visible; an...
        axes = design_space.joint_axes(finger, hand.palm, angles)
        half = JOINT_MARKER_LENGTH / 2.0
        for si, (p, a) in enumerate(zip(joints[:-1], axes)):
            perp = abs(finger.segments[si].joint.phi - math.pi / 2) < 1e-9
            ax.plot(*zip(p - half * a, p + half * a),
                    color="#1e1e23" if perp else "#d02828",
                    linewidth=3.0, solid_capstyle="round", zorder=5)
        ax.scatter(*joints[-1], s=34, c="#f5d93c", depthshade=False, zorder=6)

    span = 0.16
    ax.set_xlim(-span / 2, span / 2)
    ax.set_ylim(-span / 2, span / 2)
    ax.set_zlim(0.0, span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-58)
    if title:
        ax.set_title(title, fontsize=8, pad=0)


def _grid(items, out: str, cols: int = 3, flex: float = 0.0) -> None:
    rows = (len(items) + cols - 1) // cols
    fig = plt.figure(figsize=(3.3 * cols, 3.2 * rows), dpi=130)
    for i, (hand, title) in enumerate(items):
        draw(fig.add_subplot(rows, cols, i + 1, projection="3d"), hand, title, flex)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(items)} hands)")


def _png_main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=6, help="how many seeds to draw")
    ap.add_argument("--lineage", action="store_true",
                    help="render a mutation sequence instead of a seed population")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--flex", type=float, default=0.0,
                    help="drive every joint to this angle, in degrees. Fingers rest "
                         "pointing straight out of their face; opposition is what "
                         "FLEXION produces, so use this to see a hand close.")
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()

    if not args.lineage:
        pop = gen_init_pop.seed_population(args.seed, args.seeds)
        _grid([(h, f"seed {i}  |  {h.n_fingers}f  {h.n_joints}j")
               for i, h in enumerate(pop)], args.out, flex=math.radians(args.flex))
        return

    rng = random.Random(args.seed)
    hand = gen_init_pop.seed_population(args.seed, 1)[0]
    items = [(hand, f"seed {args.seed}  |  {hand.n_fingers}f  {hand.n_joints}j")]
    ops = list(mutate_design.OPERATORS)
    while len(items) <= args.steps:
        op = ops[rng.randrange(len(ops))]
        child = mutate_design.mutate(rng, hand, op)
        if child is None:
            continue
        hand = child
        items.append((hand, f"{op}  |  {hand.n_fingers}f  {hand.n_joints}j"))
    _grid(items, args.out, flex=math.radians(args.flex))


if __name__ == "__main__":
    # "png" selects the renderer and is consumed here, so each half keeps its own flags and...
    if len(sys.argv) > 1 and sys.argv[1] == "png":
        del sys.argv[1]
        _png_main()
    else:
        main()
