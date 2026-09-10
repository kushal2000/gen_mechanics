"""Look at a hand: interactively in a browser, or as a PNG."""

from __future__ import annotations

import argparse
import pathlib
import sys
import math
import random

import numpy as np
import trimesh
import viser
from viser.extras import ViserUrdf

from hand_sampler import build
from hand_sampler import design_space
from hand_sampler import robot_param_constants as rpc
from hand_sampler import mutate_design
from hand_sampler import gen_init_pop
from hand_sampler.design_space import face_frame

# assets/urdf/table_narrow.urdf, at reset.table_reset_z. Surface at z = 0.53.
TABLE_EXTENTS = (0.475, 0.4, 0.3)
TABLE_CENTRE_Z = 0.38

FLEXION_RGB = (0.25, 0.45, 0.95)
ABDUCTION_RGB = (0.98, 0.60, 0.10)

# a joint marker: half the link's radius, so it reads as a hinge pin rather than a bulge, and...
JOINT_MARKER_RADIUS = 0.5 * design_space.CAPSULE_RADIUS
JOINT_MARKER_LENGTH = 2.0 * design_space.CAPSULE_RADIUS


def _align_z(v: np.ndarray) -> np.ndarray:
    """Rotation carrying +z onto ``v``."""
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
    """A capsule whose TOTAL tip-to-tip extent spans p0 -> p1 exactly."""
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
    """A stub cylinder centred on a joint, lying along its hinge axis."""
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
    ap.add_argument("--population", default=None,
                    help="a population JSON (or name) to open a stored design from")
    ap.add_argument("--experiment", default=None,
                    help="a drift experiment folder under assets/populations/: adds a "
                         "ROUND axis, so the walk itself is what you scrub through")
    ap.add_argument("--design", type=int, default=0,
                    help="which design to open")
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    rng = random.Random(args.seed)

    rounds = _experiment_rounds(args.experiment) if args.experiment else None
    load_round = _round_loader(rounds) if rounds else None
    if rounds:
        designs = load_round(len(rounds) - 1)          # open on the latest round
    else:
        designs = _load_designs(args.population) if args.population else None
    start = (designs[args.design] if designs
             else gen_init_pop.seed_population(args.seed, 1)[0])
    state: dict = {
        "hand": start,
        "lineage": [],          # (operator, rng_state) -- replayed, not snapshotted
        "seed": args.seed,
        "last_op": None,
        "angles": {},
        "context": True,
    }

    # Walking the stored population, when there is one to walk. The file is the
    # only way to reach design 8113 without re-running the sampler, and a slider
    # over 24576 of them is the point of having written it down.
    round_picker = None
    if rounds:
        with server.gui.add_folder(f"mutation rounds ({len(rounds)} snapshots)"):
            round_picker = server.gui.add_slider(
                "round", min=0, max=len(rounds) - 1, step=1,
                initial_value=len(rounds) - 1)
            round_info = server.gui.add_markdown("")
            btn_rescan = server.gui.add_button("rescan folder")

    picker = None
    if designs is not None:
        with server.gui.add_folder(f"population ({len(designs)} designs)"):
            picker = server.gui.add_slider("design", min=0, max=len(designs) - 1,
                                           step=1, initial_value=args.design)
            btn_prev = server.gui.add_button("prev")
            btn_next = server.gui.add_button("next")
            btn_random = server.gui.add_button("random")

    with server.gui.add_folder("design"):
        info = server.gui.add_markdown("")
        op_dropdown = server.gui.add_dropdown("operator", ("(random)",) + mutate_design.OPERATORS)
        btn_mutate = server.gui.add_button("mutate")
        btn_undo = server.gui.add_button("undo")
        btn_reseed = server.gui.add_button("new seed")
        cb_context = server.gui.add_checkbox("show arm + table", True)

    with server.gui.add_folder("pose"):
        btn_flex = server.gui.add_button("flex all")
        btn_unflex = server.gui.add_button("unflex all")
        status = server.gui.add_markdown("")

    @cb_context.on_update
    def _(_) -> None:
        state["context"] = bool(cb_context.value)
        rebuild_scene()

    @btn_flex.on_click
    def _(_) -> None:
        set_all_joints(math.degrees(design_space.JOINT_LIMIT[1]))

    @btn_unflex.on_click
    def _(_) -> None:
        set_all_joints(0.0)

    joint_folder = server.gui.add_folder("joints")
    sliders: list = []

    # ViserUrdf, not a pile of meshes rebuilt per frame. update_cfg() moves
    # transforms only, so dragging a joint no longer reloads the arm; and what
    # you are moving is the articulation the simulator authors -- palm slab,
    # capsules as cylinder-plus-caps, ghost slots -- rather than a drawing of
    # the design. The cost is the flexion/abduction colouring and the hinge
    # pins, which live on in the `png` renderer.
    scene: dict = {"urdf": None, "handle": None, "cfg": None, "names": []}

    def rebuild_static() -> None:
        server.scene.reset()
        scene["handle"] = None
        if state["context"]:
            table = trimesh.creation.box(extents=TABLE_EXTENTS)
            table.apply_translation((0.0, 0.0, TABLE_CENTRE_Z))
            server.scene.add_mesh_simple("/table", table.vertices, table.faces,
                                         color=(209, 143, 89))
            server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1)
        else:
            server.scene.add_grid("/grid", width=0.6, height=0.6, cell_size=0.05)
        # The arm's base, so the robot stands where the env puts it.
        server.scene.add_frame("/robot", show_axes=False,
                               position=tuple(rpc.BASE_POS) if state["context"] else (0.0, 0.0, 0.0))

    def rebuild_hand() -> None:
        """New geometry: a different design is a different URDF."""
        if scene["handle"] is not None:
            scene["handle"].remove()
        urdf = urdf_for(state["hand"])
        scene["urdf"] = urdf
        scene["handle"] = ViserUrdf(server, urdf, root_node_name="/robot")
        scene["names"] = list(urdf.actuated_joint_names)
        cfg = np.zeros(len(scene["names"]))
        for i, name in enumerate(scene["names"]):
            if name in rpc.ARM_DEFAULT_JOINT_POS:
                cfg[i] = rpc.ARM_DEFAULT_JOINT_POS[name]
        scene["cfg"] = cfg
        push_cfg()

    def push_cfg() -> None:
        """The whole point: transforms only, no mesh rebuild."""
        cfg = scene["cfg"].copy()
        for (fi, si), value in state["angles"].items():
            name = f"f{fi}_j{si}"
            if name in scene["names"]:
                cfg[scene["names"].index(name)] = value
        scene["handle"].update_cfg(cfg)

    def rebuild_scene() -> None:
        rebuild_static()
        rebuild_hand()

    def rebuild_sliders() -> None:
        for handle in sliders:
            handle.remove()
        sliders.clear()
        state["angles"].clear()
        lo, hi = design_space.JOINT_LIMIT
        with joint_folder:
            for fi, finger in enumerate(state["hand"].fingers):
                for si, seg in enumerate(finger.segments):
                    label = f"f{fi}.j{si}  {math.degrees(seg.joint.theta):.0f}d"
                    handle = server.gui.add_slider(
                        label, min=math.degrees(lo), max=math.degrees(hi),
                        step=1.0, initial_value=0.0)
                    sliders.append(handle)

                    def on_change(_, fi=fi, si=si, h=handle) -> None:
                        state["angles"][(fi, si)] = math.radians(h.value)
                        push_cfg()          # transforms only

                    handle.on_update(on_change)

    def set_all_joints(degrees: float) -> None:
        """Every joint of the hand together -- the pose that shows whether a
        design can actually close on something."""
        for handle in sliders:
            handle.value = degrees
        for fi, finger in enumerate(state["hand"].fingers):
            for si in range(finger.n_joints):
                state["angles"][(fi, si)] = math.radians(degrees)
        push_cfg()

    def refresh(msg: str = "") -> None:
        info.content = describe(state["hand"], state["last_op"])
        status.content = msg
        # Sliders first: they clear the old design's angles, and rebuild_hand
        # pushes a configuration built from whatever is in there.
        rebuild_sliders()
        rebuild_hand()

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

    # Once, before anything else draws: refresh() rebuilds only the hand, so
    # without this the table, the grid and the /robot base frame the URDF hangs
    # off are never created at all.
    rebuild_static()

    if round_picker is not None:
        def show_round(i: int) -> None:
            nonlocal designs
            i = max(0, min(int(i), len(rounds) - 1))
            designs = load_round(i)
            n_joints = sum(h.n_joints for h in designs) / len(designs)
            n_fing = sum(h.n_fingers for h in designs) / len(designs)
            round_info.content = (
                f"**round {rounds[i][0]}** of {rounds[-1][0]} &nbsp;|&nbsp; "
                f"{len(designs)} designs &nbsp;|&nbsp; "
                f"{n_joints:.2f} joints/hand &nbsp;|&nbsp; {n_fing:.2f} fingers/hand")
            picker.max = len(designs) - 1
            show_design(min(int(picker.value), len(designs) - 1))

        @round_picker.on_update
        def _(_) -> None:
            show_round(round_picker.value)

        @btn_rescan.on_click
        def _(_) -> None:
            # The walk writes rounds as it goes, so a folder grows under you.
            nonlocal rounds, load_round
            rounds = _experiment_rounds(args.experiment)
            load_round = _round_loader(rounds)
            round_picker.max = len(rounds) - 1
            show_round(round_picker.value)

    if picker is not None:
        def show_design(index: int) -> None:
            index = max(0, min(int(index), len(designs) - 1))
            state["hand"] = designs[index]
            state["lineage"].clear()      # a stored design is a fresh start,
            state["last_op"] = None       # not a step in the current lineage
            state["angles"] = {}
            refresh(f"design **{index}** of {args.population}")

        @picker.on_update
        def _(_) -> None:
            show_design(picker.value)

        @btn_prev.on_click
        def _(_) -> None:
            picker.value = max(0, int(picker.value) - 1)

        @btn_next.on_click
        def _(_) -> None:
            picker.value = min(len(designs) - 1, int(picker.value) + 1)

        @btn_random.on_click
        def _(_) -> None:
            picker.value = rng.randrange(len(designs))

        if round_picker is not None:
            show_round(len(rounds) - 1)
        else:
            show_design(args.design)
    else:
        refresh(f"seed {args.seed}")

    # main() returning tears the server down with it, so hold the thread here
    # rather than relying on the caller's shell to stay open.
    server.sleep_forever()
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


def urdf_for(hand) -> "yourdfpy.URDF":
    """The design as the ARM + HAND articulation, ready for ViserUrdf.

    Not the design-space drawing: this is `build.urdf_for_viewing` grafted onto
    the arm, so what you move is the thing the simulator authors -- palm slab,
    capsules as cylinder-plus-caps, ghost slots and all.
    """
    import tempfile
    import xml.etree.ElementTree as ET

    import yourdfpy

    from coevolution.pose_viewer import ARM_URDF_PATH, _graft_hand_onto_arm

    with tempfile.TemporaryDirectory() as tmp:
        text = build.urdf_for_viewing(hand, pathlib.Path(tmp) / "d.urdf").read_text()
    root = ET.fromstring(_graft_hand_onto_arm(text, ARM_URDF_PATH))
    # yourdfpy resolves a relative mesh against the file it loaded; there is no
    # file here, so make them absolute instead of writing into the asset tree.
    for mesh in root.findall(".//mesh"):
        name = mesh.get("filename")
        if name and not name.startswith(("/", "http")):
            mesh.set("filename", str((ARM_URDF_PATH.parent / name).resolve()))
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(ET.tostring(root, encoding="unicode"))
        path = f.name
    try:
        return yourdfpy.URDF.load(path, load_meshes=True, build_collision_scene_graph=False)
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def _experiment_rounds(directory) -> list:
    """``[(round, path)]`` for a drift experiment folder, in round order."""
    import re

    d = pathlib.Path(directory)
    if not d.is_dir():
        d = pathlib.Path("assets/populations") / directory
    if not d.is_dir():
        raise SystemExit(f"no experiment folder at {directory!r}")
    out = []
    for f in d.glob("round_*.json"):
        m = re.fullmatch(r"round_(\d+)", f.stem)
        if m:
            out.append((int(m.group(1)), f))
    if not out:
        raise SystemExit(f"{d} holds no round_NNNN.json snapshots")
    return sorted(out)


def _round_loader(rounds, keep: int = 3):
    """Load a round on demand, keeping a few. 24576 designs is 18 MB a round and
    a 500-round walk is 9 GB, so holding them all is not an option."""
    from hand_sampler import population_io

    cache: dict = {}
    order: list = []

    def load(index: int):
        r, path = rounds[index]
        if r not in cache:
            cache[r] = population_io.load_population(path)
            order.append(r)
            while len(order) > keep:
                cache.pop(order.pop(0), None)
        return cache[r]

    return load


def _load_designs(population: str):
    """Designs from a population file, given a path or a bare population name."""
    from hand_sampler import population_io

    path = pathlib.Path(population)
    if not path.exists():
        path = population_io.default_path(population)
    if not path.exists():
        raise SystemExit(
            f"no population at {population!r}; write one with\n"
            f"    python -m hand_sampler.population_io {population}")
    return population_io.load_population(path)


def _parse_indices(spec: str, count: int) -> list[int]:
    """``0,5,17`` or ``8100-8105``, or any mix. Out of range is an error, not a wrap."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = (int(v) for v in part.rsplit("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < count]
    if bad:
        raise SystemExit(f"design index out of range for {count} designs: {bad}")
    return out


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
    ap.add_argument("--population", default=None,
                    help="a population JSON written by hand_sampler.population_io, "
                         "or a bare population name to resolve under assets/populations")
    ap.add_argument("--designs", default=None,
                    help="which designs to draw from --population: indices and ranges, "
                         "e.g. 0,5,17 or 8100-8105")
    args = ap.parse_args()

    if args.population:
        hands = _load_designs(args.population)
        picked = (_parse_indices(args.designs, len(hands)) if args.designs
                  else list(range(min(args.seeds, len(hands)))))
        _grid([(hands[i], f"design {i}  |  {hands[i].n_fingers}f  {hands[i].n_joints}j")
               for i in picked], args.out, flex=math.radians(args.flex))
        return

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
