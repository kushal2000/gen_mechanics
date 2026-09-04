"""Interactive viewer: draw a hand, drive its joints, step through mutations.

    python -m hand_sampler.viewer --seed 0

The point is to make the GRAMMAR inspectable. A mutation operator set is a claim
about which designs are adjacent to which, and that claim is far easier to check
by eye than by reading enumerations.

Joints are coloured by ``theta``, which replaced the old FE/AA enum: blue is pure
flexion, orange pure abduction, everything between is a design the old space
could not name. A joint's zero offset shows in the readout as ``+30o``, and in
the scene as its link resting away from the grey face-normal stub. A joint
whose ``phi`` has left perpendicular-to-bone is ringed -- which cannot happen
while phi is pinned, but is ready for when it is not.
Undo replays the lineage rather than storing snapshots, so a broken inverse shows
up as the history and the scene disagreeing.
"""

from __future__ import annotations

import argparse
import math
import random

import numpy as np
import trimesh
import viser

from hand_sampler import genotype as G
from hand_sampler import kinematics as K
from hand_sampler import mutate as M
from hand_sampler import sample as S
from hand_sampler.kinematics import face_frame

FLEXION_RGB = (0.25, 0.45, 0.95)
ABDUCTION_RGB = (0.98, 0.60, 0.10)


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

    v = d / (L + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, v)
    s, c = float(np.linalg.norm(axis)), float(z @ v)
    T = np.eye(4)
    if s < 1e-9:
        T[:3, :3] = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        a = axis / s
        Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        T[:3, :3] = np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
    T[:3, 3] = (np.asarray(p0, float) + np.asarray(p1, float)) / 2.0
    mesh.apply_transform(T)
    return mesh


def theta_colour(theta: float) -> tuple[int, int, int]:
    """Blue at pure flexion, orange at pure abduction, blended between."""
    t = (theta % math.pi) / (math.pi / 2)
    t = t if t <= 1.0 else 2.0 - t          # fold: theta and pi-theta look alike
    rgb = [(1 - t) * a + t * b for a, b in zip(FLEXION_RGB, ABDUCTION_RGB)]
    return tuple(int(255 * c) for c in rgb)


def describe(hand: G.Hand, last_op: str | None) -> str:
    seps = K.mount_separations(hand)
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
        "hand": S.seed_population(args.seed, 1)[0],
        "lineage": [],          # (operator, rng_state) -- replayed, not snapshotted
        "seed": args.seed,
        "last_op": None,
        "angles": {},
    }

    with server.gui.add_folder("design"):
        info = server.gui.add_markdown("")
        op_dropdown = server.gui.add_dropdown("operator", ("(random)",) + M.OPERATORS)
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
            joints, capsules = K.forward_kinematics(finger, hand.palm, angles)

            # the capsule carries the segment it belongs to; indexing by the
            # capsule's own position would silently read the wrong joint.
            for n, (p0, p1, r, si) in enumerate(capsules):
                mesh = capsule_mesh(p0, p1, r)
                server.scene.add_mesh_simple(
                    f"/f{fi}/link{n}", mesh.vertices, mesh.faces,
                    color=theta_colour(finger.segments[si].joint.theta))

            for si, p in enumerate(joints[:-1]):
                perp = abs(finger.segments[si].joint.phi - math.pi / 2) < 1e-9
                server.scene.add_icosphere(
                    f"/f{fi}/j{si}",
                    radius=G.CAPSULE_RADIUS * (0.55 if perp else 0.75),
                    position=tuple(p),
                    color=(30, 30, 35) if perp else (220, 40, 40))
            server.scene.add_icosphere(f"/f{fi}/tip", radius=G.CAPSULE_RADIUS * 0.45,
                                       position=tuple(joints[-1]),
                                       color=(250, 220, 60))

            # the face normal, so a finger offset away from it reads as offset
            p0 = K.mount_position(finger.mount, hand.palm)
            d = K.mount_direction(finger.mount, hand.palm)
            server.scene.add_spline_catmull_rom(
                f"/f{fi}/dir", np.stack([p0, p0 + 0.02 * d]),
                color=(200, 200, 210), line_width=2.0)

    def rebuild_sliders() -> None:
        for s in sliders:
            s.remove()
        sliders.clear()
        lo, hi = G.JOINT_LIMIT
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
        child = M.mutate(rng, before, op)
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
        state["hand"] = S.seed_population(state["seed"], 1)[0]
        state["lineage"].clear()
        state["last_op"] = None
        state["angles"] = {}
        refresh(f"seed {state['seed']}")

    refresh(f"seed {args.seed}")
    print(f"viewer on http://localhost:{args.port}")
    while True:
        import time
        time.sleep(1.0)


if __name__ == "__main__":
    main()
