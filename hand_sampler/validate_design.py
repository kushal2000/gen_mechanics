"""The cheap validator: bounds, packing, and the articulation envelope.

First of two tiers. This one is O(1) per design with no sampling and runs on
EVERY mutation; ``gates/`` does configuration-dependent self-collision once,
before evaluation. They are not substitutes -- collision cannot express a
fabrication limit, and bounds cannot catch a self-intersection.

Loosening the grammar did not delete the ``minimal/`` constraints, it moved them
here, where they can be audited in one place.

Every check returns a REASON, not a bool, so an operator can reflect the one
offending value back into range rather than discarding the candidate -- which
would pile probability mass wherever candidates are easy to generate.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from hand_sampler import design_space
from hand_sampler.design_space import (
    base_capsules, mount_position, mount_uv_bounds, segment_distance,
)

_TOL = 1e-9


def _on_grid(value: float, quantum: float) -> bool:
    return abs(value - round(value / quantum) * quantum) < _TOL


# --- individual rules -------------------------------------------------------

def check_palm(palm: design_space.Palm) -> list[str]:
    out: list[str] = []
    for name, value, (lo, hi) in (
        ("thickness", palm.thickness, design_space.PALM_THICKNESS_RANGE),
        ("width", palm.width, design_space.PALM_WIDTH_RANGE),
        ("length", palm.length, design_space.PALM_LENGTH_RANGE),
    ):
        if not lo - _TOL <= value <= hi + _TOL:
            out.append(f"palm.{name} = {value:.4f} outside [{lo}, {hi}]")
        if not _on_grid(value, design_space.PALM_QUANTUM):
            out.append(f"palm.{name} = {value:.4f} off the {design_space.PALM_QUANTUM} m grid")
    return out


def check_segment(seg: design_space.Segment, where: str) -> list[str]:
    out: list[str] = []

    # One joint per link, so every length is a real link -- no zero-length coincident-joint case...
    if not design_space.MIN_LINK_LENGTH - _TOL <= seg.length <= design_space.MAX_LINK_LENGTH + _TOL:
        out.append(f"{where}.length = {seg.length:.4f} outside "
                   f"[{design_space.MIN_LINK_LENGTH}, {design_space.MAX_LINK_LENGTH}]")
    if not _on_grid(seg.length, design_space.LINK_QUANTUM):
        out.append(f"{where}.length = {seg.length:.4f} off the "
                   f"{design_space.LINK_QUANTUM} m grid")

    # Outside these ranges a hand has more than one spelling, which breaks design identity...
    if not 0.0 - _TOL <= seg.joint.theta < math.pi:
        out.append(f"{where}.theta = {seg.joint.theta:.4f} outside [0, pi)")
    # phi is PINNED at pi/2 for now (DESIGN.md 11), so this is an equality rather than a range.
    if abs(seg.joint.phi - math.pi / 2) > _TOL:
        out.append(f"{where}.phi = {seg.joint.phi:.4f}; phi is pinned at pi/2 "
                   f"(perpendicular hinges) until off-axis joints are enabled")
    # The zero offset is where the link sits at neutral, so it must be an angle the joint could...
    lo, hi = design_space.JOINT_LIMIT
    if not lo - _TOL <= seg.joint.offset <= hi + _TOL:
        out.append(f"{where}.offset = {seg.joint.offset:.4f} outside "
                   f"the joint's own travel [{lo:.4f}, {hi:.4f}]")

    for name, value in (("theta", seg.joint.theta), ("phi", seg.joint.phi),
                        ("offset", seg.joint.offset)):
        if not _on_grid(value, design_space.ANGLE_QUANTUM):
            out.append(f"{where}.{name} = {value:.4f} off the angle grid")
    return out


def check_finger(finger: design_space.Finger, i: int, palm: design_space.Palm) -> list[str]:
    """Rules for one finger. ``palm`` is REQUIRED, not defaulted: mount bounds
    depend on the face spans, so a default would validate against the wrong hand
    -- the same mount is illegal on a 60 mm face and legal on a 100 mm one.
    """
    where = f"finger[{i}]"
    out: list[str] = []

    if finger.n_joints > design_space.MAX_JOINTS_PER_FINGER:
        out.append(f"{where} has {finger.n_joints} joints, envelope allows "
                   f"{design_space.MAX_JOINTS_PER_FINGER}")

    lo_u, hi_u, lo_v, hi_v = mount_uv_bounds(finger.mount.face, palm)
    for name, value, lo, hi in (("u", finger.mount.u, lo_u, hi_u),
                                ("v", finger.mount.v, lo_v, hi_v)):
        if not lo - _TOL <= value <= hi + _TOL:
            out.append(f"{where}.mount.{name} = {value:.4f} outside "
                       f"[{lo:.3f}, {hi:.3f}]; a mount must stay "
                       f"{design_space.MOUNT_EDGE_MARGIN * 1000:.0f} mm from the face edge "
                       f"or its capsule hangs off the palm")

    for j, seg in enumerate(finger.segments):
        out.extend(check_segment(seg, f"{where}.segment[{j}]"))
    return out


def check_packing(hand: design_space.Hand) -> list[str]:
    """Mount separation -- two floors, because the geometry differs by face.

    ACROSS faces the floor is loose: mounts either side of an edge leave along
    different normals and can clear each other. WITHIN a face that fails
    completely -- fingers run parallel, so their base capsules overlap whenever
    the mounts are closer than the capsules are wide.

    Both are NECESSARY conditions, not sufficient ones. Whether fingers intersect
    along their length depends on configuration, which is the gate's job.
    """
    out: list[str] = []
    pos = [(f.mount.face, mount_position(f.mount, hand.palm)) for f in hand.fingers]
    for (face_a, pa), (face_b, pb) in combinations(pos, 2):
        d = float(np.linalg.norm(pa - pb))
        same = face_a == face_b
        floor = design_space.MIN_SAME_FACE_SEPARATION if same else design_space.MIN_MOUNT_SEPARATION
        if d < floor - _TOL:
            out.append(
                f"two mounts {d * 1000:.1f} mm apart"
                + (f" on {face_a}, minimum is {floor * 1000:.0f} mm "
                   f"(capsules touch at {2 * design_space.CAPSULE_RADIUS * 1000:.0f} mm)"
                   if same else
                   f" across faces, minimum is {floor * 1000:.0f} mm"))
            return out

    out += check_base_clearance(hand)
    return out


def check_base_clearance(hand: design_space.Hand) -> list[str]:
    """Proximal links must not intersect each other at the rest pose.

    Separation constrains where a finger STARTS, not where it POINTS, and two
    fingers rooted a legal distance apart can lean together until their base
    links cross. One closed-form segment-segment distance per pair, no sampling.

    Proximal links at rest only: collisions further out depend on flexion and
    belong to the gate. This rules out designs broken before they move.
    """
    out: list[str] = []
    caps = base_capsules(hand)
    floor = 2.0 * design_space.CAPSULE_RADIUS
    for (p0, p1), (q0, q1) in combinations(caps, 2):
        d = segment_distance(p0, p1, q0, q1)
        if d < floor - _TOL:
            out.append(f"two base links {d * 1000:.1f} mm apart at rest; capsules "
                       f"intersect below {floor * 1000:.0f} mm")
            break
    return out


def check_envelope(hand: design_space.Hand) -> list[str]:
    """The one constraint the simulator imposes back onto the genotype. Exceeding
    it is not a design that scores badly, it is one that cannot be loaded
    alongside the others in a single Articulation view.
    """
    out: list[str] = []
    if hand.n_fingers > design_space.MAX_FINGERS:
        out.append(f"{hand.n_fingers} fingers, envelope allows {design_space.MAX_FINGERS}")
    if hand.n_fingers < design_space.MIN_FINGERS:
        out.append(f"{hand.n_fingers} fingers, minimum is {design_space.MIN_FINGERS}")
    return out


# --- the whole hand ---------------------------------------------------------

def check(hand: design_space.Hand) -> list[str]:
    """Every reason this hand is not a legal design. Empty means legal."""
    out = check_envelope(hand)
    out += check_palm(hand.palm)
    for i, f in enumerate(hand.fingers):
        out += check_finger(f, i, hand.palm)
    out += check_packing(hand)
    return out


def is_valid(hand: design_space.Hand) -> bool:
    return not check(hand)


def require_valid(hand: design_space.Hand) -> design_space.Hand:
    """Raise with every reason at once. Operators reject silently and retry; this
    is for callers that believe they built something legal."""
    reasons = check(hand)
    if reasons:
        raise ValueError("invalid hand:\n  " + "\n  ".join(reasons))
    return hand
