"""The cheap validator: bounds, packing, and the articulation envelope."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from hand_sampler import design_space
from hand_sampler.design_space import (
    base_capsules, mount_position, segment_distance,
)

_TOL = 1e-9


def _on_grid(value: float, quantum: float) -> bool:
    return abs(value - round(value / quantum) * quantum) < _TOL


# --- individual rules -------------------------------------------------------

def check_palm(palm: design_space.Palm) -> list[str]:
    """Thickness is all a palm stores. Its outline is derived from the mounts,
    so what used to be checked here is checked on them -- see check_layout."""
    out: list[str] = []
    lo, hi = design_space.PALM_THICKNESS_RANGE
    if not lo - _TOL <= palm.thickness <= hi + _TOL:
        out.append(f"palm.thickness = {palm.thickness:.4f} outside [{lo}, {hi}]")
    if not _on_grid(palm.thickness, design_space.PALM_QUANTUM):
        out.append(f"palm.thickness = {palm.thickness:.4f} off the "
                   f"{design_space.PALM_QUANTUM} m grid")
    return out


def check_layout(hand: design_space.Hand) -> list[str]:
    """Every mount in the annulus and on its grids.

    This is what bounds complexity now that a palm has no size of its own: a
    finger starts between PALM_MIN_RADIUS and MAX_MOUNT_RADIUS of the centre,
    and inside that ring MIN_MOUNT_SEPARATION decides how many will fit.
    """
    out: list[str] = []
    turn = 2.0 * math.pi
    for i, f in enumerate(hand.fingers):
        m = f.mount
        where = f"finger[{i}].mount"
        # The ring is a bound on the DISTANCE, which is why it survived the grid
        # going cartesian: it is how far a finger may be from the centre.
        if not design_space.PALM_MIN_RADIUS - _TOL <= m.radius <= \
                design_space.MAX_MOUNT_RADIUS + _TOL:
            out.append(f"{where} sits {m.radius * 1000:.0f} mm out, outside the "
                       f"[{design_space.PALM_MIN_RADIUS * 1000:.0f}, "
                       f"{design_space.MAX_MOUNT_RADIUS * 1000:.0f}] mm ring")
        for name, value in (("y", m.y), ("z", m.z)):
            if not _on_grid(value, design_space.PALM_QUANTUM):
                out.append(f"{where}.{name} = {value:.4f} off the "
                           f"{design_space.PALM_QUANTUM} m grid")
        if not -_TOL <= m.facing < turn - _TOL:
            out.append(f"{where}.facing = {m.facing:.4f} outside [0, 2pi); a "
                       f"full turn is the same direction again and a hand "
                       f"gets one spelling")
        if not _on_grid(m.facing, design_space.ANGLE_QUANTUM):
            out.append(f"{where}.facing = {m.facing:.4f} off the angle grid")
    return out


def check_arm_clearance(hand: design_space.Hand, links=None) -> list[str]:
    """Nothing the hand owns may reach behind the arm's own face.

    The arm is entirely behind ARM_FACE_Z in the palm frame and the hand is
    bolted to its flange, so one plane settles it for the palm plate and every
    link at once.

    This is what a wedge on the mount BEARING could not do. A bearing says where
    a finger starts; what reaches the arm is where it points and how far it
    goes, and those are facing, lean and length. Without this rule 62% of the
    hands a mutation walk reaches have something inside the arm after fifty
    steps and 78% after four hundred, as deep as 178 mm -- and the wedge was
    never what stood between: removing it moved that number by five points.

    Nearly free, because every operator shuffles its candidates and takes the
    first that validates: a site that would reach the arm becomes a different
    site rather than a failed move. On one population of 60 parents the rule
    costs 0.2 points of mutation success overall, and only perturb_length moves
    at all.

    A NECESSARY condition at the rest pose, like check_base_clearance: the arm's
    own earlier links swing in this frame as its last joint turns, and what a
    hand sweeps once its joints move is the gate's business.
    """
    z, who = design_space.rearmost(hand, links)
    if z >= design_space.ARM_FACE_Z - _TOL:
        return []
    whose = "the palm" if who is None else f"finger {who}"
    return [f"{whose} reaches to z = {z * 1000:.0f} mm, "
            f"{(design_space.ARM_FACE_Z - z) * 1000:.0f} mm inside the arm, "
            f"which stops at {design_space.ARM_FACE_Z * 1000:.0f} mm"]


def check_segment(seg: design_space.Segment, where: str, terminal: bool) -> list[str]:
    """``terminal`` is required, not defaulted: the same link is legal in one
    position and illegal in another, and a default would decide that silently."""
    out: list[str] = []

    # One joint per link, so every length is a real link. The floor depends on
    # whether the link seats a motor: every link drives the joint distal to it,
    # except the last, which has no child joint to drive.
    floor = (design_space.MIN_DISTAL_LINK_LENGTH if terminal
             else design_space.MIN_LINK_LENGTH)
    if not floor - _TOL <= seg.length <= design_space.MAX_LINK_LENGTH + _TOL:
        out.append(f"{where}.length = {seg.length:.4f} outside "
                   f"[{floor}, {design_space.MAX_LINK_LENGTH}]"
                   + ("" if terminal else " for a motor-carrying link"))
    if not _on_grid(seg.length, design_space.LINK_QUANTUM):
        out.append(f"{where}.length = {seg.length:.4f} off the "
                   f"{design_space.LINK_QUANTUM} m grid")

    if seg.joint.kind not in design_space.JOINT_KINDS:
        out.append(f"{where}.joint.kind = {seg.joint.kind!r} is not one of "
                   f"{design_space.JOINT_KIND_NAMES}")
    # A generated joint has no assembly angle: travel is symmetric about the rest
    # pose, and the link is aimed by its LEAN. offset survives only on hands that
    # were measured rather than drawn, which carry an axis_override too.
    if abs(seg.joint.offset) > _TOL and seg.joint.axis_override is None:
        out.append(f"{where}.joint.offset = {seg.joint.offset:.4f}; a generated "
                   f"joint has no assembly angle, the lean aims the link")

    if not 0 <= seg.lean < design_space.N_LEANS:
        out.append(f"{where}.lean = {seg.lean} is not one of the "
                   f"{design_space.N_LEANS} mounting directions")
    return out


def check_finger(finger: design_space.Finger, i: int, palm: design_space.Palm) -> list[str]:
    """Rules for one finger."""
    where = f"finger[{i}]"
    out: list[str] = []

    if finger.n_joints > design_space.MAX_JOINTS_PER_FINGER:
        out.append(f"{where} has {finger.n_joints} joints, envelope allows "
                   f"{design_space.MAX_JOINTS_PER_FINGER}")

    if finger.reach > design_space.MAX_FINGER_LENGTH + _TOL:
        out.append(f"{where} reaches {finger.reach * 1000:.0f} mm fully extended, "
                   f"maximum is {design_space.MAX_FINGER_LENGTH * 1000:.0f} mm")

    last = finger.n_joints - 1
    for j, seg in enumerate(finger.segments):
        out.extend(check_segment(seg, f"{where}.segment[{j}]", terminal=(j == last)))
    return out


def check_packing(hand: design_space.Hand, links=None) -> list[str]:
    """One separation floor for every pair of mounts, whatever faces they are on.

    A NECESSARY condition, not a sufficient one: whether fingers intersect along
    their length depends on configuration, which is the gate's job. What this
    rules out is the pair that cannot work at any configuration, because their
    base capsules already overlap where they leave the palm.
    """
    out: list[str] = []
    floor = design_space.MIN_MOUNT_SEPARATION
    pos = [mount_position(f.mount) for f in hand.fingers]
    for pa, pb in combinations(pos, 2):
        d = float(np.linalg.norm(pa - pb))
        if d < floor - _TOL:
            out.append(f"two mounts {d * 1000:.1f} mm apart, minimum is "
                       f"{floor * 1000:.0f} mm (capsules touch at "
                       f"{2 * design_space.CAPSULE_RADIUS * 1000:.0f} mm)")
            return out

    out += check_base_clearance(hand, links)
    return out


def check_base_clearance(hand: design_space.Hand, links=None) -> list[str]:
    """No two links may intersect at the rest pose.

    This used to look at ``base_capsules`` -- the proximal link of each finger
    and nothing else. That is every pair a two-joint hand HAS, so generation 0
    never noticed; mutation makes longer fingers, and at 5.7 joints/hand 1.9% of
    designs had non-adjacent links overlapping by as much as 19.6 mm, a link
    being 20.3 mm across. The simulator then starts every episode resolving a
    contact the design was told was legal, and it gets worse the further the
    population drifts.

    Consecutive links within a finger are excluded: they meet at their shared
    joint by construction, so their core segments touch at a point whatever the
    design says.
    """
    out: list[str] = []
    links = design_space.rest_capsules(hand) if links is None else links
    floor = 2.0 * design_space.CAPSULE_RADIUS
    for (fi, si, p0, p1), (fj, sj, q0, q1) in combinations(links, 2):
        if fi == fj and abs(si - sj) <= 1:
            continue
        d = segment_distance(p0, p1, q0, q1)
        if d < floor - _TOL:
            who = (f"finger {fi} link {si} and finger {fj} link {sj}" if fi != fj
                   else f"finger {fi} links {si} and {sj}")
            out.append(f"{who} are {d * 1000:.1f} mm apart at rest; capsules "
                       f"intersect below {floor * 1000:.0f} mm")
            break
    return out


def check_envelope(hand: design_space.Hand) -> list[str]:
    """The one constraint the simulator imposes back onto the genotype."""
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
    out += check_layout(hand)
    for i, f in enumerate(hand.fingers):
        out += check_finger(f, i, hand.palm)
    # Both of these walk every link at the rest pose, so the walk happens once
    # here and they share it.
    links = design_space.rest_capsules(hand)
    out += check_packing(hand, links)
    out += check_arm_clearance(hand, links)
    return out


def is_valid(hand: design_space.Hand) -> bool:
    return not check(hand)


def require_valid(hand: design_space.Hand) -> design_space.Hand:
    """Raise with every reason at once."""
    reasons = check(hand)
    if reasons:
        raise ValueError("invalid hand:\n  " + "\n  ".join(reasons))
    return hand
