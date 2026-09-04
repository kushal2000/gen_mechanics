"""Mutation operators: the neighbourhood structure local search walks.

A sampler answers "draw me a hand"; evolution needs "given this hand, which hands
are one step away", and the answer to the second defines the search topology.

THREE PROPERTIES, ALL DELIBERATE.

*Closure.* Every operator returns a hand passing ``validate.check`` or raises.

*Unit steps.* Every structural operator changes joint count by exactly +-1. If
the only structural move added a whole 3-joint finger, complexity would jump in
threes and the performance-vs-motors front -- the headline plot -- would have
holes in it.

*Exact inverses.* Add-then-remove returns the ORIGINAL hand, not a nearby one.
That is what the angle and length grids buy, and it is load-bearing for
proceeding without an explicit parsimony penalty: the argument that added
complexity must pay for itself assumes additions and removals are equally
available, which is not automatic. ``Stats`` exists to catch a drift.

NINE OPERATORS. Two rules decide the count. An operator REACHABLE BY CHAINING
others earns nothing and adds a second place the same rule can drift -- which is
why ``remount`` is gone, folded into ``move_mount`` once a step overflowing a
face carries onto the next, and why a mount-orientation operator is gone: a
joint's ZERO OFFSET reproduces exactly what it did from the base joint, and does
more from any other joint.

But operators that differ only in WHERE they attach are kept apart, even though a
tree makes them one operation. Splitting a link and growing a new finger were
briefly a single ``add_node`` drawing uniformly over pooled sites, and that hid
two things: a new finger competed against every splittable link, so it was rare;
and once the palm filled, splits kept succeeding under the same name, masking
that palm capacity had run out. Separate operators make the mutation mix
controllable and the failure modes legible.

Reflection rather than clipping when a step leaves a range, and mount steps in
METRES rather than u/v fractions -- both because the alternative biases the step
distribution by position. The mount carries no orientation at all: a finger
leaves along its face normal, and aiming it is the base joint's offset.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

import numpy as np

from hand_sampler import genotype as G
from hand_sampler import validate as V
from hand_sampler.kinematics import (
    face_frame, face_from_normal, mount_direction, mount_position,
    mount_uv_bounds,
)

OPERATORS: tuple[str, ...] = (
    # structural -- these move complexity, +-1 joint each
    "split_link", "merge_links", "add_finger", "remove_finger",
    # parametric -- complexity fixed
    "perturb_axis", "perturb_offset", "perturb_length", "move_mount",
    "perturb_palm",
)

STRUCTURAL: tuple[str, ...] = OPERATORS[:4]

_REDRAWS = 8
"""How many independent draws a whole-hand operator tries before giving up.
Each joint or link reflects into range on its own, so a rejection means a
whole-hand rule caught it and a different draw is likely to pass."""

MOUNT_STEP_M = 0.005
"""One step across a palm face, in metres. See the module docstring."""

_GROWS: tuple[tuple[str, str], ...] = (("split_link", "merge_links"),
                                       ("add_finger", "remove_finger"))
"""Structural pairs, growing operator first. Every entry must appear in
``Stats.ratchet``; naming them once here is what stopped a filter on "add"
silently dropping the split/merge pair when it was added."""

_INVERSE = {a: b for a, b in _GROWS} | {b: a for a, b in _GROWS}


class MutationImpossible(ValueError):
    """This operator cannot act on this hand. Distinct from "acted and produced
    something invalid", which is a bug rather than a state of the world."""


# --- numeric helpers --------------------------------------------------------

def reflect(x: float, lo: float, hi: float) -> float:
    """Fold ``x`` back into ``[lo, hi]``. Reflection, not clipping."""
    if hi <= lo:
        return lo
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    return lo + (y if y <= span else 2.0 * span - y)


def snap(x: float, quantum: float) -> float:
    return round(x / quantum) * quantum


def wrap_theta(theta: float) -> float:
    """theta is periodic with period pi -- a hinge and its negation coincide."""
    return theta % math.pi


# --- structural -------------------------------------------------------------

def split_link(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Divide one link in two, inserting a joint. +1 joint, within a finger.

    Needs a link of at least 2 x MIN_LINK_LENGTH to divide, so a finger of short
    links cannot deepen until ``perturb_length`` grows one. That coupling between
    depth and length is geometry, not a limitation (DESIGN.md 5).
    """
    moves = []
    for fi, finger in enumerate(hand.fingers):
        if finger.n_joints >= G.MAX_JOINTS_PER_FINGER:
            continue
        for si, seg in enumerate(finger.segments):
            n_lo = round(G.MIN_LINK_LENGTH / G.LINK_QUANTUM)
            n_tot = round(seg.length / G.LINK_QUANTUM)
            for n_a in range(n_lo, n_tot - n_lo + 1):
                moves.append((fi, si, n_a * G.LINK_QUANTUM))
    if not moves:
        raise MutationImpossible("no link is long enough to divide")

    rng.shuffle(moves)
    for fi, si, a in moves:
        finger = hand.fingers[fi]
        segments = list(finger.segments)
        old = segments[si]
        segments[si] = G.Segment(old.joint, a)
        segments.insert(si + 1, G.Segment(
            G.Joint(theta=_draw_theta(rng), phi=math.pi / 2), old.length - a))
        out = G.with_finger(hand, fi, replace(finger, segments=tuple(segments)))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no split produced a valid hand")


def merge_links(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Join two links, removing the joint between them. -1 joint, within a
    finger, and the exact inverse of ``split_link``.

    Only acts on fingers with at least two joints: emptying a finger is
    ``remove_finger``'s job. The merge preserves reach, so split-then-merge
    returns the original link rather than a shorter finger.
    """
    moves = [(fi, si) for fi, f in enumerate(hand.fingers) if f.n_joints >= 2
             for si in range(f.n_joints)]
    if not moves:
        raise MutationImpossible("every finger has a single joint")

    rng.shuffle(moves)
    for fi, si in moves:
        out = G.with_finger(hand, fi, _merge_out(hand.fingers[fi], si))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no merge stayed within bounds")


def add_finger(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Attach a new SINGLE-JOINT finger to the palm. +1 joint.

    Single-joint so the step stays at one joint and pairs exactly with
    ``remove_finger``. This operator is why the package exists: the previous
    design space had no finger-count operator and recorded that absence as
    permanent for any descended population, so a topology class that died could
    not come back.
    """
    if hand.n_fingers >= G.MAX_FINGERS:
        raise MutationImpossible(f"already at {G.MAX_FINGERS} fingers")
    out = _new_finger(rng, hand)
    if out is None:
        raise MutationImpossible("no room on the palm for another mount")
    return _accept(out, "add_finger")


def remove_finger(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Delete a SINGLE-JOINT finger. -1 joint, the exact inverse of ``add_finger``.

    Restricted to one-joint fingers so the step stays at one joint and stays
    invertible. A deeper finger is removed by merging it down first, which is
    more local and reversible at every intermediate step.
    """
    if hand.n_fingers <= G.MIN_FINGERS:
        raise MutationImpossible(f"already at the minimum of {G.MIN_FINGERS}")
    candidates = [i for i, f in enumerate(hand.fingers) if f.n_joints == 1]
    if not candidates:
        raise MutationImpossible("no single-joint finger; merge one down first")

    rng.shuffle(candidates)
    for i in candidates:
        out = replace(hand, fingers=tuple(f for k, f in enumerate(hand.fingers)
                                          if k != i))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no finger could be removed")


def _merge_out(finger: G.Finger, si: int) -> G.Finger:
    """Drop segment ``si``, folding its link into a neighbour.

    Proximal first (the exact inverse of a split), else distal, else the proximal
    merge CLAMPED to the ceiling. The clamp is unreachable from ``split_link`` -- a
    split needs its original within the ceiling, so the parts always merge back
    inside it -- so it costs nothing in exactness. Without it, a finger whose
    adjacent links summed past the ceiling could not shed that joint at all,
    making removal unavailable exactly where links are long.
    """
    segments = list(finger.segments)
    freed = segments.pop(si).length

    proximal = si - 1 if si > 0 else 0
    candidates = [proximal]
    distal = si if si < len(segments) else None      # post-pop index of si + 1
    if distal is not None and distal != proximal:
        candidates.append(distal)

    for into in candidates:
        merged = segments[into].length + freed
        if merged <= G.MAX_LINK_LENGTH + 1e-9:
            segments[into] = G.Segment(segments[into].joint, merged)
            return replace(finger, segments=tuple(segments))

    segments[proximal] = G.Segment(segments[proximal].joint, G.MAX_LINK_LENGTH)
    return replace(finger, segments=tuple(segments))


def _draw_theta(rng: random.Random) -> float:
    n = round(math.pi / G.ANGLE_QUANTUM)
    return (rng.randrange(n) * G.ANGLE_QUANTUM) % math.pi


def _new_finger(rng: random.Random, hand: G.Hand) -> G.Hand | None:
    """A fresh single-joint finger where there is room, or None if nowhere.

    Single-joint so the step stays at one joint and stays invertible. Placement
    is ENUMERATED, not rejection-sampled: on a crowded palm nearly every random
    draw violates separation, and a retry budget makes ``add_finger`` quietly stop
    being able to add fingers as space runs out -- a reachability hole wearing a
    timeout's clothing. Enumeration also treats every legal site alike.
    """
    sites = _free_mount_sites(hand)
    if not sites:
        return None

    n_len = round((G.MAX_LINK_LENGTH - G.MIN_LINK_LENGTH) / G.LINK_QUANTUM)
    rng.shuffle(sites)
    for face, u, v in sites:
        finger = G.Finger(
            mount=G.Mount(face, u, v),
            segments=(G.Segment(
                G.Joint(theta=_draw_theta(rng), phi=math.pi / 2),
                length=G.MIN_LINK_LENGTH + rng.randint(0, n_len) * G.LINK_QUANTUM),),
        )
        out = replace(hand, fingers=hand.fingers + (finger,))
        if V.is_valid(out):
            return out
    return None


MOUNT_GRID_M = 0.005
"""Spacing of candidate mount sites, in metres on the face. Fine enough that a
gap large enough to hold a finger is not missed, coarse enough that enumerating
every face is cheap."""


def _free_mount_sites(hand: G.Hand) -> list[tuple[str, float, float]]:
    """Every grid site on the palm with room for another mount.

    Laid out inside the edge margin, so a candidate never sits where a finger
    would hang off the palm. Only the separation floors are checked here, those
    being the constraints that depend on the other fingers; the chosen site still
    goes through the full validator. Vectorised per face because ``add_finger``
    runs on every mutation attempt.
    """
    if not hand.fingers:
        return []
    existing = np.array([mount_position(f.mount, hand.palm) for f in hand.fingers])
    same_face = np.array([f.mount.face for f in hand.fingers])
    sites: list[tuple[str, float, float]] = []

    for face in G.FINGER_FACES:
        centre, _, t_u, t_v, span_u, span_v = face_frame(face, hand.palm)
        lo_u, hi_u, lo_v, hi_v = mount_uv_bounds(face, hand.palm)
        n_u = max(1, int((hi_u - lo_u) * span_u / MOUNT_GRID_M))
        n_v = max(1, int((hi_v - lo_v) * span_v / MOUNT_GRID_M))

        us = np.linspace(lo_u, hi_u, n_u + 1)
        vs = np.linspace(lo_v, hi_v, n_v + 1)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        flat_u, flat_v = uu.ravel(), vv.ravel()

        # positions[k] = centre + (u-0.5) span_u t_u + (v-0.5) span_v t_v
        pos = (centre
               + np.outer((flat_u - 0.5) * span_u, t_u)
               + np.outer((flat_v - 0.5) * span_v, t_v))

        d = np.linalg.norm(pos[:, None, :] - existing[None, :, :], axis=2)
        floors = np.where(same_face == face,
                          G.MIN_SAME_FACE_SEPARATION, G.MIN_MOUNT_SEPARATION)
        ok = (d >= floors[None, :]).all(axis=1)

        sites.extend((face, float(u), float(v))
                     for u, v in zip(flat_u[ok], flat_v[ok]))
    return sites


# --- parametric -------------------------------------------------------------

def perturb_axis(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Step EVERY joint's theta by one quantum, each independently up or down.

    A whole-hand move rather than a single-joint one, so it explores orientation
    far faster than one joint at a time -- at the cost of locality, since a
    20-joint hand has all 20 axes changed at once. Redrawn a few times if the
    result does not validate, which is cheap because theta reflects into range
    per joint and only the whole-hand rules can fail.

    Only theta. ``phi`` is pinned at pi/2, first on the held-back list
    (DESIGN.md 11.4).
    """
    for _ in range(_REDRAWS):
        fingers = []
        for f in hand.fingers:
            segments = tuple(
                G.Segment(
                    G.Joint(snap(wrap_theta(sg.joint.theta
                                            + G.ANGLE_QUANTUM * rng.choice((-1, 1))),
                                 G.ANGLE_QUANTUM) % math.pi,
                            sg.joint.phi, sg.joint.offset),
                    sg.length)
                for sg in f.segments)
            fingers.append(replace(f, segments=segments))
        out = replace(hand, fingers=tuple(fingers))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no whole-hand axis perturbation validated")


def perturb_length(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Step EVERY link by one quantum, each independently up or down.

    A whole-hand move, like ``perturb_axis``. The only operator that changes
    total reach -- ``split_link`` divides and ``merge_links`` rejoins, both
    reach-preserving -- so making it act on every link is what lets a hand grow
    or shrink at a useful rate rather than one 5 mm step per mutation.

    Each length reflects into ``[MIN_LINK_LENGTH, MAX_LINK_LENGTH]`` on its own,
    so only the whole-hand rules (base clearance, packing) can reject a draw.
    """
    for _ in range(_REDRAWS):
        fingers = []
        for f in hand.fingers:
            segments = tuple(
                G.Segment(sg.joint,
                          snap(reflect(sg.length
                                       + G.LINK_QUANTUM * rng.choice((-1, 1)),
                                       G.MIN_LINK_LENGTH, G.MAX_LINK_LENGTH),
                               G.LINK_QUANTUM))
                for sg in f.segments)
            fingers.append(replace(f, segments=segments))
        out = replace(hand, fingers=tuple(fingers))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no whole-hand length perturbation validated")


def move_mount(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Slide one mount across the palm surface, CROSSING FACE EDGES.

    Steps in metres, not u/v fractions. A step that overflows a face carries onto
    the face across that edge, which is what lets this one operator do the work a
    separate ``remount`` would: on an axis-aligned box a face's tangents are its
    neighbours' normals, so no cube net is needed.
    """
    order = list(range(hand.n_fingers))
    rng.shuffle(order)
    for fi in order:
        finger = hand.fingers[fi]
        du_m = MOUNT_STEP_M * rng.choice((-1, 0, 1))
        dv_m = MOUNT_STEP_M * rng.choice((-1, 0, 1))
        if du_m == 0.0 and dv_m == 0.0:
            continue

        mount = _step_mount(finger.mount, hand.palm, du_m, dv_m)
        if mount is None or mount == finger.mount:
            continue
        out = G.with_finger(hand, fi, replace(finger, mount=mount))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no mount could move without violating a bound")


def _step_mount(mount: G.Mount, palm: G.Palm, du_m: float, dv_m: float
                ) -> G.Mount | None:
    """One step on the palm surface, wrapping onto a neighbouring face if needed.

    Movement is bounded by MOUNT_EDGE_MARGIN, and a crossing JUMPS that band
    rather than walking through it: the margin forbids a mount being within a
    capsule radius of an edge, so there is no legal position AT one to pass
    through.

    A crossing needs nothing done to the finger's aim. A finger leaves along its
    face normal and its tilt is the base joint's offset, so moving to a new face
    rotates the world direction by the angle between normals while leaving the
    tilt relative to the face untouched -- which is what a mount orientation had
    to be carried across by hand.
    """
    _, _, t_u, t_v, span_u, span_v = face_frame(mount.face, palm)
    lo_u, hi_u, lo_v, hi_v = mount_uv_bounds(mount.face, palm)
    u, v = mount.u + du_m / span_u, mount.v + dv_m / span_v

    if lo_u <= u <= hi_u and lo_v <= v <= hi_v:
        return replace(mount, u=u, v=v)

    if u > hi_u:
        cross, u = t_u, hi_u
    elif u < lo_u:
        cross, u = -t_u, lo_u
    elif v > hi_v:
        cross, v = t_v, hi_v
    else:
        cross, v = -t_v, lo_v

    face = face_from_normal(cross)
    if face is None:
        # The step points at the wrist or a large face. CLAMP rather than refuse:
        # the thin axis has a 5 mm band against a 5 mm step, so refusing froze it.
        clamped = replace(mount, u=u, v=v)
        return None if clamped == mount else clamped

    landing = mount_position(replace(mount, u=u, v=v), palm)
    centre, _, t_u2, t_v2, span_u2, span_v2 = face_frame(face, palm)
    d = landing - centre
    lo_u2, hi_u2, lo_v2, hi_v2 = mount_uv_bounds(face, palm)
    u2 = min(max(0.5 + float(np.dot(d, t_u2)) / span_u2, lo_u2), hi_u2)
    v2 = min(max(0.5 + float(np.dot(d, t_v2)) / span_v2, lo_v2), hi_v2)
    return G.Mount(face, u2, v2)


def perturb_offset(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Step EVERY joint's zero offset by one quantum, independently up or down.

    A joint's offset is where its link sits when the actuator is at neutral --
    the angle it is assembled at. Structural, costing no motor, and it carries
    the joint's travel with it.

    This replaces a mount-orientation operator that could only aim a whole finger
    from its base. An offset on the base joint reproduces exactly what that did
    (verified to 2e-12 over every reachable rest direction), and an offset
    further out gives a finger a resting curl, which no mount orientation could
    express. One primitive covering both, applied at every joint rather than only
    the first.

    Whole-hand, matching ``perturb_axis``: offset and theta are the same kind of
    per-joint angle on the same grid, so they explore at the same rate.
    """
    lo, hi = G.JOINT_LIMIT
    for _ in range(_REDRAWS):
        fingers = []
        for f in hand.fingers:
            segments = tuple(
                G.Segment(
                    G.Joint(sg.joint.theta, sg.joint.phi,
                            snap(reflect(sg.joint.offset
                                         + G.ANGLE_QUANTUM * rng.choice((-1, 1)),
                                         lo, hi), G.ANGLE_QUANTUM)),
                    sg.length)
                for sg in f.segments)
            fingers.append(replace(f, segments=segments))
        out = replace(hand, fingers=tuple(fingers))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no whole-hand offset perturbation validated")


def perturb_palm(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Step one palm dimension. The palm is the SEPARATION LEVER: the previous
    operator set measured it moving separation 3.2 mm against an in-face mount
    step's 1.2 mm, because fingers usually sit on different faces and an in-face
    step is largely perpendicular to the inter-mount vector.

    Thickness is not mutated. Mounts are normalised precisely so a resize does
    not invalidate them -- every mount rides it.
    """
    ranges = {"width": G.PALM_WIDTH_RANGE, "length": G.PALM_LENGTH_RANGE,
              "thickness": G.PALM_THICKNESS_RANGE}
    dims = list(G.MUTABLE_PALM_DIMS)
    rng.shuffle(dims)
    for name in dims:
        lo, hi = ranges[name]
        step = G.PALM_STEP * rng.choice((-1, 1))
        value = snap(reflect(getattr(hand.palm, name) + step, lo, hi), G.PALM_QUANTUM)
        out = replace(hand, palm=replace(hand.palm, **{name: value}))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no palm dimension could be stepped")


# --- dispatch and instrumentation -------------------------------------------

_FUNCS = {
    "split_link": split_link, "merge_links": merge_links,
    "add_finger": add_finger, "remove_finger": remove_finger,
    "perturb_axis": perturb_axis, "perturb_length": perturb_length,
    "perturb_offset": perturb_offset,
    "move_mount": move_mount, "perturb_palm": perturb_palm,
}


def _accept(hand: G.Hand, op: str) -> G.Hand:
    """Closure check. An operator building something illegal is a bug here."""
    reasons = V.check(hand)
    if reasons:
        raise MutationImpossible(f"{op} produced an invalid hand: {reasons[0]}")
    return hand


@dataclass
class Stats:
    """Per-operator attempt and success counts.

    A raw rate gap is NOT evidence of a ratchet: several operators are
    structurally gated in ways that are the design working -- ``merge_links``
    cannot act on single-joint fingers, ``remove_finger`` cannot act at
    MIN_FINGERS, ``add_finger`` cannot act on a full palm. Near a boundary the gap looks alarming
    and is arithmetic.

    The honest instrument is PER-MOVE BALANCE: from a hand at a given joint
    count, does one structural operator raise complexity as often as it lowers
    it? See ``tests/test_grammar.py::test_operators_are_unbiased``. These counts
    are still worth logging, because what they catch is a CHANGE.
    """

    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)

    def record(self, op: str, ok: bool) -> None:
        self.attempts[op] = self.attempts.get(op, 0) + 1
        self.successes[op] = self.successes.get(op, 0) + int(ok)

    def rate(self, op: str) -> float:
        n = self.attempts.get(op, 0)
        return self.successes.get(op, 0) / n if n else float("nan")

    def ratchet(self) -> dict[str, float]:
        """Success-rate gap per add/remove pair. Diagnostic only -- what matters
        is whether it MOVES between runs, not its value near a boundary."""
        return {f"{a} - {b}": self.rate(a) - self.rate(b) for a, b in _GROWS}

    def report(self) -> str:
        rows = [f"  {op:<20s} {self.successes.get(op,0):>6d}/"
                f"{self.attempts.get(op,0):<6d} {self.rate(op):6.1%}"
                for op in OPERATORS if self.attempts.get(op)]
        gaps = "".join(f"\n  {k:<28s} {v:+.1%}" for k, v in self.ratchet().items())
        return ("operator            success/attempts   rate\n"
                + "\n".join(rows) + "\n\nratchet (add - remove):" + gaps)


def apply(rng: random.Random, hand: G.Hand, op: str) -> G.Hand:
    """One operator, once. Raises ``MutationImpossible`` if it cannot act."""
    if op not in _FUNCS:
        raise KeyError(f"{op!r} is not an operator; use {OPERATORS}")
    return _FUNCS[op](rng, hand)


def mutate(rng: random.Random, hand: G.Hand, op: str | None = None,
           stats: Stats | None = None) -> G.Hand | None:
    """One child, or ``None`` if the operator could not act.

    Returning None rather than retrying a different operator is deliberate: a
    silent retry would reweight toward whichever operators are easy to apply,
    which is the bias ``Stats`` exists to measure.
    """
    op = op or OPERATORS[rng.randrange(len(OPERATORS))]
    try:
        child = apply(rng, hand, op)
    except MutationImpossible:
        if stats is not None:
            stats.record(op, False)
        return None
    if stats is not None:
        stats.record(op, True)
    return child
