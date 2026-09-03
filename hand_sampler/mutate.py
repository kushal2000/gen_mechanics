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

SEVEN OPERATORS, AND NO MORE. One reachable by chaining others earns nothing and
adds a second place the same rule can drift. ``add_finger``/``remove_finger``
folded into ``add_node``/``remove_node`` (in a tree, attaching to a joint and to
the palm are the same operation), and ``remount`` folded into ``move_mount`` once
a step overflowing a face carries onto the next. ``perturb_axis`` and
``perturb_direction`` survive that test: a joint axis decides which way a joint
SWEEPS, a mount direction which way the finger POINTS AT REST, and no sequence of
axis changes tilts a rest pose.

Reflection rather than clipping when a step leaves a range, mount steps in METRES
rather than u/v fractions, and direction jittered in the tangent plane rather
than by stepping alpha and beta -- all three because the alternative biases the
step distribution by position. Roll does not exist here; it is gauge.
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
    "add_node", "remove_node",
    # parametric -- complexity fixed
    "perturb_axis", "perturb_length", "move_mount",
    "perturb_direction", "perturb_palm",
)

STRUCTURAL: tuple[str, ...] = OPERATORS[:2]

MOUNT_STEP_M = 0.005
"""One step across a palm face, in metres. See the module docstring."""

_INVERSE = {"add_node": "remove_node", "remove_node": "add_node"}


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


def _canonical_mount(face: str, u: float, v: float,
                     alpha: float, beta: float) -> G.Mount:
    """One spelling per physical mount.

    alpha folds into [0, pi/2] rather than wrapping: a tilt of -a about azimuth b
    is a tilt of +a about b + pi. At alpha = 0 the azimuth names nothing, so beta
    is zeroed.
    """
    alpha = snap(reflect(alpha, 0.0, math.pi / 2), G.ANGLE_QUANTUM)
    beta = snap(beta % (2.0 * math.pi), G.ANGLE_QUANTUM)
    if alpha < 1e-9:
        beta = 0.0
    return G.Mount(face=face, u=u, v=v, alpha=alpha, beta=beta)


# --- structural -------------------------------------------------------------

def add_node(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Attach one joint to the tree. +1 joint, whatever the parent.

    Two kinds of attach point: *split* an existing link (needing one of at least
    2 x MIN_LINK_LENGTH), or start a NEW finger from the palm -- the palm is
    simply another possible parent. Separating the second into its own operator
    built a wall, and that wall is what froze finger count in the previous design
    space permanently, since no operator touched it.

    Candidates are pooled and drawn uniformly rather than choosing a kind first,
    which would make a new finger as likely as a new knuckle regardless of how
    many knuckle sites exist.
    """
    moves: list[tuple] = []
    for fi, finger in enumerate(hand.fingers):
        if finger.n_joints >= G.MAX_JOINTS_PER_FINGER:
            continue
        for si, seg in enumerate(finger.segments):
            n_lo = round(G.MIN_LINK_LENGTH / G.LINK_QUANTUM)
            n_tot = round(seg.length / G.LINK_QUANTUM)
            for n_a in range(n_lo, n_tot - n_lo + 1):
                moves.append(("split", fi, si, n_a * G.LINK_QUANTUM))
    if hand.n_fingers < G.MAX_FINGERS:
        moves.append(("palm", -1, -1, 0.0))

    if not moves:
        raise MutationImpossible("nowhere left to attach a joint")

    rng.shuffle(moves)
    for kind, fi, si, a in moves:
        if kind == "palm":
            out = _new_finger(rng, hand)
            if out is not None:
                return out
            continue

        finger = hand.fingers[fi]
        segments = list(finger.segments)
        old = segments[si]
        segments[si] = G.Segment(old.joint, a)
        segments.insert(si + 1, G.Segment(
            G.Joint(theta=_draw_theta(rng), phi=math.pi / 2), old.length - a))
        out = G.with_finger(hand, fi, replace(finger, segments=tuple(segments)))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no attach point produced a valid hand")


def remove_node(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Detach one joint. -1 joint, and the exact inverse of ``add_node``.

    Removing a joint with siblings folds its link into a neighbour, so reach is
    preserved and split-then-remove returns the original link. Removing a
    finger's LAST joint removes the finger with it -- that case falls out rather
    than needing an operator of its own.
    """
    moves = [(fi, si) for fi, f in enumerate(hand.fingers)
             for si in range(f.n_joints)]
    if not moves:
        raise MutationImpossible("no joints")

    rng.shuffle(moves)
    for fi, si in moves:
        finger = hand.fingers[fi]

        if finger.n_joints == 1:
            if hand.n_fingers <= G.MIN_FINGERS:
                continue
            out = replace(hand, fingers=tuple(f for k, f in enumerate(hand.fingers)
                                              if k != fi))
        else:
            out = G.with_finger(hand, fi, _merge_out(finger, si))

        if V.is_valid(out):
            return out
    raise MutationImpossible("no joint could be removed without breaking a bound")


def _merge_out(finger: G.Finger, si: int) -> G.Finger:
    """Drop segment ``si``, folding its link into a neighbour.

    Proximal first (the exact inverse of a split), else distal, else the proximal
    merge CLAMPED to the ceiling. The clamp is unreachable from ``add_node`` -- a
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
    draw violates separation, and a retry budget makes ``add_node`` quietly stop
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
            mount=_canonical_mount(face, u, v, alpha=0.0, beta=0.0),
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
    goes through the full validator. Vectorised per face because ``add_node``
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
    """Step one joint's theta by one angle quantum.

    Only theta. ``phi`` is PINNED at pi/2 -- every hinge perpendicular to its
    link -- and is the first entry on the held-back list in DESIGN.md 11. An
    off-perpendicular hinge is a real mechanism (the link sweeps a cone of
    half-angle phi rather than a flat fan) but it reads as a joint with two axes
    unless you already know what you are looking at, so it is held back until the
    space needs the complexity.
    """
    fi = rng.randrange(hand.n_fingers)
    finger = hand.fingers[fi]
    si = rng.randrange(finger.n_joints)
    seg = finger.segments[si]

    step = G.ANGLE_QUANTUM * rng.choice((-1, 1))
    theta = snap(wrap_theta(seg.joint.theta + step), G.ANGLE_QUANTUM) % math.pi
    joint = G.Joint(theta, seg.joint.phi)

    finger = G.with_segment(finger, si, G.Segment(joint, seg.length))
    return _accept(G.with_finger(hand, fi, finger), "perturb_axis")


def perturb_length(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Step one link by one quantum. The only operator that changes total reach --
    ``add_node`` splits and ``remove_node`` merges, both reach-preserving."""
    moves = [(fi, si)
             for fi, f in enumerate(hand.fingers)
             for si, s in enumerate(f.segments) if s.length > 1e-9]
    if not moves:
        raise MutationImpossible("every segment is zero-length")

    rng.shuffle(moves)
    for fi, si in moves:
        finger = hand.fingers[fi]
        seg = finger.segments[si]
        step = G.LINK_QUANTUM * rng.choice((-1, 1))
        length = snap(reflect(seg.length + step, G.MIN_LINK_LENGTH,
                              G.MAX_LINK_LENGTH), G.LINK_QUANTUM)
        out = G.with_finger(hand, fi,
                            G.with_segment(finger, si, G.Segment(seg.joint, length)))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no link could be stepped without breaking a bound")


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

    Bounded by MOUNT_EDGE_MARGIN, and a crossing JUMPS that band rather than
    walking through it -- the margin forbids a mount being within a capsule
    radius of an edge, so there is no legal position AT one to pass through.

    ``(alpha, beta)`` are PRESERVED across an edge, so the finger keeps its
    relationship to its face and its world direction rotates with the normal.
    Preserving the world direction instead turns alpha = 0 into alpha = 90,
    laying the finger flat along the surface it is bolted to.
    """
    _, n, t_u, t_v, span_u, span_v = face_frame(mount.face, palm)
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
        # Nowhere to cross to. CLAMP rather than refuse: the thin axis has a
        # 5 mm band against a 5 mm step, so refusing froze that axis entirely.
        clamped = replace(mount, u=u, v=v)
        return None if clamped == mount else clamped

    landing = mount_position(replace(mount, u=u, v=v), palm)

    centre, _, t_u2, t_v2, span_u2, span_v2 = face_frame(face, palm)
    d = landing - centre
    lo_u2, hi_u2, lo_v2, hi_v2 = mount_uv_bounds(face, palm)
    u2 = min(max(0.5 + float(np.dot(d, t_u2)) / span_u2, lo_u2), hi_u2)
    v2 = min(max(0.5 + float(np.dot(d, t_v2)) / span_v2, lo_v2), hi_v2)

    # (alpha, beta) carry over UNCHANGED, so the finger keeps its relationship to
    # the face it is on and its world direction rotates with the face normal.
    return _canonical_mount(face, u2, v2, mount.alpha, mount.beta)


def perturb_direction(rng: random.Random, hand: G.Hand) -> G.Hand:
    """Tilt one finger, by jittering its direction IN THE TANGENT PLANE.

    Not by stepping alpha and beta: they are polar about the face normal, so beta
    names nothing at alpha = 0 and a fixed beta step is a vanishing angular move
    near it. Perturbing the vector gives a step of uniform angular size wherever
    the finger already points.
    """
    order = list(range(hand.n_fingers))
    rng.shuffle(order)
    for fi in order:
        finger = hand.fingers[fi]
        _, n, t_u, t_v, _, _ = face_frame(finger.mount.face, hand.palm)
        d = mount_direction(finger.mount, hand.palm)

        psi = rng.uniform(0.0, 2.0 * math.pi)
        d_new = d + math.tan(G.ANGLE_QUANTUM) * (math.cos(psi) * t_u
                                                 + math.sin(psi) * t_v)
        d_new /= np.linalg.norm(d_new)

        # back to polar about the face normal
        alpha = math.acos(float(np.clip(np.dot(d_new, n), -1.0, 1.0)))
        tangential = d_new - float(np.dot(d_new, n)) * n
        beta = (math.atan2(float(np.dot(tangential, t_v)),
                           float(np.dot(tangential, t_u)))
                if np.linalg.norm(tangential) > 1e-9 else 0.0)

        mount = _canonical_mount(finger.mount.face, finger.mount.u,
                                 finger.mount.v, alpha, beta)
        if mount == finger.mount:
            continue
        out = G.with_finger(hand, fi, replace(finger, mount=mount))
        if V.is_valid(out):
            return out
    raise MutationImpossible("no finger direction could be perturbed")


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
    "add_node": add_node, "remove_node": remove_node,
    "perturb_axis": perturb_axis, "perturb_length": perturb_length,
    "move_mount": move_mount, "perturb_direction": perturb_direction,
    "perturb_palm": perturb_palm,
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
    structurally gated in ways that are the design working -- ``remove_node``
    cannot act on a hand of single-joint fingers at MIN_FINGERS, ``add_node``
    cannot split links that are too short. Near a boundary the gap looks alarming
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
        return {f"{a} - {b}": self.rate(a) - self.rate(b)
                for a, b in _INVERSE.items() if a.startswith("add")}

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
