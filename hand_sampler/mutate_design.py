"""Mutation operators: the neighbourhood structure local search walks."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

import numpy as np

from hand_sampler import design_space
from hand_sampler import validate_design
from hand_sampler.design_space import (
    mount_position,
)

OPERATORS: tuple[str, ...] = (
    # structural -- these move complexity, +-1 joint each
    "split_link", "merge_links", "add_finger", "remove_finger",
    # parametric -- complexity fixed
    "perturb_kind", "perturb_length", "move_mount", "aim_mount",
    "perturb_lean",
)

STRUCTURAL: tuple[str, ...] = OPERATORS[:4]

_STEPS: tuple[int, ...] = (-1, 0, 1)
"""Per-element step for the three whole-hand perturbations.

The ``0`` lets an element hold still, which is what makes these moves local again:
without it a 20-joint hand could never change one axis and leave the rest alone.
It also means every element can draw ``0`` and return the hand untouched -- a child
identical to its parent that is NOT a MutationImpossible, so it would inflate the
null rate invisibly. Each operator rejects that draw and redraws.
"""

_REDRAWS = 8
"""How many independent draws a whole-hand operator tries before giving up."""

MOUNT_STEP_M = 0.005
"""One step across a palm face, in metres. See the module docstring."""

_GROWS: tuple[tuple[str, str], ...] = (("split_link", "merge_links"),
                                       ("add_finger", "remove_finger"))
"""Structural pairs, growing operator first."""



class MutationImpossible(ValueError):
    """This operator cannot act on this hand."""


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


# --- structural -------------------------------------------------------------

def split_link(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Divide one link in two, inserting a joint."""
    moves = []
    for fi, finger in enumerate(hand.fingers):
        if finger.n_joints >= design_space.MAX_JOINTS_PER_FINGER:
            continue
        last = finger.n_joints - 1
        for si, seg in enumerate(finger.segments):
            # Splitting makes the proximal half carry a motor whatever it was
            # before; only the distal half can still be a motor-free fingertip.
            lo_a = round(design_space.MIN_LINK_LENGTH / design_space.LINK_QUANTUM)
            lo_b = round((design_space.MIN_DISTAL_LINK_LENGTH if si == last
                          else design_space.MIN_LINK_LENGTH)
                         / design_space.LINK_QUANTUM)
            n_tot = round(seg.length / design_space.LINK_QUANTUM)
            for n_a in range(lo_a, n_tot - lo_b + 1):
                moves.append((fi, si, n_a * design_space.LINK_QUANTUM))
    if not moves:
        raise MutationImpossible("no link is long enough to divide")

    rng.shuffle(moves)
    for fi, si, a in moves:
        finger = hand.fingers[fi]
        segments = list(finger.segments)
        old = segments[si]
        segments[si] = design_space.Segment(old.joint, a)
        segments.insert(si + 1, design_space.Segment(
            design_space.Joint(kind=_draw_kind(rng)), old.length - a))
        out = design_space.with_finger(hand, fi, replace(finger, segments=tuple(segments)))
        if validate_design.is_valid(out):
            return out
    raise MutationImpossible("no split produced a valid hand")


def merge_links(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Join two links, removing the joint between them."""
    moves = [(fi, si) for fi, f in enumerate(hand.fingers) if f.n_joints >= 2
             for si in range(f.n_joints)]
    if not moves:
        raise MutationImpossible("every finger has a single joint")

    rng.shuffle(moves)
    for fi, si in moves:
        out = design_space.with_finger(hand, fi, _merge_out(hand.fingers[fi], si))
        if validate_design.is_valid(out):
            return out
    raise MutationImpossible("no merge stayed within bounds")


def add_finger(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Attach a new SINGLE-JOINT finger to the palm."""
    if hand.n_fingers >= design_space.MAX_FINGERS:
        raise MutationImpossible(f"already at {design_space.MAX_FINGERS} fingers")
    out = _new_finger(rng, hand)
    if out is None:
        raise MutationImpossible("no room on the palm for another mount")
    return _accept(out, "add_finger")


def remove_finger(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Delete a SINGLE-JOINT finger."""
    if hand.n_fingers <= design_space.MIN_FINGERS:
        raise MutationImpossible(f"already at the minimum of {design_space.MIN_FINGERS}")
    candidates = [i for i, f in enumerate(hand.fingers) if f.n_joints == 1]
    if not candidates:
        raise MutationImpossible("no single-joint finger; merge one down first")

    rng.shuffle(candidates)
    for i in candidates:
        out = replace(hand, fingers=tuple(f for k, f in enumerate(hand.fingers)
                                          if k != i))
        if validate_design.is_valid(out):
            return out
    raise MutationImpossible("no finger could be removed")


def _merge_out(finger: design_space.Finger, si: int) -> design_space.Finger:
    """Drop segment ``si``, folding its link into a neighbour."""
    segments = list(finger.segments)
    freed = segments.pop(si).length

    proximal = si - 1 if si > 0 else 0
    candidates = [proximal]
    distal = si if si < len(segments) else None      # post-pop index of si + 1
    if distal is not None and distal != proximal:
        candidates.append(distal)

    for into in candidates:
        merged = segments[into].length + freed
        if merged <= design_space.MAX_LINK_LENGTH + 1e-9:
            segments[into] = design_space.Segment(segments[into].joint, merged)
            return replace(finger, segments=tuple(segments))

    segments[proximal] = design_space.Segment(segments[proximal].joint, design_space.MAX_LINK_LENGTH)
    return replace(finger, segments=tuple(segments))


def _draw_kind(rng: random.Random) -> int:
    """A hinge, not a roll: a fresh joint that spins its own link about itself
    moves nothing a capsule can see, so it would be a wasted motor."""
    return rng.choice((design_space.FLEXION, design_space.ABDUCTION))


def _new_finger(rng: random.Random, hand: design_space.Hand) -> design_space.Hand | None:
    """A fresh single-joint finger where there is room, or None if nowhere."""
    sites = _free_mount_sites(hand)
    if not sites:
        return None

    # A fresh finger has a single segment, which is therefore terminal: its own
    # joint is driven from the palm, and it has no child joint to carry a motor.
    n_len = round((design_space.MAX_LINK_LENGTH - design_space.MIN_DISTAL_LINK_LENGTH)
                  / design_space.LINK_QUANTUM)
    rng.shuffle(sites)
    for mount in sites:
        finger = design_space.Finger(
            mount=mount,
            segments=(design_space.Segment(
                design_space.Joint(kind=_draw_kind(rng)),
                length=design_space.MIN_DISTAL_LINK_LENGTH
                + rng.randint(0, n_len) * design_space.LINK_QUANTUM),),
        )
        out = replace(hand, fingers=hand.fingers + (finger,))
        if validate_design.is_valid(out):
            return out
    return None


def _free_mount_sites(hand: design_space.Hand) -> list[design_space.Mount]:
    """Every place on the palm with room for another finger.

    The ring between PALM_MIN_RADIUS and MAX_MOUNT_RADIUS, on the radius grid
    and the angle grid, minus the wrist's wedge and minus anywhere too close to
    a finger that is already there.
    """
    if not hand.fingers:
        return []
    existing = np.array([mount_position(f.mount) for f in hand.fingers])
    q = design_space.PALM_QUANTUM
    turn = 2.0 * math.pi
    n_r = int(round((design_space.MAX_MOUNT_RADIUS
                     - design_space.PALM_MIN_RADIUS) / q))
    n_a = int(round(turn / design_space.ANGLE_QUANTUM))

    out: list[design_space.Mount] = []
    for i in range(n_r + 1):
        r = design_space.PALM_MIN_RADIUS + i * q
        for k in range(n_a):
            bearing = k * design_space.ANGLE_QUANTUM
            if design_space.in_wrist_nogo(bearing):
                continue
            pos = design_space.mount_position(
                design_space.Mount(r, bearing, bearing))
            if (np.linalg.norm(existing - pos, axis=1)
                    >= design_space.MIN_MOUNT_SEPARATION).all():
                out.append(design_space.Mount(r, bearing, bearing))
    return out


KIND_WEIGHTS: dict[int, int] = {design_space.ROLL: 1,
                                design_space.FLEXION: 6,
                                design_space.ABDUCTION: 2}
"""How ``perturb_kind`` picks what a joint becomes. NOT a uniform draw.

Flexion is the only kind that closes a hand, so it is favoured; roll is the kind
that is useful least often -- LEAP spends one joint of sixteen on it -- so it is
favoured least. This makes the operator BIASED on purpose: flexion is easier to
enter than to leave, and a population left to drift with no selection settles at
roughly 20 / 45 / 35 roll / flexion / abduction instead of 33 each.

The ceiling is structural. The operator must change the kind it lands on, so a
flexion joint always leaves flexion, and no weight pushes the resting share past
about 47 percent -- well short of LEAP's 62. Getting there is selection's job,
not the operator's.

Note this bias is invisible to the balance tests, which measure JOINT COUNT, and
perturb_kind does not change it.
"""


def _weighted_order(rng: random.Random, options: list[int]) -> list[int]:
    """``options`` shuffled with KIND_WEIGHTS, best-weighted first most often."""
    out = []
    pool = list(options)
    while pool:
        w = [KIND_WEIGHTS[k] for k in pool]
        pick = rng.choices(range(len(pool)), weights=w, k=1)[0]
        out.append(pool.pop(pick))
    return out


def perturb_kind(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Change ONE joint to one of the other two kinds, favouring flexion.

    There are only three, so a step is a jump rather than a nudge and there is
    no neighbourhood to respect: any kind reaches any other in one move, and the
    move back is one move too -- though not at the same probability, see
    KIND_WEIGHTS.
    """
    places = [(fi, si) for fi, f in enumerate(hand.fingers)
              for si in range(f.n_joints)]
    rng.shuffle(places)
    for fi, si in places:
        finger = hand.fingers[fi]
        here = finger.segments[si].joint.kind
        options = _weighted_order(
            rng, [k for k in design_space.JOINT_KINDS if k != here])
        for kind in options:
            segments = tuple(replace(s, joint=replace(s.joint, kind=kind)) if k == si else s
                             for k, s in enumerate(finger.segments))
            out = design_space.with_finger(hand, fi, replace(finger, segments=segments))
            if validate_design.is_valid(out):
                return out
    raise MutationImpossible("no joint could change kind without violating a bound")


def perturb_length(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Step EVERY link by one quantum, each independently up, down or not at all.

    Each length reflects into its OWN range: the last link of a finger has the
    motor-free floor, every other link the motor-carrying one.
    """
    for _ in range(_REDRAWS):
        fingers = []
        for f in hand.fingers:
            last = f.n_joints - 1
            segments = tuple(
                design_space.Segment(sg.joint,
                          snap(reflect(sg.length
                                       + design_space.LINK_QUANTUM * rng.choice(_STEPS),
                                       design_space.MIN_DISTAL_LINK_LENGTH if k == last
                                       else design_space.MIN_LINK_LENGTH,
                                       design_space.MAX_LINK_LENGTH),
                               design_space.LINK_QUANTUM))
                for k, sg in enumerate(f.segments))
            fingers.append(replace(f, segments=segments))
        out = replace(hand, fingers=tuple(fingers))
        if out != hand and validate_design.is_valid(out):   # see _STEPS
            return out
    raise MutationImpossible("no whole-hand length perturbation validated")


def move_mount(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Step ONE finger's base: one radius quantum out or in, or one angle
    quantum around the palm.

    Where it SITS only. Which way it points is aim_mount's job -- a thumb has to
    be able to slide around the palm without swinging the finger with it.
    """
    q = design_space.PALM_QUANTUM
    a = design_space.ANGLE_QUANTUM
    steps = [(+q, 0.0), (-q, 0.0), (0.0, +a), (0.0, -a)]
    order = list(range(hand.n_fingers))
    rng.shuffle(order)
    for fi in order:
        finger = hand.fingers[fi]
        rng.shuffle(steps)
        for dr, da in steps:
            mount = replace(finger.mount,
                            radius=snap(finger.mount.radius + dr, q),
                            bearing=(finger.mount.bearing + da) % (2.0 * math.pi))
            out = design_space.with_finger(hand, fi, replace(finger, mount=mount))
            if validate_design.is_valid(out):
                return out
    raise MutationImpossible("no mount could move without violating a bound")


def aim_mount(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Turn ONE finger one angle quantum about its own base, in the palm plane.

    Separate from move_mount because sitting somewhere and pointing somewhere
    are different questions: a thumb reaches back across the palm from the side.
    """
    a = design_space.ANGLE_QUANTUM
    order = list(range(hand.n_fingers))
    rng.shuffle(order)
    for fi in order:
        finger = hand.fingers[fi]
        for da in ([+a, -a] if rng.random() < 0.5 else [-a, +a]):
            mount = replace(finger.mount,
                            facing=(finger.mount.facing + da) % (2.0 * math.pi))
            out = design_space.with_finger(hand, fi, replace(finger, mount=mount))
            if validate_design.is_valid(out):
                return out
    raise MutationImpossible("no mount could be re-aimed without violating a bound")


def perturb_lean(rng: random.Random, hand: design_space.Hand) -> design_space.Hand:
    """Tip ONE segment one step: straight on to a lean, or a lean to a
    perpendicular one.

    Single-segment rather than whole-hand, like move_mount: a lean swings
    everything distal to it, so tipping them all at once almost never validates.
    """
    places = [(fi, si) for fi, f in enumerate(hand.fingers)
              for si in range(f.n_joints)]
    rng.shuffle(places)
    for fi, si in places:
        finger = hand.fingers[fi]
        seg = finger.segments[si]
        options = list(design_space.LEAN_NEIGHBOURS[seg.lean])
        rng.shuffle(options)
        for nxt in options:
            segments = tuple(replace(s, lean=nxt) if k == si else s
                             for k, s in enumerate(finger.segments))
            out = design_space.with_finger(hand, fi, replace(finger, segments=segments))
            if validate_design.is_valid(out):
                return out
    raise MutationImpossible("no segment could be re-leaned without violating a bound")


# --- dispatch and instrumentation -------------------------------------------

_FUNCS = {
    "split_link": split_link, "merge_links": merge_links,
    "add_finger": add_finger, "remove_finger": remove_finger,
    "perturb_kind": perturb_kind, "perturb_length": perturb_length,
    "move_mount": move_mount, "aim_mount": aim_mount,
    "perturb_lean": perturb_lean,
}
assert set(_FUNCS) == set(OPERATORS), (
    f"operator table and OPERATORS disagree: {set(OPERATORS) ^ set(_FUNCS)}")


def _accept(hand: design_space.Hand, op: str) -> design_space.Hand:
    """Closure check. An operator building something illegal is a bug here."""
    reasons = validate_design.check(hand)
    if reasons:
        raise MutationImpossible(f"{op} produced an invalid hand: {reasons[0]}")
    return hand


@dataclass
class Stats:
    """Per-operator attempt and success counts."""

    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)

    def record(self, op: str, ok: bool) -> None:
        self.attempts[op] = self.attempts.get(op, 0) + 1
        self.successes[op] = self.successes.get(op, 0) + int(ok)

    def rate(self, op: str) -> float:
        n = self.attempts.get(op, 0)
        return self.successes.get(op, 0) / n if n else float("nan")

    def ratchet(self) -> dict[str, float]:
        """Success-rate gap per add/remove pair."""
        return {f"{a} - {b}": self.rate(a) - self.rate(b) for a, b in _GROWS}

    def report(self) -> str:
        rows = [f"  {op:<20s} {self.successes.get(op,0):>6d}/"
                f"{self.attempts.get(op,0):<6d} {self.rate(op):6.1%}"
                for op in OPERATORS if self.attempts.get(op)]
        gaps = "".join(f"\n  {k:<28s} {v:+.1%}" for k, v in self.ratchet().items())
        return ("operator            success/attempts   rate\n"
                + "\n".join(rows) + "\n\nratchet (add - remove):" + gaps)


def apply(rng: random.Random, hand: design_space.Hand, op: str) -> design_space.Hand:
    """One operator, once. Raises ``MutationImpossible`` if it cannot act."""
    if op not in _FUNCS:
        raise KeyError(f"{op!r} is not an operator; use {OPERATORS}")
    return _FUNCS[op](rng, hand)


def mutate(rng: random.Random, hand: design_space.Hand, op: str | None = None,
           stats: Stats | None = None) -> design_space.Hand | None:
    """One child, or ``None`` if the operator could not act."""
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
