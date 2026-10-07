"""Generation 0: the seed population."""

from __future__ import annotations

import math
import random

from hand_sampler import design_space
from hand_sampler import validate_design

SEED_BEARING_PAIRS: tuple[tuple[float, float], ...] = (
    (-30.0, 30.0),
    (-45.0, 45.0),
    (-60.0, 60.0),
    (0.0, 90.0),
)
"""Where generation 0 puts its two fingers, as bearings in degrees from +z.

Pairs rather than free draws so a seed starts somewhere plain and symmetric, and
all well clear of the arm behind the hand. The last pair is the lopsided
one, a finger forward and a thumb out to the side.
"""

SEED_JOINTS = (1, 2)
"""One or two joints per finger, so a hand starts with 2 to 4 motors."""

SEED_KIND_WEIGHTS: dict[int, int] = {design_space.FLEXION: 3,
                                     design_space.ABDUCTION: 1}
"""Generation 0 draws hinges only, and favours the one that closes a hand.

No roll: a roll joint spins its own link about its own axis, which a capsule
cannot show and a one-joint finger cannot use, so a seed that drew one would be
a wasted motor.

Flexion 3 to 1 over abduction because abduction has NO curl authority at all --
its axis is the grasp direction, so it spreads a hand and never closes it. Drawn
evenly, only 38 percent of generation 0 had the two closing fingers a grasp
needs; at 3 to 1 that is 67 percent. Seeding flexion alone would make it 100,
and is left undone deliberately: generation 0 should start somewhere plain, not
somewhere already solved.
"""
SEED_KINDS = tuple(SEED_KIND_WEIGHTS)
SEED_LENGTHS = (0.035, 0.040, 0.045, 0.050)
SEED_RADII = (0.025, 0.030, 0.035)
"""How far from the centre a seed finger starts. Just outside the palm's own
disc, so generation 0 is a small hand that mutation can grow outward."""


def _draw_seed_kind(rng: random.Random) -> int:
    kinds = list(SEED_KIND_WEIGHTS)
    return rng.choices(kinds, weights=[SEED_KIND_WEIGHTS[k] for k in kinds], k=1)[0]


def seed_finger(rng: random.Random, bearing_deg: float) -> design_space.Finger:
    """One seed finger, sitting and pointing the same way.

    A seed leaves along its own bearing -- straight out from the centre -- which
    is the plain thing to do. Mutation separates the two angles later.
    """
    n = rng.choice(SEED_JOINTS)
    segments = tuple(
        design_space.Segment(design_space.Joint(kind=_draw_seed_kind(rng)),
                             length=rng.choice(SEED_LENGTHS))
        for _ in range(n)
    )
    angle = math.radians(bearing_deg) % (2.0 * math.pi)
    # A bearing is still the readable way to SAY where a seed goes; a mount is
    # spelled on the 5 mm grid, so the polar pair is snapped onto it here. The
    # snap moves a seed by up to 3.5 mm, which the ring bound then has to hold,
    # hence the clamp.
    r = rng.choice(SEED_RADII)
    mount = design_space.Mount.polar(r, angle, angle)
    # Snapping can pull a seed inside the ring -- (25 mm, 45 deg) lands at
    # (20, 20), which is 28 out, but (20 mm, 45) lands at (15, 15), which is 21.
    # Push it one ring out and re-snap rather than drawing again.
    while mount.radius < design_space.PALM_MIN_RADIUS:
        r += design_space.PALM_QUANTUM
        mount = design_space.Mount.polar(r, angle, angle)
    return design_space.Finger(mount=mount, segments=segments)


def seed_hand(rng: random.Random) -> design_space.Hand:
    """One seed. Retries rather than repairs -- see `seed_population`."""
    bearings = SEED_BEARING_PAIRS[rng.randrange(len(SEED_BEARING_PAIRS))]
    return design_space.Hand(
        palm=design_space.Palm(design_space.PALM_THICKNESS),
        fingers=tuple(seed_finger(rng, b) for b in bearings))


def seed_population(seed: int, count: int, max_tries: int = 50) -> list[design_space.Hand]:
    """``count`` valid seeds, by rejection rather than repair."""
    rng = random.Random(seed)
    out: list[design_space.Hand] = []
    while len(out) < count:
        for _ in range(max_tries):
            hand = seed_hand(rng)
            if validate_design.is_valid(hand):
                out.append(hand)
                break
        else:
            raise RuntimeError(
                f"could not draw a valid seed in {max_tries} tries; the seed "
                f"constants and validate.py have drifted apart")
    return out
