"""Generation 0: the seed population.

Deliberately simple and deliberately CONVENTIONAL -- two fingers on adjacent
faces, one or two joints each, hinges perpendicular to their links, axes at pure
flexion or pure abduction, every joint at zero offset. Starting at the
conventional corner and letting the
search leave it is what makes a result legible; seeding at the unconventional
corner would conflate "evolution found this" with "we put it there".

Rejects and redraws rather than clamping an invalid draw, for the same reason
``mutate.reflect`` folds rather than clips: clamping piles probability mass on
whichever boundary was violated.
"""

from __future__ import annotations

import math
import random

from hand_sampler import genotype as G
from hand_sampler import validate as V
from hand_sampler.kinematics import mount_uv_bounds

SEED_FACE_PAIRS: tuple[tuple[str, str], ...] = (
    ("+y", "+z"),
    ("-y", "+z"),
)
"""ADJACENT faces only; the opposite pair is excluded on measurement.

Two fingers on opposite faces cannot pinch. A flexion joint's axis is
perpendicular to both the finger and GRASP_DIR, so the finger sweeps in a plane
parallel to x -- and two such planes stay a palm-width apart however far the
fingers flex, converging to exactly the palm width at 90 degrees. Only an
abduction joint moves a fingertip laterally.

Closest approach with 2-joint fingers on a 60 mm palm: +y/+z 6.0 mm, -y/+z
13.4 mm, +y/-y 44.1 mm against a 40 mm object. An earlier seed set used the
opposite pair and 58% of the population could not touch the object at any joint
angles. The two pairs here are mirror images, which is narrower than it looks:
``mutate.move_mount`` walks a finger across face edges, so nothing is
reachable-only-if-seeded."""

SEED_JOINTS = (1, 2)
"""One or two joints per finger, so a hand starts with 2 to 4 motors.

The floor matters: with MIN_FINGERS = 2, forcing two joints per finger would put
the whole population at four motors, and performance AGAINST MOTOR COUNT is the
result being chased. A two-motor hand belongs in that plot.

Not every one-joint seed can reach the object -- a one-joint fingertip traces an
ARC and two arcs often miss, where two joints trace an annulus that reliably
overlaps. Measured closure: 74% at 2 motors, 84% at 3, 100% at 4. That is a
gradient, not the sparse-fitness trap: the trap is every seed scoring zero, and
here ``split_link`` is the one-step path from the failures to the successes."""

SEED_THETAS = (0.0, math.pi / 2)   # pure flexion, pure abduction
SEED_LENGTHS = (0.030, 0.035, 0.040, 0.045, 0.050)
SEED_PALM = (
    (0.020, 0.050, 0.050),
    (0.025, 0.060, 0.060),
    (0.025, 0.070, 0.060),
)


def seed_finger(rng: random.Random, face: str, palm: G.Palm) -> G.Finger:
    n = rng.choice(SEED_JOINTS)
    segments = tuple(
        G.Segment(G.Joint(theta=rng.choice(SEED_THETAS), phi=math.pi / 2),
                  length=rng.choice(SEED_LENGTHS))
        for _ in range(n)
    )
    return G.Finger(mount=G.Mount(face, *_seed_uv(rng, face, palm)),
                    segments=segments)


def _seed_uv(rng: random.Random, face: str, palm: G.Palm) -> tuple[float, float]:
    """Where on a face a seed finger mounts -- NOT the same rule on every face.

    Normalised (u, v) mean different physical directions per face: on `+-y` the v
    axis runs along the palm's LENGTH, so biasing it high puts a finger toward
    the fingertip end, but on `+z` it runs across the WIDTH, where the same bias
    means "shoved to one side". One rule everywhere broke the seed set's mirror
    symmetry, and `-y`/`+z` seeds could not reach the object in 5 of 6 cases.

    Clamped into ``mount_uv_bounds`` so a seed never starts on a face edge.
    """
    lo_u, hi_u, lo_v, hi_v = mount_uv_bounds(face, palm)
    u = 0.5 * (lo_u + hi_u)
    v = 0.5 * (lo_v + hi_v) if face == "+z" else rng.uniform(0.55, 0.85)
    return u, min(max(v, lo_v), hi_v)


def seed_hand(rng: random.Random) -> G.Hand:
    """One seed. Retries rather than repairs -- see `seed_population`."""
    faces = SEED_FACE_PAIRS[rng.randrange(len(SEED_FACE_PAIRS))]
    palm = G.Palm(*SEED_PALM[rng.randrange(len(SEED_PALM))])
    return G.Hand(palm=palm,
                  fingers=tuple(seed_finger(rng, f, palm) for f in faces))


def seed_population(seed: int, count: int, max_tries: int = 50) -> list[G.Hand]:
    """``count`` valid seeds, by rejection rather than repair."""
    rng = random.Random(seed)
    out: list[G.Hand] = []
    while len(out) < count:
        for _ in range(max_tries):
            hand = seed_hand(rng)
            if V.is_valid(hand):
                out.append(hand)
                break
        else:
            raise RuntimeError(
                f"could not draw a valid seed in {max_tries} tries; the seed "
                f"constants and validate.py have drifted apart")
    return out
