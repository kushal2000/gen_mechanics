"""Generation 0: the seed population."""

from __future__ import annotations

import math
import random

from hand_sampler import design_space
from hand_sampler import validate_design
from hand_sampler.design_space import mount_uv_bounds

SEED_FACE_PAIRS: tuple[tuple[str, str], ...] = (
    ("+y", "+z"),
    ("-y", "+z"),
)
"""ADJACENT faces only; the opposite pair is excluded on measurement."""

SEED_JOINTS = (1, 2)
"""One or two joints per finger, so a hand starts with 2 to 4 motors."""

SEED_THETAS = (0.0, math.pi / 2)   # pure flexion, pure abduction
SEED_LENGTHS = (0.030, 0.035, 0.040, 0.045, 0.050)
SEED_PALM = (
    (0.020, 0.050, 0.050),
    (0.025, 0.060, 0.060),
    (0.025, 0.070, 0.060),
)


def seed_finger(rng: random.Random, face: str, palm: design_space.Palm) -> design_space.Finger:
    n = rng.choice(SEED_JOINTS)
    segments = tuple(
        design_space.Segment(design_space.Joint(theta=rng.choice(SEED_THETAS), phi=math.pi / 2),
                  length=rng.choice(SEED_LENGTHS))
        for _ in range(n)
    )
    return design_space.Finger(mount=design_space.Mount(face, *_seed_uv(rng, face, palm)),
                    segments=segments)


def _seed_uv(rng: random.Random, face: str, palm: design_space.Palm) -> tuple[float, float]:
    """Where on a face a seed finger mounts -- NOT the same rule on every face."""
    lo_u, hi_u, lo_v, hi_v = mount_uv_bounds(face, palm)
    u = 0.5 * (lo_u + hi_u)
    v = 0.5 * (lo_v + hi_v) if face == "+z" else rng.uniform(0.55, 0.85)
    return u, min(max(v, lo_v), hi_v)


def seed_hand(rng: random.Random) -> design_space.Hand:
    """One seed. Retries rather than repairs -- see `seed_population`."""
    faces = SEED_FACE_PAIRS[rng.randrange(len(SEED_FACE_PAIRS))]
    palm = design_space.Palm(*SEED_PALM[rng.randrange(len(SEED_PALM))])
    return design_space.Hand(palm=palm,
                  fingers=tuple(seed_finger(rng, f, palm) for f in faces))


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
