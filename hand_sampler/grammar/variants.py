"""Named ``Distribution`` variants and operator mixtures for the E0
evolvability experiments. Every variant below is built from EXISTING
``Distribution`` fields (no new field was needed for these six), via
``dataclasses.replace`` on ``DEFAULT_DISTRIBUTION`` -- so each differs from
``G_FULL`` in exactly the field(s) named in its own docstring, everything
else identical to current default sampling behaviour.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Tuple

from .derive import OPERATORS, SMALL_STEP_OPERATORS
from .distributions import DEFAULT_DISTRIBUTION, Distribution

# G_FULL is intentionally the *same object* as DEFAULT_DISTRIBUTION (not a
# copy): every other module's DEFAULT_DISTRIBUTION-keyed behaviour (replay
# hashes, existing tests) is automatically also "G_FULL" behaviour.
G_FULL: Distribution = DEFAULT_DISTRIBUTION

# No palm joints: every additional palm body is a fixed joint.
G_NOPALMJOINT: Distribution = replace(DEFAULT_DISTRIBUTION, palm_joint_probability=0.0)

# No in-digit branching.
G_NOBRANCH: Distribution = replace(DEFAULT_DISTRIBUTION, branch_probability=0.0)

# No coupled modules: renormalize module_probabilities with "Coupled"'s
# weight zeroed (the other weights are unchanged, so their *relative*
# proportions among R/C/P are identical to G_FULL's).
G_NOCOUPLE: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    module_probabilities=tuple(
        (kind, 0.0 if kind == "Coupled" else weight) for kind, weight in DEFAULT_DISTRIBUTION.module_probabilities
    ),
)

# Fully serial hands: no additional palm bodies (root is still always the
# one palm body -- see rules.py), no branching, no palm joints (moot once
# palm_body_count_range is (0, 0), but set for documentation/robustness).
G_SERIAL: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    palm_body_count_range=(0, 0),
    branch_probability=0.0,
    palm_joint_probability=0.0,
)

# I14 fix 5: an "insertion" sub-Distribution (see distributions.Distribution
# .insertion / derive.py's growth operators) that growth operators draw NEW
# digit/phalanx material from: phalanges 1..3 (no growth op should ever
# insert a full-size, up-to-6-phalanx digit/subtree in one step), no
# branching. Everything else (palm-body-related fields included -- "palm
# bodies as before") is identical to DEFAULT_DISTRIBUTION.
_INSERTION_DIST: Distribution = replace(
    DEFAULT_DISTRIBUTION, phalanx_count_range=(1, 3), branch_probability=0.0,
)

# G_FULL, but every growth operator inserts new material from
# ``_INSERTION_DIST`` instead of the (otherwise identical) outer
# distribution.
G_FULL_INS: Distribution = replace(DEFAULT_DISTRIBUTION, insertion=_INSERTION_DIST)

# G_NOBRANCH, with the same insertion sub-distribution wired in.
G_NOBRANCH_INS: Distribution = replace(G_NOBRANCH, insertion=_INSERTION_DIST)

# G_FULL_SMALL: the SAME sampling distribution as G_FULL, paired with the
# small-step operator mixture (see derive.py's SMALL_STEP_OPERATORS) for use
# with vary(..., operators=...) rather than the coarse-grained OPERATORS
# default. A distinct object from a bare Distribution: (dist, operators).
G_FULL_SMALL: Tuple[Distribution, Tuple[str, ...]] = (G_FULL, SMALL_STEP_OPERATORS)

# Grammar 0.5 (I16 support-widening, priority 1): the rest-bend primitive,
# a 15-degree/5-mm grid (each rpy component in {-30,-15,0,15,30} deg, each
# offset component in {-5,0,5} mm, full Cartesian product -- see
# distributions.Distribution.bend_probability's docstring), probability 0.3.
_BEND_DEG_GRID: Tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
_BEND_OFFSET_MM_GRID: Tuple[float, ...] = (-5.0, 0.0, 5.0)

_BEND_RPY_CHOICES_RAD: Tuple[Tuple[float, float, float], ...] = tuple(
    tuple(v * math.pi / 180.0 for v in combo) for combo in itertools.product(_BEND_DEG_GRID, repeat=3)
)
_BEND_OFFSET_CHOICES_M: Tuple[Tuple[float, float], ...] = tuple(
    tuple(v / 1000.0 for v in combo) for combo in itertools.product(_BEND_OFFSET_MM_GRID, repeat=2)
)

G_BEND: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    bend_rpy_choices_rad=_BEND_RPY_CHOICES_RAD,
    bend_offset_choices_m=_BEND_OFFSET_CHOICES_M,
    bend_probability=0.3,
)

# Grammar 0.5 (I16 support-widening, priority 2): continuous, sign-normalised
# revolute limits, range widened to (-180, 180) deg. NOTE (E11): generated
# designs sampled under G_CONT may exceed the grammar's usual (-45, 110) deg
# support range -- that is this variant's entire point (support widening),
# not a defect.
G_CONT: Distribution = replace(
    DEFAULT_DISTRIBUTION,
    limits_continuous=True,
    revolute_limit_range_deg=(-180.0, 180.0),
)

# Every named Distribution variant above, for iteration by experiment code.
NAMED_DISTRIBUTIONS = {
    "G_FULL": G_FULL,
    "G_NOPALMJOINT": G_NOPALMJOINT,
    "G_NOBRANCH": G_NOBRANCH,
    "G_NOCOUPLE": G_NOCOUPLE,
    "G_SERIAL": G_SERIAL,
    "G_FULL_INS": G_FULL_INS,
    "G_NOBRANCH_INS": G_NOBRANCH_INS,
    "G_BEND": G_BEND,
    "G_CONT": G_CONT,
}

__all__ = [
    "G_FULL",
    "G_NOPALMJOINT",
    "G_NOBRANCH",
    "G_NOCOUPLE",
    "G_SERIAL",
    "G_FULL_INS",
    "G_NOBRANCH_INS",
    "G_FULL_SMALL",
    "G_BEND",
    "G_CONT",
    "NAMED_DISTRIBUTIONS",
    "OPERATORS",
    "SMALL_STEP_OPERATORS",
]
