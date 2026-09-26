"""Named ``Distribution`` variants and operator mixtures for the E0
evolvability experiments. Every variant below is built from EXISTING
``Distribution`` fields (no new field was needed for these six), via
``dataclasses.replace`` on ``DEFAULT_DISTRIBUTION`` -- so each differs from
``G_FULL`` in exactly the field(s) named in its own docstring, everything
else identical to current default sampling behaviour.
"""

from __future__ import annotations

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

# Every named Distribution variant above, for iteration by experiment code.
NAMED_DISTRIBUTIONS = {
    "G_FULL": G_FULL,
    "G_NOPALMJOINT": G_NOPALMJOINT,
    "G_NOBRANCH": G_NOBRANCH,
    "G_NOCOUPLE": G_NOCOUPLE,
    "G_SERIAL": G_SERIAL,
    "G_FULL_INS": G_FULL_INS,
    "G_NOBRANCH_INS": G_NOBRANCH_INS,
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
    "NAMED_DISTRIBUTIONS",
    "OPERATORS",
    "SMALL_STEP_OPERATORS",
]
