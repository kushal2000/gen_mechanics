"""Shared "reduced start" builder for E2/E3 (and E5).

Moved out of ``e2_drift.py`` (I14 fix 11): ``e3_reach.py`` needs the exact
same ``build_start`` that ``e2_drift.py`` defines, and importing it directly
from ``e2_drift`` created a module-import cycle through ``runner.py`` (which
imports both ``e2_drift`` and ``e3_reach`` at its own module level to
register them): ``runner -> e2_drift -> runner`` (for ``register``/
``run_experiment``) while ``runner -> e3_reach -> e2_drift`` for
``build_start``. Whichever of ``runner``/``e2_drift``/``e3_reach`` is
imported FIRST determines whether Python's partial-module caching happens
to paper over the cycle -- fragile, and exactly the kind of import order
sensitivity that should not exist. This module has no dependency on
``runner``, ``e2_drift``, or ``e3_reach``, so nothing importing it can ever
be part of a cycle; ``e2_drift.py`` re-exports ``build_start`` from here
(``from .starts import build_start``) so every existing caller (including
``e2_drift.build_start`` and the unfinished ``e5_evolve.py``, which imports
it as ``from .e2_drift import build_start``) keeps working unchanged.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..derive import Derivation, VariationImpossible, generate, vary
from ..variants import G_SERIAL

_MAX_REDUCE_ATTEMPTS = 500


def _hand_step(derivation: Derivation):
    return next(s for s in derivation.steps if s.path == "hand")


def _top_level_digit_steps(derivation: Derivation) -> List:
    return [s for s in derivation.steps if s.production == "Digit" and s.params.get("top_level")]


def _digit_count_and_max_phalanx(derivation: Derivation) -> Tuple[int, int]:
    digit_count = _hand_step(derivation).params["digit_count"]
    phalanx_counts = [s.params["phalanx_count"] for s in _top_level_digit_steps(derivation)]
    return digit_count, (max(phalanx_counts) if phalanx_counts else 0)


def build_start(seed: int, dist=G_SERIAL, max_digits: int = 2, max_phalanges: int = 2):
    """``generate(seed, dist)`` reduced (via ``remove_digit``/``delete_phalanx``)
    until ``digit_count <= max_digits`` and every digit's ``phalanx_count <=
    max_phalanges``, or ``_MAX_REDUCE_ATTEMPTS`` retries are exhausted.
    Returns ``(derivation, achieved_digit_count, achieved_max_phalanx,
    reduced_fully: bool)``."""
    rng = np.random.default_rng([int(seed), 1_000_000])  # distinguishes this phase from per-mixture rngs (mix_idx 0..2)
    derivation, _ = generate(seed, dist)
    attempts = 0
    while attempts < _MAX_REDUCE_ATTEMPTS:
        digit_count, max_phalanx = _digit_count_and_max_phalanx(derivation)
        if digit_count <= max_digits and max_phalanx <= max_phalanges:
            return derivation, digit_count, max_phalanx, True
        op = "remove_digit" if digit_count > max_digits else "delete_phalanx"
        attempts += 1
        try:
            derivation = vary(derivation, rng, dist, operator=op)
        except VariationImpossible:
            continue
    digit_count, max_phalanx = _digit_count_and_max_phalanx(derivation)
    return derivation, digit_count, max_phalanx, False
