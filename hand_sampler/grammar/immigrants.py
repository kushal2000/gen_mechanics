"""Stratified immigrant prior (grammar 0.5, iteration B; process
recommendation 2 in ``balanced-grammar-synthesis.md`` section 4: "random
immigrants: 5-10% of each generation's evaluations from the initialisation
prior, unconditionally").

A plain ``sample_derivation(rng, dist)`` draw is already the initialisation
prior, but a handful of coarse structural axes (digit count, palm-joint
count, whether any coupling is present, whether any prismatic joint is
present) are far from uniformly represented in it -- e.g. ``Coupled``
modules require an earlier revolute phalanx in the same digit, so they are
scarce whenever ``digit_count`` is low. ``sample_immigrant``/
``stratified_population`` below rejection-sample the SAME prior
(``sample_derivation``) so every stratum in a grid over those four axes gets
deliberate representation, without changing what "the prior" means (every
individual draw is still an ordinary, valid ``sample_derivation`` output --
nothing here resamples or edits a field in isolation, unlike ``vary``'s
operators).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np

from .derive import Derivation, derive, sample_derivation
from .distributions import DEFAULT_DISTRIBUTION, Distribution
from .kinematics import KinematicModel
from .phenodist import _digit_count

_DEFAULT_MAX_TRIES = 200


class Stratum(NamedTuple):
    digit_count: int
    palm_joint_bucket: int  # 0, 1, or 2 (meaning "2 or more")
    has_coupling: bool
    has_prismatic: bool


def _palm_joint_count(model: KinematicModel) -> int:
    """Number of non-fixed joints whose parent AND child are both
    palm-flagged bodies -- i.e. an additional palm body's own joint (see
    ``derive.py``'s ``_op_toggle_palm_joint``), never a digit's mount
    joint (whose child is never palm-flagged)."""
    palm_names = {b.name for b in model.bodies if b.palm}
    return sum(1 for j in model.joints if j.parent in palm_names and j.child in palm_names and j.type != "fixed")


def stratum_of(model: KinematicModel, dist: Distribution = DEFAULT_DISTRIBUTION) -> Stratum:
    """The ``Stratum`` cell a derived ``KinematicModel`` falls into, for
    ``dist``'s own digit-count bucketing (see ``_digit_bucket_ranges``)."""
    digit_bucket = _digit_bucket_index(_digit_count(model), dist)
    palm_joint_bucket = min(_palm_joint_count(model), 2)
    has_coupling = len(model.couplings) > 0
    has_prismatic = any(j.type == "prismatic" for j in model.joints)
    return Stratum(digit_bucket, palm_joint_bucket, has_coupling, has_prismatic)


_N_DIGIT_BUCKETS = 3


def _digit_bucket_ranges(dist: Distribution) -> List[range]:
    """``dist.digit_count_range`` (inclusive) split into (up to)
    ``_N_DIGIT_BUCKETS`` roughly-equal contiguous sub-ranges ("low/mid/high"
    thirds, degrading gracefully to fewer buckets when the range itself is
    narrower than that): e.g. the default ``(1, 6)`` splits into
    ``{1,2}``/``{3,4}``/``{5,6}``. A per-VALUE grid (one bucket per digit
    count) was tried first and rejected: ``has_coupling`` becomes
    combinatorially rare as digit_count grows (a ``Coupled`` module needs an
    earlier revolute phalanx in the SAME digit -- empirically, under
    ``G_FULL_INS``, P(no coupling anywhere on the hand) falls from ~51% at
    digit_count=1 to ~1.7% at digit_count=6), so a lone "digit_count==6"
    bucket combined with "no coupling" is too rare to land reliably within
    ``sample_immigrant``'s 200-try budget even across ``stratified_population``
    's whole cycle; three coarser buckets both reduce the total stratum
    count (more cycle "slots", hence more independent 200-try rounds, per
    stratum) and merge each rare high-digit-count value into a bucket with
    its neighbours, raising the per-round hit probability enough that the
    combined per-stratum budget across a 400-immigrant cycle reliably lands
    at least once."""
    lo, hi = dist.digit_count_range
    values = list(range(lo, hi + 1))
    n_buckets = min(_N_DIGIT_BUCKETS, len(values))
    if n_buckets <= 0:
        return []
    # Split ``values`` into ``n_buckets`` contiguous, roughly-equal chunks.
    base, extra = divmod(len(values), n_buckets)
    out: List[range] = []
    start = 0
    for i in range(n_buckets):
        size = base + (1 if i < extra else 0)
        chunk = values[start:start + size]
        out.append(range(chunk[0], chunk[-1] + 1))
        start += size
    return out


def _digit_bucket_index(digit_count: int, dist: Distribution) -> int:
    for i, r in enumerate(_digit_bucket_ranges(dist)):
        if digit_count in r:
            return i
    # Outside every bucket (a digit_count off ``dist``'s own range, e.g. a
    # hand hand-authored or produced by an operator that can exceed the
    # sampling range): clamp to the nearest end bucket rather than raising.
    ranges = _digit_bucket_ranges(dist)
    if not ranges:
        return 0
    return 0 if digit_count < ranges[0].start else len(ranges) - 1


def all_strata(dist: Distribution = DEFAULT_DISTRIBUTION) -> List[Stratum]:
    """Every cell of the grid: one bucket per digit-count sub-range in
    ``dist.digit_count_range`` (see ``_digit_bucket_ranges``), x palm-joint
    bucket {0, 1, 2+}, x has_coupling {False, True}, x has_prismatic
    {False, True}. ``Stratum.digit_count`` holds the bucket INDEX (0-based),
    not a raw digit count."""
    n_buckets = max(1, len(_digit_bucket_ranges(dist)))
    return [
        Stratum(d, p, c, pr)
        for d in range(n_buckets)
        for p in (0, 1, 2)
        for c in (False, True)
        for pr in (False, True)
    ]


def sample_immigrant(rng: np.random.Generator, dist: Distribution = DEFAULT_DISTRIBUTION,
                      strata: Optional[Sequence[Stratum]] = None,
                      max_tries: int = _DEFAULT_MAX_TRIES) -> Optional[Derivation]:
    """Pick one stratum from ``strata`` (default ``all_strata(dist)``)
    UNIFORMLY at random, then rejection-sample ``sample_derivation(rng,
    dist)`` (up to ``max_tries`` attempts) until the derived model lands in
    it. Returns ``None`` if no draw landed in the chosen stratum within
    ``max_tries`` attempts (an "empty" -- or merely rare -- stratum for this
    ``dist``; see ``stratum_occupancy`` for a population-level report)."""
    pool = list(strata) if strata is not None else all_strata(dist)
    if not pool:
        return None
    target = pool[int(rng.integers(0, len(pool)))]
    for _ in range(max_tries):
        derivation = sample_derivation(rng, dist)
        model = derive(derivation)
        if stratum_of(model, dist) == target:
            return derivation
    return None


_DEFAULT_ROUNDS = 6


def stratified_population(n: int, seed: int, dist: Distribution = DEFAULT_DISTRIBUTION,
                           strata: Optional[Sequence[Stratum]] = None,
                           max_tries: int = _DEFAULT_MAX_TRIES, rounds: int = _DEFAULT_ROUNDS) -> List[Derivation]:
    """``n`` immigrants, CYCLING deterministically through ``strata``
    (default ``all_strata(dist)``) in order (item ``i`` targets
    ``strata[i % len(strata)]``) so occupancy is balanced by construction
    (every stratum gets ``n // len(strata)`` or one more "slot", rather than
    ``sample_immigrant``'s own per-call uniform pick, which would only
    balance occupancy in expectation).

    A slot whose FIRST ``sample_immigrant`` call (``max_tries`` attempts)
    misses is retried against the SAME target, up to ``rounds`` times total,
    before giving up -- concentrating any extra search budget on the
    stratum that actually needs it, rather than falling back to an
    unconstrained draw immediately (which would land in whatever stratum is
    already common, silently inflating its count and widening the
    occupancy ratio -- the first design tried here, and rejected for
    exactly that reason). Only after every round fails does a slot fall
    back to an unconstrained ``sample_derivation`` draw (still a valid
    prior sample), so the returned list always has exactly ``n`` elements;
    use ``stratum_occupancy`` on the result to see which strata (if any)
    still ended up empty."""
    pool = list(strata) if strata is not None else all_strata(dist)
    rng = np.random.default_rng(seed)
    out: List[Optional[Derivation]] = [None] * n
    counts: Dict[Stratum, int] = {s: 0 for s in pool}
    pending = list(range(n))
    for _round in range(max(1, rounds)):
        if not pending:
            break
        still_pending: List[int] = []
        for i in pending:
            target = pool[i % len(pool)] if pool else None
            derivation = sample_immigrant(rng, dist, strata=[target] if target is not None else None,
                                           max_tries=max_tries)
            if derivation is None:
                still_pending.append(i)
            else:
                out[i] = derivation
                counts[target] += 1
        pending = still_pending

    # Rebalancing pass (instead of an immediately-unconstrained fallback,
    # which would land in whatever stratum is already common and widen the
    # occupancy ratio -- see this function's own docstring): any slot still
    # unfilled after every round above retargets the CURRENTLY
    # least-occupied stratum (recomputed each attempt), for a few more
    # rounds, before finally falling back to an unconstrained draw.
    for _round in range(max(1, rounds)):
        if not pending or not pool:
            break
        still_pending: List[int] = []
        for i in pending:
            target = min(pool, key=lambda s: counts[s])
            derivation = sample_immigrant(rng, dist, strata=[target], max_tries=max_tries)
            if derivation is None:
                still_pending.append(i)
            else:
                out[i] = derivation
                counts[target] += 1
        pending = still_pending

    for i in pending:
        out[i] = sample_derivation(rng, dist)
    return out  # type: ignore[return-value]


def stratum_occupancy(population: Sequence[Derivation], dist: Distribution = DEFAULT_DISTRIBUTION,
                       strata: Optional[Sequence[Stratum]] = None) -> Dict[str, int]:
    """``{str(stratum): count}`` over every stratum in ``strata`` (default
    ``all_strata(dist)``), classifying each member of ``population`` by its
    OWN derived model (not by which stratum it was originally targeted at --
    this is a report of what actually came out). Strata with count 0 are
    included (report of "empty strata")."""
    pool = list(strata) if strata is not None else all_strata(dist)
    counts: Dict[str, int] = {str(s): 0 for s in pool}
    for derivation in population:
        model = derive(derivation)
        s = stratum_of(model, dist)
        key = str(s)
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = counts.get(key, 0) + 1
    return counts
