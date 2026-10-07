"""Wider length ranges, the finger-length limit, and the coarse and fine stages.

1. Length support: the one grammar's bones span 0-90 mm, its palm 15-160 mm
   and its palm parts 10-80 mm; sampling still draws from the old priors
   (15-80 mm bones, 20-80 mm palm and palm parts), so the sampled hands are
   unchanged; mutation reaches both ends of every support range. Variants
   without support ranges keep their sampling ranges as support.
2. ``max_finger_length_mm``: the longest base-to-fingertip sum of bone lengths
   (branches count their host's bones up to the one they grow from); 250 mm in
   the simulator limits. Sampling and every operator keep it constructively
   (raw operator output already within it), a cap that never binds changes no
   draw, and a hand already over it never gets longer.
3. Coarse and fine pools: ``EVOLUTION_OPERATORS_COARSE`` is the current pool;
   ``EVOLUTION_OPERATORS_FINE`` changes one parameter of one part by exactly
   one fine unit (1 mm, 1% of the host, 5 deg), never the structure, stays in
   the support and the limits, and ``fine_lengthen_segment`` /
   ``fine_shorten_segment`` are an exact inverse pair.
4. The fine grid contains the coarse one: sampled and coarse-mutated hands
   are on the fine grid and in ``coverage(..., resolution="fine")``; fine
   chains stay in the fine support.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar import limits as L
from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    EVOLUTION_OPERATORS_COARSE,
    EVOLUTION_OPERATORS_FINE,
    FINE_LENGTH_STEP_PAIR,
    INVERSE_OF,
    STAGE_OPERATORS,
    TARGETABLE_OPERATORS,
    VariationImpossible,
    _OPERATOR_FNS,
    apply_operator,
    derivation_to_json,
    derive,
    fine_axis_indices,
    fine_bend_component_grid,
    sample_derivation,
    vary,
)
from hand_sampler.grammar.distributions import (
    DEG,
    link_length_support_m,
    palm_body_length_support_m,
    root_length_support_m,
)
from hand_sampler.grammar.limits import FINGER_LENGTH_CAP_MM, SIMULATOR, GenerationLimits
from hand_sampler.grammar.variants import G_FULL, G_V1, G_V3S, GRAMMAR_BASE, NAMED_DISTRIBUTIONS, build_distribution

GRAMMAR = build_distribution()
NO_SUPPORT = replace(GRAMMAR, link_length_support_m=None, root_length_support_m=None, palm_body_length_support_m=None)
SIM_NO_CAP = replace(SIMULATOR, max_finger_length_mm=None)


def _lengths_mm(d):
    return L.Structure.from_steps(d.steps).finger_lengths()


# ---------------------------------------------------------------------------
# 1. Length support
# ---------------------------------------------------------------------------


def test_support_ranges():
    assert link_length_support_m(GRAMMAR) == (0.0, 0.090)
    assert root_length_support_m(GRAMMAR) == (0.015, 0.160)
    assert palm_body_length_support_m(GRAMMAR) == (0.010, 0.080)
    # the sampling priors are unchanged
    assert GRAMMAR.link_length_range_m == (0.015, 0.080) and GRAMMAR.palm_length_range_m == (0.020, 0.080)
    for name, d in NAMED_DISTRIBUTIONS.items():
        d = d[0] if isinstance(d, tuple) else d
        assert link_length_support_m(d) == d.link_length_range_m, name
        assert root_length_support_m(d) == palm_body_length_support_m(d) == d.palm_length_range_m, name


@pytest.mark.parametrize("limits", [None, SIMULATOR], ids=["none", "SIMULATOR"])
def test_support_ranges_do_not_change_sampling(limits):
    for seed in range(300):
        a = sample_derivation(seed, GRAMMAR, limits=limits)
        b = sample_derivation(seed, NO_SUPPORT, limits=limits)
        assert derivation_to_json(a) == derivation_to_json(b)
        for s in a.steps:
            if s.production == "Phalanx":
                assert 0.015 - 1e-12 <= s.params["length"] <= 0.080 + 1e-12
            elif s.production == "PalmBody":
                assert 0.020 - 1e-12 <= s.params["length"] <= 0.080 + 1e-12
        assert 0.020 - 1e-12 <= next(s for s in a.steps if s.path == "hand").params["root_length"] <= 0.080 + 1e-12


def _walk(d, dist, op, target_of, n=40):
    """Apply ``op`` (targeted at the segment ``target_of(d)``) until it stops."""
    rng = np.random.default_rng(0)
    for _ in range(n):
        child = apply_operator(d, rng, dist, op, target=target_of(d))
        if child is None:
            return d
        d = child
    return d


def test_mutation_reaches_both_ends_of_the_support():
    d = sample_derivation(2, GRAMMAR, limits=SIMULATOR)
    ph = next(s for s in d.steps if s.production == "Phalanx")
    uid = ph.params["uid"]
    seg = lambda x: next(s for s in x.steps if s.params.get("uid") == uid).params["length"]   # noqa: E731
    for op, end in (("lengthen_segment", 0.090), ("shorten_segment", 0.0),
                    ("fine_lengthen_segment", 0.090), ("fine_shorten_segment", 0.0)):
        out = _walk(d, GRAMMAR, op, lambda _x: uid, n=100)
        assert abs(seg(out) - end) < 1e-9, (op, seg(out))
    derive(_walk(d, GRAMMAR, "shorten_segment", lambda _x: uid))          # a 0 mm bone derives
    rng = np.random.default_rng(1)
    roots = set()
    for _ in range(4000):
        child = apply_operator(d, rng, GRAMMAR, "step_root_length")
        if child is not None:
            d = child
            roots.add(round(next(s for s in d.steps if s.path == "hand").params["root_length"], 6))
    assert min(roots) == 0.015 and max(roots) == 0.160
    # palm parts: 10-80 mm
    hand = next(sample_derivation(s, GRAMMAR, limits=SIMULATOR) for s in range(200)
                if any(st.production == "PalmBody" for st in sample_derivation(s, GRAMMAR, limits=SIMULATOR).steps))
    pb = next(s for s in hand.steps if s.production == "PalmBody").params["uid"]
    pl = lambda x: next(s for s in x.steps if s.params.get("uid") == pb).params["length"]   # noqa: E731
    assert abs(pl(_walk(hand, GRAMMAR, "shorten_segment", lambda _x: pb, 40)) - 0.010) < 1e-9
    assert abs(pl(_walk(hand, GRAMMAR, "lengthen_segment", lambda _x: pb, 40)) - 0.080) < 1e-9


def test_coverage_and_conform_use_the_support():
    d = sample_derivation(2, GRAMMAR, limits=SIMULATOR)
    uid = next(s for s in d.steps if s.production == "Phalanx").params["uid"]
    zero = _walk(d, GRAMMAR, "shorten_segment", lambda _x: uid)
    assert coverage(derive(zero), GRAMMAR).in_support
    assert not coverage(derive(zero), G_V3S).in_support           # a variant without 0 mm bones


# ---------------------------------------------------------------------------
# 2. Finger length
# ---------------------------------------------------------------------------


def test_finger_length_definition():
    for seed in range(60):
        d = sample_derivation(seed, G_FULL)
        st = L.Structure.from_steps(d.steps)
        fl = st.finger_lengths()
        bones = {}
        for s in d.steps:
            if s.production == "Phalanx":
                bones.setdefault(s.params["digit_id"], {})[s.params["p"]] = s.params["length"]
        digits = {s.params["digit_id"]: s.params for s in d.steps if s.production == "Digit"}

        def tip(did):
            p = digits[did]
            own = sum(bones[did].values())
            if p["top_level"]:
                return own
            host, k = p["mount"][1:].rsplit("p", 1)          # "d{host}p{k}"
            return tip_base(host, int(k)) + own

        def tip_base(host, k):
            p = digits[host]
            up = sum(bones[host][i] for i in range(k))
            if p["top_level"]:
                return up
            hh, kk = p["mount"][1:].rsplit("p", 1)
            return tip_base(hh, int(kk)) + up

        top_of = st.top_of()
        want = {t: max(tip(e) for e in digits if top_of[e] == t) for t in fl}
        assert all(abs(fl[t] - want[t]) < 1e-12 for t in fl), (seed, fl, want)


def test_finger_length_presets():
    assert FINGER_LENGTH_CAP_MM == 250.0 == round(221 * 1.1 / 50) * 50
    assert SIMULATOR.max_finger_length_mm == 250.0
    assert GenerationLimits().max_finger_length_mm is None
    with pytest.raises(ValueError):
        GenerationLimits(max_finger_length_mm=0)
    d = next(sample_derivation(s, G_V1) for s in range(200)
             if max(_lengths_mm(sample_derivation(s, G_V1)).values()) > 0.25)
    rep = L.check(d, SIMULATOR)
    assert "max_finger_length_mm" in rep.failing and "finger length" in rep.summary()


@pytest.mark.parametrize("name", ["GRAMMAR", "G_V1", "G_FULL"])
def test_sampling_keeps_the_cap(name):
    dist = {"GRAMMAR": GRAMMAR, "G_V1": G_V1, "G_FULL": G_FULL}[name]
    lim = SIMULATOR if name != "G_FULL" else GenerationLimits(max_finger_length_mm=150.0)
    cap = lim.max_finger_length_mm / 1000
    binds = 0
    for seed in range(400):
        d = sample_derivation(seed, dist, limits=lim)
        assert L.check(d, lim).ok
        assert max(_lengths_mm(d).values()) <= cap + 1e-9
        free = sample_derivation(seed, dist, limits=replace(lim, max_finger_length_mm=None))
        binds += max(_lengths_mm(free).values()) > cap + 1e-9
    assert binds > 10                       # the cap binds on these hands


def test_a_cap_that_never_binds_changes_no_draw():
    never = replace(SIMULATOR, max_finger_length_mm=6 * 80.0)        # 6 bones of the longest sampled length
    for seed in range(200):
        assert derivation_to_json(sample_derivation(seed, GRAMMAR, limits=never)) == \
            derivation_to_json(sample_derivation(seed, GRAMMAR, limits=SIM_NO_CAP))
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    a = b = sample_derivation(0, GRAMMAR, limits=SIM_NO_CAP)
    for _ in range(150):
        try:
            a = vary(a, rng_a, GRAMMAR, operators=EVOLUTION_OPERATORS, limits=never)
        except VariationImpossible:
            a = None
        try:
            b = vary(b, rng_b, GRAMMAR, operators=EVOLUTION_OPERATORS, limits=SIM_NO_CAP)
        except VariationImpossible:
            b = None
        assert (a is None) == (b is None)
        if a is None:
            break
        assert derivation_to_json(a) == derivation_to_json(b)


@pytest.mark.parametrize("name", ["GRAMMAR", "G_FULL"])
def test_operators_keep_the_cap_constructively(name):
    """Raw operator output (before vary's checks) is within the cap; tight caps
    with branches allowed exercise insert, branch, regrow, delete and resample."""
    dist = GRAMMAR if name == "GRAMMAR" else G_FULL
    for lim in (SIMULATOR, GenerationLimits(max_finger_length_mm=120.0)):
        rng = np.random.default_rng(3)
        n_out = 0
        for c in range(20):
            d = sample_derivation(c, dist, limits=lim)
            for _ in range(10):
                for op, fn in _OPERATOR_FNS.items():
                    out = fn(rng, dist, d, lim=L.context(lim, d.steps))
                    if out is not None:
                        n_out += 1
                        assert L.check(out, lim).ok, (op, L.check(out, lim).summary())
                try:
                    d = vary(d, rng, dist, operators=tuple(_OPERATOR_FNS), limits=lim)
                except VariationImpossible:
                    pass
        assert n_out > 1000


def test_lengthening_stops_at_the_cap():
    lim = GenerationLimits(max_finger_length_mm=100.0)
    d = sample_derivation(5, GRAMMAR, limits=lim)
    rng = np.random.default_rng(0)
    for op in ("lengthen_segment", "fine_lengthen_segment", "insert_phalanx"):
        for _ in range(400):
            child = apply_operator(d, rng, GRAMMAR, op, limits=lim)
            if child is not None:
                d = child
    assert max(_lengths_mm(d).values()) <= 0.100 + 1e-9
    assert max(_lengths_mm(d).values()) >= 0.095                   # it got there
    full = [t for t, v in _lengths_mm(d).items() if v > 0.099]
    assert full


def test_a_hand_over_the_cap_never_gets_longer():
    rng = np.random.default_rng(9)
    n = 0
    for seed in range(80):
        d = sample_derivation(seed, G_V1)
        base = L.check(d, SIMULATOR).excess["max_finger_length_mm"]
        if not base:
            continue
        for _ in range(8):
            try:
                child = vary(d, rng, G_V1, operators=EVOLUTION_OPERATORS + EVOLUTION_OPERATORS_FINE, limits=SIMULATOR)
            except VariationImpossible:
                continue
            assert L.check(child, SIMULATOR).excess["max_finger_length_mm"] <= base
            n += 1
    assert n > 50


# ---------------------------------------------------------------------------
# 3. Coarse and fine pools
# ---------------------------------------------------------------------------


def test_pools():
    assert EVOLUTION_OPERATORS_COARSE is EVOLUTION_OPERATORS
    assert STAGE_OPERATORS == {"coarse": EVOLUTION_OPERATORS, "fine": EVOLUTION_OPERATORS_FINE}
    assert set(EVOLUTION_OPERATORS_FINE) == {
        "fine_step_axis", "fine_step_limits", "fine_slide_mount", "fine_shift_mount", "fine_turn_mount",
        "fine_step_bend_rpy", "fine_step_root_length", "fine_step_radius", "fine_step_segment_length"}
    assert not set(EVOLUTION_OPERATORS_FINE) & set(EVOLUTION_OPERATORS)
    assert FINE_LENGTH_STEP_PAIR == ("fine_lengthen_segment", "fine_shorten_segment")
    assert INVERSE_OF["fine_lengthen_segment"] == "fine_shorten_segment"
    assert INVERSE_OF["fine_shorten_segment"] == "fine_lengthen_segment"
    assert set(FINE_LENGTH_STEP_PAIR) | {"fine_step_segment_length"} <= TARGETABLE_OPERATORS


def _flat(d):
    """uid (or 'hand') -> {parameter path: value}."""
    out = {}
    for s in d.steps:
        key = "hand" if s.path == "hand" else s.params["uid"]
        flat = {"mount_offset": (0.0, 0.0)} if s.production in ("Digit", "PalmBody") else {}
        for k, v in s.params.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        out[key] = (s.production, flat)
    return out


def _angle_diff_deg(a, b):
    return abs((math.degrees(a - b) + 180.0) % 360.0 - 180.0)


def _one_fine_unit(parent, child, op):
    a, b = _flat(parent), _flat(child)
    assert set(a) == set(b)
    changes = []
    for key in a:
        assert a[key][0] == b[key][0]
        pa, pb = a[key][1], b[key][1]
        assert set(pa) == set(pb)
        for k in pa:
            if pa[k] != pb[k]:
                changes.append((key, k, pa[k], pb[k]))
    assert len(changes) == 1, (op, changes)
    _, k, old, new = changes[0]
    if k in ("length", "root_length", "capsule_radius_m"):
        assert abs(abs(new - old) - 0.001) < 1e-9, (op, k, old, new)
    elif k == "mount_frac":
        assert abs(abs(new - old) - 0.01) < 1e-9
    elif k == "mount_offset":
        d = [abs(x - y) for x, y in zip(old, new)]
        assert sorted(d)[0] < 1e-12 and abs(sorted(d)[1] - 0.001) < 1e-9
    elif k in ("mount_rpy", "direction_rpy", "bend_rpy"):
        d = sorted(_angle_diff_deg(x, y) for x, y in zip(old, new))
        assert d[0] < 1e-9 and d[1] < 1e-9 and abs(d[2] - 5.0) < 1e-6, (op, d)
    elif k in ("axis", "module.axis"):
        ia, ib = fine_axis_indices(old), fine_axis_indices(new)
        assert ia is not None and ib is not None
        assert (ia[0] == ib[0] and min((ia[1] - ib[1]) % 72, (ib[1] - ia[1]) % 72) == 1) or \
            (abs(ia[0] - ib[0]) == 1 and (ia[1] == ib[1] or 0 in (ia[0], ib[0]) or 36 in (ia[0], ib[0])))
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(old, new))))))
        assert ang <= 5.0 + 1e-6
    elif k in ("limits", "module.limits"):
        d = sorted(abs(math.degrees(x - y)) for x, y in zip(old, new))
        assert d[0] < 1e-9 and abs(d[1] - 5.0) < 1e-6
    else:
        raise AssertionError((op, k))
    return k


def _in_support(d, dist):
    hand = next(s for s in d.steps if s.path == "hand").params
    lo, hi = root_length_support_m(dist)
    assert lo - 1e-9 <= hand["root_length"] <= hi + 1e-9
    assert min(dist.capsule_radius_choices_m) - 1e-9 <= hand["capsule_radius_m"] <= max(dist.capsule_radius_choices_m) + 1e-9
    rlo, rhi = (v * DEG for v in dist.revolute_limit_range_deg)
    for s in d.steps:
        p = s.params
        if s.production in ("Digit", "PalmBody"):
            assert -1e-9 <= p["mount_frac"] <= 1 + 1e-9
            assert all(abs(v) <= dist.mount_lateral_max_m + 1e-9 for v in p.get("mount_offset", (0.0, 0.0))
                       ) or (s.production == "Digit" and dist.mount_on_host_surface)
        if s.production == "PalmBody":
            lo, hi = palm_body_length_support_m(dist)
            assert lo - 1e-9 <= p["length"] <= hi + 1e-9
        if s.production == "Phalanx":
            lo, hi = link_length_support_m(dist)
            assert lo - 1e-9 <= p["length"] <= hi + 1e-9
            if p["module"]["kind"] == "R":
                assert rlo - 1e-9 <= p["module"]["limits"][0] < p["module"]["limits"][1] <= rhi + 1e-9
            for comp in range(3):
                g = fine_bend_component_grid(dist, comp)
                assert g[0] - 1e-9 <= p["bend_rpy"][comp] <= g[-1] + 1e-9


@pytest.mark.parametrize("name", ["GRAMMAR", "G_V3S"])
def test_each_fine_operator_moves_one_parameter_by_one_unit(name):
    dist = GRAMMAR if name == "GRAMMAR" else G_V3S
    rng = np.random.default_rng(21)
    seen = {}
    for c in range(25):
        d = sample_derivation(c, dist, limits=SIMULATOR)
        struct0 = L.Structure.from_steps(d.steps)
        for _ in range(25):
            for op in EVOLUTION_OPERATORS_FINE:
                child = apply_operator(d, rng, dist, op, limits=SIMULATOR)
                if child is None:
                    continue
                seen.setdefault(op, set()).add(_one_fine_unit(d, child, op))
                st = L.Structure.from_steps(child.steps)
                assert (st.palm, st.digits, st.kinds) == (struct0.palm, struct0.digits, struct0.kinds)
                assert L.check(child, SIMULATOR).ok
                _in_support(child, dist)
            try:
                d = vary(d, rng, dist, operators=EVOLUTION_OPERATORS_FINE, limits=SIMULATOR)
            except VariationImpossible:
                pass
    want = set(EVOLUTION_OPERATORS_FINE) - ({"fine_shift_mount"} if name == "G_V3S" else set())
    assert set(seen) == want, set(EVOLUTION_OPERATORS_FINE) - set(seen)


def test_fine_length_pair_is_an_exact_inverse():
    rng = np.random.default_rng(5)
    n = 0
    for c in range(30):
        d = sample_derivation(c, GRAMMAR, limits=SIMULATOR)
        for s in d.steps:
            if s.production not in ("Phalanx", "PalmBody"):
                continue
            for a, b in (FINE_LENGTH_STEP_PAIR, FINE_LENGTH_STEP_PAIR[::-1]):
                child = apply_operator(d, rng, GRAMMAR, a, target=s.params["uid"], limits=SIMULATOR)
                if child is None:
                    continue
                back = apply_operator(child, rng, GRAMMAR, b, target=s.params["uid"], limits=SIMULATOR)
                assert back is not None and back.steps == d.steps
                n += 1
    assert n > 300


def test_fine_steps_respect_the_cap():
    lim = replace(SIMULATOR, max_finger_length_mm=120.0)
    rng = np.random.default_rng(2)
    for c in range(20):
        d = sample_derivation(c, GRAMMAR, limits=lim)
        for _ in range(60):
            try:
                d = vary(d, rng, GRAMMAR, operators=("fine_lengthen_segment",) + EVOLUTION_OPERATORS_FINE, limits=lim)
            except VariationImpossible:
                continue
            assert L.check(d, lim).ok


# ---------------------------------------------------------------------------
# 4. The fine grid contains the coarse one
# ---------------------------------------------------------------------------


def _on_fine_grid(d):
    def whole(v, unit):
        return abs(v / unit - round(v / unit)) < 1e-6

    hand = next(s for s in d.steps if s.path == "hand").params
    assert whole(hand["root_length"], 0.001) and whole(hand["capsule_radius_m"], 0.001)
    for s in d.steps:
        p = s.params
        if s.production in ("Digit", "PalmBody"):
            assert whole(p["mount_frac"], 0.01)
            rpy = p["mount_rpy"] if s.production == "Digit" else p["direction_rpy"]
            assert all(whole(math.degrees(a), 5.0) for a in rpy)
        if s.production == "PalmBody":
            assert whole(p["length"], 0.001) and fine_axis_indices(p["axis"]) is not None
        if s.production == "Phalanx":
            assert whole(p["length"], 0.001) and fine_axis_indices(p["module"]["axis"]) is not None
            assert all(whole(math.degrees(a), 5.0) for a in p["bend_rpy"])


@pytest.mark.parametrize("name", ["GRAMMAR", "G_FULL", "G_V3S"])
def test_coarse_hands_are_fine_hands(name):
    dist = {"GRAMMAR": GRAMMAR, "G_FULL": G_FULL, "G_V3S": G_V3S}[name]
    rng = np.random.default_rng(8)
    for c in range(40):
        d = sample_derivation(c, dist, limits=SIMULATOR)
        for k in range(6):
            _on_fine_grid(d)
            cov_c, cov_f = coverage(derive(d), dist), coverage(derive(d), dist, resolution="fine")
            assert cov_f.in_support or not cov_c.in_support, cov_f.out_of_support
            if cov_c.in_support:
                assert cov_f.in_support
            try:
                d = vary(d, rng, dist, operators=EVOLUTION_OPERATORS, limits=SIMULATOR)
            except VariationImpossible:
                pass


def test_fine_chains_stay_in_the_fine_support():
    rng = np.random.default_rng(13)
    n_fine_only = 0
    for c in range(25):
        d = sample_derivation(c, GRAMMAR, limits=SIMULATOR)
        for _ in range(30):
            d = vary(d, rng, GRAMMAR, operators=EVOLUTION_OPERATORS_FINE, limits=SIMULATOR)
        cov = coverage(derive(d), GRAMMAR, resolution="fine")
        assert cov.in_support, cov.out_of_support
        n_fine_only += not coverage(derive(d), GRAMMAR).in_support
    assert n_fine_only > 20                 # fine values leave the coarse grid


def test_resolution_is_validated():
    with pytest.raises(ValueError):
        coverage(derive(sample_derivation(0, GRAMMAR)), GRAMMAR, resolution="medium")
