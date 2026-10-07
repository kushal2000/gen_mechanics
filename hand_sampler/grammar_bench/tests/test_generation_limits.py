"""Generation limits (``hand_sampler/grammar/limits.py``): the layer between
what the grammar can express and what the simulator can build.

1. ``limits=None`` is byte-identical to the code before limits existed
   (digests of derivation JSON and phenotype hashes recorded on the parent
   commit e77d28a), and ``GenerationLimits()`` (or any limits that never
   bind) equals ``None``.
2. ``SIMULATOR`` is exactly the simulator envelope plus a finger-length cap:
   within ``SIMULATOR`` without the cap iff ``grammar_envelope._admit_structural``
   admits the derived model.
3. Every design sampled under ``SIMULATOR`` (2000 per variant) and every step
   of long random mutation chains passes ``_admit_structural``, and the
   operators reach that constructively (their raw output is within the
   limits; no rejection).
4. Each limit on its own is respected by sampling and mutation.
5. The grammar's no-empty-palm rule (``Distribution.palm_body_needs_digit``,
   not a limit) is kept by sampling and mutation, with or without limits.

``grammar_envelope.py`` imports ``isaacsimenvs/__init__.py`` (Isaac Lab) when
imported normally, so it is loaded by file path, as the grammar viewer does.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar import limits as L
from hand_sampler.grammar.canonical import phenotype_hash
from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    EVOLUTION_OPERATORS_V1,
    MINIMAL_STRUCTURAL_OPERATORS,
    OPERATORS,
    SMALL_STEP_OPERATORS,
    _OPERATOR_FNS,
    TARGETABLE_OPERATORS,
    VariationImpossible,
    apply_operator,
    derivation_to_json,
    derive,
    sample_derivation,
    vary,
    vary_tracked,
)
from dataclasses import replace

from hand_sampler.grammar.limits import SIMULATOR, SIMULATOR_ENVELOPE, GenerationLimits
from hand_sampler.grammar.variants import NAMED_DISTRIBUTIONS, build_distribution

UNLIMITED = GenerationLimits()

REPO = Path(__file__).resolve().parents[3]


def _envelope():
    name = "isaacsimenvs.inhand_reorient.scene.grammar_envelope"
    if name in sys.modules:
        return sys.modules[name]
    root = REPO / "isaacsimenvs"
    for pkg, path in (("isaacsimenvs", root), ("isaacsimenvs.inhand_reorient", root / "inhand_reorient"),
                      ("isaacsimenvs.inhand_reorient.scene", root / "inhand_reorient" / "scene")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    for mod, path in (("isaacsimenvs.inhand_reorient.palm_calibration", root / "inhand_reorient" / "palm_calibration.py"),
                      (name, root / "inhand_reorient" / "scene" / "grammar_envelope.py")):
        if mod not in sys.modules:
            spec = importlib.util.spec_from_file_location(mod, str(path))
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod] = module
            spec.loader.exec_module(module)
    return sys.modules[name]


GE = _envelope()


def _dist(name):
    d = NAMED_DISTRIBUTIONS[name]
    return d[0] if isinstance(d, tuple) else d


def _admitted(d) -> bool:
    return GE._admit_structural(derive(d)).ok


# ---------------------------------------------------------------------------
# 1. limits=None is byte-identical; UNLIMITED equals None
# ---------------------------------------------------------------------------

# "evo" is the 17-operator pool the digests below were recorded with
# (EVOLUTION_OPERATORS before step_segment_length was added).
POOLS = {"evo": EVOLUTION_OPERATORS_V1, "default": None, "minimal": MINIMAL_STRUCTURAL_OPERATORS + SMALL_STEP_OPERATORS}


def sample_digest(name, n, **kw):
    h, p = hashlib.sha256(), hashlib.sha256()
    for seed in range(n):
        d = sample_derivation(seed, _dist(name), **kw)
        h.update(derivation_to_json(d).encode())
        p.update(phenotype_hash(derive(d)).encode())
    return h.hexdigest()[:16], p.hexdigest()[:16]


def chain_digest(name, pool, n_chains, n_steps, **kw):
    dist = _dist(name)
    h = hashlib.sha256()
    for c in range(n_chains):
        rng = np.random.default_rng(1000 + c)
        d = sample_derivation(c, dist)
        for _ in range(n_steps):
            try:
                if POOLS[pool] is None:
                    d = vary(d, rng, dist, **kw)
                else:
                    d = vary(d, rng, dist, operators=POOLS[pool], **kw)
                h.update(derivation_to_json(d).encode())
            except VariationImpossible:
                h.update(b"X")
    return h.hexdigest()[:16]


def tracked_digest(name, n_chains, n_steps, **kw):
    dist = _dist(name)
    h = hashlib.sha256()
    for c in range(n_chains):
        rng = np.random.default_rng(2000 + c)
        d = sample_derivation(c, dist)
        for k in range(n_steps):
            child, op = vary_tracked(d, rng, dist, EVOLUTION_OPERATORS_V1, **kw)
            h.update(op.encode())
            if child is not None:
                d = child
                h.update(derivation_to_json(d).encode())
            op2 = EVOLUTION_OPERATORS_V1[k % len(EVOLUTION_OPERATORS_V1)]
            res = apply_operator(d, rng, dist, op2, **kw)
            h.update(b"N" if res is None else derivation_to_json(res).encode())
    return h.hexdigest()[:16]


# Recorded with the code of commit e77d28a (before limits existed): 150 seeds
# per variant (derivation-JSON digest, phenotype-hash digest), 6 mutation
# chains x 20 steps per (variant, operator pool), and 4 x 15 vary_tracked +
# apply_operator steps.
SAMPLES = {
    'G_BEND': ('db5928fe4c44d4bc', '1ce6907c66edf75f'),
    'G_CONT': ('474417ae80724702', '2198cfc6e95efd94'),
    'G_FULL': ('e547b3603eee84af', 'aa76bad30837ff8f'),
    'G_FULL_INS': ('e547b3603eee84af', 'aa76bad30837ff8f'),
    'G_NOBRANCH': ('4f3b93257a19345c', '2132d73f4534cda9'),
    'G_NOBRANCH_INS': ('4f3b93257a19345c', '2132d73f4534cda9'),
    'G_NOCOUPLE': ('e47e1a00b6ba274c', 'b279f2ebaf4bab79'),
    'G_NOPALMJOINT': ('08536e130fcf8097', 'f37ed9958a3f0156'),
    'G_SERIAL': ('828591168dad5084', 'b82419889d04350e'),
    'G_V1': ('e5e65aa811003d1d', '60fe9bfdd7e843f3'),
    'G_V1S': ('2786faa3a2a9542d', 'c2907f876f433ca1'),
    'G_V2': ('e5cba91adb5d78ed', '0da832c775b2d9f7'),
    'G_V2S': ('c7a5b45467525a0c', '510d5292edb8a654'),
    'G_V3': ('8457bd4d1e9a0afc', 'e8d7109ce0694683'),
    'G_V3S': ('ae0eb7d3436f77f7', '69ef34818b32ee4b'),
}
CHAINS = {
    ('G_BEND', 'default'): '46e9bab680981a78',
    ('G_BEND', 'evo'): '9d544f8654875473',
    ('G_BEND', 'minimal'): '39112a766fabe1a4',
    ('G_CONT', 'default'): '261de3d2ea69d94b',
    ('G_CONT', 'evo'): '25d4d73c98face36',
    ('G_CONT', 'minimal'): 'ca5c12a71124ecd0',
    ('G_FULL', 'default'): 'a9f1f3bbe9dff7d8',
    ('G_FULL', 'evo'): '23d1be151174a1ef',
    ('G_FULL', 'minimal'): '41b69539af3142ad',
    ('G_FULL_INS', 'default'): '5cf90761b7342686',
    ('G_FULL_INS', 'evo'): '23d1be151174a1ef',
    ('G_FULL_INS', 'minimal'): '41b69539af3142ad',
    ('G_V1', 'default'): 'a4efc7b2d7e7094b',
    ('G_V1', 'evo'): 'c06ac3769413b345',
    ('G_V1', 'minimal'): '3491200a8c3a4551',
    ('G_V2S', 'default'): '3a1be871f4e68987',
    ('G_V2S', 'evo'): '39b3dc3028583170',
    ('G_V2S', 'minimal'): '5a2a7904d1c07dcd',
    ('G_V3S', 'default'): 'e905ed40a21a1b2f',
    ('G_V3S', 'evo'): '1c00bcb0b3565e48',
    ('G_V3S', 'minimal'): '2c787e3ac2e3b2c7',
}
TRACKED = {'G_FULL': '4210b8b823591c1e', 'G_NOBRANCH_INS': '22a1fcf3100164f6', 'G_V1S': 'aeb55e51d5dc2fff'}

# Limits that never bind on any of the hands above (every cap far beyond what
# the variants sample or the chains reach): they must not perturb a single draw.
LOOSE = GenerationLimits(max_digits=60, max_joints_per_digit=400, max_palm_bodies=60, max_jointed_palm_bodies=60,
                         max_digits_per_jointed_palm_body=60, max_finger_chains=120)

LIMIT_ARGS = {"none": {}, "None": {"limits": None}, "UNLIMITED": {"limits": UNLIMITED}, "LOOSE": {"limits": LOOSE}}


@pytest.mark.parametrize("mode", list(LIMIT_ARGS))
@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_sampling_byte_identical(name, mode):
    assert sample_digest(name, 150, **LIMIT_ARGS[mode]) == SAMPLES[name]


@pytest.mark.parametrize("mode", list(LIMIT_ARGS))
@pytest.mark.parametrize("key", sorted(CHAINS))
def test_mutation_chains_byte_identical(key, mode):
    assert chain_digest(*key, 6, 20, **LIMIT_ARGS[mode]) == CHAINS[key]


@pytest.mark.parametrize("mode", list(LIMIT_ARGS))
@pytest.mark.parametrize("name", sorted(TRACKED))
def test_tracked_and_apply_operator_byte_identical(name, mode):
    assert tracked_digest(name, 4, 15, **LIMIT_ARGS[mode]) == TRACKED[name]


def test_unlimited_is_the_default_and_limits_nothing():
    assert UNLIMITED.is_unlimited and not SIMULATOR.is_unlimited
    assert not hasattr(L, "DEFAULT_LIMITS") and not hasattr(L, "UNLIMITED") and not hasattr(L, "PRESETS")
    assert "require_digit_on_palm_body" not in L.LIMIT_KEYS
    for seed in range(40):
        assert L.check(sample_derivation(seed, _dist("G_FULL")), UNLIMITED).ok


# ---------------------------------------------------------------------------
# 2. SIMULATOR is exactly the envelope's structural admission
# ---------------------------------------------------------------------------


def test_simulator_preset_matches_envelope_constants():
    assert SIMULATOR.max_digits == GE.MAX_DIGITS == GE.N_FINGERS
    assert SIMULATOR.max_joints_per_digit == GE.MAX_JOINTS_PER_DIGIT == GE.N_JOINTS_PER_FINGER
    assert SIMULATOR.max_jointed_palm_bodies == GE.MAX_JOINTED_PALM_BODIES
    assert SIMULATOR.max_finger_chains == GE.N_FINGERS
    assert SIMULATOR.allowed_modules == ("R",) and not SIMULATOR.allow_branches
    assert not SIMULATOR.allow_stacked_palm_joints and SIMULATOR.max_digits_per_jointed_palm_body == 1
    # SIMULATOR adds the finger-length cap to the envelope's shape
    assert SIMULATOR.max_finger_length_mm == 250.0 and SIMULATOR_ENVELOPE == replace(SIMULATOR,
                                                                                     max_finger_length_mm=None)


def _equivalence_designs():
    for name in ("G_FULL", "G_NOBRANCH", "G_NOCOUPLE", "G_SERIAL", "G_V1", "G_V3S"):
        for seed in range(150):
            yield sample_derivation(seed, _dist(name))
    # mutated structures the sampler alone does not produce (stacked and
    # re-parented palm bodies, digits moved between hosts, branch re-attachment)
    pool = EVOLUTION_OPERATORS + ("remove_palm_body", "regrow_subtree", "resample_parameter", "add_digit")
    for c in range(40):
        rng = np.random.default_rng(c)
        d = sample_derivation(500 + c, _dist("G_FULL"))
        for _ in range(25):
            try:
                d = vary(d, rng, _dist("G_FULL"), operators=pool)
            except VariationImpossible:
                continue
            yield d


def test_simulator_limits_equal_admit_structural():
    """The envelope's shape is exactly SIMULATOR_ENVELOPE; SIMULATOR (which
    also caps finger length) is inside it."""
    n = n_ok = 0
    for d in _equivalence_designs():
        rep = L.check(d, SIMULATOR_ENVELOPE)
        oracle = GE._admit_structural(derive(d))
        assert rep.ok == oracle.ok, (rep.failing, oracle.reasons)
        if L.check(d, SIMULATOR).ok:
            assert oracle.ok
        n += 1
        n_ok += oracle.ok
    assert n > 1500 and 0 < n_ok < n, (n, n_ok)


# ---------------------------------------------------------------------------
# 3. SIMULATOR: everything generated or mutated passes _admit_structural
# ---------------------------------------------------------------------------

SIM_VARIANTS = ("G_V1", "G_V1S", "G_V2S", "G_V3S", "G_FULL")


@pytest.mark.parametrize("name", SIM_VARIANTS)
def test_simulator_sampling_always_admitted(name):
    dist = _dist(name)
    digits = set()
    for seed in range(2000):
        d = sample_derivation(seed, dist, limits=SIMULATOR)
        assert L.check(d, SIMULATOR).ok
        assert _admitted(d), (name, seed, GE._admit_structural(derive(d)).reasons)
        digits.add(sum(1 for s in d.steps if s.production == "Digit"))
    assert max(digits) == 5            # the caps are reached, not just respected


@pytest.mark.parametrize("pool", ["evo", "wide"])
@pytest.mark.parametrize("name", SIM_VARIANTS)
def test_simulator_mutation_chains_always_admitted(name, pool):
    """50 chains x 50 steps, each step a random operator (VariationImpossible
    skipped), every child admitted."""
    dist = _dist(name)
    ops = EVOLUTION_OPERATORS if pool == "evo" else tuple(_OPERATOR_FNS)
    n_children = 0
    for c in range(50):
        rng = np.random.default_rng(10_000 + c)
        d = sample_derivation(c, dist, limits=SIMULATOR)
        for _ in range(50):
            try:
                d = vary(d, rng, dist, operators=ops, limits=SIMULATOR)
            except VariationImpossible:
                continue
            n_children += 1
            assert _admitted(d), GE._admit_structural(derive(d)).reasons
    assert n_children > 1200


@pytest.mark.parametrize("name", ("G_FULL", "G_V1S", "G_BEND"))
def test_operators_respect_limits_constructively(name):
    """The raw operator output (before vary's derive/limit check) is already
    within the limits: no operator relies on rejection."""
    dist = _dist(name)
    for limits in (SIMULATOR, GenerationLimits(max_digits=3, max_joints_per_digit=4, max_jointed_palm_bodies=1,
                                               allow_stacked_palm_joints=False, max_finger_chains=3,
                                               max_digits_per_jointed_palm_body=1, max_palm_bodies=2)):
        rng = np.random.default_rng(7)
        for c in range(25):
            d = sample_derivation(c, dist, limits=limits)
            for _ in range(12):
                for op, fn in _OPERATOR_FNS.items():
                    ctx = L.context(limits, d.steps)
                    out = fn(rng, dist, d, lim=ctx)
                    if out is not None:
                        assert L.check(out, limits).ok, (op, L.check(out, limits).failing)
                try:
                    d = vary(d, rng, dist, operators=tuple(_OPERATOR_FNS), limits=limits)
                except VariationImpossible:
                    pass


def test_vary_tracked_and_apply_operator_take_limits():
    dist = _dist("G_V1")
    rng = np.random.default_rng(3)
    d = sample_derivation(0, dist, limits=SIMULATOR)
    for _ in range(200):
        child, _op = vary_tracked(d, rng, dist, EVOLUTION_OPERATORS, limits=SIMULATOR)
        if child is not None:
            assert _admitted(child)
            d = child
    assert apply_operator(d, rng, dist, "add_branch_digit", limits=SIMULATOR) is None
    with pytest.raises(VariationImpossible):
        vary(d, rng, dist, operator="add_branch_digit", limits=SIMULATOR)


def test_operator_inapplicable_at_a_cap():
    """add_minimal_digit at the digit cap is inapplicable; under no limits it applies."""
    dist = _dist("G_V1")
    lim = GenerationLimits(max_digits=2, max_finger_chains=5)
    seed = next(s for s in range(200)
                if sum(1 for st in sample_derivation(s, dist, limits=lim).steps
                       if st.production == "Digit") == 2)
    d = sample_derivation(seed, dist, limits=lim)
    rng = np.random.default_rng(0)
    assert apply_operator(d, rng, dist, "add_minimal_digit", limits=lim) is None
    assert apply_operator(d, rng, dist, "add_minimal_digit") is not None


def test_mutation_never_worsens_a_hand_outside_the_limits():
    """A hand already outside the limits (e.g. a G_FULL sample, like a
    projected commercial hand) can still be mutated, but no limit gets worse."""
    dist = _dist("G_FULL")
    rng = np.random.default_rng(11)
    for seed in range(40):
        d = sample_derivation(seed, dist)
        base = L.check(d, SIMULATOR).excess
        for _ in range(10):
            try:
                child = vary(d, rng, dist, operators=EVOLUTION_OPERATORS, limits=SIMULATOR)
            except VariationImpossible:
                continue
            ex = L.check(child, SIMULATOR).excess
            assert all(ex[k] <= base[k] for k in ex), (ex, base)


# ---------------------------------------------------------------------------
# 4. Each limit on its own
# ---------------------------------------------------------------------------


def _measure(d):
    st = L.Structure.from_steps(d.steps)
    return {
        "digits": len(st.top_digits()),
        "joints": max(st.joints_per_digit().values(), default=0),
        "palm": len(st.palm),
        "jointed": len(st.jointed()),
        "stacked": len(st.stacked()),
        "carried": max(st.carried_counts().values(), default=0),
        "chains": st.finger_chains(),
        "branches": len(st.branching_links()) + sum(1 for s in d.steps if s.production == "Digit"
                                                    and not s.params["top_level"]),
        "kinds": set(st.kinds.values()),
    }


SINGLE_LIMITS = [
    (GenerationLimits(max_digits=3), lambda m: m["digits"] <= 3),
    (GenerationLimits(max_joints_per_digit=3), lambda m: m["joints"] <= 3),
    (GenerationLimits(max_palm_bodies=1), lambda m: m["palm"] <= 1),
    (GenerationLimits(max_jointed_palm_bodies=1), lambda m: m["jointed"] <= 1),
    (GenerationLimits(max_jointed_palm_bodies=0), lambda m: m["jointed"] == 0),
    (GenerationLimits(allow_stacked_palm_joints=False), lambda m: m["stacked"] == 0),
    (GenerationLimits(max_digits_per_jointed_palm_body=1), lambda m: m["carried"] <= 1),
    (GenerationLimits(max_finger_chains=3), lambda m: m["chains"] <= 3),
    (GenerationLimits(allow_branches=False), lambda m: m["branches"] == 0),
    (GenerationLimits(allowed_modules=("R", "P")), lambda m: m["kinds"] <= {"R", "P"}),
    (GenerationLimits(allowed_modules=("R", "Coupled")), lambda m: m["kinds"] <= {"R", "Coupled"}),
]


@pytest.mark.parametrize("limits,ok", SINGLE_LIMITS, ids=[repr({k: v for k, v in vars(lim).items()
                                                                  if v != getattr(UNLIMITED, k)})
                                                            for lim, _ in SINGLE_LIMITS])
def test_each_limit_respected(limits, ok):
    dist = _dist("G_FULL")
    binding = 0
    for seed in range(300):
        d = sample_derivation(seed, dist, limits=limits)
        assert ok(_measure(d)) and L.check(d, limits).ok
        binding += not L.check(sample_derivation(seed, dist), limits).ok
    assert binding > 0           # the limit does bind on the free grammar
    pool = tuple(_OPERATOR_FNS)
    for c in range(15):
        rng = np.random.default_rng(c)
        d = sample_derivation(c, dist, limits=limits)
        for _ in range(30):
            try:
                d = vary(d, rng, dist, operators=pool, limits=limits)
            except VariationImpossible:
                continue
            assert ok(_measure(d)) and L.check(d, limits).ok


def test_joint_budget_counts_branches():
    """With branches allowed, a top-level digit's joints include its branches."""
    dist = _dist("G_FULL")
    lim = GenerationLimits(max_joints_per_digit=4)
    branched = 0
    for seed in range(400):
        d = sample_derivation(seed, dist, limits=lim)
        branched += any(s.production == "Digit" and not s.params["top_level"] for s in d.steps)
        assert max(L.Structure.from_steps(d.steps).joints_per_digit().values()) <= 4
    assert branched > 0


def test_limits_validation():
    with pytest.raises(ValueError):
        GenerationLimits(allowed_modules=("Coupled",))
    with pytest.raises(ValueError):
        GenerationLimits(allowed_modules=("X",))
    with pytest.raises(ValueError):
        GenerationLimits(max_digits=0)
    assert GenerationLimits(allowed_modules=("Coupled", "R")).allowed_modules == ("R", "Coupled")


def test_report_lines():
    d = next(sample_derivation(s, _dist("G_FULL")) for s in range(100)
             if not L.check(sample_derivation(s, _dist("G_FULL")), SIMULATOR).ok)
    rep = L.check(d, SIMULATOR)
    assert rep.summary().startswith("no (") and all(k in L.LIMIT_KEYS for k in rep.failing)


# ---------------------------------------------------------------------------
# 5. No empty palm parts: a rule of the grammar (palm_body_needs_digit)
# ---------------------------------------------------------------------------


def _ruled(name):
    return replace(_dist(name), palm_body_needs_digit=True)


def test_the_one_grammar_has_the_rule():
    assert build_distribution().palm_body_needs_digit
    assert not any((d[0] if isinstance(d, tuple) else d).palm_body_needs_digit for d in NAMED_DISTRIBUTIONS.values())


@pytest.mark.parametrize("limits", [None, SIMULATOR], ids=["no limits", "SIMULATOR"])
@pytest.mark.parametrize("name", ["G_FULL", "G_V1", "G_V1S", "G_V2", "G_V2S", "G_V3S"])
def test_no_empty_palm_parts_sampled_or_mutated(name, limits):
    dist = _ruled(name)
    n_palm = 0
    for seed in range(400):
        d = sample_derivation(seed, dist, limits=limits)
        st = L.Structure.from_steps(d.steps)
        assert not st.empty_palm_bodies() and (limits is None or L.check(d, limits).ok)
        n_palm += len(st.palm)
    assert n_palm > 100
    for c in range(15):
        rng = np.random.default_rng(c)
        d = sample_derivation(c, dist, limits=limits)
        for _ in range(40):
            try:
                d = vary(d, rng, dist, operators=EVOLUTION_OPERATORS + ("remove_digit_minimal", "regrow_subtree",
                                                                        "resample_parameter", "remove_palm_body"),
                         limits=limits)
            except VariationImpossible:
                continue
            assert not L.Structure.from_steps(d.steps).empty_palm_bodies()


def test_without_the_rule_palm_parts_may_be_empty():
    dist = _dist("G_FULL")
    assert any(L.Structure.from_steps(sample_derivation(s, dist, limits=SIMULATOR).steps).empty_palm_bodies()
               for s in range(200))


def test_add_palm_body_brings_a_finger():
    dist = _ruled("G_FULL")
    rng = np.random.default_rng(5)
    n = 0
    for seed in range(60):
        d = sample_derivation(seed, dist)
        child = apply_operator(d, rng, dist, "add_palm_body")
        if child is None:
            continue
        a, b = L.Structure.from_steps(d.steps), L.Structure.from_steps(child.steps)
        (new,) = set(b.palm) - set(a.palm)
        assert len(b.top_digits()) == len(a.top_digits()) + 1
        assert [m for m, _ in b.digits.values()].count(new) == 1 and not b.empty_palm_bodies()
        n += 1
        # without the rule it adds an empty palm body, as before
        plain = apply_operator(d, np.random.default_rng(seed), _dist("G_FULL"), "add_palm_body")
        if plain is not None:
            assert len(L.Structure.from_steps(plain.steps).top_digits()) == len(a.top_digits())
    assert n > 20


def test_removing_the_last_finger_removes_the_palm_part():
    dist = _ruled("G_FULL")
    n = 0
    for seed in range(200):
        d = sample_derivation(seed, dist)
        st = L.Structure.from_steps(d.steps)
        lonely = [t for t in st.top_digits() if st.digits[t][0] in st.palm_leaves()
                  and [m for m, _ in st.digits.values()].count(st.digits[t][0]) == 1]
        if not lonely or len(st.top_digits()) < 2:
            continue
        for rng_seed in range(20):
            child = apply_operator(d, np.random.default_rng(rng_seed), dist, "remove_digit")
            if child is None:
                continue
            b = L.Structure.from_steps(child.steps)
            assert not b.empty_palm_bodies()
            if set(st.top_digits()) - set(b.top_digits()) <= set(lonely):
                assert len(b.palm) < len(st.palm)
                n += 1
        if n > 15:
            break
    assert n > 15


def test_remove_a_finger_removes_any_finger():
    """The current pool's 'remove a finger' (remove_digit) removes fingers of
    any length, so it is not an exact inverse of 'add a finger'
    (add_minimal_digit, one joint)."""
    assert "remove_digit" in EVOLUTION_OPERATORS and "remove_digit_minimal" not in EVOLUTION_OPERATORS
    assert "remove_palm_body_empty" not in EVOLUTION_OPERATORS and "remove_palm_body_empty" in EVOLUTION_OPERATORS_V1
    dist = _dist("G_V1")
    lengths = set()
    for seed in range(60):
        d = sample_derivation(seed, dist, limits=SIMULATOR)
        child = apply_operator(d, np.random.default_rng(seed), dist, "remove_digit", limits=SIMULATOR)
        if child is None:
            continue
        a, b = L.Structure.from_steps(d.steps), L.Structure.from_steps(child.steps)
        (gone,) = set(a.top_digits()) - set(b.top_digits())
        lengths.add(a.digits[gone][1])
    assert max(lengths) >= 3


# ---------------------------------------------------------------------------
# 6. Two fingers on one palm joint (on hold): the grammar side follows the
#    envelope, and the slot layout of every design the envelope admits today
#    is pinned, so a new layout can be checked for backward compatibility.
# ---------------------------------------------------------------------------


def test_simulator_fingers_per_palm_joint_follows_the_envelope():
    """SIMULATOR's ``max_digits_per_jointed_palm_body`` is what one carrier
    slot of the envelope can take: 1 today. A layout that carries several
    finger chains per palm joint declares ``MAX_DIGITS_PER_CARRIER`` (None:
    no cap, only the 5 chains bind); set SIMULATOR to match and the
    equivalence test above checks the rest."""
    assert SIMULATOR.max_digits_per_jointed_palm_body == getattr(GE, "MAX_DIGITS_PER_CARRIER", 1)


def _layout_digest(models):
    import json
    h, n = hashlib.sha256(), 0
    for m in models:
        if not GE._admit_structural(m).ok:
            h.update(b"-")
            continue
        d = GE.canonicalize(m)
        payload = [d.slot_valid.astype(bool).tolist(), np.round(d.slot_origin, 9).tolist(),
                   np.round(d.slot_axis, 9).tolist(), np.round(d.slot_limits, 9).tolist(),
                   np.round(d.slot_length, 9).tolist(), list(d.slot_joint_name), list(d.slot_body_name),
                   list(d.finger_digit_id)]
        h.update(json.dumps(payload).encode())
        n += 1
    return h.hexdigest()[:16], n


# (digest, admitted designs) of canonicalize's 32-slot tables, recorded on
# 8b454cb and unchanged since: 300 samples per variant with limits=None
# (byte-identical by section 1) and the exact projections of the manifest's
# commercial hands.
LAYOUTS = {
    "G_FULL": ("70a058c378b82e1d", 7),
    "G_V1": ("3b909d849986c38d", 230),
    "G_V3S": ("3696e28272f00c78", 224),
    "G_NOBRANCH": ("b986f4e7d69e6ee2", 15),
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_slot_layout_of_admitted_designs_unchanged(name):
    assert _layout_digest(derive(sample_derivation(s, _dist(name))) for s in range(300)) == LAYOUTS[name]


def test_slot_layout_of_commercial_hands_unchanged():
    from hand_sampler.grammar.adapters.projection import project_to_derivation
    from hand_sampler.grammar.adapters.urdf import load_urdf
    from hand_sampler.grammar.experiments import e13_representation as e13

    models = []
    for hand, path, avail, _ in e13._cases(e13._load_manifest()):
        if avail != "available":
            continue
        imp = load_urdf(path, hand_root=hand.get("hand_root"))
        pr = project_to_derivation(imp.model, palm_joints=hand.get("palm_joints", ()),
                                   tip_frames=hand.get("tip_frames", {}))
        models.append(derive(pr.derivation))
    if len(models) < 15:
        pytest.skip("local-only: not every manifest hand is available on this machine")
    assert _layout_digest(models) == ("f380213c7bf21cc7", 14)
