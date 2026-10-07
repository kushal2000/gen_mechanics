"""The one grammar (``variants.build_distribution``): one base distribution
with capability grids wide enough for the commercial hands, plus three
on/off generation rules (surface mounting, spacing, curl and opposition).

1. Under SIMULATOR limits its samples have the same rule properties as G_V3S
   (surface mounts, a straight first bone and a 15-45 deg curl after it,
   transverse hinge axes, an opposing last finger), plus spacing; it is not
   byte-identical to G_V3S (different base counts and mount fractions).
2. The support-only grids do not change sampling at all: the same draws as
   the base without them.
3. Switching a rule off removes its property.
4. The support grids are reachable: step_bend_rpy steps support-only bends,
   step_limits moves an off-menu range, step_mount steps a lateral offset.
5. Every named variant is unchanged (test_generation_limits' digests cover
   them); the composer leaves NAMED_DISTRIBUTIONS alone.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.derive import (
    DerivationStep,
    apply_operator,
    derivation_to_json,
    derive,
    sample_derivation,
)
from hand_sampler.grammar.distributions import DEG, lateral_offset_choices_m
from hand_sampler.grammar.fk import forward_kinematics, rpy_to_matrix
from hand_sampler.grammar.limits import SIMULATOR, check
from hand_sampler.grammar.variants import (
    DEFAULT_DISTRIBUTION,
    G_V3S,
    GRAMMAR_BASE,
    NAMED_DISTRIBUTIONS,
    RULES,
    build_distribution,
)

N = 300
CURL = {round(v, 9) for v in (15 * DEG, 30 * DEG, 45 * DEG)}


def _rule_stats(dist, n=N):
    """Fractions of sampled hands (under SIMULATOR) with each rule property."""
    surface = curl = band = oppose = spaced = 0
    n_multi = 0
    min_sep = []
    for seed in range(n):
        d = sample_derivation(seed, dist, limits=SIMULATOR)
        assert check(d, SIMULATOR).ok
        m = derive(d)
        hand = next(s for s in d.steps if s.path == "hand").params
        r = hand["capsule_radius_m"]
        top = [s for s in d.steps if s.production == "Digit" and s.params["top_level"]]
        phal = [s for s in d.steps if s.production == "Phalanx"]
        surface += all(abs(math.hypot(*s.params.get("mount_offset", (0, 0))) - r) < 1e-9 for s in top)
        curl += all((s.params["p"] == 0 and tuple(s.params["bend_rpy"]) == (0.0, 0.0, 0.0))
                    or (s.params["p"] > 0 and round(s.params["bend_rpy"][1], 9) in CURL) for s in phal)
        band += all(60 - 1e-6 <= math.degrees(math.acos(max(-1, min(1, s.params["module"]["axis"][2])))) <= 120 + 1e-6
                    for s in phal)
        T = forward_kinematics(m, {})
        bases = np.array([T[f"d{s.params['digit_id']}p1"][:3, 3] for s in top])
        if len(top) >= 2:
            n_multi += 1
            fwd = [T[f"d{s.params['digit_id']}p1"][:3, :3] @ np.array([0, 0, 1.0]) for s in top]
            others = np.mean(fwd[:-1], axis=0)
            oppose += float(np.dot(fwd[-1], others / (np.linalg.norm(others) + 1e-12))) < 0.0
            dd = np.linalg.norm(bases[:, None] - bases[None], axis=2) + np.eye(len(bases))
            min_sep.append(float(dd.min()))
    return {"surface": surface / n, "curl": curl / n, "band": band / n,
            "oppose": oppose / max(1, n_multi), "min_sep_mm": 1000 * float(np.median(min_sep))}


@pytest.fixture(scope="module")
def stats():
    return {
        "grammar": _rule_stats(build_distribution()),
        "v3s": _rule_stats(G_V3S),
        "off": _rule_stats(build_distribution(False, False, False)),
        "no_spacing": _rule_stats(build_distribution(True, False, True)),
    }


def test_rules_and_defaults():
    assert RULES == ("surface", "spacing", "curl_opposition")
    assert build_distribution() == build_distribution(True, True, True)
    assert "G_V3S" in NAMED_DISTRIBUTIONS and NAMED_DISTRIBUTIONS["G_V3S"] is G_V3S
    assert build_distribution() not in NAMED_DISTRIBUTIONS.values()


def test_same_rule_properties_as_v3s(stats):
    g, v = stats["grammar"], stats["v3s"]
    assert g["surface"] == v["surface"] == 1.0
    assert g["curl"] == v["curl"] == 1.0
    assert g["band"] == v["band"] == 1.0
    assert g["oppose"] > 0.9 and v["oppose"] > 0.9
    # spacing on top of V3s: finger bases further apart than without it
    assert g["min_sep_mm"] > stats["no_spacing"]["min_sep_mm"] + 5.0


def test_rules_off_remove_their_properties(stats):
    off = stats["off"]
    assert off["surface"] == 0.0 and off["band"] < 0.5 and off["oppose"] < 0.8
    # no rule: no rest bend at all (curl off), straight fingers
    for seed in range(50):
        d = sample_derivation(seed, build_distribution(False, False, False), limits=SIMULATOR)
        assert all(tuple(s.params["bend_rpy"]) == (0.0, 0.0, 0.0) for s in d.steps if s.production == "Phalanx")
        assert all(s.params.get("mount_offset", (0.0, 0.0)) == (0.0, 0.0) for s in d.steps
                   if s.production in ("Digit", "PalmBody"))


def test_not_byte_identical_to_v3s():
    same = sum(derivation_to_json(sample_derivation(s, build_distribution(spacing=False), limits=SIMULATOR))
               == derivation_to_json(sample_derivation(s, G_V3S, limits=SIMULATOR)) for s in range(30))
    assert same < 30


@pytest.mark.parametrize("rules", [(True, True, True), (False, False, False), (True, False, True)])
def test_support_grids_do_not_change_sampling(rules):
    """GRAMMAR_BASE's support-only fields never touch a random draw."""
    full = build_distribution(*rules)
    plain = replace(full, mount_lateral_grid_m=None, mount_lateral_sampled=True, bend_support_rpy_choices_rad=(),
                    limits_support_continuous=False, revolute_limit_range_deg=DEFAULT_DISTRIBUTION.revolute_limit_range_deg)
    for seed in range(60):
        for limits in (None, SIMULATOR):
            assert derivation_to_json(sample_derivation(seed, full, limits=limits)) == \
                derivation_to_json(sample_derivation(seed, plain, limits=limits))


def test_samples_are_in_support():
    d = build_distribution()
    for seed in range(30):
        cov = coverage(derive(sample_derivation(seed, d, limits=SIMULATOR)), d)
        assert cov.in_support, cov.out_of_support


def _with(d, production, fn):
    steps = []
    done = False
    for s in d.steps:
        if not done and s.production == production:
            s, done = DerivationStep(path=s.path, production=s.production, params=fn(dict(s.params))), True
        steps.append(s)
    return replace(d, steps=tuple(steps))


def test_support_grids_are_reachable():
    dist = build_distribution()
    rng = np.random.default_rng(0)
    d = sample_derivation(3, dist, limits=SIMULATOR)
    # a support-only bend (roll 30 deg) on a bone after the first is stepped along the full grid
    bent = _with(d, "Phalanx", lambda p: {**p, "bend_rpy": (30 * DEG, 0.0, 0.0)} if p["p"] else p)
    seen = set()
    for _ in range(60):
        child = apply_operator(bent, rng, dist, "step_bend_rpy")
        if child is not None:
            seen.update(tuple(round(math.degrees(v)) for v in s.params["bend_rpy"])
                        for s in child.steps if s.production == "Phalanx")
    assert any(t[0] in (15, 45) for t in seen)
    # an off-menu range (a real hand's) moves one bound by 15 deg inside +/-180
    odd = _with(d, "Phalanx", lambda p: {**p, "module": {**p["module"], "limits": (-0.3, 1.9)}})
    moved = 0
    for _ in range(400):
        child = apply_operator(odd, rng, dist, "step_limits")
        if child is None:
            continue
        lims = [s.params["module"]["limits"] for s in child.steps if s.production == "Phalanx"]
        for lo, hi in lims:
            if (lo, hi) != (-0.3, 1.9) and (abs(lo + 0.3) < 1e-9 or abs(hi - 1.9) < 1e-9):
                assert abs(abs((lo + 0.3) + (hi - 1.9)) - 15 * DEG) < 1e-9
                moved += 1
    assert moved > 0
    # a lateral offset on the grid is stepped by one grid step
    grid = lateral_offset_choices_m(dist)
    lat = _with(d, "Digit", lambda p: {**p, "mount_offset": (grid[20], grid[13])})
    stepped = 0
    for _ in range(400):
        child = apply_operator(lat, rng, dist, "step_mount")
        if child is None:
            continue
        a = next(s for s in lat.steps if s.production == "Digit").params["mount_offset"]
        b = next(s for s in child.steps if s.production == "Digit").params["mount_offset"]
        if a != b:
            assert abs(max(abs(np.subtract(a, b))) - 0.005) < 1e-12
            stepped += 1
    assert stepped > 0
