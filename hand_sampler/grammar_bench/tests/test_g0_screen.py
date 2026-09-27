"""G0 CPU grammar screen (plan revision, 2026-09-27, step 2): acceptance
tests for the three new generative rules the screen's V1-V3 variants use,
plus a determinism/no-regression check.

  1. Mount-spacing rule (V2, I29): ``Distribution.mount_min_separation_m``
     spreads top-level digit mounts across hosts/fracs instead of drawing
     each digit's (host, frac) i.i.d.; default (``None``) leaves
     ``sample_derivation`` byte-identical to before this field existed.
  2. Curl axis prior (V3, I30): ``Distribution.digit_axis_elevation_band_deg``
     restricts a digit phalanx's revolute axis to a band around the
     horizontal (hinge-like) plane; default (``None``) is unchanged.
  3. Opposition prior (V3, I30): ``Distribution.opposition_prior`` makes the
     LAST top-level digit's mount orientation oppose the mean forward
     direction of the earlier digits, when there are >= 2 digits; default
     (``False``) is unchanged.
  4. No-regression: with every new field at its default, ``sample_derivation``
     output is byte-identical (same ``phenotype_hash``) to a pre-recorded
     baseline for a handful of seeds under ``G_FULL``/``G_SERIAL`` -- the
     existing grammar suite (3124 passed, 1 skipped) is the primary
     regression guard; this test is a narrow, fast pin on top of it.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.canonical import phenotype_hash
from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION, sample_axis
from hand_sampler.grammar.fk import rpy_to_matrix
from hand_sampler.grammar.variants import G_FULL, G_SERIAL

# ---------------------------------------------------------------------------
# 4. No-regression pin (run first: cheapest signal that nothing else broke).
# ---------------------------------------------------------------------------

# Recorded once, pre-change, from the working tree at commit a50514a (the
# tip of `martin/hand-grammar` before this screen's grammar-rule commit).
_BASELINE_HASHES = {
    ("G_FULL", 0): "2eda8925369864f24c5ad92e8cf743e82ad3fae2c7ccadb054443debd01ca1e0",
    ("G_FULL", 1): "30df2435a1772a7d743867bfb478c9e815857e9b56039d625c1f5d89416c22e7",
    ("G_SERIAL", 0): "d33d7032180b5a6ebe4d3a5094eb86dec3ce2c7c1de4188b56faf6c6a50d0c37",
    ("G_SERIAL", 1): "8bfec8f9d0976c74003751c8ea62261268eeab769307c929b67e01e204349baa",
}


@pytest.mark.parametrize("name,dist,seed", [
    ("G_FULL", G_FULL, 0), ("G_FULL", G_FULL, 1),
    ("G_SERIAL", G_SERIAL, 0), ("G_SERIAL", G_SERIAL, 1),
])
def test_default_sampling_unchanged(name, dist, seed):
    # New fields at their defaults must not perturb the RNG stream or the
    # derived geometry at all -- sanity: the three new fields all read as
    # "off".
    assert dist.mount_min_separation_m is None
    assert dist.digit_axis_elevation_band_deg is None
    assert dist.opposition_prior is False
    d = sample_derivation(seed, dist)
    m = derive(d)
    h = phenotype_hash(m)
    assert h == _BASELINE_HASHES[(name, seed)]


# ---------------------------------------------------------------------------
# 1. Mount-spacing rule (V2).
# ---------------------------------------------------------------------------


def _top_level_mount_points(model, derivation):
    """(host, mount world-position-at-q0) for every top-level digit, read
    straight off the derived model (root-mounted digits' joint origin, at
    q=0, in the ROOT frame -- exact since root/palm chains have no rotation
    at q=0 along the way for these variants: revolute-only, all limits
    bracket 0 is not required, but q=0 is always a legal FK input)."""
    from hand_sampler.grammar.fk import forward_kinematics
    transforms = forward_kinematics(model, {})
    out = []
    top_ids = {s.params["digit_id"] for s in derivation.steps
               if s.production == "Digit" and s.params.get("top_level")}
    for s in derivation.steps:
        if s.production == "Digit" and s.params["digit_id"] in top_ids:
            body_name = f"d{s.params['digit_id']}p1"
            T = transforms.get(body_name)
            if T is not None:
                out.append((s.params["mount"], np.array(T[:3, 3])))
    return out


def test_mount_spacing_increases_separation():
    base = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(4, 5), palm_body_count_range=(0, 0),
        branch_probability=0.0,
    )
    spaced = replace(base, mount_min_separation_m=0.03)

    def min_pairwise_sep(dist, n_seeds=60):
        seps = []
        for seed in range(n_seeds):
            d = sample_derivation(seed, dist)
            m = derive(d)
            pts = _top_level_mount_points(m, d)
            if len(pts) < 2:
                continue
            same_host = {}
            for host, p in pts:
                same_host.setdefault(host, []).append(p)
            for host, plist in same_host.items():
                for i in range(len(plist)):
                    for j in range(i + 1, len(plist)):
                        seps.append(float(np.linalg.norm(plist[i] - plist[j])))
        return seps

    seps_base = min_pairwise_sep(base)
    seps_spaced = min_pairwise_sep(spaced)
    assert seps_base, "expected some same-host digit pairs in the unspaced baseline"
    assert seps_spaced, "expected some same-host digit pairs under spacing too"
    # The spacing rule must materially reduce (ideally eliminate) exact or
    # near-exact mount collisions (two digits landing on the same host at
    # the same frac -> 0 mm separation) relative to i.i.d. sampling.
    n_collisions_base = sum(1 for s in seps_base if s < 1e-6)
    n_collisions_spaced = sum(1 for s in seps_spaced if s < 1e-6)
    assert n_collisions_base > 0, "baseline i.i.d. sampling should produce some coincident mounts at this digit count"
    assert n_collisions_spaced == 0, "the spacing rule must never place two same-host digits at the identical mount"
    assert float(np.mean(seps_spaced)) > float(np.mean(seps_base))


def test_mount_spacing_deterministic_and_derivable():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(3, 5), palm_body_count_range=(0, 2),
        branch_probability=0.0, mount_min_separation_m=0.025,
    )
    for seed in range(30):
        d1 = sample_derivation(seed, dist)
        d2 = sample_derivation(seed, dist)
        m1 = derive(d1)
        m2 = derive(d2)
        assert phenotype_hash(m1) == phenotype_hash(m2)


# ---------------------------------------------------------------------------
# 2. Curl axis prior (V3).
# ---------------------------------------------------------------------------


def test_axis_band_restricts_elevation():
    lo, hi = 60.0, 120.0
    rng = np.random.default_rng(0)
    for _ in range(500):
        axis = sample_axis(rng, elevation_band_deg=(lo, hi))
        z = axis[2]
        # elevation = angle from +z; band 60-120 deg means |cos(elevation)|
        # is bounded away from 1 -- z = cos(elevation) in [cos(120), cos(60)].
        assert -0.500001 <= z <= 0.500001


def test_axis_band_none_matches_unbanded_stream():
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    for _ in range(50):
        a = sample_axis(rng_a)
        b = sample_axis(rng_b, elevation_band_deg=None)
        assert a == b


def test_digit_axis_elevation_band_applied_to_sampled_hands():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(2, 4), palm_body_count_range=(0, 0),
        branch_probability=0.0,
        module_probabilities=(("R", 1.0), ("C", 0.0), ("P", 0.0), ("Coupled", 0.0)),
        digit_axis_elevation_band_deg=(60.0, 120.0),
    )
    max_abs_z = 0.0
    n_joints = 0
    for seed in range(40):
        d = sample_derivation(seed, dist)
        m = derive(d)
        for j in m.joints:
            if j.type == "revolute" and not any(b.name == j.parent and b.palm for b in m.bodies):
                n_joints += 1
                max_abs_z = max(max_abs_z, abs(j.axis[2]))
    assert n_joints > 0
    assert max_abs_z <= 0.500001


# ---------------------------------------------------------------------------
# 3. Opposition prior (V3).
# ---------------------------------------------------------------------------


def test_opposition_prior_opposes_last_digit():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(3, 3), palm_body_count_range=(0, 0),
        branch_probability=0.0, opposition_prior=True,
    )
    n_checked = 0
    for seed in range(60):
        d = sample_derivation(seed, dist)
        top = [s for s in d.steps if s.production == "Digit" and s.params.get("top_level")]
        assert len(top) == 3
        fwds = [rpy_to_matrix(tuple(s.params["mount_rpy"])) @ np.array([0.0, 0.0, 1.0]) for s in top]
        mean_others = np.mean(fwds[:-1], axis=0)
        if np.linalg.norm(mean_others) < 1e-9:
            continue
        mean_others = mean_others / np.linalg.norm(mean_others)
        dot = float(np.dot(fwds[-1], mean_others))
        assert dot < 0.0, f"seed {seed}: last digit does not oppose the others (dot={dot})"
        n_checked += 1
    assert n_checked >= 40


def test_opposition_prior_off_by_default():
    assert DEFAULT_DISTRIBUTION.opposition_prior is False
    d = sample_derivation(0, DEFAULT_DISTRIBUTION)
    m = derive(d)
    assert m is not None
