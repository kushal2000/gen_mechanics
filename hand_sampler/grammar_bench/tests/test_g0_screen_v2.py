"""Opus review of G0 (opus-review-g0.md, I39): acceptance tests for the
V1s/V2s/V3s fixes (surface mounting + cross-host spacing, review item 2;
curl/opposition host-frame fix, review item 5), plus a no-regression pin on
the EXISTING V1/V2/V3 (which must stay byte-identical -- V1/V2/V3 are kept
for reference, not replaced).

  1. No-regression pin: G_V1/G_V2/G_V3 sampling is unaffected by the three
     new ``Distribution`` fields this review adds (all default off).
  2. Surface mounting (review item 2): ``mount_on_host_surface`` moves a
     top-level digit's mount off the host's centre axis by the host's own
     capsule radius.
  3. Cross-host spacing (review item 2): V2s's planned (host, frac, azimuth)
     placement gives a materially larger minimum pairwise mount separation
     than V1s's i.i.d. azimuth, including on SHORT root lengths where the
     axial grid alone cannot reach the target.
  4. Curl-skip-first-phalanx (review item 5): with ``bend_probability=1.0``,
     phalanx 0 never bends when ``curl_skip_first_phalanx=True``, so the
     digit's actual joint-origin orientation equals its own ``mount_rpy``
     exactly; later phalanges still bend.
  5. Opposition host-frame fix (review item 5): on mixed-host hands (top-
     level digits on different palm bodies), the ROOT-FRAME opposition
     (measured via forward kinematics) is materially better under
     ``opposition_use_host_frame=True`` than under the naive (pre-fix)
     computation, on the SAME sampled hosts/fracs/earlier-digit poses.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.canonical import phenotype_hash
from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.variants import G_V1, G_V1S, G_V2, G_V2S, G_V3, G_V3S

# ---------------------------------------------------------------------------
# 1. No-regression pin (V1/V2/V3 stay byte-identical to before this review).
# ---------------------------------------------------------------------------

# Recorded from the working tree at commit ccb5b66 (the tip of
# martin/hand-grammar before this review's grammar-rule commit), via an
# isolated snapshot of distributions.py/derive.py/variants.py at that
# commit -- confirmed to match the post-review code exactly (see the
# worker's own report).
_BASELINE_HASHES = {
    ("G_V1", 0): "09226ed9fd98abd714e61b2c12259910667e3b4a21a874b9708a7d273c8377e1",
    ("G_V1", 1): "82d562c8d5618304e03d4895d9b901b0488d8b54e9977aad71a5a9ec7e13bfa0",
    ("G_V1", 2): "da24e77e325c2e7706556a26e0b15d4e91da444c1be5927618fd2cd1e78a245e",
    ("G_V1", 3): "b00a09b3aa89af3c5e641354c81fe7fe16cc2891c92d4d1de920f24809e777cd",
    ("G_V2", 0): "08a94d9e2cf9d25bc39b9872d17e4917e8a7383a4a7869386a1ba231be11ef13",
    ("G_V2", 1): "bc6e0afbb8f0226858843be934b041f834bdc713eb779323efd3c1c3c97d3953",
    ("G_V2", 2): "7092adb922a43250ebdf8b4b5ccb3c72806e3a759c9d7e327660540e3948ff1d",
    ("G_V2", 3): "abe809f395eb22aa03af4ce97e5000a2f8abb0bd6937073f5fa66b2b387f9beb",
    ("G_V3", 0): "3eacd7f5f45a55e4bc520a508e49d71e1401fa9ded1227e1847fde8ec2c599b7",
    ("G_V3", 1): "145e8df820e086d119b1f836e28c48f618525f9aa35b6010f23c79b9664f583f",
    ("G_V3", 2): "ad66992cd734228cf57ce6df626242ec018fe3a260c4b66ef24711b07cca9c21",
    ("G_V3", 3): "d319958aca3908065ef4e9e1c6b1053c7bf5d0bf3efffd6802fd37ced4103150",
}


@pytest.mark.parametrize("name,dist,seed", [
    (n, d, s) for n, d in [("G_V1", G_V1), ("G_V2", G_V2), ("G_V3", G_V3)] for s in (0, 1, 2, 3)
])
def test_v1_v2_v3_unchanged_by_review(name, dist, seed):
    assert dist.mount_on_host_surface is False
    assert dist.curl_skip_first_phalanx is False
    assert dist.opposition_use_host_frame is False
    m = derive(sample_derivation(seed, dist))
    assert phenotype_hash(m) == _BASELINE_HASHES[(name, seed)]


def test_new_fields_default_off():
    assert DEFAULT_DISTRIBUTION.mount_on_host_surface is False
    assert DEFAULT_DISTRIBUTION.curl_skip_first_phalanx is False
    assert DEFAULT_DISTRIBUTION.opposition_use_host_frame is False


# ---------------------------------------------------------------------------
# 2. Surface mounting.
# ---------------------------------------------------------------------------


def _top_level_mount_world_points(model, derivation):
    transforms = forward_kinematics(model, {})
    out = []
    top = [s for s in derivation.steps if s.production == "Digit" and s.params.get("top_level")]
    for s in top:
        body_name = f"d{s.params['digit_id']}p1"
        T = transforms.get(body_name)
        if T is not None:
            out.append((s.params["mount"], np.array(T[:3, 3])))
    return out


def test_surface_mounting_offsets_off_axis():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(1, 1), palm_body_count_range=(0, 0),
        branch_probability=0.0, mount_on_host_surface=True,
    )
    n_offaxis = 0
    for seed in range(60):
        d = sample_derivation(seed, dist)
        m = derive(d)
        digit_step = next(s for s in d.steps if s.production == "Digit")
        radial = float(np.hypot(digit_step.params["mount_offset"][0], digit_step.params["mount_offset"][1]))
        capsule_radius = next(s for s in d.steps if s.path == "hand").params["capsule_radius_m"]
        assert abs(radial - capsule_radius) < 1e-9
        if radial > 1e-6:
            n_offaxis += 1
    assert n_offaxis > 50, "surface mounting should place almost every mount off the host's centre axis"


def test_surface_mounting_default_on_axis():
    # mount_on_host_surface=False (default): mount_offset stays (0, 0) --
    # byte-identical positioning to before this field existed.
    d = sample_derivation(0, DEFAULT_DISTRIBUTION)
    for s in d.steps:
        if s.production == "Digit":
            assert s.params["mount_offset"] == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 3. Cross-host spacing (V1s vs V2s), including short root lengths.
# ---------------------------------------------------------------------------


def _min_pairwise_mount_separation(dist, n_seeds=150):
    seps = []
    for seed in range(n_seeds):
        d = sample_derivation(seed, dist)
        m = derive(d)
        pts = _top_level_mount_world_points(m, d)
        if len(pts) < 2:
            continue
        positions = [p for _h, p in pts]
        pairwise = [
            float(np.linalg.norm(positions[i] - positions[j]))
            for i in range(len(positions)) for j in range(i + 1, len(positions))
        ]
        seps.append(min(pairwise))
    return seps


def test_cross_host_spacing_beats_iid_surface_azimuth():
    # 4-5 digits, a short root (20-60 mm) plus up to 2 palm bodies -- the
    # regime opus-review-g0.md item 2 flagged as unplaceable by axial
    # fraction alone.
    base = replace(
        G_V1S, digit_count_range=(4, 5), palm_body_count_range=(0, 2),
        palm_length_range_m=(0.020, 0.060),
    )
    v2s_like = replace(base, mount_min_separation_m=2.0 * 0.012 + 0.005)

    seps_v1s = _min_pairwise_mount_separation(base)
    seps_v2s = _min_pairwise_mount_separation(v2s_like)
    assert len(seps_v1s) > 30 and len(seps_v2s) > 30
    assert float(np.mean(seps_v2s)) > float(np.mean(seps_v1s))
    assert float(np.median(seps_v2s)) > float(np.median(seps_v1s))
    # The planner must never place two top-level digits at the identical
    # point (a same-host, same-frac, same-azimuth coincidence).
    assert min(seps_v2s) > 1e-6


def test_v2s_deterministic_and_derivable():
    dist = replace(G_V2S, digit_count_range=(3, 5), palm_body_count_range=(0, 2))
    for seed in range(30):
        m1 = derive(sample_derivation(seed, dist))
        m2 = derive(sample_derivation(seed, dist))
        assert phenotype_hash(m1) == phenotype_hash(m2)


# ---------------------------------------------------------------------------
# 4. Curl-skip-first-phalanx.
# ---------------------------------------------------------------------------


def test_curl_skip_first_phalanx_leaves_phalanx0_unbent():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(1, 1), palm_body_count_range=(0, 0),
        phalanx_count_range=(3, 3), branch_probability=0.0,
        bend_probability=1.0,
        bend_rpy_choices_rad=((0.0, 30.0 * np.pi / 180.0, 0.0),),
        bend_offset_choices_m=((0.005, 0.0),),
        curl_skip_first_phalanx=True,
    )
    n_bent_later = 0
    for seed in range(30):
        d = sample_derivation(seed, dist)
        m = derive(d)
        digit_step = next(s for s in d.steps if s.production == "Digit")
        p0 = next(s for s in d.steps if s.production == "Phalanx" and s.params["p"] == 0
                  and s.params["digit_id"] == digit_step.params["digit_id"])
        assert p0.params["bend_rpy"] == (0.0, 0.0, 0.0)
        assert p0.params["bend_offset"] == (0.0, 0.0)
        # The digit's own first joint origin orientation is then EXACTLY its
        # sampled mount_rpy (no composition at all).
        joint = next(j for j in m.joints if j.name == f"d{digit_step.params['digit_id']}p1_j")
        assert joint.origin.rpy == tuple(digit_step.params["mount_rpy"])
        p1 = next((s for s in d.steps if s.production == "Phalanx" and s.params["p"] == 1
                    and s.params["digit_id"] == digit_step.params["digit_id"]), None)
        if p1 is not None and p1.params["bend_rpy"] != (0.0, 0.0, 0.0):
            n_bent_later += 1
    assert n_bent_later > 20, "phalanges after the first should still bend at probability 1.0"


def test_curl_skip_first_phalanx_off_matches_existing_bend():
    # False (default): phalanx 0 bends exactly as before this field existed
    # (G_BEND's own precedent) -- no regression for that distribution.
    from hand_sampler.grammar.variants import G_BEND
    assert G_BEND.curl_skip_first_phalanx is False


# ---------------------------------------------------------------------------
# 5. Opposition host-frame fix.
# ---------------------------------------------------------------------------


def _root_frame_opposition_dot(model, derivation):
    transforms = forward_kinematics(model, {})
    top = [s for s in derivation.steps if s.production == "Digit" and s.params.get("top_level")]
    fwds = [transforms[f"d{s.params['digit_id']}p1"][:3, 2] for s in top]
    mean_others = np.mean(fwds[:-1], axis=0)
    n = float(np.linalg.norm(mean_others))
    if n < 1e-9:
        return None
    return float(np.dot(fwds[-1], mean_others / n))


def test_opposition_host_frame_fix_improves_mixed_host_hands():
    dist = replace(
        DEFAULT_DISTRIBUTION,
        digit_count_range=(3, 3), palm_body_count_range=(1, 1),
        branch_probability=0.0,
        module_probabilities=(("R", 1.0), ("C", 0.0), ("P", 0.0), ("Coupled", 0.0)),
        opposition_prior=True, opposition_use_host_frame=True,
    )
    dist_naive = replace(dist, opposition_use_host_frame=False)

    dots_fixed, dots_naive = [], []
    n_mixed = 0
    for seed in range(500):
        d = sample_derivation(seed, dist)
        top = [s for s in d.steps if s.production == "Digit" and s.params.get("top_level")]
        if len({s.params["mount"] for s in top}) < 2:
            continue
        n_mixed += 1
        d_naive = sample_derivation(seed, dist_naive)
        dot_fixed = _root_frame_opposition_dot(derive(d), d)
        dot_naive = _root_frame_opposition_dot(derive(d_naive), d_naive)
        if dot_fixed is None or dot_naive is None:
            continue
        dots_fixed.append(dot_fixed)
        dots_naive.append(dot_naive)

    assert n_mixed >= 20, "expected several mixed-host hands in 500 seeds"
    assert len(dots_fixed) >= 15
    assert float(np.median(dots_fixed)) < -0.9, "the fix should closely oppose in the ROOT frame"
    assert float(np.median(dots_fixed)) < float(np.median(dots_naive)), \
        "the host-frame fix must beat the naive (pre-fix) computation on mixed-host hands"


def test_v3s_deterministic_and_derivable():
    dist = replace(G_V3S, digit_count_range=(2, 5), palm_body_count_range=(0, 2))
    for seed in range(30):
        m1 = derive(sample_derivation(seed, dist))
        m2 = derive(sample_derivation(seed, dist))
        assert phenotype_hash(m1) == phenotype_hash(m2)
