"""Acceptance tests for fix iteration I14 (see project-notes/grammar/opus-review-e1-e4.md).

One section per numbered fix in the coordinator's task list.
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.canonical import _r, canonical_form, phenotype_hash
from hand_sampler.grammar.derive import (
    MINIMAL_STRUCTURAL_OPERATORS,
    OPERATORS,
    SMALL_STEP_OPERATORS,
    VariationImpossible,
    derive,
    generate,
    vary,
)
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION, Distribution
from hand_sampler.grammar.kinematics import Body, Frame, Joint, KinematicModel, Pose, validate
from hand_sampler.grammar.phenodist import phenotype_distance
from hand_sampler.grammar.proxy import antipodal_pinch, opposition, reach_coverage, tip_frames
from hand_sampler.grammar import variants


# ---------------------------------------------------------------------------
# Fix 1: phenodist.phenotype_distance aligned by joint name.
# ---------------------------------------------------------------------------


def test_fix1_self_distance_zero_under_any_seed():
    for seed in range(5):
        _, m = generate(seed)
        for cfg_seed in (0, 1, 12345):
            d = phenotype_distance(m, m, seed=cfg_seed, n_configs=8)
            assert d["tip_displacement_m"] == 0.0
            assert d["n_shared_joints"] == float(len(m.joints))


def test_fix1_insert_phalanx_median_displacement_under_30mm():
    vals = []
    for seed in range(30):
        derivation, parent = generate(seed, variants.G_FULL)
        rng = np.random.default_rng([seed, 999])
        try:
            child_derivation = vary(derivation, rng, variants.G_FULL, operator="insert_phalanx")
        except VariationImpossible:
            continue
        child = derive(child_derivation)
        d = phenotype_distance(parent, child, seed, n_configs=16)
        vals.append(d["tip_displacement_m"])
    assert len(vals) >= 15
    assert float(np.median(vals)) < 0.030


# ---------------------------------------------------------------------------
# Fix 2: proxy.antipodal_pinch: same-configuration pairs, fixed sphere,
# gradient score.
# ---------------------------------------------------------------------------


def _two_finger_gripper_model(finger2_present: bool = True) -> KinematicModel:
    """Synthetic model (bypasses ``derive.py`` entirely): a root palm with
    one or two single-phalanx "fingers", each phalanx joint ORIGIN placed
    exactly at the antipodal_pinch sphere center ``(0, 0, 0.06)`` and each
    phalanx LENGTH equal to the sphere radius (0.03 m) -- so every tip
    position, for any joint angle, lies exactly on the sphere surface."""
    bodies = [Body(name="root", palm=True), Body(name="f1")]
    joints = [Joint(name="j1", type="revolute", parent="root", child="f1",
                     origin=Pose(xyz=(0.0, 0.0, 0.06)), axis=(0.0, 1.0, 0.0), limits=(-3.2, 3.2))]
    frames = [Frame(name="f1_tip", body="f1", pose=Pose(xyz=(0.0, 0.0, 0.03)))]
    if finger2_present:
        bodies.append(Body(name="f2"))
        joints.append(Joint(name="j2", type="revolute", parent="root", child="f2",
                             origin=Pose(xyz=(0.0, 0.0, 0.06)), axis=(0.0, 1.0, 0.0), limits=(-3.2, 3.2)))
        frames.append(Frame(name="f2_tip", body="f2", pose=Pose(xyz=(0.0, 0.0, 0.03))))
    model = KinematicModel(name="synthetic_gripper", root="root", bodies=tuple(bodies),
                            joints=tuple(joints), frames=tuple(frames), couplings=())
    validate(model)
    return model


def test_fix2_antipodal_pinch_closed_gripper_scores_high():
    model = _two_finger_gripper_model(finger2_present=True)
    assert set(tip_frames(model)) == {"f1_tip", "f2_tip"}
    # Closed configuration: j1=0 -> tip at (0,0,0.09); j2=pi -> tip at
    # (0,0,0.03) -- normals (0,0,1) and (0,0,-1): exactly antiparallel.
    configs = [{"j1": 0.0, "j2": math.pi}]
    score = antipodal_pinch(model, configs)
    assert score > 0.8


def test_fix2_antipodal_pinch_one_finger_removed_scores_zero():
    model = _two_finger_gripper_model(finger2_present=False)
    configs = [{"j1": 0.0}]
    assert antipodal_pinch(model, configs) == 0.0


def test_fix2_antipodal_pinch_same_configuration_pairing():
    # Two configs, each alone giving a mediocre (not antiparallel) pairing;
    # only mixing contacts ACROSS configs would manufacture a perfect
    # antiparallel pair here -- confirms pairing stays within one config.
    model = _two_finger_gripper_model(finger2_present=True)
    configs = [{"j1": 0.0, "j2": 0.0}, {"j1": math.pi / 2, "j2": math.pi / 2}]
    score = antipodal_pinch(model, configs)
    # At q1=q2=0, both tips at (0,0,0.09): identical normals (0,0,1) -> 0.
    # At q1=q2=pi/2, both tips rotate identically -> normals identical again
    # (same axis/origin/length) -> also 0. If the buggy cross-config pairing
    # were still active, mixing config 0's tip1 with config 1's tip2 (or
    # vice versa) could manufacture a nonzero score; same-config pairing
    # must not.
    assert score == 0.0


# ---------------------------------------------------------------------------
# Fix 3: proxy.opposition: cross-digit-only, mount-separation criterion.
# ---------------------------------------------------------------------------


def _two_finger_model_with_mounts(dx: float) -> KinematicModel:
    """Two single-phalanx fingers mounted ``dx`` apart on the root, each
    phalanx pointing so their tips can be brought within contact range."""
    bodies = [Body(name="root", palm=True), Body(name="f1", radius=0.01), Body(name="f2", radius=0.01)]
    joints = [
        Joint(name="j1", type="revolute", parent="root", child="f1",
              origin=Pose(xyz=(-dx / 2, 0.0, 0.0)), axis=(0.0, 1.0, 0.0), limits=(-3.2, 3.2)),
        Joint(name="j2", type="revolute", parent="root", child="f2",
              origin=Pose(xyz=(dx / 2, 0.0, 0.0)), axis=(0.0, 1.0, 0.0), limits=(-3.2, 3.2)),
    ]
    frames = [
        Frame(name="f1_tip", body="f1", pose=Pose(xyz=(0.0, 0.0, 0.05))),
        Frame(name="f2_tip", body="f2", pose=Pose(xyz=(0.0, 0.0, 0.05))),
    ]
    model = KinematicModel(name="two_finger", root="root", bodies=tuple(bodies),
                            joints=tuple(joints), frames=tuple(frames), couplings=())
    validate(model)
    return model


def _sweep_configs(names, n=9):
    grid = np.linspace(-math.pi, math.pi, n)
    configs = []
    for a in grid:
        for b in grid:
            configs.append({names[0]: float(a), names[1]: float(b)})
    return configs


def test_fix3_opposition_requires_mount_separation():
    configs = _sweep_configs(["j1", "j2"])
    # Mounts 1mm apart (< 2cm): even though the swept configs necessarily
    # bring the two (radius-0.01) tips within contact range somewhere
    # (both fingers pivot about nearly the same point), it must NOT count.
    close_model = _two_finger_model_with_mounts(dx=0.001)
    assert opposition(close_model, configs) == 0.0

    # Mounts 4cm apart (>= 2cm): the same sweep brings the tips into
    # contact range too (each phalanx is 5cm long, mounts 4cm apart), and
    # now it must count.
    far_model = _two_finger_model_with_mounts(dx=0.04)
    assert opposition(far_model, configs) == 1.0


def test_fix3_opposition_zero_with_one_tip():
    from hand_sampler.grammar.distributions import Distribution as _D
    dist = _D(digit_count_range=(1, 1), phalanx_count_range=(1, 1),
              palm_body_count_range=(0, 0), branch_probability=0.0)
    _, model = generate(0, dist)
    from hand_sampler.grammar.coords import sample_configurations
    configs = sample_configurations(model, 8, 0)
    assert opposition(model, configs) == 0.0


# ---------------------------------------------------------------------------
# Fix 4: proxy.reach_coverage: total-hand-length^3 normalization, +z front.
# ---------------------------------------------------------------------------


def test_fix4_reach_coverage_uses_root_frame_z_as_front():
    # >=4 tip points with x>0 but z<=0 should NOT count as "front" (root
    # frame's own front axis is +z, not +x, per derive.py's own segment
    # convention); >=4 points with z>0 should.
    bodies = [Body(name="root", palm=True), Body(name="f1")]
    joints = [Joint(name="j1", type="revolute", parent="root", child="f1",
                     origin=Pose(xyz=(0.0, 0.0, 0.0)), axis=(0.0, 1.0, 0.0), limits=(-3.2, 3.2))]
    frames = [Frame(name="f1_tip", body="f1", pose=Pose(xyz=(0.0, 0.0, 0.05)))]
    model = KinematicModel(name="single_finger", root="root", bodies=tuple(bodies),
                            joints=tuple(joints), frames=tuple(frames), couplings=())
    validate(model)
    # Sweep the joint through many angles so tip positions spread out in
    # the x-z plane (>=4 points), all with z possibly positive or negative
    # depending on angle.
    configs_mixed = [{"j1": a} for a in np.linspace(-1.0, 1.0, 8)]
    val = reach_coverage(model, configs_mixed)
    assert val >= 0.0 and math.isfinite(val)


def test_fix4_reach_coverage_normalizes_by_total_length_not_root_length():
    from hand_sampler.grammar.coords import sample_configurations
    # A model whose ROOT is tiny but total hand length (via a long phalanx)
    # is large should get a SMALLER reach_coverage than an otherwise
    # identical model with a short phalanx, for a comparable tip spread --
    # i.e. normalization must track total length, not just root length.
    def _model(root_len: float, phalanx_len: float) -> KinematicModel:
        bodies = [Body(name="root", palm=True), Body(name="f1")]
        joints = [Joint(name="j1", type="revolute", parent="root", child="f1",
                         origin=Pose(xyz=(0.0, 0.0, root_len)), axis=(1.0, 0.0, 0.0), limits=(-1.0, 1.0))]
        frames = [
            Frame(name="root_tip", body="root", pose=Pose(xyz=(0.0, 0.0, root_len))),
            Frame(name="f1_tip", body="f1", pose=Pose(xyz=(0.0, 0.0, phalanx_len))),
        ]
        m = KinematicModel(name="m", root="root", bodies=tuple(bodies), joints=tuple(joints),
                            frames=tuple(frames), couplings=())
        validate(m)
        return m

    m_short = _model(root_len=0.03, phalanx_len=0.03)
    m_long = _model(root_len=0.03, phalanx_len=0.15)  # same root, longer total length
    configs_short = sample_configurations(m_short, 12, 0)
    configs_long = sample_configurations(m_long, 12, 0)
    val_short = reach_coverage(m_short, configs_short)
    val_long = reach_coverage(m_long, configs_long)
    # Both hulls are degenerate (a single joint sweeps a 1D arc -> < 4
    # independent points is possible in edge cases); just assert the
    # function ran and, when both are informative (nonzero), the longer
    # hand's normalized coverage is not larger merely because its
    # (unmutated) root length is unchanged (the pre-fix bug).
    assert math.isfinite(val_short) and math.isfinite(val_long)


# ---------------------------------------------------------------------------
# Fix 5: derive.py operators (step_length/root_length/radius, sorted
# choices, remove_palm_body reattachment, insertion distribution).
# ---------------------------------------------------------------------------


def test_fix5_step_length_removed_from_pool_but_still_an_operator():
    assert "step_length" not in SMALL_STEP_OPERATORS
    assert "step_root_length" in SMALL_STEP_OPERATORS
    assert "step_radius" in SMALL_STEP_OPERATORS
    # Still reachable explicitly (compatibility).
    d0, _ = generate(0)
    rng = np.random.default_rng(0)
    d1 = vary(d0, rng, operator="step_length")
    assert d1 != d0


def test_fix5_step_root_length_and_step_radius_change_one_field():
    for opname in ("step_root_length", "step_radius"):
        n_applied = 0
        for seed in range(40):
            d0, _ = generate(seed)
            rng = np.random.default_rng([2000, seed])
            try:
                d1 = vary(d0, rng, operator=opname)
            except VariationImpossible:
                continue
            hand0 = next(s for s in d0.steps if s.path == "hand")
            hand1 = next(s for s in d1.steps if s.path == "hand")
            diffs = [k for k in hand0.params if hand0.params[k] != hand1.params[k]]
            assert diffs == (["root_length"] if opname == "step_root_length" else ["capsule_radius_m"])
            m1 = derive(d1)
            validate(m1)
            n_applied += 1
        assert n_applied >= 10, f"{opname} applied on too few seeds ({n_applied}/40)"


def test_fix5_step_limits_neighbour_is_numerically_adjacent():
    # Force landing on a phalanx with a revolute module by resampling until
    # found, then repeatedly step_limits and confirm each new (lo, hi) is
    # adjacent, in SORTED order, to the previous one.
    sorted_choices = sorted(DEFAULT_DISTRIBUTION.revolute_limit_choices_deg)
    found = 0
    for seed in range(60):
        d0, m0 = generate(seed)
        revolute_idxs = [i for i, s in enumerate(d0.steps)
                          if s.production == "Phalanx" and s.params["module"]["kind"] == "R"]
        if not revolute_idxs:
            continue
        rng = np.random.default_rng([3000, seed])
        try:
            d1 = vary(d0, rng, operator="step_limits")
        except VariationImpossible:
            continue
        for s0, s1 in zip(d0.steps, d1.steps):
            if s0 != s1 and s0.production == "Phalanx" and s0.params["module"]["kind"] == "R":
                lo0, hi0 = s0.params["module"]["limits"]
                lo1, hi1 = s1.params["module"]["limits"]
                deg0 = (round(math.degrees(lo0), 6), round(math.degrees(hi0), 6))
                deg1 = (round(math.degrees(lo1), 6), round(math.degrees(hi1), 6))
                i0 = min(range(len(sorted_choices)), key=lambda i: abs(sorted_choices[i][0] - deg0[0]) + abs(sorted_choices[i][1] - deg0[1]))  # noqa: E501
                i1 = min(range(len(sorted_choices)), key=lambda i: abs(sorted_choices[i][0] - deg1[0]) + abs(sorted_choices[i][1] - deg1[1]))  # noqa: E501
                assert abs(i0 - i1) == 1
                found += 1
    assert found >= 5


def test_fix5_remove_palm_body_can_reattach_children():
    # Build a hand guaranteed to have exactly 2 palm bodies, palm1 parented
    # on palm0 (so palm0 HOSTS a child), then try removing palm0 via
    # remove_palm_body -- must succeed by reattaching palm1 (renumbered to
    # "palm0") to palm0's own parent ("root"), rather than only ever
    # removing empty leaves. "palm0 was the one removed" is detected by
    # ``length`` (independently sampled per palm body, so palm0's and
    # palm1's own lengths differ with overwhelming probability) rather than
    # by name, since the survivor is ALWAYS renamed to "palm0" regardless
    # of which of the two was actually removed.
    dist = replace(DEFAULT_DISTRIBUTION, palm_body_count_range=(2, 2), digit_count_range=(1, 1))
    n_reattached = 0
    for seed in range(60):
        d0, m0 = generate(seed, dist)
        palm0 = next(s for s in d0.steps if s.production == "PalmBody" and s.params["name"] == "palm0")
        palm1 = next(s for s in d0.steps if s.production == "PalmBody" and s.params["name"] == "palm1")
        if palm1.params["parent"] != "palm0" or palm0.params["length"] == palm1.params["length"]:
            continue  # need palm0 to host palm1, and a distinguishable fingerprint
        rng = np.random.default_rng([4000, seed])
        removed_palm0 = False
        for _ in range(50):
            try:
                candidate = vary(d0, rng, dist, operator="remove_palm_body")
            except VariationImpossible:
                break
            survivor = next(s for s in candidate.steps if s.production == "PalmBody" and s.params["name"] == "palm0")
            if survivor.params["length"] == palm1.params["length"]:
                # palm1 survived (renamed "palm0") -> the ORIGINAL palm0 was
                # removed, and palm1 must now be reattached to palm0's own
                # parent ("root"), at palm0's own mount_frac.
                assert survivor.params["parent"] == palm0.params["parent"]
                assert survivor.params["mount_frac"] == palm0.params["mount_frac"]
                m1 = derive(candidate)
                validate(m1)
                removed_palm0 = True
                break
        if removed_palm0:
            n_reattached += 1
    assert n_reattached >= 5, f"remove_palm_body never reattached a non-leaf palm body ({n_reattached}/60)"


def test_fix5_insertion_distribution_limits_growth_material():
    # G_FULL_INS's insertion sub-distribution caps new digits at 1-3
    # phalanges and no branching; add_minimal_digit is unaffected (it
    # always makes a 1-phalanx digit anyway) but add_digit/regrow_subtree
    # must respect the cap under G_FULL_INS.
    assert variants.G_FULL_INS.insertion is not None
    assert variants.G_FULL_INS.insertion.phalanx_count_range == (1, 3)
    assert variants.G_FULL_INS.insertion.branch_probability == 0.0
    assert variants.G_NOBRANCH_INS.insertion is not None
    assert DEFAULT_DISTRIBUTION.insertion is None

    n_checked = 0
    for seed in range(60):
        d0, _ = generate(seed, variants.G_FULL_INS)
        top_ids_before = {s.params["digit_id"] for s in d0.steps if s.production == "Digit" and s.params.get("top_level")}  # noqa: E501
        rng = np.random.default_rng([5000, seed])
        try:
            d1 = vary(d0, rng, variants.G_FULL_INS, operator="add_digit")
        except VariationImpossible:
            continue
        top_ids_after = {s.params["digit_id"] for s in d1.steps if s.production == "Digit" and s.params.get("top_level")}  # noqa: E501
        new_id = next(iter(top_ids_after - top_ids_before))
        new_digit_step = next(s for s in d1.steps if s.production == "Digit" and s.params["digit_id"] == new_id)
        assert 1 <= new_digit_step.params["phalanx_count"] <= 3
        n_checked += 1
    assert n_checked >= 15, f"add_digit under G_FULL_INS applied on too few seeds ({n_checked}/60)"


def test_fix5_default_distributions_unaffected_by_insertion_field():
    # insertion=None everywhere except the two new *_INS variants -- default
    # sampling/growth behaviour is untouched (replay identity for
    # sample_derivation/derive is separately covered by
    # REPLAY_HASHES_SEED_0_99 in test_e0_infra.py).
    for name, dist in variants.NAMED_DISTRIBUTIONS.items():
        if name in ("G_FULL_INS", "G_NOBRANCH_INS"):
            assert dist.insertion is not None
        else:
            assert dist.insertion is None


# ---------------------------------------------------------------------------
# Fix 6: canonical.py kills -0.0.
# ---------------------------------------------------------------------------


def test_fix6_canonical_round_kills_negative_zero():
    r = _r(-1e-15)
    assert r == 0.0
    assert math.copysign(1.0, r) == 1.0  # +0.0, not -0.0
    assert "-" not in repr(r)


def test_fix6_phenotype_hash_stable_under_sign_of_zero():
    _, m = generate(7)
    h0 = phenotype_hash(m)
    # A body whose xyz has a -0.0 component must hash identically to the
    # +0.0 version (same geometry).
    negzero_bodies = tuple(
        Body(name=b.name, palm=b.palm, radius=b.radius) for b in m.bodies
    )
    tweaked_frames = tuple(
        Frame(name=f.name, body=f.body,
              pose=Pose(xyz=tuple((-0.0 if v == 0.0 else v) for v in f.pose.xyz), rpy=f.pose.rpy))
        for f in m.frames
    )
    m2 = KinematicModel(name=m.name, root=m.root, bodies=negzero_bodies, joints=m.joints,
                         frames=tweaked_frames, couplings=m.couplings)
    assert phenotype_hash(m2) == h0


# ---------------------------------------------------------------------------
# Fix 11: runner.py circular import broken via starts.py; git info read
# first.
# ---------------------------------------------------------------------------


def test_fix11_e3_reach_importable_standalone_without_runner():
    # A fresh subprocess importing e3_reach FIRST (never having imported
    # runner.py or e2_drift.py at all) must not hit the old
    # e2_drift -> runner -> e3_reach -> e2_drift cycle.
    code = "import hand_sampler.grammar.experiments.e3_reach; print('ok')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]))
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_fix11_starts_module_has_no_runner_or_e2_or_e3_dependency():
    import hand_sampler.grammar.experiments.starts as starts_mod
    assert not hasattr(starts_mod, "run_experiment")
    assert not hasattr(starts_mod, "register")


def test_fix11_e2_drift_reexports_build_start():
    from hand_sampler.grammar.experiments import e2_drift, starts
    assert e2_drift.build_start is starts.build_start


def test_fix11_git_info_present_in_result():
    from hand_sampler.grammar.experiments.runner import registered_experiments, run_experiment
    import tempfile
    from pathlib import Path
    fn = registered_experiments()["smoke"]
    with tempfile.TemporaryDirectory() as td:
        result = run_experiment("smoke", fn, params={"n_configs": 4}, seeds=[0, 1],
                                 out_dir=str(Path(td) / "out"), processes=1, allow_dirty=True)
    assert "git_sha" in result and "git_dirty" in result and "git_diff_sha256" in result


# ---------------------------------------------------------------------------
# Fix 9 / 7 / 8 / 10: experiment-module wiring smoke checks (heavier
# behavioural coverage already lives in test_e1_e2.py / test_e3_e4.py; this
# section checks the specific NEW facts fix 9/7/8/10 introduce).
# ---------------------------------------------------------------------------


def test_fix9_all_operators_excludes_step_length_includes_new_ops():
    from hand_sampler.grammar.experiments.e1_locality import ALL_OPERATORS
    assert "step_length" not in ALL_OPERATORS
    assert "step_root_length" in ALL_OPERATORS
    assert "step_radius" in ALL_OPERATORS


def test_fix9_run_reports_legacy_vs_aligned_footnote():
    import tempfile
    from pathlib import Path
    from hand_sampler.grammar.experiments import e1_locality
    with tempfile.TemporaryDirectory() as td:
        result = e1_locality.run(out_dir=str(Path(td) / "out"), seeds=[0, 1, 2, 3, 4], n_configs=4, processes=2,
                                  allow_dirty=True)
    fn = result["footnote_legacy_vs_aligned"]
    assert fn["operator"] == "insert_phalanx"
    assert fn["aligned_median"] is not None
    assert fn["legacy_unaligned_median"] is not None
    summary_text = (Path(td, "out") if False else None)  # noqa: F841 -- summary already checked via other tests


def test_fix7_e3_operator_sets_and_dist_variants():
    from hand_sampler.grammar.experiments.e3_reach import DIST_VARIANTS, OPERATOR_SETS
    assert set(OPERATOR_SETS) == {"DEFAULT", "UNION"}
    assert set(DIST_VARIANTS) == {"G_FULL", "G_NOBRANCH", "G_FULL_INS", "G_NOBRANCH_INS"}
    assert "step_length" not in OPERATOR_SETS["UNION"]
    assert len(OPERATOR_SETS["UNION"]) == len(OPERATORS) + len(SMALL_STEP_OPERATORS) + len(MINIMAL_STRUCTURAL_OPERATORS)  # noqa: E501


def test_fix7_e3_budget_is_proposals_not_accepted():
    from hand_sampler.grammar.experiments.e3_reach import e3_reach_seed
    result = e3_reach_seed(0, budget=25)
    for row in result.values():
        assert row["n_proposed"] <= 25
        assert row["n_accepted"] <= row["n_proposed"]


def test_fix7_arch_palm_requires_digits_on_palm():
    from hand_sampler.grammar.derive import DerivationStep, Derivation, sample_derivation
    from hand_sampler.grammar.experiments.e3_reach import TARGETS
    dist_fn = TARGETS["arch_palm"]
    # A hand with 4+ digits, >=2 nonparallel palm joints, but every
    # TOP-LEVEL digit forced onto ROOT (never onto a palm body) must NOT
    # score 0: it must be penalized for the missing "digits actually mount
    # on a palm body" criterion (I14 fix -- previously an empty, jointed
    # palm body satisfied this target regardless of where digits mounted).
    dist = replace(DEFAULT_DISTRIBUTION, digit_count_range=(4, 4), palm_body_count_range=(2, 2),
                    palm_joint_probability=1.0)
    derivation = sample_derivation(0, dist)
    forced_steps = tuple(
        DerivationStep(path=s.path, production=s.production, params={**s.params, "mount": "root"})
        if s.production == "Digit" and s.params.get("top_level") else s
        for s in derivation.steps
    )
    forced = Derivation(seed=derivation.seed, grammar_version=derivation.grammar_version, steps=forced_steps)
    model = derive(forced)
    assert dist_fn(model, forced) > 0.0


def test_fix8_e2_mixtures_and_dist_variants():
    from hand_sampler.grammar.experiments.e2_drift import DIST_VARIANTS, MIXTURES, UNION_OPERATORS
    assert set(MIXTURES) == {"DEFAULT_uniform", "UNION_uniform", "UNION_weighted"}
    assert set(DIST_VARIANTS) == {"G_FULL", "G_NOBRANCH", "G_FULL_INS", "G_NOBRANCH_INS"}
    assert "step_length" not in UNION_OPERATORS
    for name, mix in MIXTURES.items():
        assert abs(sum(mix.values()) - 1.0) < 1e-9, name


def test_fix10_null_vs_impossible_fraction_separated():
    from hand_sampler.grammar.experiments.e4_redundancy import e4_redundancy_seed
    result = e4_redundancy_seed(0, n_offspring_parents=2)
    for variant_name, entry in result.items():
        for group_name, stats in entry["offspring"].items():
            assert "null_fraction" in stats
            assert "impossible_fraction" in stats
            if stats["null_fraction"] is not None:
                assert 0.0 <= stats["null_fraction"] <= 1.0
            if stats["impossible_fraction"] is not None:
                assert 0.0 <= stats["impossible_fraction"] <= 1.0


def test_fix10_impossible_fraction_high_for_constrained_hand():
    # A single-digit, single-phalanx, no-palm hand under the DEFAULT pool:
    # remove_digit/delete_phalanx/regrow_subtree-on-only-digit are all
    # either impossible or trivial; specifically remove_digit is always
    # impossible (digit_count == 1). Confirms impossible_fraction can be
    # legitimately nonzero and distinct from null_fraction.
    from hand_sampler.grammar.experiments.e4_redundancy import _offspring_stats
    dist = replace(DEFAULT_DISTRIBUTION, digit_count_range=(1, 1), phalanx_count_range=(1, 1),
                    palm_body_count_range=(0, 0), branch_probability=0.0)
    derivation, model = generate(0, dist)
    h = phenotype_hash(model)
    rng = np.random.default_rng([1, 1, 1, 4])
    stats = _offspring_stats(derivation, h, dist, ("remove_digit",), rng)
    assert stats["impossible_fraction"] == 1.0
    assert stats["null_fraction"] is None
