"""Acceptance tests for fix iteration I15 (see project-notes/grammar/opus-review-final.md).

One section per numbered fix in the coordinator's task list (Part A items
1, 2, 4, 5, 7), plus the new envelope module and a tiny E5b run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar.derive import (
    Derivation,
    VariationImpossible,
    derive,
    generate,
    joint_identity,
    vary,
)
from hand_sampler.grammar.envelope import fits_envelope
from hand_sampler.grammar.experiments import runner  # noqa: F401 -- import first to avoid the e2_drift<->runner cycle.
from hand_sampler.grammar.experiments import e5b_evolve_sim
from hand_sampler.grammar.experiments.runner import run_experiment
from hand_sampler.grammar.kinematics import Body, Frame, Joint, KinematicModel, Pose, validate
from hand_sampler.grammar.phenodist import phenotype_distance
from hand_sampler.grammar.proxy import reach_coverage
from hand_sampler.grammar import variants


# ---------------------------------------------------------------------------
# I15 fix 1: stable joint uid + joint_identity + uid-aware phenotype_distance.
# ---------------------------------------------------------------------------


def test_fix1_joint_identity_covers_every_joint_with_unique_uids():
    for seed in range(20):
        d, m = generate(seed, variants.G_FULL)
        ji = joint_identity(d)
        assert set(ji.keys()) == {j.name for j in m.joints}
        uids = list(ji.values())
        assert len(set(uids)) == len(uids), "uid collision within one derivation"


def test_fix1_insert_phalanx_child_aligned_by_uid_matches_every_parent_joint():
    """The child of ``insert_phalanx`` has exactly one NEW joint; every
    pre-existing parent joint must be matched by ``joint_identity``'s uid
    (not merely by name, which insert/delete_phalanx renumbering can
    break)."""
    n_checked = 0
    for seed in range(60):
        d, parent = generate(seed, variants.G_FULL)
        rng = np.random.default_rng([seed, 999])
        try:
            cd = vary(d, rng, variants.G_FULL, operator="insert_phalanx")
        except VariationImpossible:
            continue
        child = derive(cd)
        ji_parent, ji_child = joint_identity(d), joint_identity(cd)
        dist = phenotype_distance(parent, child, seed, n_configs=8, identity_a=ji_parent, identity_b=ji_child)
        assert dist["n_shared_joints"] == len(parent.joints)
        assert dist["joint_count_delta"] == 1.0
        n_checked += 1
    assert n_checked >= 15


def _find_step_coupling_case(seed_range=range(400)):
    """A (seed, child_derivation) pair where ``step_coupling`` changed a
    Coupled module's multiplier/offset -- deterministic search, no reliance
    on any particular seed succeeding."""
    for seed in seed_range:
        d, parent = generate(seed, variants.G_FULL)
        rng = np.random.default_rng([seed, 12345])
        try:
            cd = vary(d, rng, variants.G_FULL, operator="step_coupling")
        except VariationImpossible:
            continue
        child = derive(cd)
        yield seed, d, parent, cd, child


def test_fix1_step_coupling_child_shows_nonzero_tip_displacement():
    """Coupled dependents must be recomputed via the CHILD's own coupling
    (coords.q_from_u), never copied from the parent's raw value -- so a
    changed multiplier/offset must show up as nonzero tip displacement."""
    found_nonzero = 0
    checked = 0
    for seed, d, parent, cd, child in _find_step_coupling_case():
        ji_parent, ji_child = joint_identity(d), joint_identity(cd)
        dist = phenotype_distance(parent, child, seed, n_configs=16, identity_a=ji_parent, identity_b=ji_child)
        checked += 1
        if dist["tip_displacement_m"] > 0.0:
            found_nonzero += 1
        if checked >= 60:
            break
    assert checked >= 20
    assert found_nonzero >= checked // 4, (found_nonzero, checked)


def test_fix1_shared_independent_value_clamped_into_child_limits():
    """A shared independent joint whose value would be out-of-range for the
    child's own (possibly different) declared limits must be clamped, not
    passed through raw."""
    bodies = [Body(name="root", palm=True), Body(name="f1")]
    joints_a = [Joint(name="j1", type="revolute", parent="root", child="f1",
                       origin=Pose(xyz=(0.0, 0.0, 0.0)), axis=(0.0, 1.0, 0.0), limits=(-3.0, 3.0))]
    joints_b = [Joint(name="j1", type="revolute", parent="root", child="f1",
                       origin=Pose(xyz=(0.0, 0.0, 0.0)), axis=(0.0, 1.0, 0.0), limits=(-0.1, 0.1))]
    frames = [Frame(name="f1_tip", body="f1", pose=Pose(xyz=(0.0, 0.0, 0.05)))]
    a = KinematicModel(name="a", root="root", bodies=tuple(bodies), joints=tuple(joints_a),
                        frames=tuple(frames), couplings=(), independent=("j1",))
    b = KinematicModel(name="b", root="root", bodies=tuple(bodies), joints=tuple(joints_b),
                        frames=tuple(frames), couplings=(), independent=("j1",))
    validate(a)
    validate(b)
    # a's admissible box for j1 is (-3, 3); with n_configs=0 the only samples
    # are the extremal ones (all_lower/all_upper hit +-3 exactly), which
    # must be clamped into b's own (-0.1, 0.1) limits.
    dist = phenotype_distance(a, b, seed=0, n_configs=0)
    assert dist["tip_displacement_m"] > 0.0
    assert dist["tip_displacement_m"] < 0.05 * 2  # sane upper bound (never blows past the segment length)


# ---------------------------------------------------------------------------
# I15 fix 2: reach_coverage normalised by a fixed reference length.
# ---------------------------------------------------------------------------


def test_fix2_reach_coverage_not_penalized_by_extra_digit():
    """Adding a THIRD non-overlapping finger (more reachable volume, same
    per-body-length metric) must not shrink reach_coverage the way dividing
    by the hand's own total length would."""
    def _model(n_fingers: int) -> KinematicModel:
        bodies = [Body(name="root", palm=True)]
        joints = []
        frames = []
        for i in range(n_fingers):
            name = f"f{i}"
            bodies.append(Body(name=name))
            joints.append(Joint(name=f"j{i}", type="revolute", parent="root", child=name,
                                 origin=Pose(xyz=(0.01 * i, 0.0, 0.0)), axis=(1.0, 0.0, 0.0),
                                 limits=(-0.4, 0.4)))
            frames.append(Frame(name=f"{name}_tip", body=name, pose=Pose(xyz=(0.0, 0.02, 0.05))))
        m = KinematicModel(name="m", root="root", bodies=tuple(bodies), joints=tuple(joints),
                            frames=tuple(frames), couplings=())
        validate(m)
        return m

    configs2 = [{"j0": v, "j1": v} for v in np.linspace(-0.4, 0.4, 6)]
    configs3 = [{"j0": v, "j1": v, "j2": v} for v in np.linspace(-0.4, 0.4, 6)]
    r2 = reach_coverage(_model(2), configs2)
    r3 = reach_coverage(_model(3), configs3)
    assert r3 >= r2, (r2, r3)


# ---------------------------------------------------------------------------
# I15 fix 4: delete_phalanx may delete a branch-hosting phalanx (branch
# re-attached); remove_digit_minimal may remove a digit with up to 2
# phalanges.
# ---------------------------------------------------------------------------


def test_fix4_delete_phalanx_can_remove_branch_hosting_phalanx():
    """Search for a (seed, rng) pair where the phalanx actually deleted
    hosts a branch, and confirm: exactly one joint removed (the deleted
    phalanx's own), and the branch's own joints all survive (present in
    the child, matched by uid against the parent)."""
    found = 0
    for seed in range(300):
        d, parent = generate(seed, variants.G_FULL)
        branch_phalanx_uids = {
            s.params["uid"] for s in d.steps
            if s.production == "Phalanx" and s.params["branch_digit_count"] > 0
        }
        if not branch_phalanx_uids:
            continue
        ji_parent = joint_identity(d)
        uid_to_joint_parent = {v: k for k, v in ji_parent.items()}
        for trial in range(30):
            rng = np.random.default_rng([seed, trial])
            try:
                cd = vary(d, rng, variants.G_FULL, operator="delete_phalanx")
            except VariationImpossible:
                continue
            deleted_uids = branch_phalanx_uids - {s.params["uid"] for s in cd.steps if s.production == "Phalanx"}
            if not deleted_uids:
                continue
            # A branch-hosting phalanx was actually deleted this trial.
            child = derive(cd)
            assert len(child.joints) == len(parent.joints) - 1
            ji_child = joint_identity(cd)
            # Every joint the parent had, OTHER than the deleted phalanx's
            # own, must still be present in the child (matched by uid) --
            # i.e. the branch's own joints were relocated, not orphaned.
            deleted_uid = next(iter(deleted_uids))
            deleted_joint_name = uid_to_joint_parent[deleted_uid]
            for jname, uid in ji_parent.items():
                if jname == deleted_joint_name:
                    continue
                assert uid in ji_child.values(), f"parent joint {jname!r} (uid {uid}) missing from child"
            found += 1
            break
        if found >= 3:
            break
    assert found >= 3, "could not find a delete_phalanx application removing a branch-hosting phalanx"


def test_fix4_remove_digit_minimal_can_remove_two_phalanx_digit():
    """Build a derivation with a >=2-phalanx, branch-free, top-level digit
    by growing one with add_minimal_digit + insert_phalanx, then confirm
    remove_digit_minimal can remove it (digit_count drops by exactly one,
    joint_count drops by exactly 2)."""
    from hand_sampler.grammar.experiments.starts import build_start

    d, _dc, _mp, _full = build_start(0, variants.G_NOBRANCH, max_digits=2, max_phalanges=2)
    m = derive(d)
    rng = np.random.default_rng([1])
    # Grow a fresh 1-phalanx digit, then grow it to 2 phalanges.
    d2 = vary(d, rng, variants.G_NOBRANCH, operator="add_minimal_digit")
    top_ids_before = {s.params["digit_id"] for s in d.steps if s.production == "Digit" and s.params.get("top_level")}
    new_ids = {s.params["digit_id"] for s in d2.steps if s.production == "Digit" and s.params.get("top_level")} - top_ids_before  # noqa: E501
    assert len(new_ids) == 1
    new_digit_id = next(iter(new_ids))

    # Force-insert a second phalanx onto exactly that new digit by retrying
    # insert_phalanx until it lands on new_digit_id (small digit count means
    # this is the only 1-phalanx digit most of the time, but retry to be safe).
    d3 = None
    for trial in range(200):
        rng2 = np.random.default_rng([2, trial])
        try:
            cand = vary(d2, rng2, variants.G_NOBRANCH, operator="insert_phalanx")
        except VariationImpossible:
            continue
        cand_phalanx_count = next(
            s.params["phalanx_count"] for s in cand.steps if s.production == "Digit" and s.params["digit_id"] == new_digit_id  # noqa: E501
        )
        if cand_phalanx_count == 2:
            d3 = cand
            break
    assert d3 is not None, "could not grow the new digit to 2 phalanges"
    m3 = derive(d3)

    found = False
    for trial in range(200):
        rng3 = np.random.default_rng([3, trial])
        try:
            d4 = vary(d3, rng3, variants.G_NOBRANCH, operator="remove_digit_minimal")
        except VariationImpossible:
            continue
        new_digit_ids_after = {
            s.params["digit_id"] for s in d4.steps if s.production == "Digit" and s.params.get("top_level")
        }
        if new_digit_id not in new_digit_ids_after:
            m4 = derive(d4)
            assert len(m4.joints) == len(m3.joints) - 2
            found = True
            break
    assert found, "remove_digit_minimal never removed the 2-phalanx digit across 200 trials"


# ---------------------------------------------------------------------------
# I15 fix 5: run_experiment refuses a dirty tree unless allow_dirty=True.
# ---------------------------------------------------------------------------


def test_fix5_run_experiment_refuses_dirty_tree_by_default(monkeypatch):
    from hand_sampler.grammar.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_git_info", lambda repo_dir: {"sha": "deadbeef", "dirty": True})
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(RuntimeError):
            run_experiment(
                "smoke", lambda seed, **kw: {"x": 1.0}, params={}, seeds=[0],
                out_dir=str(Path(td) / "out"),
            )


def test_fix5_allow_dirty_records_git_diff_sha256(monkeypatch):
    from hand_sampler.grammar.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_git_info", lambda repo_dir: {"sha": "deadbeef", "dirty": True})
    monkeypatch.setattr(runner_mod, "_git_diff_sha256", lambda repo_dir: "abc123")
    with tempfile.TemporaryDirectory() as td:
        result = run_experiment(
            "smoke", lambda seed, **kw: {"x": 1.0}, params={}, seeds=[0],
            out_dir=str(Path(td) / "out"), allow_dirty=True,
        )
    assert result["git_dirty"] is True
    assert result["git_diff_sha256"] == "abc123"


def test_fix5_clean_tree_runs_without_allow_dirty(monkeypatch):
    from hand_sampler.grammar.experiments import runner as runner_mod

    monkeypatch.setattr(runner_mod, "_git_info", lambda repo_dir: {"sha": "deadbeef", "dirty": False})
    with tempfile.TemporaryDirectory() as td:
        result = run_experiment(
            "smoke", lambda seed, **kw: {"x": 1.0}, params={}, seeds=[0],
            out_dir=str(Path(td) / "out"),
        )
    assert result["git_dirty"] is False
    assert result["git_diff_sha256"] is None


# ---------------------------------------------------------------------------
# I15 fix 7: E3 arch_palm target requires digits mounted on a JOINTED palm body.
# ---------------------------------------------------------------------------


def test_fix7_arch_palm_rejects_digits_on_unjointed_palm_body():
    from hand_sampler.grammar.experiments.e3_reach import _dist_arch_palm
    from hand_sampler.grammar.derive import Derivation, DerivationStep

    def _mk(mount_on_jointed: bool) -> float:
        hand = DerivationStep(path="hand", production="Hand",
                               params={"digit_count": 4, "palm_body_count": 2, "root_length": 0.05,
                                       "capsule_radius_m": 0.01})
        palm0 = DerivationStep(path="palm/0", production="PalmBody", params={
            "name": "palm0", "parent": "root", "mount_frac": 0.5, "length": 0.03,
            "direction_rpy": (0.0, 0.0, 0.0), "has_joint": True, "axis": (1.0, 0.0, 0.0), "limits": (-0.3, 0.3),
        })
        palm1 = DerivationStep(path="palm/1", production="PalmBody", params={
            "name": "palm1", "parent": "root", "mount_frac": 0.5, "length": 0.03,
            "direction_rpy": (0.0, 1.0, 0.0), "has_joint": True, "axis": (0.0, 1.0, 0.0), "limits": (-0.3, 0.3),
        })
        digit_mount = "palm0" if mount_on_jointed else "root"
        steps = [hand, palm0, palm1]
        for i in range(1, 5):
            digit_id = str(i)
            steps.append(DerivationStep(path=f"digit/{digit_id}", production="Digit", params={
                "digit_id": digit_id, "mount": digit_mount if i <= 2 else "root",
                "mount_frac": 0.0, "mount_rpy": (0.0, 0.0, 0.0),
                "phalanx_count": 1, "top_level": True, "depth": 0,
            }))
            steps.append(DerivationStep(path=f"digit/{digit_id}/phalanx/0", production="Phalanx", params={
                "digit_id": digit_id, "p": 0,
                "module": {"kind": "R", "axis": (1.0, 0.0, 0.0), "limits": (-0.3, 0.3)},
                "length": 0.03, "branch_digit_count": 0,
            }))
        derivation = Derivation(seed=0, grammar_version="0.3", steps=tuple(steps))
        model = derive(derivation)
        return _dist_arch_palm(model, derivation)

    # Import GRAMMAR_VERSION to avoid hardcoding a mismatched version string.
    from hand_sampler.grammar.rules import GRAMMAR_VERSION
    assert GRAMMAR_VERSION == "0.3"
    d_unjointed = _mk(mount_on_jointed=False)
    d_jointed = _mk(mount_on_jointed=True)
    assert d_jointed < d_unjointed
    assert d_jointed == 0.0


# ---------------------------------------------------------------------------
# Envelope.
# ---------------------------------------------------------------------------


def test_envelope_accepts_small_serial_hand():
    from hand_sampler.grammar.experiments.starts import build_start

    d, _dc, _mp, _full = build_start(0, variants.G_SERIAL, max_digits=2, max_phalanges=2)
    m = derive(d)
    ok, reasons = fits_envelope(m)
    assert ok, reasons


def test_envelope_rejects_too_many_digits():
    for seed in range(50):
        d, m = generate(seed, variants.G_FULL)
        from hand_sampler.grammar.phenodist import _digit_count
        if _digit_count(m) > 5:
            ok, reasons = fits_envelope(m, max_digits=5)
            assert not ok
            assert any("digit_count" in r for r in reasons)
            return
    pytest.fail("no seed in range(50) produced > 5 digits under G_FULL")


def test_envelope_rejects_palm_joint_when_disallowed():
    for seed in range(50):
        d, m = generate(seed, variants.G_FULL)
        has_palm_joint = any(
            s.production == "PalmBody" and s.params["has_joint"] for s in d.steps
        )
        if has_palm_joint:
            ok, reasons = fits_envelope(m, allow_palm_joints=False)
            assert not ok
            assert any("palm joint" in r for r in reasons)
            ok2, _ = fits_envelope(m, allow_palm_joints=True, max_digits=999, max_joints_per_digit=999)
            return
    pytest.fail("no seed in range(50) produced a palm joint under G_FULL")


def test_envelope_rejects_branch_when_disallowed():
    for seed in range(200):
        d, m = generate(seed, variants.G_FULL)
        if any(s.production == "Phalanx" and s.params["branch_digit_count"] > 0 for s in d.steps):
            ok, reasons = fits_envelope(m, allow_branches=False, max_digits=999, max_joints_per_digit=999)
            assert not ok
            assert any("branch" in r for r in reasons)
            return
    pytest.fail("no seed in range(200) produced a branch under G_FULL")


# ---------------------------------------------------------------------------
# E5b: tiny run into a tmp dir, required keys, determinism.
# ---------------------------------------------------------------------------


def test_e5b_tiny_run_result_json_keys_and_determinism(tmp_path):
    conditions = e5b_evolve_sim.all_conditions()[:2]
    kwargs = dict(
        restarts=[0, 1], conditions=conditions, mu=4, lam=4, generations=3, n_proxy_configs=4,
        processes=1, allow_dirty=True,
    )
    r1 = e5b_evolve_sim.run(out_dir=str(tmp_path / "run1"), **kwargs)
    r2 = e5b_evolve_sim.run(out_dir=str(tmp_path / "run2"), **kwargs)
    assert r1["per_seed"] == r2["per_seed"]
    assert r1["aggregate"] == r2["aggregate"]

    assert r1["aggregate"]["n_conditions"] == len(conditions)
    for key, c in r1["aggregate"]["per_condition"].items():
        for req in ("n_restarts", "final_best_raw_proxy", "final_mean_motors", "final_mean_joints",
                    "final_mean_digits", "target_reach_rate", "clone_rate",
                    "rejection_rate_per_operator", "improvement_rate_per_operator"):
            assert req in c, f"{key} missing {req!r}"
    for row in r1["per_seed"]:
        assert row["ok"], row["error"]
        res = row["result"]
        for req in ("trajectory", "op_stats", "op_improve", "n_clone_events", "n_mutate_calls",
                    "final_best_raw_proxy", "reached_target", "sigma"):
            assert req in res, f"missing {req!r} in per-restart result"
        assert len(res["trajectory"]) == kwargs["generations"] + 1

    assert (tmp_path / "run1" / "result.json").exists()
    assert (tmp_path / "run1" / "summary.md").exists()
    json.loads((tmp_path / "run1" / "result.json").read_text())


def test_e5b_envelope_rejects_palm_joint_proposals_when_pool_has_palm_ops():
    """Sanity: running a few generations under UNION_weighted (which
    includes palm operators) with the default (palm-refusing) envelope
    must actually record some envelope_rejected count for at least one
    palm operator -- confirms the envelope is wired into the mutate loop,
    not merely defined."""
    result = e5b_evolve_sim.e5b_evolve_seed(
        (0, "G_NOBRANCH", "UNION_weighted", "antipodal_pinch", "0.01"),
        mu=8, lam=8, generations=5, n_proxy_configs=4,
    )
    palm_ops = ("add_palm_body", "toggle_palm_joint")
    total_rejected = sum(result["op_stats"][op]["envelope_rejected"] for op in palm_ops if op in result["op_stats"])
    assert total_rejected > 0
