"""Grammar 0.5, iteration A acceptance tests:

  1. The rest-bend primitive (``bend_rpy``/``bend_offset``) defaults to zero
     and never touches the RNG under the default ``Distribution``, so every
     existing seed's replay hash (``test_e0_infra.REPLAY_HASHES_SEED_0_99``)
     is unchanged.
  2. ``variants.G_BEND`` produces at least one nonzero bend across a batch
     of seeds, and every one of those designs' forward kinematics matches
     an independent Pinocchio oracle on the exported URDF.
  3. ``step_bend_rpy``/``step_bend_offset`` each change exactly one
     component of one Phalanx step's own bend field.
  4. Continuous revolute limits (``variants.G_CONT``) sample inside
     ``revolute_limit_range_deg`` with ``lo < hi``, and ``step_limits``
     keeps ``lo < hi`` under repeated application.
  5. ``canonical.normalize_axis_sign`` invariance: a flipped-axis/negated-
     limits joint canonicalizes/hashes identically to its un-flipped
     equivalent.
  6. Coverage: ``limits_continuous`` relax reports fewer out-of-support
     items than the default on a real hand (Allegro), and a synthetic
     coupling whose dependent limits are strictly tighter than (but
     contained in) the source's affine image is never flagged
     ``coupling_limits_outside_image`` -- only noted as
     ``dependent_limits_tighter``.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.canonical import canonical_form, normalize_axis_sign, phenotype_hash
from hand_sampler.grammar.coords import q_from_u, sample_configurations
from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.derive import Derivation, DerivationStep, derive, generate, sample_derivation, vary
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION, DEG
from hand_sampler.grammar.kinematics import AffineCoupling, Body, Joint, KinematicModel
from hand_sampler.grammar.variants import G_BEND, G_CONT
from hand_sampler.grammar_bench.tests.test_e0_infra import REPLAY_HASHES_SEED_0_99
from hand_sampler.grammar_bench.tolerances import ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent.parent
REAL_DIR = BENCH_DIR / "fixtures" / "real"
ALLEGRO_URDF = REAL_DIR / "allegro_right" / "allegro_hand_description_right.urdf"

ORACLE_PYTHON = Path("/home/singularity/anaconda3/envs/piper/bin/python")
ORACLE_FK_SCRIPT = BENCH_DIR / "refgen" / "oracle_fk.py"

N_BEND_SEEDS = 20


def _oracle_available() -> bool:
    return ORACLE_PYTHON.is_file() and ORACLE_FK_SCRIPT.is_file()


def to_json_model(model):
    from hand_sampler.grammar.adapters.json_io import to_json
    return to_json(model)


# ---------------------------------------------------------------------------
# 1. Bend fields default to zero; replay hashes hold.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_bend_fields_default_zero_and_replay_holds(seed):
    derivation, model = generate(seed, DEFAULT_DISTRIBUTION)
    for s in derivation.steps:
        if s.production != "Phalanx":
            continue
        assert s.params["bend_rpy"] == (0.0, 0.0, 0.0)
        assert s.params["bend_offset"] == (0.0, 0.0)

    import hashlib
    h = hashlib.sha256(to_json_model(model).encode("utf-8")).hexdigest()
    assert h == REPLAY_HASHES_SEED_0_99[seed], f"seed={seed}: replay hash changed"


# ---------------------------------------------------------------------------
# 2. G_BEND: at least one nonzero bend, FK matches Pinocchio oracle.
# ---------------------------------------------------------------------------


def test_g_bend_has_nonzero_bend_and_matches_oracle():
    if not _oracle_available():
        pytest.skip("oracle-unavailable: piper conda interpreter not found")

    any_nonzero = False
    n_compared = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for seed in range(N_BEND_SEEDS):
            derivation, model = generate(seed, G_BEND)
            for s in derivation.steps:
                if s.production == "Phalanx" and (
                    s.params["bend_rpy"] != (0.0, 0.0, 0.0) or s.params["bend_offset"] != (0.0, 0.0)
                ):
                    any_nonzero = True

            text, _losses = to_urdf(model)
            configs = sample_configurations(model, 3, seed)

            urdf_path = tmp / f"{seed}.urdf"
            urdf_path.write_text(text)
            cfg_path = tmp / f"{seed}.configs.json"
            cfg_path.write_text(json.dumps(configs))
            out_path = tmp / f"{seed}.out.json"

            proc = subprocess.run(
                [str(ORACLE_PYTHON), str(ORACLE_FK_SCRIPT), "--urdf", str(urdf_path),
                 "--configs", str(cfg_path), "--out", str(out_path)],
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"oracle_fk.py failed on seed={seed}:\n{proc.stdout}\n{proc.stderr}"

            oracle_out = json.loads(out_path.read_text())
            orig_names = [b.name for b in model.bodies]
            for u, oracle_poses in zip(configs, oracle_out["poses"]):
                q = q_from_u(model, u)
                ours = fk.forward_kinematics(model, q)
                assert set(orig_names) <= set(oracle_poses)
                for name in orig_names:
                    T_oracle = np.array(oracle_poses[name])
                    pos_err = fk.position_error(ours[name], T_oracle)
                    rot_err = fk.rotation_error(ours[name], T_oracle)
                    n_compared += 1
                    assert pos_err <= ORACLE_POS_M, f"seed={seed} body={name} pos_err={pos_err}"
                    assert rot_err <= ORACLE_ROT_RAD, f"seed={seed} body={name} rot_err={rot_err}"

            urdf_path.unlink()
            cfg_path.unlink()
            out_path.unlink()

    assert any_nonzero, "expected at least one nonzero bend across G_BEND seeds"
    assert n_compared > 0


# ---------------------------------------------------------------------------
# 2b. I22 fix 1: _compose_bend_rpy is a proper rotation composition
# (R(mount) @ R(bend)), not componentwise Euler addition -- a nonzero
# phalanx-0 bend rotates about the MOUNT frame's own local axes, and a
# continuation phalanx's bend is unchanged (base orientation is identity).
# ---------------------------------------------------------------------------


def _one_digit_derivation(mount_rpy, bend_rpy_p0, bend_rpy_p1):
    from hand_sampler.grammar.rules import GRAMMAR_VERSION

    steps = [
        DerivationStep(path="hand", production="Hand", params={
            "digit_count": 1, "palm_body_count": 0, "root_length": 0.05, "capsule_radius_m": 0.01,
        }),
        DerivationStep(path="digit/1", production="Digit", params={
            "digit_id": "1", "mount": "root", "mount_frac": 0.5, "mount_rpy": mount_rpy,
            "phalanx_count": 2, "top_level": True, "depth": 0, "uid": 0,
        }),
        DerivationStep(path="digit/1/phalanx/0", production="Phalanx", params={
            "digit_id": "1", "p": 0, "module": {"kind": "R", "axis": (0.0, 1.0, 0.0), "limits": (-0.3, 0.9)},
            "length": 0.02, "branch_digit_count": 0, "uid": 1,
            "bend_rpy": bend_rpy_p0, "bend_offset": (0.0, 0.0),
        }),
        DerivationStep(path="digit/1/phalanx/1", production="Phalanx", params={
            "digit_id": "1", "p": 1, "module": {"kind": "R", "axis": (0.0, 1.0, 0.0), "limits": (-0.3, 0.9)},
            "length": 0.02, "branch_digit_count": 0, "uid": 2,
            "bend_rpy": bend_rpy_p1, "bend_offset": (0.0, 0.0),
        }),
    ]
    return Derivation(seed=-1, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))


def test_phalanx0_bend_rotates_about_mount_frame_local_axes():
    mount_rpy = (0.3, -0.4, 0.5)
    bend_rpy_p0 = (0.2, 0.1, -0.15)
    bend_rpy_p1 = (0.05, -0.07, 0.09)  # continuation bend, should be UNCHANGED

    derivation = _one_digit_derivation(mount_rpy, bend_rpy_p0, bend_rpy_p1)
    m = derive(derivation)

    j0 = next(j for j in m.joints if j.name == "d1p1_j")
    R0 = fk.rpy_to_matrix(j0.origin.rpy)
    expected0 = fk.rpy_to_matrix(mount_rpy) @ fk.rpy_to_matrix(bend_rpy_p0)
    assert np.allclose(R0, expected0, atol=1e-12), "phalanx-0 bend must compose as R(mount) @ R(bend)"

    j1 = next(j for j in m.joints if j.name == "d1p2_j")
    assert j1.origin.rpy == bend_rpy_p1, "a continuation phalanx's bend_rpy must be unchanged (base orientation is identity)"


# ---------------------------------------------------------------------------
# 3. step_bend_rpy / step_bend_offset change exactly one component.
# ---------------------------------------------------------------------------


def _phalanx_params_by_uid(derivation: Derivation):
    return {s.params["uid"]: s.params for s in derivation.steps if s.production == "Phalanx"}


def test_step_bend_rpy_changes_one_component():
    rng = np.random.default_rng(12345)
    changed_any = False
    for _ in range(50):
        derivation = sample_derivation(int(rng.integers(0, 1_000_000)), G_BEND)
        before = _phalanx_params_by_uid(derivation)
        try:
            after_derivation = vary(derivation, rng, dist=G_BEND, operator="step_bend_rpy")
        except Exception:
            continue
        after = _phalanx_params_by_uid(after_derivation)
        diffs = []
        for uid, p_before in before.items():
            p_after = after[uid]
            if p_before["bend_rpy"] != p_after["bend_rpy"]:
                diffs.append((p_before["bend_rpy"], p_after["bend_rpy"]))
            else:
                assert p_before["bend_offset"] == p_after["bend_offset"]
                assert p_before["module"] == p_after["module"]
                assert p_before["length"] == p_after["length"]
        assert len(diffs) == 1, f"expected exactly one changed Phalanx step, got {diffs}"
        old_rpy, new_rpy = diffs[0]
        n_component_diffs = sum(1 for a, b in zip(old_rpy, new_rpy) if abs(a - b) > 1e-12)
        assert n_component_diffs == 1, f"expected exactly one changed rpy component: {old_rpy} -> {new_rpy}"
        changed_any = True
    assert changed_any, "step_bend_rpy never produced a change over 50 attempts"


def test_step_bend_offset_changes_one_component():
    rng = np.random.default_rng(54321)
    changed_any = False
    for _ in range(50):
        derivation = sample_derivation(int(rng.integers(0, 1_000_000)), G_BEND)
        before = _phalanx_params_by_uid(derivation)
        try:
            after_derivation = vary(derivation, rng, dist=G_BEND, operator="step_bend_offset")
        except Exception:
            continue
        after = _phalanx_params_by_uid(after_derivation)
        diffs = []
        for uid, p_before in before.items():
            p_after = after[uid]
            if p_before["bend_offset"] != p_after["bend_offset"]:
                diffs.append((p_before["bend_offset"], p_after["bend_offset"]))
            else:
                assert p_before["bend_rpy"] == p_after["bend_rpy"]
        assert len(diffs) == 1, f"expected exactly one changed Phalanx step, got {diffs}"
        old_off, new_off = diffs[0]
        n_component_diffs = sum(1 for a, b in zip(old_off, new_off) if abs(a - b) > 1e-12)
        assert n_component_diffs == 1, f"expected exactly one changed offset component: {old_off} -> {new_off}"
        changed_any = True
    assert changed_any, "step_bend_offset never produced a change over 50 attempts"


# ---------------------------------------------------------------------------
# 4. Continuous limits sample inside range with lo < hi; step_limits keeps
#    lo < hi.
# ---------------------------------------------------------------------------


def test_continuous_limits_inside_range_and_lo_lt_hi():
    lo_deg, hi_deg = G_CONT.revolute_limit_range_deg
    lo_rad, hi_rad = lo_deg * DEG, hi_deg * DEG
    for seed in range(30):
        derivation = sample_derivation(seed, G_CONT)
        for s in derivation.steps:
            if s.production != "Phalanx" or s.params["module"]["kind"] != "R":
                continue
            lo, hi = s.params["module"]["limits"]
            assert lo < hi, f"seed={seed}: lo={lo} hi={hi}"
            assert lo_rad - 1e-9 <= lo <= hi_rad + 1e-9
            assert lo_rad - 1e-9 <= hi <= hi_rad + 1e-9


def test_step_limits_continuous_keeps_lo_lt_hi_within_range():
    lo_deg, hi_deg = G_CONT.revolute_limit_range_deg
    lo_rad, hi_rad = lo_deg * DEG, hi_deg * DEG
    rng = np.random.default_rng(999)
    derivation = sample_derivation(1, G_CONT)
    for _ in range(200):
        try:
            derivation = vary(derivation, rng, dist=G_CONT, operator="step_limits")
        except Exception:
            continue
        for s in derivation.steps:
            if s.production != "Phalanx" or s.params["module"]["kind"] != "R":
                continue
            lo, hi = s.params["module"]["limits"]
            assert lo < hi
            assert lo_rad - 1e-9 <= lo <= hi_rad + 1e-9
            assert lo_rad - 1e-9 <= hi <= hi_rad + 1e-9


# ---------------------------------------------------------------------------
# 5. normalize_axis_sign invariance / hash equality.
# ---------------------------------------------------------------------------


def test_normalize_axis_sign_invariance():
    axis = (0.0, 1.0, 0.0)
    limits = (-1.2, 0.7)
    norm_a = normalize_axis_sign(axis, limits)
    norm_b = normalize_axis_sign((-axis[0], -axis[1], -axis[2]), (-limits[1], -limits[0]))
    assert norm_a[0] == norm_b[0]
    assert norm_a[1] == norm_b[1]
    # The chosen representative has a positive first-nonzero axis component.
    assert norm_a[0][1] > 0.0


def _flipped_axis_model(flip: bool) -> KinematicModel:
    axis = (0.0, 0.0, -1.0) if flip else (0.0, 0.0, 1.0)
    limits = (-1.8, 0.2) if flip else (-0.2, 1.8)
    bodies = (Body(name="root", palm=True), Body(name="d1p1"))
    joints = (
        Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1", axis=axis, limits=limits),
    )
    return KinematicModel(name="sign_test", root="root", bodies=bodies, joints=joints)


def test_normalize_axis_sign_canonical_hash_equality():
    model_a = _flipped_axis_model(flip=False)
    model_b = _flipped_axis_model(flip=True)
    assert phenotype_hash(model_a) == phenotype_hash(model_b)
    canon_a, canon_b = canonical_form(model_a), canonical_form(model_b)
    ja = canon_a.joints[0]
    jb = canon_b.joints[0]
    assert ja.axis == jb.axis
    assert ja.limits == jb.limits


# ---------------------------------------------------------------------------
# 6. Coverage: limits_continuous on Allegro; dependent-tighter containment.
# ---------------------------------------------------------------------------


def _flagged_identifier_count(items) -> int:
    """Each ``coverage`` item is ``"<category>:<comma-joined names>"`` (or a
    bare category with no colon) -- count individual flagged names, not
    list entries, since one relaxation can shrink a single comma-joined
    entry (e.g. from 16 joint names down to 1) without changing the outer
    list's own length."""
    total = 0
    for item in items:
        _, _, rest = item.partition(":")
        total += len(rest.split(",")) if rest else 1
    return total


def test_coverage_allegro_limits_continuous_relax_fewer_items():
    if not ALLEGRO_URDF.is_file():
        pytest.skip("local-only: Allegro fixture not present")
    model = load_urdf(ALLEGRO_URDF).model
    baseline = coverage(model, DEFAULT_DISTRIBUTION, relax=frozenset())
    relaxed = coverage(model, DEFAULT_DISTRIBUTION, relax=frozenset({"limits_continuous"}))
    n_base = _flagged_identifier_count(baseline.missing_constructs + baseline.out_of_support)
    n_relaxed = _flagged_identifier_count(relaxed.missing_constructs + relaxed.out_of_support)
    assert n_relaxed < n_base, f"baseline={n_base} relaxed={n_relaxed}"


def _dependent_tighter_model() -> KinematicModel:
    bodies = (Body(name="root", palm=True), Body(name="d1p1"), Body(name="d1p2"))
    joints = (
        Joint(name="d1p1_j", type="revolute", parent="root", child="d1p1",
              axis=(1.0, 0.0, 0.0), limits=(-1.0, 1.0)),
        Joint(name="d1p2_j", type="revolute", parent="d1p1", child="d1p2",
              axis=(1.0, 0.0, 0.0), limits=(-0.5, 0.5)),
    )
    couplings = (AffineCoupling(dependent="d1p2_j", source="d1p1_j", multiplier=1.0, offset=0.0),)
    return KinematicModel(name="dependent_tighter_test", root="root", bodies=bodies, joints=joints,
                           couplings=couplings)


def test_coverage_dependent_tighter_limits_not_flagged_outside_image():
    model = _dependent_tighter_model()
    result = coverage(model, DEFAULT_DISTRIBUTION)
    assert not any(item.startswith("coupling_limits_outside_image") for item in result.missing_constructs), (
        result.missing_constructs
    )
    assert any(note == "dependent_limits_tighter:d1p2_j" for note in result.notes), result.notes
