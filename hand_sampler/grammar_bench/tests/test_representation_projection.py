"""Representation-check plan, items 1 and 2.

Item 1 (below, ``test_derive_*``): ``derive()`` is the representation
compiler, not the grammar -- it must accept structurally valid but
off-distribution values a real hand's projection needs (a mount point off a
palm body's own segment line, phalanx length 0, ``mount_frac`` outside
``[0, 1]``, more palm bodies than any ``Distribution`` samples, off-grid
axes/limits/lengths) with no notion of "in range" at all (see ``rules.py``'s
updated module docstring). ``test_e0_infra.py``'s own
``REPLAY_HASHES_SEED_0_99`` already covers "replay hashes unchanged" for the
zero-default case; it is not duplicated here.

Item 2: ``project_to_derivation`` projects a
real hand's ``KinematicModel`` into a ``Derivation``; ``derive()`` of that
``Derivation`` must reproduce the original model's forward kinematics to
numerical precision (the projection is an exact gauge change, not an
approximation -- see ``adapters/projection.py``'s module docstring).

For each hand: project, derive, then compare forward kinematics between the
original model and the derived model over zero plus 16 random configurations
(every ORIGINAL movable, non-mimic joint sampled independently within its
own declared limits; mimic/coupled joints take their driven value). Checks
(via ``name_map``, with ``root_transform`` applied to bring the derived
model's FK into the original root frame):

- every mapped joint's position matches within 1e-6 m;
- every mapped joint's world-frame axis direction matches within 1e-6 rad
  (angle between the two unit vectors);
- every digit's fingertip position matches within 1e-6 m;
- the movable-joint count is conserved.

The plan's acceptance bar is 5 mm / 10 degrees; since the projection is
exact by construction, this file holds it to a much tighter, numerical-
precision bar instead (see the plan's own note on this).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

from hand_sampler.grammar.adapters.projection import ProjectionFailure, project_to_derivation
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.derive import Derivation, DerivationStep, derive, validate_derivation
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.kinematics import Body, Joint, KinematicModel, MOVABLE_TYPES, Pose
from hand_sampler.grammar.rules import GRAMMAR_VERSION


# --------------------------------------------------------------------------
# Item 1: derive() as a pure representation compiler (no range checks)
# --------------------------------------------------------------------------


def _minimal_derivation(
    *, palm_body_count: int = 0, palm_overrides: Optional[Dict[str, object]] = None,
    phalanx_length: float = 0.02, phalanx_axis=(0.0, 1.0, 0.0), phalanx_limits=(-0.3, 0.9),
    mount_frac: float = 0.5,
) -> Derivation:
    """One root + ``palm_body_count`` chained palm bodies (all fixed, no
    joint) + one digit of one revolute phalanx mounted on the last palm body
    (or root if none). ``palm_overrides`` is merged into the LAST palm
    body's params (e.g. to set a nonzero ``mount_offset``)."""
    steps = [DerivationStep(path="hand", production="Hand", params={
        "digit_count": 1, "palm_body_count": palm_body_count, "root_length": 0.05,
        "capsule_radius_m": 0.01,
    })]
    palm_names = []
    for i in range(palm_body_count):
        parent = "root" if i == 0 else palm_names[-1]
        p = {"name": f"palm{i}", "parent": parent, "mount_frac": 0.5, "length": 0.03,
             "direction_rpy": (0.0, 0.0, 0.0), "has_joint": False, "axis": (1.0, 0.0, 0.0),
             "limits": None, "uid": i}
        if palm_overrides and i == palm_body_count - 1:
            p.update(palm_overrides)
        steps.append(DerivationStep(path=f"palm/{i}", production="PalmBody", params=p))
        palm_names.append(f"palm{i}")
    mount = palm_names[-1] if palm_names else "root"
    steps.append(DerivationStep(path="digit/1", production="Digit", params={
        "digit_id": "1", "mount": mount, "mount_frac": mount_frac, "mount_rpy": (0.0, 0.0, 0.0),
        "phalanx_count": 1, "top_level": True, "depth": 0, "uid": 1000,
    }))
    steps.append(DerivationStep(path="digit/1/phalanx/0", production="Phalanx", params={
        "digit_id": "1", "p": 0, "module": {"kind": "R", "axis": phalanx_axis, "limits": phalanx_limits},
        "length": phalanx_length, "branch_digit_count": 0, "uid": 1001,
        "bend_rpy": (0.0, 0.0, 0.0), "bend_offset": (0.0, 0.0),
    }))
    return Derivation(seed=-1, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))


def test_derive_accepts_zero_length_phalanx() -> None:
    d = _minimal_derivation(phalanx_length=0.0)
    assert not validate_derivation(d)
    model = derive(d)
    T = forward_kinematics(model, {})
    tip = T["d1p1_tip"][:3, 3]
    joint = T["d1p1"][:3, 3]
    assert np.linalg.norm(tip - joint) < 1e-12


@pytest.mark.parametrize("mount_frac", [-0.3, 1.7])
def test_derive_accepts_mount_frac_outside_unit_interval(mount_frac: float) -> None:
    d = _minimal_derivation(palm_body_count=1, mount_frac=mount_frac)
    assert not validate_derivation(d)
    model = derive(d)
    T = forward_kinematics(model, {})
    # mount is on "palm0" (length 0.03, itself sitting at z = 0.5 * 0.05 =
    # 0.025 on the root): joint position along palm0's own z-axis at exactly
    # palm0_z + mount_frac * length, even outside [0, 1].
    palm0_z = 0.5 * 0.05
    expected_z = palm0_z + mount_frac * 0.03
    actual_z = T["d1p1"][2, 3]
    assert abs(actual_z - expected_z) < 1e-12


def test_derive_accepts_seven_palm_bodies() -> None:
    d = _minimal_derivation(palm_body_count=7)
    assert not validate_derivation(d)
    model = derive(d)
    palm_bodies = [b for b in model.bodies if b.palm]
    assert len(palm_bodies) == 8  # root + 7
    from hand_sampler.grammar.kinematics import validate
    validate(model)  # re-checks palm connectivity explicitly


def test_derive_accepts_off_grid_axis_limits_length() -> None:
    axis = (0.1234567, 0.4815162, math.sqrt(1 - 0.1234567**2 - 0.4815162**2))
    d = _minimal_derivation(phalanx_length=0.031415926, phalanx_axis=axis, phalanx_limits=(-0.123456, 0.987654))
    assert not validate_derivation(d)
    model = derive(d)
    j = next(jj for jj in model.joints if jj.name == "d1p1_j")
    assert j.limits == (-0.123456, 0.987654)
    assert tuple(j.axis) == axis
    T = forward_kinematics(model, {})
    length = float(np.linalg.norm(T["d1p1_tip"][:3, 3] - T["d1p1"][:3, 3]))
    assert abs(length - 0.031415926) < 1e-9


def test_derive_accepts_nonzero_palm_mount_offset() -> None:
    d = _minimal_derivation(palm_body_count=1, palm_overrides={"mount_offset": (0.011, -0.023)})
    assert not validate_derivation(d)
    model = derive(d)
    T = forward_kinematics(model, {})
    # root has no rotation, so palm0's joint origin in the world frame is
    # exactly (mount_offset_x, mount_offset_y, mount_frac * root_length).
    pos = T["palm0"][:3, 3]
    expected = np.array([0.011, -0.023, 0.5 * 0.05])
    assert np.linalg.norm(pos - expected) < 1e-12


def test_derive_palm_mount_offset_default_is_byte_identical() -> None:
    """Omitting ``mount_offset`` entirely (an old-style params dict) and
    passing the explicit zero default must derive to the identical model."""
    from hand_sampler.grammar.canonical import phenotype_hash

    d_explicit = _minimal_derivation(palm_body_count=1, palm_overrides={"mount_offset": (0.0, 0.0)})
    steps = list(d_explicit.steps)
    palm_idx = next(i for i, s in enumerate(steps) if s.production == "PalmBody")
    old_params = dict(steps[palm_idx].params)
    del old_params["mount_offset"]
    steps[palm_idx] = DerivationStep(path=steps[palm_idx].path, production="PalmBody", params=old_params)
    d_omitted = Derivation(seed=-1, grammar_version=GRAMMAR_VERSION, steps=tuple(steps))

    h_explicit = phenotype_hash(derive(d_explicit))
    h_omitted = phenotype_hash(derive(d_omitted))
    assert h_explicit == h_omitted

BENCH_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BENCH_DIR.parent.parent
MANIFEST = json.loads((BENCH_DIR / "manifest.json").read_text())

POS_TOL_M = 1e-6
AXIS_TOL_RAD = 1e-6
N_RANDOM_CONFIGS = 16


def _manifest_hand(hand_id: str) -> dict:
    return next(h for h in MANIFEST["hands"] if h["id"] == hand_id)


def _local_only_path(hand_id: str) -> Optional[Path]:
    """Resolve a ``commit_allowed=false`` hand via ``manifest.json``'s
    ``source_root`` + ``source_path``, mirroring ``evaluate._resolve_hand``.
    Returns ``None`` if unavailable on this machine."""
    hand = _manifest_hand(hand_id)
    source_root = MANIFEST.get("source_root")
    source_path = hand.get("source_path")
    if not source_root or not source_path:
        return None
    p = Path(source_root) / source_path
    return p if p.is_file() else None


# (case name, urdf path, hand_root, palm_joints, tip_frames)
CasesT = Tuple[str, Path, Optional[str], Sequence[str], Dict[str, str]]

FIXTURE_CASES: list = [
    ("coupled_finger", BENCH_DIR / "fixtures/analytic/coupled_finger.urdf", None, (), {}),
    ("allegro_right", BENCH_DIR / "fixtures/real/allegro_right/allegro_hand_description_right.urdf", None, (), {}),
    ("leap_right", BENCH_DIR / "fixtures/real/leap_right/leap_hand_right.urdf", None, (), {}),
    ("barrett_bh", BENCH_DIR / "fixtures/real/barrett_bh/bhand_model.urdf", None, (), {}),
    ("dclaw", BENCH_DIR / "fixtures/real/dclaw/dclaw_gripper.urdf", None, (), {}),
    ("wuji_right", BENCH_DIR / "fixtures/real/wuji_right/right.urdf", None, (), {}),
    ("xhand_right", BENCH_DIR / "fixtures/real/xhand_right/xhand_right.urdf", None, (), {}),
    ("tesollo_dg5f_right", BENCH_DIR / "fixtures/real/tesollo_dg5f_right/dg5f_right.urdf", None, (), {}),
    # I22 wrists audit: hand_root must be the palm (right_palm), not None --
    # None let load_urdf auto-pick right_root, which kept the movable
    # right_wrist joint above the palm (see manifest.json's orca_right note).
    ("orca_right", BENCH_DIR / "fixtures/real/orca_right/orcahand_right.urdf", "right_palm", (), {}),
    (
        "sharpa_left_on_iiwa14",
        REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
        "left_hand_C_MC",
        (),
        {},
    ),
]

LOCAL_ONLY_IDS = ["ability_right", "inspire_right"]


def _resolve_local_only(hand_id: str) -> CasesT:
    p = _local_only_path(hand_id)
    if p is None:
        pytest.skip(f"local-only:{hand_id} source not present at manifest source_root on this machine")
    return (hand_id, p, None, (), {})


ALL_CASE_IDS = [c[0] for c in FIXTURE_CASES] + LOCAL_ONLY_IDS


def _get_case(name: str) -> CasesT:
    for c in FIXTURE_CASES:
        if c[0] == name:
            return c
    return _resolve_local_only(name)


def _rotation_angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cos_t = float(np.dot(a, b) / (na * nb))
    cos_t = max(-1.0, min(1.0, cos_t))
    return float(math.acos(cos_t))


def _check_hand(name: str, path: Path, hand_root: Optional[str], palm_joints, tip_frames) -> None:
    if not path.is_file():
        pytest.skip(f"local-only:{name} fixture not present on this machine: {path}")

    result = load_urdf(path, hand_root=hand_root)
    model = result.model
    n_movable_orig = sum(1 for j in model.joints if j.type in MOVABLE_TYPES)

    pr = project_to_derivation(model, palm_joints=palm_joints, tip_frames=tip_frames)
    issues = validate_derivation(pr.derivation)
    assert not issues, f"{name}: validate_derivation issues: {issues}"

    derived = derive(pr.derivation)
    n_movable_der = sum(1 for j in derived.joints if j.type in MOVABLE_TYPES)
    assert n_movable_der == n_movable_orig, (
        f"{name}: movable joint count not conserved: orig={n_movable_orig} derived={n_movable_der}"
    )

    dependents = {c.dependent for c in model.couplings}
    independent_joints = [j for j in model.joints if j.type in MOVABLE_TYPES and j.name not in dependents]
    derived_joint_by_name = {j.name: j for j in derived.joints}

    rng = np.random.default_rng(20260926)
    max_pos = 0.0
    max_axis = 0.0
    max_tip = 0.0

    for trial in range(N_RANDOM_CONFIGS + 1):
        q_orig: Dict[str, float] = {}
        for j in independent_joints:
            if trial == 0:
                q_orig[j.name] = 0.0
                continue
            if j.type == "continuous":
                q_orig[j.name] = float(rng.uniform(-math.pi, math.pi))
            else:
                lo, hi = j.limits
                q_orig[j.name] = float(rng.uniform(lo, hi))

        T_orig = forward_kinematics(model, q_orig)
        q_der = {pr.name_map[jn]: v for jn, v in q_orig.items() if jn in pr.name_map}
        T_der = forward_kinematics(derived, q_der)
        RT = pr.root_transform

        for j in model.joints:
            djn = pr.name_map.get(j.name)
            if djn is None:
                continue
            derived_body = djn[:-2]  # joint "<body>_j" -> body "<body>" (derive.py's own convention)
            if derived_body not in T_der:
                continue
            T_o = T_orig[j.child]
            T_d = RT @ T_der[derived_body]
            max_pos = max(max_pos, float(np.linalg.norm(T_o[:3, 3] - T_d[:3, 3])))

            dj = derived_joint_by_name[djn]
            axis_o = T_o[:3, :3] @ np.asarray(j.axis, dtype=float)
            axis_d = T_d[:3, :3] @ np.asarray(dj.axis, dtype=float)
            max_axis = max(max_axis, _rotation_angle_between(axis_o, axis_d))

        # Fingertips: every original body named as a fingertip target in
        # ``name_map`` (mapped to a derived "..._tip" FRAME, not a body).
        for orig_body, derived_name in pr.name_map.items():
            if not derived_name.endswith("_tip"):
                continue
            if orig_body not in T_orig:
                continue
            derived_frames = {f.name: f for f in derived.frames}
            if derived_name not in derived_frames:
                continue
            fr = derived_frames[derived_name]
            T_body_der = T_der[fr.body]
            from hand_sampler.grammar.fk import pose_to_matrix
            T_tip_der = T_body_der @ pose_to_matrix(fr.pose)
            T_tip_der_root = RT @ T_tip_der
            max_tip = max(max_tip, float(np.linalg.norm(T_orig[orig_body][:3, 3] - T_tip_der_root[:3, 3])))

    print(
        f"[{name}] max_pos_m={max_pos:.3e} max_axis_rad={max_axis:.3e} max_tip_m={max_tip:.3e} "
        f"merged_fixed={len(pr.report['merged_fixed_joints'])} palm_bodies={pr.report['palm_bodies']} "
        f"fingertip_undefined={pr.report['fingertip_undefined']} "
        f"coupling_as_independent={pr.report['coupling_as_independent']}"
    )
    assert max_pos <= POS_TOL_M, f"{name}: max joint position error {max_pos} m exceeds {POS_TOL_M} m"
    assert max_axis <= AXIS_TOL_RAD, f"{name}: max joint axis error {max_axis} rad exceeds {AXIS_TOL_RAD} rad"
    assert max_tip <= POS_TOL_M, f"{name}: max fingertip error {max_tip} m exceeds {POS_TOL_M} m"


@pytest.mark.parametrize("case_name", ALL_CASE_IDS)
def test_projection_reproduces_fk(case_name: str) -> None:
    name, path, hand_root, palm_joints, tip_frames = _get_case(case_name)
    _check_hand(name, path, hand_root, palm_joints, tip_frames)


def test_projection_reports_fingertip_undefined_for_coincident_leaf() -> None:
    """barrett_bh's finger tip links coincide with their own last joint (no
    fixed offset beyond it), so the projection must flag this explicitly
    rather than silently emitting a zero-length phalanx with no note."""
    path = BENCH_DIR / "fixtures/real/barrett_bh/bhand_model.urdf"
    if not path.is_file():
        pytest.skip("local-only:barrett_bh fixture not present")
    model = load_urdf(path).model
    pr = project_to_derivation(model)
    assert pr.report["fingertip_undefined"], "expected at least one fingertip_undefined note for barrett_bh"


def test_projection_coupled_finger_couplings() -> None:
    """I22 decision (one motor per joint): the analytic ``coupled_finger``
    fixture's two mimic joints (``joint2``, ``joint3``, both sourced off the
    digit's own first, revolute phalanx ``joint1``) must be emitted as
    independent revolute modules with their OWN declared limits -- never a
    ``Coupled`` module, even though an earlier revolute source is available
    in the same digit (this is a deliberate contract change from the
    original representation-check plan item 2, which asked for ``Coupled``
    here; the project owner's I22 decision overrides it: coupled/mimic
    joints are a future item, and the projection must never emit
    ``Coupled``)."""
    path = BENCH_DIR / "fixtures/analytic/coupled_finger.urdf"
    model = load_urdf(path).model
    pr = project_to_derivation(model)
    derived = derive(pr.derivation)
    assert len(derived.couplings) == 0
    assert set(pr.report["coupling_as_independent"]) == {"joint2", "joint3"}
    d1p2_j = next(j for j in derived.joints if j.name == "d1p2_j")
    d1p3_j = next(j for j in derived.joints if j.name == "d1p3_j")
    assert d1p2_j.limits == (-1.4, 1.6)  # joint2's own declared limits, unchanged
    assert d1p3_j.limits == (-0.5, 0.8)  # joint3's own declared limits, unchanged


# --------------------------------------------------------------------------
# I22 fix 3: an un-annotated in-digit branch raises ProjectionFailure
# (branch_not_projected), live now that the old ">= 2 leaves" auto-include
# rule is gone.
# --------------------------------------------------------------------------


def _synthetic_branchy_model() -> KinematicModel:
    """root -[joint0]-> mid -[joint1]-> leaf1
                              -[joint2]-> leaf2
    ``mid`` has 2 movable children and is not the root and not named by any
    ``palm_joints`` entry -- a genuine in-digit branch this schema cannot
    project without an explicit annotation."""
    bodies = (Body(name="root"), Body(name="mid"), Body(name="leaf1"), Body(name="leaf2"))
    joints = (
        Joint(name="joint0", type="revolute", parent="root", child="mid",
              origin=Pose(xyz=(0.0, 0.0, 0.05)), axis=(0.0, 1.0, 0.0), limits=(-1.0, 1.0)),
        Joint(name="joint1", type="revolute", parent="mid", child="leaf1",
              origin=Pose(xyz=(0.01, 0.0, 0.02)), axis=(0.0, 1.0, 0.0), limits=(-1.0, 1.0)),
        Joint(name="joint2", type="revolute", parent="mid", child="leaf2",
              origin=Pose(xyz=(-0.01, 0.0, 0.02)), axis=(0.0, 1.0, 0.0), limits=(-1.0, 1.0)),
    )
    return KinematicModel(name="synthetic_branchy", root="root", bodies=bodies, joints=joints)


def test_unannotated_in_digit_branch_raises_branch_not_projected() -> None:
    model = _synthetic_branchy_model()
    with pytest.raises(ProjectionFailure) as excinfo:
        project_to_derivation(model)
    assert excinfo.value.report["branch_not_projected"] == ["mid"]
    assert "branch_not_projected" in str(excinfo.value)


def test_annotating_the_branch_joint_as_palm_joints_fixes_it() -> None:
    """Naming ``joint0`` (whose child ``mid`` is the branch point) in
    ``palm_joints`` folds ``mid`` into the palm set, exactly the same fix
    the manifest audit applied to the real hands that needed it (SVH's
    ``right_hand_j5``, Shadow's ``rh_LFJ5``, ARMS's ``CMC4``/``CMC5``)."""
    model = _synthetic_branchy_model()
    pr = project_to_derivation(model, palm_joints=["joint0"])
    assert not pr.report["branch_not_projected"]
    assert "mid" in pr.report["palm_bodies"] or "palm0" in pr.report["palm_bodies"]
    derived = derive(pr.derivation)
    assert sum(1 for j in derived.joints if j.type in MOVABLE_TYPES) == 3
