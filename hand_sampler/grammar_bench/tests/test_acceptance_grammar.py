"""Iteration-3 acceptance tests for the hand-kinematics grammar
(``hand_sampler/grammar/rules.py`` + ``distributions.py`` + ``derive.py``):

  1. Structural validity, mobility/admissible-box sanity, model+derivation
     JSON round trips, and replay determinism, over 500 seeds.
  2. URDF export/reimport (parses, kinematically equivalent) over 200 seeds,
     plus an independent Pinocchio oracle check on the exported files.
  3. A support audit over 1000 seeds proving the grammar actually reaches
     every construct the design calls out (branches, palm joints, oblique
     axes, couplings with negative multiplier/nonzero offset, ...).
  4. ``vary``: every operator, applied over 100 seeds, either yields a valid,
     different derivation or (``remove_digit`` at 1 digit only) raises
     ``VariationImpossible``.

Convention note (URDF re-import): URDF has no concept of ``Body.palm`` /
``Body.radius``, and the ``"<body>_tip"`` ``Frame`` this grammar attaches to
every segment-producing body (carrying its length/direction for the later
geometry step, per the design note) is exported as an *extra* fixed-joint
link/joint pair that reimports as an ordinary body -- both losses are
inherent to the URDF format itself (see ``adapters/urdf.py``'s own
docstrings) and are not grammar bugs. So "re-import equals" is checked here
as: every *original* joint's type/parent/child/axis/limits/origin and every
coupling round-trip exactly, and forward kinematics for every *original*
body is identical (well under any float-noise tolerance) between the
pre-export model and the reimported one -- i.e. kinematic equivalence,
which is what to_urdf/load_urdf actually promise.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from urdf_parser_py.urdf import URDF

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.json_io import from_json, to_json
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.coords import (
    admissible_box,
    independent_joints,
    movable_joints,
    q_from_u,
    sample_configurations,
)
from hand_sampler.grammar.derive import (
    OPERATORS,
    VariationImpossible,
    derivation_from_json,
    derivation_to_json,
    derive,
    generate,
    sample_derivation,
    segment,
    vary,
)
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.kinematics import validate
from hand_sampler.grammar_bench.tolerances import ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent.parent
ORACLE_PYTHON = Path("/home/singularity/anaconda3/envs/piper/bin/python")
ORACLE_FK_SCRIPT = BENCH_DIR / "refgen" / "oracle_fk.py"

# Same-code FK comparison (our own forward_kinematics on both sides of a
# URDF export/reimport round trip): error should be at float-precision, far
# tighter than the independent-oracle tolerance.
REIMPORT_POS_M = 1e-9
REIMPORT_ROT_RAD = 1e-9

N_STRUCTURAL_SEEDS = 500
N_URDF_SEEDS = 200
N_AUDIT_SEEDS = 1000
N_VARY_SEEDS = 100


def _oracle_available() -> bool:
    return ORACLE_PYTHON.is_file() and ORACLE_FK_SCRIPT.is_file()


# ---------------------------------------------------------------------------
# 1. Structural validity / mobility / admissible box / round trips / replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(N_STRUCTURAL_SEEDS))
def test_grammar_structural_validity(seed):
    derivation, model = generate(seed)

    validate(model)  # raises ModelError on any failure

    indep = independent_joints(model)
    mov = movable_joints(model)
    assert len(indep) == len(mov) - len(model.couplings)

    box, _conflicts = admissible_box(model)
    assert box, "admissible box must be non-empty"
    assert set(box) == set(indep)
    for name, (lo, hi) in box.items():
        assert lo <= hi, f"seed={seed} joint={name} inverted box ({lo}, {hi})"

    s = to_json(model)
    assert from_json(s) == model
    assert to_json(from_json(s)) == s

    dj = derivation_to_json(derivation)
    assert derivation_from_json(dj) == derivation

    d2 = sample_derivation(seed, DEFAULT_DISTRIBUTION)
    assert d2 == derivation
    assert derive(d2) == model

    palm_names = {b.name for b in model.bodies if b.palm}
    assert model.root in palm_names
    parent_of = {j.child: j.parent for j in model.joints}
    for name in palm_names - {model.root}:
        cur = name
        seen = set()
        while cur != model.root:
            assert cur not in seen
            seen.add(cur)
            cur = parent_of[cur]
            assert cur in palm_names or cur == model.root, (
                f"seed={seed} palm body {name!r} does not connect to root through palm bodies"
            )

    child_joint_count: dict = {}
    for j in model.joints:
        child_joint_count[j.parent] = child_joint_count.get(j.parent, 0) + 1

    for step in derivation.steps:
        if step.production != "Digit":
            continue
        if step.params["top_level"]:
            assert step.params["mount"] in palm_names, (
                f"seed={seed} digit {step.params['digit_id']} mounts on non-palm body {step.params['mount']!r}"
            )
        else:
            # A branch digit measures what the grammar claims: its mount is
            # the host phalanx's own body (never a palm body), and that body
            # must genuinely have >= 2 child joints (its own next phalanx,
            # if any, plus one joint per branch digit mounted on it).
            mount = step.params["mount"]
            assert mount not in palm_names, (
                f"seed={seed} branch digit {step.params['digit_id']} mounts on palm body {mount!r}"
            )
            assert child_joint_count.get(mount, 0) >= 2, (
                f"seed={seed} branch digit {step.params['digit_id']} mount {mount!r} "
                f"has {child_joint_count.get(mount, 0)} child joints, expected >= 2"
            )


# ---------------------------------------------------------------------------
# 1b. Iteration-5 geometric convention: every mount sits ON its host's own
# segment (never off it, on a sphere around the host's origin), every body
# -- the root included -- owns a real segment, and ``derive.segment()``
# agrees with ``forward_kinematics`` on the "<body>_tip" frames it reads.
# ---------------------------------------------------------------------------

N_GEOMETRY_SEEDS = 500


@pytest.mark.parametrize("seed", range(N_GEOMETRY_SEEDS))
def test_grammar_joint_origins_on_host_segment(seed):
    """Every joint's origin is ``(0, 0, t)`` in its parent's own frame, with
    ``0 <= t <= L`` where ``L`` is the parent body's own segment length (its
    "<body>_tip" frame's z value) -- the convention fixed in iteration 5
    (see rules.py's module docstring). This covers palm-to-palm mounts,
    digit/branch mounts onto a palm or phalanx body, and phalanx-to-phalanx
    continuations alike, since ``derive.py`` now builds all three the same
    way: ``Trans(0, 0, frac * L) * Rot(rpy)``."""
    _derivation, model = generate(seed)
    tip_len = {f.body: float(f.pose.xyz[2]) for f in model.frames if f.name == f"{f.body}_tip"}
    for j in model.joints:
        x, y, z = j.origin.xyz
        assert abs(x) < 1e-12 and abs(y) < 1e-12, (
            f"seed={seed} joint {j.name!r} origin {j.origin.xyz} is not purely along its parent's z-axis"
        )
        L = tip_len[j.parent]
        assert -1e-12 <= z <= L + 1e-12, (
            f"seed={seed} joint {j.name!r} origin t={z} lies outside host segment [0, {L}]"
        )


@pytest.mark.parametrize("seed", range(N_GEOMETRY_SEEDS))
def test_grammar_root_is_a_real_palm_segment(seed):
    """Every hand has a palm: the root is always a palm body with its own
    (> 0) segment, so the base of the tree is never a length-0 gap."""
    _derivation, model = generate(seed)
    palm_names = {b.name for b in model.bodies if b.palm}
    tip_len = {f.body: float(f.pose.xyz[2]) for f in model.frames if f.name == f"{f.body}_tip"}
    assert model.root in palm_names
    assert model.root in tip_len
    assert tip_len[model.root] > 0.0


@pytest.mark.parametrize("seed", range(N_GEOMETRY_SEEDS))
def test_grammar_palm_mount_point_on_parent_segment(seed):
    """A palm body's mount point (root frame, q=0) sits exactly at
    ``mount_frac`` along its parent's own segment, so the parent's segment
    is never left with an unowned stretch, and no two palm bodies share a
    piece of segment (a child's segment starts exactly where the parent's
    still continues, generally in a different direction). Palm-body-to-
    palm-body connectivity through the root is already checked by
    ``validate`` (called inside ``generate``/``derive``) on every seed
    here."""
    derivation, model = generate(seed)
    for step in derivation.steps:
        if step.production != "PalmBody":
            continue
        name = step.params["name"]
        parent = step.params["parent"]
        frac = step.params["mount_frac"]
        p_start, p_end = segment(model, parent)
        c_start, _c_end = segment(model, name)
        expected = tuple(ps + frac * (pe - ps) for ps, pe in zip(p_start, p_end))
        err = max(abs(a - b) for a, b in zip(c_start, expected))
        assert err < 1e-9, (
            f"seed={seed} palm body {name!r} mount point off parent {parent!r}'s segment (err={err:.3e})"
        )


def test_grammar_segment_helper_matches_forward_kinematics():
    """``derive.segment(model, body)`` is a q=0 forward-kinematics lookup of
    ``body``'s own origin and its "<body>_tip" frame; check it agrees with
    calling ``forward_kinematics`` directly, for every body of the first 50
    generated hands."""
    for seed in range(50):
        _derivation, model = generate(seed)
        transforms = fk.forward_kinematics(model, {})
        for b in model.bodies:
            start, end = segment(model, b.name)
            assert np.allclose(start, tuple(transforms[b.name][:3, 3]), atol=1e-12)
            assert np.allclose(end, tuple(transforms[f"{b.name}_tip"][:3, 3]), atol=1e-12)


# ---------------------------------------------------------------------------
# 2. URDF export / reimport + independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(N_URDF_SEEDS))
def test_grammar_urdf_export_reimport(seed):
    _derivation, model = generate(seed)
    text, _losses = to_urdf(model)
    parsed = URDF.from_xml_string(text.encode("utf-8"))
    assert parsed is not None

    reimported = load_urdf(text).model
    assert reimported.root == model.root
    assert reimported.couplings == model.couplings

    reimported_joints = {j.name: j for j in reimported.joints}
    for j in model.joints:
        rj = reimported_joints[j.name]
        assert rj.type == j.type
        assert rj.parent == j.parent
        assert rj.child == j.child
        assert rj.origin == j.origin
        assert rj.axis == j.axis
        assert rj.limits == j.limits

    configs = sample_configurations(model, 4, seed)
    orig_names = [b.name for b in model.bodies]
    for u in configs:
        q = q_from_u(model, u)
        ours = fk.forward_kinematics(model, q)
        reim = fk.forward_kinematics(reimported, q)
        for name in orig_names:
            pos_err = fk.position_error(ours[name], reim[name])
            rot_err = fk.rotation_error(ours[name], reim[name])
            assert pos_err <= REIMPORT_POS_M, f"seed={seed} body={name} pos_err={pos_err}"
            assert rot_err <= REIMPORT_ROT_RAD, f"seed={seed} body={name} rot_err={rot_err}"


def test_grammar_urdf_export_matches_oracle():
    if not _oracle_available():
        pytest.skip("oracle-unavailable: piper conda interpreter not found")

    n_compared = 0
    max_pos = 0.0
    max_rot = 0.0
    t0 = time.time()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for seed in range(N_URDF_SEEDS):
            model = derive(sample_derivation(seed, DEFAULT_DISTRIBUTION))
            text, _losses = to_urdf(model)
            configs = sample_configurations(model, 8, seed)

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
                for name in orig_names:
                    T_oracle = np.array(oracle_poses[name])
                    pos_err = fk.position_error(ours[name], T_oracle)
                    rot_err = fk.rotation_error(ours[name], T_oracle)
                    max_pos = max(max_pos, pos_err)
                    max_rot = max(max_rot, rot_err)
                    n_compared += 1
                    assert pos_err <= ORACLE_POS_M, f"seed={seed} body={name} pos_err={pos_err}"
                    assert rot_err <= ORACLE_ROT_RAD, f"seed={seed} body={name} rot_err={rot_err}"

            urdf_path.unlink()
            cfg_path.unlink()
            out_path.unlink()

    elapsed = time.time() - t0
    assert n_compared > 0
    print(
        f"grammar oracle check: {N_URDF_SEEDS} seeds, {n_compared} body-pose comparisons, "
        f"max_pos={max_pos:.3e}, max_rot={max_rot:.3e}, elapsed={elapsed:.1f}s"
    )


# ---------------------------------------------------------------------------
# 3. Support audit: every construct the design calls out must be reachable.
# ---------------------------------------------------------------------------


def test_grammar_support_audit():
    counts = {
        "one_digit": 0,
        "five_plus_digits": 0,
        "six_plus_phalanges": 0,
        "in_digit_branch": 0,
        "palm_tree": 0,
        "palm_joint": 0,
        "two_nonparallel_palm_joints": 0,
        "nonperpendicular_axis": 0,
        "coupling_nonzero_offset": 0,
        "coupling_negative_multiplier": 0,
        "asymmetric_limits": 0,
        "continuous_joint": 0,
        "prismatic_joint": 0,
        "nonidentity_mount_rotation": 0,
    }

    for seed in range(N_AUDIT_SEEDS):
        derivation = sample_derivation(seed, DEFAULT_DISTRIBUTION)
        model = derive(derivation)
        hand = next(s for s in derivation.steps if s.path == "hand").params
        if hand["digit_count"] == 1:
            counts["one_digit"] += 1
        if hand["digit_count"] >= 5:
            counts["five_plus_digits"] += 1

        # Structural honesty: measure the *derived model*, not a derivation-step
        # flag. in_digit_branch = some non-palm body has >= 2 child joints
        # (real in-digit branching). palm_tree = some palm body has >= 2
        # palm-body children (real palm fan-out, not a chain).
        palm_names = {b.name for b in model.bodies if b.palm}
        child_joint_count: dict = {}
        palm_child_count: dict = {}
        for j in model.joints:
            child_joint_count[j.parent] = child_joint_count.get(j.parent, 0) + 1
            if j.parent in palm_names and j.child in palm_names:
                palm_child_count[j.parent] = palm_child_count.get(j.parent, 0) + 1
        if any(name not in palm_names and n >= 2 for name, n in child_joint_count.items()):
            counts["in_digit_branch"] += 1
        if any(n >= 2 for n in palm_child_count.values()):
            counts["palm_tree"] += 1

        for s in derivation.steps:
            if s.production == "PalmBody" and s.params["has_joint"]:
                counts["palm_joint"] += 1
            if s.production == "Digit":
                rpy = s.params["mount_rpy"]
                if any(abs(v) > 1e-9 for v in rpy):
                    counts["nonidentity_mount_rotation"] += 1
                if s.params["phalanx_count"] >= 6:
                    counts["six_plus_phalanges"] += 1
            if s.production == "Phalanx":
                mod = s.params["module"]
                if mod["kind"] == "C":
                    counts["continuous_joint"] += 1
                if mod["kind"] == "P":
                    counts["prismatic_joint"] += 1
                if mod["kind"] in ("R", "P") and abs(mod["limits"][0] + mod["limits"][1]) > 1e-9:
                    counts["asymmetric_limits"] += 1
                if mod["kind"] == "Coupled":
                    if abs(mod["offset"]) > 1e-9:
                        counts["coupling_nonzero_offset"] += 1
                    if mod["multiplier"] < 0.0:
                        counts["coupling_negative_multiplier"] += 1

        # nonperpendicular_axis: measured on the DERIVED MODEL (fk.py's own
        # convention), not on the raw sampled axis against the world X/Y/Z
        # axes. A joint's frame equals its child body's own frame at q=0,
        # and every phalanx body's segment direction is (0, 0, 1) in that
        # same child frame (its "<body>_tip" frame is always (0, 0, length),
        # see rules.py) -- so the segment direction, expressed in the
        # joint's own frame, is exactly (0, 0, 1) already: no forward-
        # kinematics transform is needed, ``|axis . (0,0,1)|`` after
        # normalizing is just ``|axis_z|``. Counted once per model (not once
        # per phalanx encountered), matching every other key in this audit.
        if any(
            abs(float(j.axis[2])) / float(np.linalg.norm(j.axis)) > 1e-9
            for j in model.joints
            if j.child not in palm_names and float(np.linalg.norm(j.axis)) > 1e-12
        ):
            counts["nonperpendicular_axis"] += 1

        # two_nonparallel_palm_joints: a palm joint's axis is stored in its
        # own (local) frame, and different palm bodies can sit at different
        # orientations relative to the root (each PalmBody's direction_rpy
        # chains onto the previous one) -- so comparing raw stored axes
        # compares vectors expressed in different frames. Move every
        # palm-joint axis into the ROOT frame at q=0 via forward_kinematics
        # (a joint axis is a direction, so only the rotation part of its
        # child body's root-frame transform applies) before cross-producting
        # them. Counted once per model: stop at the first nonparallel pair.
        transforms0 = fk.forward_kinematics(model, {})
        palm_joint_root_axes = [
            transforms0[j.child][:3, :3] @ np.array(j.axis, dtype=float)
            for j in model.joints
            if j.child in palm_names and j.type != "fixed"
        ]
        found_nonparallel = False
        for i in range(len(palm_joint_root_axes)):
            for j2 in range(i + 1, len(palm_joint_root_axes)):
                cross_norm = float(np.linalg.norm(np.cross(palm_joint_root_axes[i], palm_joint_root_axes[j2])))
                if cross_norm > 1e-6:
                    found_nonparallel = True
                    break
            if found_nonparallel:
                break
        if found_nonparallel:
            counts["two_nonparallel_palm_joints"] += 1

    print(f"grammar support audit over {N_AUDIT_SEEDS} seeds: {counts}")
    for key, n in counts.items():
        assert n >= 1, f"support audit: never observed {key!r} in {N_AUDIT_SEEDS} seeds"


# ---------------------------------------------------------------------------
# 4. vary operators
# ---------------------------------------------------------------------------


def _digit_phalanx_counts(derivation):
    return [s.params["phalanx_count"] for s in derivation.steps if s.production == "Digit"]


def _assert_impossible_is_legitimate(operator, derivation):
    """``VariationImpossible`` is only acceptable when the operator's own
    precondition genuinely cannot be met for this derivation -- never as a
    stand-in for a bug. ``resample_parameter``/``perturb_parameter`` always
    have a step to act on (every derivation has >=1 digit with >=1
    phalanx), so they must never raise."""
    hand = next(s for s in derivation.steps if s.path == "hand").params
    counts = _digit_phalanx_counts(derivation)
    if operator == "remove_digit":
        assert hand["digit_count"] == 1, (
            f"remove_digit raised VariationImpossible but digit_count={hand['digit_count']} != 1"
        )
    elif operator == "add_digit":
        assert hand["digit_count"] >= DEFAULT_DISTRIBUTION.digit_count_range[1], (
            f"add_digit raised VariationImpossible but digit_count={hand['digit_count']} "
            f"< max {DEFAULT_DISTRIBUTION.digit_count_range[1]}"
        )
    elif operator == "delete_phalanx":
        # Mirrors derive.py's ``_op_delete_phalanx`` deletable-candidate
        # logic exactly: a phalanx that hosts a branch is refused outright,
        # and the digit's current-last phalanx is refused too when deleting
        # it would promote an under-branched (exactly 1 branch digit)
        # phalanx to "last". VariationImpossible is legitimate only when no
        # digit has any phalanx left that clears both checks.
        def _has_deletable(s):
            digit_id = s.params["digit_id"]
            phalanx_count = s.params["phalanx_count"]
            if phalanx_count <= 1:
                return False
            last_idx = phalanx_count - 1
            phalanx_by_p = {
                pp.params["p"]: pp for pp in derivation.steps
                if pp.production == "Phalanx" and pp.params["digit_id"] == digit_id
            }
            second_last = phalanx_by_p.get(last_idx - 1)
            last_deletion_safe = not (second_last is not None and second_last.params["branch_digit_count"] == 1)
            return any(
                pp.params["branch_digit_count"] == 0 and (p_idx != last_idx or last_deletion_safe)
                for p_idx, pp in phalanx_by_p.items()
            )

        assert not any(_has_deletable(s) for s in derivation.steps if s.production == "Digit"), (
            f"delete_phalanx raised VariationImpossible but some digit has a deletable phalanx: {counts}"
        )
    elif operator == "insert_phalanx":
        assert all(c >= DEFAULT_DISTRIBUTION.phalanx_count_range[1] for c in counts), (
            f"insert_phalanx raised VariationImpossible but some digit has room to grow: {counts}"
        )
    else:
        pytest.fail(f"operator {operator!r} raised VariationImpossible on a derivation with no legitimate reason")


@pytest.mark.parametrize("operator", OPERATORS)
def test_grammar_vary_operator(operator):
    n_applied = 0
    n_impossible = 0
    for seed in range(N_VARY_SEEDS):
        derivation = sample_derivation(seed, DEFAULT_DISTRIBUTION)
        rng = np.random.default_rng(1_000_000 + seed)
        try:
            varied = vary(derivation, rng, DEFAULT_DISTRIBUTION, operator=operator)
        except VariationImpossible:
            n_impossible += 1
            _assert_impossible_is_legitimate(operator, derivation)
            continue
        n_applied += 1
        assert varied != derivation
        model = derive(varied)  # must be a valid model (raises otherwise)
        validate(model)

    print(f"vary operator={operator!r}: applied={n_applied} impossible={n_impossible} of {N_VARY_SEEDS} seeds")
    assert n_applied > 0


def test_grammar_vary_replays_exactly_with_branch():
    """A derivation containing an in-digit branch must still replay exactly
    after ``vary``: ``derive`` is a pure function of the derivation, so
    applying any operator and deriving the result twice must agree, and the
    varied derivation must itself survive a JSON round trip unchanged."""
    branch_seed = None
    for seed in range(200):
        derivation = sample_derivation(seed, DEFAULT_DISTRIBUTION)
        if any(s.production == "Phalanx" and s.params["branch_digit_count"] > 0 for s in derivation.steps):
            branch_seed = seed
            break
    assert branch_seed is not None, "no seed with a branch found in the first 200 seeds"

    derivation = sample_derivation(branch_seed, DEFAULT_DISTRIBUTION)
    n_checked = 0
    for operator in OPERATORS:
        rng = np.random.default_rng(3_000_000 + branch_seed)
        try:
            varied = vary(derivation, rng, DEFAULT_DISTRIBUTION, operator=operator)
        except VariationImpossible:
            continue
        model_a = derive(varied)
        model_b = derive(varied)
        assert model_a == model_b, f"operator={operator} seed={branch_seed}: replay mismatch"
        assert derive(derivation_from_json(derivation_to_json(varied))) == model_a, (
            f"operator={operator} seed={branch_seed}: derivation JSON round trip changed the model"
        )
        n_checked += 1

    assert n_checked > 0, "no operator applied to the branch-containing derivation"
