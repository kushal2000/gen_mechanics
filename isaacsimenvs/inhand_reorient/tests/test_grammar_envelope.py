"""CPU tests for `isaacsimenvs/inhand_reorient/scene/grammar_envelope.py`
(design note section 5, items 1, 2, 3, 5, 6). Numpy + `hand_sampler` only,
but importing `isaacsimenvs` eagerly pulls in `gymnasium` (see
`test_urdf_cutter.py`'s docstring) -- run with the isaacsim venv:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_grammar_envelope.py -q
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hand_sampler import robot_param_constants as rpc
from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.variants import G_SERIAL

from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf

N_SEEDS = 200
REVOLUTE_ONLY = (("R", 1.0), ("C", 0.0), ("P", 0.0), ("Coupled", 0.0))

DEFAULT_R = replace(
    DEFAULT_DISTRIBUTION, module_probabilities=REVOLUTE_ONLY, digit_count_range=(1, 5),
    branch_probability=0.0, palm_body_count_range=(0, 2),
)
G_SERIAL_R = replace(G_SERIAL, module_probabilities=REVOLUTE_ONLY, digit_count_range=(1, 5))

VARIANTS = [("DEFAULT_R", DEFAULT_R), ("G_SERIAL_R", G_SERIAL_R)]

PROJECTED_HAND_IDS = [
    "allegro_right", "leap_right", "barrett_bh", "ability_right", "inspire_right", "dclaw",
    "wuji_right", "xhand_right", "tesollo_dg5f_right", "orca_right", "sharpa_left_on_iiwa14",
    "shadow_right_local", "svh_right", "arms_skel",
]


def _admitted_models(dist, n=N_SEEDS, seed0=0):
    out = []
    for seed in range(seed0, seed0 + n):
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        result = ge.admit(model)
        if result.ok:
            out.append((seed, model))
    return out


# --------------------------------------------------------------------------
# 1. FK on 200 designs per variant, plus every admissible projected hand.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,dist", VARIANTS)
def test_fk_matches_grammar_reference_sampled(name, dist):
    admitted = _admitted_models(dist)
    assert admitted, f"{name}: no designs admitted out of {N_SEEDS} seeds"
    max_err = 0.0
    rng = np.random.default_rng(12345)
    for seed, model in admitted:
        design = ge.canonicalize(model, source=f"{name}:{seed}")
        valid = design.slot_valid
        for trial in range(3):
            q = np.zeros(ge.N_SLOTS)
            if trial != 0:
                for idx in range(ge.N_SLOTS):
                    if valid[idx]:
                        lo, hi = design.slot_limits[idx]
                        q[idx] = rng.uniform(lo, hi)
            Ta = ge.authored_fk(design, q)
            Tr = ge.grammar_fk_reference(design, q)
            max_err = max(max_err, float(np.abs(Ta[valid][:, :3, 3] - Tr[valid][:, :3, 3]).max(initial=0.0)))
            max_err = max(max_err, float(np.abs(Ta[valid][:, :3, :3] - Tr[valid][:, :3, :3]).max(initial=0.0)))
    assert max_err < 1e-9, f"{name}: authored_fk vs grammar_fk_reference max error {max_err}"


def test_fk_matches_grammar_reference_projected():
    max_err = 0.0
    rng = np.random.default_rng(999)
    n_checked = 0
    for hand_id in PROJECTED_HAND_IDS:
        entry, status, reason = pf.projected_entry(hand_id)
        if status != "admitted":
            continue
        model = derive(pf.derivation_from_dict(entry.derivation_dict))
        design = ge.canonicalize(model, source=f"projected:{hand_id}")
        valid = design.slot_valid
        for trial in range(3):
            q = np.zeros(ge.N_SLOTS)
            if trial != 0:
                for idx in range(ge.N_SLOTS):
                    if valid[idx]:
                        lo, hi = design.slot_limits[idx]
                        q[idx] = rng.uniform(lo, hi)
            Ta = ge.authored_fk(design, q)
            Tr = ge.grammar_fk_reference(design, q)
            max_err = max(max_err, float(np.abs(Ta[valid][:, :3, 3] - Tr[valid][:, :3, 3]).max(initial=0.0)))
            max_err = max(max_err, float(np.abs(Ta[valid][:, :3, :3] - Tr[valid][:, :3, :3]).max(initial=0.0)))
        n_checked += 1
    assert n_checked >= 5, f"expected several admissible projected hands, only checked {n_checked}"
    assert max_err < 1e-9, f"projected hands: authored_fk vs grammar_fk_reference max error {max_err}"


# --------------------------------------------------------------------------
# 2. Admission cases -- every hand with palm joints, SVH included.
# --------------------------------------------------------------------------


def test_svh_admitted_with_a_follower_carrier():
    """SVH carries two fingers on one palm joint: the first is the leader
    (its carrier is the palm joint), the second a follower tied to it."""
    design = _project_and_canonicalize("svh_right")
    roles = ge.carrier_roles(design)
    assert roles.count(ge.LEADER) == 1 and roles.count(ge.FOLLOWER) == 1
    lead = ge.carrier_slot(roles.index(ge.LEADER))
    fol = ge.carrier_slot(roles.index(ge.FOLLOWER))
    assert design.slot_tie[fol] == lead and design.slot_valid[lead] and not design.slot_valid[fol]
    assert roles.index(ge.FOLLOWER) == roles.index(ge.LEADER) + 1      # grouped, leader first


@pytest.mark.parametrize("hand_id", ["sharpa_left_on_iiwa14", "shadow_right_local", "arms_skel", "svh_right"])
def test_articulated_palm_hands_admitted(hand_id):
    entry, status, reason = pf.projected_entry(hand_id)
    assert status == "admitted", f"{hand_id}: expected admitted, got {status} ({reason})"


@pytest.mark.parametrize("hand_id", PROJECTED_HAND_IDS)
def test_every_projected_hand_admitted(hand_id):
    entry, status, reason = pf.projected_entry(hand_id)
    if status == "unavailable":
        pytest.skip(reason)
    assert status == "admitted", f"{hand_id}: {status} ({reason})"


def test_admission_rates_in_a_sane_range():
    # `ge.admit`'s default args now include the rest-overlap AND
    # spawn-height gates (review items 2/4), not just the envelope-shape
    # check `_admitted_models` used to be measuring alone -- the faithful
    # filter (root capsule + pairs across ghost carriers, opus-review-
    # phase2.md item 4) rejects most structurally-fine designs on rest
    # overlap. Measured over 1500 seeds each: G_SERIAL 21.5%, DEFAULT 17.7%
    # (see the worker report); this smaller 200-seed sample is noisier, so
    # the bounds are generous around that, not a tight sanity check.
    for name, dist in VARIANTS:
        admitted = _admitted_models(dist)
        rate = len(admitted) / N_SEEDS
        assert 0.05 <= rate <= 0.6, f"{name}: admission rate {rate} outside a sane range"


# --------------------------------------------------------------------------
# 3. Table invariants.
# --------------------------------------------------------------------------


def _some_designs(n=25):
    out = []
    for seed, model in _admitted_models(DEFAULT_R, n=80)[:n]:
        out.append(ge.canonicalize(model, source=f"inv:{seed}"))
    return out


def test_envelope_topology_is_fixed_across_designs():
    designs = _some_designs()
    assert len(designs) >= 5
    for design in designs:
        assert design.slot_valid.shape == (ge.N_SLOTS,)
        assert design.slot_origin.shape == (ge.N_SLOTS, 4, 4)
        assert design.slot_axis.shape == (ge.N_SLOTS, 3)
        assert design.slot_limits.shape == (ge.N_SLOTS, 2)
    # Parent/child topology (SLOT_PARENT) never depends on the design: 6
    # finger slots, each a carrier on the root and a chain of 5 finger joints.
    assert ge.N_SLOTS == 36 and ge.N_FINGERS == 6 and ge.N_JOINTS_PER_FINGER == 5
    for f in range(ge.N_FINGERS):
        assert ge.SLOT_PARENT[ge.carrier_slot(f)] == ge.ROOT_SENTINEL
        assert ge.SLOT_PARENT[ge.finger_slot(f, 0)] == ge.carrier_slot(f)
        for d in range(1, ge.N_JOINTS_PER_FINGER):
            assert ge.SLOT_PARENT[ge.finger_slot(f, d)] == ge.finger_slot(f, d - 1)
        assert ge.SLOT_NAMES[ge.carrier_slot(f)] == f"f{f}_cj" and ge.SLOT_NAMES[ge.finger_slot(f, 4)] == f"f{f}_j4"
    assert len(set(ge.SLOT_NAMES)) == ge.N_SLOTS


def test_ghost_slots_have_old_sampler_convention():
    designs = _some_designs()
    for design in designs:
        for idx in range(ge.N_SLOTS):
            if not design.slot_valid[idx] and design.slot_tie[idx] < 0:
                lo, hi = design.slot_limits[idx]
                assert (lo, hi) == ge.GHOST_LIMITS
                assert design.slot_length[idx] == ge.GHOST_LENGTH_M


def test_token_boxes_shape_and_finiteness():
    for design in _some_designs(10):
        boxes = ge.token_boxes(design)
        assert boxes.shape == (ge.N_SLOTS, 4, 3)
        assert np.isfinite(boxes).all()


def test_build_population_table_shapes():
    designs = _some_designs(12)
    pop = ge.build_population(designs, n_sweep=10)
    n = len(designs)
    assert pop.joint_link_boxes.shape == (n, ge.N_SLOTS, 4, 3)
    assert pop.joint_valid.shape == (n, ge.N_SLOTS)
    assert pop.joint_tie.shape == (n, ge.N_SLOTS)
    assert pop.tip_offsets.shape == (n, ge.N_FINGERS)
    assert pop.joint_limits.shape == (n, ge.N_SLOTS, 2)
    assert pop.default_joint_pos.shape == (n, ge.N_SLOTS)
    assert pop.hand_scale.shape == (n,)
    assert pop.fingertip_valid.shape == (n, ge.N_FINGERS)
    assert pop.fingertip_offsets.shape == (n, ge.N_FINGERS, 3)
    assert pop.palm_center.shape == (n, 3)
    assert pop.palm_keypoints.shape == (n, 4, 3)
    assert pop.palm_frame.shape == (n, 7)
    assert pop.base_rot.shape == (n, 4)
    assert pop.spawn_offset.shape == (n, 3)
    assert not hasattr(pop, "drive"), "GrammarPopulation.drive was removed (review item 11: unused, inconsistent)"
    # A design with limits containing 0 must exist somewhere in a real
    # population, but this is a per-generator invariant (make_grammar_
    # population.py orders env 0), not a per-table one; here we only check
    # every valid joint has SOME finite, ordered limit.
    valid = pop.joint_valid
    lo, hi = pop.joint_limits[..., 0], pop.joint_limits[..., 1]
    assert np.isfinite(pop.joint_limits[valid]).all()
    assert (lo[valid] <= hi[valid]).all()


# --------------------------------------------------------------------------
# 5. Mass positive (semi-)definite.
# --------------------------------------------------------------------------


def test_mass_props_positive():
    for design in _some_designs(15):
        props = ge.mass_props(design)
        assert props, "mass_props returned no bodies"
        for name, (mass, com, inertia) in props.items():
            assert mass > 0.0, f"{name}: non-positive mass {mass}"
            assert np.isfinite(com).all()
            assert np.isfinite(inertia).all()
            # Symmetric, positive semi-definite (up to float noise).
            assert np.allclose(inertia, inertia.T, atol=1e-12)
            eigvals = np.linalg.eigvalsh(inertia)
            assert eigvals.min() > -1e-12, f"{name}: inertia eigenvalues {eigvals}"


# --------------------------------------------------------------------------
# 6. palm_up sanity: DClaw +z, Allegro about +x.
# --------------------------------------------------------------------------


def _project_and_canonicalize(hand_id):
    entry, status, reason = pf.projected_entry(hand_id)
    assert status == "admitted", f"{hand_id}: {status} ({reason})"
    model = derive(pf.derivation_from_dict(entry.derivation_dict))
    return ge.canonicalize(model, source=f"projected:{hand_id}")


def test_palm_up_dclaw_normal_about_plus_z():
    design = _project_and_canonicalize("dclaw")
    result = ge.palm_up(design, n_sweep=20)
    # "about +z" -- dominant component is z, and it's positive.
    dominant_axis = int(np.argmax(np.abs(result.normal)))
    assert dominant_axis == 2
    assert result.normal[2] > 0


def test_palm_up_allegro_normal_about_plus_x():
    design = _project_and_canonicalize("allegro_right")
    result = ge.palm_up(design, n_sweep=20)
    dominant_axis = int(np.argmax(np.abs(result.normal)))
    assert dominant_axis == 0
    assert result.normal[0] > 0


# --------------------------------------------------------------------------
# Review item 1: fingertip ghost-slot translation.
# --------------------------------------------------------------------------


def test_ghost_tip_marker_sits_at_the_real_last_link_tip():
    """The first ghost slot after a finger's real chain must be translated
    by (0,0,L_last) so it (and the whole identity-chained ghost tail after
    it, including whichever slot every population/observation caller reads
    as "the fingertip body") sits exactly at the real last link's TIP, not
    its base."""
    checked = 0
    designs = [ge.canonicalize(m, source=f"probe:{s}") for s, m in _admitted_models(DEFAULT_R, n=300)]
    for design in designs:
        rng = np.random.default_rng(0)
        q = np.array([rng.uniform(*design.slot_limits[i]) if design.slot_valid[i] else 0.0
                      for i in range(ge.N_SLOTS)])
        for qq in (np.zeros(ge.N_SLOTS), q):
            T = ge.authored_fk(design, qq)
            tips = ge.tip_fk(design, T)
            for f in range(ge.N_FINGERS):
                used = [ge.finger_slot(f, d) for d in range(ge.N_JOINTS_PER_FINGER)
                        if design.slot_valid[ge.finger_slot(f, d)]]
                if not used:
                    continue
                last = max(used)
                tip_of_last = (T[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3]
                # the fingertip body sits at the real tip (`tip_offsets`), whatever the chain's length
                assert np.allclose(tips[f], tip_of_last, atol=1e-9), (design.source, f, last)
                if len(used) < ge.N_JOINTS_PER_FINGER:
                    assert np.allclose(T[last + 1][:3, 3], tip_of_last, atol=1e-9)   # first ghost at the tip
                checked += 1
    assert checked > 40, f"only checked {checked} finger chains, want a real sample"


def test_every_finger_has_a_fingertip_including_five_joint_fingers():
    """Every finger slot ends in a fingertip body, so `fingertip_valid` is
    True for every finger, including one whose 5 joints fill its slot (the
    SHARPA and Shadow thumbs), whose fingertip body sits one link length
    beyond its last link."""
    for hand_id in ("sharpa_left_on_iiwa14", "shadow_right_local"):
        design = _project_and_canonicalize(hand_id)
        pu = ge.palm_up(design, n_sweep=0)
        assert (pu.fingertip_valid == ge.finger_valid(design)).all()
        full = [f for f in range(ge.N_FINGERS) if design.slot_valid[ge.LAST_FINGER_SLOTS[f]]]
        assert full, hand_id
        offs = ge.tip_offsets(design)
        for f in range(ge.N_FINGERS):
            expected = float(design.slot_length[ge.LAST_FINGER_SLOTS[f]]) if f in full else 0.0
            assert offs[f] == pytest.approx(expected)


# --------------------------------------------------------------------------
# Review items 2/3/4: the full `admit` gate (overlap, spawn height,
# projected-hand exemption) and `viability_report`.
# --------------------------------------------------------------------------


def test_admit_includes_root_capsule_and_ghost_carrier_pairs():
    """G_SERIAL seed 7 is structurally fine but has a known bad rest overlap
    among root-mounted fingers (see test_make_grammar_population.py's own
    test of this seed) -- `admit()`'s default args must catch it, and
    `rest_overlap_pairs` must report at least one pair against the root
    pseudo-node (`ge.ROOT_NODE`) somewhere across a reasonable sample,
    confirming the root capsule is actually being checked, not merely
    present in the code path."""
    seen_root_pair = False
    for seed in range(300):
        model = derive(sample_derivation(seed, G_SERIAL_R))
        if not ge._admit_structural(model).ok:
            continue
        design = ge.canonicalize(model, source=f"probe:{seed}")
        pairs = ge.rest_overlap_pairs(design)
        if any(i == ge.ROOT_NODE or j == ge.ROOT_NODE for i, j, _pen in pairs):
            seen_root_pair = True
            break
    assert seen_root_pair, "no root-capsule overlap pair found in 300 structurally admitted G_SERIAL_R designs"
    assert not ge.admit(derive(sample_derivation(7, G_SERIAL_R))).ok


def test_admit_check_overlap_false_exempts_but_records_pairs():
    entry, status, reason = pf.projected_entry("sharpa_left_on_iiwa14")
    assert status == "admitted", (status, reason)
    assert entry.exempt_overlap
    assert entry.filtered_pairs, "SHARPA is known to overlap by ~20mm; expected recorded filtered_pairs"

    model = derive(pf.derivation_from_dict(entry.derivation_dict))
    assert ge.admit(model, check_overlap=False).ok, "exempted: should pass"
    strict = ge.admit(model, check_overlap=True)
    assert not strict.ok, "SHARPA should fail the DEFAULT (non-exempt) overlap check"


def test_viability_report_shape_admitted_and_rejected():
    admitted_report = None
    for seed, model in _admitted_models(DEFAULT_R, n=50):
        admitted_report = ge.viability_report(model)
        break
    assert admitted_report is not None
    assert admitted_report["admitted"] is True
    assert admitted_report["reasons"] == []
    assert admitted_report["max_rest_overlap_mm"] <= ge.MAX_REST_PENETRATION_M * 1000.0
    assert admitted_report["spawn_height_mm"] >= 0.0
    assert admitted_report["digit_count"] >= 1
    assert admitted_report["joint_count"] >= 1

    # A structurally-fine design rejected on rest overlap (G_SERIAL seed 7).
    derivation = sample_derivation(7, G_SERIAL_R)
    model = derive(derivation)
    rejected_report = ge.viability_report(model)
    assert rejected_report["admitted"] is False
    assert any("rest-overlap" in r for r in rejected_report["reasons"])
    assert rejected_report["max_rest_overlap_mm"] > ge.MAX_REST_PENETRATION_M * 1000.0
    assert rejected_report["digit_count"] is not None  # structural check passed

    # A structurally-rejected design (7 fingers).
    from hand_sampler.grammar_bench.splits import split_hand

    seven = ge.viability_report(derive(split_hand(7)))
    assert seven["admitted"] is False and seven["digit_count"] is None
    assert any("digit_count 7" in r for r in seven["reasons"])


def test_palm_up_base_rot_is_a_unit_quaternion():
    for design in _some_designs(8):
        result = ge.palm_up(design, n_sweep=10)
        assert abs(float(np.linalg.norm(result.base_rot_wxyz)) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# Opus review G0 item 1: rest_overlap_pairs must model the BUILT capsule
# (core [r, L-r]), not the full [0, L] segment.
# --------------------------------------------------------------------------


def test_capsule_core_endpoints_match_author_grammar_built_geometry():
    """`ge.capsule_core_endpoints_local` must equal what `author_grammar.
    _author_body_and_collider` actually builds: a capsule mesh centered at
    local z=`length/2`, cylindrical part `rpc.cylinder_part(length, radius)`,
    hemispherical caps of `radius` on each end -- so the caps' own centers
    (the core segment PhysX's capsule-capsule distance query uses) sit at
    `length/2 -+ cylinder_part/2`. Reimplements that arithmetic independently
    here (not by calling the function under test) as the cross-check, over
    every valid slot AND the root capsule of a broad sample of designs,
    including the `length < 2r` (fully-clamped, sphere-like) case."""
    checked = 0
    saw_clamped_case = False
    for name, dist in VARIANTS:
        for seed, model in _admitted_models(dist, n=150):
            design = ge.canonicalize(model, source=f"{name}:{seed}")
            radius = design.capsule_radius_m
            lengths = [float(design.slot_length[i]) for i in range(ge.N_SLOTS) if design.slot_valid[i]]
            lengths.append(design.root_length_m)
            for length in lengths:
                expected_height = rpc.cylinder_part(max(length, 1e-6), radius)
                expected_z0 = length / 2.0 - expected_height / 2.0
                expected_z1 = length / 2.0 + expected_height / 2.0
                z0, z1 = ge.capsule_core_endpoints_local(length, radius)
                assert abs(z0 - expected_z0) < 1e-12, (name, seed, length, radius)
                assert abs(z1 - expected_z1) < 1e-12, (name, seed, length, radius)
                # Overall extent always reaches the full nominal segment,
                # whether via a nonzero core (length >= 2r) or the sphere's
                # own radius (length < 2r).
                assert z0 - radius <= 1e-9
                assert z1 + radius >= length - 1e-9
                if length < 2.0 * radius:
                    saw_clamped_case = True
                    assert abs(z0 - z1) < 1e-12, "length < 2r must degenerate to a single point (sphere)"
                checked += 1
    assert checked > 200, f"only checked {checked} capsule endpoints, want a broad sample"
    assert saw_clamped_case, "no length < 2r (clamped/sphere) case seen -- want to exercise that branch too"


def test_rest_overlap_pairs_endpoints_are_2r_shorter_than_before():
    """Direct regression for the review's own finding ("the oracle's
    capsules are 2r longer than PhysX's"): for a valid slot whose length
    exceeds 2r, the capsule-core endpoint distance used internally by
    `rest_overlap_pairs` must be `length - 2r`, not `length`."""
    for seed, model in _admitted_models(DEFAULT_R, n=60):
        design = ge.canonicalize(model, source=f"probe:{seed}")
        radius = design.capsule_radius_m
        for idx in range(ge.N_SLOTS):
            if not design.slot_valid[idx]:
                continue
            length = float(design.slot_length[idx])
            if length <= 2.0 * radius:
                continue
            z0, z1 = ge.capsule_core_endpoints_local(length, radius)
            assert abs((z1 - z0) - (length - 2.0 * radius)) < 1e-12


# --------------------------------------------------------------------------
# Opus review G0 item 3: overlap must be checked at BOTH q=0 and default_q.
# --------------------------------------------------------------------------


def test_default_q_overlap_catches_designs_q0_alone_missed():
    """Some designs are clean at q=0 but overlap past the 3mm threshold once
    curled to their own `default_q` (opus review G0: "17/351 (V1) and 37/483
    (V3) admitted designs overlap by more than 3 mm at that pose"). Finds a
    concrete such design and confirms `admit()` actually rejects it (citing
    default_q), not merely that `rest_overlap_pairs(design, q=...)` CAN
    report a pair there."""
    found_seed = None
    for seed in range(1500):
        derivation = sample_derivation(seed, DEFAULT_R)
        model = derive(derivation)
        structural = ge._admit_structural(model)
        if not structural.ok:
            continue
        design = ge.canonicalize(model, source=f"probe:{seed}")
        pairs_q0 = ge.rest_overlap_pairs(design)
        if any(pen > ge.MAX_REST_PENETRATION_M for _, _, pen in pairs_q0):
            continue  # already caught at q=0 -- not the case we want
        pu = ge.palm_up(design, n_sweep=0)
        pairs_default = ge.rest_overlap_pairs(design, q=pu.default_q)
        if any(pen > ge.MAX_REST_PENETRATION_M for _, _, pen in pairs_default):
            found_seed = seed
            break
    assert found_seed is not None, "no q0-clean-but-default_q-bad design found in 1500 seeds"

    derivation = sample_derivation(found_seed, DEFAULT_R)
    model = derive(derivation)
    result = ge.admit(model)
    assert not result.ok
    assert any("default_q" in r for r in result.reasons)


def test_rest_overlap_pairs_q_zero_default_matches_explicit_zeros():
    """`q=None` (the default) must behave exactly like passing an explicit
    all-zeros vector."""
    design = _some_designs(1)[0]
    pairs_default_arg = ge.rest_overlap_pairs(design)
    pairs_explicit_zero = ge.rest_overlap_pairs(design, q=np.zeros(ge.N_SLOTS))
    assert pairs_default_arg == pairs_explicit_zero


# --------------------------------------------------------------------------
# Opus review G0 item 4: spawn point (no hull_extent overshoot) and the
# dense batched-FK reach sweep.
# --------------------------------------------------------------------------


def test_authored_fk_batch_matches_authored_fk_per_sample():
    rng = np.random.default_rng(7)
    for seed, model in _admitted_models(DEFAULT_R, n=20):
        design = ge.canonicalize(model, source=f"batch:{seed}")
        valid = design.slot_valid
        n_batch = 6
        q_batch = np.zeros((n_batch, ge.N_SLOTS))
        for i in range(n_batch):
            for idx in range(ge.N_SLOTS):
                if valid[idx]:
                    lo, hi = design.slot_limits[idx]
                    q_batch[i, idx] = rng.uniform(lo, hi)
        T_batch = ge.authored_fk_batch(design, q_batch)
        assert T_batch.shape == (n_batch, ge.N_SLOTS, 4, 4)
        for i in range(n_batch):
            T_single = ge.authored_fk(design, q_batch[i])
            assert np.allclose(T_batch[i], T_single, atol=1e-9), (seed, i)


def test_spawn_offset_is_tip_centroid_plus_clearance_no_overshoot():
    """opus review G0 item 4: the spawn point must be the fingertip
    centroid at default_q plus (object half size + 5 mm) along the
    calibrated normal -- NOT the old `hull_extent` overshoot past the
    farthest mid-curl fingertip."""
    object_half_size = 0.03
    checked = 0
    for seed, model in _admitted_models(DEFAULT_R, n=120):
        design = ge.canonicalize(model, source=f"spawn:{seed}")
        pu = ge.palm_up(design, n_sweep=0, object_half_size=object_half_size)
        T_mid = ge.authored_fk(design, pu.default_q)
        tip_pts = []
        for f in range(ge.N_FINGERS):
            used = [ge.finger_slot(f, d) for d in range(ge.N_JOINTS_PER_FINGER)
                    if design.slot_valid[ge.finger_slot(f, d)]]
            if not used:
                continue
            last = max(used)
            tip_pts.append((T_mid[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3])
        if not tip_pts:
            continue
        tip_centroid = np.mean(np.stack(tip_pts), axis=0)
        expected = tip_centroid + pu.normal * (object_half_size + 0.005)
        assert np.allclose(pu.spawn_offset, expected, atol=1e-9), (seed,)
        checked += 1
    assert checked > 10, f"only checked {checked} designs"


def test_dense_reach_sweep_uses_at_least_4000_samples_by_default():
    design = _some_designs(1)[0]
    result = ge.palm_up(design)
    assert result.n_swept >= 4000


@pytest.mark.parametrize("hand_id", ["sharpa_left_on_iiwa14", "wuji_right", "xhand_right"])
def test_five_finger_commercial_hands_reach_with_at_least_three(hand_id):
    """opus review G0 item 4's validation criterion: five-finger projected
    commercial hands must reach the (fixed) spawn point with at least 3
    fingertips under the dense sweep (the old 200-sample sweep put many
    fingers, even fully extended, out of the spawn point's reach)."""
    design = _project_and_canonicalize(hand_id)
    n_digits = sum(1 for d in design.finger_digit_id if d is not None)
    assert n_digits == 5, f"{hand_id}: expected 5 digits, got {n_digits}"
    pu = ge.palm_up(design)
    assert pu.reachable_fingertips >= 3, f"{hand_id}: only {pu.reachable_fingertips} fingertips reach"


# --------------------------------------------------------------------------
# Joints at one point (a 0 mm bone, or one shorter than a capsule radius)
# --------------------------------------------------------------------------


def _hand_with_a_short_bone(length_m: float):
    """A one-grammar hand with a 3+ joint finger whose middle bone is
    `length_m` long and bent 60 deg at both its joints."""
    import math

    from hand_sampler.grammar.derive import DerivationStep
    from hand_sampler.grammar.limits import SIMULATOR
    from hand_sampler.grammar.variants import build_distribution

    dist = build_distribution()
    for seed in range(200):
        d = sample_derivation(seed, dist, limits=SIMULATOR)
        digits = [s for s in d.steps if s.production == "Digit" and s.params["phalanx_count"] >= 3]
        if not digits:
            continue
        did = digits[0].params["digit_id"]
        steps = []
        for s in d.steps:
            if s.production == "Phalanx" and s.params["digit_id"] == did and s.params["p"] in (1, 2):
                p = dict(s.params)
                p["bend_rpy"] = (0.0, math.radians(60.0), 0.0)
                if s.params["p"] == 1:
                    p["length"] = length_m
                s = DerivationStep(path=s.path, production=s.production, params=p)
            steps.append(s)
        dd = replace(d, steps=tuple(steps))
        model = derive(dd)
        if ge._admit_structural(model).ok:
            design = ge.canonicalize(model)
            f = next(i for i, x in enumerate(design.finger_digit_id) if x == did)
            return design, ge.finger_slot(f, 0)
    raise AssertionError("no suitable hand")


def test_bodies_either_side_of_a_0mm_bone_are_adjacent_and_filtered():
    design, base = _hand_with_a_short_bone(0.0)
    a, c = base, base + 2                        # the bones before and after the 0 mm bone (base + 1)
    assert (a, c) in ge.adjacent_pairs(design) and (c, a) in ge.adjacent_pairs(design)
    assert (a, c) in design.filtered_pairs        # authoring collision-filters them
    for q in (None, ge.palm_up(design, n_sweep=0).default_q):
        assert not any({i, j} == {a, c} for i, j, _ in ge.rest_overlap_pairs(design, q=q))
    # the same pair, with the middle bone 30 mm long, is a pair the check sees
    design30, base30 = _hand_with_a_short_bone(0.030)
    assert (base30, base30 + 2) not in ge.adjacent_pairs(design30)
    assert (base30, base30 + 2) not in design30.filtered_pairs


def test_without_the_chain_rule_the_0mm_bone_overlaps():
    """What the chain rule removes: across a 0 mm bone bent 60 deg twice, the
    neighbours' capsules overlap by more than the 3 mm gate."""
    from hand_sampler.design_space import segment_distance

    design, base = _hand_with_a_short_bone(0.0)
    T = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    ends = []
    for idx in (base, base + 2):
        z0, z1 = ge.capsule_core_endpoints_local(float(design.slot_length[idx]), design.capsule_radius_m)
        ends.append([(T[idx] @ np.array([0.0, 0.0, z, 1.0]))[:3] for z in (z0, z1)])
    pen = 2 * design.capsule_radius_m - segment_distance(*ends[0], *ends[1])
    assert pen > ge.MAX_REST_PENETRATION_M


def test_short_bone_pairs_survive_mark_filtered_pairs():
    design, base = _hand_with_a_short_bone(0.0)
    marked = ge.mark_filtered_pairs(design, extra_qs=[ge.palm_up(design, n_sweep=0).default_q])
    assert set(design.filtered_pairs) <= set(marked.filtered_pairs)


def test_sampled_designs_keep_parent_child_adjacency_only():
    """No sampled bone is shorter than a capsule radius, so sampled designs
    get no short-bone pairs and the overlap check is unchanged for them."""
    from hand_sampler.grammar.limits import SIMULATOR
    from hand_sampler.grammar.variants import build_distribution

    n = 0
    for dist in (DEFAULT_R, build_distribution()):
        for seed in range(150):
            model = derive(sample_derivation(seed, dist, limits=SIMULATOR if dist is not DEFAULT_R else None))
            if not ge._admit_structural(model).ok:
                continue
            design = ge.canonicalize(model)
            assert design.filtered_pairs == ge.mount_pairs(design)
            parent_only = set()
            for idx in range(ge.N_SLOTS):
                if design.slot_valid[idx]:
                    p = ge._effective_parent(design, idx)
                    parent_only |= {(idx, p), (p, idx)}
            assert ge.adjacent_pairs(design) == parent_only
            n += 1
    assert n > 150


def test_zero_length_link_mass_is_floored_at_a_sphere():
    import math

    from isaacsimenvs.inhand_reorient.scene import author_grammar as ag

    for r in (0.008, 0.010, 0.012):
        m, inertia = ag._link_mass_props(0.0, r, True)
        sphere = 4.0 / 3.0 * math.pi * r ** 3 * rpc.GEN_LINK_DENSITY_KG_M3
        assert m == pytest.approx(sphere) and min(inertia) >= 0.4 * sphere * r * r * (1 - 1e-9)
        # a link long enough to outweigh the sphere is unchanged
        m30, i30 = ag._link_mass_props(0.030, r, True)
        assert m30 == pytest.approx(math.pi * r * r * 0.030 * rpc.GEN_LINK_DENSITY_KG_M3)
        assert i30[2] == pytest.approx(0.5 * m30 * r * r)
    assert ag._link_mass_props(0.0, 0.01, False) == (rpc.VIRTUAL_LINK_MASS_KG, (rpc.VIRTUAL_LINK_INERTIA,) * 3)


def test_fingers_are_filtered_against_the_body_they_sit_on():
    """A finger's PhysX joint parent is its own carrier. On a locked carrier
    it really sits on the palm, on a follower carrier on the leader's palm
    part; both are exempt from the overlap check against that body and
    collision-filtered against it (`mount_pairs`). A leader's own finger
    sits on the leader's body, its joint parent, which PhysX excludes
    itself, so it gets no pair."""
    from hand_sampler.grammar_bench.splits import split_hand
    from hand_sampler.grammar.limits import SIMULATOR
    from hand_sampler.grammar.variants import build_distribution

    models = [derive(split_hand(r, p)) for r, p in ((3, (2, 1)), (2, (2,)), (0, (3, 3)), (5, ()))]
    models += [derive(sample_derivation(s, build_distribution(), limits=SIMULATOR)) for s in range(200)]
    seen = {ge.LOCKED: 0, ge.LEADER: 0, ge.FOLLOWER: 0}
    for model in models:
        if not ge._admit_structural(model).ok:
            continue
        design = ge.canonicalize(model)
        pairs = set(design.filtered_pairs)
        adjacent = ge.adjacent_pairs(design)
        for f, role in enumerate(ge.carrier_roles(design)):
            base, c = ge.finger_slot(f, 0), ge.carrier_slot(f)
            if not design.slot_valid[base]:
                continue
            seen[role] += 1
            on = {ge.LOCKED: ge.ROOT_NODE, ge.LEADER: c, ge.FOLLOWER: int(design.slot_tie[c])}[role]
            assert ge._effective_parent(design, base) == on and (base, on) in adjacent
            assert ((min(on, base), max(on, base)) in pairs) == (role != ge.LEADER)
        assert set(ge.mount_pairs(design)) <= pairs
    assert all(v > 0 for v in seen.values()), seen


def test_followers_move_with_their_leader():
    """A follower carrier has its leader's frame, so at any joint vector the
    two bodies coincide (`authored_fk` ties the follower to the leader, as
    the mimic joint does); its finger matches the grammar's FK."""
    from hand_sampler.grammar_bench.splits import split_hand

    design = ge.canonicalize(derive(split_hand(3, (2, 1))))
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = np.array([rng.uniform(*design.slot_limits[i]) if design.slot_valid[i] else 0.0
                      for i in range(ge.N_SLOTS)])
        q_bad = q.copy()
        q_bad[design.slot_tie >= 0] = 1.234            # ignored: followers take their leader's value
        Ta, Tb, Tr = ge.authored_fk(design, q), ge.authored_fk(design, q_bad), ge.grammar_fk_reference(design, q)
        assert np.allclose(Ta, Tb)
        real = [i for i in range(ge.N_SLOTS) if design.slot_body_name[i] is not None]
        assert np.abs(Ta[real] - Tr[real]).max() < 1e-9
        for s in np.nonzero(design.slot_tie >= 0)[0]:
            assert np.allclose(Ta[s], Ta[design.slot_tie[s]])
    pu = ge.palm_up(design, n_sweep=0)
    fol = np.nonzero(design.slot_tie >= 0)[0]
    assert np.allclose(pu.default_q[fol], pu.default_q[design.slot_tie[fol]])
