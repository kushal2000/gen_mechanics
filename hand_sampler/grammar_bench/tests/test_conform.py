"""Conforming projected commercial hands to the grammar's own discretisation
(``adapters/conform.py``), and the opt-in lateral mounts it can use
(``Distribution.mount_lateral_grid_m``, the ``G_WIDE`` reference variant).

1. Every manifest hand, conformed to G_FULL / G_V1 / G_V3S, is a valid
   derivation that ``coverage`` (an independent support audit) finds
   in-support, while the exact projection is not; every value lies on the
   variant's grid.
2. Conforming a hand the grammar sampled reproduces it exactly (closed-loop
   snapping), so conform is the identity on the grammar's own hands.
3. The grid-stepping operators that cannot act on an exact projection act on
   the conformed hand, and no operator raises on it.
4. The report: SIMULATOR violations (none: every hand fits the simulator,
   SVH's two fingers on one palm joint included), rule conflicts (co-located
   joints, lateral finger offsets), fidelity
   (G_WIDE closer to the hand than G_FULL).
5. ``mount_lateral_grid_m``: off by default (sampling unchanged, see
   ``test_generation_limits``'s digests); when set, offsets on the grid for
   digits and palm bodies, stepped by ``step_mount``, drawn by the growth
   operators.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import pytest

from hand_sampler.grammar.adapters import conform as C
from hand_sampler.grammar.adapters.projection import project_to_derivation
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coverage import coverage
from hand_sampler.grammar.derive import (
    EVOLUTION_OPERATORS,
    EVOLUTION_OPERATORS_FINE,
    VariationImpossible,
    _axis_grid_indices,
    apply_operator,
    derive,
    sample_derivation,
    vary,
)
from hand_sampler.grammar.distributions import (DEG, lateral_offset_choices_m, link_length_support_m,
                                                palm_body_length_support_m, root_length_support_m)
from hand_sampler.grammar.experiments import e13_representation as e13
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.limits import SIMULATOR, Structure
from hand_sampler.grammar.variants import G_FULL, G_V1, G_V3S, G_WIDE, NAMED_DISTRIBUTIONS, build_distribution


def _hands():
    out = {}
    for hand, path, avail, _ in e13._cases(e13._load_manifest()):
        if avail != "available":
            continue
        imp = load_urdf(path, hand_root=hand.get("hand_root"))
        pr = project_to_derivation(imp.model, palm_joints=hand.get("palm_joints", ()),
                                   tip_frames=hand.get("tip_frames", {}))
        out[hand["id"]] = (imp.model, pr)
    return out


HANDS = _hands()
HAND_IDS = sorted(HANDS)
GRAMMAR = build_distribution()
VARIANTS = {"G_FULL": G_FULL, "G_V1": G_V1, "G_V3S": G_V3S, "G_WIDE": G_WIDE, "GRAMMAR": GRAMMAR}

if not HANDS:  # pragma: no cover
    pytest.skip("local-only: no manifest hand available on this machine", allow_module_level=True)


def _on_grid(d, dist):
    """Every value of ``d`` is one ``dist`` can generate (raises otherwise)."""
    grid = dist.link_length_grid_m
    fracs = set(dist.mount_frac_choices)
    angles = {round((k * 15.0 - 180.0) * DEG, 12) for k in range(24)}
    lat = set(lateral_offset_choices_m(dist))

    def on_len(v, rng):
        k = (v - rng[0]) / grid
        assert abs(k - round(k)) < 1e-9 and rng[0] - 1e-9 <= v <= rng[1] + 1e-9, (v, rng)

    hand = next(s for s in d.steps if s.path == "hand").params
    on_len(hand["root_length"], root_length_support_m(dist))
    assert hand["capsule_radius_m"] in dist.capsule_radius_choices_m
    for s in d.steps:
        p = s.params
        if s.production == "PalmBody":
            on_len(p["length"], palm_body_length_support_m(dist))
            assert p["mount_frac"] in fracs
            assert all(round(a, 12) in angles for a in p["direction_rpy"])
            _axis_grid_indices(p["axis"])
            assert all(v in lat for v in p.get("mount_offset", (0.0, 0.0)))
            if p["has_joint"] and dist.limits_support_continuous:
                a, b = dist.revolute_limit_range_deg
                assert a * DEG - 1e-9 <= p["limits"][0] < p["limits"][1] <= b * DEG + 1e-9
            elif p["has_joint"]:
                assert any(abs(lo * DEG - p["limits"][0]) < 1e-9 and abs(hi * DEG - p["limits"][1]) < 1e-9
                           for lo, hi in dist.palm_joint_limit_choices_deg)
        elif s.production == "Digit":
            assert p["mount_frac"] in fracs
            assert all(round(a, 12) in angles for a in p["mount_rpy"])
            off = p.get("mount_offset", (0.0, 0.0))
            if dist.mount_on_host_surface and not (dist.mount_lateral_grid_m and not dist.mount_lateral_sampled):
                assert abs(math.hypot(*off) - hand["capsule_radius_m"]) < 1e-9
            else:
                assert all(v in lat for v in off)
        elif s.production == "Phalanx":
            on_len(p["length"], link_length_support_m(dist))
            mod = p["module"]
            _axis_grid_indices(mod["axis"])
            if mod["kind"] == "R":
                lo, hi = mod["limits"]
                if dist.limits_continuous or dist.limits_support_continuous:
                    a, b = dist.revolute_limit_range_deg
                    assert a * DEG - 1e-9 <= lo < hi <= b * DEG + 1e-9
                else:
                    assert any(abs(a * DEG - lo) < 1e-9 and abs(b * DEG - hi) < 1e-9
                               for a, b in dist.revolute_limit_choices_deg)
            pair = (tuple(p["bend_rpy"]), tuple(p["bend_offset"]))
            assert pair in {(tuple(r), tuple(o)) for r, o in C._bend_support(dist, p["p"] == 0)}, pair


@pytest.mark.parametrize("variant", ["GRAMMAR", "G_FULL", "G_V1", "G_V3S"])
@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_conformed_hand_is_in_the_grammar(hand_id, variant):
    dist = VARIANTS[variant]
    model, pr = HANDS[hand_id]
    if not (hand_id == "coupled_finger" and variant == "GRAMMAR"):
        # the exact projection is not (the analytic coupled finger's only
        # off-support value was its 0 mm bone, which the grammar now has)
        assert not coverage(derive(pr.derivation), dist).in_support
    cd, rep = C.conform_to_grammar(pr.derivation, dist, SIMULATOR)
    assert rep.valid, rep.error
    cov = coverage(derive(cd), dist)
    assert cov.in_support, cov.out_of_support
    _on_grid(cd, dist)


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_conformed_to_g_wide_on_grid_and_closer(hand_id):
    model, pr = HANDS[hand_id]
    cd, rep = C.conform_to_grammar(pr.derivation, G_WIDE)
    assert rep.valid
    _on_grid(cd, G_WIDE)
    wide = C.fidelity(model, pr.name_map, rep.root_transform(pr.root_transform), derive(cd))
    cf, rf = C.conform_to_grammar(pr.derivation, G_FULL)
    full = C.fidelity(model, pr.name_map, rf.root_transform(pr.root_transform), derive(cf))
    assert wide["max_pos_mm"] <= full["max_pos_mm"] + 1e-6
    assert "lateral_mount_offset" not in rep.conflict_features()


@pytest.mark.parametrize("name", ["G_FULL", "G_V1", "G_V1S", "G_V2S", "G_V3S", "G_CONT", "G_NOBRANCH"])
def test_conform_reproduces_grammar_hands(name):
    dist = NAMED_DISTRIBUTIONS[name]
    for seed in range(25):
        d = sample_derivation(seed, dist)
        cd, rep = C.conform_to_grammar(d, dist, exact=("radius",))
        assert rep.root_shift_m == 0.0 and not rep.conflicts
        T0, T1 = forward_kinematics(derive(d), {}), forward_kinematics(derive(cd), {})
        assert max(float(np.abs(T0[b] - T1[b]).max()) for b in T0) < 1e-9


def test_exact_projection_is_exact():
    for hand_id in HAND_IDS:
        model, pr = HANDS[hand_id]
        f = C.fidelity(model, pr.name_map, pr.root_transform, derive(pr.derivation))
        assert f["max_pos_mm"] < 1e-6 and f["max_axis_deg"] < 1e-3 and f["within_target"]


GRID_STEPS = ("step_axis", "step_limits", "step_mount", "step_root_length", "step_radius", "step_segment_length")


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_operators_act_on_conformed_hands(hand_id):
    model, pr = HANDS[hand_id]
    cd, _ = C.conform_to_grammar(pr.derivation, G_FULL)
    status = C.operator_applicability(cd, G_FULL, EVOLUTION_OPERATORS)
    assert not any(v.startswith("error") for v in status.values()), status
    assert all(status[op] == "ok" for op in GRID_STEPS), status
    for op in ("insert_phalanx", "delete_phalanx", "add_palm_body", "add_minimal_digit"):
        assert status[op] == "ok", (op, status[op])
    rng = np.random.default_rng(0)
    d = cd
    n = 0
    for _ in range(40):                       # a mutation chain stays valid
        try:
            d = vary(d, rng, G_FULL, operators=EVOLUTION_OPERATORS)
        except VariationImpossible:
            continue
        derive(d)
        n += 1
    assert n >= 20


def test_the_exact_projection_misses_grid_steps():
    """Before conforming, some grid-stepping operator cannot act on every hand
    (step_limits: the URDF limits are off the menu)."""
    missing = 0
    for hand_id in HAND_IDS:
        _, pr = HANDS[hand_id]
        status = C.operator_applicability(pr.derivation, G_FULL, GRID_STEPS)
        missing += status["step_limits"] != "ok"
    assert missing >= len(HAND_IDS) - 1


def test_report_names_the_known_conflicts():
    if "svh_right" in HANDS:
        _, pr = HANDS["svh_right"]
        cd, rep = C.conform_to_grammar(pr.derivation, G_FULL, SIMULATOR)
        assert rep.limits.ok                  # two fingers on one palm joint fit the simulator
        assert "colocated_joints" in rep.conflict_features()
    if "allegro_right" in HANDS:
        _, pr = HANDS["allegro_right"]
        _, rep = C.conform_to_grammar(pr.derivation, G_FULL, SIMULATOR)
        assert rep.limits.ok and "lateral_mount_offset" in rep.conflict_features()
    if "shadow_right_local" in HANDS:
        _, pr = HANDS["shadow_right_local"]
        _, rep = C.conform_to_grammar(pr.derivation, G_FULL)
        assert "colocated_joints" in rep.conflict_features()
    if "coupled_finger" in HANDS:
        _, pr = HANDS["coupled_finger"]
        _, rep = C.conform_to_grammar(pr.derivation, G_V3S)
        assert "forced_curl" in rep.conflict_features()


def test_exact_classes_restore_fidelity():
    """Diagnosis: with every class exact, the conformed hand is the projection."""
    model, pr = HANDS[HAND_IDS[0]]
    cd, rep = C.conform_to_grammar(pr.derivation, G_FULL, exact=C.PARAMETER_CLASSES)
    f = C.fidelity(model, pr.name_map, rep.root_transform(pr.root_transform), derive(cd))
    assert f["max_pos_mm"] < 1e-6 and f["max_axis_deg"] < 1e-3


# ---------------------------------------------------------------------------
# Lateral mounts (opt-in)
# ---------------------------------------------------------------------------

LAT = replace(G_FULL, mount_lateral_grid_m=0.005, mount_lateral_max_m=0.03)


def test_lateral_mounts_off_by_default():
    for dist in NAMED_DISTRIBUTIONS.values():
        dist = dist[0] if isinstance(dist, tuple) else dist
        if dist is G_WIDE:
            continue
        assert dist.mount_lateral_grid_m is None
    assert lateral_offset_choices_m(G_FULL) == (0.0,)


def test_lateral_mounts_sampled_on_the_grid():
    grid = set(lateral_offset_choices_m(LAT))
    assert max(grid) == 0.03 and len(grid) == 13
    seen = set()
    for seed in range(60):
        d = sample_derivation(seed, LAT)
        for s in d.steps:
            if s.production == "PalmBody" or (s.production == "Digit" and s.params["top_level"]):
                off = s.params["mount_offset"]
                assert all(v in grid for v in off)
                seen.update(off)
            elif s.production == "Digit":
                assert s.params["mount_offset"] == (0.0, 0.0)          # branch digits stay on the axis
        derive(d)
    assert len(seen) > 8


def test_lateral_mounts_mutated():
    rng = np.random.default_rng(1)
    moved = grown = 0
    for seed in range(40):
        d = sample_derivation(seed, LAT)
        child = apply_operator(d, rng, LAT, "step_mount")
        if child is not None:
            a = {s.params["uid"]: s.params.get("mount_offset") for s in d.steps if "uid" in s.params}
            b = {s.params["uid"]: s.params.get("mount_offset") for s in child.steps if "uid" in s.params}
            diff = [u for u in a if a[u] != b[u]]
            if diff:
                (u,) = diff
                delta = np.subtract(b[u], a[u])
                assert abs(abs(delta).max() - 0.005) < 1e-12 and abs(delta).min() == 0.0
                moved += 1
        for op in ("add_palm_body", "add_minimal_digit"):
            child = apply_operator(d, rng, LAT, op)
            if child is not None:
                new = [s for s in child.steps if s.params.get("uid") not in {t.params.get("uid") for t in d.steps}
                       and s.production in ("PalmBody", "Digit")]
                assert all("mount_offset" in s.params for s in new)
                grown += 1
    assert moved > 5 and grown > 20


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_the_one_grammar_contains_the_hands(hand_id):
    """Against the one grammar (all rules on) the only rule conflicts left are
    the ones its capability does not cover yet (see the README): joints at
    one point, bone lengths outside 15-80 mm, mounts off a palm part, and the
    15 deg grid where it costs more than 10 deg."""
    model, pr = HANDS[hand_id]
    cd, rep = C.conform_to_grammar(pr.derivation, GRAMMAR, SIMULATOR)
    assert set(rep.conflict_features()) <= {"colocated_joints", "link_length_range", "mount_off_segment", "rest_bend"}
    f = C.fidelity(model, pr.name_map, rep.root_transform(pr.root_transform), derive(cd))
    _, rf = C.conform_to_grammar(pr.derivation, G_V1)
    old = C.fidelity(model, pr.name_map, rf.root_transform(pr.root_transform),
                     derive(C.conform_to_grammar(pr.derivation, G_V1)[0]))
    assert f["max_pos_mm"] <= old["max_pos_mm"] + 1e-6


# ---------------------------------------------------------------------------
# The fine resolution (conform_to_grammar(..., resolution="fine"))
# ---------------------------------------------------------------------------

# Hands that miss the 5 mm / 10 deg target at the fine resolution, and why (see
# the README): a small bend between two bones (DClaw's 2.8 deg) is not on the
# 5 deg rest-bend grid, whose angles from straight are 0, 5, 7.1, 10, ... deg,
# and near straight the joint axis cannot be matched as well, so on a long
# bone the next joint lands a few millimetres off. A 1 mm lateral joint offset
# (``bend_offset``) in the fine support brings every hand inside.
FINE_MISSES = {"dclaw"}


def _fine_on_grid(d):
    def whole(v, unit):
        return abs(v / unit - round(v / unit)) < 1e-6

    from hand_sampler.grammar.derive import fine_axis_indices
    hand = next(s for s in d.steps if s.path == "hand").params
    assert whole(hand["root_length"], 0.001) and whole(hand["capsule_radius_m"], 0.001)
    for s in d.steps:
        p = s.params
        if s.production in ("Digit", "PalmBody"):
            assert whole(p["mount_frac"], 0.01)
            assert all(whole(v, 0.001) for v in p.get("mount_offset", (0.0, 0.0)))
            rpy = p["mount_rpy"] if s.production == "Digit" else p["direction_rpy"]
            assert all(whole(math.degrees(a), 5.0) for a in rpy)
        if s.production == "PalmBody":
            assert whole(p["length"], 0.001) and fine_axis_indices(p["axis"]) is not None
        if s.production == "Phalanx":
            assert whole(p["length"], 0.001) and fine_axis_indices(p["module"]["axis"]) is not None
            assert all(whole(math.degrees(a), 5.0) for a in p["bend_rpy"])


@functools.lru_cache(maxsize=None)
def _fine(hand_id):
    model, pr = HANDS[hand_id]
    out = {}
    for res in ("coarse", "fine"):
        cd, rep = C.conform_to_grammar(pr.derivation, GRAMMAR, SIMULATOR, resolution=res)
        f = C.fidelity(model, pr.name_map, rep.root_transform(pr.root_transform), derive(cd))
        out[res] = (cd, rep, f)
    return out


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_fine_conform_is_in_the_fine_grammar(hand_id):
    cd, rep, _ = _fine(hand_id)["fine"]
    assert rep.valid, rep.error
    cov = coverage(derive(cd), GRAMMAR, resolution="fine")
    assert cov.in_support, cov.out_of_support
    _fine_on_grid(cd)
    # within the simulator limits (SVH's two fingers on one palm joint included), and the
    # simulator builds it
    assert rep.limits.ok, rep.limits.failing
    from hand_sampler.grammar_bench.tests.test_generation_limits import GE
    assert GE._admit_structural(derive(cd)).ok, GE._admit_structural(derive(cd)).reasons
    assert not Structure.from_steps(cd.steps).empty_palm_bodies()          # the grammar's palm rule
    assert not rep.conflicts, rep.conflict_features()


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_fine_conform_fidelity(hand_id):
    res = _fine(hand_id)
    f, fc = res["fine"][2], res["coarse"][2]
    worst = max(f["max_pos_mm"], f["max_tip_mm"] or 0.0)
    assert worst <= max(fc["max_pos_mm"], fc["max_tip_mm"] or 0.0) + 1e-6          # at least as close as coarse
    if hand_id in FINE_MISSES:
        assert worst <= 8.0 and f["max_axis_deg"] <= 10.0, f
    else:
        assert f["within_target"], f


@pytest.mark.parametrize("hand_id", HAND_IDS)
def test_fine_operators_act_on_fine_conformed_hands(hand_id):
    cd, _, _ = _fine(hand_id)["fine"]
    status = C.operator_applicability(cd, GRAMMAR, EVOLUTION_OPERATORS_FINE, limits=SIMULATOR)
    assert all(v == "ok" for v in status.values()), status


@pytest.mark.parametrize("name", ["GRAMMAR", "G_V3S"])
def test_fine_conform_reproduces_grammar_hands(name):
    """A sampled hand is on the fine grid already: conforming it at the fine
    resolution keeps every joint where it was (to within the 1 mm lateral
    grid, which a surface-mounted finger's offset is not on)."""
    dist = GRAMMAR if name == "GRAMMAR" else G_V3S
    for seed in range(15):
        d = sample_derivation(seed, dist, limits=SIMULATOR)
        cd, rep = C.conform_to_grammar(d, dist, exact=("radius",), resolution="fine")
        T0, T1 = forward_kinematics(derive(d), {}), forward_kinematics(derive(cd), {})
        assert max(float(np.linalg.norm(T0[b][:3, 3] - T1[b][:3, 3])) for b in T0) < 1.5e-3
        m0, m1 = derive(d), derive(cd)
        ax = {j.name: T0[j.child][:3, :3] @ np.asarray(j.axis) for j in m0.joints if j.type != "fixed"}
        for j in m1.joints:
            if j.name in ax:
                assert float(np.dot(ax[j.name], T1[j.child][:3, :3] @ np.asarray(j.axis))) > math.cos(math.radians(2.0))


def test_resolution_is_validated_by_conform():
    _, pr = HANDS[HAND_IDS[0]]
    with pytest.raises(ValueError):
        C.conform_to_grammar(pr.derivation, GRAMMAR, resolution="medium")
