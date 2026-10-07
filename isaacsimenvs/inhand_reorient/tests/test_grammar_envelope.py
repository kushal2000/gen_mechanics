"""CPU tests for the 36-slot envelope of the locked grammar
(`scene/grammar_envelope.py`), population files (`scene/population_file.py`),
the population builder and the commercial-hand helpers. Run with the isaacsim
venv:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_grammar_envelope.py -q
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hand_sampler.grammar import build_conformed
from hand_sampler.grammar import derive as gdv
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar import viability as gvb
from hand_sampler.grammar.hand import EVOLUTION_RULES, NO_RULES, Finger, Hand, Joint, PalmJoint, check

from isaacsimenvs.inhand_reorient import make_grammar_population as mkpop
from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf
from isaacsimenvs.inhand_reorient.scene import projected_hands as ph


def _random_hands(n, rules=NO_RULES, seed=0, mutate=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = gops.random_hand(rng, rules)
        for _ in range(mutate):
            h, _, _ = gops.mutate(h, rng, rules)
        out.append(h)
    return out


SPECIAL = {"split321": mkpop.split_321_hand(), "zero_links": mkpop.zero_link_hand(), "oblique": mkpop.oblique_hand()}


# --------------------------------------------------------------------------
# canonicalize / FK
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand", list(SPECIAL.values()) + _random_hands(30, mutate=8), ids=lambda h: "")
def test_authored_fk_matches_the_grammar(hand):
    """Authored frames (the dual-frame z-joint convention PhysX uses) equal
    the grammar's FK, at zero and at random joint vectors, tied slots
    following their source."""
    d = ge.canonicalize(hand)
    rng = np.random.default_rng(1)
    for q in [np.zeros(ge.N_SLOTS)] + [ge.dofs_to_slot_q(d, gdv.sample_q(hand, 1, rng)[0]) for _ in range(3)]:
        A = ge.authored_fk(d, q)
        B = ge.grammar_fk_reference(d, q)
        for s in range(ge.N_SLOTS):
            if d.slot_body_name[s] is not None:
                assert np.allclose(A[s], B[s], atol=1e-9), s
        tips = ge.tip_fk(d, A)
        g = gdv.fk(hand, ge.slot_q_to_dofs(d, q)).tips
        for f in range(ge.N_FINGERS):
            i = d.finger_index[f]
            if i >= 0:
                assert np.allclose(tips[f], g[i], atol=1e-9)
    Q = np.stack([ge.dofs_to_slot_q(d, gdv.sample_q(hand, 1, rng)[0]) for _ in range(5)])
    TB = ge.authored_fk_batch(d, Q)
    for k in range(5):
        assert np.allclose(TB[k], ge.authored_fk(d, Q[k]), atol=1e-9)


def test_every_grammar_hand_fits_the_envelope():
    for h in _random_hands(200, NO_RULES, seed=2, mutate=5):
        d = ge.canonicalize(h)
        assert int(d.slot_real[list(ge.FINGER_BASE_SLOTS)].sum()) == len(h.fingers)
        assert int(sum(len(f.joints) for f in h.fingers)) == sum(
            int(d.slot_real[ge.finger_slot(f, k)]) for f in range(6) for k in range(5))


def test_split_321_roles_and_ties():
    d = ge.canonicalize(SPECIAL["split321"])
    roles = ge.carrier_roles(d)
    assert roles == (ge.LOCKED, ge.LOCKED, ge.LOCKED, ge.LEADER, ge.FOLLOWER, ge.LEADER)
    assert d.finger_index == (2, 3, 4, 0, 1, 5)
    assert d.slot_tie[ge.carrier_slot(4)] == ge.carrier_slot(3) and d.slot_gear[ge.carrier_slot(4)] == 1.0
    # coupled joints: mimic joints on the joint before them, gear 1.1, not policy-controlled
    s = ge.finger_slot(0, 2)
    assert d.slot_real[s] and not d.slot_valid[s]
    assert d.slot_tie[s] == ge.finger_slot(0, 1) and d.slot_gear[s] == pytest.approx(1.1)
    q = np.zeros(ge.N_SLOTS)
    q[ge.finger_slot(0, 1)] = 0.5
    q[ge.carrier_slot(3)] = 0.2
    t = ge.tied_q(d, q)
    assert t[s] == pytest.approx(0.55) and t[ge.carrier_slot(4)] == pytest.approx(0.2)


def test_coupling_chains_resolve_to_the_independent_joint():
    h = Hand(fingers=(Finger(y=0, z=90, joints=(Joint("hinge", (0, 0), 40), Joint("coupled", (0, 0), 30),
                                                 Joint("coupled", (0, 0), 25))),
                      Finger(y=40, z=90, joints=(Joint("hinge", (0, 0), 40),))))
    d = ge.canonicalize(h)
    s = ge.finger_slot(0, 2)
    assert d.slot_tie[s] == ge.finger_slot(0, 0) and d.slot_gear[s] == pytest.approx(1.21)


def test_tie_columns_and_gears():
    d = ge.canonicalize(SPECIAL["split321"])
    perm = np.arange(ge.N_SLOTS)[::-1]
    cols = ge.tie_columns(d.slot_tie[None], perm)
    gears = ge.tie_gears(d.slot_tie[None], d.slot_gear[None], perm)
    q = np.random.default_rng(0).normal(size=ge.N_SLOTS)
    x = q[perm][None]
    tied = np.take_along_axis(x, cols, 1) * gears
    assert np.allclose(tied[0], ge.tied_q(d, q)[perm])


def test_sliding_joints_are_prismatic_slots():
    rec = build_conformed.load().get("dex1")
    if rec is None:
        pytest.skip("local-only:dex1 not in conformed_hands.json")
    d = ge.canonicalize(rec["hand"])
    assert d.slot_prismatic[ge.finger_slot(0, 0)] and d.slot_prismatic[ge.finger_slot(1, 0)]
    lo, hi = d.slot_limits[ge.finger_slot(0, 0)]
    assert (lo, hi) == pytest.approx((-0.02, 0.025))


def test_ghost_tail_sits_at_the_fingertip():
    for h in _random_hands(20, seed=3):
        d = ge.canonicalize(h)
        T = ge.authored_fk(d, np.zeros(ge.N_SLOTS))
        tips = ge.tip_fk(d, T)
        for f in range(6):
            i = d.finger_index[f]
            if i >= 0:
                assert np.allclose(tips[f], gdv.tip_zero_m(h.fingers[i]), atol=1e-12)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_link_colliders_are_8_vertex_cores_and_plates_fit_gpu_hulls():
    from scipy.spatial import ConvexHull

    for h in list(SPECIAL.values()) + _random_hands(40, seed=4):
        d = ge.canonicalize(h)
        for s in range(ge.N_SLOTS):
            if d.slot_real[s] and ge.slot_joint_index(s) >= 0:
                pts = ge.link_core_points(d, s)
                assert pts.shape == (8, 3)
                ext = pts.max(0) - pts.min(0)
                assert ext.min() >= 1e-3 - 1e-12          # >= 1 mm core: GPU cooking
        for node in [ge.ROOT_NODE] + [c for c in ge.CARRIER_SLOTS if d.slot_valid[c]]:
            pts = ge.plate_points(d, node)
            hull = ConvexHull(pts)
            assert len(hull.vertices) <= 64
            # polygons (faces): 2 caps + one side per outline edge
            assert len(pts) // 2 + 2 <= 64


def test_token_boxes_cover_rounded_links():
    d = ge.canonicalize(SPECIAL["zero_links"])
    boxes = ge.token_boxes(d)
    assert boxes.shape == (36, 4, 3)
    s = ge.finger_slot(0, 1)
    c, h = ge.link_core(d, s)
    extent = 2 * (h + ge.LINK_RADIUS_M)
    side = sorted(np.linalg.norm(boxes[s, 1:] - boxes[s, 0], axis=1))
    assert np.allclose(side, sorted(extent), atol=1e-6)


def test_construction_pairs_cover_0mm_links_and_mounts():
    d = ge.canonicalize(SPECIAL["zero_links"])
    pairs = set(d.filtered_pairs)
    # the base link sits on the palm through a locked carrier: filtered against the root
    assert (ge.ROOT_NODE, ge.finger_slot(0, 0)) in pairs
    # link 1 meets the palm through the 0 mm link 0
    assert (ge.ROOT_NODE, ge.finger_slot(0, 1)) in pairs
    # links 1 and 3 meet through the 0 mm link 2
    assert (ge.finger_slot(0, 1), ge.finger_slot(0, 3)) in pairs
    d2 = ge.canonicalize(SPECIAL["split321"])
    # a follower's finger sits on the leader's section
    assert (ge.carrier_slot(3), ge.finger_slot(4, 0)) in set(d2.filtered_pairs)


def test_overlap_pairs_match_viability_c1():
    for h in _random_hands(30, EVOLUTION_RULES, seed=5):
        d = ge.canonicalize(h)
        a = sorted(round(o * 1e3, 6) for _, _, o in ge.overlap_pairs(d))
        b = sorted(round(o, 6) for _, _, o in gvb.overlaps(h))
        assert a == b


# --------------------------------------------------------------------------
# admission, palm_up
# --------------------------------------------------------------------------


def test_admit_is_c1_for_sampled_and_exempts_commercial():
    bad = Hand(fingers=(Finger(y=0, z=80, joints=(Joint("hinge", (0, 0), 50),)),
                        Finger(y=19, z=80, facing=355, joints=(Joint("hinge", (0, 0), 50),))))
    assert not ge.admit(bad).ok and ge.admit(bad, check_overlap=False).ok
    for hid, rec in build_conformed.load().items():
        assert ge.admit(rec["hand"], check_overlap=False, check_rim=False).ok, hid


def test_palm_up_is_exact():
    d = ge.canonicalize(SPECIAL["split321"])
    pu = ge.palm_up(d)
    up = ge._quat_apply_wxyz(pu.base_rot_wxyz, [1.0, 0.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])
    assert pu.spawn_offset[0] == pytest.approx(0.037 / 2 + 0.03 + 0.005)
    assert np.allclose(ge.tied_q(d, pu.default_q), pu.default_q)
    # the start pose: flexion joints at 0.35 of their upper limit, others 0
    for s in range(ge.N_SLOTS):
        if d.slot_valid[s] and ge.slot_joint_index(s) >= 0:
            lo, hi = d.slot_limits[s]
            assert pu.default_q[s] in (pytest.approx(0.0), pytest.approx(0.35 * hi))


def test_viability_report():
    r = ge.viability_report(SPECIAL["split321"])
    assert set(r) >= {"admitted", "viable", "c1_ok", "c2_ok", "c2_margin_mm", "digit_count", "joint_count"}
    assert r["digit_count"] == 6


# --------------------------------------------------------------------------
# population files
# --------------------------------------------------------------------------


def test_population_round_trip_and_tamper(tmp_path):
    entries = mkpop.collect_sampled(3) + mkpop.collect_commercial(["allegro_right"])
    path = tmp_path / "p.json"
    pf.write_population(path, entries)
    designs = pf.load_population(path)
    assert [d.source for d in designs] == [e.source for e in entries]
    pop = ge.build_population(designs)
    assert pop.joint_valid.shape == (4, 36) and pop.joint_gear.shape == (4, 36)
    doc = json.loads(path.read_text())
    doc["designs"][0]["hand"]["fingers"][0]["y"] += 1
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="sha256"):
        pf.load_population(path)


def test_population_refuses_another_layout(tmp_path):
    path = tmp_path / "p.json"
    pf.write_population(path, mkpop.collect_sampled(1), envelope="grammar_envelope/2")
    with pytest.raises(ValueError, match="layout"):
        pf.load_population(path)


def test_sampled_entries_are_viable():
    entries, rej = pf.sampled_entries(range(40))
    assert entries
    for e in entries:
        assert gvb.is_viable(e.hand) and check(e.hand, EVOLUTION_RULES) == []


@pytest.mark.parametrize("hand_id", sorted(build_conformed.load()))
def test_every_commercial_hand_is_admitted(hand_id):
    entry, status, reason = pf.commercial_entry(hand_id)
    assert status == "admitted", reason
    d = pf._design(entry.hand, entry.source, True)
    # after filtering, no overlapping pair is left unfiltered at zero or at the start pose
    for q in (None, ge.palm_up(d).default_q):
        for a, b, _ in ge.overlap_pairs(d, q):
            assert (min(a, b), max(a, b)) in set(d.filtered_pairs)


def test_validation_population_has_its_hands():
    entries = mkpop.validation_entries(n_random=2)
    sources = [e.source for e in entries]
    for hid in mkpop.VALIDATION_COMMERCIAL_HANDS:
        assert f"commercial:{hid}" in sources
    for name in SPECIAL:
        assert f"sampled:{name}" in sources


# --------------------------------------------------------------------------
# commercial helpers
# --------------------------------------------------------------------------


def test_commercial_slot_values_use_the_zero_pose_difference():
    rec = build_conformed.load()["allegro_right"]
    d = ge.canonicalize(rec["hand"], source="commercial:allegro_right")
    m = ph.joint_slot_map(d, "allegro_right")
    assert len(m) == 16
    zero = ph.slot_values(d, "allegro_right", {n: 0.0 for n in m})
    q_off = ge.dofs_to_slot_q(d, np.asarray(rec["q_off"]))
    for n, s in m.items():
        assert zero[s] == pytest.approx(q_off[s])


def test_urdf_equivalent_placement_keeps_the_palm():
    d = ge.canonicalize(build_conformed.load()["allegro_right"]["hand"], source="commercial:allegro_right")
    entry = {"base_pos": [0.0, 0.0, 0.5], "base_rot": [1.0, 0.0, 0.0, 0.0], "spawn_offset_local": [0.0, 0.0, 0.1]}
    pl = ph.urdf_equivalent_placement(d, "allegro_right", entry)
    T = build_conformed.load()["allegro_right"]["palm_T"]
    # the spawn point is the same physical point in both frames
    p_world_urdf = np.array([0.0, 0.0, 0.6])
    from scipy.spatial.transform import Rotation

    w, x, y, z = pl["base_rot_wxyz"]
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    p_world_grammar = np.asarray(pl["base_pos"]) + R @ np.asarray(pl["spawn_offset"])
    assert np.allclose(p_world_urdf, p_world_grammar, atol=1e-9)
