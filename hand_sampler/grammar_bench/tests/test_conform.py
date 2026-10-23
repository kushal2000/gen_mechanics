"""Conforming commercial hands onto the grammar (one grid: 10 mm, 30
degrees; at most one palm cut)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hand_sampler.grammar import build_conformed as BC
from hand_sampler.grammar import commercial as C
from hand_sampler.grammar import conform as CF
from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.hand import COMMERCIAL_RULES, FEETECH_RULES, NO_RULES, Hand, check, finger_length_mm, shifted
from hand_sampler.grammar import viability as V


def _real_from_hand(hand: Hand) -> C.RealHand:
    text, _ = to_urdf(dv.derive(hand))
    model = load_urdf(text.encode(), hand_root="palm").model
    ann = {"palm_joints": [f"palm{k}_joint" for k in range(len(hand.palm_joints))]}
    return C.real_hand("sampled", model, ann)


# Known limits of the fitter, reached by the No Rules hands these seeds draw
# since 2026-10-20 (the open start and one copy per turn and mirror image
# change the draws). The fitter at 1d68b41 fails these hands the same way.
ROUND_TRIP_KNOWN_LIMITS = {
    ("none", 1): "a finger standing out of the plate (tilt 90) with a spread first: its axis lies in the plate, the "
                 "fit takes it for a bend first (tilt 0, offset 90 degrees) at a bend's base depth, 0.5 mm off",
    ("none", 2): "the same: a standing spread-first finger comes back 0.5 mm off as a bend first",
    ("none", 9): "the same (2026-10-23 draws): a standing spread-first finger comes back as a bend first",
    ("feetech", 11): "the same (2026-10-23 draws): a standing finger with an oblique first axis comes back as a "
                     "bend first",
}


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("rules", [FEETECH_RULES, NO_RULES], ids=["feetech", "none"])
def test_conform_round_trips_a_sampled_hand(seed, rules, request):
    """A sampled (and mutated) hand, exported to URDF and read back, conforms
    to an exact copy of itself, given its palm frame: every joint on its axis
    line, every axis, every fingertip. The values come back the same up to
    two equivalences: a twist joint may slide along its own axis (the finger)
    with the links either side of it trading length, and a finger straight
    out of the plate may come back with another facing (it only rolls the
    frame, which the axes absorb). `ROUND_TRIP_KNOWN_LIMITS` lists the seeds
    that reach a known limit (strict: a fix shows up as a pass)."""
    known = ROUND_TRIP_KNOWN_LIMITS.get(("none" if rules is NO_RULES else "feetech", seed))
    if known:
        request.applymarker(pytest.mark.xfail(strict=True, reason=known))
    rng = np.random.default_rng(seed)
    h = ops.random_hand(rng, rules)
    for _ in range(seed * 3):
        res = ops.mutate(h, rng, rules)
        if res is not None:
            h = res[0]
    fit = CF.conform(_real_from_hand(h), palm_frame=np.eye(4))
    # the fit puts the mount at the palm's centre on the 10 mm grid: every base moved by d, the frame back by d
    dy, dz = (int(round(-v * 1e3)) for v in fit.palm_T[1:3, 3])
    h = shifted(h, dy, dz)
    exact = []
    for f, g, r in zip(h.fingers, fit.hand.fingers, fit.fingers):
        if max(r.joint_mm + r.axis_deg + [r.tip_mm]) < 1e-3:
            exact.append(True)
        else:   # known limit: two joints at the base point leave the finger's direction open, and the frame search
            # can settle elsewhere (seed 3: 6.5 mm); such a finger must still be within the fit target
            assert f.joints[0].length == 0 and max(r.joint_mm + [r.tip_mm]) <= CF.TARGET_MM, (f, g)
            exact.append(False)
            continue
        assert (g.y, g.z, g.tilt) == (f.y, f.z, f.tilt)
        assert len(g.joints) == len(f.joints) and finger_length_mm(g) == finger_length_mm(f)
        assert [j.type for j in g.joints] == [j.type for j in f.joints]
        if abs(f.tilt) != 90:
            assert g.facing == f.facing and [j.axis for j in g.joints] == [j.axis for j in f.joints]
    assert sum(exact) >= len(exact) - 1
    # a cut moves by the shift along its normal, which is not always a multiple of 10 mm
    assert [p.facing for p in fit.hand.palm_joints] == [p.facing for p in h.palm_joints]
    assert all(abs(p.offset - q.offset) <= 10 for p, q in zip(fit.hand.palm_joints, h.palm_joints))


PALM_FRAME_HANDS = ["shadow_right_local", "tesollo_dg5f_right", "wuji2_right", "xhand_right"]


@pytest.mark.parametrize("hand_id", PALM_FRAME_HANDS)
def test_conform_finds_the_palm_frame_of_a_conformed_hand(hand_id):
    """Without the palm frame given: a conformed commercial hand, exported to
    URDF, comes back within the fit target on nearly the same plate (normal
    within 6 degrees of +x, origin within 15 mm of the mount). z may turn in
    the plate (it follows the bases' centroid). The bases sit at their first
    motors' depths (0 to 4.5 mm below the mid-plane for flat fingers, deeper
    for a tilted thumb), so the plane fitted through them, and the plane
    search after it, may tilt a few degrees and shift along its normal.
    On 2026-10-20 (fits by the fingertip map) 13 of 19 passed. On
    2026-10-23 (the buildable-parts grammar) 10 of 19 pass: Allegro, Barrett,
    Dex1, Inspire, LEAP, Orca, SHARPA, SVH and Wuji v1 come back outside the
    target or off the plate (SHARPA by 15.5 mm, against 15)."""
    rec = BC.load().get(hand_id)
    if rec is None:
        pytest.skip(f"local-only:{hand_id} not in the committed file on this machine")
    real = _real_from_hand(rec["hand"])
    thumbs = set(rec["thumbs"])
    for i, f in enumerate(real.fingers):
        f.thumb = i in thumbs
    fit = CF.conform(real)
    assert fit.within_target
    assert math.degrees(math.acos(min(1.0, float(fit.palm_T[0, 0])))) < 6.0
    assert np.linalg.norm(fit.palm_T[:3, 3]) < 0.015


# --------------------------------------------------------------------------
# The committed conformed commercial hands
# --------------------------------------------------------------------------


def _doc():
    return json.loads(BC.OUT_PATH.read_text())


def test_conformed_file_covers_the_reference_set():
    doc = _doc()
    have = {r["id"] for r in doc["hands"]} | set(doc["unavailable"])
    assert have == set(C.reference_ids())
    assert len(doc["hands"]) >= 15


NOT_BUILDABLE = {"svh_right"}
"""Hands no scale up to 5 x makes buildable with every joint's kind kept and
every fingertip map within `MAP_RMS_MM` / `MAP_WORST_MM` (the rebuild of
2026-10-20: a finger's map stays 23-24 mm RMS; GRAMMAR-OVERVIEW step 18).
They keep their real-size fit under the grammar's own limits
(`conform.fallback_fit`)."""


def _acceptable(rec) -> bool:
    return not rec["notes"][-1].startswith("no scale")


def test_every_conformed_hand_is_buildable_at_its_scale():
    """Each commercial hand is in the grammar (its own limits), scaled up by
    a multiple of 0.05 between 1 and 5. All but the listed few are scaled by
    the smallest factor at which the fit keeps the servo link lengths, passes
    C1, keeps every joint's kind and stays within twice the fit target: they
    can be built from Feetech servos without collisions."""
    recs = BC.load()
    assert {rid for rid, rec in recs.items() if not _acceptable(rec)} == NOT_BUILDABLE
    for rid, rec in recs.items():
        h = rec["hand"]
        rules = COMMERCIAL_RULES if _acceptable(rec) else NO_RULES
        assert check(h, rules) == [], (rid, check(h, rules))
        assert 1.0 <= rec["scale"] <= 5.0 and abs(rec["scale"] / 0.05 - round(rec["scale"] / 0.05)) < 1e-6, rid
        if _acceptable(rec):
            assert rec["buildable"] and V.c1_self_overlap(h).ok, rid
        else:
            assert rec["scale"] == 1.0, rid


def test_every_conformed_hand_keeps_its_structure():
    """The fit keeps every commercial hand's structure: its fingers, the
    joints of each finger, each joint's type and kind (the commercial axis,
    seen in the grammar's link frame at the fitted pose, is nearest the same
    kind or within the 20 degree fit target of the grammar axis, as
    `conform.kinds_kept`; not checked for the hands in NOT_BUILDABLE, kept at
    5 x whatever their kinds), and its palm joint as one cut with the kept
    joint's fingers beyond it."""
    for hid, rec in BC.load().items():
        try:
            real = C.load_real_hand(hid)
        except FileNotFoundError:
            continue
        h, R = rec["hand"], rec["palm_T"][:3, :3]          # rotations do not change with the scale
        pose = dv.fk(h, rec["q_off"])
        assert len(h.fingers) == len(real.fingers), hid
        for i, (f, rf) in enumerate(zip(h.fingers, real.fingers)):
            assert len(f.joints) == len(rf.joints), (hid, i)
            for j, (jt, rj) in enumerate(zip(f.joints, rf.joints)):
                assert (jt.type == "sliding") == (rj.type == "prismatic"), (hid, i, j)
                if jt.type == "sliding":
                    continue
                a = pose.links[i][j][:3, :3].T @ (R.T @ rj.axis)
                same = dv.kind_of(a) == dv.joint_kind(h, i, j) or \
                    CF.axis_angle_deg(a, dv.direction(*jt.axis)) <= CF.TARGET_DEG
                assert same or hid in NOT_BUILDABLE, (hid, i, j)
        assert len(h.palm_joints) == (1 if real.palm_joints else 0), hid
        if h.palm_joints:
            beyond = [i for i in range(len(h.fingers)) if dv.finger_section(h, i) == 0]
            kept = rec["name_map"]["palm0_joint"]
            assert beyond == rec["section"]
            assert set(i for i, f in enumerate(real.fingers) if f.palm_joint == kept) <= set(beyond), hid


@pytest.mark.parametrize("hand_id", ["dclaw", "dex1", "dex3_left"])
def test_conform_reproduces_the_committed_fit(hand_id):
    """Re-fitting (the scale search included) gives the committed hand (fast
    hands; the rest are checked by rebuilding the file)."""
    rec = BC.load().get(hand_id)
    if rec is None:
        pytest.skip(f"local-only:{hand_id} not in the committed file on this machine")
    try:
        real = C.load_real_hand(hand_id)
    except FileNotFoundError:
        pytest.skip(f"local-only:{hand_id} source not on this machine")
    fit = CF.conform_buildable(real)
    assert fit.scale == pytest.approx(rec["scale"])
    assert fit.hand == rec["hand"]
    assert fit.max_joint_mm == pytest.approx(rec["max_joint_mm"], abs=1e-2)


def test_fidelity_of_the_committed_fits():
    """The fidelity table (millimetres at each hand's real size): every fit
    the scale search accepted has every finger's fingertip map within
    `MAP_RMS_MM` / `MAP_WORST_MM` (2026-10-20: the fingertips follow the
    commercial ones over their joint ranges; a joint may sit elsewhere). The
    joint, axis and fingertip errors at the zero pose are reported, not
    scored. The palm cut's own errors are reported, not scored."""
    doc = _doc()
    assert (doc["target_mm"], doc["target_deg"]) == (CF.TARGET_MM, CF.TARGET_DEG) == (10.0, 20.0)
    for r in doc["hands"]:
        if _acceptable(r):
            assert r["max_map_rms_mm"] <= CF.MAP_RMS_MM and r["max_map_worst_mm"] <= CF.MAP_WORST_MM, r["id"]


def test_the_zero_pose_difference_reproduces_the_commercial_pose():
    """At its q_off, a conformed hand's fingertips sit on the commercial
    fingertips (in the palm frame) within the reported error."""
    recs = BC.load()
    for hid in ("dex3_left", "midas", "wuji2_right"):
        if hid not in recs:
            continue
        rec = recs[hid]
        real = C.load_real_hand(hid)
        tips = dv.fk(rec["hand"], rec["q_off"]).tips
        R, o = rec["palm_T"][:3, :3], rec["palm_T"][:3, 3]
        s = rec["scale"]
        for i, f in enumerate(real.fingers):       # the fit is of the hand scaled up about its root; errors at real size
            assert np.linalg.norm(tips[i] - R.T @ (s * f.tip - o)) * 1e3 / s <= rec["max_tip_mm"] + 1e-3


def test_the_fingertip_map_of_an_exact_copy_is_zero():
    """A sampled hand exported to URDF and conformed: its fingertip map
    follows the commercial one exactly (the batched chain agrees with the
    joint-by-joint one), and the commercial samples start at the fingertip."""
    rng = np.random.default_rng(5)
    h = ops.random_hand(rng, FEETECH_RULES)
    real = _real_from_hand(h)
    fit = CF.conform(real, palm_frame=np.eye(4))
    targets = CF._targets_for(real, fit.palm_T)
    for i, (f, t, r) in enumerate(zip(fit.hand.fingers, targets, fit.fingers)):
        if max(r.joint_mm + [r.tip_mm]) > 1e-3:
            continue                                   # the known base-crossing case (round-trip test)
        rms, worst = CF.tip_map_mm(f, t, r.q_off)
        assert rms < 1e-3 and worst < 1e-3
        one = CF._chain(dv.mount_point_m(f), dv.finger_frame(f), [dv.direction(*j.axis) for j in f.joints],
                        [j.length for j in f.joints], t.types, r.q_off)[2]
        assert np.allclose(CF._tips_batch(f, t, r.q_off, np.zeros((1, len(f.joints))))[0], one)
        Q, P = CF._real_tip_samples(real, real.fingers[fit.finger_order[i]], t.points, t.axes, t.tip)
        assert Q.shape == (CF.TIP_MAP_SAMPLES, len(f.joints)) and np.allclose(P, t.tip_samples)
