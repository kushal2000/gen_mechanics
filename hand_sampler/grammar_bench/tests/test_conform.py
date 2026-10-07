"""Conforming commercial hands onto the grammar (fine grid)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from hand_sampler.grammar import build_conformed as BC
from hand_sampler.grammar import commercial as C
from hand_sampler.grammar import conform as CF
from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar.adapters.urdf import load_urdf, to_urdf
from hand_sampler.grammar.hand import EVOLUTION_RULES, NO_RULES, Hand, check, hand_from_dict


def _real_from_hand(hand: Hand) -> C.RealHand:
    text, _ = to_urdf(dv.derive(hand))
    model = load_urdf(text.encode(), hand_root="palm").model
    ann = {"palm_joints": [f"palm{k}_joint" for k in range(len(hand.palm_joints))]}
    return C.real_hand("sampled", model, ann)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("rules", [EVOLUTION_RULES, NO_RULES], ids=["evolution", "none"])
def test_conform_round_trips_a_sampled_hand(seed, rules):
    """A sampled (and mutated) hand, exported to URDF and read back, conforms
    to exactly itself, given its palm frame (a finger pointing straight out of
    the plate may come back with an equivalent facing and axes)."""
    rng = np.random.default_rng(seed)
    h = ops.random_hand(rng, rules, "coarse")
    for _ in range(seed * 3):
        h, _, _ = ops.mutate(h, rng, rules, "fine" if seed % 2 else "coarse")
    fit = CF.conform(_real_from_hand(h), palm_frame=np.eye(4))
    assert fit.max_joint_mm < 1e-3 and fit.max_axis_deg < 1e-3 and fit.max_tip_mm < 1e-3
    for f, g in zip(h.fingers, fit.hand.fingers):
        if abs(f.tilt) == 90:
            # a finger straight out of the plate: its facing only rolls the frame, which the axis
            # directions absorb (and the derived signs follow), so equivalent values may come back
            assert (g.y, g.z, g.tilt, g.palm_joint) == (f.y, f.z, f.tilt, f.palm_joint)
            assert [j.length for j in g.joints] == [j.length for j in f.joints]
        else:
            assert g == f
    assert fit.hand.palm_joints == h.palm_joints


@pytest.mark.parametrize("hand_id", ["allegro_right", "leap_right", "shadow_right_local", "inspire_right"])
def test_conform_finds_the_palm_frame_of_a_conformed_hand(hand_id):
    """Without the palm frame given: a conformed commercial hand, exported to
    URDF, comes back on the same plate (normal +x, origin at the wrist
    centre; z may turn in the plate, it follows the bases' centroid) and
    within the fit target."""
    rec = BC.load().get(hand_id)
    if rec is None:
        pytest.skip(f"local-only:{hand_id} not in the committed file on this machine")
    real = _real_from_hand(rec["hand"])
    thumbs = set(rec["thumbs"])
    for i, f in enumerate(real.fingers):
        f.thumb = i in thumbs
    fit = CF.conform(real)
    assert fit.within_target
    assert np.allclose(fit.palm_T[:3, 0], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(fit.palm_T[:3, 3], 0.0, atol=1e-9)


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


def test_every_conformed_hand_is_in_the_grammar_on_the_fine_grid():
    for rid, rec in BC.load().items():
        assert check(rec["hand"], NO_RULES) == [], (rid, check(rec["hand"], NO_RULES))


def test_conformed_hands_follow_the_evolution_rules_except_sliding_joints():
    for rid, rec in BC.load().items():
        v = check(rec["hand"], EVOLUTION_RULES)
        assert all("sliding joints not allowed" in m for m in v), (rid, v)
        assert bool(v) == any(j.type == "sliding" for f in rec["hand"].fingers for j in f.joints)


@pytest.mark.parametrize("hand_id", ["dex1", "inspire_right", "leap_right", "dex3_left", "wuji2_left", "midas"])
def test_conform_reproduces_the_committed_fit(hand_id):
    """Re-fitting gives the committed hand (fast hands; the rest are checked by
    rebuilding the file)."""
    rec = BC.load().get(hand_id)
    if rec is None:
        pytest.skip(f"local-only:{hand_id} not in the committed file on this machine")
    try:
        real = C.load_real_hand(hand_id)
    except FileNotFoundError:
        pytest.skip(f"local-only:{hand_id} source not on this machine")
    fit = CF.conform(real)
    assert fit.hand == rec["hand"]
    assert fit.max_joint_mm == pytest.approx(rec["max_joint_mm"], abs=1e-2)


def test_fidelity_of_the_committed_fits():
    """The fidelity table: every hand but the reported misses is within 5 mm /
    10 degrees, and a pure zero-pose difference is allowed (q_off)."""
    doc = _doc()
    misses = sorted(r["id"] for r in doc["hands"] if not r["within_target"])
    assert len(misses) <= 2, misses
    for r in doc["hands"]:
        assert r["max_joint_mm"] < 12 and r["max_axis_deg"] < 15 and r["max_tip_mm"] < 10, r["id"]


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
        for i, f in enumerate(real.fingers):
            assert np.linalg.norm(tips[i] - R.T @ (f.tip - o)) * 1e3 <= rec["max_tip_mm"] + 1e-3
