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
    to exactly itself, given its palm frame."""
    rng = np.random.default_rng(seed)
    h = ops.random_hand(rng, rules, "coarse")
    for _ in range(seed * 3):
        h, _, _ = ops.mutate(h, rng, rules, "fine" if seed % 2 else "coarse")
    fit = CF.conform(_real_from_hand(h), palm_frame=np.eye(4))
    assert fit.hand == h
    assert fit.max_joint_mm < 1e-3 and fit.max_axis_deg < 1e-3 and fit.max_tip_mm < 1e-3


def test_conform_finds_the_palm_frame_of_a_typical_hand():
    """Without the palm frame given: a row of bending fingers and a thumb
    (the fingers close toward the grasp side, which fixes the normal) comes back
    on the same plate (the in-plane direction of z follows the bases'
    centroid, so it may turn) and within the fit target."""
    rng = np.random.default_rng(100)
    found = 0
    for _ in range(40):
        h = ops.random_hand(rng, EVOLUTION_RULES, "coarse")
        nonthumb = [f for f in h.fingers if f.facing == 0]
        flexing = [f for f in nonthumb if any(j.axis == (0, 0) for j in f.joints)]
        if len(h.fingers) < 4 or len(flexing) < 3 or h.palm_joints:
            continue
        real = _real_from_hand(h)
        for f, g in zip(real.fingers, h.fingers):
            f.thumb = g.facing != 0
        fit = CF.conform(real)
        assert fit.within_target
        assert np.allclose(fit.palm_T[:3, 0], [1.0, 0.0, 0.0], atol=1e-6)     # the same plate and normal
        assert np.allclose(fit.palm_T[:3, 3], 0.0, atol=1e-9)                # the wrist centre
        found += 1
    assert found >= 3


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
