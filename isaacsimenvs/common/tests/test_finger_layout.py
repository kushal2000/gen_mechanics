"""The joint -> (finger, slot) derivation on every hand we have. Kit-free."""
from __future__ import annotations

import json

import numpy as np
from pathlib import Path

import pytest

from isaacsimenvs.common.robots import REGISTRY, get_robot_spec
from isaacsimenvs.common.robots.finger_layout import (DEFAULT_GRID, N_FINGERS, N_HAND_SLOTS, N_SLOTS, PALM_SLOT,
                                                      SlotGrid, finger_layout, grid_for, template_layout)

ROOT = Path(__file__).resolve().parents[3]
GRAMMAR = ROOT / "assets/urdf/27oct_grammar_random_hands"


def test_commercial_hands_split_into_their_fingers():
    expect = {"allegro_left_uniform_handonly": [4, 4, 4, 4], "dex3_left_uniform_handonly": [3, 2, 2],
              "shadow_left_uniform_handonly": [4, 4, 4, 5, 5], "wuji2_left_uniform_handonly": [4, 4, 4, 4, 4],
              "wuji2_left_uniform_handonly_only_index_middle": [4, 4]}
    for name, lengths in expect.items():
        lay = finger_layout(get_robot_spec(name))
        assert [len(f) for f in lay.fingers] == lengths, name
        assert lay.palm == ()
        assert len(set(lay.slot_of_joint)) == lay.n_joints


def test_every_registered_hand_only_spec_fits_the_grid():
    for name, spec in REGISTRY.items():
        if spec.num_arm_joints or not spec.urdf_path:
            continue
        lay = finger_layout(spec)
        lay.validate(name)
        assert len(lay.fingers) == spec.num_fingertips, name


@pytest.mark.parametrize("hand", ["coarse_s0_000", "coarse_s0_025", "coarse_s0_099"])
def test_grammar_hands_match_their_genotype(hand):
    spec = get_robot_spec(str(GRAMMAR / hand / f"{hand}.spec.json"))
    design = json.loads((GRAMMAR / hand / f"{hand}.design.json").read_text())
    lay = finger_layout(spec)
    assert [len(f) for f in lay.fingers] == [len(f["joints"]) for f in design["fingers"]]
    assert len(lay.palm) == len(design["palm_joints"])
    names = spec.hand_joint_names
    for i, f in enumerate(lay.fingers):                          # palm -> tip is f{i}_j0, f{i}_j1, ...
        assert [names[j] for j in f] == [f"f{i}_j{k}" for k in range(len(f))]
    if lay.palm:
        assert names[lay.palm[0]] == "palm0_joint" and lay.slot_of_joint[lay.palm[0]] == PALM_SLOT


def test_all_hundred_grammar_hands_fit():
    for d in sorted(p for p in GRAMMAR.iterdir() if p.is_dir()):
        finger_layout(get_robot_spec(str(d / f"{d.name}.spec.json"))).validate(d.name)


def test_slots_round_trip_and_the_template_is_finger_major():
    lay = finger_layout(get_robot_spec("shadow_left_uniform_handonly"))
    jos = lay.joint_of_slot
    assert len(jos) == N_HAND_SLOTS and sum(j >= 0 for j in jos) == lay.n_joints
    assert all(jos[s] == j for j, s in enumerate(lay.slot_of_joint))
    t = template_layout(DEFAULT_GRID)
    assert t.slot_of_joint == tuple(range(N_HAND_SLOTS))
    assert len(t.fingers) == N_FINGERS and all(len(f) == N_SLOTS for f in t.fingers) and t.palm == (PALM_SLOT,)


def test_a_finger_major_hand_set_scatters_each_hand_to_its_slots():
    from isaacsimenvs.common.robots import get_finger_layout
    from isaacsimenvs.common.robots.multi_hand import hand_set, slot_layout, spec_names
    ref = "multi:finger:shadow_left_uniform_handonly+dex3_left_uniform_handonly"
    assert slot_layout(ref) == "finger" and slot_layout("multi:uniform") == "dense"
    assert spec_names(ref) == ("shadow_left_uniform_handonly", "dex3_left_uniform_handonly")
    hs = hand_set(ref)
    assert (hs.n_joints, hs.n_tips, hs.layout) == (N_HAND_SLOTS, N_FINGERS, "finger")
    assert len(hs.template.hand_joint_names) == N_HAND_SLOTS
    for h, spec in enumerate(hs.specs):
        lay = finger_layout(spec)
        assert hs.joint_slots[h] == lay.slot_of_joint
        assert list(hs.joint_valid[h].nonzero()[0]) == sorted(lay.slot_of_joint)
        assert hs.fingertip_valid[h].sum() == spec.num_fingertips == len(lay.fingers)
    dense = hand_set("multi:shadow_left_uniform_handonly+dex3_left_uniform_handonly")
    assert dense.layout == "dense" and dense.n_joints == 22 and dense.joint_slots[1] == tuple(range(7))
    assert get_finger_layout(ref).slot_of_joint == tuple(range(N_HAND_SLOTS))
    with pytest.raises(ValueError):
        get_finger_layout("multi:shadow_left_uniform_handonly+dex3_left_uniform_handonly")


def test_cut_hands_mark_their_moving_fingers_and_the_set_sizes_its_grid():
    from isaacsimenvs.common.robots.multi_hand import hand_set
    lays = {d.name: finger_layout(get_robot_spec(str(d / f"{d.name}.spec.json")))
            for d in sorted(p for p in GRAMMAR.iterdir() if p.is_dir())}
    cut = {n: l for n, l in lays.items() if l.palm}
    assert cut and all(1 <= sum(l.moving) <= 2 for l in cut.values())        # the data: a cut moves 1 or 2 fingers
    assert all(l.moving == () for n, l in lays.items() if n not in cut)
    g = grid_for(lays.values())
    assert (g.n_rigid, g.n_moving, g.n_fingers, g.n_hand_slots) == (5, 2, 7, 36)
    for n, l in lays.items():                                                # placement respects the sections
        p = l.with_grid(g)
        for f, s in enumerate(p.finger_slot_of_finger):
            if l.palm:
                assert g.is_moving_slot(s) == l.moving[f], n
        assert sorted(j for j in p.joint_of_slot if j >= 0) == list(range(l.n_joints))
        assert len(set(p.finger_slot_of_finger)) == l.n_fingers
    # a hand set on that grid: the template is 36 slots + 7 tips; tips sit at their finger's slot
    ref = "multi:template:" + "+".join(str(GRAMMAR / n / f"{n}.spec.json") for n in ("coarse_s0_000", "coarse_s0_025"))
    hs = hand_set(ref)
    assert hs.layout == "template" and hs.grid == grid_for([lays["coarse_s0_000"], lays["coarse_s0_025"]])
    assert hs.n_joints == hs.grid.n_hand_slots and hs.n_tips == hs.grid.n_fingers
    for h in range(2):
        assert list(hs.fingertip_valid[h].nonzero()[0]) == sorted(hs.finger_slots[h])
        assert hs.joint_slots[h] == hs.layouts[h].slot_of_joint
    two = next(l for l in cut.values() if sum(l.moving) == 2)
    with pytest.raises(ValueError):
        two.with_grid(SlotGrid(n_rigid=6, n_moving=1))                       # two moving fingers, one moving slot
    assert two.with_grid(SlotGrid(n_rigid=6, n_moving=0)).finger_slot_of_finger == tuple(range(two.n_fingers))


def test_joint_moving_flags_follow_the_cut():
    from isaacsimenvs.common.robots.finger_layout import joint_moving, joint_moving_of
    from isaacsimenvs.common.robots.multi_hand import hand_set
    spec = get_robot_spec(str(GRAMMAR / "coarse_s0_000/coarse_s0_000.spec.json"))
    lay = finger_layout(spec)
    mv = joint_moving(lay)
    assert mv == joint_moving_of(spec) and len(mv) == spec.num_hand_joints
    names = spec.hand_joint_names
    assert all(names[j].startswith("f4_") == m for j, m in enumerate(mv))     # hand 000: finger 4 rides the cut
    assert not any(joint_moving_of(get_robot_spec("shadow_left_uniform_handonly")))
    hs = hand_set("multi:template:" + str(GRAMMAR / "coarse_s0_000/coarse_s0_000.spec.json"))
    t = hs.per_env(np.zeros(3, dtype=np.int64))["joint_moving"]
    assert t.shape == (3, hs.n_joints) and t[0][list(lay.with_grid(hs.grid).slot_of_joint)].tolist() == list(mv)
