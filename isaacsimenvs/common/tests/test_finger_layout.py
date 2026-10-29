"""The joint -> (finger, slot) derivation on every hand we have. Kit-free."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaacsimenvs.common.robots import REGISTRY, get_robot_spec
from isaacsimenvs.common.robots.finger_layout import (N_FINGERS, N_HAND_SLOTS, N_SLOTS, PALM_SLOT, finger_layout,
                                                      template_layout)

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
    t = template_layout(N_HAND_SLOTS)
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
