"""Multi-hand scene layout (robots/multi_hand.py): slots, per-env tables, env blocks. Kit-free."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mh():
    from isaacsimenvs.common.robots import multi_hand
    return multi_hand


def test_template_pads_to_the_largest_hand(mh):
    hs = mh.hand_set("multi:uniform")
    assert hs.n_hands == 8
    assert hs.n_joints == max(s.num_hand_joints for s in hs.specs) == 22
    assert hs.n_tips == 5
    t = hs.template
    assert t.num_hand_joints == 22 and t.num_fingertips == 5 and t.num_arm_joints == 0
    assert t.hand_joint_names == tuple(f"s{k}" for k in range(22))
    t.validate()


def test_per_env_tables_follow_each_hand(mh):
    hs = mh.hand_set("multi:allegro_left_uniform_handonly+dex3_left_uniform_handonly")
    idx = hs.hand_of_env(10)
    assert idx.tolist() == [0, 1] * 5                         # round-robin across envs
    t = hs.per_env(idx)
    allegro, dex3 = hs.specs
    assert t["joint_valid"][0].sum() == allegro.num_hand_joints == 16
    assert t["joint_valid"][9].sum() == dex3.num_hand_joints == 7
    assert t["fingertip_valid"][0].sum() == 4 and t["fingertip_valid"][9].sum() == 3
    np.testing.assert_allclose(t["joint_link_bbox_local"][9, :7], np.asarray(dex3.joint_link_boxes, np.float32))
    assert not t["joint_link_bbox_local"][9, 7:].any()          # ghost slots carry nothing
    # every 2048-env SAPG block holds every hand
    full = mh.hand_set("multi:uniform").hand_of_env(12288)
    for b in range(6):
        assert len(np.unique(full[b * 2048:(b + 1) * 2048])) == 8
    np.testing.assert_allclose(t["palm_keypoints"][0], np.asarray(allegro.palm_keypoints, np.float32))
    np.testing.assert_allclose(t["fingertip_offsets"][9, :3], np.asarray(dex3.fingertip_offsets, np.float32))


def test_uneven_split_and_bad_refs(mh):
    hs = mh.hand_set("multi:uniform")
    sizes = np.bincount(hs.hand_of_env(12289), minlength=8)
    assert sizes.max() - sizes.min() <= 1 and sizes.sum() == 12289
    with pytest.raises(KeyError):
        mh.hand_set("multi:not_a_hand")


def test_get_robot_spec_takes_a_multi_ref_of_spec_files(mh):
    """A multi ref whose last member is a spec JSON path ends in .spec.json too; it is still a hand set (the
    joint transformer looks the template up this way)."""
    from isaacsimenvs.common.robots import get_robot_spec
    a = "assets/urdf/unified_dynamics_commercial_hands/allegro/allegro_left.spec.json"
    b = "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.spec.json"
    t = get_robot_spec(f"multi:{a}+{b}")
    assert t.num_hand_joints == max(get_robot_spec(a).num_hand_joints, get_robot_spec(b).num_hand_joints)


def test_a_manifest_names_its_whole_family(mh):
    hs = mh.hand_set("multi:assets/urdf/27oct_grammar_random_hands/manifest.json")
    assert hs.n_hands == 100
    assert hs.specs[0].hand_name == "coarse_s0_000" and hs.specs[-1].hand_name == "coarse_s0_099"
