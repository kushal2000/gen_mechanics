"""The per-finger token mode (token_mode: finger): the static joint -> finger-slot maps, the decoder's scatter back to
canonical order, ghosts masked, and the finger-major multi-hand template. CPU, small widths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from coevolution.networks.joint_transformer import JointTransformerNet, _WithArmJoints
from isaacsimenvs.common.obs_layout import field_offsets
from isaacsimenvs.common.robots import get_finger_layout, get_robot_spec
from isaacsimenvs.common.robots.finger_layout import N_FINGERS, N_HAND_SLOTS, N_SLOTS, PALM_SLOT
from isaacsimenvs.common.robots.multi_hand import hand_set

ROOT = Path(__file__).resolve().parents[3]
GRAMMAR = ROOT / "assets/urdf/27oct_grammar_random_hands"
FIELDS = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets", "joint_link_bbox",
          "joint_lower", "joint_upper", "joint_enabled", "palm_pos", "palm_rot", "keypoints_rel_palm",
          "keypoints_rel_goal", "task_onehot"]
EXTRA = {"task_onehot": 8}
N_WRIST = 6
MULTI = "multi:finger:allegro_left_uniform_handonly+dex3_left_uniform_handonly"


def _grammar_hand(with_palm_joint: bool) -> str:
    for d in sorted(p for p in GRAMMAR.iterdir() if p.is_dir()):
        design = json.loads((d / f"{d.name}.design.json").read_text())
        if bool(design["palm_joints"]) == with_palm_joint:
            return str(d / f"{d.name}.spec.json")
    raise AssertionError


def _params(robot_spec, **over):
    p = dict(robot_spec=robot_spec, obs_list=FIELDS, extra_arm_joints=N_WRIST, extra_field_sizes=EXTRA,
             token_mode="finger", d_model=32, n_layers=2, n_heads=4, ff_mult=2, decoder_units=[32],
             arm_head_units=[32], value_head_units=[32], affine_norm=False, masked_arm_dims=[3, 4, 5],
             space={"continuous": {"fixed_sigma": "fixed"}})
    p.update(over)
    return p


def _net_and_obs(robot_spec, enabled: torch.Tensor, **over):
    """A finger-mode net on ``robot_spec`` (+ the 6 wrist joints) and a random observation whose joint_enabled
    columns are ``enabled`` (B, n_hand)."""
    spec = _WithArmJoints(get_robot_spec(robot_spec), N_WRIST)
    off = field_offsets(FIELDS, spec, EXTRA)
    obs_dim = max(e for _, e in off.values())
    torch.manual_seed(0)
    net = JointTransformerNet(_params(robot_spec, **over), actions_num=N_WRIST + spec.num_hand_joints,
                              input_shape=(obs_dim,)).eval()
    obs = torch.randn(enabled.shape[0], obs_dim)
    s, e = off["joint_enabled"]
    obs[:, s:e] = enabled.float()
    return net, obs, off


def test_shapes_palm_joint_and_locked_wrist_dims():
    hand = _grammar_hand(with_palm_joint=True)
    lay = get_finger_layout(hand)
    n_hand = lay.n_joints
    net, obs, _ = _net_and_obs(hand, torch.ones(4, n_hand, dtype=torch.bool))
    with torch.no_grad():
        mu, sigma, value, _ = net({"obs": obs})
    assert mu.shape == (4, N_WRIST + n_hand) and sigma.shape == mu.shape and value.shape == (4, 1)
    assert torch.isfinite(mu).all() and torch.isfinite(value).all()
    assert int(net.slot_of_joint[lay.palm[0]]) == PALM_SLOT
    assert net.slot_gather.shape == (N_HAND_SLOTS,) and net.token_proj.in_features == N_SLOTS * (net.token_dim + 1)
    assert net.arm_head[-1].out_features == N_WRIST + 1          # the palm-cut joint's action
    assert (mu[:, [3, 4, 5]] == 0).all() and (sigma[:, [3, 4, 5]] == 0).all()
    assert (mu[:, N_WRIST:] != 0).all()                          # every real hand joint acts


def test_a_finger_acts_from_its_own_token_only():
    """Without attention (n_layers 0) the finger slots are independent: perturbing finger 0's joint columns moves
    only finger 0's actions -- the gather into slots and the scatter back are the right static maps."""
    hand = _grammar_hand(with_palm_joint=False)
    lay = get_finger_layout(hand)
    n_hand = lay.n_joints
    net, obs, off = _net_and_obs(hand, torch.ones(3, n_hand, dtype=torch.bool), n_layers=0)
    obs2 = obs.clone()
    cols = []
    for field in ("joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets"):
        s, _ = off[field]
        cols += [s + N_WRIST + j for j in lay.fingers[0]]
    for field, w in (("joint_link_bbox", 12), ("joint_lower", 1), ("joint_upper", 1)):
        s, _ = off[field]
        cols += [s + j * w + i for j in lay.fingers[0] for i in range(w)]
    obs2[:, cols] += 1.0
    with torch.no_grad():
        mu1, _, _, _ = net({"obs": obs})
        mu2, _, _, _ = net({"obs": obs2})
    f0 = [N_WRIST + j for j in lay.fingers[0]]
    rest = [i for i in range(N_WRIST, N_WRIST + n_hand) if i not in f0]
    assert not torch.allclose(mu1[:, f0], mu2[:, f0])
    assert torch.allclose(mu1[:, rest], mu2[:, rest], atol=1e-6)
    assert torch.allclose(mu1[:, :N_WRIST], mu2[:, :N_WRIST], atol=1e-6)   # no palm cut: the wrist sees nothing


def test_finger_major_template_and_ghost_slots():
    hs = hand_set(MULTI)
    assert hs.layout == "finger" and hs.n_joints == N_HAND_SLOTS and hs.n_tips == N_FINGERS
    for h, spec in enumerate(hs.specs):
        lay = get_finger_layout(spec.name)
        assert hs.joint_slots[h] == lay.slot_of_joint
        assert hs.joint_valid[h].sum() == spec.num_hand_joints
        assert hs.joint_valid[h][list(lay.slot_of_joint)].all()
        boxes = torch.as_tensor(hs.joint_link_boxes[h])[list(lay.slot_of_joint)]
        assert torch.allclose(boxes, torch.tensor(spec.joint_link_boxes))
    assert get_finger_layout(MULTI).slot_of_joint == tuple(range(N_HAND_SLOTS))
    # The template net: per-env ghosts (from joint_enabled) get no action.
    enabled = torch.as_tensor(hs.per_env(hs.hand_of_env(6))["joint_valid"])
    net, obs, _ = _net_and_obs(MULTI, enabled)
    with torch.no_grad():
        mu, sigma, value, _ = net({"obs": obs})
    assert mu.shape == (6, N_WRIST + N_HAND_SLOTS)
    ghost = torch.cat([torch.zeros(6, N_WRIST, dtype=torch.bool), ~enabled], 1)
    assert (mu[ghost] == 0).all() and (sigma[ghost] == 0).all()
    real = torch.cat([torch.zeros(6, N_WRIST, dtype=torch.bool), enabled], 1)
    assert (mu[real] != 0).all()


def test_dense_multi_ref_is_refused_in_finger_mode():
    dense = "multi:allegro_left_uniform_handonly+dex3_left_uniform_handonly"
    with pytest.raises(ValueError, match="multi:finger:"):
        get_finger_layout(dense)
    with pytest.raises(ValueError, match="hand_global_skip"):
        _net_and_obs(_grammar_hand(False), torch.ones(1, 1, dtype=torch.bool), hand_global_skip=True)


def test_joint_mode_is_unchanged_by_default():
    hand = _grammar_hand(with_palm_joint=True)
    n_hand = get_finger_layout(hand).n_joints
    net, obs, _ = _net_and_obs(hand, torch.ones(2, n_hand, dtype=torch.bool), token_mode="joint")
    assert hasattr(net, "mu_head") and not hasattr(net, "decoder") and not hasattr(net, "slot_gather")
    with torch.no_grad():
        mu, _, _, _ = net({"obs": obs})
    assert mu.shape == (2, N_WRIST + n_hand)


def test_packed_tokens_refuses_what_it_cannot_do():
    hand = _grammar_hand(with_palm_joint=False)
    n_hand = get_finger_layout(hand).n_joints
    for over, msg in (({"compile_net": False}, "compile_net"), ({"mask_ghost_actions": False}, "mask_ghost_actions"),
                      ({"parallel_block": True, "compile_net": True}, "parallel_block"),
                      ({"compile_net": True}, "head dim")):
        with pytest.raises(ValueError, match=msg):
            _net_and_obs(hand, torch.ones(1, n_hand, dtype=torch.bool), packed_tokens=True,
                         **({"compile_net": True} | over))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention runs on CUDA")
@pytest.mark.parametrize("mode", ["finger", "joint"])
def test_packed_tokens_match_padded_on_a_mixed_template_set(mode):
    """The packed encoder (real tokens only, FlexAttention document mask) gives the padded network's outputs on every
    real dimension, per env over a set with ghosts, a cut hand and a spilled finger -- and stays right when a batch
    outgrows its capacity (one regrow)."""
    ref = "multi:template:" + "+".join(str(GRAMMAR / n / f"{n}.spec.json")
                                       for n in ("coarse_s0_026", "coarse_s0_000", "coarse_s0_006"))
    hs = hand_set(ref)
    enabled = torch.as_tensor(hs.per_env(hs.hand_of_env(600))["joint_valid"])
    net0, obs, _ = _net_and_obs(ref, enabled, token_mode=mode, compile_net=True, d_model=64)    # head dim 16
    net1, _, _ = _net_and_obs(ref, enabled, token_mode=mode, compile_net=True, packed_tokens=True, d_model=64)
    net1.load_state_dict(net0.state_dict())
    net0, net1, obs = net0.cuda().eval(), net1.cuda().eval(), obs.cuda()
    with torch.no_grad():
        a, b = net0({"obs": obs}), net1({"obs": obs})
    for x, y in zip(a[:3], b[:3]):
        assert torch.allclose(x, y, atol=1e-5), float((x - y).abs().max())
    assert 600 in net1._pack_caps
    # a batch of the same size with every slot real overflows the capacity: it regrows and is still exact
    obs2 = obs.clone()
    obs2[:, net1.enabled_index] = 1.0
    with torch.no_grad():
        a, b = net0({"obs": obs2}), net1({"obs": obs2})
    assert net1._pack_caps[600] >= 600 * (net1.n_fingers + 1 if mode == "finger" else net1.n_hand + 1)
    for x, y in zip(a[:3], b[:3]):
        assert torch.allclose(x, y, atol=1e-5), float((x - y).abs().max())
