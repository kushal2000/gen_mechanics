"""ActionCfg.finger_target: the "range" and "home" maps between finger actions and joint targets. Kit-free."""
from __future__ import annotations

import torch

from isaacsimenvs.inhand_olympics import action_map


class _Stub:
    def __init__(self, mode, lo, hi, home):
        self.mode, self.lo, self.hi, self.home = mode, lo, hi, home

    def action_to_target(self, a):
        return action_map.action_to_target(a, self.lo, self.hi, self.home, self.mode)

    def target_to_action(self, t):
        return action_map.target_to_action(t, self.lo, self.hi, self.home, self.mode)


LO = torch.tensor([[-0.5, 0.0, -1.0, 0.0]])        # a bend (-30..90 deg), a spread-like, a twist, a ghost (0, 0)
HI = torch.tensor([[1.57, 0.6, 1.0, 0.0]])
HOME = torch.tensor([[0.0, 0.1, 0.0, 0.0]])


def test_range_zero_is_the_middle_and_round_trips():
    e = _Stub("range", LO, HI, HOME)
    assert torch.allclose(e.action_to_target(torch.zeros(1, 4)), (LO + HI) / 2)
    a = torch.linspace(-1, 1, 9).repeat(4, 1).T.unsqueeze(0).reshape(9, 4)
    for row in a:
        t = e.action_to_target(row[None])
        assert torch.allclose(e.target_to_action(t)[:, :3], row[None, :3], atol=1e-6)


def test_home_zero_is_the_default_pose_and_the_limits_are_plus_minus_one():
    e = _Stub("home", LO, HI, HOME)
    assert torch.allclose(e.action_to_target(torch.zeros(1, 4)), HOME)
    assert torch.allclose(e.action_to_target(torch.ones(1, 4)), HI)
    assert torch.allclose(e.action_to_target(-torch.ones(1, 4)), LO)
    # asymmetric gains: +0.5 goes halfway to the upper limit, -0.5 halfway to the lower
    assert torch.allclose(e.action_to_target(torch.full((1, 4), 0.5)), HOME + 0.5 * (HI - HOME))
    assert torch.allclose(e.action_to_target(torch.full((1, 4), -0.5)), HOME - 0.5 * (HOME - LO))
    for v in (-1.0, -0.3, 0.0, 0.4, 1.0):
        t = e.action_to_target(torch.full((1, 4), v))
        assert torch.allclose(e.target_to_action(t)[:, :3], torch.full((1, 3), v), atol=1e-6)
    # a ghost slot (limits 0, 0) is 0 whatever the action, and reads back as 0
    assert float(e.action_to_target(torch.full((1, 4), 0.7))[0, 3]) == 0.0
    assert float(e.target_to_action(torch.zeros(1, 4))[0, 3]) == 0.0
