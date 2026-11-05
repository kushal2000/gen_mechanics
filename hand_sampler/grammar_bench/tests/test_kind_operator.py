"""The `kind` operator (6 Nov): on the 90-degree axis grid, any joint becomes any other kind in one edit."""
import numpy as np

from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as ops

RULES = gh.FEETECH_COARSE_NOCUT_RULES
KINDS = set(ops.JOINT_KINDS.values())


def _hands(n=6, seed=3):
    rng = np.random.default_rng(seed)
    return [ops.random_hand(rng, RULES) for _ in range(n)]


def test_every_joint_has_both_other_kinds_as_moves():
    for h in _hands():
        moves = ops.OPERATOR_BY_NAME["kind"].moves(h, RULES)
        n_joints = sum(len(f.joints) for f in h.fingers)
        assert len(moves) == 2 * n_joints
        for i, j, k in moves:
            assert ops.JOINT_KINDS[k] != gh.canonical_axis(*h.fingers[i].joints[j].axis)


def test_a_kind_child_changes_exactly_that_joints_axis():
    for h in _hands():
        op = ops.OPERATOR_BY_NAME["kind"]
        for (i, j, k), child in ops.iter_valid(h, op, RULES):
            raw = op.apply(h, (i, j, k), RULES)
            assert raw.fingers[i].joints[j].axis == ops.JOINT_KINDS[k]
            for a, (f0, f1) in enumerate(zip(h.fingers, raw.fingers)):
                for b, (j0, j1) in enumerate(zip(f0.joints, f1.joints)):
                    if (a, b) != (i, j):
                        assert j0.axis == j1.axis
            assert {tuple(jt.axis) for f in child.fingers for jt in f.joints} <= KINDS


def test_axis_steps_are_for_the_fine_grid_only():
    h = _hands(1)[0]
    assert ops.OPERATOR_BY_NAME["axis_az"].moves(h, RULES) == []
    assert ops.OPERATOR_BY_NAME["axis_el"].moves(h, RULES) == []
    assert ops.OPERATOR_BY_NAME["kind"].moves(h, gh.FEETECH_RULES) == []      # 30-degree grid: axis_az / axis_el
