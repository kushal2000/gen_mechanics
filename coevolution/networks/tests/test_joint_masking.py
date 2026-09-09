"""A padded design's ghost joints must not reach the policy."""

from __future__ import annotations

import pytest
import torch

from coevolution.networks.joint_transformer import _EncoderLayer, _ParallelEncoderLayer


@pytest.fixture(params=[_EncoderLayer, _ParallelEncoderLayer], ids=["serial", "parallel"])
def block(request):
    torch.manual_seed(0)
    return request.param(d_model=64, n_heads=1, ff_mult=2, dropout=0.0).eval()


def test_all_valid_mask_is_a_no_op(block):
    """SHARPA has no ghosts, so masking must not perturb it at all."""
    x = torch.randn(4, 31, 64)
    with torch.no_grad():
        assert torch.equal(block(x), block(x, torch.ones(4, 31, dtype=torch.bool)))


def test_masked_keys_change_the_output_and_stay_finite(block):
    x = torch.randn(4, 31, 64)
    mask = torch.ones(4, 31, dtype=torch.bool)
    mask[:, 5:-1] = False                      # the global token is never masked
    with torch.no_grad():
        out = block(x, mask)
    assert torch.isfinite(out).all(), "an all -inf row would softmax to NaN"
    assert not torch.allclose(out, block(x))


def test_a_ghost_token_cannot_influence_a_real_one(block):
    """Changing ghost content must leave the real tokens untouched."""
    x = torch.randn(2, 31, 64)
    mask = torch.ones(2, 31, dtype=torch.bool)
    mask[:, 3:-1] = False
    y = x.clone()
    y[:, 3:-1] = torch.randn_like(y[:, 3:-1])
    with torch.no_grad():
        a, b = block(x, mask), block(y, mask)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)


def test_masked_pooling_ignores_ghosts():
    joints = torch.randn(4, 30, 64)
    valid = torch.zeros(4, 30, dtype=torch.bool)
    valid[:, :3] = True
    w = valid.unsqueeze(-1).to(joints.dtype)
    pooled = (joints * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    assert torch.allclose(pooled, joints[:, :3].mean(dim=1), atol=1e-6)


# --- per-design shape layouts (materials.py) ---------------------------------

def _materials_helpers():
    """Load the pure-python helpers without materials.py's Kit imports."""
    src = open("isaacsimenvs/pose_reaching_6d/scene_utils/materials.py").read()
    head = src[:src.index("def _bucketed(")]
    head = "\n".join(l for l in head.splitlines()
                     if not l.startswith(("from .", "import torch", "from isaaclab")))
    ns: dict = {"__name__": "materials_helpers"}
    exec(compile(head, "materials_helpers", "exec"), ns)
    return ns


def test_fingertips_come_from_the_collider_record_not_the_template():
    """A short finger's template tip is a ghost and carries no shape."""
    ns = _materials_helpers()
    record = {"f0_link0": 1, "f0_link1": 1, "f1_link0": 1}
    assert ns["fingertips_from_record"](record, 5, 6) == {"f0_link1", "f1_link0"}


def test_a_reconstructed_layout_matches_a_measured_one():
    """Measuring every design is ~96 min at 24k, so layouts are reconstructed."""
    ns = _materials_helpers()
    links = ["iiwa14_link_7", "f0_link0", "f0_link1", "f1_link0", "f1_link1"]
    record = {"iiwa14_link_7": 1, "f0_link0": 1, "f0_link1": 1, "f1_link0": 1}
    measured = [("iiwa14_link_7", 0, 3), ("f0_link0", 3, 4), ("f0_link1", 4, 5),
                ("f1_link0", 5, 6), ("f1_link1", 6, 6)]
    arm = ns["arm_counts_from"](measured, record)
    layout, total = ns["shape_layouts_from_record"](links, {0: record}, arm)[0]
    assert layout == measured
    assert total == 6
