"""Input normalisation and the attention mask of the joint transformer.

Until 2026-09-29 rl_games standardised every observation column (``models.py`` ``norm_obs``)
before the network saw it. ``joint_enabled`` is a constant 1 on a fixed hand, so it arrived as
~0, and the mask ``joint_enabled > 0.5`` marked every joint a ghost: each joint attended to the
global token alone and no joint's action could depend on another joint's input (found in trained
real-SHARPA and Allegro checkpoints: d mu_k / d token_j was exactly 0 for every j != k).

Now the network normalises its own input and rl_games' normaliser is off for it:
  * joint_enabled is read RAW, only for the attention mask, and is not a token feature;
  * token fields are standardised on statistics POOLED over real joints, so a per-joint
    constant (limits, where a link sits) survives instead of normalising to 0.

Kit-free: the robot spec is stubbed (the layout needs only joint counts).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import coevolution.networks.joint_transformer as JT

ROOT = Path(__file__).resolve().parents[3]
_sp = importlib.util.spec_from_file_location(
    "layout", ROOT / "isaacsimenvs/pose_reaching_6d/obs_utils/layout.py")
L = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(L)

N_HAND = 5
FIELDS = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
          "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
          "object_keypoints_rel_joint", "keypoints_rel_goal", "object_vel"]
SPEC = SimpleNamespace(num_arm_joints=0, num_hand_joints=N_HAND, num_joints=N_HAND,
                       num_fingertips=2, hand_joint_names=[f"j{i}" for i in range(N_HAND)])
LAYOUT = L.build_token_layout(SPEC, FIELDS)
OBS_DIM = L.compute_obs_dim(FIELDS, SPEC)
OFF = L.field_offsets(FIELDS, SPEC)
EN_A, EN_B = OFF["joint_enabled"]
TOKEN_COLS = torch.tensor(LAYOUT["token_columns"])


@pytest.fixture(autouse=True)
def _stub_spec(monkeypatch):
    monkeypatch.setattr(JT.JointTransformerNet, "_build_layout",
                        lambda self, d: L.build_token_layout(SPEC, self.field_list))


def _net(**kw):
    torch.manual_seed(0)
    params = dict(d_model=16, n_layers=2, n_heads=1, ff_mult=2, mu_head_units=[16],
                  value_head_units=[16], robot_spec="stub", obs_list=FIELDS,
                  space={"continuous": {"fixed_sigma": "fixed", "mu_activation": "None",
                                        "sigma_activation": "None"}})
    return JT.JointTransformerNet(params, actions_num=N_HAND, input_shape=(OBS_DIM,), **kw)


def _obs(enabled: torch.Tensor) -> torch.Tensor:
    """Raw observations: per-joint constants differ between joints, as on a real hand."""
    torch.manual_seed(1)
    n = enabled.shape[0]
    x = torch.randn(n, OBS_DIM)
    x[:, EN_A:EN_B] = enabled
    lo, up = OFF["joint_lower"], OFF["joint_upper"]
    x[:, lo[0]:lo[1]] = torch.linspace(-1.0, 0.0, N_HAND)          # constant per joint
    x[:, up[0]:up[1]] = torch.linspace(0.5, 2.0, N_HAND)
    return x


def _trained(net, x, steps=50):
    """Fold statistics in the way a run does (training-mode forwards), then evaluate."""
    net.train()
    with torch.no_grad():
        for _ in range(steps):
            net({"obs": x})
    return net.eval()


def _cross_joint_sensitivity(net, x):
    x = x.clone().requires_grad_(True)
    mu = net({"obs": x})[0]
    other = 0.0
    for k in range(N_HAND):
        g, = torch.autograd.grad(mu[:, k].sum(), x, retain_graph=True)
        per_joint = g[:, TOKEN_COLS].abs().sum(dim=(0, 2))
        other += (per_joint.sum() - per_joint[k]).item()
    return other


def test_rl_games_input_normalisation_is_refused():
    with pytest.raises(ValueError, match="normalize_input"):
        _net(normalize_input=True)


def test_joint_enabled_is_not_a_token_feature():
    en_cols = set(range(EN_A, EN_B))
    assert not en_cols & set(TOKEN_COLS.reshape(-1).tolist())
    assert LAYOUT["enabled_columns"] == list(range(EN_A, EN_B))
    assert LAYOUT["token_dim"] == 31, "5 proprio + 12 box + 2 limits + 12 object keypoints"


def test_a_fixed_hand_keeps_every_joint():
    x = _obs(torch.ones(64, N_HAND))
    net = _trained(_net(), x)
    assert net._trunk(x)[3].all()


def test_joints_can_see_each_other():
    """The decentralized policy's whole point: joint k's action depends on joint j's state."""
    x = _obs(torch.ones(64, N_HAND))
    assert _cross_joint_sensitivity(_trained(_net(), x), x) > 1e-4


def test_a_population_masks_exactly_its_ghosts():
    enabled = torch.ones(64, N_HAND)
    enabled[:, 4] = 0.0                  # a ghost in every design
    enabled[::2, 3] = 0.0                # a ghost in half of them
    x = _obs(enabled)
    assert torch.equal(_trained(_net(), x)._trunk(x)[3], enabled > 0.5)


def test_a_ghost_cannot_influence_a_real_joint():
    enabled = torch.ones(64, N_HAND)
    enabled[:, 4] = 0.0
    x = _obs(enabled)
    net = _trained(_net(), x)
    y = x.clone()
    y[:, TOKEN_COLS[4]] = 100 * torch.randn_like(y[:, TOKEN_COLS[4]])
    with torch.no_grad():
        assert torch.allclose(net({"obs": x})[0][:, :4], net({"obs": y})[0][:, :4], atol=1e-5)


def test_ghosts_stay_out_of_the_token_statistics():
    enabled = torch.ones(64, N_HAND)
    enabled[:, 4] = 0.0
    x = _obs(enabled)
    y = x.clone()
    y[:, TOKEN_COLS[4]] = 100 * torch.randn_like(y[:, TOKEN_COLS[4]])
    a, b = _trained(_net(), x), _trained(_net(), y)
    assert torch.allclose(a.token_norm.running_mean, b.token_norm.running_mean)
    assert torch.allclose(a.token_norm.running_var, b.token_norm.running_var)


def test_per_joint_constants_survive_pooled_normalisation():
    """Per-column normalisation sent every joint's limits to 0; pooled keeps them distinct."""
    x = _obs(torch.ones(64, N_HAND))
    net = _trained(_net(), x)
    with torch.no_grad():
        raw = x[:, TOKEN_COLS.reshape(-1)].view(64, N_HAND, -1)
        normed = net.token_norm(raw)
    lower_slot = 5 + 12                  # proprio (5), box (12), then joint_lower
    lower = normed[0, :, lower_slot]
    assert lower.unique().numel() == N_HAND and lower.abs().max() > 0.5


def test_statistics_move_only_in_training_mode():
    x = _obs(torch.ones(16, N_HAND))
    net = _net().eval()
    before = net.token_norm.running_mean.clone()
    with torch.no_grad():
        net({"obs": x})
    assert torch.equal(before, net.token_norm.running_mean)
    net.train()
    with torch.no_grad():
        net({"obs": x})
    assert not torch.equal(before, net.token_norm.running_mean)
