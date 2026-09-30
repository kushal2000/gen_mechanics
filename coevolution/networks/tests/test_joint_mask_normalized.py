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


# --- the normaliser against rl_games', and on rl_games' schedule -----------------------------

def test_running_norm_matches_rl_games_running_mean_std():
    """Unweighted, _RunningNorm must be rl_games' RunningMeanStd: same statistics, same output."""
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    torch.manual_seed(3)
    ref, ours = RunningMeanStd((7,)), JT._RunningNorm(7)
    ref.train(); ours.train()
    for i in range(6):
        x = torch.randn(257, 7) * torch.tensor([1e-3, 0.1, 1, 10, 100, 1, 1]) + i
        ref(x)
        ours.update(x)
    ref.eval(); ours.eval()
    # rl_games takes batch moments in float32, ours in float64: equal to float32 rounding.
    assert torch.allclose(ours.running_mean, ref.running_mean, rtol=1e-5, atol=1e-7)
    assert torch.allclose(ours.running_var, ref.running_var, rtol=1e-5, atol=1e-9)
    assert torch.allclose(ours.count, ref.count)
    x = torch.randn(50, 7) * 20
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def _rl_games_model():
    """The real rl_games model class around the network, built the way a2c_continuous does."""
    from rl_games.algos_torch.models import ModelA2CContinuousLogStd
    builder = JT.JointTransformerBuilder()
    builder.load(dict(d_model=16, n_layers=2, n_heads=1, ff_mult=2, mu_head_units=[16],
                      value_head_units=[16], robot_spec="stub", obs_list=FIELDS,
                      space={"continuous": {"fixed_sigma": "fixed", "mu_activation": "None",
                                            "sigma_activation": "None"}}))
    return ModelA2CContinuousLogStd(builder).build(dict(
        input_shape=(OBS_DIM,), actions_num=N_HAND, normalize_input=False,
        normalize_value=False, value_size=1, num_seqs=1))


def test_statistics_follow_rl_games_schedule():
    """Rollout (eval): frozen. First mini-epoch (train): update. After rl_games calls
    model.running_mean_std.eval(): frozen, although the model is still in training mode."""
    model = _rl_games_model()
    algo = SimpleNamespace(model=model, normalize_input=False)
    assert JT.install_rl_games_hook(algo) and algo.normalize_input
    net = model.a2c_network
    x = _obs(torch.ones(32, N_HAND))
    call = lambda: model({"is_train": True, "obs": x, "prev_actions": torch.zeros(32, N_HAND)})
    count = lambda: net.token_norm.count.item()

    model.eval(); c0 = count()
    with torch.no_grad():
        model({"is_train": False, "obs": x})
    assert count() == c0, "rollout must not move the statistics"

    model.train()                                  # set_train()
    with torch.no_grad():
        call(); call()                             # mini-epoch 1, two minibatches
    c1 = count()
    assert c1 > c0
    model.running_mean_std.eval()                  # a2c_common: freeze after mini-epoch 1
    assert model.training and not net.token_norm.training
    with torch.no_grad():
        call(); call()                             # mini-epochs 2..
    assert count() == c1, "statistics moved after the first mini-epoch"

    model.eval(); model.train()                    # next update
    with torch.no_grad():
        call()
    assert count() > c1


def test_the_hook_adds_no_state_and_checkpoints_round_trip():
    model = _rl_games_model()
    algo = SimpleNamespace(model=model, normalize_input=False)
    JT.install_rl_games_hook(algo)
    assert model.running_mean_std.state_dict() == {}
    keys = model.state_dict().keys()
    assert not [k for k in keys if k.startswith("running_mean_std")]
    assert "a2c_network.token_norm.running_mean" in keys
    model.train()
    with torch.no_grad():
        model({"is_train": True, "obs": _obs(torch.ones(8, N_HAND)),
               "prev_actions": torch.zeros(8, N_HAND)})
    fresh = _rl_games_model()
    fresh.load_state_dict(model.state_dict())
    assert torch.equal(fresh.a2c_network.token_norm.running_var, model.a2c_network.token_norm.running_var)


def test_the_hook_ignores_other_networks():
    algo = SimpleNamespace(model=SimpleNamespace(a2c_network=torch.nn.Linear(2, 2)), normalize_input=True)
    assert not JT.install_rl_games_hook(algo) and algo.normalize_input


def test_activation_checkpointing_is_exact():
    """grad_checkpoint must change memory only: identical outputs and gradients."""
    x = _obs(torch.ones(16, N_HAND))
    torch.manual_seed(0)
    plain = _net()
    ckpt = _net()
    ckpt.load_state_dict(plain.state_dict())
    ckpt.grad_checkpoint = True
    for net in (plain, ckpt):
        net.train()
        with torch.no_grad():
            net({"obs": x})                                   # same statistics update in both
    outs, grads = [], []
    for net in (plain, ckpt):
        net.token_norm.eval(); net.global_norm.eval()
        net.zero_grad()
        mu, _, value, _ = net({"obs": x})
        (mu.square().sum() + value.sum()).backward()
        outs.append(torch.cat([mu.flatten(), value.flatten()]).detach())
        grads.append(torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None]))
    assert torch.allclose(outs[0], outs[1], atol=1e-6)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


# --- ghost action dimensions ----------------------------------------------------------------

def _mixed_population_obs():
    """Envs from different designs: each env has its OWN ghost slots."""
    enabled = torch.ones(12, N_HAND)
    enabled[0:4, 4] = 0.0                 # design A: slot 4 is a ghost
    enabled[4:8, 2:4] = 0.0               # design B: slots 2 and 3
    return _obs(enabled), enabled > 0.5   # design C (8:12): no ghosts


def test_ghost_actions_are_constant_per_env():
    x, real = _mixed_population_obs()
    net = _trained(_net(), x)
    net.mask_ghost_actions = True
    mu, logstd, _, _ = net({"obs": x})
    assert torch.equal(mu[~real], torch.zeros_like(mu[~real]))
    assert torch.equal(logstd[~real], torch.zeros_like(logstd[~real]))


def test_real_actions_are_unchanged_by_the_ghost_mask():
    x, real = _mixed_population_obs()
    net = _trained(_net(), x)
    with torch.no_grad():
        mu0, ls0, v0, _ = net({"obs": x})
        net.mask_ghost_actions = True
        mu1, ls1, v1, _ = net({"obs": x})
    assert torch.equal(mu0[real], mu1[real]) and torch.equal(ls0[real], ls1[real])
    assert torch.equal(v0, v1)


def test_ghost_dimensions_carry_no_gradient():
    """Their log-prob must not depend on any parameter, so they cancel out of the PPO ratio."""
    x, real = _mixed_population_obs()
    net = _trained(_net(), x)
    net.mask_ghost_actions = True
    net.token_norm.eval(); net.global_norm.eval()
    mu, logstd, _, _ = net({"obs": x})
    a = torch.randn_like(mu)
    logp = torch.distributions.Normal(mu, logstd.exp()).log_prob(a)
    net.zero_grad()
    logp[~real].sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert all(torch.count_nonzero(g) == 0 for g in grads), "a ghost dimension reached a parameter"


def test_arm_dimensions_are_never_masked(monkeypatch):
    spec = SimpleNamespace(num_arm_joints=2, num_hand_joints=N_HAND, num_joints=N_HAND + 2,
                           num_fingertips=2, hand_joint_names=[f"j{i}" for i in range(N_HAND)])
    monkeypatch.setattr(JT.JointTransformerNet, "_build_layout",
                        lambda self, d: L.build_token_layout(spec, self.field_list))
    obs_dim = L.compute_obs_dim(FIELDS, spec)
    a, b = L.field_offsets(FIELDS, spec)["joint_enabled"]
    torch.manual_seed(0)
    params = dict(d_model=16, n_layers=1, n_heads=1, ff_mult=2, mu_head_units=[16],
                  value_head_units=[16], arm_head_units=[16], robot_spec="stub", obs_list=FIELDS,
                  mask_ghost_actions=True,
                  space={"continuous": {"fixed_sigma": "fixed", "mu_activation": "None",
                                        "sigma_activation": "None"}})
    net = JT.JointTransformerNet(params, actions_num=N_HAND + 2, input_shape=(obs_dim,)).eval()
    x = torch.randn(6, obs_dim)
    x[:, a:b] = 1.0
    x[:, a + 1] = 0.0                                   # hand slot 1 is a ghost
    mu, _, _, _ = net({"obs": x})
    assert (mu[:, :2] != 0).all(), "arm dimensions were masked"
    assert (mu[:, 2 + 1] == 0).all() and (mu[:, [2, 4, 5, 6]] != 0).all()
