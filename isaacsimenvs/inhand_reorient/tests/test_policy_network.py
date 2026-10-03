"""CPU tests for ``policy_network.py``: the rl_games actor-critic whose
state-independent policy log-std is projected into a configured range before
every forward pass (I41: the population controller's std grew to ~250).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_policy_network.py -q
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from isaacsimenvs.inhand_reorient import policy_network as pn

CFG = Path(__file__).resolve().parents[3] / "coevolution" / "cfg" / "train"


def _params(logstd_max=0.0, logstd_min=None):
    params = copy.deepcopy(yaml.safe_load((CFG / "InHandReorientSAPG.yaml").read_text())["params"]["network"])
    params["name"] = pn.NETWORK_NAME
    params["space"]["continuous"]["logstd_max"] = logstd_max
    if logstd_min is not None:
        params["space"]["continuous"]["logstd_min"] = logstd_min
    return params


def _build(params, n_obs=10, n_act=4, n_envs=6):
    builder = pn.BoundedSigmaA2CBuilder()
    builder.load(params)
    net = builder.build("x", actions_num=n_act, input_shape=(n_obs + 1,), num_seqs=n_envs, value_size=1,
                        type="extra_param", coef_ids=torch.tensor([50.0]), coef_id_idx=n_obs)
    obs = torch.randn(n_envs, n_obs + 1)
    obs[:, -1] = 50.0
    states = tuple(s for s in net.get_default_rnn_state())
    return net, {"obs": obs, "rnn_states": states, "is_train": False}


def test_network_is_registered_with_rl_games():
    from rl_games.algos_torch import model_builder

    assert pn.NETWORK_NAME in model_builder.NETWORK_REGISTRY


def test_sigma_above_the_bound_is_projected_onto_it():
    net, inp = _build(_params(logstd_max=0.0))
    with torch.no_grad():
        net.sigma.fill_(5.0)
    _mu, logstd, _value, _states = net(inp)
    assert torch.all(logstd == 0.0)
    assert torch.all(net.sigma == 0.0)  # the parameter itself, so the optimizer continues from the bound


def test_sigma_inside_the_bound_is_untouched_and_trainable():
    net, inp = _build(_params(logstd_max=0.5, logstd_min=-3.0))
    with torch.no_grad():
        net.sigma.fill_(-1.0)
    _mu, logstd, _value, _states = net(inp)
    assert torch.allclose(logstd, torch.full_like(logstd, -1.0))
    logstd.sum().backward()
    assert net.sigma.grad is not None and torch.all(net.sigma.grad != 0)


def test_lower_bound_is_applied_when_configured():
    net, inp = _build(_params(logstd_max=0.5, logstd_min=-2.0))
    with torch.no_grad():
        net.sigma.fill_(-9.0)
    _mu, logstd, _value, _states = net(inp)
    assert torch.all(logstd == -2.0)


def test_no_bounds_configured_behaves_like_the_stock_network():
    params = _params()
    del params["space"]["continuous"]["logstd_max"]
    net, inp = _build(params)
    with torch.no_grad():
        net.sigma.fill_(7.0)
    _mu, logstd, _value, _states = net(inp)
    assert torch.all(logstd == 7.0)


def test_population_config_uses_the_bounded_network():
    doc = yaml.safe_load((CFG / "InHandReorientPopSAPG.yaml").read_text())["params"]
    assert doc["network"]["name"] == pn.NETWORK_NAME
    assert doc["network"]["space"]["continuous"]["logstd_max"] <= 0.5


def _ppo_params(ghost_tail=None):
    params = copy.deepcopy(yaml.safe_load((CFG / "InHandAnyRotatePPO.yaml").read_text())["params"]["network"])
    params["name"] = pn.NETWORK_NAME
    if ghost_tail is not None:
        params["space"]["continuous"]["ghost_mask_tail"] = ghost_tail
    return params


def _build_plain(params, obs, n_act):
    builder = pn.BoundedSigmaA2CBuilder()
    builder.load(params)
    net = builder.build("x", actions_num=n_act, input_shape=(obs.shape[1],), num_seqs=obs.shape[0], value_size=1)
    states = tuple(s for s in net.get_default_rnn_state())
    return net, {"obs": obs, "rnn_states": states, "is_train": False}


def test_ghost_action_dims_are_held_constant_and_carry_no_gradient():
    """ghost_mask_tail N: the last N observation values are each action
    slot's (normalised) sign, +1 real / -1 ghost; a slot reading < 0 gets
    mean 0 and log-std 0 with no gradient, so a hand's ghost joints add
    nothing to the PPO ratio or the update of another hand's real joints."""
    n_act = 4
    obs = torch.randn(3, 8 + n_act)
    obs[:, -n_act:] = torch.tensor([[1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    net, inp = _build_plain(_ppo_params(ghost_tail=n_act), obs, n_act)
    with torch.no_grad():
        net.sigma.fill_(-0.7)
    mu, logstd, _v, _s = net(inp)
    ghost = obs[:, -n_act:] < 0
    assert torch.all(mu[ghost] == 0.0) and torch.all(logstd[ghost] == 0.0)
    assert torch.all(logstd[~ghost] == -0.7)
    (mu[ghost].sum() + logstd[ghost].sum()).backward(retain_graph=True)
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert all(torch.all(g == 0) for g in grads)
    # a normalised sign of exactly 0 (the slot is the same for every env) counts as real
    assert not ghost[2].any()


def test_without_ghost_mask_all_dims_are_free():
    n_act = 3
    obs = torch.randn(2, 5 + n_act)
    obs[:, -n_act:] = -1.0
    net, inp = _build_plain(_ppo_params(), obs, n_act)
    mu, logstd, _v, _s = net(inp)
    mu.sum().backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in net.parameters())


def test_slot_sign_observation_width_and_default_off():
    from isaacsimenvs.inhand_reorient import anyrotate_profile as ar

    assert ar.anyrotate_field_width("hora_slot_sign", 32, 5) == 32
    h = yaml.safe_load((CFG.parent / "task" / "InHandReorient.yaml").read_text())["hora"]
    assert h["ghost_action_mask"] is False
