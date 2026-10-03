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
    rnn = net.get_default_rnn_state()
    states = tuple(rnn) if rnn is not None else None
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


def _design_params(k=None, ghost_tail=None):
    params = _ppo_params(ghost_tail)
    if k is not None:
        params["per_design_nets"] = k
    return params


def _design_obs(designs, k, n_body=6, ghost=None):
    """Body values, then a one-hot of width k (as rl_games' input
    normalisation leaves it: positive for the env's design, negative
    elsewhere), then an optional ghost-sign tail."""
    n = len(designs)
    onehot = torch.full((n, k), -0.8)
    onehot[torch.arange(n), torch.tensor(designs)] = 1.3
    parts = [torch.randn(n, n_body), onehot] + ([ghost] if ghost is not None else [])
    return torch.cat(parts, dim=-1)


def test_per_design_nets_route_each_sample_to_its_designs_network():
    """per_design_nets K: K independent actor-critics (trunk, mean, log-std,
    value); each sample uses the one its design one-hot selects."""
    k, n_act = 3, 4
    obs = _design_obs([0, 2, 1, 2], k)
    net, inp = _build_plain(_design_params(k), obs, n_act)
    with torch.no_grad():
        for d, sub in enumerate(net.design_subnets()):
            sub.sigma.fill_(-0.5 - d)
    mu, logstd, value, _s = net(inp)
    for i, d in enumerate([0, 2, 1, 2]):
        mu_d, logstd_d, value_d, _ = net.design_forward(d, inp)
        assert torch.allclose(mu[i], mu_d[i]) and torch.allclose(value[i], value_d[i])
        assert torch.allclose(logstd[i], torch.full_like(logstd[i], -0.5 - d))
    # design 1's samples train only design 1's network
    (mu[2].sum() + value[2].sum() + logstd[2].sum()).backward()
    for d, sub in enumerate(net.design_subnets()):
        own = [p.grad for p in sub.own_parameters() if p.grad is not None and torch.any(p.grad != 0)]
        assert (len(own) > 0) == (d == 1)


def test_per_design_nets_with_ghost_mask_tail():
    k, n_act = 2, 3
    ghost = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]])
    obs = _design_obs([1, 0], k, ghost=ghost)
    net, inp = _build_plain(_design_params(k, ghost_tail=n_act), obs, n_act)
    mu, logstd, _v, _s = net(inp)
    assert torch.all(mu[ghost < 0] == 0) and torch.all(logstd[ghost < 0] == 0)
    mu_1, _, _, _ = net.design_forward(1, inp)
    assert torch.allclose(mu[0, ghost[0] > 0], mu_1[0, ghost[0] > 0])


def test_per_design_nets_off_keeps_the_stock_parameters():
    obs = torch.randn(2, 9)
    stock, _ = _build_plain(_ppo_params(), obs, 3)
    off, _ = _build_plain(_design_params(None), obs, 3)
    one, _ = _build_plain(_design_params(1), obs, 3)
    names = sorted(n for n, _ in stock.named_parameters())
    assert sorted(n for n, _ in off.named_parameters()) == names
    assert sorted(n for n, _ in one.named_parameters()) == names


def test_design_id_observation_field():
    from isaacsimenvs.inhand_reorient import anyrotate_profile as ar
    from isaacsimenvs.inhand_reorient import hora_profile as hp

    assert ar.anyrotate_field_width("hora_design_id_8", 32, 5) == 8
    oh = hp.design_onehot(torch.tensor([0, 2, 1]), 4)
    assert oh.tolist() == [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
    with pytest.raises(ValueError):
        hp.design_onehot(torch.tensor([0, 4]), 4)
    h = yaml.safe_load((CFG.parent / "task" / "InHandReorient.yaml").read_text())["hora"]
    assert h["design_id_obs"] == 0


def test_group_running_norm_keeps_separate_statistics_per_design():
    norm = pn.GroupRunningNorm(2, 3)
    norm.train()
    g = torch.tensor([0] * 200 + [1] * 200)
    x = torch.cat([torch.randn(200, 3) * 0.1 + 0.5, torch.randn(200, 3) * 4.0 - 2.0])
    y = norm(x, g)
    for d in (0, 1):
        assert torch.allclose(y[g == d].mean(0), torch.zeros(3), atol=0.05)
        assert torch.allclose(y[g == d].std(0), torch.ones(3), atol=0.05)
    norm.eval()  # eval: no update
    before = norm.mean.clone()
    norm(x * 3.0, g)
    assert torch.equal(norm.mean, before)


def test_per_design_input_norm_normalises_the_body_and_keeps_the_tail():
    """per_design_input_norm: the observation body is normalised by each
    design's own running statistics before the network (shared or
    per-design); the design one-hot and slot-sign tail pass unchanged."""
    k, n_act = 2, 3
    ghost = torch.tensor([[1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]]).repeat(50, 1)
    obs = _design_obs([0, 1] * 50, k, ghost=ghost)
    obs[1::2, :6] = obs[1::2, :6] * 10.0 + 3.0  # design 1's body on another scale
    params = _design_params(None, ghost_tail=n_act)
    params["per_design_input_norm"] = k
    net, inp = _build_plain(params, obs, n_act)
    net.train()
    body, tail = net.design_normalised_obs(obs)
    assert torch.equal(tail, obs[:, -(k + n_act):])
    for d in (0, 1):
        assert torch.allclose(body[d::2].mean(0), torch.zeros(6), atol=1e-3)
    mu, logstd, _v, _s = net(inp)
    assert torch.all(mu[ghost < 0] == 0)


def test_per_design_heads_share_the_trunk_and_route_the_heads():
    """per_design_heads K: one trunk, K mean heads, K log-std vectors and K
    value heads; each sample uses its design's heads."""
    k, n_act = 2, 3
    obs = _design_obs([0, 1, 1], k)
    params = _ppo_params()
    params["per_design_heads"] = k
    net, inp = _build_plain(params, obs, n_act)
    with torch.no_grad():
        net.sigma.fill_(-0.3)
        net.sigma_heads[0].fill_(-1.7)
    mu, logstd, value, _s = net(inp)
    trunk = net.actor_mlp(obs)
    assert torch.allclose(mu[0], net.mu(trunk)[0]) and torch.allclose(value[0], net.value(trunk)[0])
    assert torch.allclose(mu[1], net.mu_heads[0](trunk)[1]) and torch.allclose(value[2], net.value_heads[0](trunk)[2])
    assert torch.all(logstd[0] == -0.3) and torch.all(logstd[1:] == -1.7)
    (mu[1].sum() + value[1].sum()).backward()
    assert net.mu.weight.grad is None or torch.all(net.mu.weight.grad == 0)  # design 0's head untouched
    assert torch.any(net.mu_heads[0].weight.grad != 0)
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in net.actor_mlp.parameters())  # shared trunk
