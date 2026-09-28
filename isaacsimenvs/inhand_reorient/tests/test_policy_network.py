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
