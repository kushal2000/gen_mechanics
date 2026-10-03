"""Per-group PPO advantage normalisation in the vendored rl_games
(``config.group_advantage_norm``, off by default): the env publishes a
group id per env in its infos (``hora.ppo_group_info``), and each group's
advantages are normalised over that group's own samples. Kit-free."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_group_normalization_gives_each_group_zero_mean_unit_std():
    from rl_games.algos_torch import torch_ext

    g = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2])
    x = torch.tensor([1.0, 2.0, 3.0, 100.0, 300.0, 500.0, 700.0, 5.0])
    y = torch_ext.group_normalization(x, g)
    for k in (0, 1):
        m = g == k
        assert abs(float(y[m].mean())) < 1e-5
        assert abs(float(y[m].std()) - 1.0) < 1e-4
    assert float(y[g == 2]) == 0.0  # a single-sample group is only centred


def _agent(group_norm: bool):
    from rl_games.common.a2c_common import ContinuousA2CBase

    agent = object.__new__(ContinuousA2CBase)
    captured = {}
    agent.normalize_value = False
    agent.normalize_advantage = True
    agent.normalize_rms_advantage = False
    agent.is_rnn = False
    agent.has_central_value = False
    agent.group_advantage_norm = group_norm
    agent.dataset = SimpleNamespace(update_values_dict=lambda d: captured.update(d))
    return agent, captured


def _batch():
    horizon, n = 4, 6
    groups = torch.tensor([0, 1, 0, 1, 0, 1])  # per env
    from rl_games.common.custom_utils import swap_and_flatten01

    g_buf = groups.unsqueeze(0).expand(horizon, n)  # what play_steps records per step
    gen = torch.Generator().manual_seed(0)
    adv = torch.randn(horizon, n, 1, generator=gen) * torch.where(groups == 1, 50.0, 1.0).unsqueeze(-1)
    values = torch.zeros(horizon, n, 1)
    batch = {"obses": torch.zeros(horizon * n, 3), "returns": swap_and_flatten01(adv + values),
             "values": swap_and_flatten01(values), "dones": torch.zeros(horizon * n),
             "actions": torch.zeros(horizon * n, 2), "neglogpacs": torch.zeros(horizon * n),
             "mus": torch.zeros(horizon * n, 2), "sigmas": torch.ones(horizon * n, 2),
             "ppo_groups": swap_and_flatten01(g_buf)}
    return batch, swap_and_flatten01(groups.unsqueeze(0).expand(horizon, n))


def test_prepare_dataset_normalises_advantages_per_group_when_enabled():
    agent, captured = _agent(True)
    batch, g = _batch()
    agent.prepare_dataset(batch)
    adv = captured["advantages"]
    for k in (0, 1):
        assert abs(float(adv[g == k].std()) - 1.0) < 1e-4
        assert abs(float(adv[g == k].mean())) < 1e-5


def test_prepare_dataset_default_keeps_batch_global_normalisation():
    agent, captured = _agent(False)
    batch, g = _batch()
    agent.prepare_dataset(batch)
    adv = captured["advantages"]
    assert abs(float(adv.std()) - 1.0) < 1e-4
    assert float(adv[g == 0].std()) < 0.2  # the small-spread group is squashed by the large one


def test_group_norm_without_group_ids_raises():
    agent, _ = _agent(True)
    batch, _ = _batch()
    batch.pop("ppo_groups")
    with pytest.raises(KeyError):
        agent.prepare_dataset(batch)


def test_ppo_group_info_defaults_off():
    h = yaml.safe_load((REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())["hora"]
    assert h["ppo_group_info"] is False
    agent = yaml.safe_load((REPO_ROOT / "coevolution/cfg/train/InHandAnyRotatePPO.yaml").read_text())
    assert agent["params"]["config"].get("group_advantage_norm", False) is False


def test_play_steps_records_group_ids_in_batch_order():
    from rl_games.common.custom_utils import swap_and_flatten01

    agent, _ = _agent(True)
    agent.horizon_length, agent.num_actors, agent.ppo_device = 3, 4, "cpu"
    agent.group_info_key = "ppo_group"
    groups = torch.tensor([2, 0, 1, 0])
    for n in range(3):
        agent._record_ppo_group(n, {"ppo_group": groups})
    expected = swap_and_flatten01(groups.unsqueeze(0).expand(3, 4))
    assert torch.equal(agent._ppo_groups_batch(), expected)
    with pytest.raises(KeyError):
        agent._record_ppo_group(0, {})
