"""CPU tests for the collaborator's training recipe on the in-hand joint-token
transformer: SAPG support in ``token_policy.py`` (the exploration-coefficient
column as a learned embedding on the global vector, one log-std row per
exploration block) and the two recipe configs
``InHandHoraTokenTeamPPO.yaml`` (R1, plain PPO) and
``InHandHoraTokenTeamSAPG.yaml`` (R2, R1 plus SAPG with 6 blocks), both
taken from ``PoseReachJointTransformerSAPG.yaml``.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_token_team_recipe.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from isaacsimenvs.inhand_reorient import token_layout as tl

CFG = Path(__file__).resolve().parents[3] / "coevolution" / "cfg"
J = 32
N_BLOCKS = 6
COEF_IDS = torch.linspace(50.0, 0.0, N_BLOCKS)  # a2c_common's embedding_genvec, one value per block
R1 = "InHandHoraTokenTeamPPO.yaml"
R2 = "InHandHoraTokenTeamSAPG.yaml"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CFG / "train" / name).read_text())["params"]


def _same(a, b) -> bool:
    """Equal as numbers when both parse as one (PyYAML reads "1e-4" as a string)."""
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def _sapg_params(logstd_max=None):
    p = {
        "name": "inhand_joint_transformer", "separate": False, "d_model": 16, "n_layers": 2, "n_heads": 1,
        "ff_mult": 2, "value_head_units": [32], "arm_head_units": [8], "design_id_obs": 0,
        "per_design_global_norm": False,
        "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None", "fixed_sigma": "coef_cond",
                                 "mu_init": {"name": "default"},
                                 "sigma_init": {"name": "const_initializer", "val": 0}}},
    }
    if logstd_max is not None:
        p["space"]["continuous"]["logstd_max"] = logstd_max
    return p


def _sapg_net(params):
    """Built the way a2c_continuous builds it under SAPG (mixed_expl_learn_param)."""
    from isaacsimenvs.inhand_reorient import token_policy as tp

    b = tp.InhandJointTransformerBuilder()
    b.load(params)
    obs_dim = tl.layout(J)["obs_dim"]
    return b.build("x", actions_num=J, input_shape=(obs_dim + 1,), num_seqs=1, value_size=1,
                   type="extra_param", coef_ids=COEF_IDS.clone(), coef_id_idx=obs_dim)


def _obs(n_valid, blocks, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(blocks)
    tok = torch.randn(n, J, tl.TOKEN_DIM, generator=g)
    enabled = torch.zeros(n, J, dtype=torch.bool)
    enabled[:, :n_valid] = True
    tok[..., tl.ENABLED_COL] = enabled.float()
    tok = tok * enabled.unsqueeze(-1)
    coef = COEF_IDS[torch.tensor(blocks)].unsqueeze(-1)
    return torch.cat([tok.reshape(n, -1), torch.randn(n, tl.GLOBAL_DIM, generator=g), coef], dim=-1), enabled


# --------------------------------------------------------------------------
# SAPG path of the network
# --------------------------------------------------------------------------


def test_sapg_build_and_shapes():
    net = _sapg_net(_sapg_params()).eval()
    assert net.sigma.shape == (N_BLOCKS, J)
    assert net.global_proj.in_features == tl.GLOBAL_DIM + 32  # the coefficient embedding joins the global vector
    obs, _ = _obs(16, [0, 1, 2, 3, 4, 5])
    mu, logstd, value, states = net({"obs": obs})
    assert mu.shape == (6, J) and logstd.shape == (6, J) and value.shape == (6, 1) and states is None


def test_each_block_reads_its_own_logstd_row_and_ghosts_stay_masked():
    net = _sapg_net(_sapg_params()).eval()
    with torch.no_grad():
        net.sigma.copy_(-0.1 * torch.arange(1, N_BLOCKS + 1).unsqueeze(1).expand(N_BLOCKS, J))
    obs, enabled = _obs(16, [0, 3, 5, 3])
    _mu, logstd, _v, _ = net({"obs": obs})
    for i, b in enumerate([0, 3, 5, 3]):
        assert torch.allclose(logstd[i][enabled[i]], net.sigma[b][enabled[i]])
    assert torch.all(logstd[~enabled] == 0)


def test_the_exploration_coefficient_reaches_the_network():
    """The same observation under two blocks gives different outputs (the
    learned per-block embedding on the global vector)."""
    net = _sapg_net(_sapg_params()).eval()
    obs, enabled = _obs(16, [0, 0])
    obs[1, -1] = COEF_IDS[4]
    mu, _s, v, _ = net({"obs": obs})
    assert not torch.allclose(mu[0][enabled[0]], mu[1][enabled[1]])
    assert not torch.allclose(v[0], v[1])


def test_entropy_bonus_gives_ghost_logstd_no_gradient():
    """I41: SAPG's entropy bonus inflated the MLP's ghost-slot log-std. Here
    an entropy-like objective (the sum of the per-sample log-std) reaches only
    the real columns of the rows its blocks use."""
    net = _sapg_net(_sapg_params()).train()
    obs, enabled = _obs(10, [0, 2, 2, 5])
    _mu, logstd, _v, _ = net({"obs": obs})
    logstd.sum().backward()
    g = net.sigma.grad
    assert torch.all(g[:, 10:] == 0)  # ghost columns, every block
    assert torch.all(g[[0, 2, 5], :10] != 0) and torch.all(g[[1, 3, 4]] == 0)


def test_logstd_max_projects_every_block_row():
    net = _sapg_net(_sapg_params(logstd_max=0.0)).eval()
    with torch.no_grad():
        net.sigma.fill_(0.7)
        net.sigma[2].fill_(-0.4)
    obs, enabled = _obs(12, [1, 2])
    _mu, logstd, _v, _ = net({"obs": obs})
    assert torch.all(net.sigma[[0, 1, 3, 4, 5]] == 0.0) and torch.all(net.sigma[2] == -0.4)
    assert torch.all(logstd[0][enabled[0]] == 0.0) and torch.all(logstd[1][enabled[1]] == -0.4)


def test_ghost_tokens_change_nothing_under_sapg():
    net = _sapg_net(_sapg_params()).eval()
    obs, enabled = _obs(7, [1, 4, 0])
    mu, _s, v, _ = net({"obs": obs})
    noisy = obs.clone()
    tok = noisy[:, : J * tl.TOKEN_DIM].view(3, J, tl.TOKEN_DIM)
    junk = torch.randn_like(tok) * 50.0
    junk[..., tl.ENABLED_COL] = 0.0
    tok[~enabled] = junk[~enabled]
    mu2, _s2, v2, _ = net({"obs": noisy})
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(v, v2, atol=1e-5)


def test_plain_ppo_path_unchanged():
    """Without SAPG kwargs the network is the plain-PPO one (one log-std row)."""
    from isaacsimenvs.inhand_reorient import token_policy as tp

    p = _sapg_params()
    p["space"]["continuous"]["fixed_sigma"] = "fixed"
    b = tp.InhandJointTransformerBuilder()
    b.load(p)
    net = b.build("x", actions_num=J, input_shape=(tl.layout(J)["obs_dim"],), num_seqs=1, value_size=1)
    assert net.sigma.shape == (J,) and net.global_proj.in_features == tl.GLOBAL_DIM


# --------------------------------------------------------------------------
# Recipe configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [R1, R2])
def test_recipe_matches_the_collaborators_config(name):
    """lr 1e-4 adaptive (KL 0.016), clip 0.1, 2 mini-epochs, horizon 16,
    d_model 128, 4 layers: PoseReachJointTransformerSAPG.yaml's numbers."""
    team = _yaml("PoseReachJointTransformerSAPG.yaml")
    ours = _yaml(name)
    for key in ("learning_rate", "lr_schedule", "kl_threshold", "e_clip", "mini_epochs", "horizon_length",
                "gamma", "tau", "critic_coef", "grad_norm", "bounds_loss_coef", "entropy_coef",
                "normalize_value", "normalize_advantage", "mixed_precision", "seq_length"):
        assert _same(ours["config"][key], team["config"][key]), key
    for key in ("d_model", "n_layers", "n_heads", "ff_mult", "value_head_units"):
        assert ours["network"][key] == team["network"][key], key
    assert ours["network"]["name"] == "inhand_joint_transformer"
    assert ours["config"]["normalize_input"] is False  # the network normalises valid tokens itself
    assert ours["network"]["space"]["continuous"]["logstd_max"] == 0.0
    assert ours["config"].get("group_advantage_norm", False) is False


def test_r2_is_r1_plus_the_sapg_block():
    r1, r2, team = _yaml(R1), _yaml(R2), _yaml("PoseReachJointTransformerSAPG.yaml")
    sapg_keys = {"use_others_experience", "off_policy_ratio", "expl_type", "expl_reward_type",
                 "expl_reward_coef_embd_size", "expl_reward_coef_scale", "expl_coef_block_size"}
    for key in sapg_keys - {"expl_coef_block_size"}:
        assert r2["config"][key] == team["config"][key], key
    assert r1["config"]["expl_type"] == "none" and r1["config"]["use_others_experience"] == "none"
    diff = {k for k in set(r1["config"]) | set(r2["config"]) if r1["config"].get(k) != r2["config"].get(k)}
    assert diff - {"name", "full_experiment_name"} <= sapg_keys
    assert r1["network"]["space"]["continuous"]["fixed_sigma"] == "fixed"
    assert r2["network"]["space"]["continuous"]["fixed_sigma"] == "coef_cond"
    r1n = {k: v for k, v in r1["network"].items() if k != "space"}
    r2n = {k: v for k, v in r2["network"].items() if k != "space"}
    assert r1n == r2n


def test_sapg_block_size_gives_six_blocks_and_minibatches_divide():
    """num_envs = 6 x expl_coef_block_size (players.py hardcodes 6 blocks);
    the minibatch divides the PPO batch and SAPG's augmented batch."""
    r1, r2 = _yaml(R1)["config"], _yaml(R2)["config"]
    envs = N_BLOCKS * r2["expl_coef_block_size"]
    assert envs == 12288
    batch = envs * r2["horizon_length"]
    assert batch % r1["minibatch_size"] == 0
    assert (batch + batch // N_BLOCKS) % r2["minibatch_size"] == 0


def test_entry_points_registered():
    import gymnasium as gym

    import isaacsimenvs.inhand_reorient  # noqa: F401

    kw = gym.spec("GenMech-InHandReorient-Direct-v0").kwargs
    assert Path(kw["rl_games_hora_token_team_ppo_cfg_entry_point"]).name == R1
    assert Path(kw["rl_games_hora_token_team_sapg_cfg_entry_point"]).name == R2


@pytest.mark.parametrize("name", [R1, R2])
def test_recipe_network_builds_from_the_yaml(name):
    from isaacsimenvs.inhand_reorient import token_policy as tp

    params = _yaml(name)["network"]
    b = tp.InhandJointTransformerBuilder()
    b.load(params)
    obs_dim = tl.layout(J)["obs_dim"]
    if params["space"]["continuous"]["fixed_sigma"] == "coef_cond":
        net = b.build("x", actions_num=J, input_shape=(obs_dim + 1,), num_seqs=1, value_size=1,
                      type="extra_param", coef_ids=COEF_IDS.clone(), coef_id_idx=obs_dim)
        obs, enabled = _obs(16, [0, 5])
    else:
        net = b.build("x", actions_num=J, input_shape=(obs_dim,), num_seqs=1, value_size=1)
        obs, enabled = _obs(16, [0, 5])
        obs = obs[:, :-1]
    mu, logstd, value, _ = net.eval()({"obs": obs})
    assert mu.shape == (2, J) and torch.all(mu[~enabled] == 0) and torch.all(logstd <= 0.0)
