"""CPU tests for the per-joint token observation (``token_layout.py``) and
the team's joint-token transformer on the in-hand envelope
(``token_policy.py``): layout, mask integrity (the RAW enabled column is the
mask; ghost tokens change nothing), valid-token normalisation, and
per-design invariance.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_token_policy.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from isaacsimenvs.inhand_reorient import token_layout as tl

CFG = Path(__file__).resolve().parents[3] / "coevolution" / "cfg"
J = 32


def test_layout_constants_and_columns():
    assert tl.TOKEN_DIM == 33 and tl.ENABLED_COL == 20 and tl.GLOBAL_DIM == 12
    lay = tl.layout(J, design_id_width=2)
    assert lay["obs_dim"] == J * 33 + 12 + 2
    assert lay["token_columns"][1] == list(range(33, 66))
    assert lay["global_slice"] == (J * 33, J * 33 + 12)
    assert lay["design_slice"] == (J * 33 + 12, J * 33 + 14)
    assert tl.layout(J)["obs_dim"] == J * 33 + 12


def test_slot_body_names_follow_the_authored_bodies():
    assert tl.slot_body_name(0) == "f0_link0"
    assert tl.slot_body_name(29) == "f4_link5"
    assert tl.slot_body_name(30) == "pc0" and tl.slot_body_name(31) == "pc1"


def test_assemble_zeroes_ghost_tokens_and_keeps_enabled_raw():
    n = 2
    enabled = torch.zeros(n, J, dtype=torch.bool)
    enabled[:, :5] = True
    ones = lambda w: torch.ones(n, J, w)
    flat = tl.assemble(ones(3), ones(3), ones(12), ones(2), enabled, ones(12))
    tok = flat.view(n, J, tl.TOKEN_DIM)
    assert torch.all(tok[:, 5:] == 0)
    assert torch.all(tok[:, :5, tl.ENABLED_COL] == 1) and torch.all(tok[:, :5] == 1)


def test_field_widths_and_default_off():
    from isaacsimenvs.inhand_reorient import anyrotate_profile as ar

    assert ar.anyrotate_field_width("hora_tokens", J, 5) == J * tl.TOKEN_DIM
    assert ar.anyrotate_field_width("hora_token_global", J, 5) == tl.GLOBAL_DIM
    h = yaml.safe_load((CFG / "task" / "InHandReorient.yaml").read_text())["hora"]
    assert h["token_obs"] is False


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------


def _params(design_id=0, per_design_global=False):
    return {
        "name": "inhand_joint_transformer", "separate": False, "d_model": 16, "n_layers": 2, "n_heads": 2,
        "ff_mult": 2, "value_head_units": [32], "arm_head_units": [8], "design_id_obs": design_id,
        "per_design_global_norm": per_design_global,
        "space": {"continuous": {"mu_activation": "None", "sigma_activation": "None", "fixed_sigma": "fixed",
                                 "mu_init": {"name": "default"},
                                 "sigma_init": {"name": "const_initializer", "val": 0}}},
    }


def _net(params):
    from isaacsimenvs.inhand_reorient import token_policy as tp

    b = tp.InhandJointTransformerBuilder()
    b.load(params)
    k = int(params.get("design_id_obs") or 0)
    return b.build("x", actions_num=J, input_shape=(tl.layout(J, k)["obs_dim"],), num_seqs=1, value_size=1)


def _obs(n_valid, n=4, design=None, k=0, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(n, J, tl.TOKEN_DIM, generator=g) * scale
    enabled = torch.zeros(n, J, dtype=torch.bool)
    for i, v in enumerate(n_valid if isinstance(n_valid, list) else [n_valid] * n):
        enabled[i, :v] = True
    tok[..., tl.ENABLED_COL] = enabled.float()
    tok = tok * enabled.unsqueeze(-1)
    parts = [tok.reshape(n, -1), torch.randn(n, tl.GLOBAL_DIM, generator=g) * scale]
    if k:
        parts.append(torch.nn.functional.one_hot(torch.tensor(design if design is not None else [0] * n), k).float())
    return torch.cat(parts, dim=-1), enabled


def test_network_registered_and_shapes():
    from rl_games.algos_torch import model_builder

    from isaacsimenvs.inhand_reorient import token_policy as tp

    assert tp.NETWORK_NAME in model_builder.NETWORK_REGISTRY
    net = _net(_params()).eval()
    obs, _ = _obs(10)
    mu, sigma, value, states = net({"obs": obs})
    assert mu.shape == (4, J) and sigma.shape == (4, J) and value.shape == (4, 1) and states is None


def test_obs_width_mismatch_raises():
    from isaacsimenvs.inhand_reorient import token_policy as tp

    b = tp.InhandJointTransformerBuilder()
    b.load(_params())
    with pytest.raises(ValueError):
        b.build("x", actions_num=J, input_shape=(tl.layout(J)["obs_dim"] + 1,), num_seqs=1, value_size=1)


def test_ghost_tokens_get_no_action_and_change_nothing():
    """The RAW enabled column is the mask: ghost tokens' features (whatever
    their values) change no valid output, and ghost actions are mean 0,
    log-std 0, with no gradient."""
    net = _net(_params()).eval()
    obs, enabled = _obs([6, 16, 3, 22])
    mu, sigma, value, _ = net({"obs": obs})
    assert torch.all(mu[~enabled] == 0) and torch.all(sigma[~enabled] == 0)
    noisy = obs.clone().view(4, -1)
    tok = noisy[:, : J * tl.TOKEN_DIM].view(4, J, tl.TOKEN_DIM)
    junk = torch.randn_like(tok) * 50.0
    junk[..., tl.ENABLED_COL] = 0.0
    tok[~enabled] = junk[~enabled]
    mu2, sigma2, value2, _ = net({"obs": noisy})
    assert torch.allclose(mu, mu2, atol=1e-5) and torch.allclose(value, value2, atol=1e-5)
    (mu2[~enabled].sum() + sigma2[~enabled].sum()).backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in net.parameters())


def test_enabled_flag_survives_large_feature_values():
    """No input normaliser runs before the mask (rl_games' normalize_input
    is off for this network), so the enabled column reaches it raw."""
    net = _net(_params()).eval()
    obs, enabled = _obs(5, scale=1e3)
    mu, _s, _v, _ = net({"obs": obs})
    assert torch.all(mu[:, 5:] == 0) and torch.all(mu[:, :5] != 0)


def test_token_norm_uses_valid_tokens_only():
    """The same valid tokens with more ghost padding give the same statistics."""
    from isaacsimenvs.inhand_reorient import token_policy as tp

    x = torch.randn(64, 8, 5) * 3.0 + 1.0
    valid = torch.zeros(64, 8, dtype=torch.bool)
    valid[:, :3] = True
    a = tp.MaskedRunningNorm(5, skip_cols=(2,)).train()
    a(x, valid)
    x_pad = torch.cat([x, torch.zeros(64, 8, 5)], dim=1)
    valid_pad = torch.cat([valid, torch.zeros(64, 8, dtype=torch.bool)], dim=1)
    b = tp.MaskedRunningNorm(5, skip_cols=(2,)).train()
    b(x_pad, valid_pad)
    assert torch.allclose(a.mean, b.mean) and torch.allclose(a.var, b.var)
    assert torch.allclose(a.mean, x[:, :3].reshape(-1, 5).mean(0), atol=1e-3)
    y = a.eval()(x, valid)
    assert torch.equal(y[..., 2][valid], x[..., 2][valid])  # the skipped column passes raw
    assert torch.all(y[~valid] == 0)


def test_per_design_global_norm_keeps_each_designs_statistics():
    net = _net(_params(design_id=2, per_design_global=True)).train()
    obs, _ = _obs(8, n=200, design=[0] * 100 + [1] * 100, k=2)
    g0, g1 = slice(0, 100), slice(100, 200)
    gs, ge_ = tl.layout(J, 2)["global_slice"]
    obs[g1, gs:ge_] = obs[g1, gs:ge_] * 20.0 + 7.0
    net({"obs": obs})
    m = net.global_norm.mean
    assert torch.allclose(m[0], obs[g0, gs:ge_].mean(0), atol=1e-3)
    assert torch.allclose(m[1], obs[g1, gs:ge_].mean(0), atol=1e-3)


def test_outputs_do_not_depend_on_other_designs_in_the_batch():
    """Per-design invariance (eval): a design's actions and value are the
    same whatever other designs share its batch."""
    net = _net(_params(design_id=2, per_design_global=True)).eval()
    a, _ = _obs([6, 6, 16, 16], design=[0, 0, 1, 1], k=2, seed=1)
    b, _ = _obs([6, 6, 22, 9], design=[0, 0, 1, 1], k=2, seed=2)
    b[:2] = a[:2]
    mu_a, _, v_a, _ = net({"obs": a})
    mu_b, _, v_b, _ = net({"obs": b})
    assert torch.allclose(mu_a[:2], mu_b[:2], atol=1e-6) and torch.allclose(v_a[:2], v_b[:2], atol=1e-6)


def test_token_order_equivariance():
    """Permuting a design's valid tokens permutes its actions (shared head,
    no learned joint identity)."""
    net = _net(_params()).eval()
    obs, enabled = _obs(6, n=1)
    tok = obs[:, : J * tl.TOKEN_DIM].view(1, J, tl.TOKEN_DIM)
    perm = torch.tensor([3, 0, 5, 1, 4, 2] + list(range(6, J)))
    obs_p = torch.cat([tok[:, perm].reshape(1, -1), obs[:, J * tl.TOKEN_DIM:]], dim=-1)
    mu, _, v, _ = net({"obs": obs})
    mu_p, _, v_p, _ = net({"obs": obs_p})
    assert torch.allclose(mu[:, perm], mu_p, atol=1e-5) and torch.allclose(v, v_p, atol=1e-5)


def test_train_config_bypasses_the_rl_games_input_normaliser():
    doc = yaml.safe_load((CFG / "train" / "InHandHoraTokenPPO.yaml").read_text())["params"]
    assert doc["network"]["name"] == "inhand_joint_transformer"
    assert doc["config"]["normalize_input"] is False


def test_mu_head_init_scale_shrinks_the_initial_action_means():
    """mu_head_init_scale s < 1: the action head's last layer starts at s x
    its default weights and zero bias, so the initial mean actions (which
    HORA integrates into the joint targets every step) start near 0;
    default 1 keeps the team network's init."""
    obs, enabled = _obs(12, n=64)
    base = _net(_params()).eval()
    p = _params()
    p["mu_head_init_scale"] = 0.01
    small = _net(p).eval()
    mu_b, *_ = base({"obs": obs})
    mu_s, *_ = small({"obs": obs})
    assert mu_s[enabled].abs().max() < 0.05 * mu_b[enabled].abs().max()
    assert torch.all(mu_s[~enabled] == 0)
