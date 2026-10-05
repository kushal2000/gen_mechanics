"""CPU tests for the GET-Zero-style distillation (``distill/``) and the env's
``hora.teacher_obs`` option: the expert's flat observation, the DAgger
schedule and action mix, the aggregated dataset, the masked action loss,
expert loading and routing, first-episode evaluation, checkpoints, the
student config, and the Kit-safe entry point.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_distill.py -q
"""

from __future__ import annotations

import ast
import math
import types
from pathlib import Path

import pytest
import torch
import yaml

from isaacsimenvs.inhand_reorient import token_layout as tl
from isaacsimenvs.inhand_reorient.distill import dagger as dg

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "coevolution" / "cfg"
J = 32


def _yaml(name):
    return yaml.safe_load((CFG / "train" / name).read_text())["params"]


# --------------------------------------------------------------------------
# Env option: hora.teacher_obs
# --------------------------------------------------------------------------


def test_flat_observation_is_the_solo_mlp_input():
    from isaacsimenvs.inhand_reorient import anyrotate_profile as ar
    from isaacsimenvs.inhand_reorient import hora_profile as hp

    n = 3
    hist = torch.arange(n * 3 * 2 * J, dtype=torch.float32).view(n, 3, 2 * J)
    priv = -torch.arange(n * 9, dtype=torch.float32).view(n, 9)
    flat = hp.flat_observation(hist, priv)
    width = ar.anyrotate_field_width("hora_proprio_hist", J, 5) + ar.anyrotate_field_width("hora_priv", J, 5)
    assert flat.shape == (n, width) == (n, 201)
    assert torch.equal(flat[:, : 2 * J], hist[:, 0]) and torch.equal(flat[:, 4 * J: 6 * J], hist[:, 2])
    assert torch.equal(flat[:, -9:], priv)


def _hora_cfg(token_obs, teacher_obs):
    ns = types.SimpleNamespace
    hora = ns(sim_dt=1 / 120, decimation=6, episode_length_s=20.0, friction=1.0, action_scale=1 / 24,
              z_axis_frame="world_up", morph_obs=False, token_obs=token_obs, teacher_obs=teacher_obs,
              ghost_action_mask=False, disjoint_slots=False, design_id_obs=0)
    anyrotate = ns(hand_orientation_randomization=False, restitution=0.0)
    sim = ns(dt=0.0, render_interval=1, physics_material=ns(static_friction=0, dynamic_friction=0, restitution=0))
    return ns(task_profile="hora", hora=hora, anyrotate=anyrotate, sim=sim, scene=ns(clone_in_fabric=True),
              obs=ns(obs_list=(), state_list=()), decimation=1, episode_length_s=1.0)


def test_teacher_obs_is_off_by_default_and_needs_token_obs():
    from isaacsimenvs.inhand_reorient import anyrotate_profile as ar

    h = yaml.safe_load((CFG / "task" / "InHandReorient.yaml").read_text())["hora"]
    assert h["teacher_obs"] is False
    with pytest.raises(ValueError, match="teacher_obs"):
        ar.apply_anyrotate_to_cfg(_hora_cfg(token_obs=False, teacher_obs=True))
    cfg = _hora_cfg(token_obs=True, teacher_obs=True)
    ar.apply_anyrotate_to_cfg(cfg)
    assert cfg.obs.obs_list == ("hora_tokens", "hora_token_global")  # the student's observation is unchanged


def test_hooks_emit_the_teacher_obs_from_the_shared_helper():
    """``anyrotate_hooks`` needs Kit; check its source: the flat path and the
    teacher path both build the MLP input with ``hp.flat_observation``."""
    src = (ROOT / "isaacsimenvs" / "inhand_reorient" / "anyrotate_hooks.py").read_text()
    assert 'obs["teacher_obs"] = hp.flat_observation(env._hora_hist, priv)' in src
    assert "parts = [hp.flat_observation(env._hora_hist, priv)]" in src


# --------------------------------------------------------------------------
# DAgger schedule, mix, loss, dataset
# --------------------------------------------------------------------------


def test_beta_schedule_warmup_then_linear_then_floor():
    assert [dg.beta_at(i, 3, 4) for i in (0, 2)] == [1.0, 1.0]
    assert dg.beta_at(3, 3, 4) == 1.0 and dg.beta_at(5, 3, 4) == pytest.approx(0.5)
    assert dg.beta_at(7, 3, 4) == 0.0 and dg.beta_at(100, 3, 4) == 0.0
    assert dg.beta_at(100, 3, 4, beta_min=0.2) == pytest.approx(0.2)
    assert dg.beta_at(5, 3, 0, beta_min=0.1) == pytest.approx(0.1)


def test_mix_actions_picks_whole_rows():
    n = 4000
    e, s = torch.ones(n, J), -torch.ones(n, J)
    a, used = dg.mix_actions(e, s, 1.0)
    assert torch.equal(a, e) and bool(used.all())
    a, used = dg.mix_actions(e, s, 0.0)
    assert torch.equal(a, s) and not bool(used.any())
    a, used = dg.mix_actions(e, s, 0.3, generator=torch.Generator().manual_seed(0))
    assert torch.equal(a[used], e[used]) and torch.equal(a[~used], s[~used])
    assert abs(float(used.float().mean()) - 0.3) < 0.03


def _tokens(n_valid, n=None, seed=0):
    n_valid = list(n_valid)
    n = len(n_valid)
    g = torch.Generator().manual_seed(seed)
    tok = torch.randn(n, J, tl.TOKEN_DIM, generator=g)
    enabled = torch.zeros(n, J, dtype=torch.bool)
    for i, v in enumerate(n_valid):
        enabled[i, :v] = True
    tok[..., tl.ENABLED_COL] = enabled.float()
    tok = tok * enabled.unsqueeze(-1)
    return torch.cat([tok.reshape(n, -1), torch.randn(n, tl.GLOBAL_DIM, generator=g)], dim=-1), enabled


def test_valid_mask_is_the_raw_enabled_column():
    obs, enabled = _tokens([3, 16, 22])
    assert torch.equal(dg.valid_from_tokens(obs, J, tl.TOKEN_DIM, tl.ENABLED_COL), enabled)


def test_masked_action_mse_ignores_ghost_columns():
    _, valid = _tokens([2, 5])
    mu, y = torch.zeros(2, J), torch.zeros(2, J)
    y[0, :2] = 1.0  # sample 0: error 1 on both real joints
    y[1, :5] = 2.0  # sample 1: error 4 on all five
    loss, per = dg.masked_action_mse(mu, y, valid)
    assert torch.allclose(per, torch.tensor([1.0, 4.0])) and float(loss) == pytest.approx(2.5)
    y2 = y.clone()
    y2[~valid] = 100.0  # ghost labels carry no loss
    assert float(dg.masked_action_mse(mu, y2, valid)[0]) == pytest.approx(2.5)
    m, c = dg.per_design_mean(per, torch.tensor([1, 1]), 3)
    assert torch.allclose(m, torch.tensor([0.0, 2.5, 0.0])) and torch.equal(c, torch.tensor([0.0, 2.0, 0.0]))


def test_aggregated_dataset_is_a_ring_buffer():
    ds = dg.AggregatedDataset(5, 2, 1, "cpu")
    for k in range(3):  # 9 rows into 5 slots
        obs = torch.arange(3 * k, 3 * k + 3, dtype=torch.float32).unsqueeze(-1).expand(3, 2)
        ds.add(obs, obs[:, :1] * 10, torch.full((3,), k))
    assert len(ds) == 5 and ds.total == 9
    assert sorted(ds.obs[:, 0].tolist()) == [4.0, 5.0, 6.0, 7.0, 8.0]  # the newest rows
    o, y, d = ds.sample(64, generator=torch.Generator().manual_seed(0))
    assert torch.equal(y[:, 0], o[:, 0] * 10) and set(o[:, 0].tolist()) <= {4.0, 5.0, 6.0, 7.0, 8.0}
    assert torch.equal(d, (o[:, 0] // 3).long())
    big = dg.AggregatedDataset(4, 1, 1, "cpu")
    big.add(torch.arange(10.0).unsqueeze(-1), torch.zeros(10, 1), torch.zeros(10))
    assert sorted(big.obs[:, 0].tolist()) == [6.0, 7.0, 8.0, 9.0] and big.total == 10


# --------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------


def _expert_params():
    return _yaml("InHandAnyRotatePPO.yaml")


def _trained_like_expert(seed=0):
    torch.manual_seed(seed)
    model = dg.build_rlg_model(_expert_params(), 201, J)
    model.train()
    with torch.no_grad():  # move the input normaliser away from its initial statistics
        model.norm_obs(torch.randn(512, 201) * 3.0 + 1.0)
    return model


def test_expert_loads_with_its_frozen_input_normaliser(tmp_path):
    model = _trained_like_expert()
    path = tmp_path / "expert.pth"
    torch.save({0: {"model": model.state_dict(), "epoch": 7}}, path)  # the vendored rl_games' format
    pol = dg.load_expert(_expert_params(), path, 201, J)
    assert not pol.training and all(not p.requires_grad for p in pol.parameters())
    mean0 = pol.model.running_mean_std.running_mean.clone()
    obs = torch.randn(64, 201) * 20.0
    model.eval()
    with torch.no_grad():
        ref = model.a2c_network({"obs": model.norm_obs(obs.clamp(-10.0, 10.0))})[0].clamp(-1.0, 1.0)
        out = pol(obs)
    assert torch.allclose(out, ref, atol=1e-6)
    assert torch.equal(pol.model.running_mean_std.running_mean, mean0)  # eval: the statistics stay frozen
    assert float(out.abs().max()) <= 1.0


def test_expert_bank_routes_each_env_to_its_own_design():
    design = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    experts = [lambda o: torch.full((o.shape[0], J), 1.0), lambda o: o[:, :J] * 0 + 2.0]
    bank = dg.ExpertBank(experts, design, [1, 0, None])
    a = bank.act(torch.zeros(7, 201), J)
    assert torch.equal(a[:, 0], torch.tensor([2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0])) and bank.missing == [2]
    assert dg.resolve_experts(["a", "b"], {"b": "x.pth"}) == [None, "x.pth"]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_first_episode_tracker_counts_each_env_once():
    design = torch.tensor([0, 0, 1, 1])
    tr = dg.FirstEpisodeTracker(design, 2)
    f = lambda *x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
    tr.update(torch.tensor([True, False, False, False]), f(1.0, 9, 9, 9), f(-0.5, 9, 9, 9))
    tr.update(torch.tensor([True, True, False, True]), f(5.0, 3.0, 9, 20.0), f(9, 2 * math.pi, 9, 4 * math.pi))
    assert not tr.all_done
    tr.update(torch.tensor([False, False, True, False]), f(0, 0, 10.0, 0), f(0, 0, math.pi, 0))
    assert tr.all_done
    s = dg.summarise(tr.sums(20.0), ["a", "b"])
    assert s["a"]["episodes"] == 2 and s["a"]["ttt_mean_s"] == pytest.approx(2.0)  # env 0's first episode: 1 s
    assert s["a"]["rotations_mean"] == pytest.approx(0.5)  # (0 clamped + 1) / 2
    assert s["a"]["rotations_signed_mean"] == pytest.approx((-0.5 / (2 * math.pi) + 1) / 2)
    assert s["b"]["ttt_mean_s"] == pytest.approx(15.0) and s["b"]["timeout_frac"] == pytest.approx(0.5)
    assert s["b"]["rad_per_s"] == pytest.approx(5 * math.pi / 30.0)
    both = dg.add_sums(dg.add_sums(None, tr.sums(20.0)), tr.sums(20.0))
    assert dg.summarise(both, ["a", "b"])["a"]["episodes"] == 4


# --------------------------------------------------------------------------
# Student, training step, checkpoints, entry point
# --------------------------------------------------------------------------


def test_distill_config_uses_the_team_transformer_and_registers():
    p, team = _yaml("InHandHoraTokenDistill.yaml"), _yaml("InHandHoraTokenTeamPPO.yaml")
    assert p["network"] == team["network"] and p["network"]["name"] == "inhand_joint_transformer"
    assert p["config"]["normalize_input"] is False  # the RAW enabled column stays the mask
    d = p["distill"]
    for k in ("rollout_steps", "grad_steps", "minibatch", "learning_rate", "buffer_capacity", "warmup_iters",
              "decay_iters", "beta_min", "log_every_iters", "ckpt_every_iters", "eval_every_s", "eval_rounds"):
        assert k in d, k
    import gymnasium as gym

    import isaacsimenvs.inhand_reorient  # noqa: F401  registers the task
    kw = gym.spec("GenMech-InHandReorient-Direct-v0").kwargs
    assert Path(kw["rl_games_hora_token_distill_cfg_entry_point"]).name == "InHandHoraTokenDistill.yaml"
    assert Path(kw[d["expert_agent"]]).name == "InHandAnyRotatePPO.yaml"


def _small_student():
    from isaacsimenvs.inhand_reorient import token_policy  # noqa: F401  registers the network

    p = _yaml("InHandHoraTokenDistill.yaml")
    p["network"].update(d_model=16, n_layers=1, value_head_units=[16])
    return dg.build_rlg_model(p, tl.layout(J)["obs_dim"], J)


def test_student_imitates_a_per_joint_expert_and_ghosts_get_nothing():
    """A few masked-MSE steps on a toy labelling (each real joint's action
    is its token's first feature) cut the loss; ghost columns stay 0 with
    no gradient."""
    torch.manual_seed(0)
    student = _small_student()
    from isaacsimenvs.inhand_reorient.token_policy import InhandJointTransformerNet

    assert isinstance(student.a2c_network, InhandJointTransformerNet)
    obs, valid = _tokens([4, 16, 22, 9] * 16, seed=1)
    y = obs[:, : J * tl.TOKEN_DIM].view(-1, J, tl.TOKEN_DIM)[..., 0].clamp(-1, 1) * valid
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    losses = []
    student.train()
    for _ in range(60):
        mu = student.a2c_network({"obs": student.norm_obs(obs)})[0]
        loss, _ = dg.masked_action_mse(mu, y, dg.valid_from_tokens(obs, J, tl.TOKEN_DIM, tl.ENABLED_COL))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]
    student.eval()
    mu = dg.MeanPolicy(student, obs_clip=10.0)(obs)
    assert torch.all(mu[~valid] == 0)


def test_checkpoint_roundtrip_restores_student_optimizer_and_state(tmp_path):
    student = _small_student()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    obs, _ = _tokens([5, 7])
    student.train()
    student.a2c_network({"obs": obs})[0].sum().backward()
    opt.step()
    path = tmp_path / "student_last.pth"
    dg.save_checkpoint(path, student, opt, {"iteration": 12, "samples": 345, "train_s": 6.5})
    raw = torch.load(path, weights_only=False)
    assert set(raw) == {0} and raw[0]["epoch"] == 12 and raw[0]["frame"] == 345 and "model" in raw[0]
    fresh = _small_student()
    opt2 = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    st = dg.load_checkpoint(path, fresh, opt2)
    assert st == {"iteration": 12, "samples": 345, "train_s": 6.5}
    for (k, a), (_, b) in zip(student.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), k
    assert opt2.state_dict()["state"].keys() == opt.state_dict()["state"].keys()
    assert not list(tmp_path.glob("*.tmp"))


def test_entry_point_boots_kit_before_isaaclab_imports():
    """``run.py`` imports only the standard library at module level; the
    AppLauncher import comes first inside ``main``."""
    src = (ROOT / "isaacsimenvs" / "inhand_reorient" / "distill" / "run.py").read_text()
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    mods = {(n.module if isinstance(n, ast.ImportFrom) else n.names[0].name).split(".")[0] for n in top}
    assert mods <= {"__future__", "json", "os", "sys", "time", "pathlib"}, mods
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    imports = [n for n in ast.walk(main) if isinstance(n, (ast.Import, ast.ImportFrom))]
    first_heavy = next(n for n in imports if (getattr(n, "module", None) or n.names[0].name).split(".")[0]
                       not in {"argparse"})
    assert first_heavy.module == "isaaclab.app" and first_heavy.names[0].name == "AppLauncher"


# --------------------------------------------------------------------------
# Expert selection
# --------------------------------------------------------------------------


def _row(ttt, rot, n=100):
    return {"episodes": n, "ttt_mean_s": ttt, "rotations_mean": rot, "rad_per_s": rot * 2 * math.pi / ttt}


def test_select_experts_picks_the_checkpoint_and_applies_the_keep_rule():
    from isaacsimenvs.inhand_reorient.distill import select_experts as se

    evals = {"best": {"a": _row(13.45, 1.329), "b": _row(0.44, 0.058), "c": _row(15.0, 0.02)},
             "last": {"a": _row(11.49, 1.319), "b": _row(0.52, 0.063), "c": _row(15.0, 0.019)}}
    zero = {"a": _row(17.4, 0.009), "b": _row(18.9, 0.004), "c": _row(16.0, 0.012)}
    maps = {k: {s: f"{k}/{s}.pth" for s in "abc"} for k in ("best", "last")}
    rows = se.select(evals, zero, maps)
    assert rows["a"]["checkpoint"] == "best" and rows["a"]["path"] == "best/a.pth" and rows["a"]["kept"]
    assert rows["b"]["checkpoint"] == "last" and rows["b"]["kept"]  # a flick still turns a full turn in 30 s
    assert not rows["c"]["kept"]  # 0.05 rad/s while holding: no full turn within 30 s
    assert se.keep(_row(1.0, 0.03), _row(15.0, 0.02)) is False  # below twice the zero-action rotation
    assert se.keep(_row(1.0, 0.05), _row(15.0, 0.02)) is True
