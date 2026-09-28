"""Fake-env CPU tests for ``nan_guard.py`` (per-env non-finite physics guard)
and its per-design counting in ``design_scoring.py``. Neither module imports
``isaaclab`` at module scope, so both run under plain pytest with a fake env
exposing only the torch attributes they read.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_nan_guard.py -q
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from isaacsimenvs.inhand_reorient import design_scoring as ds
from isaacsimenvs.inhand_reorient import nan_guard as ng

NAN = float("nan")
INF = float("inf")


def _fake_env(num_envs: int = 4, n_joints: int = 3, design_idx=None):
    env = SimpleNamespace()
    env.num_envs = num_envs
    env.device = "cpu"
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1.0 / 120.0), decimation=2, episode_length_s=10.0,
        reward=SimpleNamespace(drop_penalty=50.0), obs=SimpleNamespace(clamp_abs_observations=10.0),
    )
    env.object = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.zeros(num_envs, 3), root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(num_envs, 1),
        root_lin_vel_w=torch.zeros(num_envs, 3), root_ang_vel_w=torch.zeros(num_envs, 3),
    ))
    env.robot = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(num_envs, n_joints), joint_vel=torch.zeros(num_envs, n_joints),
    ))
    env._rot_error = torch.full((num_envs,), 1.0)
    env._prev_rot_error = torch.full((num_envs,), 1.0)
    env._obj_pos_palm = torch.zeros(num_envs, 3)
    env._spawn_obj_pos_palm = torch.full((num_envs, 3), 0.05)
    env._obj_quat_palm = torch.tensor([[1.0, 0, 0, 0]]).repeat(num_envs, 1)
    env._goal_quat_palm = env._obj_quat_palm.clone()
    env._obj_quat_rel_goal = env._obj_quat_palm.clone()
    env._obj_lin_vel_palm = torch.zeros(num_envs, 3)
    env._obj_ang_vel_palm = torch.zeros(num_envs, 3)
    env._fingertip_pos_palm = torch.zeros(num_envs, 6)
    env._successes = torch.zeros(num_envs, dtype=torch.long)
    env._current_success_tolerance = 0.4
    env._goal_curriculum_stage = 0
    if design_idx is not None:
        n_designs = int(max(design_idx) + 1)
        env.hand_tables = SimpleNamespace(
            n_designs=n_designs,
            designs=[SimpleNamespace(source=f"design:{i}", sha256=f"sha:{i}") for i in range(n_designs)],
        )
        env.scene_record = {"design_idx": torch.as_tensor(design_idx, dtype=torch.long)}
    else:
        env.hand_tables = None
        env.scene_record = None
    ng.allocate_guard_buffers(env)
    return env


# --------------------------------------------------------------------------
# nonfinite_rows
# --------------------------------------------------------------------------


def test_nonfinite_rows_flags_nan_and_inf_per_row_across_tensors():
    a = torch.zeros(4, 3)
    b = torch.zeros(4)
    a[1, 2] = NAN
    b[3] = -INF
    mask = ng.nonfinite_rows(a, b, None)
    assert mask.tolist() == [False, True, False, True]


def test_nonfinite_rows_of_all_finite_inputs_is_all_false():
    mask = ng.nonfinite_rows(torch.randn(5, 2, 3), torch.randn(5))
    assert mask.dtype == torch.bool
    assert not mask.any()


# --------------------------------------------------------------------------
# detect + sanitize (the _get_dones side)
# --------------------------------------------------------------------------


def test_detect_flags_envs_whose_object_or_joint_state_is_non_finite():
    env = _fake_env(num_envs=5)
    env.object.data.root_pos_w[1, 0] = NAN
    env.object.data.root_ang_vel_w[2, 1] = INF
    env.robot.data.joint_vel[4, 0] = NAN
    mask = ng.detect_nonfinite_envs(env)
    assert mask.tolist() == [False, True, True, False, True]


def test_detect_flags_a_non_finite_rotation_error_even_with_finite_state():
    """A non-normalised quaternion can make the derived error NaN on its own."""
    env = _fake_env(num_envs=3)
    env._rot_error[0] = NAN
    assert ng.detect_nonfinite_envs(env).tolist() == [True, False, False]


def test_guard_step_state_replaces_flagged_geometry_with_finite_placeholders():
    env = _fake_env(num_envs=3)
    env.object.data.root_pos_w[1] = NAN
    env._rot_error[1] = NAN
    env._obj_pos_palm[1] = NAN
    env._obj_quat_rel_goal[1] = NAN
    env._fingertip_pos_palm[1] = NAN
    mask = ng.guard_step_state(env)
    assert mask.tolist() == [False, True, False]
    assert env._nonfinite_mask.tolist() == [False, True, False]
    assert env._rot_error[1].item() == pytest.approx(math.pi)  # worst possible error, never a spurious success
    assert torch.equal(env._obj_pos_palm[1], env._spawn_obj_pos_palm[1])
    assert torch.equal(env._obj_quat_rel_goal[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(env._fingertip_pos_palm[1], torch.zeros(6))
    # untouched envs keep their values
    assert env._rot_error[0].item() == 1.0
    for name in ng.GEOMETRY_CACHE_FIELDS:
        assert torch.isfinite(getattr(env, name)).all(), name


def test_guard_step_state_counts_flagged_env_steps():
    env = _fake_env(num_envs=3)
    env.robot.data.joint_pos[0, 0] = NAN
    ng.guard_step_state(env)
    ng.guard_step_state(env)
    assert env._nonfinite_env_steps_total == 2


def test_add_nonfinite_termination_terminates_flagged_envs_and_records_the_reason():
    env = _fake_env(num_envs=4)
    env._nonfinite_mask = torch.tensor([False, True, False, False])
    env._termination_reasons = {"drop": torch.tensor([False, False, True, False])}
    terminated = torch.tensor([False, False, True, False])
    out = ng.add_nonfinite_termination(env, terminated)
    assert out.tolist() == [False, True, True, False]
    assert env._termination_reasons["nonfinite"].tolist() == [False, True, False, False]


def test_add_nonfinite_termination_always_publishes_the_reason_key():
    """So `episode_final/done_nonfinite` is logged (as 0) on healthy runs too."""
    env = _fake_env(num_envs=2)
    env._nonfinite_mask = torch.zeros(2, dtype=torch.bool)
    env._termination_reasons = {"drop": torch.zeros(2, dtype=torch.bool)}
    ng.add_nonfinite_termination(env, torch.zeros(2, dtype=torch.bool))
    assert "nonfinite" in env._termination_reasons


# --------------------------------------------------------------------------
# rewards and observations
# --------------------------------------------------------------------------


def test_sanitize_reward_gives_flagged_envs_the_drop_penalty_and_keeps_terms_finite():
    env = _fake_env(num_envs=3)
    env._nonfinite_mask = torch.tensor([False, True, False])
    reward = torch.tensor([1.0, NAN, 2.0])
    env._reward_terms = {"rotation_rew": torch.tensor([1.0, NAN, 2.0]), "total_reward": reward.clone()}
    out = ng.sanitize_reward(env, reward)
    assert out.tolist() == [1.0, -50.0, 2.0]
    assert env._reward_terms["rotation_rew"].tolist() == [1.0, 0.0, 2.0]
    assert env._reward_terms["total_reward"].tolist() == [1.0, -50.0, 2.0]


def test_sanitize_reward_also_catches_a_non_finite_reward_on_an_unflagged_env():
    env = _fake_env(num_envs=2)
    env._nonfinite_mask = torch.zeros(2, dtype=torch.bool)
    env._reward_terms = {}
    out = ng.sanitize_reward(env, torch.tensor([INF, 0.5]))
    assert torch.isfinite(out).all()
    assert out[1].item() == 0.5


def test_sanitize_observations_zeroes_non_finite_entries_and_counts_rows():
    env = _fake_env(num_envs=3)
    obs = torch.zeros(3, 4)
    obs[2, 1] = NAN
    obs[0, 3] = INF
    out = ng.sanitize_observations(env, {"policy": obs, "critic": obs.clone()})
    for key in ("policy", "critic"):
        assert torch.isfinite(out[key]).all()
    assert out["policy"][2, 1].item() == 0.0
    assert out["policy"][0, 3].item() == 10.0  # clamped to the obs bound, not left at inf
    assert env._nonfinite_obs_rows_total == 2


def test_sanitize_observations_is_a_passthrough_for_finite_obs():
    env = _fake_env(num_envs=2)
    obs = torch.randn(2, 5)
    out = ng.sanitize_observations(env, {"policy": obs, "critic": obs})
    assert torch.equal(out["policy"], obs)
    assert env._nonfinite_obs_rows_total == 0


# --------------------------------------------------------------------------
# per-design counting in design_scoring
# --------------------------------------------------------------------------


def test_design_scoring_counts_nonfinite_resets_per_design(tmp_path):
    env = _fake_env(num_envs=4, design_idx=[0, 1, 0, 1])
    ds.allocate_scoring_buffers(env)
    env._score_output_path = tmp_path / "per_design_scores_rank0.json"
    ds.reset_scoring_state(env, torch.arange(4))
    ds.step_scoring_state(env)
    env._termination_reasons = {
        "drop": torch.tensor([True, False, False, False]),
        "nonfinite": torch.tensor([False, True, False, True]),
        "timeout": torch.zeros(4, dtype=torch.bool),
    }
    ds.bank_done_episodes(env, torch.zeros(4))
    ds.maybe_write(env)
    doc = json.loads(env._score_output_path.read_text())
    assert doc["designs"]["1"]["nonfinite_resets"] == 2
    assert doc["designs"]["1"]["nonfinite_resets_total"] == 2
    assert doc["designs"]["0"]["nonfinite_resets"] == 0
    assert doc["nonfinite_resets"] == 2
    assert doc["nonfinite_resets_total"] == 2

    # the window counter resets on write; the cumulative one does not
    ds.step_scoring_state(env)
    env._termination_reasons = {
        "drop": torch.zeros(4, dtype=torch.bool), "nonfinite": torch.tensor([False, True, False, False]),
    }
    ds.bank_done_episodes(env, torch.zeros(4))
    ds.maybe_write(env)
    doc = json.loads(env._score_output_path.read_text())
    assert doc["designs"]["1"]["nonfinite_resets"] == 1
    assert doc["designs"]["1"]["nonfinite_resets_total"] == 3
    assert doc["nonfinite_resets_total"] == 3


def test_design_scoring_without_a_nonfinite_reason_reports_zero():
    env = _fake_env(num_envs=2, design_idx=[0, 1])
    ds.allocate_scoring_buffers(env)
    ds.reset_scoring_state(env, torch.arange(2))
    ds.step_scoring_state(env)
    env._termination_reasons = {"drop": torch.tensor([True, True])}
    ds.bank_done_episodes(env, torch.zeros(2))
    snap = ds.read_snapshot(env)
    assert snap["nonfinite_resets"] == 0
    assert snap["designs"]["0"]["nonfinite_resets"] == 0
