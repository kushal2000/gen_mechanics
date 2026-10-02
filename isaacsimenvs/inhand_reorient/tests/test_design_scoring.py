"""Fake-env CPU tests for ``design_scoring.py`` (Part C: graded per-design
scoring). ``design_scoring.py`` imports no ``isaaclab``/``hydra`` at module
scope (see its own docstring), so it is directly testable under plain
pytest with a fake env exposing only the plain-torch attributes it reads --
same discipline as ``test_population_env_wiring.py``.

Run with the isaacsim venv (see test_grammar_envelope.py's docstring):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_design_scoring.py -q
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import torch

from isaacsimenvs.inhand_reorient import design_scoring as ds

DEVICE = "cpu"


class _FakeDesign:
    def __init__(self, source: str, sha256: str):
        self.source = source
        self.sha256 = sha256


class _FakeTables:
    def __init__(self, n_designs: int):
        self.n_designs = n_designs
        self.designs = [_FakeDesign(f"design:{i}", f"sha:{i}") for i in range(n_designs)]


def _fake_env(num_envs: int, design_idx=None, episode_length_s: float = 10.0, dt: float = 1.0 / 60.0,
              decimation: int = 2):
    env = SimpleNamespace()
    env.num_envs = num_envs
    env.device = DEVICE
    env.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=dt), decimation=decimation, episode_length_s=episode_length_s,
    )
    env._rot_error = torch.zeros(num_envs)
    env._successes = torch.zeros(num_envs, dtype=torch.long)
    env._current_success_tolerance = 0.2
    env._goal_curriculum_stage = 1
    if design_idx is not None:
        n_designs = int(max(design_idx) + 1)
        env.hand_tables = _FakeTables(n_designs)
        env.scene_record = {"design_idx": torch.as_tensor(design_idx, dtype=torch.long)}
    else:
        env.hand_tables = None
        env.scene_record = None
    ds.allocate_scoring_buffers(env)
    return env


def test_single_hand_path_banks_as_design_0():
    env = _fake_env(4, design_idx=None)
    assert env._score_n_designs == 1
    assert torch.equal(env._score_design_idx, torch.zeros(4, dtype=torch.long))


def test_graded_fitness_components_formula():
    goals = torch.tensor([0.0, 1.0, 2.0])
    rotation_progress = torch.tensor([0.0, math.pi / 2, -1.0])  # last one: error got WORSE (immediate drop)
    time_held_s = torch.tensor([0.0, 5.0, 20.0])  # last one exceeds episode_max_s
    episode_max_s = 10.0
    rot_term, time_term, fitness = ds.graded_fitness_components(goals, rotation_progress, time_held_s, episode_max_s)

    # env 0: no progress, no time held, no goals -> everything 0.
    assert fitness[0].item() == 0.0
    # env 1: half the rotation budget (pi/2 out of pi) -> 0.5 * 0.5 = 0.25; half the time budget -> 0.25*0.5=0.125.
    assert abs(rot_term[1].item() - 0.25) < 1e-6
    assert abs(time_term[1].item() - 0.125) < 1e-6
    assert abs(fitness[1].item() - (1.0 + 0.25 + 0.125)) < 1e-6
    # env 2: negative rotation_progress clamps to 0 (never penalized below goals); time clamps at the weight cap.
    assert rot_term[2].item() == 0.0
    assert abs(time_term[2].item() - ds.TIME_WEIGHT) < 1e-6
    assert abs(fitness[2].item() - (2.0 + 0.0 + ds.TIME_WEIGHT)) < 1e-6


def test_bank_done_episodes_keys_by_design_idx_at_done_time():
    """3 designs over 7 envs (shared fixture style with
    test_population_env_wiring.py) -- a drop at different rotation
    progress/time per env must land in the RIGHT design's running sums."""
    design_idx = [0, 0, 1, 1, 1, 2, 2]
    env = _fake_env(7, design_idx=design_idx, episode_length_s=8.0, dt=1.0 / 60.0, decimation=2)
    # step_dt = dt * decimation = 1/30 s.
    step_dt = env.cfg.sim.dt * env.cfg.decimation

    env_ids = torch.arange(7)
    env._rot_error[:] = torch.tensor([2.0, 2.0, 1.5, 1.5, 1.5, 0.5, 0.5])
    ds.reset_scoring_state(env, env_ids)
    assert torch.equal(env._score_start_error, env._rot_error)
    assert torch.equal(env._score_min_error, env._rot_error)
    assert torch.equal(env._score_elapsed_steps, torch.zeros(7, dtype=torch.long))

    # Simulate 5 steps of progress before a drop, tracking the min error.
    for step_error in (1.8, 1.6, 1.0, 0.9, 0.9):
        env._rot_error[:] = step_error
        ds.step_scoring_state(env)
    assert torch.equal(env._score_elapsed_steps, torch.full((7,), 5, dtype=torch.long))
    # envs 0-4 started above 0.9 and dipped to 0.9; envs 5,6 started BELOW
    # 0.9 (0.5) and the uniform step sequence never went lower, so their
    # min stays at their own starting error.
    assert torch.allclose(env._score_min_error, torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5]))

    env._successes[:] = torch.tensor([0, 1, 0, 2, 0, 0, 3])
    drop = torch.tensor([True, True, False, False, True, False, True])
    timeout = torch.tensor([False, False, True, True, False, False, False])
    env._termination_reasons = {"drop": drop, "timeout": timeout}
    reward = torch.zeros(7)
    ds.bank_done_episodes(env, reward)

    # design 0 (envs 0,1): both done (drop). start error 2.0, min 0.9 ->
    # rotation_progress 1.1 each; goals 0 and 1; elapsed 5 steps.
    assert env._score_episodes[0].item() == 2
    assert env._score_goals_sum[0].item() == 1.0
    assert abs(env._score_rotation_progress_sum[0].item() - 2 * 1.1) < 1e-6
    assert abs(env._score_time_held_sum[0].item() - 2 * 5 * step_dt) < 1e-6

    # design 1 (envs 2,3,4): env 4 dropped, envs 2/3 timed out; all 3 done.
    assert env._score_episodes[1].item() == 3
    assert env._score_goals_sum[1].item() == 2.0  # 0 + 2 + 0

    # design 2 (envs 5,6): env 5 not done, env 6 dropped -> only 1 episode banked.
    assert env._score_episodes[2].item() == 1
    assert env._score_goals_sum[2].item() == 3.0
    assert env._score_succeeded[2].item() == 1.0


def test_ghost_column_style_free_channel_does_not_leak_into_return():
    """Sanity: `bank_done_episodes` accumulates the exact `reward` tensor
    passed in (no hidden masking of its own) -- masking ghost actions is
    obs_utils's job (review item 9), not this module's; this just confirms
    the return-tracking arithmetic is a plain running sum reset at
    `reset_scoring_state`."""
    env = _fake_env(2, design_idx=[0, 0])
    env_ids = torch.arange(2)
    ds.reset_scoring_state(env, env_ids)
    for r in (1.0, 2.0, 3.0):
        env._rot_error[:] = 0.5
        ds.step_scoring_state(env)
        ds.bank_done_episodes(env, torch.tensor([r, r]))  # never done: no _termination_reasons
    assert torch.allclose(env._score_return_running, torch.tensor([6.0, 6.0]))

    env._successes[:] = 0
    env._termination_reasons = {"timeout": torch.tensor([True, True])}
    ds.bank_done_episodes(env, torch.tensor([1.0, 1.0]))
    # Both envs belong to design 0 and are both done: running return was 6.0
    # each, +1.0 this step = 7.0 each, summed into design 0's bucket = 14.0.
    assert torch.allclose(env._score_return_sum, torch.tensor([14.0]))


def test_maybe_write_and_flush_produce_valid_json_and_reset_the_window(tmp_path):
    env = _fake_env(2, design_idx=[0, 0])
    env._score_output_path = tmp_path / "per_design_scores_rank0.json"

    env_ids = torch.arange(2)
    ds.reset_scoring_state(env, env_ids)
    env._rot_error[:] = 0.5
    ds.step_scoring_state(env)
    env._successes[:] = torch.tensor([1, 0])
    env._termination_reasons = {"drop": torch.tensor([True, True])}
    ds.bank_done_episodes(env, torch.zeros(2))

    ds.maybe_write(env)
    doc = json.loads(env._score_output_path.read_text())
    assert doc["n_designs"] == 1
    assert doc["episodes"] == 2
    assert "0" in doc["designs"]
    row = doc["designs"]["0"]
    assert row["episodes"] == 2
    assert row["design_sha256"] == "sha:0"
    assert row["envs_per_design"] == 2
    assert "graded_fitness" in row
    assert "success_tolerance" in doc and "goal_curriculum_stage" in doc
    comp = row["graded_fitness_components"]
    assert abs((comp["goal_term_mean"] + comp["rotation_term_mean"] + comp["time_term_mean"])
               - row["graded_fitness"]) < 1e-6

    # The window resets after a write: banking nothing new and flushing
    # again must show 0 episodes in the (now-empty) new window.
    ds.flush(env)
    doc2 = json.loads(env._score_output_path.read_text())
    assert doc2["episodes"] == 0
    assert doc2["designs"] == {}


def test_reset_scoring_state_only_touches_the_given_env_ids():
    env = _fake_env(4, design_idx=[0, 0, 1, 1])
    env._rot_error[:] = torch.tensor([1.0, 1.0, 2.0, 2.0])
    ds.reset_scoring_state(env, torch.arange(4))
    ds.step_scoring_state(env)  # elapsed_steps -> 1 for all

    # Partial reset of env 0 only, with a different starting error.
    env._rot_error[0] = 3.0
    ds.reset_scoring_state(env, torch.tensor([0]))
    assert env._score_start_error[0].item() == 3.0
    assert env._score_elapsed_steps[0].item() == 0
    # envs 1-3 untouched.
    assert env._score_start_error[1].item() == 1.0
    assert env._score_elapsed_steps[1].item() == 1


def test_snapshot_goal_mode_code_is_isaaclab_under_the_repose_profile():
    """The isaaclab_repose profile has no goal curriculum; its snapshots
    report goal mode 3 (NVIDIA's random goals), not the legacy curriculum's
    unused stage."""
    env = SimpleNamespace(_repose=True)
    assert ds._goal_mode_code_safe(env) == 3.0


def test_anyrotate_banks_rotation_about_the_axis_and_time_to_terminate():
    """Under the anyrotate profile the graded fitness is AnyRotate's own two
    metrics: rotations about k (rotation_progress = the episode's rotation
    about k, rad) plus 0.25 x the fraction of the episode before termination;
    goals are still counted but not in the fitness."""
    env = SimpleNamespace(_anyrotate=True)
    assert ds._goal_mode_code_safe(env) == 4.0
    rot = torch.tensor([2 * math.pi, -0.5])
    ttt = torch.tensor([30.0, 6.0])
    goals = torch.tensor([3.0, 0.0])
    rp, rot_term, time_term, fitness = ds.anyrotate_fitness_components(goals, rot, ttt, 30.0)
    assert rp.tolist() == [2 * math.pi, -0.5]
    assert rot_term.tolist() == [1.0, 0.0]
    assert time_term.tolist() == [0.25, 0.05]
    assert [round(v, 6) for v in fitness.tolist()] == [1.25, 0.05]
