"""CPU tests for the goal/tolerance curriculum decoupling rule (I26).

``goal_curriculum.py`` imports nothing (not even isaaclab-triggering
pose_reaching_6d modules), so it is reachable under plain pytest -- no Kit,
no ``.venv_isaacsim`` requirement beyond ``torch`` itself (which needs no
Kit; only ``isaaclab.*`` does -- see that module's docstring and
``tests/test_config_sanity.py``'s for the general rule this package follows).
"""

from __future__ import annotations

import torch

from isaacsimenvs.inhand_reorient.goal_curriculum import (
    GOAL_MODE_CODES, goal_curriculum_mode, goal_mode_code, update_goal_curriculum,
)


class _ResetCfg:
    def __init__(self, stages=("axis", "full"), enabled=True, interval=3000, threshold=3.0,
                 goal_sampling_type="absolute"):
        self.goal_curriculum_enabled = enabled
        self.goal_curriculum_stages = stages
        self.goal_curriculum_interval = interval
        self.goal_curriculum_success_threshold = threshold
        self.goal_sampling_type = goal_sampling_type


class _TerminationCfg:
    def __init__(self, target_success_tolerance=0.1):
        self.target_success_tolerance = target_success_tolerance


class _Cfg:
    def __init__(self, **kwargs):
        self.reset = _ResetCfg(**{k: v for k, v in kwargs.items() if k in (
            "stages", "enabled", "interval", "threshold", "goal_sampling_type")})
        self.termination = _TerminationCfg(
            target_success_tolerance=kwargs.get("target_success_tolerance", 0.1))


class _Env:
    """Duck-typed stand-in: only the attributes update_goal_curriculum and
    goal_curriculum_mode actually read."""

    def __init__(
        self, *, frame_counter=10_000, last_curriculum_update=6_000,
        current_success_tolerance=0.1, last_goal_curriculum_update=0,
        goal_curriculum_stage=0, successes=(5.0, 5.0, 5.0), **cfg_kwargs,
    ):
        self.cfg = _Cfg(**cfg_kwargs)
        self._frame_counter = frame_counter
        self._last_curriculum_update = last_curriculum_update
        self._current_success_tolerance = current_success_tolerance
        self._last_goal_curriculum_update = last_goal_curriculum_update
        self._goal_curriculum_stage = goal_curriculum_stage
        self._prev_episode_successes = torch.tensor(successes)


# --- goal_curriculum_mode -------------------------------------------------


def test_mode_reads_the_current_stage_when_enabled():
    env = _Env(stages=("axis", "full"), goal_curriculum_stage=0)
    assert goal_curriculum_mode(env) == "axis"
    env._goal_curriculum_stage = 1
    assert goal_curriculum_mode(env) == "full"


def test_mode_falls_back_to_static_type_when_disabled():
    env = _Env(enabled=False, goal_sampling_type="delta")
    assert goal_curriculum_mode(env) == "delta"


def test_mode_clamps_an_out_of_range_stage_index():
    env = _Env(stages=("axis", "full"), goal_curriculum_stage=99)
    assert goal_curriculum_mode(env) == "full"


def test_goal_mode_code_mapping():
    assert goal_mode_code("axis") == GOAL_MODE_CODES["axis"] == 0.0
    assert goal_mode_code("delta") == 1.0
    assert goal_mode_code("full") == goal_mode_code("absolute") == 2.0
    assert goal_mode_code("nonsense") == -1.0


# --- update_goal_curriculum: the I26 decoupling rule ----------------------


def test_no_advance_while_tolerance_curriculum_above_its_floor():
    # Interval/threshold satisfied, but the tolerance curriculum has not
    # reached target_success_tolerance yet -- must not advance.
    env = _Env(
        current_success_tolerance=0.25, target_success_tolerance=0.1,
        last_curriculum_update=1_000,  # not this frame, so that alone can't block it
        frame_counter=10_000, last_goal_curriculum_update=0,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_no_advance_on_the_same_frame_the_tolerance_curriculum_touched():
    # At the floor, but the tolerance curriculum's own bookkeeping updated
    # THIS exact frame (its eligibility check fired) -- must not also
    # advance the goal curriculum this step, even though every other
    # condition (own interval/threshold) is satisfied.
    env = _Env(
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=10_000,
        last_goal_curriculum_update=0,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_advances_once_floor_reached_own_interval_elapsed_and_threshold_met():
    env = _Env(
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,  # a stale touch, not this frame
        last_goal_curriculum_update=0, interval=3000, threshold=3.0,
        successes=(4.0, 4.0, 4.0),
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 1
    assert env._last_goal_curriculum_update == 10_000


def test_respects_its_own_interval_even_after_the_floor_is_reached():
    env = _Env(
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,
        last_goal_curriculum_update=9_000,  # only 1000 frames ago, interval is 3000
        interval=3000,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_respects_its_own_success_threshold():
    env = _Env(
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,
        last_goal_curriculum_update=0, threshold=3.0,
        successes=(1.0, 2.0, 0.0),  # mean 1.0 < 3.0
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_no_advance_with_no_completed_episodes_yet():
    env = _Env(
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,
        last_goal_curriculum_update=0, successes=(),
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_stays_at_the_last_stage_once_reached():
    env = _Env(
        stages=("axis", "full"), goal_curriculum_stage=1,
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,
        last_goal_curriculum_update=0,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 1  # unchanged, no IndexError


def test_disabled_never_advances():
    env = _Env(
        enabled=False,
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=5_000,
        last_goal_curriculum_update=0,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0


def test_single_stage_never_advances_even_when_every_other_condition_holds():
    # The E-R0 default: goal_curriculum_stages=("axis",). Nothing to advance
    # to, so this must be a cheap no-op regardless of tolerance/frame state.
    env = _Env(
        stages=("axis",),
        current_success_tolerance=0.1, target_success_tolerance=0.1,
        frame_counter=10_000, last_curriculum_update=10_000,  # even same-frame
        last_goal_curriculum_update=0,
    )
    update_goal_curriculum(env)
    assert env._goal_curriculum_stage == 0
