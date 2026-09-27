"""``get_curriculum_state``/``set_curriculum_state``'s shared 4-key contract,
and the generic ``extra_curriculum_state``/``restore_extra_curriculum_state``
hook an env can opt into.

The hook itself lives in ``pose_reaching_6d.reward_utils.curriculum`` (this
module); no env-specific state does -- see
``isaacsimenvs.inhand_reorient.reward_utils`` for the goal-difficulty
curriculum state that actually uses it (I24, Phase 1c).

``curriculum.py`` itself imports nothing (not even stdlib beyond
``__future__``), so it needs no Kit booted in isolation -- but reaching it
through the normal dotted path (``isaacsimenvs.pose_reaching_6d.reward_utils.
curriculum``) still executes ``reward_utils/__init__.py`` first (ordinary
Python package-import semantics), which imports ``termination.py`` ->
``reset_utils`` -> ``isaaclab.utils.math`` and needs Kit already booted.
Loaded here via ``importlib.util.spec_from_file_location`` instead, the same
work-around ``test_sharpa_physics_surface.py`` uses for the same reason. Run
with ``.venv_isaacsim/bin/python`` (for ``gymnasium``, which importing
``isaacsimenvs`` itself -- unavoidable: it is this test file's own parent
package -- needs), same as ``isaacsimenvs/inhand_reorient/tests``:
    OMNI_KIT_ACCEPT_EULA=YES PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        .venv_isaacsim/bin/python -m pytest -p no:cacheprovider \
        isaacsimenvs/pose_reaching_6d/tests/test_curriculum.py -q
No Kit boot happens either way: OMNI_KIT_ACCEPT_EULA is set only because
other tests in the same invocation may need it, not this file.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "_curriculum_under_test",
    ROOT / "isaacsimenvs/pose_reaching_6d/reward_utils/curriculum.py",
)
curriculum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curriculum)

CURRICULUM_STATE_VERSION = curriculum.CURRICULUM_STATE_VERSION
get_curriculum_state = curriculum.get_curriculum_state
set_curriculum_state = curriculum.set_curriculum_state


class _StubTermination:
    def __init__(self):
        self.resume_success_tolerance = 0.0
        self.success_tolerance = 0.4


class _StubCfg:
    def __init__(self):
        # A fresh _StubTermination per _StubCfg (not a shared class
        # attribute): test_..._resume_override_wins mutates
        # resume_success_tolerance on its own env, and that must not leak
        # into any other test's env.
        self.termination = _StubTermination()


class _PoseReachLikeEnv:
    """No ``extra_curriculum_state``/``restore_extra_curriculum_state`` --
    what a ``PoseReachEnv`` (or anything else predating the hook) looks like
    to these functions."""

    def __init__(self):
        self.cfg = _StubCfg()
        self._current_success_tolerance = 0.4
        self._frame_counter = 100
        self._last_curriculum_update = 90


def test_get_curriculum_state_is_exactly_the_old_four_keys_with_no_hook():
    env = _PoseReachLikeEnv()
    state = get_curriculum_state(env)
    assert state == {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": 0.4,
        "frame_counter": 100,
        "last_curriculum_update": 90,
    }


def test_set_curriculum_state_restores_the_four_fields_with_no_hook():
    env = _PoseReachLikeEnv()
    set_curriculum_state(env, {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": 0.2,
        "frame_counter": 500,
        "last_curriculum_update": 480,
    })
    assert env._current_success_tolerance == 0.2
    assert env._frame_counter == 500
    assert env._last_curriculum_update == 480


def test_set_curriculum_state_tolerates_none_and_missing_keys_with_no_hook():
    env = _PoseReachLikeEnv()
    set_curriculum_state(env, None)  # no checkpoint state at all: unchanged
    assert env._current_success_tolerance == 0.4
    assert env._frame_counter == 100

    set_curriculum_state(env, {"version": CURRICULUM_STATE_VERSION})  # no numeric fields
    assert env._current_success_tolerance == 0.4  # unchanged: no current_success_tolerance key
    assert env._frame_counter == 100  # unchanged: no frame_counter key


def test_set_curriculum_state_resume_override_wins_with_no_hook():
    env = _PoseReachLikeEnv()
    env.cfg.termination.resume_success_tolerance = 0.35
    set_curriculum_state(env, {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": 0.05,
        "frame_counter": 500,
        "last_curriculum_update": 480,
    })
    # The checkpoint's tolerance is overridden by the explicit resume value;
    # everything else (frame counters) still restores from the checkpoint.
    assert env._current_success_tolerance == 0.4  # unchanged: override wins, doesn't set 0.05
    assert env._frame_counter == 500


class _EnvWithExtraHook(_PoseReachLikeEnv):
    """Opts into the generic hook, mirroring ``InHandReorientEnv``'s own
    ``extra_curriculum_state``/``restore_extra_curriculum_state`` (I24)."""

    def __init__(self):
        super().__init__()
        self.goal_stage = 0

    def extra_curriculum_state(self) -> dict:
        return {"goal_stage": self.goal_stage}

    def restore_extra_curriculum_state(self, state: dict) -> None:
        self.goal_stage = int(state.get("goal_stage", self.goal_stage))


def test_get_curriculum_state_merges_in_the_extra_hook_when_present():
    env = _EnvWithExtraHook()
    env.goal_stage = 1
    state = get_curriculum_state(env)
    assert state["goal_stage"] == 1
    assert set(state) == {
        "version", "current_success_tolerance", "frame_counter",
        "last_curriculum_update", "goal_stage",
    }


def test_set_curriculum_state_calls_the_extra_hook_when_present():
    env = _EnvWithExtraHook()
    set_curriculum_state(env, {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": 0.3,
        "frame_counter": 10,
        "last_curriculum_update": 5,
        "goal_stage": 2,
    })
    assert env.goal_stage == 2
    assert env._current_success_tolerance == 0.3


def test_set_curriculum_state_extra_hook_tolerates_a_checkpoint_missing_its_key():
    env = _EnvWithExtraHook()
    env.goal_stage = 3
    # A checkpoint written before the goal curriculum existed has neither key.
    set_curriculum_state(env, {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": 0.3,
        "frame_counter": 10,
        "last_curriculum_update": 5,
    })
    assert env.goal_stage == 3  # unchanged
