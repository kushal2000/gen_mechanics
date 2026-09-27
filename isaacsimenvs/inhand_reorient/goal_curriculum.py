"""Pure-python (stdlib only, no isaaclab) goal-difficulty curriculum logic.

Kept out of ``reward_utils.py``/``reset_utils.py`` deliberately -- the same
split ``palm_calibration.py`` uses relative to ``calibrate_palm_up.py``/
``scene_utils.py`` (see that module's docstring): both of those import
``isaacsimenvs.pose_reaching_6d`` modules that bootstrap Kit on import
(``isaaclab.*`` at module scope, transitively through
``pose_reaching_6d.reward_utils``'s package ``__init__``), so nothing that
has to stay testable under plain pytest (no Kit) can live there. This module
imports nothing at all: it is duck-typed over ``env``, and the one tensor
operation it does (``.float().numel()/.mean().item()`` on
``env._prev_episode_successes``) works identically on a real torch tensor in
production or a plain torch tensor built in a CPU test (torch itself needs no
Kit -- only ``isaaclab.*`` does).

Decoupling rule (I26). A 60-minute SHARPA run let the goal curriculum
(``axis`` -> ``full``) and the tolerance curriculum (0.40 -> 0.36) both
advance in the SAME step (frame ~24227, both keyed off the same statistic,
threshold and interval); successes collapsed from 0.22 to 0.03 and never
recovered in the remaining 40 minutes. The mitigation implemented in
``update_goal_curriculum`` below: the goal curriculum only advances once the
TOLERANCE curriculum has already reached its floor
(``termination.target_success_tolerance``), evaluated against the goal
curriculum's OWN interval/threshold -- and, even once at the floor, never on
a frame where the tolerance curriculum's own bookkeeping just touched
(``env._last_curriculum_update == env._frame_counter``, which
``update_tolerance_curriculum`` sets whenever ITS eligibility check fires,
whether or not the clamped value actually moved). Call this strictly AFTER
``update_tolerance_curriculum`` each step (``env.py::_get_dones`` does), so
``env._current_success_tolerance``/``env._last_curriculum_update`` already
reflect this frame's tolerance-curriculum outcome.
"""

from __future__ import annotations

__all__ = ["GOAL_MODE_CODES", "goal_curriculum_mode", "goal_mode_code", "update_goal_curriculum"]

# Numeric encoding for TensorBoard (which has no string-scalar tags): the
# active mode's code, alongside the stage index, in
# reset_utils.log_step_metrics. "full" and "absolute" share a code (both mean
# "fully random goal", see reset_utils._sample_goal) since the curriculum
# never distinguishes them as different DIFFICULTIES, only different names
# for the same terminal stage.
GOAL_MODE_CODES: dict[str, float] = {"axis": 0.0, "delta": 1.0, "full": 2.0, "absolute": 2.0}


def goal_curriculum_mode(env) -> str:
    """Which sampling mode is active right now: the staged curriculum's
    current stage when enabled, else the static ``goal_sampling_type``."""
    cfg = env.cfg.reset
    if not cfg.goal_curriculum_enabled:
        return cfg.goal_sampling_type
    stage = min(env._goal_curriculum_stage, len(cfg.goal_curriculum_stages) - 1)
    return cfg.goal_curriculum_stages[stage]


def goal_mode_code(mode: str) -> float:
    """``GOAL_MODE_CODES[mode]``, or ``-1.0`` for an unrecognised mode (never
    happens in production -- ``_sample_goal`` would already have raised --
    but keeps this a total function for logging call sites)."""
    return GOAL_MODE_CODES.get(mode, -1.0)


def update_goal_curriculum(env) -> None:
    """Widen the goal-sampling curriculum (``reset_utils._sample_goal``'s
    modes) once completed episodes average enough goals AND the tolerance
    curriculum has settled at its floor (I26 decoupling rule, module
    docstring above). Reads ``env._frame_counter``/``env._current_success_
    tolerance``/``env._last_curriculum_update`` AFTER
    ``update_tolerance_curriculum`` has already updated them this step --
    call this right after that function, not before (see
    ``env.py::_get_dones``).
    """
    cfg = env.cfg.reset
    if not cfg.goal_curriculum_enabled or len(cfg.goal_curriculum_stages) <= 1:
        return  # nothing staged, or nothing left to advance to

    term = env.cfg.termination
    if env._current_success_tolerance > term.target_success_tolerance:
        return  # tolerance curriculum has not reached its floor yet
    if env._last_curriculum_update == env._frame_counter:
        return  # tolerance curriculum's own bookkeeping just touched THIS
        # frame (its eligibility check fired, whether or not the clamped
        # value moved) -- never advance the goal curriculum the same step
    if env._frame_counter - env._last_goal_curriculum_update < cfg.goal_curriculum_interval:
        return

    successes = env._prev_episode_successes.float()
    if successes.numel() == 0 or successes.mean().item() < cfg.goal_curriculum_success_threshold:
        return

    env._last_goal_curriculum_update = env._frame_counter
    if env._goal_curriculum_stage >= len(cfg.goal_curriculum_stages) - 1:
        return
    env._goal_curriculum_stage += 1
    stage_name = cfg.goal_curriculum_stages[env._goal_curriculum_stage]
    print(f"[inhand_reorient] goal curriculum -> stage {env._goal_curriculum_stage} "
          f"({stage_name!r}) at frame {env._frame_counter}", flush=True)
