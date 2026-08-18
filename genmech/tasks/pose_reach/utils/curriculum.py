"""Curriculum state that has to survive a checkpoint.

Free functions over a duck-typed env, in a module that imports nothing from
Isaac Sim, so the logic can be tested without booting Kit. ``PoseReachEnv``
delegates to these; the rl_games adapter ferries the dict they produce.

WHY THIS EXISTS. ``_current_success_tolerance`` lives on the env, and rl_games
persists env state only through ``vec_env.get_env_state()`` -- a hook whose own
docstring says it is for "stateful training sessions, i.e. with adaptive
curriculums". Neither Isaac Lab wrapper overrode it, so every checkpoint this
repo wrote carried ``env_state=None`` and every resume silently restarted the
curriculum at its beginning.

That is worse than a difficulty reset: the tolerance also scales the keypoint
reward (``obs_utils``), so a continuation trained an EASIER task on a DIFFERENT
reward scale than the checkpoint was written under, and the reward curve jumped
in a way that read as the fine-tune working unusually well. Nothing raised.
"""

from __future__ import annotations

CURRICULUM_STATE_VERSION = 1


def _resume_override(env) -> float:
    """The explicit ``resume_success_tolerance``, or 0.0 when unset.

    0.0 is the sentinel rather than None because Isaac Lab's
    ``update_class_from_dict`` type-checks a CLI override against the current
    value's type -- a None-defaulted field cannot be overridden from hydra at
    all, which cost one 20-second job to discover.
    """
    return float(getattr(env.cfg.termination, "resume_success_tolerance", 0.0) or 0.0)


def initial_success_tolerance(env) -> float:
    """Where the curriculum starts for this run: the resume point, or the start."""
    override = _resume_override(env)
    return float(override if override > 0.0 else env.cfg.termination.success_tolerance)


def get_curriculum_state(env) -> dict:
    """The curriculum state a resume needs, as plain JSON-able data."""
    return {
        "version": CURRICULUM_STATE_VERSION,
        "current_success_tolerance": float(env._current_success_tolerance),
        "frame_counter": int(env._frame_counter),
        "last_curriculum_update": int(env._last_curriculum_update),
    }


def set_curriculum_state(env, state: dict | None) -> None:
    """Restore what :func:`get_curriculum_state` saved.

    Tolerant of None and of missing keys: checkpoints written before this
    existed carry ``env_state=None`` and must still load. An explicit
    ``resume_success_tolerance`` WINS over the checkpoint -- it is a direct
    instruction, and it is the only way to place the curriculum when continuing
    one of those older checkpoints.
    """
    if not state:
        print("[curriculum] checkpoint carries no curriculum state; keeping the "
              "configured tolerance", flush=True)
        return

    version = state.get("version")
    if version != CURRICULUM_STATE_VERSION:
        print(f"[curriculum] WARNING: checkpoint curriculum state version "
              f"{version} != {CURRICULUM_STATE_VERSION}; restoring what matches",
              flush=True)

    env._frame_counter = int(state.get("frame_counter", env._frame_counter))
    env._last_curriculum_update = int(
        state.get("last_curriculum_update", env._last_curriculum_update))

    ckpt_tol = state.get("current_success_tolerance")
    override = _resume_override(env)
    if override > 0.0:
        print(f"[curriculum] checkpoint tolerance {ckpt_tol} overridden by "
              f"resume_success_tolerance={override}", flush=True)
        return
    if ckpt_tol is not None:
        env._current_success_tolerance = float(ckpt_tol)
        print(f"[curriculum] restored success tolerance "
              f"{env._current_success_tolerance:.6f} from the checkpoint "
              f"(frame {env._frame_counter})", flush=True)


__all__ = ["CURRICULUM_STATE_VERSION", "get_curriculum_state",
           "initial_success_tolerance", "set_curriculum_state"]
