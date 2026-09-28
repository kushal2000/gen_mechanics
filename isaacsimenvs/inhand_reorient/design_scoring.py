"""Per-design graded scoring, banked INSIDE the env at episode end (plan
revision 2026-09-27, step 1: "Score each design with partial credit: time
held, rotation achieved before a drop, and goals per unit time, keyed by
design index at done time").

Supersedes ``coevolution/design_rewards.py``'s ``DesignRewardWrapper`` for
this task -- that wrapper's bugs (opus-review-phase2.md item 3:
``scene_record.robot_design_index`` read by attribute off a dict, and
``extras["successes"]`` stale by one episode) both stem from banking
OUTSIDE the env, one step removed from the state it needs. Banking here
instead, with direct access to ``env._successes``/``env._rot_error`` at the
exact moment of ``_get_rewards()`` (after ``compute_terminations`` has
updated ``_successes`` for this step, before ``_reset_idx`` clears
anything), avoids both: there is no dict/attribute mismatch to get wrong,
and no off-by-one episode.

Kit-free (torch + stdlib only, no ``isaaclab``/``hydra`` at module scope --
``hydra`` is imported lazily inside ``_resolve_output_path``) so the
banking/fitness arithmetic is directly CPU-testable under plain pytest,
same discipline as ``drop_detection.py``/``goal_curriculum.py``.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch

__all__ = [
    "allocate_scoring_buffers", "reset_scoring_state", "step_scoring_state",
    "bank_done_episodes", "maybe_write", "flush", "flush_all", "read_snapshot",
    "graded_fitness_components", "GOAL_WEIGHT", "ROTATION_WEIGHT", "TIME_WEIGHT",
]

# graded_fitness, per episode:
#     goals + ROTATION_WEIGHT * clip(rotation_progress / pi, 0, 1)
#           + TIME_WEIGHT * min(time_held_s / episode_max_s, 1)
# `goals` (an integer count, usually 0-3) dominates whenever any goal is
# reached at all; the two partial-credit terms (rotation progress towards a
# goal that was never reached, and time survived without dropping) are
# scaled to top out at 0.5/0.25 respectively so a design that reaches zero
# goals but visibly makes progress or holds the object still scores above a
# design that drops immediately with no progress -- exactly the "partial
# credit" the plan revision asks for (E-R0's finding: a success-only
# fitness scores nearly every random grammar hand 0, leaving evolution
# nothing to climb).
GOAL_WEIGHT = 1.0
ROTATION_WEIGHT = 0.5
TIME_WEIGHT = 0.25

EPISODE_MAX_S_FALLBACK = 10.0
WRITE_EVERY_N_STEPS = int(os.environ.get("GENMECH_DESIGN_SCORE_WRITE_EVERY", "200"))
"""Cadence for `maybe_write`'s periodic flush -- matches the order of
magnitude of `design_rewards.py`'s own `flush_every=50` steps default,
loosened since this module's per-write JSON is larger (per-design
components, not just sum/count).

Overridable via `GENMECH_DESIGN_SCORE_WRITE_EVERY` -- needed because
`train.py`'s own exit path uses `os._exit(0)` (Kit's shutdown hangs
otherwise), which runs no `atexit` handlers, so `flush`'s own
`atexit.register` (this module's only OTHER path to disk) never fires
there. `train.py` is out of scope to edit (this branch's edit rule), so
there is no hook to call `flush` explicitly the way `train.py` calls
`design_rewards.flush_all()` -- the periodic write is the only mechanism
available; a short smoke (a few hundred steps or fewer) needs a smaller
value to actually produce a file at all."""


def _design_idx(env) -> torch.Tensor:
    """`(num_envs,)` long: which design each env holds. All-zeros (one
    virtual "design 0") on the single-hand path, per Part C's own
    requirement ("Also bank in the single-hand path as design 0")."""
    record = getattr(env, "scene_record", None)
    if record is not None and "design_idx" in record:
        return record["design_idx"].to(dtype=torch.long)
    return torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def _n_designs(env) -> int:
    tables = getattr(env, "hand_tables", None)
    return int(tables.n_designs) if tables is not None else 1


def _design_sha256(env, d: int) -> str:
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return ""
    return getattr(tables.designs[d], "sha256", "") or ""


def _design_source(env, d: int) -> str:
    tables = getattr(env, "hand_tables", None)
    if tables is None:
        return "single_hand"
    return tables.designs[d].source


def graded_fitness_components(
    goals: torch.Tensor, rotation_progress: torch.Tensor, time_held_s: torch.Tensor, episode_max_s: float,
):
    """`(rotation_term, time_term, fitness)`, all `(k,)` -- PER-EPISODE (not
    yet averaged over episodes). See the module docstring's formula.
    `goals`/`rotation_progress`/`time_held_s` are the raw per-episode
    quantities; `rotation_progress` may be negative (error got WORSE than
    at episode start, e.g. an immediate drop) and is clamped to 0 here, so
    a design is never penalized below its `goals`/`time_held` credit for
    that."""
    rotation_term = ROTATION_WEIGHT * torch.clamp(rotation_progress, min=0.0) / math.pi
    rotation_term = torch.clamp(rotation_term, max=ROTATION_WEIGHT)
    time_term = TIME_WEIGHT * torch.clamp(time_held_s / max(float(episode_max_s), 1e-6), 0.0, 1.0)
    fitness = GOAL_WEIGHT * goals + rotation_term + time_term
    return rotation_term, time_term, fitness


def _resolve_output_path(env) -> Optional[Path]:
    """The hydra run dir, if this process is running under
    `coevolution/train.py`'s `@hydra_task_config_with_yaml`-decorated
    `main()` (this env is constructed INSIDE that same call stack, so
    `HydraConfig.get()` is already populated by the time `__init__` runs --
    same singleton `train.py` itself reads for `DesignRewardWrapper`'s own
    `output_path`, just read from a different point in the same process
    instead of threaded through as an argument, since this module cannot
    edit `train.py`). `None` (write disabled, just log) when there is no
    Hydra context -- an ad-hoc script, a diagnostic, or a CPU test."""
    try:
        from hydra.core.hydra_config import HydraConfig

        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # noqa: BLE001 -- no Hydra context; not fatal, just disables the periodic write
        return None

    rank = int(os.environ.get("RANK", "0"))
    return run_dir / f"per_design_scores_rank{rank}.json"


def allocate_scoring_buffers(env) -> None:
    """Run once, after the scene is up (mirrors
    `reset_utils.allocate_state_buffers`, called right after it)."""
    n, device = env.num_envs, env.device
    env._score_start_error = torch.zeros(n, device=device)
    env._score_min_error = torch.zeros(n, device=device)
    env._score_elapsed_steps = torch.zeros(n, device=device, dtype=torch.long)
    env._score_return_running = torch.zeros(n, device=device)

    n_designs = _n_designs(env)
    env._score_design_idx = _design_idx(env)
    env._score_n_designs = n_designs
    env._score_episodes = torch.zeros(n_designs, device=device)
    env._score_goals_sum = torch.zeros(n_designs, device=device)
    env._score_succeeded = torch.zeros(n_designs, device=device)  # episodes with >= 1 goal
    env._score_time_held_sum = torch.zeros(n_designs, device=device)
    env._score_rotation_progress_sum = torch.zeros(n_designs, device=device)
    env._score_return_sum = torch.zeros(n_designs, device=device)
    env._score_rotation_term_sum = torch.zeros(n_designs, device=device)
    env._score_time_term_sum = torch.zeros(n_designs, device=device)
    env._score_fitness_sum = torch.zeros(n_designs, device=device)
    # Episodes ended by nan_guard's "nonfinite" termination (a physics
    # blow-up), per design: this window's count and the run's running total
    # (never reset, so the last write carries everything up to it).
    env._score_nonfinite_sum = torch.zeros(n_designs, device=device)
    env._score_nonfinite_total = torch.zeros(n_designs, device=device)

    env._score_total_steps = 0
    env._score_window_steps = 0
    env._score_t0 = time.time()
    env._score_window_t0 = time.time()
    env._score_output_path = _resolve_output_path(env)
    if env._score_output_path is not None:
        atexit.register(flush, env)
    print(f"[design_scoring] banking per-design graded scores for {n_designs} design(s), "
          f"{n} envs -> {env._score_output_path if env._score_output_path is not None else '(no Hydra context: write disabled)'}",
          flush=True)


def reset_scoring_state(env, env_ids: torch.Tensor) -> None:
    """Called from `reset_utils.reset_env_state`, AFTER
    `update_palm_frame_geometry` has refreshed `env._rot_error` for the
    just-written goal/object state -- `env._rot_error[env_ids]` is this
    episode's STARTING error, per Part C's own definition of
    `rotation_progress`."""
    env._score_start_error[env_ids] = env._rot_error[env_ids]
    env._score_min_error[env_ids] = env._rot_error[env_ids]
    env._score_elapsed_steps[env_ids] = 0
    env._score_return_running[env_ids] = 0.0
    # design_idx never changes mid-run today (one population, loaded once),
    # but refreshing here is cheap and keeps this correct if that changes.
    env._score_design_idx = _design_idx(env)


def step_scoring_state(env) -> None:
    """Called once per step, from `_get_dones`, AFTER
    `compute_intermediate_values` has refreshed `env._rot_error` and BEFORE
    `compute_terminations` -- so a DONE step's own error still counts
    towards `_score_min_error` before it is read in `bank_done_episodes`."""
    env._score_elapsed_steps += 1
    env._score_min_error = torch.minimum(env._score_min_error, env._rot_error)
    env._score_total_steps += 1
    env._score_window_steps += 1


def bank_done_episodes(env, reward: torch.Tensor) -> None:
    """Called from `_get_rewards`, AFTER `compute_rewards` (so `reward` is
    this step's FULL, final reward for an episode ending this step) and
    AFTER `compute_terminations` already ran (inside `_get_dones`, earlier
    this same env.step() call) -- so `env._successes`/`env._termination_
    reasons` already reflect this step, and `_reset_idx` has not yet run.
    Uses `env._termination_reasons` (set by `compute_terminations`) rather
    than a separate `(terminated, truncated)` argument, since that dict IS
    the done mask, already computed once this step."""
    env._score_return_running += reward
    reasons = getattr(env, "_termination_reasons", None)
    if not reasons:
        return
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for mask in reasons.values():
        done = done | mask
    ids = done.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
        return

    step_dt = float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))
    episode_max_s = float(getattr(env.cfg, "episode_length_s", EPISODE_MAX_S_FALLBACK))

    d = env._score_design_idx[ids]
    goals = env._successes[ids].float()
    rotation_progress = env._score_start_error[ids] - env._score_min_error[ids]
    time_held_s = env._score_elapsed_steps[ids].float() * step_dt
    ep_return = env._score_return_running[ids]

    rot_term, time_term, fitness = graded_fitness_components(goals, rotation_progress, time_held_s, episode_max_s)
    rotation_progress_clamped = torch.clamp(rotation_progress, min=0.0)

    ones = torch.ones_like(goals)
    env._score_episodes.index_add_(0, d, ones)
    env._score_goals_sum.index_add_(0, d, goals)
    env._score_succeeded.index_add_(0, d, (goals > 0).float())
    env._score_time_held_sum.index_add_(0, d, time_held_s)
    env._score_rotation_progress_sum.index_add_(0, d, rotation_progress_clamped)
    env._score_return_sum.index_add_(0, d, ep_return)
    env._score_rotation_term_sum.index_add_(0, d, rot_term)
    env._score_time_term_sum.index_add_(0, d, time_term)
    env._score_fitness_sum.index_add_(0, d, fitness)
    nonfinite = reasons.get("nonfinite")
    if nonfinite is not None:
        nf = nonfinite[ids].float()
        env._score_nonfinite_sum.index_add_(0, d, nf)
        env._score_nonfinite_total.index_add_(0, d, nf)

    if env._score_output_path is not None and env._score_window_steps >= WRITE_EVERY_N_STEPS:
        maybe_write(env)


def _snapshot_payload(env) -> dict:
    n_designs = env._score_n_designs
    episodes = env._score_episodes.cpu()
    goals_sum = env._score_goals_sum.cpu()
    succeeded = env._score_succeeded.cpu()
    time_held_sum = env._score_time_held_sum.cpu()
    rotation_progress_sum = env._score_rotation_progress_sum.cpu()
    return_sum = env._score_return_sum.cpu()
    rotation_term_sum = env._score_rotation_term_sum.cpu()
    time_term_sum = env._score_time_term_sum.cpu()
    fitness_sum = env._score_fitness_sum.cpu()
    nonfinite_sum = env._score_nonfinite_sum.cpu()
    nonfinite_total = env._score_nonfinite_total.cpu()
    design_idx = env._score_design_idx.cpu()

    tables = getattr(env, "hand_tables", None)
    designs_payload = {}
    for i in range(n_designs):
        n_ep = float(episodes[i])
        if n_ep <= 0:
            continue
        goals_per_episode = float(goals_sum[i]) / n_ep
        time_held_mean_s = float(time_held_sum[i]) / n_ep
        time_held_sum_s = float(time_held_sum[i])
        designs_payload[str(i)] = {
            "source": _design_source(env, i),
            "design_sha256": _design_sha256(env, i),
            "envs_per_design": int((design_idx == i).sum()),
            "episodes": int(n_ep),
            "goals_sum": float(goals_sum[i]),
            "goals_per_episode": goals_per_episode,
            "succeeded": int(succeeded[i]),
            "success_rate": float(succeeded[i]) / n_ep,
            "time_held_sum_s": time_held_sum_s,
            "time_held_mean_s": time_held_mean_s,
            "rotation_progress_sum_rad": float(rotation_progress_sum[i]),
            "rotation_progress_mean_rad": float(rotation_progress_sum[i]) / n_ep,
            "goals_per_second_held": (float(goals_sum[i]) / time_held_sum_s) if time_held_sum_s > 0 else 0.0,
            # design_rewards.py-compatible names, kept as a superset:
            "return_sum": float(return_sum[i]),
            "return_mean": float(return_sum[i]) / n_ep,
            "graded_fitness": float(fitness_sum[i]) / n_ep,
            "graded_fitness_components": {
                "goal_term_mean": GOAL_WEIGHT * goals_per_episode,
                "rotation_term_mean": float(rotation_term_sum[i]) / n_ep,
                "time_term_mean": float(time_term_sum[i]) / n_ep,
            },
            "nonfinite_resets": int(nonfinite_sum[i]),
            "nonfinite_resets_total": int(nonfinite_total[i]),
        }

    return {
        "rank": int(os.environ.get("RANK", "0")),
        "steps": int(env._score_total_steps),
        "window_steps": int(env._score_window_steps),
        "success_tolerance": float(getattr(env, "_current_success_tolerance", float("nan"))),
        "goal_curriculum_stage": int(getattr(env, "_goal_curriculum_stage", -1)),
        "goal_mode_code": _goal_mode_code_safe(env),
        "elapsed_s": round(time.time() - env._score_t0, 1),
        "window_elapsed_s": round(time.time() - env._score_window_t0, 1),
        "n_designs": n_designs,
        "episodes": int(episodes.sum().item()),
        "nonfinite_resets": int(nonfinite_sum.sum().item()),
        "nonfinite_resets_total": int(nonfinite_total.sum().item()),
        "graded_fitness_formula": (
            f"goals + {ROTATION_WEIGHT} * clip(rotation_progress / pi, 0, 1) "
            f"+ {TIME_WEIGHT} * min(time_held_s / episode_max_s, 1), averaged over this window's episodes"
        ),
        "designs": designs_payload,
    }


def _goal_mode_code_safe(env) -> int:
    try:
        from .goal_curriculum import goal_curriculum_mode, goal_mode_code

        return goal_mode_code(goal_curriculum_mode(env))
    except Exception:  # noqa: BLE001
        return -1


def maybe_write(env) -> None:
    """Write the current window's snapshot, then reset the window's
    accumulators (review item 3's lesson generalised: a design's fitness
    should reflect ITS CURRENT policy/curriculum, not a running average
    blended across curriculum changes over the whole run -- "tolerance and
    goal stage for each write window" only means something if each window
    is its own, non-overlapping slice)."""
    if env._score_output_path is None:
        return
    payload = _snapshot_payload(env)
    tmp = env._score_output_path.with_suffix(".tmp")
    env._score_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(env._score_output_path)  # atomic: a reader never sees half a file

    env._score_episodes.zero_()
    env._score_goals_sum.zero_()
    env._score_succeeded.zero_()
    env._score_time_held_sum.zero_()
    env._score_rotation_progress_sum.zero_()
    env._score_return_sum.zero_()
    env._score_rotation_term_sum.zero_()
    env._score_time_term_sum.zero_()
    env._score_fitness_sum.zero_()
    env._score_nonfinite_sum.zero_()
    env._score_window_steps = 0
    env._score_window_t0 = time.time()


def read_snapshot(env) -> dict:
    """Public wrapper around the current window's snapshot -- for a caller
    (e.g. `evaluate_population.py`) that wants the accumulated per-design
    tallies as a dict WITHOUT writing/resetting anything (unlike
    `maybe_write`/`flush`)."""
    return _snapshot_payload(env)


def flush(env) -> None:
    """Write whatever is left in the current window, WITHOUT resetting it
    (called from `atexit` -- nothing runs after this)."""
    if env._score_output_path is None:
        return
    payload = _snapshot_payload(env)
    tmp = env._score_output_path.with_suffix(".tmp")
    env._score_output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(env._score_output_path)


def flush_all() -> None:
    """No-op placeholder for symmetry with `coevolution/design_rewards.
    flush_all` -- this module's `flush(env)` is registered directly via
    `atexit.register(flush, env)` per env instance (there is only ever one
    env per process here), so there is no separate registry to walk."""
    return None
