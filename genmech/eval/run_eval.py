"""Evaluate one policy under one held-out condition.

Runs a batched rollout: N envs each chain live-sampled goals until they fail or
complete the required run of consecutive successes, giving N independent trials
from a single Kit boot. simtoolreal's DexToolBench eval used ``num_envs=1`` and
one process per case, which is fine for 24 real-object trajectories but far too
slow for a multi-axis sweep.

**On the headline metric.** Goals are sampled live, matching training, so there
is no trajectory to complete a percentage of. What is measured is how many
consecutive goals a policy chains before falling or timing out. That outcome is
strongly bimodal -- a policy either fails to establish a grasp and scores zero,
or grasps and chains the lot -- so the mean describes almost no individual
episode. ``full_completion_rate`` and ``zero_goal_rate`` are the honest summary
and are reported alongside it.

One process per condition, because Kit cannot be torn down and re-created
in-process. ``launch_sweep.py`` fans these out over a SLURM array.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m genmech.eval.run_eval \\
        --robot_spec sharpa_iiwa14 \\
        --checkpoint pretrained_policy/model.pth \\
        --policy_config pretrained_policy/config.yaml \\
        --condition nominal/nominal \\
        --out results/smoke/sharpa_iiwa14/nominal/nominal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

CONTROL_HZ = 60.0


def _set_by_path(cfg, dotted: str, value) -> None:
    """Apply one dotted-path override to a configclass instance."""
    obj = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(obj, part):
            raise KeyError(f"no config section {part!r} while setting {dotted!r}")
        obj = getattr(obj, part)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"no config field {dotted!r}")
    setattr(obj, leaf, value)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--policy_config", required=True)
    parser.add_argument("--condition", required=True, help="axis/name, e.g. dr/hard")
    parser.add_argument("--suite", default="full")
    parser.add_argument("--out", required=True, help="Directory for result.json")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Hard cap on rollout steps. 0 derives it from the "
                             "goal count, which is the right behaviour: the cap "
                             "exists only to bound a pathological run, and a "
                             "hand-picked value is either too small (silently "
                             "truncating) or needlessly large (one straggler "
                             "then drags the whole batch to the ceiling).")
    parser.add_argument("--rl_device", default="cuda")
    parser.add_argument(
        "--override", action="append", default=[],
        help="Extra cfg override as dotted.path=JSON, repeatable. Applied AFTER "
             "the condition, so it wins. Use to score a checkpoint under its own "
             "training protocol rather than the suite's, e.g. "
             "--override termination.success_steps=10 "
             "--override 'reset.target_volume_mins=[-0.35,-0.1,0.68]'. Recorded "
             "in result.json so a re-protocoled run is never mistaken for a "
             "suite-standard one.",
    )
    parser.add_argument(
        "--sapg_expl_coef", type=float, default=50.0,
        help="Trailing exploration-coefficient obs column for SAPG checkpoints. "
             "Negative disables it (plain PPO). Getting this wrong is silent.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    # Resolve the condition before booting Kit, so a bad name fails in a second.
    from genmech.eval.suites import get_suite, resolve_overrides

    conditions = {c.id: c for c in get_suite(args.suite)}
    if args.condition not in conditions:
        raise SystemExit(
            f"unknown condition {args.condition!r} in suite {args.suite!r}.\n"
            f"Available: {sorted(conditions)}"
        )
    condition = conditions[args.condition]
    overrides = resolve_overrides(condition)

    extra: dict = {}
    for item in args.override:
        key, _, raw = item.partition("=")
        if not raw:
            raise SystemExit(f"--override {item!r} is not dotted.path=value")
        try:
            extra[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            extra[key.strip()] = raw  # bare string
    if extra:
        # Applied after the condition so a deliberate protocol change wins over
        # the suite default.
        overrides = {**overrides, **extra}
        print(f"[eval] CLI overrides (non-standard protocol): {extra}")

    from genmech.robots import get_robot_spec
    spec = get_robot_spec(args.robot_spec)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    print(f"[eval] {spec.name} x {condition.id}  (seed {condition.seed})")
    print(f"[eval] {condition.note}")

    t_boot = time.perf_counter()
    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import genmech.tasks  # noqa: F401  registers GenMech-PoseReach-Direct-v0
    from genmech.eval.rl_player import RlPlayer
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg
    from genmech.utils.paths import resolve as resolve_repo_path

    torch.manual_seed(condition.seed)
    torch.cuda.manual_seed_all(condition.seed)
    np.random.seed(condition.seed)

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = args.robot_spec
    for key, value in overrides.items():
        if key == "reset.fixed_trajectory_file" and value:
            value = str(resolve_repo_path(value))
        _set_by_path(cfg, key, value)

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None

    n_act = int(inner.cfg.action_space)
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=n_act,
        config_path=args.policy_config,
        checkpoint_path=str(checkpoint),
        device=args.rl_device,
        sapg_expl_coef=(args.sapg_expl_coef if args.sapg_expl_coef >= 0 else None),
        num_envs=args.num_envs,
    )
    player.player.init_rnn()

    n_goals = int(cfg.termination.max_consecutive_successes)
    N = args.num_envs
    dev = inner.device

    # ~5 s per goal at 60 Hz. 200/goal truncated 2 of 128 envs once the dwell
    # requirement was restored (success_steps=10 makes each goal slower), and a
    # truncated env reports a lower bound, so the cap is set above where that
    # was observed. Still far below the 600-steps-per-goal worst case a run of
    # pure per-goal timeouts would need. Envs that do hit it are counted as
    # `unfinished`, so truncation is never silent.
    max_steps = args.max_steps or (300 * n_goals)
    print(f"[eval] {n_goals} goals per episode, step cap {max_steps}"
          f"{' (derived)' if not args.max_steps else ' (explicit)'}")

    obs, _ = env.reset()
    # Match the reference driver's timing: one physics tick before the first
    # policy action (the first step doubles as the reset trigger).
    obs, _, _, _, _ = env.step(torch.zeros((N, n_act), device=dev))

    # Per-env accumulators. Each env runs exactly one trajectory; once it
    # finishes we stop counting it, so envs that terminate early do not get
    # scored on a second, freshly-reset trajectory.
    done_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    goals_reached = torch.zeros(N, dtype=torch.long, device=dev)
    episode_steps = torch.zeros(N, dtype=torch.long, device=dev)
    first_goal_step = torch.full((N,), -1, dtype=torch.long, device=dev)
    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    reward_sum = torch.zeros(N, device=dev)
    term_fall = torch.zeros(N, dtype=torch.bool, device=dev)
    term_hand_far = torch.zeros(N, dtype=torch.bool, device=dev)
    term_timeout = torch.zeros(N, dtype=torch.bool, device=dev)
    term_complete = torch.zeros(N, dtype=torch.bool, device=dev)

    t_roll = time.perf_counter()
    for step in range(max_steps):
        policy_obs = obs["policy"].to(args.rl_device)
        action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, rew, terminated, truncated, _ = env.step(action.to(dev))

        active = ~done_mask
        if not bool(active.any()):
            break

        episode_steps = torch.where(active, episode_steps + 1, episode_steps)
        reward_sum = torch.where(active, reward_sum + rew, reward_sum)
        ever_lifted |= active & inner._lifted_object.bool()

        # _successes counts goals reached within the current episode; it is
        # zeroed by the env on reset, so read it before folding in `done`.
        live = inner._successes.clone()
        newly_scored = active & (live > goals_reached)
        goals_reached = torch.where(newly_scored, live, goals_reached)
        first_goal_step = torch.where(
            newly_scored & (first_goal_step < 0),
            torch.full_like(first_goal_step, step),
            first_goal_step,
        )

        done_now = active & (terminated | truncated)
        if bool(done_now.any()):
            # step() already reset those envs, so the episode's final count
            # lives in _prev_episode_successes.
            final = inner._prev_episode_successes
            goals_reached = torch.where(
                done_now & (final > goals_reached), final, goals_reached
            )
            # Attribute the termination. fall / hand_far are read off the
            # env's own predicates; the rest is trajectory completion vs the
            # per-goal timeout.
            complete = done_now & (goals_reached >= n_goals)
            term_complete |= complete
            fell = done_now & ~complete & terminated
            term_fall |= fell
            term_timeout |= done_now & ~complete & truncated
            done_mask |= done_now

        if (step + 1) % 500 == 0:
            print(f"[eval] step {step + 1}: {int(done_mask.sum())}/{N} envs finished, "
                  f"mean goals {float(goals_reached.float().mean()):.2f}/{n_goals}",
                  flush=True)

    roll_sec = time.perf_counter() - t_roll
    unfinished = int((~done_mask).sum())
    if unfinished:
        print(f"[eval] WARNING: {unfinished}/{N} envs hit the {max_steps}-step "
              f"cap without terminating; their scores are lower bounds.")

    # Fraction of the required consecutive-goal chain the policy completed.
    #
    # NOT "percent of a trajectory": goals are sampled live, so there is no
    # trajectory. An episode ends on n_goals consecutive successes, a fall, or a
    # per-goal timeout, making this a normalised CHAIN LENGTH.
    #
    # The distribution is strongly bimodal -- policies either fail to establish a
    # grasp and score 0, or grasp and chain the lot -- so the mean describes
    # almost no individual episode and must be read alongside complete/zero
    # rates, which is why both are reported.
    goals_np = goals_reached.cpu().numpy().astype(float)
    goal_pct = (goals_reached.float() / n_goals * 100.0).cpu().numpy()
    steps_np = episode_steps.cpu().numpy()
    fg = first_goal_step.cpu().numpy()
    reached_any = fg >= 0

    def _mean(x) -> float:
        return float(np.mean(x)) if len(x) else float("nan")

    def _sem(x) -> float:
        return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    result = {
        "schema_version": 1,
        "robot_spec": spec.name,
        "arm": spec.arm_name,
        "hand": spec.hand_name,
        "axis": condition.axis,
        "condition": condition.name,
        "condition_id": condition.id,
        "suite": args.suite,
        "seed": condition.seed,
        "note": condition.note,
        "num_envs": N,
        "n_goals": n_goals,
        "policy": {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "config": str(Path(args.policy_config).resolve()),
            "sapg_expl_coef": args.sapg_expl_coef if args.sapg_expl_coef >= 0 else None,
        },
        # Headline metrics. goal_pct is the primary; reward is diagnostic only
        # (docs/methodology.md §4).
        # HEADLINE: expected number of goals reached per episode.
        #
        # Uses every episode's full information instead of thresholding it away
        # -- an env reaching 9 goals and one reaching 0 are identical under a
        # completion rate, which discards real signal. It also removes a
        # researcher degree of freedom: with a threshold metric, n_goals has to
        # be tuned so the reference policy lands mid-range, and that tuning moves
        # the headline. Here n_goals is just a ceiling.
        #
        # The distribution is bimodal, but that is a fact about its shape, not a
        # defect in the mean: the expected number of goals is the quantity of
        # interest regardless of how the mass is arranged.
        "goals_reached_mean": float(np.mean(goals_np)),
        "goals_reached_sem": _sem(goals_np),
        "goals_reached_max": n_goals,
        # Saturation check: if many episodes hit the ceiling the mean compresses
        # and n_goals should be raised.
        "ceiling_rate": float(np.mean(goals_np >= n_goals)),
        "goal_pct_mean": _mean(goal_pct),      # same quantity, percent of ceiling
        "goal_pct_sem": _sem(goal_pct),
        # The honest summary of a bimodal outcome.
        "full_completion_rate": float(np.mean(goal_pct >= 100.0)),
        "zero_goal_rate": float(np.mean(goal_pct <= 0.0)),
        "bimodality": float(np.mean((goal_pct <= 0.0) | (goal_pct >= 100.0))),
        "any_goal_rate": float(np.mean(reached_any)),
        "time_to_first_goal_sec": _mean(fg[reached_any] / CONTROL_HZ),
        "mean_episode_sec": _mean(steps_np / CONTROL_HZ),
        "lift_rate": float(ever_lifted.float().mean()),
        "reward_mean": float((reward_sum / episode_steps.clamp(min=1)).mean()),
        "termination_counts": {
            "trajectory_complete": int(term_complete.sum()),
            "fall_or_hand_far": int(term_fall.sum()),
            "timeout": int(term_timeout.sum()),
            "unfinished": unfinished,
        },
        "per_env": {
            "goal_pct": goal_pct.tolist(),
            "episode_steps": steps_np.tolist(),
            "first_goal_step": fg.tolist(),
        },
        "overrides": {k: (list(v) if isinstance(v, tuple) else v)
                      for k, v in overrides.items()},
        # Non-empty means this run did NOT use the suite's standard protocol and
        # must not be pooled with runs that did.
        "cli_overrides": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in extra.items()},
        "wall_clock": {"boot_sec": t_roll - t_boot, "rollout_sec": roll_sec},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"\n[eval] {spec.name} x {condition.id}: "
        f"goals={result['goals_reached_mean']:.2f} +/- {result['goals_reached_sem']:.2f}"
        f" of {n_goals}   "
        f"(complete={result['full_completion_rate']:.0%} zero={result['zero_goal_rate']:.0%})  "
        f"lift={result['lift_rate']:.0%}  ({roll_sec:.0f}s)"
    )
    if result["ceiling_rate"] > 0.4:
        print(f"[eval] NOTE: {result['ceiling_rate']:.0%} of episodes hit the "
              f"{n_goals}-goal ceiling; the mean is compressed. Raise n_goals.")
    print(f"[eval] wrote {out_dir / 'result.json'}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
