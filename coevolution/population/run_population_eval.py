"""Score every embodiment in a population on the object it trained against.

One Kit boot, one articulation view, ``num_envs`` designs stepping together.
Environment ``i`` holds design ``i`` and object ``i % pool_size``, which is
exactly the pairing training used -- so reproducing "what each design was
trained on" needs no lookup table, only the same asset config. The assignment is
verified against PhysX at startup by ``_verify_robot_design_assignment``.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m coevolution.population.run_population_eval \\
        --run_dir train_dir/.../mec_population24k_seed0_2026-08-17_15-13-28 \\
        --checkpoint train_dir/.../nn/0_pose_reach_sapg.pth \\
        --episodes 10 --out results/pop_eval

``--policy_config`` is optional: the run's own ``agent`` block IS the rl_games
config RlPlayer wants under the key ``train``, so it is synthesised from
``--run_dir`` unless given.

**Why each design needs more than one episode.** At the k = n operating point a
design gets a single environment, so a one-episode score carries the full
variance of one goal sequence. Worse, the object is *fixed* per environment for
the whole run, so a design's score is confounded with its one object draw --
designs sharing a pool index are the only fair direct comparison
(``dump_assignment.py`` emits those groups). Averaging over episodes removes the
goal-sequence noise; it does not remove the object confound, and nothing here
can. Read the per-object-type breakdown before ranking designs against each
other.

**On the protocol.** Assets, actuation and observations come from the training
run's own config, so the embodiment and its object are exactly what trained.
Goals and termination come from the frozen eval protocol in
``coevolution/eval/suites.py`` (deterministic-ish reset, success_steps=10,
eval_success_tolerance=0.01), so numbers are comparable with the rest of
``results/``. Training's own ``max_consecutive_successes`` (50) is NOT used --
``--goals_per_episode`` sets it, default 10, matching the suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

CONTROL_HZ = 60.0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_dir", required=True,
                   help="Training run whose asset/actuation config to reproduce")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--policy_config", default=None,
                   help="YAML with a top-level 'train:' key. Default: synthesised "
                        "from the run's own agent block, which is exactly that block "
                        "under a different name")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--episodes", type=int, default=10,
                   help="Episodes to average per design (default 10)")
    p.add_argument("--num_envs", type=int, default=0,
                   help="0 = the run's population count (one env per design)")
    p.add_argument("--population_count", type=int, default=0,
                   help="0 = the run's value. Must equal num_envs for a 1:1 "
                        "design-to-env mapping")
    p.add_argument("--goals_per_episode", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0,
                   help="0 = derived: 300 * goals * episodes")
    p.add_argument("--success_tolerance", type=float, default=None,
                   help="Pins termination.eval_success_tolerance in metres. Default "
                        "is the suite's 0.01 (the curriculum FLOOR); a mid-training "
                        "checkpoint usually needs looser to give a readable ranking")
    p.add_argument("--dr", default="off", choices=("off", "train", "hard"),
                   help="Domain randomization profile (default off, matching the "
                        "population runs)")
    p.add_argument("--rl_device", default="cuda")
    p.add_argument("--sapg_expl_coef", type=float, default=50.0,
                   help="50.0 for SAPG checkpoints; negative means plain PPO. Wrong "
                        "value shifts the obs vector by one slot, silently")
    p.add_argument("--population_path", default=None,
                   help="Evaluate a DIFFERENT population under this run's policy "
                        "and protocol: a directory holding manifest.json. Takes "
                        "precedence over the run's robot_population_seed. Used to "
                        "score mutant populations, whose designs are index-aligned "
                        "to the parents' -- child i sits in env i and therefore "
                        "holds object i % pool_size, the same object parent i had, "
                        "so child-minus-parent is paired within-object.")
    p.add_argument("--condition", default=None,
                   help="A committed EvalCondition id, 'axis/name' (see "
                        "coevolution/eval/suites.py). Its overrides are applied ON TOP "
                        "of --dr, so the condition wins on any field it names. Use "
                        "with --dr off: the dr_knob conditions pin every DR field "
                        "explicitly, so nothing falls back to a configclass "
                        "default that the training run never used.")
    p.add_argument("--record_value", action="store_true",
                   help="Also record the critic's mean V(s) per design. Free: the "
                        "action forward pass already computes it. Lets us ask "
                        "whether the value function agrees with measured "
                        "performance -- i.e. whether the policy KNOWS which "
                        "designs are good, which would be a fitness estimate "
                        "costing one forward pass instead of a rollout.")
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="Record cumulative goals every N steps (0 = off). Turns "
                        "one run into a measurement at EVERY budget up to "
                        "step_budget: two runs differing only in --env_seed then "
                        "give the reliability-vs-budget curve from one pair of "
                        "jobs, instead of one job per budget.")
    p.add_argument("--env_seed", type=int, default=None,
                   help="Pins DirectRLEnvCfg.seed, which seeds the ROLLOUT: the "
                        "object's random spawn orientation and the goal sequences. "
                        "Everything else is unaffected -- the object pool carries "
                        "its own np.random.seed(object_seed), and the design->env "
                        "map is index arithmetic. So two runs differing only in "
                        "this are independent replicates of the same measurement, "
                        "which is how you measure the reliability ceiling instead "
                        "of modelling it.")
    p.add_argument("--step_budget", type=int, default=0,
                   help="THROUGHPUT MODE. >0 runs exactly this many steps and "
                        "scores each design by TOTAL goals hit in them, across "
                        "however many episodes it gets. No design is frozen out "
                        "early, so every design receives an identical step "
                        "budget and the number is a rate, not a per-episode "
                        "average. Also splits the budget in half and records "
                        "each half, so split-half reliability of the metric is "
                        "computable without a second run.")
    p.add_argument("--progress_every", type=int, default=250)
    return p


def main() -> None:
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the run's config BEFORE booting Kit -- a typo here should cost a
    # second, not a two-minute Isaac Sim startup.
    from coevolution.population.run_config import (
        apply_run_fields, eval_protocol, get_by_path, load_run_config,
        run_env_dict, set_by_path, synthesise_policy_config,
    )

    run_cfg = load_run_config(run_dir)
    run_env = run_env_dict(run_cfg)

    if get_by_path(run_env, "assets.robot_population_seed") is None:
        raise SystemExit(
            f"{run_dir.name} is not a population run (robot_population_seed is None)."
        )

    run_count = int(get_by_path(run_env, "assets.robot_population_count"))
    num_envs_hint = args.num_envs or args.population_count or run_count

    # RlPlayer reads cfg["train"], but a run's .hydra/config.yaml stores the same
    # rl_games block under "agent". Rather than make the caller hand-assemble a
    # file (and risk pairing a checkpoint with someone else's network shape),
    # synthesise it from the run that produced the checkpoint.
    policy_config = args.policy_config
    if policy_config is None:
        policy_config = synthesise_policy_config(
            run_cfg, out_dir / "policy_config.yaml", num_envs_hint)
        print(f"[pop] synthesised policy config from {run_dir.name}/.hydra -> "
              f"{policy_config}")

    protocol = eval_protocol(args.dr, args.goals_per_episode, args.success_tolerance)
    condition_note = None
    if args.condition:
        from coevolution.eval.suites import condition_by_id
        cond = condition_by_id(args.condition)
        condition_note = cond.note
        # After the DR profile, so the condition is authoritative on every field
        # it names and merely inherits the rest.
        protocol.update(cond.overrides)
        print(f"[pop] condition {cond.id}: {cond.note}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    # Hashed now, not at the end: the training job that wrote it rotates
    # checkpoints, and losing a finished eval to a deleted file is a bad trade.
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    t_boot = time.perf_counter()
    app = AppLauncher(args).app

    import numpy as np
    import torch

    import isaacsimenvs  # noqa: F401
    from coevolution.population.object_pool import (
        pool_index_for_env, reconstruct_pool,
    )
    from coevolution.eval.rl_player import RlPlayer
    from hand_sampler.population import load_population
    from isaacsimenvs.pose_reaching_6d.env import PoseReachEnv
    from isaacsimenvs.pose_reaching_6d.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    # One env class now; a population is what makes it multi-embodiment. The
    # merged cfg defaults to no population (seed -1), so ask for one before the
    # run's own config gets a chance to override it.
    cfg.assets.robot_population_seed = 0
    apply_run_fields(cfg, run_env)

    count = args.population_count or int(cfg.assets.robot_population_count)
    num_envs = args.num_envs or count
    if num_envs != count:
        # Not fatal, but it breaks "env i is design i": designs would repeat or
        # be dropped, and every per-design score below would be mislabelled.
        raise SystemExit(
            f"num_envs ({num_envs}) must equal population_count ({count}) so each "
            "design gets exactly one env. Pass both to run a smaller subset."
        )
    cfg.assets.robot_population_count = count
    cfg.scene.num_envs = num_envs
    if args.env_seed is not None:
        cfg.seed = int(args.env_seed)
        print(f"[pop] env seed pinned to {cfg.seed}")
    if args.population_path:
        # Set AFTER apply_run_fields so it wins over whatever the run carried.
        cfg.assets.robot_population_path = str(Path(args.population_path).resolve())
        print(f"[pop] population overridden: {cfg.assets.robot_population_path}")

    for key, value in protocol.items():
        set_by_path(cfg, key, value)

    tol = protocol["termination.eval_success_tolerance"]
    print(f"[pop] {count} designs, {num_envs} envs, {args.episodes} episodes each, "
          f"{args.goals_per_episode} goals/episode, DR={args.dr}, "
          f"success tolerance {tol * 100:.1f} cm", flush=True)

    env = PoseReachEnv(cfg=cfg)
    inner = env
    inner._replay_target_lab_order = None
    spec = inner.scene_record.robot_spec
    n_act = int(cfg.action_space)

    player = RlPlayer(
        num_observations=int(cfg.observation_space), num_actions=n_act,
        config_path=policy_config, checkpoint_path=args.checkpoint,
        device=args.rl_device,
        sapg_expl_coef=args.sapg_expl_coef if args.sapg_expl_coef >= 0 else None,
        num_envs=num_envs,
    )
    player.player.init_rnn()
    boot_sec = time.perf_counter() - t_boot
    print(f"[pop] obs {cfg.observation_space}, act {n_act}, boot {boot_sec:.0f}s",
          flush=True)

    # --- rollout ----------------------------------------------------------
    dev = inner.device
    N, E = num_envs, args.episodes
    n_goals = args.goals_per_episode
    max_steps = args.max_steps or (300 * n_goals * E)
    budget_mode = args.step_budget > 0
    if budget_mode:
        max_steps = int(args.step_budget)
        # The per-episode matrices are sized for a target episode count that
        # throughput mode does not have. A design that fails instantly banks
        # many short episodes, so size generously and say so if it overflows --
        # the scalar totals below are unaffected either way.
        E = max(E, 256)

    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros((N, n_act), device=dev))

    # Isaac Lab auto-resets a done env, so unlike run_eval (one trajectory per
    # env) this keeps stepping and banks each episode as it lands. Envs that
    # reach `episodes` are frozen out of the accounting but keep stepping --
    # they share a physics scene and cannot be removed from it.
    episodes_done = torch.zeros(N, dtype=torch.long, device=dev)
    goals_per_ep = torch.zeros(N, E, dtype=torch.float32, device=dev)
    lifted_eps = torch.zeros(N, dtype=torch.long, device=dev)
    ever_lifted = torch.zeros(N, dtype=torch.bool, device=dev)
    steps_per_ep = torch.zeros(N, E, dtype=torch.long, device=dev)
    ep_step = torch.zeros(N, dtype=torch.long, device=dev)
    # Throughput accounting, independent of the per-episode matrices: goals from
    # episodes that have ENDED. The in-flight episode's goals live in
    # inner._successes and are added when the budget runs out.
    banked_goals = torch.zeros(N, dtype=torch.float32, device=dev)
    value_sum = torch.zeros(N, dtype=torch.float64, device=dev)
    value_n = 0
    half_goals = torch.zeros(N, dtype=torch.float32, device=dev)
    half_step = max_steps // 2
    snap_every = int(args.snapshot_every)
    snaps: list[tuple[int, "torch.Tensor"]] = []
    term_complete = torch.zeros(N, dtype=torch.long, device=dev)
    term_fall = torch.zeros(N, dtype=torch.long, device=dev)
    term_timeout = torch.zeros(N, dtype=torch.long, device=dev)

    t_roll = time.perf_counter()
    steps_taken = 0
    for step in range(max_steps):
        policy_obs = obs["policy"].to(args.rl_device)
        if args.record_value:
            action, values = player.get_action_and_value(
                policy_obs, deterministic_actions=True)
            value_sum += values.double().to(dev)
            value_n += 1
        else:
            action = player.get_normalized_action(policy_obs, deterministic_actions=True)
        obs, _rew, terminated, truncated, _ = env.step(action.to(dev))
        steps_taken = step + 1

        # Throughput mode gives every design the same step budget, so nothing
        # is frozen out: a design that finishes its episodes early keeps going
        # and keeps scoring, which is the whole point of a rate.
        active = (torch.ones_like(episodes_done, dtype=torch.bool) if budget_mode
                  else episodes_done < E)
        if not bool(active.any()):
            break

        ep_step = torch.where(active, ep_step + 1, ep_step)
        ever_lifted |= active & inner._lifted_object.bool()

        done_now = active & (terminated | truncated)
        if bool(done_now.any()):
            # step() has already reset those envs, so the episode's final count
            # is in _prev_episode_successes, not _successes.
            final = inner._prev_episode_successes.float()
            idx = episodes_done.clamp(max=E - 1)
            rows = done_now.nonzero(as_tuple=True)[0]
            banked_goals[rows] += final[rows]
            goals_per_ep[rows, idx[rows]] = final[rows]
            steps_per_ep[rows, idx[rows]] = ep_step[rows]

            complete = done_now & (final >= n_goals)
            term_complete += complete.long()
            term_fall += (done_now & ~complete & terminated).long()
            term_timeout += (done_now & ~complete & truncated).long()

            lifted_eps += (done_now & ever_lifted).long()
            ever_lifted &= ~done_now
            ep_step = torch.where(done_now, torch.zeros_like(ep_step), ep_step)
            episodes_done += done_now.long()

        if budget_mode and (step + 1) == half_step:
            half_goals = banked_goals + inner._successes.float()

        if snap_every and (step + 1) % snap_every == 0:
            # Cumulative, including the in-flight episode -- those goals were
            # really scored inside the budget, and dropping them would penalise
            # exactly the designs still going when the snapshot lands.
            snaps.append((step + 1,
                          (banked_goals + inner._successes.float()).cpu().clone()))

        if (step + 1) % args.progress_every == 0:
            finished = int((episodes_done >= E).sum())
            banked = episodes_done.clamp(max=E)
            mean_so_far = float(
                (goals_per_ep.sum(dim=1) / banked.clamp(min=1).float()).mean())
            print(f"[pop] step {step + 1}/{max_steps}: {finished}/{N} designs done, "
                  f"mean episodes {float(banked.float().mean()):.2f}/{E}, "
                  f"running mean goals {mean_so_far:.2f}/{n_goals}", flush=True)

    roll_sec = time.perf_counter() - t_roll
    # The final, unfinished episode counts: its goals were really hit inside the
    # budget, and dropping them would penalise exactly the designs that were
    # still going when time ran out.
    total_goals = banked_goals + inner._successes.float()
    second_half_goals = total_goals - half_goals
    if budget_mode:
        over = int((episodes_done > E).sum())
        if over:
            print(f"[pop] NOTE: {over}/{N} designs banked more than {E} episodes; "
                  f"their per-episode columns saturate. Totals are exact.",
                  flush=True)
    incomplete = 0 if budget_mode else int((episodes_done < E).sum())
    if incomplete:
        print(f"[pop] WARNING: {incomplete}/{N} designs banked fewer than {E} "
              f"episodes within the {max_steps}-step cap; their means average "
              f"fewer episodes and are noisier, not biased.", flush=True)

    # --- per-design results ------------------------------------------------
    banked = episodes_done.clamp(max=E)
    ep_sum = goals_per_ep.sum(dim=1)
    mean_goals = ep_sum / banked.clamp(min=1).float()
    # Population std over the banked episodes, per design.
    sq = (goals_per_ep ** 2).sum(dim=1)
    var = (sq / banked.clamp(min=1).float()) - mean_goals ** 2
    std_goals = var.clamp(min=0).sqrt()

    ep_mask = (torch.arange(E, device=dev)[None, :] < banked[:, None])
    full_rate = ((goals_per_ep >= n_goals) & ep_mask).sum(dim=1).float() / banked.clamp(min=1).float()
    zero_rate = ((goals_per_ep <= 0) & ep_mask).sum(dim=1).float() / banked.clamp(min=1).float()

    hands = load_population(int(cfg.assets.robot_population_seed))[:count]
    pool = reconstruct_pool(
        handle_head_types=tuple(cfg.assets.handle_head_types),
        num_assets_per_type=int(cfg.assets.num_assets_per_type),
        object_seed=int(cfg.assets.object_seed),
        shuffle=bool(cfg.assets.shuffle_assets),
        density_scale=float(cfg.assets.object_density_scale),
    )
    pool_size = len(pool)

    design_index = inner.scene_record.robot_design_index.cpu().numpy()
    live_pool_index = inner.scene_record.object_pool_index.cpu().numpy()

    mg = mean_goals.cpu().numpy()
    sg = std_goals.cpu().numpy()
    bk = banked.cpu().numpy()
    fr = full_rate.cpu().numpy()
    zr = zero_rate.cpu().numpy()
    le = lifted_eps.cpu().numpy()
    gpe = goals_per_ep.cpu().numpy()
    tc, tf, tt = (term_complete.cpu().numpy(), term_fall.cpu().numpy(),
                  term_timeout.cpu().numpy())
    snap_np = [(st, v.numpy()) for st, v in snaps]
    mean_value = (value_sum / max(value_n, 1)).float().cpu().numpy()
    tg = total_goals.cpu().numpy()
    hg = half_goals.cpu().numpy()
    sh = second_half_goals.cpu().numpy()
    ed = episodes_done.cpu().numpy()

    rows = []
    for e in range(N):
        d_idx = int(design_index[e])
        p_idx = int(live_pool_index[e])
        expected = pool_index_for_env(e, pool_size)
        if p_idx != expected:
            raise RuntimeError(
                f"env {e}: live object index {p_idx} != expected {expected}. The "
                "env->pool rule changed; per-design object labels would be wrong.")
        entry = pool[p_idx]
        hand = hands[d_idx]
        rows.append({
            "env": e,
            "design_index": d_idx,
            "design": hand.name,
            "n_active_fingers": hand.n_active_fingers,
            "n_active_joints": hand.n_active_joints,
            "pool_index": p_idx,
            "object_type": entry.type,
            "object_shape": entry.shape,
            "episodes": int(bk[e]),
            "episodes_completed": int(ed[e]),
            "total_goals": float(tg[e]),
            **({"mean_value": float(mean_value[e])} if args.record_value else {}),
            **{f"goals_at_{st}": float(v[e]) for st, v in snap_np},
            "goals_first_half": float(hg[e]),
            "goals_second_half": float(sh[e]),
            "mean_goals": float(mg[e]),
            "std_goals": float(sg[e]),
            "full_completion_rate": float(fr[e]),
            "zero_goal_rate": float(zr[e]),
            "lift_rate": float(le[e]) / max(1, int(bk[e])),
            "goals_per_episode": [float(v) for v in gpe[e][:int(bk[e])]],
            "term_complete": int(tc[e]),
            "term_fall": int(tf[e]),
            "term_timeout": int(tt[e]),
        })

    record = {
        "schema_version": 1,
        "kind": "population_eval",
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "population": {
            "path": getattr(cfg.assets, "robot_population_path", None),
            "seed": int(cfg.assets.robot_population_seed),
            "count": count,
            "num_envs": num_envs,
        },
        "object_pool": {
            "handle_head_types": list(cfg.assets.handle_head_types),
            "num_assets_per_type": int(cfg.assets.num_assets_per_type),
            "object_seed": int(cfg.assets.object_seed),
            "pool_size": pool_size,
            "assignment": "training pairing reproduced: env i holds design i, object i % pool_size",
        },
        "protocol": {
            "episodes": E,
            "goals_per_episode": n_goals,
            "dr_profile": args.dr,
            "env_seed": args.env_seed,
            "condition": args.condition,
            "condition_note": condition_note,
            "eval_success_tolerance": protocol["termination.eval_success_tolerance"],
            "step_budget": (int(args.step_budget) if budget_mode else None),
            "snapshot_every": (snap_every or None),
            "snapshot_steps": [st for st, _ in snap_np],
            "metric": ("total goals in step_budget steps" if budget_mode
                       else "mean goals per episode"),
            "max_steps": max_steps,
            "steps_taken": steps_taken,
            "designs_short_of_target_episodes": incomplete,
            "overrides": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in protocol.items()},
        },
        "policy": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "config": str(policy_config),
            "sapg_expl_coef": (args.sapg_expl_coef if args.sapg_expl_coef >= 0 else None),
            "robot_spec_layout": spec.name,
        },
        "summary": {
            "mean_goals": float(mg.mean()),
            "median_goals": float(np.median(mg)),
            "std_across_designs": float(mg.std()),
            "best": max(rows, key=lambda r: r["mean_goals"])["design"],
            "best_mean_goals": float(mg.max()),
            "worst_mean_goals": float(mg.min()),
            "designs_scoring_zero": int((mg <= 0).sum()),
            "mean_total_goals": float(tg.mean()),
            "median_total_goals": float(np.median(tg)),
            "std_total_goals_across_designs": float(tg.std()),
            "max_total_goals": float(tg.max()),
            "designs_with_zero_total": int((tg <= 0).sum()),
            "split_half_r": (float(np.corrcoef(hg, sh)[0, 1])
                             if budget_mode and hg.std() > 0 and sh.std() > 0
                             else None),
        },
        "wall_clock": {"boot_sec": boot_sec, "rollout_sec": roll_sec},
        "per_design": rows,
    }

    json_path = out_dir / "population_eval.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    csv_path = out_dir / "per_design.csv"
    cols = ["design", "design_index", "n_active_fingers", "n_active_joints",
            "pool_index", "object_type", "object_shape", "total_goals",
            *(["mean_value"] if args.record_value else []),
            *[f"goals_at_{st}" for st, _ in snap_np],
            "goals_first_half", "goals_second_half", "episodes_completed",
            "episodes", "mean_goals", "std_goals", "full_completion_rate",
            "zero_goal_rate", "lift_rate", "term_complete", "term_fall",
            "term_timeout"]
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    _print_summary(record, rows, np)
    print(f"\n[pop] wrote {json_path}")
    print(f"[pop] wrote {csv_path}")

    env.close()
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _print_summary(record, rows, np) -> None:
    s = record["summary"]
    n_goals = record["protocol"]["goals_per_episode"]
    print(f"\n=== population eval: {len(rows)} designs x "
          f"{record['protocol']['episodes']} episodes ===")
    print(f"  mean goals across designs : {s['mean_goals']:.3f} / {n_goals}")
    print(f"  median                    : {s['median_goals']:.3f}")
    print(f"  spread across designs (sd): {s['std_across_designs']:.3f}")
    print(f"  best / worst design mean  : {s['best_mean_goals']:.3f} / "
          f"{s['worst_mean_goals']:.3f}")
    print(f"  designs scoring exactly 0 : {s['designs_scoring_zero']}")
    if record["protocol"].get("step_budget"):
        b = record["protocol"]["step_budget"]
        print(f"\n  --- throughput: total goals in {b} steps ---")
        print(f"  mean total goals          : {s['mean_total_goals']:.3f}")
        print(f"  median                    : {s['median_total_goals']:.3f}")
        print(f"  spread across designs (sd): {s['std_total_goals_across_designs']:.3f}")
        print(f"  best design               : {s['max_total_goals']:.3f}")
        print(f"  designs with zero total   : {s['designs_with_zero_total']}")
        r = s.get("split_half_r")
        if r is not None:
            # Spearman-Brown: the halves each carry half the budget, so their
            # correlation understates the full-budget metric's reliability.
            sb = 2 * r / (1 + r) if r > -1 else float("nan")
            print(f"  split-half r (3k vs 3k)   : {r:.3f}  "
                  f"-> full-budget reliability {sb:.3f}")
            print("    (how much of the between-design spread is real signal; "
                  "near 0 means read only grouped statistics)")

    def _group(key):
        buckets = {}
        for r in rows:
            buckets.setdefault(r[key], []).append(r["mean_goals"])
        return {k: (len(v), float(np.mean(v))) for k, v in sorted(buckets.items())}

    # The object confound, made visible: a design's score is partly its object's.
    print("\n  by object type (score is partly the object's, not the design's):")
    for k, (n, m) in sorted(_group("object_type").items(), key=lambda kv: -kv[1][1]):
        print(f"    {k:12s} {n:6d} designs   mean {m:.3f}")
    print("\n  by active finger count:")
    for k, (n, m) in _group("n_active_fingers").items():
        print(f"    {k} fingers    {n:6d} designs   mean {m:.3f}")

    top = sorted(rows, key=lambda r: -r["mean_goals"])[:5]
    print("\n  top 5 designs:")
    for r in top:
        print(f"    {r['design']}  {r['mean_goals']:6.3f} +- {r['std_goals']:.2f}  "
              f"({r['n_active_fingers']}f/{r['n_active_joints']}j, "
              f"{r['object_type']})")


if __name__ == "__main__":
    main()
