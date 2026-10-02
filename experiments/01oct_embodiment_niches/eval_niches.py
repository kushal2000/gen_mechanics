"""Phase 1 of the niche test: measure a trained hand's policy on many axes, not just goals/episode.

One process = one hand x one condition: boots Kit, builds the hand's own training env (its run's
saved config, so its own physics, joint speed cap included), runs its final checkpoint greedily
on N envs for a fixed sim time, and writes one JSON of metrics.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/01oct_embodiment_niches/eval_niches.py \
        --hand allegro --condition nominal --out debug_outputs/embodiment_niches/allegro__nominal.json

Changes from training, all deliberate:
  * no 50-goal episode cap (max_consecutive_successes 0): throughput is measured over a fixed
    window, so a fast hand is not clipped at the ceiling that saturates goals/episode;
  * greedy actions (deterministic, SAPG coefficient 0), as when the policy is deployed;
  * the condition's perturbation (see CONDITIONS), everything else as trained.

Torque is Isaac Lab's implicit-actuator estimate (stiffness x error + damping x velocity error,
clipped), not a PhysX joint-force readback: fine for comparing hands, not a wattmeter.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/30sep_ten_hands"))

TASK = "GenMech-InHandReorient-Direct-v0"

# condition -> (description, env_cfg overrides as (dotted path, value))
CONDITIONS = {
    "nominal":  ("as trained (45 mm cube)", []),
    "cube40":   ("40 mm cube", [("assets.object_pool", "cube1_40mm")]),
    "cube55":   ("55 mm cube", [("assets.object_pool", "cube1_55mm")]),
    "cube65":   ("65 mm cube", [("assets.object_pool", "cube1_65mm")]),
    "light":    ("cube mass x0.5", [("assets.object_density_scale", 0.5)]),
    "heavy":    ("cube mass x2", [("assets.object_density_scale", 2.0)]),
    # Relative: half of whatever the run trained with (1.0 for the vendor-dynamics runs, 0.5 uniform).
    "slippery": ("cube friction x0.5", [("assets.object_friction", "x0.5")]),
    # force = N(0,1) x cube mass x force_scale for one 1/60 s step; ~1 g (10) did nothing measurable
    "push":     ("random pushes ~2 g, ~1.2 per second",
                 [("domain_randomization.force_scale", 20.0),
                  ("domain_randomization.force_prob_range", (0.02, 0.02))]),
    "push_hard": ("random pushes ~5 g, ~1.2 per second",
                  [("domain_randomization.force_scale", 50.0),
                   ("domain_randomization.force_prob_range", (0.02, 0.02))]),
}


def _set(cfg, dotted: str, value) -> None:
    *path, leaf = dotted.split(".")
    for p in path:
        cfg = getattr(cfg, p)
    setattr(cfg, leaf, value)


def _summ(xs) -> dict:
    import numpy as np
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def _streaks(ST) -> dict:
    """Goals before a drop: the minimum observed, and Kaplan-Meier quantiles over censored episodes."""
    import numpy as np
    g, dropped = ST[:, 0].astype(np.int64), ST[:, 1].astype(bool)
    out = {"episodes": int(len(g)), "drops": int(dropped.sum()),
           "min": int(g[dropped].min()) if dropped.any() else None,
           "drops_before_first_goal": int((dropped & (g == 0)).sum())}
    surv, at_risk_k = 1.0, {}
    for k in np.unique(g[dropped]):
        n_risk = int((g >= k).sum())
        surv *= 1.0 - int(((g == k) & dropped).sum()) / n_risk
        at_risk_k[int(k)] = surv
    for q in (0.01, 0.05, 0.10, 0.25, 0.5):
        # smallest k at which at least a q fraction of episodes have dropped; None = never in window
        out[f"p{int(q * 100)}"] = next((k for k, s in sorted(at_risk_k.items()) if 1.0 - s >= q), None)
    out["max_observed_streak"] = int(g.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--set", default="vendor", choices=("vendor", "uniform"),
                    help="which runs: the ten-hands vendor/10 rad/s runs, or the uniform-dynamics runs")
    ap.add_argument("--condition", default="nominal", choices=list(CONDITIONS))
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seconds", type=float, default=60.0, help="sim time per env")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="", help="default: the hand's latest checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from isaaclab.app import AppLauncher
    lp = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(lp)
    app_args, _ = lp.parse_known_args([])
    app_args.headless, app_args.device = True, args.device
    app = AppLauncher(app_args).app

    import gymnasium as gym
    import numpy as np
    import torch
    import yaml
    import tempfile
    from omegaconf import OmegaConf

    import isaacsimenvs  # noqa: F401
    from coevolution.eval.rl_player import RlPlayer
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml
    from isaacsimenvs.pose_reaching_6d.obs_utils.observations import _quat_angle
    if args.set == "uniform":
        # The uniform-dynamics runs (experiments/01oct_uniform_dynamics): each hand's newest checkpoint.
        import glob, os
        logs = REPO / "debug_outputs/train_logs/01oct_uniform_dynamics"
        pths = glob.glob(f"{logs}/0_scale_train_left_{args.hand}_uniform_c01_*/rank_0/*/nn/*.pth")
        ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else pathlib.Path(max(pths, key=os.path.getmtime))
    else:
        from viser_zero_shot import HANDS, latest_checkpoint
        spec_ref, study, _ = HANDS[args.hand]
        ckpt = pathlib.Path(args.checkpoint) if args.checkpoint else latest_checkpoint(study)
    run_dir = ckpt.parents[3]
    saved = OmegaConf.load(run_dir / "rank_0/.hydra/config.yaml")
    sys.argv = [sys.argv[0]]
    result = {}

    @hydra_task_config_with_yaml(TASK, "rl_games_joint_transformer_cfg_entry_point")
    def _run(env_cfg, agent_cfg):
        env_d = OmegaConf.to_container(saved.env, resolve=True)
        for k in ("observation_space", "state_space", "action_space"):
            env_d.pop(k, None)
        env_cfg.from_dict(env_d)
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.termination.max_consecutive_successes = 0
        for path, val in CONDITIONS[args.condition][1]:
            if isinstance(val, str) and val.startswith("x"):        # relative to the trained value
                *head, leaf = path.split(".")
                obj = env_cfg
                for h in head:
                    obj = getattr(obj, h)
                val = getattr(obj, leaf) * float(val[1:])
            _set(env_cfg, path, val)

        env = gym.make(TASK, cfg=env_cfg)
        inner = env.unwrapped
        obs, _ = env.reset()

        acfg = OmegaConf.to_container(saved, resolve=True)["agent"]
        acfg["params"]["config"]["multi_gpu"] = False
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        yaml.safe_dump({"train": acfg}, f)
        f.close()
        player = RlPlayer(obs["policy"].shape[-1], inner.action_space.shape[-1], f.name, str(ckpt),
                          device=args.device, sapg_expl_coef=0.0, num_envs=inner.num_envs)

        N, dev, dt = inner.num_envs, inner.device, float(inner.step_dt)
        hid = torch.as_tensor(inner._hand_joint_ids, device=dev)
        lim = inner.robot.data.joint_pos_limits[0, hid]
        real = (lim[:, 1] - lim[:, 0]) > 1e-6                 # gen-SHARPA's ghost slots excluded
        jid = hid[real]
        J = int(jid.numel())
        vlim = inner.robot.data.joint_vel_limits[0, jid]
        steps = int(round(args.seconds / dt))

        f64 = dict(device=dev, dtype=torch.float64)
        z = lambda: torch.zeros(N, **f64)
        acc = {k: torch.zeros((), **f64) for k in (
            "work", "tau2", "work_m", "tau2_m", "dtarget", "qacc2", "lin", "lin2", "ang", "palm_d", "palm_d2",
            "obj_acc", "vel_sat", "n")}
        S1, S2 = torch.zeros(J, **f64), torch.zeros(J, J, **f64)
        absqd = torch.zeros(J, **f64)
        seg_start, seg_rot, seg_work = torch.zeros(N, dtype=torch.long, device=dev), z(), z()
        seg_from_reset = torch.ones(N, dtype=torch.bool, device=dev)
        streak = torch.zeros(N, dtype=torch.long, device=dev)    # goals since this episode began
        streak_rec = []      # (goals before the episode ended, 1 = ended by a drop / 0 = censored)

        def angle_now():
            return _quat_angle(inner.object.data.root_quat_w, inner.goal_viz.data.root_quat_w)

        seg_theta0 = angle_now().double()
        succ_rec, fail_rec = [], []      # (duration_s, theta0, rot, work, from_reset) / (kind, duration_s)
        prev_v = inner.object.data.root_lin_vel_w.clone()
        prev_qd = inner.robot.data.joint_vel[:, jid].clone()
        prev_tgt = inner._cur_targets[:, jid].clone()
        n_succ = n_fall = n_tout = n_far = 0
        t0 = time.time()
        for k in range(steps):
            with torch.no_grad():
                act = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, _, _, _, _ = env.step(act)

            qd = inner.robot.data.joint_vel[:, jid].double()
            tau = inner.robot.data.applied_torque[:, jid].double()
            # What the solver actually applied along each joint axis (drive + limit constraints).
            tau_m = inner.robot.root_physx_view.get_dof_projected_joint_forces()[:, jid].double()
            acc["work_m"] += (tau_m * qd).abs().sum() * dt
            acc["tau2_m"] += (tau_m ** 2).sum() * dt
            tgt = inner._cur_targets[:, jid]
            v = inner.object.data.root_lin_vel_w
            w = inner.object.data.root_ang_vel_w
            power = (tau * qd).abs().sum(-1)
            acc["work"] += power.sum() * dt
            acc["tau2"] += (tau ** 2).sum() * dt
            acc["dtarget"] += (tgt - prev_tgt).abs().double().sum() / dt
            acc["qacc2"] += (((qd - prev_qd.double()) / dt) ** 2).sum()
            sp = v.norm(dim=-1).double()
            acc["lin"] += sp.sum(); acc["lin2"] += (sp ** 2).sum()
            wn = w.norm(dim=-1).double()
            acc["ang"] += wn.sum()
            d = (inner.object.data.root_pos_w - inner._palm_center_pos_w).norm(dim=-1).double()
            acc["palm_d"] += d.sum(); acc["palm_d2"] += (d ** 2).sum()
            acc["obj_acc"] += ((v - prev_v).norm(dim=-1).double() / dt).sum()
            acc["vel_sat"] += (qd.abs() >= 0.95 * vlim.double()).any(-1).double().sum()
            acc["n"] += N
            S1 += qd.sum(0); S2 += qd.T @ qd; absqd += qd.abs().sum(0)
            seg_rot += wn * dt
            seg_work += power * dt

            succ = inner._is_success.bool()
            r = inner._termination_reasons
            fall, tout, far = r["fall"].bool(), r["timeout"].bool(), r["hand_far"].bool()
            ended = fall | tout | far
            th_now = angle_now().double()
            if succ.any():
                i = succ.nonzero().squeeze(-1)
                dur = (k + 1 - seg_start[i]).double() * dt
                succ_rec.append(torch.stack([dur, seg_theta0[i], seg_rot[i], seg_work[i],
                                             seg_from_reset[i].double()], -1).cpu())
            fail = ended & ~succ
            if fail.any():
                i = fail.nonzero().squeeze(-1)
                kind = torch.where(fall[i], 0, torch.where(tout[i], 1, 2)).double()
                fail_rec.append(torch.stack([kind, (k + 1 - seg_start[i]).double() * dt], -1).cpu())
            streak += succ.long()
            if ended.any():
                i = ended.nonzero().squeeze(-1)
                # Only a drop is the failure; a timeout or hand_far ends the episode without one.
                streak_rec.append(torch.stack([streak[i], fall[i].long()], -1).cpu())
                streak = torch.where(ended, torch.zeros_like(streak), streak)
            new = succ | ended
            if new.any():
                seg_start = torch.where(new, torch.full_like(seg_start, k + 1), seg_start)
                seg_theta0 = torch.where(new, th_now, seg_theta0)
                seg_rot = torch.where(new, torch.zeros_like(seg_rot), seg_rot)
                seg_work = torch.where(new, torch.zeros_like(seg_work), seg_work)
                seg_from_reset = (seg_from_reset & ~succ) | ended
            n_succ += int(succ.sum()); n_fall += int(fall.sum()); n_tout += int(tout.sum())
            n_far += int(far.sum())
            prev_v, prev_qd, prev_tgt = v.clone(), qd.clone(), tgt.clone()
        wall = time.time() - t0
        streak_rec.append(torch.stack([streak, torch.zeros_like(streak)], -1).cpu())   # still running
        ST = torch.cat(streak_rec).numpy()

        S = torch.cat(succ_rec).numpy() if succ_rec else np.zeros((0, 5))
        F = torch.cat(fail_rec).numpy() if fail_rec else np.zeros((0, 2))
        env_s = N * steps * dt
        n = float(acc["n"])
        mean = (S1 / n)
        C = (S2 / n - torch.outer(mean, mean)).cpu().numpy()
        ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        pr = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
        k90 = int(np.searchsorted(np.cumsum(ev) / max(ev.sum(), 1e-12), 0.9) + 1)
        share = np.sort((absqd / absqd.sum()).cpu().numpy())[::-1]
        k80_joints = int(np.searchsorted(np.cumsum(share), 0.8) + 1)
        mid = S[S[:, 4] == 0] if len(S) else S        # goals that started from a previous goal
        first = S[S[:, 4] == 1] if len(S) else S      # first goal after a reset (includes settling)

        result.update({
            "hand": args.hand, "set": args.set, "condition": args.condition, "condition_desc": CONDITIONS[args.condition][0],
            "checkpoint": str(ckpt), "num_envs": N, "seconds": args.seconds, "dt": dt, "env_seconds": env_s,
            "wall_s": wall, "joints": J, "hand_mass_kg": float(inner.robot.data.default_mass[0].sum()),
            "cube_mass_kg": float(inner._object_mass[0]),
            "vel_limit_rad_s": [float(vlim.min()), float(vlim.max())],
            # throughput / reliability
            "goals_per_min": 60.0 * n_succ / env_s,
            "drops_per_min": 60.0 * n_fall / env_s,
            "timeouts_per_min": 60.0 * n_tout / env_s,
            "goals_per_drop": n_succ / max(n_fall, 1),
            "counts": {"goals": n_succ, "drops": n_fall, "timeouts": n_tout, "hand_far": n_far},
            "time_per_goal_s": _summ(mid[:, 0]),
            "first_goal_after_reset_s": _summ(first[:, 0]),
            "goal_angle_deg": _summ(np.degrees(mid[:, 1])),
            # rotation the cube travelled per unit of rotation the goal required (1 = shortest path)
            "path_efficiency": float(mid[:, 1].sum() / max(mid[:, 2].sum(), 1e-9)) if len(mid) else None,
            "failed_segment_s": _summ(F[:, 1]),
            # goals an episode achieves before dropping the cube; episodes still running (or ended
            # by a timeout) at the end are censored, so the quantiles are Kaplan-Meier estimates
            "goals_before_drop": _streaks(ST),
            # effort: measured = PhysX joint forces (use these); pd = the implicit actuator's
            # PD-law estimate, which overstates effort when joints sit at the speed cap
            "work_per_goal_J": float(acc["work_m"]) / max(n_succ, 1),
            "mean_power_W_per_env": float(acc["work_m"]) / env_s,
            "torque_sq_rate": float(acc["tau2_m"]) / env_s,
            "pd_work_per_goal_J": float(acc["work"]) / max(n_succ, 1),
            "pd_torque_sq_rate": float(acc["tau2"]) / env_s,
            "target_rate_rad_s_per_joint": float(acc["dtarget"]) / (n * J),
            "joint_acc_rms": float((acc["qacc2"] / (n * J)).sqrt()),
            "joint_speed_saturated_frac": float(acc["vel_sat"]) / n,
            # object stability
            "obj_speed_mean": float(acc["lin"]) / n,
            "obj_speed_rms": float((acc["lin2"] / n).sqrt()),
            "obj_spin_mean": float(acc["ang"]) / n,
            "obj_acc_mean": float(acc["obj_acc"]) / n,
            "palm_dist_mean": float(acc["palm_d"]) / n,
            "palm_dist_std": float((acc["palm_d2"] / n - (acc["palm_d"] / n) ** 2).clamp(min=0).sqrt()),
            # coordination
            "participation_ratio": pr, "participation_frac": pr / J,
            "pcs_for_90pct": k90, "joints_for_80pct_motion": k80_joints,
        })
        env.close()

    _run()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[niches] {args.hand}/{args.condition}: {result.get('goals_per_min', float('nan')):.1f} goals/min, "
          f"{result.get('drops_per_min', float('nan')):.2f} drops/min, "
          f"{result.get('work_per_goal_J', float('nan')):.3f} J/goal -> {out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
