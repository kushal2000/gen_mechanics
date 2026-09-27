r"""Wiring diagnostics for InHandReorient (I24, Phase 1c).

Boots one Kit process, one scene (64 envs by default), and runs the checks
the Phase 1c plan asked for BEFORE any reward-design work:

  (a) random and sinusoidal actions move every joint within its limits,
      and the action -> target mapping lands where it says it will;
  (b) object/goal quaternions and the rotation-error computation use a
      consistent convention;
  (c) goal == current object orientation gives ~0 rotation error and
      triggers success after ``success_steps``;
  (d) the observation vector contains the goal and changes when it changes;
  (e) reward-term magnitudes under a zero-action vs a random-action policy.

Prints a report; does not assert (this is a diagnostic, not a test -- the
package's pytest suite can't boot Kit at all, per
``tests/test_config_sanity.py``'s docstring).

Run (single Kit process; ~1-2 min boot):
    .venv_isaacsim/bin/python isaacsimenvs/inhand_reorient/diagnostics.py \
        --hand sharpa [--num-envs 64] [--device cuda:0]
"""

from __future__ import annotations

import argparse
import math


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", default="sharpa")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import torch
    from isaaclab.utils.math import (
        quat_from_angle_axis, quat_error_magnitude, quat_inv, quat_mul, random_orientation,
    )

    from isaacsimenvs.inhand_reorient.env import InHandReorientEnv
    from isaacsimenvs.inhand_reorient.env_cfg import InHandReorientEnvCfg
    from isaacsimenvs.inhand_reorient.obs_utils import _field_width, update_palm_frame_geometry

    cfg = InHandReorientEnvCfg()
    cfg.assets.hand_id = args.hand
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device

    env = InHandReorientEnv(cfg=cfg)
    env.reset()
    device = env.device
    n = env.num_envs
    j = env.hand_spec.num_hand_joints
    # env.robot.data.joint_names, NOT env.hand_spec.hand_joint_names: the
    # articulation view's own joint order (what soft_joint_pos_limits/
    # joint_pos/default_joint_pos are indexed by) is not guaranteed to match
    # the URDF-declaration order the spec keeps (confirmed different for
    # SHARPA -- zipping the two positionally, as an earlier version of this
    # script and of calibrate_palm_up.py's curl search both did, silently
    # paired each joint's VALUE with a DIFFERENT joint's NAME).
    names = env.robot.data.joint_names

    print(f"\n=== diagnostics: hand={args.hand} envs={n} joints={j} "
          f"fingertips={env.hand_spec.num_fingertips} ===\n", flush=True)

    lower = env.robot.data.soft_joint_pos_limits[0, :, 0].clone()
    upper = env.robot.data.soft_joint_pos_limits[0, :, 1].clone()
    default_pos0 = env.robot.data.default_joint_pos[0].clone()

    # --- (a) action -> joint range of motion + neutral-point sanity --------
    def run_actions(action_fn, steps):
        env.reset()
        lo = torch.full((j,), float("inf"), device=device)
        hi = torch.full((j,), float("-inf"), device=device)
        for t in range(steps):
            env.step(action_fn(t))
            jp = env.robot.data.joint_pos
            lo = torch.minimum(lo, jp.min(dim=0).values)
            hi = torch.maximum(hi, jp.max(dim=0).values)
        return lo, hi

    torch.manual_seed(0)
    lo_r, hi_r = run_actions(lambda t: torch.rand(n, j, device=device) * 2.0 - 1.0, 300)

    def sinusoid(t):
        phase = 2.0 * math.pi * t / 240.0
        return torch.full((n, j), math.sin(phase), device=device)

    lo_s, hi_s = run_actions(sinusoid, 480)

    print("--- (a) per-joint range of motion achieved vs limits ---")
    print(f"  {'joint':30s} {'limit':>20s} {'random (frac)':>22s} {'sinusoid (frac)':>22s}")
    for i, name in enumerate(names):
        span = (upper[i] - lower[i]).item()
        lim = f"[{lower[i]:+.3f},{upper[i]:+.3f}]"
        fr = (min(hi_r[i], upper[i]) - max(lo_r[i], lower[i])).item() / span if span > 0 else float("nan")
        fs = (min(hi_s[i], upper[i]) - max(lo_s[i], lower[i])).item() / span if span > 0 else float("nan")
        print(f"  {name:30s} {lim:>20s} "
              f"[{lo_r[i]:+.3f},{hi_r[i]:+.3f}] ({fr:>5.0%})  "
              f"[{lo_s[i]:+.3f},{hi_s[i]:+.3f}] ({fs:>5.0%})")

    print("\n--- (a2) action-neutral point: default_joint_pos vs limits midpoint ---")
    for i, name in enumerate(names):
        span = (upper[i] - lower[i]).item()
        mid = 0.5 * (upper[i] + lower[i]).item()
        off = (default_pos0[i].item() - mid) / span if span > 0 else float("nan")
        print(f"  {name:30s} default={default_pos0[i]:+.3f} mid_of_limits={mid:+.3f} "
              f"offset={off:+.1%} of span")

    print("\n--- (a3) sustained action -> _cur_targets mapping (env 0) ---")
    env.reset()
    for _ in range(60):
        env.step(torch.ones(n, j, device=device))
    targets_plus = env._cur_targets[0].clone()
    env.reset()
    for _ in range(60):
        env.step(-torch.ones(n, j, device=device))
    targets_minus = env._cur_targets[0].clone()
    for i, name in enumerate(names):
        print(f"  {name:30s} limit=[{lower[i]:+.3f},{upper[i]:+.3f}] "
              f"target@a=+1:{targets_plus[i]:+.3f} target@a=-1:{targets_minus[i]:+.3f}")

    # --- (b) frame convention sanity ---------------------------------------
    print("\n--- (b) frame convention sanity ---")
    q_id = torch.zeros(1, 4, device=device)
    q_id[:, 0] = 1.0
    err_same = quat_error_magnitude(q_id, q_id)
    axis_z = torch.zeros(1, 3, device=device)
    axis_z[:, 2] = 1.0
    q90 = quat_from_angle_axis(torch.tensor([math.pi / 2], device=device), axis_z)
    err_90 = quat_error_magnitude(q_id, q90)
    print(f"  error(q, q)                     = {err_same.item():.6f} rad (expect ~0)")
    print(f"  error(identity, 90deg about z)  = {err_90.item():.6f} rad (expect {math.pi / 2:.6f})")

    env.reset()
    palm_quat_w = env.robot.data.body_quat_w[:, env.palm_body_idx]
    obj_quat_w = env.object.data.root_quat_w
    manual_obj_quat_palm = quat_mul(quat_inv(palm_quat_w), obj_quat_w)
    diff = quat_error_magnitude(manual_obj_quat_palm, env._obj_quat_palm)
    print(f"  object_quat_palm vs manual quat_inv(palm)*obj: max err {diff.max().item():.2e} rad")
    manual_goal_quat_palm = quat_mul(quat_inv(palm_quat_w), env._goal_quat_w)
    diff2 = quat_error_magnitude(manual_goal_quat_palm, env._goal_quat_palm)
    print(f"  goal_quat_palm vs manual quat_inv(palm)*goal:  max err {diff2.max().item():.2e} rad")

    # --- (P) coordinator priority check: is the calibrated "palm up" cube --
    # actually within reach of the fingertips, or resting on a face the
    # fingers can't touch (e.g. the wrist/base face)?
    print("\n--- (P) is the spawned cube within reach of the fingertips? ---")
    env.reset()
    for _ in range(20):
        env.step(torch.zeros(n, j, device=device))
    palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx]
    tip_pos_w = env.robot.data.body_pos_w[:, env.fingertip_body_idx, :]  # (n, k, 3)
    obj_pos_w = env.object.data.root_pos_w
    rel_z = tip_pos_w[:, :, 2] - palm_pos_w[:, 2].unsqueeze(1)
    print("  fingertip z relative to palm origin (world frame), mean over envs "
          "(+ = above the palm):")
    for ti, name in enumerate(env.hand_spec.fingertip_body_names):
        print(f"    {name:26s} mean_rel_z={rel_z[:, ti].mean().item():+.4f} m")
    dist_to_tips = (tip_pos_w - obj_pos_w.unsqueeze(1)).norm(dim=-1)  # (n, k)
    nearest = dist_to_tips.min(dim=1).values
    print(f"  cube center z relative to palm origin (world): "
          f"{(obj_pos_w[:, 2] - palm_pos_w[:, 2]).mean().item():+.4f} m")
    print(f"  cube-to-nearest-fingertip distance after 20 settle steps: "
          f"mean={nearest.mean().item():.4f} m min={nearest.min().item():.4f} "
          f"max={nearest.max().item():.4f}")

    print("\n  reachable-workspace sweep: object FROZEN at its spawn point (rewritten every "
          "step so it can't fall away), hand joints driven by 300 random configs:")
    env.reset()
    frozen_pos_w = env.object.data.root_pos_w.clone()
    frozen_quat_w = torch.zeros(n, 4, device=device)
    frozen_quat_w[:, 0] = 1.0
    zero_vel = torch.zeros(n, 6, device=device)
    min_dist = torch.full((n, len(env.fingertip_body_idx)), float("inf"), device=device)
    torch.manual_seed(1)
    n_sweep = 300
    for _ in range(n_sweep):
        env.step(torch.rand(n, j, device=device) * 2.0 - 1.0)
        env.object.write_root_state_to_sim(
            torch.cat([frozen_pos_w, frozen_quat_w, zero_vel], dim=-1))
        tips = env.robot.data.body_pos_w[:, env.fingertip_body_idx, :]
        min_dist = torch.minimum(min_dist, (tips - frozen_pos_w.unsqueeze(1)).norm(dim=-1))
    frac_3cm = (min_dist < 0.03).float().mean(dim=0)
    for ti, name in enumerate(env.hand_spec.fingertip_body_names):
        print(f"    {name:26s} frac_envs_reaching_3cm={frac_3cm[ti].item():.1%}  "
              f"best_dist_seen={min_dist[:, ti].min().item():.4f} m")
    print(f"  ANY fingertip within 3cm, fraction of envs: "
          f"{(min_dist.min(dim=1).values < 0.03).float().mean().item():.1%}")
    print(f"  >=2 fingertips within 3cm (same env), fraction of envs: "
          f"{((min_dist < 0.03).sum(dim=1) >= 2).float().mean().item():.1%}")

    # --- (c) goal == current object orientation -> ~0 error, success -------
    print("\n--- (c) goal == object orientation -> zero error & success ---")
    env.reset()
    env._goal_quat_w[:] = env.object.data.root_quat_w.clone()
    update_palm_frame_geometry(env)
    success_seen = torch.zeros(n, dtype=torch.bool, device=device)
    max_err_after_set = env._rot_error.max().item()
    n_hold = env.cfg.termination.success_steps + 5
    zero_action = torch.zeros(n, j, device=device)
    for _ in range(n_hold):
        env.step(zero_action)
        success_seen |= env._is_success
    print(f"  rot_error immediately after setting goal := object: {max_err_after_set:.6f} rad (expect ~0)")
    print(f"  held {n_hold} zero-action steps (success_steps={env.cfg.termination.success_steps})")
    print(f"  fraction of envs with _is_success at least once: {success_seen.float().mean().item():.1%}")

    # --- (d) observation contains the goal and reacts to it changing -------
    print("\n--- (d) observation reflects the goal ---")
    env.reset()
    obs1 = env._get_observations()["policy"].clone()
    env._goal_quat_w[:] = random_orientation(n, device=device)
    update_palm_frame_geometry(env)
    obs2 = env._get_observations()["policy"].clone()
    diff_obs = (obs1 - obs2).abs()
    offsets, c = {}, 0
    for f in env.cfg.obs.obs_list:
        w = _field_width(f, env.hand_spec)
        offsets[f] = (c, c + w)
        c += w
    for f, (s, e) in offsets.items():
        print(f"  {f:22s} idx[{s:4d}:{e:4d}] max_abs_diff={diff_obs[:, s:e].max().item():.4f}")

    # --- (e) reward-term magnitudes: zero-action vs random-action ----------
    print("\n--- (e) reward term magnitudes: zero-action vs random-action ---")

    def reward_stats(action_fn, steps=90):
        env.reset()
        sums: dict[str, float] = {}
        for t in range(steps):
            env.step(action_fn(t))
            for k, v in env._reward_terms.items():
                sums[k] = sums.get(k, 0.0) + v.mean().item()
        return {k: v / steps for k, v in sums.items()}

    zero_stats = reward_stats(lambda t: torch.zeros(n, j, device=device))
    rand_stats = reward_stats(lambda t: torch.rand(n, j, device=device) * 2.0 - 1.0)
    print(f"  {'term':24s} {'zero-action':>14s} {'random-action':>14s}")
    for k in zero_stats:
        print(f"  {k:24s} {zero_stats[k]:>14.4f} {rand_stats.get(k, float('nan')):>14.4f}")

    # --- regression check: partial-reset _cur_targets staleness fix --------
    print("\n--- regression: partial reset re-syncs _cur_targets (I24 fix) ---")
    env.reset()
    for _ in range(20):
        env.step(torch.ones(n, j, device=device))  # drive targets to the upper limit
    stale_before = env._cur_targets[0].clone()
    half = n // 2
    env_ids = torch.arange(half, device=device, dtype=torch.long)
    env._reset_idx(env_ids)
    after = env._cur_targets[env_ids]
    jp_after = env.robot.data.joint_pos[env_ids]
    max_gap = (after - jp_after).abs().max().item()
    print(f"  _cur_targets before partial reset (env 0, near upper limit): "
          f"mean={stale_before.mean().item():+.3f}")
    print(f"  max |_cur_targets - joint_pos| for the {half} reset envs right after reset: "
          f"{max_gap:.2e} (expect ~0)")

    print("\n=== diagnostics done ===", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
