"""Contract test: the action pipeline matches a self-contained numpy reference.

Two independent checks:

1. **Numeric agreement.** Feeds identical (normalized actions, prev targets)
   through ``apply_action_pipeline`` and through ``_reference_targets`` below —
   a direct numpy transcription of the pipeline simtoolreal's Isaac Gym env used
   (``isaacgymenvs/utils/observation_action_utils_sharpa.py::compute_joint_pos_targets``),
   re-expressed against the *live* joint limits so it is hand-agnostic. Targets
   start mid-range and actions stay small so no joint-limit clamp binds; inside
   the limits the two must agree exactly.

2. **Action routing (sparsity).** Feeds one-hot canonical actions and asserts
   that each one moves exactly the joint it addresses. This is the structural
   guard against the arm/hand index bug: the pipeline must route by
   ``_arm_joint_ids`` / ``_hand_joint_ids``, never by a positional ``[:, :7]``
   slice of a Lab-ordered buffer. A hand whose URDF does not happen to put the
   arm joints first would silently get arm treatment on hand joints.

    .venv_isaacsim/bin/python tests/test_action_pipeline.py \\
      --num_envs 4 --num_assets_per_type 1
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def _reference_targets(
    actions: "np.ndarray",       # (N, J) canonical order
    prev_targets: "np.ndarray",  # (N, J) canonical order
    lower: "np.ndarray",         # (J,) canonical order
    upper: "np.ndarray",         # (J,)
    n_arm: int,
    hand_moving_average: float,
    arm_moving_average: float,
    dof_speed_scale: float,
    dt: float,
) -> "np.ndarray":
    """Canonical-order reference: arm is a velocity-delta accumulator, hand is an
    absolute [-1, 1] rescale into the joint range; both moving-averaged."""
    import numpy as np

    cur = prev_targets.copy()

    # Hand: absolute scale from [-1, 1] into [lower, upper].
    a_hand = actions[:, n_arm:]
    lo, hi = lower[n_arm:], upper[n_arm:]
    cur[:, n_arm:] = 0.5 * (a_hand + 1.0) * (hi - lo) + lo
    cur[:, n_arm:] = (
        hand_moving_average * cur[:, n_arm:]
        + (1.0 - hand_moving_average) * prev_targets[:, n_arm:]
    )
    cur[:, n_arm:] = np.clip(cur[:, n_arm:], lo, hi)

    # Arm: integrate a velocity delta onto the previous target.
    lo, hi = lower[:n_arm], upper[:n_arm]
    cur[:, :n_arm] = prev_targets[:, :n_arm] + dof_speed_scale * dt * actions[:, :n_arm]
    cur[:, :n_arm] = np.clip(cur[:, :n_arm], lo, hi)
    cur[:, :n_arm] = (
        arm_moving_average * cur[:, :n_arm]
        + (1.0 - arm_moving_average) * prev_targets[:, :n_arm]
    )
    return cur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_assets_per_type", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robot_spec", type=str, default="sharpa_iiwa14")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np
    import torch

    import genmech  # noqa: F401  registers gym envs
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg
    from genmech.tasks.pose_reach.utils.action_utils import apply_action_pipeline

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.domain_randomization.use_action_delay = False

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    inner._replay_target_lab_order = None
    env.reset()

    p_c2l = inner._perm_canon_to_lab
    p_l2c = inner._perm_lab_to_canon
    arm_ids = list(inner._arm_joint_ids)
    hand_ids = list(inner._hand_joint_ids)
    n_arm = len(arm_ids)
    n_joints = n_arm + len(hand_ids)

    # The arm/hand id sets must partition the joints. Everything downstream
    # (action routing, reward penalties, limit tables) assumes it.
    assert sorted(arm_ids + hand_ids) == list(range(n_joints)), (
        f"arm_ids {arm_ids} and hand_ids {hand_ids} do not partition range({n_joints})"
    )
    print(f"[test] arm_joint_ids (Lab order) = {arm_ids}")
    print(f"[test] hand_joint_ids (Lab order) = {hand_ids}")

    lower_canon = inner._joint_lower_canon.detach().cpu().numpy().astype(np.float64)
    upper_canon = inner._joint_upper_canon.detach().cpu().numpy().astype(np.float64)
    mid_canon = 0.5 * (lower_canon + upper_canon)

    rng = np.random.default_rng(args.seed)
    N = args.num_envs
    dt = float(inner.step_dt)
    act_cfg = inner.cfg.action

    # ---- 1. numeric agreement over a chained rollout ----
    prev_canon = np.tile(mid_canon, (N, 1)).astype(np.float32)
    max_err = 0.0
    for step in range(args.steps):
        actions_canon = rng.uniform(-0.2, 0.2, size=(N, n_joints)).astype(np.float32)

        prev_lab = torch.tensor(prev_canon, device=inner.device)[:, p_c2l]
        inner._prev_targets = prev_lab.clone()
        inner._cur_targets = prev_lab.clone()
        apply_action_pipeline(inner, torch.tensor(actions_canon, device=inner.device))
        sim_targets_canon = inner._cur_targets[:, p_l2c].detach().cpu().numpy()

        ref = _reference_targets(
            actions=actions_canon.astype(np.float64),
            prev_targets=prev_canon.astype(np.float64),
            lower=lower_canon,
            upper=upper_canon,
            n_arm=n_arm,
            hand_moving_average=act_cfg.hand_moving_average,
            arm_moving_average=act_cfg.arm_moving_average,
            dof_speed_scale=act_cfg.dof_speed_scale,
            dt=dt,
        )

        err = float(np.abs(sim_targets_canon - ref).max())
        max_err = max(max_err, err)
        assert err < 1e-4, (
            f"step {step}: max target divergence {err:.2e}\n"
            f"sim: {sim_targets_canon[0].round(4)}\n"
            f"ref: {ref[0].round(4)}"
        )
        prev_canon = sim_targets_canon.astype(np.float32)

    print(f"[test] {args.steps} chained steps, max |sim - ref| target error = {max_err:.2e}")

    # ---- 2. action routing: one-hot in canonical slot i moves only joint i ----
    base_lab = torch.tensor(np.tile(mid_canon, (N, 1)).astype(np.float32),
                            device=inner.device)[:, p_c2l]
    for i in range(n_joints):
        onehot = np.zeros((N, n_joints), dtype=np.float32)
        # Hand actions are an absolute rescale, so 0.0 is already a large move;
        # compare each one-hot against the all-zero action to isolate slot i.
        onehot[:, i] = 1.0

        inner._prev_targets = base_lab.clone()
        inner._cur_targets = base_lab.clone()
        apply_action_pipeline(inner, torch.tensor(onehot, device=inner.device))
        hot = inner._cur_targets[:, p_l2c].detach().cpu().numpy()

        inner._prev_targets = base_lab.clone()
        inner._cur_targets = base_lab.clone()
        apply_action_pipeline(inner, torch.zeros((N, n_joints), device=inner.device))
        zero = inner._cur_targets[:, p_l2c].detach().cpu().numpy()

        delta = np.abs(hot - zero).max(axis=0)  # (J,) over envs
        moved = np.nonzero(delta > 1e-6)[0].tolist()
        assert moved == [i], (
            f"canonical action slot {i} moved joints {moved}, expected only [{i}]. "
            f"The pipeline is routing actions positionally instead of by "
            f"_arm_joint_ids/_hand_joint_ids."
        )
    print(f"[test] action routing: all {n_joints} canonical slots map 1:1 to joints")

    print("[test] action pipeline test OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
