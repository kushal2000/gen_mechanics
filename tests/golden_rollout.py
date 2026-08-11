"""Shared deterministic-rollout definition for the port-parity gate.

Imported by both ``capture_golden.py`` (which writes the baseline) and
``test_sharpa_parity.py`` (which checks against it), so the two can never drift
apart — if the rollout definition changes, both sides change together and the
stored baseline is invalidated on purpose rather than silently compared against
a different experiment.

**Import this only after ``AppLauncher`` has booted Kit.** Nothing here imports
isaaclab directly, but the env it drives does.

Determinism is engineered, not assumed:

- every domain-randomization source is disabled, and every reset-noise range
  zeroed, so resets draw nothing;
- the object pool is unshuffled, so env *i* gets the same object in both repos;
- the start pose is pinned;
- actions come from an independent seeded numpy RNG, so the action sequence is
  byte-identical regardless of env state;
- the torch RNG is re-seeded immediately before ``reset()``, so both repos
  consume it from the same point no matter how much scene construction drew.

What is left random is goal sampling, which draws from that re-seeded torch RNG
in identical order on both sides.
"""

from __future__ import annotations

from typing import Any


# Populated identically by compute_rewards in both repos.
REWARD_TERMS: tuple[str, ...] = (
    "fingertip_delta_rew",
    "lifting_rew",
    "lift_bonus_rew",
    "keypoint_rew",
    "kuka_actions_penalty",
    "hand_actions_penalty",
    "bonus_rew",
    "total_reward",
)

# Defaults define the experiment. Changing any of these invalidates a stored
# baseline; capture_golden.py records them into the npz so the test can refuse
# to compare mismatched traces.
DEFAULTS = dict(num_envs=32, num_assets_per_type=2, steps=200, seed=0, action_scale=0.3)


def build_cfg(cfg_cls, *, num_envs: int, num_assets_per_type: int) -> Any:
    """Return an env cfg with every stochastic source pinned."""
    cfg = cfg_cls()
    cfg.scene.num_envs = num_envs
    cfg.assets.num_assets_per_type = num_assets_per_type
    cfg.assets.shuffle_assets = False

    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.object_scale_noise_multiplier_range = (1.0, 1.0)
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (0.0001, 0.0001)
    dr.torque_prob_range = (0.0001, 0.0001)
    dr.object_friction_scale_range = (1.0, 1.0)
    dr.fingertip_friction_scale_range = (1.0, 1.0)

    rs = cfg.reset
    rs.reset_position_noise_x = 0.0
    rs.reset_position_noise_y = 0.0
    rs.reset_position_noise_z = 0.0
    rs.reset_dof_pos_random_interval_arm = 0.0
    rs.reset_dof_pos_random_interval_fingers = 0.0
    rs.reset_dof_vel_random_interval = 0.0
    rs.table_reset_z_range = 0.0
    rs.table_reset_xy_range_m = (0.0, 0.0)
    rs.table_reset_yaw_range_deg = 0.0
    rs.fixed_start_pose = (
        0.0, 0.0, rs.table_reset_z + rs.table_object_z_offset, 1.0, 0.0, 0.0, 0.0,
    )
    return cfg


def run_rollout(env, *, num_envs: int, steps: int, seed: int, action_scale: float,
                verbose: bool = True) -> dict:
    """Reset, drive a fixed action sequence, and return the recorded trace."""
    import numpy as np
    import torch

    inner = env.unwrapped
    inner._replay_target_lab_order = None
    n_act = int(inner.cfg.action_space)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.reset()

    actions_all = np.random.default_rng(seed).uniform(
        -action_scale, action_scale, size=(steps, num_envs, n_act)
    ).astype(np.float32)

    rec: dict[str, list] = {
        "policy_obs": [], "critic_obs": [], "reward": [],
        "terminated": [], "truncated": [], "cur_targets_canon": [],
        **{f"rew_{k}": [] for k in REWARD_TERMS},
    }

    for step in range(steps):
        action = torch.from_numpy(actions_all[step]).to(inner.device)
        obs, rew, terminated, truncated, _ = env.step(action)

        rec["policy_obs"].append(obs["policy"].detach().cpu().numpy())
        critic = obs.get("critic", obs.get("states"))
        rec["critic_obs"].append(critic.detach().cpu().numpy())
        rec["reward"].append(rew.detach().cpu().numpy())
        rec["terminated"].append(terminated.detach().cpu().numpy())
        rec["truncated"].append(truncated.detach().cpu().numpy())
        rec["cur_targets_canon"].append(
            inner._cur_targets[:, inner._perm_lab_to_canon].detach().cpu().numpy()
        )
        for k in REWARD_TERMS:
            rec[f"rew_{k}"].append(inner._reward_terms[k].detach().cpu().numpy())

        if verbose and (step + 1) % 50 == 0:
            print(f"[rollout] step {step + 1}/{steps}", flush=True)

    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in rec.items()}
    arrays["actions"] = actions_all
    return arrays


__all__ = ["REWARD_TERMS", "DEFAULTS", "build_cfg", "run_rollout"]
