"""Config/registration sanity that needs no booted Kit.

Runs under .venv_isaacsim's python (gymnasium, yaml) but deliberately never
imports ``env_cfg``, ``obs_utils``, ``scene_utils``, ``reward_utils``,
``reset_utils`` or ``env`` -- ANY of those pull in ``isaaclab.*`` at module
scope, and ``isaaclab/__init__.py`` calls ``bootstrap_kernel()``
unconditionally on import (not deferred to ``AppLauncher``, despite what the
README's "construct AppLauncher before any isaaclab.* import" gotcha implies
about *when* it is safe -- it turns out no import of ``isaaclab.*`` is safe
without Kit already booted, not even ``isaaclab.utils.math``). Confirmed by
running this file's original version, which imported ``env_cfg`` directly,
under plain pytest: it crashed in collection with "Unable to bootstrap inner
kit kernel: pytest: reading from stdin while output is captured!". A
derive_spaces/env_cfg check that actually exercises isaaclab therefore has
to run inside a booted Kit process (the smoke test), not here.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import yaml

import isaacsimenvs  # noqa: F401  registers the gym ids as an import side effect

REPO_ROOT = Path(__file__).resolve().parents[3]

# What env_cfg.py's dataclasses declare, transcribed here rather than
# introspected (introspecting requires importing env_cfg, which requires Kit).
_ALLOWED_KEYS = {
    "scene": {"num_envs", "env_spacing", "replicate_physics", "clone_in_fabric"},
    "assets": {"hand_id", "object_size_m", "object_density", "object_pool",
               "modify_asset_frictions", "robot_friction", "finger_tip_friction",
               "object_friction", "object_restitution"},
    "obs": {"obs_list", "state_list", "clamp_abs_observations"},
    "action": {"hand_moving_average", "dof_speed_scale"},
    "reward": {"rot_reward_scale", "rot_eps", "dist_reward_scale", "rotation_progress_scale",
               "goal_bonus", "action_penalty_scale", "action_delta_penalty_scale",
               "hand_velocity_penalty_scale", "drop_penalty"},
    "reset": {"object_spawn_offset", "object_position_noise", "joint_reset_noise",
              "goal_sampling_type", "delta_rotation_degrees", "drop_distance_m",
              "goal_curriculum_enabled", "goal_curriculum_stages", "goal_curriculum_interval",
              "goal_curriculum_success_threshold"},
    "termination": {"episode_length", "success_tolerance", "target_success_tolerance",
                    "eval_success_tolerance", "resume_success_tolerance", "success_steps",
                    "max_consecutive_successes", "force_consecutive_near_goal_steps",
                    "tolerance_curriculum_increment", "tolerance_curriculum_interval",
                    "tolerance_curriculum_success_threshold"},
}


def test_task_is_registered():
    spec = gym.spec("GenMech-InHandReorient-Direct-v0")
    assert spec.entry_point == "isaacsimenvs.inhand_reorient.env:InHandReorientEnv"
    kwargs = spec.kwargs
    assert kwargs["env_cfg_entry_point"] == "isaacsimenvs.inhand_reorient.env_cfg:InHandReorientEnvCfg"
    assert Path(kwargs["env_cfg_yaml_entry_point"]).is_file()
    assert Path(kwargs["rl_games_sapg_cfg_entry_point"]).is_file()


def test_task_yaml_parses_and_every_key_is_on_the_cfg():
    overlay = yaml.safe_load(
        (REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())
    for section, allowed in _ALLOWED_KEYS.items():
        assert section in overlay, f"task yaml is missing the {section!r} section"
        unknown = set(overlay[section]) - allowed
        assert not unknown, f"{section}: unknown keys {unknown}, not on InHandReorientEnvCfg"


def test_task_yaml_num_envs_divides_the_sapg_block_size():
    task_cfg = yaml.safe_load(
        (REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())
    train_cfg = yaml.safe_load(
        (REPO_ROOT / "coevolution/cfg/train/InHandReorientSAPG.yaml").read_text())
    block = train_cfg["params"]["config"]["expl_coef_block_size"]
    assert task_cfg["scene"]["num_envs"] % block == 0


def test_termination_curriculum_bounds_make_sense():
    task_cfg = yaml.safe_load(
        (REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())
    term = task_cfg["termination"]
    assert term["target_success_tolerance"] < term["success_tolerance"]
    assert 0.0 < term["tolerance_curriculum_increment"] < 1.0


def test_train_yaml_names_a_valid_experiment_prefix():
    train_cfg = yaml.safe_load(
        (REPO_ROOT / "coevolution/cfg/train/InHandReorientSAPG.yaml").read_text())
    name = train_cfg["params"]["config"]["name"]
    assert name.split("_")[0].isdigit(), "vendored rl_games parses policy_idx from the prefix"
