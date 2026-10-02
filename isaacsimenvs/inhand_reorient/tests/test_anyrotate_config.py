"""The anyrotate task and train configs carry AnyRotate's numbers (CoRL 2024:
Sec. 3.1/4, App. B, F, Tables 4, 5, 9). Kit-free (YAML only)."""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import pytest
import yaml

import isaacsimenvs  # noqa: F401  registers the task

REPO_ROOT = Path(__file__).resolve().parents[3]


def _task():
    return yaml.safe_load((REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())


def _train(name):
    return yaml.safe_load((REPO_ROOT / "coevolution/cfg/train" / name).read_text())["params"]


def test_sim_rates_and_episode():
    a = _task()["anyrotate"]
    assert a["sim_dt"] == pytest.approx(1 / 60)
    assert 1.0 / (a["sim_dt"] * a["decimation"]) == pytest.approx(20.0)
    assert math.ceil(a["episode_length_s"] / (a["sim_dt"] * a["decimation"])) == 600


def test_reward_weights_and_constants():
    a = _task()["anyrotate"]
    weights = {k: a[f"w_{k}"] for k in ("kp", "rot", "goal", "gc", "bc", "omega", "pose", "work", "torque", "penalty")}
    assert weights == dict(kp=1.0, rot=5.0, goal=10.0, gc=0.1, bc=0.2, omega=0.5, pose=0.5, work=0.1, torque=0.05,
                           penalty=50.0)
    assert (a["kp_a"], a["kp_b"], a["keypoint_distance_m"], a["rot_clip"]) == (50.0, 2.0, 0.05, 0.025)
    assert (a["d_tol"], a["d_max"], a["axis_dev_max_deg"], a["goal_increment_deg"]) == (0.15, 0.1, 45.0, 30.0)
    assert (a["omega_max"], a["curriculum_g_min"], a["curriculum_g_max"]) == (0.6, 1.0, 2.0)
    assert a["action_scale"] == 0.026


def test_touch_and_randomisation_constants():
    a = _task()["anyrotate"]
    assert (a["contact_threshold"], a["force_alpha"], a["force_beta"], a["force_max"]) == (0.25, 0.5, 0.6, 5.0)
    assert (a["pose_beta"], a["pose_max"]) == (0.6, 0.53)
    assert a["mass_range"] == [0.025, 0.2] and a["com_range"] == 0.01
    assert 0.025 <= a["capsule_radius"] <= 0.034 and 0.0 <= a["capsule_width"] <= 0.012
    assert 0.045 <= a["box_size"] <= 0.06
    assert (a["static_friction"], a["dynamic_friction"]) == (10.0, 10.0)
    assert (a["joint_noise"], a["tip_pos_noise"], a["tip_quat_noise"]) == (0.03, 0.005, 0.01)
    assert (a["contact_pose_noise"], a["contact_force_noise"]) == (0.0174, 0.1)
    assert a["hand_disable_gravity"] is False  # Sec. 4: gravity on hand and object
    assert a["hand_orientation_randomization"] is False and a["gravity_curriculum"] is False


def test_train_yaml_is_table_5_teacher():
    p = _train("InHandAnyRotatePPO.yaml")
    c = p["config"]
    assert p["network"]["mlp"]["units"] == [512, 256, 128] and p["network"]["mlp"]["activation"] == "elu"
    assert (c["learning_rate"], c["horizon_length"], c["minibatch_size"], c["mini_epochs"]) == (5e-3, 8, 32768, 5)
    assert (c["gamma"], c["tau"], c["e_clip"], c["kl_threshold"], c["grad_norm"]) == (0.99, 0.95, 0.2, 0.02, 1.0)
    assert c["name"].split("_")[0].isdigit()


def test_population_train_yaml_adds_only_the_logstd_bound():
    base, pop = _train("InHandAnyRotatePPO.yaml"), _train("InHandAnyRotatePopPPO.yaml")
    assert pop["network"]["name"] == "inhand_actor_critic"
    assert pop["network"]["space"]["continuous"]["logstd_max"] == 0.0
    for key, value in base["config"].items():
        if key not in ("name", "full_experiment_name"):
            assert pop["config"][key] == value, key


def test_entry_points_registered_and_default_profile_is_legacy():
    kwargs = gym.spec("GenMech-InHandReorient-Direct-v0").kwargs
    assert Path(kwargs["rl_games_anyrotate_ppo_cfg_entry_point"]).name == "InHandAnyRotatePPO.yaml"
    assert Path(kwargs["rl_games_anyrotate_pop_ppo_cfg_entry_point"]).name == "InHandAnyRotatePopPPO.yaml"
    assert _task()["task_profile"] == "legacy"


def test_grasp_cache_is_off_by_default_with_hora_and_anyrotate_numbers():
    """App. C (AnyRotate) and HORA's grasp generation; off unless a cache
    path is given, so the drop reset stays the default."""
    a = _task()["anyrotate"]
    assert a["grasp_cache"] == "" and a["grasp_cache_generate"] is False
    assert a["grasp_joint_sample_noise"] == 0.3  # App. C: U(-0.3, 0.3) rad
    assert a["grasp_min_tip_contacts"] == 2 and a["grasp_max_tip_dist_m"] == 0.1  # HORA
    assert a["grasp_max_nontip_contacts"] == -1 and a["grasp_max_mean_tip_dist_m"] == -1.0  # AnyRotate tests: opt-in
    assert a["grasp_reset_joint_noise"] == 0.0 and a["grasp_reset_obj_pos_noise"] == 0.0  # neither adds noise
    assert a["grasp_gravity_cycle"] is False and a["grasp_settle_steps"] == 0
    assert 3.0 <= a["grasp_hold_s"] <= 6.0  # HORA 3.3 s, AnyRotate 6 s
