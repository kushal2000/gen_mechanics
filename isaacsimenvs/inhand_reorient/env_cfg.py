"""Typed defaults for InHandReorient; cfg/task/InHandReorient.yaml overlays
them 1:1 (see ``PoseReachEnvCfg`` for the reason every field has a
non-None sentinel default).

``TerminationCfg`` is imported, not redefined: ``coevolution/utils/
rlgames_utils.py`` restores curriculum state through free functions
(``get_curriculum_state``/``set_curriculum_state``) that read
``env.cfg.termination.*`` by field name, so this cfg has to expose exactly
those names for resume to keep working -- reusing the class is the simplest
way to guarantee that.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaacsimenvs.pose_reaching_6d.env_cfg import TerminationCfg


@configclass
class AssetsCfg:
    """Which hand, and the object it reorients."""

    # A manifest hand id (hand_only.manifest_entry): "sharpa", "allegro_right",
    # "dclaw", ... . Used only when `hand_population` is empty (the default);
    # one fixed hand for the whole run.
    hand_id: str = "sharpa"
    # Phase 2: path to a grammar population JSON (population_file.write_population's
    # output). Empty (the default) keeps today's single-hand path byte-identical
    # (`hand_id` above, scene.replicate_physics=True, the regex spawner). When
    # set, `scene_utils.setup_scene` authors one design per env directly
    # (scene.replicate_physics/clone_in_fabric are hard-asserted False; see
    # `scene/author_grammar.py`).
    hand_population: str = ""
    object_size_m: float = 0.06  # AllegroHand/ShadowHand-literature range is 0.06-0.065 m
    object_density: float = 500.0
    object_pool: tuple[float, ...] = ()  # cube edge lengths; () = object_size_m only

    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.0
    object_friction: float = 0.7
    object_restitution: float = 0.0


@configclass
class ObsCfg:
    """Symmetric actor-critic for Phase 1 (no privileged state yet)."""

    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_actions",
        "object_pos_palm", "object_quat_palm", "object_lin_vel_palm", "object_ang_vel_palm",
        "goal_quat_palm", "object_quat_rel_goal", "fingertip_pos_palm",
    )
    state_list: tuple[str, ...] = obs_list
    clamp_abs_observations: float = 10.0


@configclass
class PhysicsCfg:
    contact_offset: float = 0.002
    rest_offset: float = 0.0


@configclass
class ActionCfg:
    hand_moving_average: float = 0.2
    dof_speed_scale: float = 1.0


@configclass
class RewardCfg:
    """Adopted from IsaacGymEnvs AllegroHand/ShadowHand (Makoviychuk et al.,
    2021): ``rot_rew = rot_reward_scale / (|rot_dist| + rot_eps)``,
    ``dist_rew = -dist_reward_scale * ||object_pos - goal_pos||`` (goal
    position = the object's own spawn point; see reward_utils.compute_rewards)
    plus a goal bonus, action penalty, and fall (drop) penalty. See I24,
    Phase 1c."""

    rot_reward_scale: float = 1.0
    rot_eps: float = 0.1
    dist_reward_scale: float = 10.0
    rotation_progress_scale: float = 0.0  # x (prev_rot_error - rot_error); off by default, A/B only
    goal_bonus: float = 250.0
    action_penalty_scale: float = 0.0002
    action_delta_penalty_scale: float = 0.0003
    hand_velocity_penalty_scale: float = 0.0003
    drop_penalty: float = 50.0


@configclass
class ResetCfg:
    """Where the object starts and how the goal is sampled."""

    object_spawn_offset: tuple[float, float, float] = (0.0, 0.0, 0.06)
    """Above the palm centre, in the palm's own frame."""
    object_position_noise: float = 0.01
    joint_reset_noise: float = 0.1
    goal_sampling_type: str = "absolute"  # "absolute" | "delta" | "axis"; used only if curriculum is off
    delta_rotation_degrees: float = 90.0
    drop_distance_m: float = 0.24
    """Terminate when the object is farther than this from its spawn point,
    measured in the palm frame, or falls below the palm (z < -drop_distance_m/2).
    0.24 m matches IsaacGymEnvs AllegroHand/ShadowHand's fall_dist; the
    previous 0.10 m terminated episodes on ordinary manipulation slip long
    before any rotation could happen (Phase 1b: drop fraction 1.00 -> 0.79
    over 2200 epochs, rot error unchanged from chance)."""

    goal_curriculum_enabled: bool = True
    """Stage the GOAL's difficulty (reset_utils._sample_goal's modes) instead
    of always sampling a fully random goal orientation, mirroring the
    existing angular-TOLERANCE curriculum below (I24, Phase 1c)."""
    goal_curriculum_stages: tuple[str, ...] = ("axis", "full")
    """Each stage is a ``_sample_goal`` mode: "axis" (rotate about the
    calibrated palm-normal axis only -- a 1-DOF spin task), "delta" (a random
    axis, up to delta_rotation_degrees), "full"/"absolute" (fully random)."""
    goal_curriculum_interval: int = 3000
    """Frames between eligibility checks; mirrors tolerance_curriculum_interval."""
    goal_curriculum_success_threshold: float = 3.0
    """Mean per-episode successes needed (over completed episodes) to advance
    one stage; mirrors tolerance_curriculum_success_threshold."""


def _default_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            gpu_max_rigid_contact_count=8388608,
            gpu_max_rigid_patch_count=4194304,
        ),
    )


@configclass
class InHandReorientEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 10.0
    action_space: int = 0        # 0 = derive from the hand spec in setup_scene
    observation_space: int = 1   # placeholder; setup_scene overwrites it
    state_space: int = 1

    sim: SimulationCfg = _default_sim_cfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.3, 0.3, 0.7), lookat=(0.0, 0.0, 0.5), resolution=(640, 480),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.6, replicate_physics=True, clone_in_fabric=True,
    )

    assets: AssetsCfg = AssetsCfg()
    obs: ObsCfg = ObsCfg()
    action: ActionCfg = ActionCfg()
    reward: RewardCfg = RewardCfg()
    physics: PhysicsCfg = PhysicsCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()


__all__ = [
    "InHandReorientEnvCfg", "AssetsCfg", "ObsCfg", "ActionCfg", "RewardCfg",
    "PhysicsCfg", "ResetCfg", "TerminationCfg",
]
