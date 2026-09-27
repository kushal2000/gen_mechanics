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
    # "dclaw", ... . One fixed hand for the whole run in Phase 1 -- no
    # population/robot_spec-by-name indirection yet (that is Phase 2's
    # grammar-to-simulator adapter).
    hand_id: str = "sharpa"
    object_size_m: float = 0.055  # one size to start; kept pluggable (object_pool below)
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
        "object_pos_palm", "object_quat_palm", "goal_quat_palm", "object_quat_rel_goal",
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
    rotation_progress_scale: float = 1.0  # x (prev_rot_error - rot_error), radians
    goal_bonus: float = 250.0
    action_penalty_scale: float = 0.0003
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
    goal_sampling_type: str = "absolute"  # "absolute" | "delta"
    delta_rotation_degrees: float = 90.0
    drop_distance_m: float = 0.10
    """Terminate when the object is farther than this from its spawn point,
    measured in the palm frame, or falls below the palm (z < -drop_distance_m/2)."""


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
