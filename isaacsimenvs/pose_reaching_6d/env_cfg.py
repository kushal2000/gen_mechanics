"""Typed defaults for the PoseReach task; cfg/task/PoseReach.yaml overlays them 1:1.

Values follow the legacy isaacgymenvs task, with ``decimation=2`` and
``sim.dt=1/120`` standing in for its ``dt=1/60, substeps=2``.

Sentinels are "" / -1 / 0.0, never None: isaaclab type-checks a hydra override
against the default's runtime type, so a None default rejects every override.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


@configclass
class AssetsCfg:
    """URDFs, the procedural object pool, the hand population, base frictions.

    Kept off the scene cfg because Isaac Lab rejects non-asset fields there.
    """

    # Selects the RobotSpec (joint names, gains, home pose, geometry, adjacency);
    # action_space and the observation dims derive from it.
    robot_spec: str = "sharpa_iiwa14"
    robot_urdf: str = ""  # overrides spec.urdf_path; the joint set must still match
    table_urdf: str = "assets/urdf/table_narrow.urdf"

    handle_head_types: tuple[str, ...] = (
        "hammer", "screwdriver", "marker", "spatula", "eraser", "brush",
    )
    num_assets_per_type: int = 100
    # Pool RNG seed; a different seed is how held-out geometry is made. 42 is simtoolreal's pool.
    object_seed: int = 42
    # Multiplies every sampled density, hence mass and inertia. Baked into the
    # URDF because runtime set_masses raises in this build.
    object_density_scale: float = 1.0
    # Shuffle so env i % len(pool) covers types uniformly; parity runs set False.
    shuffle_assets: bool = True

    # Base frictions, set once at scene init.
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5
    object_restitution: float = 0.0  # 0 in training; the object-physics eval axis raises it

    # Hand population: env i holds design i % len(population), all in one
    # articulation view (ghosting pads every design to one joint template), and
    # robot_spec is ignored. Path wins over seed; -1 / "" mean one fixed hand.
    robot_population_path: str = ""  # directory holding manifest.json
    robot_population_seed: int = -1
    robot_population_count: int = 0  # 0 = the whole population
    # A RobotPopulation injected in code (coevolution evaluates one member); not settable from hydra.
    robot_population: object | None = None


@configclass
class ObsCfg:
    """Asymmetric actor-critic: the critic sees state_list, the actor obs_list."""

    state_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
        "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
        "joint_enabled", "object_keypoints_rel_joint", "hand_scale",
        "palm_pos", "palm_rot", "palm_vel", "object_rot", "object_vel",
        "keypoints_rel_palm", "keypoints_rel_goal", "object_scales",
        "closest_keypoint_max_dist", "closest_fingertip_dist",
        "lifted_object", "progress", "successes", "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
        "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
        "joint_enabled", "object_keypoints_rel_joint", "hand_scale",
        "palm_pos", "palm_rot", "object_rot", "keypoints_rel_palm",
        "keypoints_rel_goal", "object_scales",
    )
    clamp_abs_observations: float = 10.0


@configclass
class PhysicsCfg:
    """PhysX collider offsets, authored onto every collision prim by both backends."""

    contact_offset: float = 0.002
    rest_offset: float = 0.0


@configclass
class ActionCfg:
    """Joint-position targets with moving-average smoothing."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5


@configclass
class RewardCfg:
    """Keypoint + lifting (with bonus) + distance delta + reach-goal bonus, minus action penalties."""

    keypoint_rew_scale: float = 200.0
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size: tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    fixed_size_keypoint_reward: bool = True

    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15

    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0

    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003


@configclass
class ResetCfg:
    """Initial-state noise and goal sampling, both applied at every reset."""

    # Object pose noise.
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Joint state noise.
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5
    start_arm_higher: bool = False  # joint 2 -10 deg, joint 4 +10 deg (DexToolBench eval)

    # Table. table_reset_z is the box CENTRE; the 0.3 m box puts the surface at +0.15.
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    table_reset_xy_range_m: tuple[float, float] = (0.0, 0.0)  # uniform half-widths
    table_reset_yaw_range_deg: float = 0.0  # uniform half-width about z

    # Goal sampling.
    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0
    # Debug: every reset writes this env-local (x, y, z, qw, qx, qy, qz) to GoalViz.
    fixed_goal_pose: tuple[float, float, float, float, float, float, float] | None = None


@configclass
class TerminationCfg:
    """Episode end and the success-tolerance curriculum. A goal hit zeros the
    episode clock, so truncation fires on time without progress."""

    episode_length: int = 600  # policy steps; 10 s at 60 Hz

    success_tolerance: float = 0.075  # curriculum start and upper clamp
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None  # pins the tolerance, disables the curriculum
    # Where a continued run's curriculum picks up (0 = success_tolerance). Kept
    # separate so the run record still states the curriculum as written; the
    # tolerance lives on the env and does not survive an rl_games checkpoint.
    resume_success_tolerance: float = 0.0

    success_steps: int = 10
    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False

    tolerance_curriculum_increment: float = 0.9  # multiplicative
    tolerance_curriculum_interval: int = 3000  # env steps
    tolerance_curriculum_success_threshold: float = 3.0


@configclass
class DomainRandomizationCfg:
    """Per-episode and per-step perturbations, plus init-time friction buckets."""

    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3

    # Delay and noise on the observed object pose.
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)  # per env
    joint_velocity_obs_noise_std: float = 0.1  # per step

    # Random wrench impulses on the object.
    force_scale: float = 20.0
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True

    torque_scale: float = 2.0
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True

    # Per-env friction scales, sampled once at scene init onto n_buckets values:
    # PhysX caps materials at 64K and every distinct tuple is a new one. Mass DR
    # is not exposed (set_masses raises in this build).
    object_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    fingertip_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    friction_n_buckets: int = 16


def _default_sim_cfg() -> SimulationCfg:
    """120 Hz physics, 60 Hz policy."""
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # TGS
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            # Sized for 24576-env grasping; Lab defaults overflow the patch buffer.
            gpu_max_rigid_contact_count=16777216,
            gpu_max_rigid_patch_count=8388608,
        ),
    )


@configclass
class PoseReachEnvCfg(DirectRLEnvCfg):
    """Top-level config; cfg/task/PoseReach.yaml key paths map onto these fields."""

    decimation: int = 2
    episode_length_s: float = 10.0
    # 0 = derive from the robot spec in setup_scene; a stale non-zero value raises.
    action_space: int = 0
    # Placeholders; setup_scene derives the real widths from obs.obs_list / state_list.
    observation_space: int = 778
    state_space: int = 800

    sim: SimulationCfg = _default_sim_cfg()
    # Camera for render('rgb_array'), world frame, framed on the central env.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.5, -1.5, 1.2), lookat=(0.0, 0.4, 0.5), resolution=(640, 480),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=24576,  # smaller counts must divide by SAPG's expl_coef_block_size (4096)
        env_spacing=1.2,
        # Per-env distinct assets need both: PhysX parses each env as its own
        # subtree, and the env prims exist in USD for the regex spawn to find.
        replicate_physics=False,
        clone_in_fabric=False,
    )

    assets: AssetsCfg = AssetsCfg()
    obs: ObsCfg = ObsCfg()
    action: ActionCfg = ActionCfg()
    reward: RewardCfg = RewardCfg()
    physics: PhysicsCfg = PhysicsCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    log_morphology_layout: bool = False  # print the descriptor's field map at startup
    # Only meaningful with a population. The env enforces True (a YAML overlay
    # once dropped it silently); False is the ablation and strips the field.
    include_morphology_obs: bool = True


__all__ = [
    "PoseReachEnvCfg",
    "AssetsCfg",
    "ObsCfg",
    "ActionCfg",
    "RewardCfg",
    "PhysicsCfg",
    "ResetCfg",
    "TerminationCfg",
    "DomainRandomizationCfg",
]
