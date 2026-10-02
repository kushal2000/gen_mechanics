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
    """URDFs, the procedural object pool, base frictions.

    Kept off the scene cfg because Isaac Lab rejects non-asset fields there.
    """

    # Selects the RobotSpec (joint names, gains, home pose, geometry, adjacency);
    # action_space and the observation dims derive from it.
    robot_spec: str = "sharpa_iiwa14"
    robot_urdf: str = ""  # overrides spec.urdf_path; the joint set must still match
    # A hand_sampler.HandPopulation injected in code: every env holds one of its
    # designs, and its template spec replaces robot_spec.
    robot_population: object | None = None
    # Joint friction is zero on every joint of every robot, arm and hand.
    # simtoolreal's isaacgym env set a 22-entry hand table and its own Isaac Sim
    # port dropped it; measured here (jobs 735294/735437) the table moved the
    # pretrained checkpoint from 27.780 goals/env to 27.691, a 0.089 gap on a
    # 0.24 standard error. It simply buys nothing.
    #
    # The units, once suspected of mismatching, do not: isaacgym never documents
    # them, but 21 of the 22 values are exactly 4% or 2% of that joint's URDF
    # effort limit (0.00378738 against 0.189369, to six figures), so they were
    # derived as a fraction of a torque and are torques. Isaac Sim 5.0+ applies
    # this parameter as an effort in N.m, which is the same thing. The table was
    # right; it just does not matter.
    # Or name one: robot_spec = "gen_s<seed>_n<count>" builds it from the
    # grammar. One knob, because the agent YAML interpolates the network's spec
    # from robot_spec and a second would let the two disagree.
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
    # How pool entries go to envs -- see hand_sampler.robot_spec.object_index.
    # "env_modulo" is the original rule; "design_cycle" gives every design the
    # same objects (all of them, with a pool of total_envs // n_designs).
    object_assignment: str = "env_modulo"
    # A hand-picked pool from scene_utils/objects/curated_pools.py ("diverse24")
    # instead of num_assets_per_type random draws per distribution.
    object_pool: str = ""

    # Base frictions, set once at scene init.
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5
    object_restitution: float = 0.0  # 0 in training; the object-physics eval axis raises it



@configclass
class ObsCfg:
    """Asymmetric actor-critic: the critic sees state_list, the actor obs_list."""

    state_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
        "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
        "joint_enabled", "object_keypoints_rel_joint", "palm_keypoints",
        "ee_pos", "ee_rot", "ee_vel", "object_rot", "object_vel",
        "keypoints_rel_ee", "keypoints_rel_goal", "object_scales",
        "closest_keypoint_max_dist", "closest_fingertip_dist",
        "lifted_object", "progress", "successes", "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel",
        "prev_action_targets", "joint_link_bbox", "joint_lower", "joint_upper",
        "joint_enabled", "object_keypoints_rel_joint", "palm_keypoints",
        "ee_pos", "ee_rot", "object_rot", "keypoints_rel_ee",
        "keypoints_rel_goal", "object_scales",
    )
    # Compare object and goal keypoints about their OWN centres, making
    # _keypoints_max_dist a pure rotation residual (still in metres, so the
    # tolerance curriculum and every downstream reader are unaffected). False
    # keeps the fused position+orientation metric pose reaching needs.
    orientation_only_goal: bool = False
    # Under orientation_only_goal, what the success test and the progress reward measure.
    # "keypoint": the max corner residual (axis-dependent: a fixed tolerance admits 20-35
    # deg). "angle": the true quaternion angle, as arc length r * theta at the keypoint
    # radius; the tolerance is converted back to that exact angle. Observations unchanged.
    orientation_metric: str = "keypoint"
    clamp_abs_observations: float = 10.0
    # Where joint_link_bbox and object_keypoints_rel_joint are measured from.
    # "ee": link_7's origin, metres -- one point every design shares; the palm
    # is then described by palm_keypoints alone. "palm_center": the design's
    # palm centre, / hand_scale -- what every run before 2026-09-17 trained on;
    # a saved config without this key evaluates as it trained.
    geometry_origin: str = "palm_center"
    # Every hand-relative position (tokens, keypoints_rel_ee, fingertip_pos_rel_ee, palm_keypoints) and
    # ee_rot / palm_rot in the palm's CANONICAL frame -- x grasp normal, y width, z wrist to fingertip,
    # read off palm_keypoints -- from the palm centre, in metres; ee_pos is the palm centre. Off, they
    # are in the palm body's own frame, whose axes are whatever the hand's URDF chose: the same for
    # every env of one hand, different between hands. For multi-hand scenes; supersedes geometry_origin.
    canonical_palm_frame: bool = False


@configclass
class PhysicsCfg:
    """PhysX collider offsets, authored onto every collision prim by both backends."""

    contact_offset: float = 0.002
    rest_offset: float = 0.0
    # Hand joint speed cap (rad/s) in the solver, the same for every hand joint. 0 keeps whatever
    # the converted USD carries -- the vendor URDF's <limit velocity>, which ranges 2.0 (Shadow) to
    # 15 (SHARPA) across the commercial hands and throttled the slow ones. Generated hands are 10.
    # (0 rather than None: Isaac Lab's config override refuses to put a number over a None default.)
    hand_velocity_limit: float = 0.0
    # The rest of the hand actuator, overridable the same way (0 keeps the spec's / USD's value): lets a
    # hand whose dynamics come from elsewhere -- a generated population -- run on the uniform-dynamics
    # actuator (assets/urdf/unified_dynamics_commercial_hands: 0.5 N.m, 3.0, 0.0775, 0.00058).
    hand_effort_limit: float = 0.0
    hand_stiffness: float = 0.0
    hand_damping: float = 0.0
    hand_armature: float = 0.0


@configclass
class ActionCfg:
    """Joint-position targets with moving-average smoothing."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    # Hand joint units, in the observation AND the action.
    #   "range"    (default, every run so far): observed joint angles in radians, and the action an
    #              absolute map of [-1, 1] onto each joint's OWN [lower, upper] -- so the same action is a
    #              different angle on every joint and every hand.
    #   "physical" one scale for every joint of every hand: angles (joint_pos, prev_joint_pos,
    #              prev_action_targets, joint_lower, joint_upper) observed as rad / pi, velocities as
    #              (rad/s) / JOINT_VEL_SCALE, and the action target = pi * a radians, clipped to the joint's
    #              range. With the canonical joint conventions (unified_dynamics_commercial_hands) the same
    #              number is then the same motion on every hand. Pair it with the network's
    #              raw_token_fields so the running normaliser does not re-scale these per batch.
    joint_units: str = "range"
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
    # Dense, always-positive orientation-proximity term, IsaacLab's shape: scale / (x + eps)
    # per step, x = keypoint residual / keypoint radius (so x ~ angle for small rotations).
    # 0 disables it, which is every run before 2026-09-29. Why it exists: our progress term
    # pays nothing for HOLDING a cube it cannot currently improve on, and measured on 585759
    # a policy that drops the cube earns ~2x the return of one that holds it (130-150 vs ~60
    # per episode), so PPO learns possession by epoch 50 and then unlearns it.
    orientation_proximity_scale: float = 0.0
    orientation_proximity_eps: float = 0.1
    # Charged once, on the step the object is dropped (termination.fall). 0 disables it, which
    # is every run before 2026-09-29. Why: on 585759 a policy that drops the cube earns ~2x the
    # return of one that holds it, because every reset issues a fresh start + goal = a fresh
    # keypoint-progress budget that a tumbling cube harvests quickly (plus accidental 20 deg
    # goals). A bonus of 250, a dense proximity term and gamma 0.998 each only DELAYED the
    # resulting collapse; this makes the drop itself cost, independent of the horizon.
    fall_penalty: float = 0.0

    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003


@configclass
class ResetCfg:
    """Initial-state noise and goal sampling, both applied at every reset."""

    # Object pose noise.
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    # Place the object in the palm at reset instead of above the table, and
    # latch _lifted_object so the keypoint reward is live from step 0.
    object_in_hand: bool = False
    # Jitter on that in-palm placement. Small: the table-placement noise above
    # is 0.1 m, which would put the object outside the hand entirely.
    in_hand_position_noise: float = 0.005
    # Gap between the palm face and the object at reset. A few mm is under two
    # policy steps of free fall, so no settle phase is needed, and it keeps the
    # object from starting in contact with a finger it is meant to be caged by.
    in_hand_clearance: float = 0.003
    # Where the object starts: on the palm face, or resting on the FINGERTIPS.
    # The fingertips are the grippy surface (finger_tip_friction 1.5 against the
    # palm's 0.5, so mu 1.25 vs 0.75 combined with the object), and putting the
    # object on them means the FINGERS hold it rather than the slab -- which is
    # the task. It also makes the tilt largely irrelevant, since the support is
    # no longer a plate.
    in_hand_placement: str = "palm"          # "palm" | "fingertips" | "palm_body_offset"
    # Only read when in_hand_placement == "fingertips": how far out from the palm
    # centre toward the fingertip centroid the object starts. 0 reproduces the
    # "palm" placement exactly, 1 rests it on the tips. The clearance over the
    # palm slab is interpolated with it, so intermediate values are a real
    # position rather than a blend of two formulas.
    in_hand_fingertip_fraction: float = 1.0
    # Extra gap above whatever surface is under the object at reset, on top of
    # in_hand_clearance. Only read for the "fingertips" placement. Non-zero means
    # the object is DROPPED onto the fingers rather than starting in contact with
    # them, which is what keeps it out of penetration for every orientation draw.
    in_hand_drop_margin: float = 0.0
    # Only read when in_hand_placement == "palm_body_offset": (x, y, bottom_height) in
    # metres, in the palm BODY's frame. See reset.py for the exact convention.
    in_hand_palm_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
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

    # "Dropped" as a distance from the palm rather than an absolute floor
    # height: a fixed palm-up hand has no floor to fall below. None keeps the
    # object_z_local < 0.1 test pose reaching uses.
    drop_distance_m: float | None = None
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

    # The env enforces True (a YAML overlay
    # once dropped it silently); False is the ablation and strips the field.


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
