"""PoseReachEnvCfg — typed defaults for the PoseReach goal-pose-reaching task.

Organized into sectioned sub-configclasses that mirror the YAML overlay
in cfg/task/PoseReach.yaml 1:1:

    sim                     → isaaclab.sim.SimulationCfg (+ PhysxCfg)
    scene                   → PoseReachSceneCfg(InteractiveSceneCfg)
    obs                     → ObsCfg
    action                  → ActionCfg
    reward                  → RewardCfg
    reset                   → ResetCfg   (includes goal sampling)
    termination             → TerminationCfg (includes tolerance curriculum)
    domain_randomization    → DomainRandomizationCfg

Values match the legacy isaacgymenvs/cfg/task/PoseReach.yaml defaults with
the following deliberate deviations (see plan file
.claude/plans/we-are-currently-in-twinkling-bengio.md):

  - `controlFrequencyInv` removed; Isaac Lab's `decimation=2` + `sim.dt=1/120`
    yields the same 60 Hz policy / 120 Hz physics as legacy `dt=1/60 +
    substeps=2`.
  - `fallDistance` / `fallPenalty` removed (unused in legacy env.py).
  - `useRelativeControl` removed (legacy True branch not being ported).
  - DR tree pruned to obs/action/object-state delays + force/torque impulses +
    object-scale & joint-vel obs noise (see DomainRandomizationCfg docstring).
  - Curricula pruned to tolerance curriculum only.

The Env class (`env.py:PoseReachEnv`) is a thin DirectRLEnv shell —
all DirectRLEnv hooks raise NotImplementedError. Phases B–H populate them.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


# ----------------------------------------------------------------------------
# scene — kept as plain InteractiveSceneCfg (num_envs + layout knobs only).
# Isaac Lab's InteractiveScene._add_entities_from_cfg iterates every field
# on the scene cfg and rejects anything that isn't an AssetBaseCfg-derived
# config, so the asset metadata (URDF paths, frictions, procedural knobs)
# must live under a sibling section — see AssetsCfg below.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# assets — URDFs, procedural-generation knobs, static per-material frictions
# ----------------------------------------------------------------------------


@configclass
class AssetsCfg:
    # The robot knob. Selects a RobotSpec from isaacsimenvs.pose_reaching_6d.scene_utils.robots.REGISTRY, which
    # carries the joint names and order, PD gains, home pose, palm and fingertip
    # geometry, and self-collision adjacency. action_space and the observation
    # dims are derived from it, so this is the only field a hand swap needs.
    robot_spec: str = "sharpa_iiwa14"
    # Optional URDF override for the selected spec. Empty means use
    # spec.urdf_path. Useful for sweeping a mount transform without editing the
    # spec; the joint set must still match, which the env asserts.
    robot_urdf: str = ""

    # Author each design's robot USD instead of converting its URDF.
    #
    # Only meaningful with a robot population. Kit's UrdfConverter takes ~1.17 s
    # per generated design on this cluster, ~90% of it re-importing the SAME
    # iiwa14 arm's 16 STL meshes -- about 8 hours for 24,576 designs before the
    # first gradient step. Authoring converts the arm ONCE, references it, and
    # writes the hand directly: ~29 ms per design, ~12 min for the population.
    #
    # Verified equivalent to the converted asset by
    # genmech.tools.compare_authored_robot: masses, inertias, joint limits,
    # colliders and every actuation property agree exactly, and driven
    # identically the fingertips land within 0.0013 mm and the joints within
    # 1.6e-05 rad. Defaults OFF so the conversion path stays the reference.
    author_robot_usds: bool = False

    table_urdf: str = "assets/urdf/table_narrow.urdf"
    # Per-env scale ranges applied to the table mesh at scene-build time.
    # Sampled independently per env: sx ~ U(table_scale_range_x), sy ~ U(table_scale_range_y).
    # Z is held at 1.0 so the table surface height does not move (the policy was
    # trained to expect it where it is). Note that height is NOT table_reset_z --
    # see the note on table_reset_z below. Default (1, 1) means no scaling
    # (legacy behavior).
    table_scale_range_x: tuple[float, float] = (1.0, 1.0)
    table_scale_range_y: tuple[float, float] = (1.0, 1.0)
    # Number of pre-baked USD variants spanning the scale range. Per env Isaac Lab
    # round-robins across this list at scene build, so each env's table has a
    # different XY footprint drawn from the configured ranges.
    table_scale_num_variants: int = 1

    object_name: str = "handle_head_primitives"
    # DexToolBench eval: when object_urdf is set, load that single URDF for
    # every env instead of generating the procedural pool. object_scale is the
    # policy-normalized grasp-bbox scale (the NAME_TO_OBJECT[...].scale
    # convention: metric bbox / object_base_size) and is required with it.
    object_urdf: str = ""
    object_scale: tuple[float, float, float] | None = None
    handle_head_types: tuple[str, ...] = (
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    )
    num_assets_per_type: int = 100
    # Author object USDs directly instead of converting their URDFs. The pool is
    # 1200 objects and conversion measured 0.2 s each (~240 s of a ~758 s scene
    # build); each object is one rigid body with two analytic shapes whose mass
    # and inertia generate_objects already computes in closed form, so the URDF
    # round-trip recovers numbers we started with.
    #
    # REVERTED TO OFF. Training diverges on this path even though every
    # equivalence check passes:
    #
    #   at epoch 3000, converted vs authored objects
    #     sharpa_iiwa14   rew 4000.09  vs   251.45
    #     gen_sharpa_like rew 5738.24  vs   220.49
    #
    #   -- 16-26x worse, sitting near the not-learning floor, with the two runs
    #   differing ONLY in this flag. Cause not yet found.
    #
    # Everything below is still true, and was still not sufficient. A pretrained
    # policy exercises a converged behaviour on a fixed distribution; training
    # explores, resets constantly and learns, and it is the only test that has
    # ever caught this. Do not re-enable without a training curve.
    #
    # The evidence that was mistaken for sufficient:
    #   * pretrained policy scores 5.01 +/- 0.09 goals on BOTH paths at 2048
    #     envs, matching on lift (79%), complete (31%) and zero (26%);
    #   * a 120-step rollout over 256 envs is bit-identical -- peak and settled
    #     velocity and robot joint_vel std agree to every printed digit;
    #   * per-env mass, inertia, object_scales and asset index are bit-exact
    #     across all 512 envs (check_object_identity).
    # Plus mass/inertia/centre of mass to ~1e-8 and resting poses to ~2e-7 m
    # (compare_object_assets, compare_object_physics).
    #
    # Note what the eval alone did NOT catch: an env->pool assignment bug that
    # cost 5.07 -> 3.00 goals while every asset comparison passed. The identity
    # check above is the one that closes that hole, and it is why this default
    # moved only after per-env identity was verified, not merely per-asset
    # equivalence.
    author_object_usds: bool = False
    # Diagnostic: author only one of the pair, to bisect which asset carries a
    # behavioural difference. "both" in normal use.
    author_which: str = "both"
    # RNG seed for the procedural object pool. Held-out object *geometry* is
    # generated by changing this: same size distributions, unseen draws. The
    # generator seeds numpy globally, so this must be an explicit knob rather
    # than inherited from the run seed -- an eval condition has to name the pool
    # it evaluates on. 42 reproduces simtoolreal's training pool.
    object_seed: int = 42
    # Multiplies every sampled handle/head density, and so every generated
    # object's mass and inertia. This is how the held-out object-mass axis is
    # reached: PhysX's runtime set_masses raises in this Isaac Lab build, but
    # density is baked into the URDF at generation time, so no runtime API is
    # needed. 1.0 is the training distribution.
    object_density_scale: float = 1.0

    # Shuffle the procedural pool after generation. Legacy default (True)
    # gives env i uniform coverage over types via i % len(pool). Debug/parity
    # runs set this False so pool[0] is the first matching distribution
    # (cuboid hammer ahead of cylinder hammer, etc.) — see
    # debug_differences/policy_rollout_isaacsim.py.
    shuffle_assets: bool = True

    # Static per-material frictions (set once at asset creation, not per-reset DR).
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5
    # Coefficient of restitution for the object material. 0.0 (fully inelastic)
    # is the training value; the object-physics eval axis raises it to test
    # policies against bouncier contacts.
    object_restitution: float = 0.0

# ----------------------------------------------------------------------------
# obs
# ----------------------------------------------------------------------------

    # --- hand population (leave the seed at -1 for a single hand) ------------
    #
    # With a population set, every env gets its own design, round-robin: env i
    # holds design i % len(population), the same rule the object pool uses. They
    # all run in ONE articulation view because ghosting pads each design to the
    # same 37-joint template -- view count tracks joint COUNT, not population
    # size. robot_spec is then ignored: a fixed hand and a generated one do not
    # share a joint set (29 vs 37), so it would build the spaces for a robot the
    # scene does not contain.
    #
    # EMPTY STRING / -1, NOT None. isaaclab's update_class_from_dict type-checks
    # a hydra override against the default's RUNTIME type, so `str | None = None`
    # rejects every string override with "Incorrect type under namespace ...
    # Expected: <class 'NoneType'>". Same reason reset.fixed_trajectory_file
    # defaults to "". -1 rather than None for the seed keeps 0 a usable seed.
    robot_population_path: str = ""
    """A population directory holding manifest.json. Takes precedence over the
    seed. Mutated generations have no seed to be named by, so they are addressed
    by path -- see hand_sampler.population.load_population_at."""

    robot_population_seed: int = -1
    """Cached population to draw from; -1 means no population, one hand."""

    robot_population_count: int = 0
    """0 means the whole cached population."""


@configclass
class ObsCfg:
    """Asymmetric actor-critic obs layout + clamping."""

    # Critic sees the full state list; actor sees the obs list subset.
    state_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "palm_vel",
        "object_rot",
        "object_vel",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist",
        "closest_fingertip_dist",
        "lifted_object",
        "progress",
        "successes",
        "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
    )

    clamp_abs_observations: float = 10.0


# ----------------------------------------------------------------------------
# action
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# physics
# ----------------------------------------------------------------------------


@configclass
class PhysicsCfg:
    """PhysX collider offsets, authored onto every collision prim.

    contact_offset is the distance at which PhysX starts generating contacts,
    rest_offset the separation it settles at. They were module constants in two
    places -- the converter path and the authoring path -- which had to agree by
    hand. They are one config value now, so the two backends cannot drift.
    """

    contact_offset: float = 0.002
    rest_offset: float = 0.0


@configclass
class ActionCfg:
    """Joint-position-target control with moving-average smoothing."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5


# ----------------------------------------------------------------------------
# reward
# ----------------------------------------------------------------------------


@configclass
class RewardCfg:
    """Four-term reward: keypoint + lifting (w/ bonus) + distance-delta +
    reach-goal bonus + action-magnitude penalties.
    """

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


# ----------------------------------------------------------------------------
# reset (includes goal sampling — both fire on _reset_idx)
# ----------------------------------------------------------------------------


@configclass
class ResetCfg:
    """Initial-state distribution + goal sampling (sampled at every reset)."""

    # Initial object pose noise
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Joint state noise on reset
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5

    # Offset the default arm pose (joint 2 -10deg, joint 4 +10deg) — matches
    # the gym env's startArmHigher, used for DexToolBench evaluation.
    start_arm_higher: bool = False

    # Table reset geometry.
    #
    # table_reset_z is the table's CENTRE, not its surface. The box in
    # table_narrow.urdf is 0.3 m tall and centred on its link origin, and
    # _reset_table_pose writes this value straight to the rigid body's z, so the
    # surface a tool rests on is table_reset_z + 0.15 ~= 0.53 m.
    #
    # Measured, because a comment above used to claim otherwise and cost real
    # time: objects reset at table_reset_z + table_object_z_offset = 0.63, fall,
    # and settle at 0.535 with zero velocity -- a 0.525 surface plus ~10 mm of
    # object half-thickness. Anything drawing or reasoning about "the table
    # surface" must add the half-height.
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    # Per-env XY position noise applied at reset (uniform half-widths in m).
    # Default (0, 0) keeps the table centered on the env origin (legacy behavior).
    table_reset_xy_range_m: tuple[float, float] = (0.0, 0.0)
    # Per-env yaw noise applied at reset (uniform half-width in degrees about z).
    # Default 0.0 preserves the identity quat (legacy behavior).
    table_reset_yaw_range_deg: float = 0.0

    # Goal sampling
    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0

    # Debug only — when set, every reset writes this exact env-local pose
    # to GoalViz instead of sampling. Format: (x, y, z, qw, qx, qy, qz).
    # Used by debug_differences/* to keep both envs visually aligned.
    fixed_goal_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Fixed-trajectory ablation: when ``fixed_trajectory_file`` is non-empty,
    # the env ignores ``goal_sampling_type`` and instead draws goal sequences
    # from a pre-generated pool of (N_total, K, 3+4) trajectories in the JSON
    # file. ``fixed_trajectory_count`` truncates the pool to the first N
    # (0 = use the whole file). Pair with ``termination.max_consecutive_
    # successes == K`` so episodes end exactly when a trajectory is exhausted.
    #
    # Empty-string / 0 defaults are deliberate: isaaclab's configclass type-
    # checks hydra overrides against the default value's *runtime* type, so a
    # ``str | None = None`` field rejects string overrides at parse time.
    fixed_trajectory_file: str = ""
    fixed_trajectory_count: int = 0
    # Index of the first trajectory to use from the pool. Together with
    # fixed_trajectory_count this slices [offset : offset + count], which is
    # what makes a train/held-out goal split possible from one committed file:
    # train on [0:800], evaluate on [800:1000]. Without it, every run would see
    # the same prefix.
    fixed_trajectory_offset: int = 0


# ----------------------------------------------------------------------------
# termination (includes tolerance curriculum — governs success criterion)
# ----------------------------------------------------------------------------


@configclass
class TerminationCfg:
    """Episode-end conditions + success-tolerance curriculum.

    The episode-length-extends-on-goal-hit behavior (legacy
    ``progress_buf[is_success > 0] = 0`` at env.py:2503-2505) lands in
    Phase F's ``_get_dones`` — there it zeros ``self.episode_length_buf``
    for envs that hit a goal, so the framework's default truncation check
    only fires on *time without progress*, not on total time in episode.
    """

    episode_length: int = 600  # steps (policy steps; 600 * decimation * dt = 10s)

    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None

    resume_success_tolerance: float = 0.0
    """Where the curriculum PICKS UP, when continuing a run. 0 starts at
    ``success_tolerance``.

    0.0 rather than None as the "unset" value, deliberately. Isaac Lab's
    ``update_class_from_dict`` type-checks an override against the CURRENT
    value's type, so a field defaulting to None rejects a float from the CLI
    with "Expected: <class 'NoneType'>" -- a hydra override of it cannot work at
    all. ``assets.robot_population_seed`` and ``assets.object_urdf`` use
    non-None sentinels for the same reason.

    Separate from ``success_tolerance`` on purpose. That field is the
    curriculum's definition -- its start AND its upper clamp -- so overloading
    it to resume a run would rewrite the record of what the curriculum was:
    ``.hydra/config.yaml`` would claim this run's curriculum began at 2.35 cm
    when it began at 7.5 cm, and the clamp would silently tighten with it. This
    field states the one thing that is actually different about a continuation,
    and leaves the curriculum itself as written.

    It exists because the tolerance does NOT survive a checkpoint: it lives on
    the env, and rl_games persists env state only through
    ``vec_env.get_env_state()``, which Isaac Lab's wrapper leaves as IVecEnv's
    stub returning None. Distinct from ``eval_success_tolerance``, which PINS
    the value every step and disables the curriculum."""

    success_steps: int = 10
    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False

    # Tolerance curriculum (the only curriculum in v1).
    tolerance_curriculum_increment: float = 0.9  # multiplicative per step
    tolerance_curriculum_interval: int = 3000  # env steps across all agents
    tolerance_curriculum_success_threshold: float = 3.0


# ----------------------------------------------------------------------------
# domain_randomization
# ----------------------------------------------------------------------------


@configclass
class DomainRandomizationCfg:
    """Sim2real DR set. Scoped to per-episode / per-step perturbations that
    the paper identifies as essential for transfer. Physics-param DR (gravity,
    DOF damping/stiffness/effort/friction/armature, rigid-body mass,
    rigid-shape friction/restitution) is *not* ported in v1.
    """

    # Obs / action latency
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3

    # Object state delay + noise on the observed object pose.
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    # Multiplicative per-env scale noise applied to keypoint offsets and to the
    # object_scales obs (legacy env.py:3093-3098,3193-3195).
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)

    # Per-step Gaussian noise on joint-velocity obs (legacy env.py:3251).
    joint_velocity_obs_noise_std: float = 0.1

    # Random force/torque impulses on the object body.
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

    # Per-env friction randomization, sampled ONCE at scene init (not at
    # reset). Multiplicative scales of the AssetsCfg base values. Default
    # (1.0, 1.0) is a no-op so existing runs are unaffected.
    #
    # Why init-only with bucketing: PhysX caps live materials at 64K and
    # set_material_properties creates a new material per distinct
    # (static, dynamic, restitution) tuple, so per-reset randomization
    # exhausts the limit in seconds. Init-only with discrete buckets caps
    # the material count at ~`friction_n_buckets` per axis.
    #
    # Mass randomization is not exposed: set_masses raises
    # "Failed to set rigid body masses in backend" in this Isaac Lab /
    # PhysX configuration. The proper path is Isaac Lab's
    # EventCfg.ActorMassRandomization, which is a larger refactor.
    object_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    fingertip_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    friction_n_buckets: int = 16


# ----------------------------------------------------------------------------
# Top-level configclass — composes the above, plus DirectRLEnvCfg requireds
# ----------------------------------------------------------------------------


def _default_sim_cfg() -> SimulationCfg:
    """60 Hz policy control / 120 Hz physics (matches legacy dt=1/60 + substeps=2)."""
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # 1 = TGS (matches legacy)
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            # Sized for 24576-env close-contact grasping (Lab defaults
            # overflow: "Patch buffer overflow detected" kills training).
            gpu_max_rigid_contact_count=16777216,
            gpu_max_rigid_patch_count=8388608,
        ),
    )


@configclass
class PoseReachEnvCfg(DirectRLEnvCfg):
    """Top-level configclass for the PoseReach goal-pose-reaching env.

    Structure mirrors ``cfg/task/PoseReach.yaml`` exactly — YAML overlay
    key paths resolve to these fields via ``configclass.from_dict``.
    """

    # --- DirectRLEnvCfg required fields ---
    decimation: int = 2  # 2 physics substeps per policy step
    episode_length_s: float = 10.0  # 600 policy steps * 2 * (1/120) = 10s
    # 0 means "derive from assets.robot_spec" (PoseReachEnv.__init__ sets it to
    # spec.num_joints). A non-zero value is checked against the spec and raises
    # on mismatch, so a stale override fails loudly instead of silently
    # truncating the action vector.
    action_space: int = 0
    # Obs/state sizes are derived from obs.obs_list / obs.state_list at env init.
    # Placeholder keeps the configclass instantiable before the env computes the
    # final spaces.
    observation_space: int = 140
    state_space: int = 140

    # --- Isaac Lab base fields ---
    sim: SimulationCfg = _default_sim_cfg()
    # Viewer is the camera DirectRLEnv.render('rgb_array') captures from. One
    # render product (omni.replicator) is lazily allocated at this prim path
    # on first render() call — single buffer, num_envs-independent. eye/lookat
    # are world-frame; with replicate_physics=False the central env sits near
    # world origin at large num_envs, so framing the table/robot here works.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.5, -1.5, 1.2),
        lookat=(0.0, 0.4, 0.5),
        resolution=(640, 480),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # Validated from-scratch training scale. Smaller counts must stay
        # divisible by the SAPG expl_coef_block_size (4096).
        num_envs=24576,
        env_spacing=1.2,
        # Per-env distinct USDs (MultiUsdFileCfg) require:
        #  - replicate_physics=False so PhysX parses each env as its own
        #    subtree (otherwise variants collapse into a single template;
        #    Isaac Lab also emits a hard warning — see
        #    isaaclab/scene/interactive_scene.py).
        #  - clone_in_fabric=False so the cloner replicates env_0 into the
        #    USD stage (not just Fabric). MultiUsdFileCfg's spawner resolves
        #    the regex prim_path via find_matching_prim_paths, which only
        #    sees USD prims; with clone_in_fabric=True env_1..env_{N-1}
        #    exist only in Fabric and the multi-asset spawn lands in env_0.
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # --- Sectioned sub-configs (mirror YAML sections 1:1) ---
    assets: AssetsCfg = AssetsCfg()
    obs: ObsCfg = ObsCfg()
    action: ActionCfg = ActionCfg()
    reward: RewardCfg = RewardCfg()
    physics: PhysicsCfg = PhysicsCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    log_morphology_layout: bool = False
    """Print the descriptor's field map at startup. Noisy otherwise."""

    include_morphology_obs: bool = True
    """Whether the morphology descriptor is in the observation at all.

    Only meaningful with a population. True is the only setting that trains a
    cross-embodied policy for its purpose, and the env ENFORCES it -- appending
    the field if a config dropped it -- because a YAML overlay silently removed
    it once and the only symptom was an observation 186 wide instead of 329.

    False exists for one experiment: the ablation asking whether the policy uses
    the descriptor or ignores it. It STRIPS the field rather than merely not
    adding it, so the task YAML overlay and ``obs_list=${env.obs.state_list}``
    cannot put it back."""



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
