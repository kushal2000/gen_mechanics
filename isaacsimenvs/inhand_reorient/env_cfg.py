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
    goal_curriculum_stages: tuple[str, ...] = ("axis",)
    """Each stage is a ``_sample_goal`` mode: "axis" (rotate about the
    calibrated palm-normal axis only -- a 1-DOF spin task), "delta" (a random
    axis, up to delta_rotation_degrees), "full"/"absolute" (fully random).

    Default is axis-only (I26): a 60-min SHARPA run showed full random-
    orientation reorientation is not learnable at ~1 GPU-hour, while the
    palm-normal-axis spin is. "full" stays available by config, e.g.
    ``env.reset.goal_curriculum_stages='[axis,full]'`` on the CLI -- when more
    than one stage is configured, ``goal_curriculum.update_goal_curriculum``
    never advances the goal stage on the same step the tolerance curriculum
    advances (see that module's docstring for the decoupling rule: the goal
    curriculum only advances once the tolerance curriculum has reached its
    floor, using the goal curriculum's own threshold/interval below)."""
    goal_curriculum_interval: int = 3000
    """Frames between eligibility checks (once the tolerance curriculum is at
    its floor -- see goal_curriculum_stages above); mirrors
    tolerance_curriculum_interval."""
    goal_curriculum_success_threshold: float = 3.0
    """Mean per-episode successes needed (over completed episodes) to advance
    one stage; mirrors tolerance_curriculum_success_threshold."""


@configclass
class ReposeCfg:
    """The ``isaaclab_repose`` task profile's numbers: Isaac Lab 2.3.2's
    ``Isaac-Repose-Cube-Allegro-Direct-v0`` (``AllegroHandEnvCfg``, the
    ``ALLEGRO_HAND_CFG`` asset and the DexCube USD), field names as theirs
    where they have one. Read only when ``task_profile ==
    "isaaclab_repose"``; ``repose_profile.apply_profile_to_cfg`` copies the
    env-level ones (decimation, dt, episode length, physics material) onto
    the cfg before the sim is built, so override them here
    (``env.repose.decimation=...``), not at the top level."""

    # --- env / sim (AllegroHandEnvCfg) ---
    decimation: int = 4
    episode_length_s: float = 10.0
    sim_dt: float = 1.0 / 120.0
    static_friction: float = 1.0
    """sim.physics_material: the default material of every body with no
    material of its own (the DexCube's binding points at a prim that does
    not exist, so the cube uses it too)."""
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    bounce_threshold_velocity: float = 0.2

    # --- object: DexCube (a 0.06 m collision cube) spawned at scale 1.2 ---
    object_size_m: float = 0.072
    object_density: float = 400.0
    object_mass_kg: float = 0.216
    """> 0 sets the cube's mass directly (overrides density). 0.216 kg is
    NVIDIA's effective mass: the DexCube USD authors ``physics:mass = 0.216``,
    which takes precedence over the configured density (measured in their
    running env with ``root_physx_view.get_masses()``; density 400 alone
    would give 0.149 kg)."""
    object_contact_offset: float = 0.001
    object_rest_offset: float = 0.0
    object_torsional_patch_radius: float = 0.1
    object_min_torsional_patch_radius: float = 0.008
    object_solver_position_iterations: int = 8
    object_solver_velocity_iterations: int = 0
    object_sleep_threshold: float = 0.005
    object_stabilization_threshold: float = 0.0025
    object_max_depenetration_velocity: float = 1000.0
    object_enable_gyroscopic_forces: bool = True

    # --- hand rigid-body / articulation props (ALLEGRO_HAND_CFG) ---
    hand_disable_gravity: bool = True
    hand_angular_damping: float = 0.01
    hand_max_depenetration_velocity: float = 1000.0
    hand_solver_position_iterations: int = 8
    hand_solver_velocity_iterations: int = 0
    hand_sleep_threshold: float = 0.005
    hand_stabilization_threshold: float = 0.0005

    # --- hand actuator, every joint of a single hand (negative: keep the
    # hand's own value). Defaults: NVIDIA's Allegro (ALLEGRO_HAND_CFG
    # stiffness 3.0, damping 0.1, friction 0.01, effort 0.5 N m; armature 0
    # and velocity limit 2 pi rad/s from its USD). Measured 2026-10-01: with
    # the hands' own gains, allegro_right learned half as fast and SHARPA
    # (up to 3.3 N m) knocked the cube off within 6 steps and did not learn
    # in 10 min. The grammar-population path keeps its designs' own gains.
    hand_stiffness: float = 3.0
    hand_damping: float = 0.1
    hand_armature: float = 0.0
    hand_joint_friction: float = 0.01
    hand_effort_limit: float = 0.5
    hand_velocity_limit: float = 6.283

    collision_from_visuals: bool = False
    """Collide through convex hulls of the hand's visual meshes instead of
    its URDF collision elements (``collision_from_visuals.py``), as NVIDIA's
    Allegro USD does, for every single hand. A hand's entry in
    ``hand_pose_file`` can also turn it on for that hand alone
    (allegro_right's does: its URDF collides through boxes and 12 mm tip
    spheres). Single-hand path only."""

    hand_asset: str = ""
    """"" (default): ``assets.hand_id`` from our manifest. "isaaclab_allegro":
    NVIDIA's Allegro USD and ALLEGRO_HAND_CFG, unchanged (fetched from their
    asset server), to separate asset effects from env effects."""

    # --- hand pose ---
    hand_pose_file: str = "repose_hand_poses.json"
    """Per-hand pose overrides (``palm_calibration.REPOSE_HAND_POSES_PATH``;
    relative paths are relative to this package). A hand with an entry
    there is placed as that entry says; any other hand keeps its palm-up
    calibration. Empty: palm-up calibration for every hand."""

    # --- reset ---
    reset_position_noise: float = 0.01
    reset_dof_pos_noise: float = 0.2
    reset_dof_vel_noise: float = 0.0
    in_hand_pos_offset: tuple[float, float, float] = (0.0, 0.0, -0.04)
    """Position target = the (noise-free) spawn point plus this, in WORLD
    axes (NVIDIA: ``in_hand_pos = default object pos; in_hand_pos[:, 2] -=
    0.04``)."""

    # --- reward ---
    dist_reward_scale: float = -10.0
    rot_reward_scale: float = 1.0
    rot_eps: float = 0.1
    action_penalty_scale: float = -0.0002
    reach_goal_bonus: float = 250.0
    fall_penalty: float = 0.0
    fall_dist: float = 0.24
    success_tolerance: float = 0.2
    max_consecutive_success: int = 0
    av_factor: float = 0.1

    # --- observation / control ---
    vel_obs_scale: float = 0.2
    act_moving_average: float = 1.0


@configclass
class AnyRotateCfg:
    """The ``anyrotate`` task profile: multi-axis in-hand object rotation,
    after M. Yang et al., "AnyRotate", CoRL 2024 (arXiv 2405.07391v3). Paper
    references in brackets; "adapt." marks a departure (see
    ``anyrotate_profile.py`` and the evolution README's table). Read only
    when ``task_profile == "anyrotate"``."""

    # --- env / sim [Sec. 4: dt 1/60, 20 Hz control, 600 steps = 30 s] ---
    sim_dt: float = 1.0 / 60.0
    decimation: int = 3
    episode_length_s: float = 30.0
    static_friction: float = 10.0  # [Table 4: object and hand friction 10.0]
    dynamic_friction: float = 10.0
    restitution: float = 0.0

    # --- task [Sec. 3.1, App. B.1, Table 9] ---
    axis_sampling: str = "sphere"
    """"sphere" (uniform on S^2; the paper trains on "arbitrary" axes without
    giving the distribution), "principal" (+-x/y/z) or "z" (palm normal)."""
    axis_curriculum_z_first: bool = False
    """adapt.: start with the z axis only, switch to ``axis_sampling`` once
    completed episodes average ``axis_curriculum_rotations`` rotations."""
    axis_curriculum_rotations: float = 0.5
    z_axis_frame: str = "palm"
    """What the "z" axis (``axis_sampling: z`` and the z-first stage) is:
    "palm" (default, as before): the palm body's +z; "world_up": world +z in
    the palm frame, the palm normal of the palm-up hand (the paper's z).
    They differ for allegro_right under NVIDIA's pose, whose palm +z runs
    along the fingers (horizontal)."""
    goal_increment_deg: float = 30.0  # [Table 9, theta = 30 deg]
    goal_advance: str = "reach"
    """"reach" (the paper: a new goal only when the pending one is reached;
    the "regular intervals" of Sec. 3.1 are the angular increment, Table 9).
    "timer" (A/B option): a goal pending ``goal_timer_s`` advances by the
    increment from where it was, so a stalled object falls behind."""
    goal_timer_s: float = 1.5
    goal_tol_metric: str = "rotation_rad"
    """adapt.: see anyrotate_profile.goal_reached (d_tol = 0.15 cannot be a
    keypoint distance in metres). "kp_dist_m" applies it literally."""
    d_tol: float = 0.15  # [Table 5, teacher]
    keypoint_distance_m: float = 0.05  # [App. B.1: N = 6 keypoints, 5 cm]
    kp_a: float = 50.0
    kp_b: float = 2.0
    kp_scale: float = 1.0  # the undefined numerator d_kp of Eq. 3
    rot_clip: float = 0.025  # [Eq. 4, c1]

    # --- reward weights [App. B.1] ---
    w_kp: float = 1.0
    w_rot: float = 5.0
    w_goal: float = 10.0
    w_gc: float = 0.1
    w_bc: float = 0.2
    w_omega: float = 0.5
    w_pose: float = 0.5
    w_work: float = 0.1
    w_torque: float = 0.05
    w_penalty: float = 50.0
    omega_max: float = 0.6  # [Eq. 8]
    reward_curriculum: bool = True  # [App. B.3, lambda_rew]
    curriculum_g_min: float = 1.0
    curriculum_g_max: float = 2.0
    curriculum_ema: float = 0.01
    """Weight of each completed episode in the running mean of goals per
    episode (g_eval); the paper does not say how g_eval is averaged."""

    # --- terminations [Eq. 12] ---
    d_max: float = 0.1
    axis_dev_max_deg: float = 45.0
    axis_check_grace_steps: int = 10
    """adapt.: a settle phase for the drop reset (envs reset from a cached
    grasp use ``grasp_settle_steps``): for this many steps after a reset the off-axis test is off, and at its
    end the axis-tilt reference and the first goal are re-made from the
    settled object (``anyrotate_profile.axis_tilt``). The episode's rotation
    about k (the Rot metric and the fitness) counts only after it."""

    # --- action [Sec. 3.1] ---
    action_scale: float = 0.026  # Delta theta in [-0.026, 0.026] rad
    action_eta: float = 0.5  # adapt.: the EMA coefficient eta is not given

    # --- simulated touch [App. F] ---
    contact_threshold: float = 0.25  # N, Eq. 16
    force_alpha: float = 0.5  # Eq. 17
    force_beta: float = 0.6  # Eq. 18
    force_max: float = 5.0
    pose_beta: float = 0.6  # Eq. 19
    pose_max: float = 0.53
    contact_data_per_prim: int = 32
    """Contact-point buffer per sensor body and env (Isaac Lab's
    max_contact_data_count_per_prim). 4 overflowed (CUDA device assert) for
    SHARPA's mesh colliders at 8192 envs."""

    # --- object [Sec. 4, Table 4] ---
    object_shape: str = "box"
    """adapt.: one shape per run ("box" or "capsule"), not the paper's mixed
    capsule/box set with per-episode dimensions. Box by default: without the
    paper's grasp cache the object is dropped onto the open palm, and a
    capsule (radius 3 cm, nearly a ball) rolls off it (Kit check, 16 envs,
    zero actions: 90 drops in 6 s vs 0 for the box)."""
    capsule_radius: float = 0.0295  # Table 4 range [0.025, 0.034], midpoint
    capsule_width: float = 0.006  # [0.000, 0.012]
    box_size: float = 0.0525  # [0.045, 0.06]
    mass_range: tuple[float, float] = (0.025, 0.2)
    com_range: float = 0.01
    object_contact_offset: float = 0.002
    object_rest_offset: float = 0.0

    # --- domain randomisation of observations [Table 4] ---
    obs_noise: bool = True
    joint_noise: float = 0.03
    tip_pos_noise: float = 0.005
    tip_quat_noise: float = 0.01
    contact_pose_noise: float = 0.0174
    contact_force_noise: float = 0.1

    # --- reset ---
    reset_joint_noise: float = 0.1
    """Drop reset (no ``grasp_cache``, or a design without a stable grasp):
    the object dropped onto the palm-up hand from its spawn point, joints at
    the canonical pose plus U(-noise, noise). The paper starts episodes from
    a cache of stable grasps (App. C; ``grasp_cache`` below)."""
    hand_orientation_randomization: bool = False
    """The paper samples the hand orientation per episode; the team keeps a
    stationary palm-up hand (not implemented: True raises)."""

    # --- stable-grasp cache [App. C; HORA, CoRL 2022] (grasp_cache.py) ---
    grasp_cache: str = ""
    """Path of a grasp cache (.npz, ``grasp_cache.py``). Empty (default):
    no cache, the object is dropped onto the hand as above. Set: every
    reset starts from a cached stable grasp of the env's design (joint
    positions, the PD targets that held it, object pose in the palm frame;
    HORA's reset), the pose penalty's q0 is that episode's initial joint
    pose (HORA's ``init_pose_buf``), and a design without a stable grasp
    keeps the drop reset but scores 0 (``design_scoring``)."""
    grasp_cache_generate: bool = False
    """Generate grasps at env start for designs the cache lacks, and save
    them into it (``grasp_cache_gen.generate``). False: a missing design
    raises."""
    grasp_cache_prune: bool = False
    """Save only this env's designs back into the cache (the evolution
    driver: every elite is in every generation's population)."""
    grasp_per_design: int = 1000
    """Grasps kept per design (AnyRotate: 10000 per object; HORA: 50000 per
    object scale)."""
    grasp_gen_max_rounds: int = 60
    grasp_gen_max_minutes: float = 15.0
    grasp_gen_rounds_without_grasp: int = 15
    """A design with no stable grasp after this many rounds is non-viable
    (15 rounds at 128 envs per design: 1920 candidates; a design with a
    0.2% yield then has a 2% chance of being declared non-viable, 22% at
    6 rounds)."""
    grasp_gen_seed: int = 0
    grasp_hold_s: float = 3.0
    """Hold with the PD targets fixed (HORA: 50 steps at 15 Hz = 3.3 s;
    AnyRotate: 120 steps at 20 Hz = 6 s while turning gravity)."""
    grasp_presettle_steps: int = 5
    """adapt.: control steps for the hand to settle at its targets before
    the object is placed (the fingertip centroid is read after them)."""
    grasp_joint_sample_noise: float = 0.3  # [App. C: U(-0.3, 0.3) rad]; HORA 0.25
    grasp_curl_frac: float = 0.75
    """adapt.: share of candidates curled to a random fraction of every
    joint's range instead of canonical + noise (grammar designs have no
    hand-made grasp pose). pop_v3s_32: 2.9% of curled candidates with the
    object at the fingertips held it, 0.43% of canonical ones."""
    grasp_tip_place_frac: float = 0.9
    """adapt.: share of candidates with the object at the fingertip
    centroid; the rest at the hand's spawn point (pop_v3s_32: 1.7% vs
    0.03% held; allegro_right: 3.6% vs 0.5%)."""
    grasp_obj_pos_noise: float = 0.01
    grasp_max_disp_m: float = 0.02  # HORA: a fall of at most 1.5 cm
    grasp_max_lin_speed: float = 0.05
    grasp_max_ang_speed: float = 0.5
    grasp_min_tip_contacts: int = 2  # HORA >= 2 (AnyRotate: "greater than 2")
    grasp_max_nontip_contacts: int = -1  # -1 off (HORA); AnyRotate 0
    grasp_max_tip_dist_m: float = 0.1  # HORA: every fingertip within 0.1 m
    grasp_max_mean_tip_dist_m: float = -1.0  # -1 off; AnyRotate: total 0.2 over 4 tips
    grasp_gravity_cycle: bool = False  # AnyRotate: gravity along +-x, +-y, +-z during the hold
    grasp_max_joint_speed: float = -1.0
    """Peak real-joint speed (rad/s) allowed over the hold; -1 off. The
    report records it per design either way."""
    grasp_reset_joint_noise: float = 0.0
    """U(-noise, noise) rad on a cached grasp's joints and targets at reset.
    0: neither HORA nor AnyRotate perturbs a cached grasp."""
    grasp_reset_obj_pos_noise: float = 0.0
    grasp_settle_steps: int = 0
    """Settle phase (``axis_check_grace_steps``) for envs reset from a cached
    grasp: none, as the paper (the grasp is already at rest)."""

    # --- hand (palm-up placement and actuator as in the isaaclab_repose port) ---
    hand_pose_file: str = "repose_hand_poses.json"
    collision_from_visuals: bool = False
    hand_disable_gravity: bool = False  # [Sec. 4: "Gravity is enabled for both the hand and the object"]
    hand_angular_damping: float = 0.01
    hand_max_depenetration_velocity: float = 1000.0
    hand_solver_position_iterations: int = 8
    hand_solver_velocity_iterations: int = 0
    hand_sleep_threshold: float = 0.005
    hand_stabilization_threshold: float = 0.0005
    hand_stiffness: float = 3.0
    hand_damping: float = 0.1
    hand_armature: float = 0.0
    hand_joint_friction: float = 0.01
    hand_effort_limit: float = 0.5
    hand_velocity_limit: float = 6.283
    population_hand_actuator: bool = False
    population_palm_collider: str = "capsule"
    """Population path: "capsule" (default: the root capsule only),
    "mount_hull" (also a convex palm spanning the root capsule and every
    finger mount, ``projected_hands.palm_hull_points``) or
    "mount_hull_filtered" (the same, collision-filtered against each
    finger's first two links)."""
    population_capsule_radius: float = -1.0
    """> 0: collider radius of every real link and the root (masses keep the
    design's radius). -1: the design's own (projections: 0.01 m)."""
    population_projected_pose: bool = False
    """Population path: a projected commercial hand with an entry in
    ``hand_pose_file`` stands where its single-hand URDF asset does (pose,
    spawn point, default joints; ``projected_hands.urdf_equivalent_placement``)
    instead of at its analytic palm-up calibration."""
    grasp_canonical_profile: str = "palm_up"
    """Population grasp search: "palm_up" (default, each design's calibrated
    curl), "opposition" (one candidate pose per finger as the opposing digit
    at HORA's thumb fractions, the others at its finger fractions; a design
    keeps the opposing finger that held the most grasps) or "hora_like" (every finger at HORA's allegro pose as range
    fractions, ``grasp_cache.HORA_LIKE_PROFILE``). 2026-10-02: the projected
    allegro learned from HORA-pose grasps (8-11 s held at 15-30 min) and not
    from palm-up ones (1.1-1.3 s)."""
    grasp_projected_canonical: bool = False
    """Grasp search: a projected commercial hand starts from its single-hand
    canonical grasp pose (``grasp_cache.CANONICAL_GRASP_POSES``, mapped onto
    its slots) instead of its palm-up curl."""
    """Population path: give every real joint of every design the actuator
    above (hand_stiffness, hand_damping, hand_effort_limit; for the hora
    profile HORA's Allegro PD 3.0 / 0.1, torque clip 0.5 N m) instead of the
    designs' own (3.93 / 0.15, 1.0 N m). False (default): own gains."""

    # --- optional gravity curriculum (NVIDIA Dexsuite ADR, 2025) ---
    gravity_curriculum: bool = False
    """Isaac Lab Dexsuite's adr_curriculum: a per-env difficulty in
    [0, gravity_max_difficulty] rises by 1 when an episode succeeds (>=
    ``gravity_promote_goals`` goals) and falls by 1 otherwise; gravity =
    (mean difficulty / max) x (0, 0, -9.81), left at 0 while that fraction is
    below 0.1."""
    gravity_max_difficulty: int = 10
    gravity_promote_goals: int = 1
    gravity_promote_metric: str = "goals"
    """"goals" (Dexsuite: success) or "rotations" (>= ``gravity_promote_rotations``
    rotations about k per episode; the hora profile has no goals)."""
    gravity_promote_rotations: float = 0.25
    gravity_promotion_only: bool = False


@configclass
class HoraCfg:
    """The ``hora`` task profile: HORA (Qi et al., CoRL 2022), numbers from
    its released ``configs/task/AllegroHandHora.yaml`` and
    ``allegro_hand_hora.py`` (``hora_profile.py``). The scene, object, hand
    and grasp cache are the anyrotate block's; ``apply_anyrotate_to_cfg``
    writes this block's timing, friction and action scale into it."""

    sim_dt: float = 1.0 / 120.0  # sim.dt 0.0083333
    decimation: int = 6  # controlFrequencyInv 6: 20 Hz
    episode_length_s: float = 20.0  # episodeLength 400 steps
    action_scale: float = 1.0 / 24.0  # targets = prev_targets + 1/24 * actions
    friction: float = 1.0
    """adapt.: HORA randomises friction U(0.3, 3.0) per env; one value here."""
    angvel_clip_min: float = -0.5
    angvel_clip_max: float = 0.5
    rotate_reward_scale: float = 1.0
    obj_linvel_penalty_scale: float = -0.3
    pose_diff_penalty_scale: float = -0.3
    torque_penalty_scale: float = -0.1
    work_penalty_scale: float = -2.0
    joint_noise_scale: float = 0.02  # randomization.jointNoiseScale, U(-1, 1) x 0.02
    drop_dz: float = 0.015
    """adapt.: HORA ends an episode below an absolute height
    (reset_height_threshold 0.645 m; objects start at 0.65-0.66 m); here a fall
    of ``drop_dz`` below the episode's start height."""
    z_axis_frame: str = "world_up"
    """The rotation axis: the palm normal of the palm-up hand (HORA: world z)."""

    # --- shared controllers across hands (options; HORA trains one hand) ---
    morph_obs: bool = False
    """Population path: add a morphology context to actor and critic
    (``hora_morph``): per slot validity, joint axis and origin in the palm
    frame at q = 0, link length, limits and the canonical grasp pose; per
    fingertip its palm-frame position (current q) and mask; digit count,
    hand scale, capsule radius."""
    ghost_action_mask: bool = False
    """Append each action slot's sign (+1 real joint, -1 ghost) as the last
    observation values, for ``policy_network``'s ``ghost_mask_tail`` (the
    agent must use ``network.name: inhand_actor_critic`` with
    ``space.continuous.ghost_mask_tail`` = the joint count)."""
    per_design_reward_norm: bool = False
    """Divide each env's reward by its design's running reward RMS
    (``hora_profile.DesignRewardScale``)."""
    reward_norm_decay: float = 0.999
    reward_norm_floor: float = 0.05
    ppo_group_info: bool = False
    """Publish each env's design index in the step infos (``extras
    ["ppo_group"]``) for the vendored rl_games' ``group_advantage_norm``
    (advantages normalised within each design)."""
    token_obs: bool = False
    """Population path: the per-joint token observation of the team's
    joint-token transformer (``token_layout.py``; agent
    ``rl_games_hora_token_ppo_cfg_entry_point``) instead of HORA's flat
    history and privileged values."""
    fall_penalty: float = 0.0
    """Added to the reward on the step the object drops (HORA has no fall
    penalty, so flicking the object and dropping it is a cheap local
    optimum for a per-hand expert); e.g. -20. 0 = off."""
    alive_bonus: float = 0.0
    """Added to the reward on every step the object has not dropped. 0 = off."""
    teacher_obs: bool = False
    """With ``token_obs``: also emit, under the observation key
    ``teacher_obs``, the flat observation a per-hand MLP expert was trained
    on (HORA's 3-frame joint history and its 9 privileged values,
    ``hora_profile.flat_observation``; no design one-hot), so one env step
    serves a token student and every env's own expert (``distill/``)."""
    design_id_obs: int = 0
    """Population path: append a one-hot of each env's design index, this
    wide (>= the population size; 0 = off), before the slot signs. Read by
    ``policy_network``'s ``per_design_nets``."""
    disjoint_slots: bool = False
    """Population path: give every design its own policy columns for its
    real joints (``hora_profile.disjoint_slot_map``; per-joint
    observations and actions are permuted per env), so no two designs
    share an observation or action column. Needs at most 32 real joints
    over the whole population."""


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
    task_profile: str = "legacy"
    """"anyrotate": multi-axis rotation after AnyRotate (CoRL 2024; ``anyrotate``
    below, ``anyrotate_profile.py``). "legacy" (default): this env's original spec (SAPG-era reward,
    tolerance and goal curricula, palm-normal-axis goals), every field
    outside ``repose`` exactly as before. "isaaclab_repose": NVIDIA's 2021
    in-hand cube reorientation spec (``repose`` below,
    ``repose_profile.py``), selectable with ``env.task_profile=
    isaaclab_repose`` but not adopted (2026-10-01: the team chose a newer
    spec); kept as a verified port and framework sanity check."""
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
    repose: ReposeCfg = ReposeCfg()
    anyrotate: AnyRotateCfg = AnyRotateCfg()
    hora: HoraCfg = HoraCfg()


__all__ = [
    "InHandReorientEnvCfg", "AssetsCfg", "ObsCfg", "ActionCfg", "RewardCfg",
    "PhysicsCfg", "ResetCfg", "ReposeCfg", "AnyRotateCfg", "HoraCfg", "TerminationCfg",
]
