"""Config for the in-hand olympics env: one hand on a 6-DOF dummy wrist, palm down above a table.

Every task is a sequence of 6D goal poses for the pen; the task decides where the pen starts and how the next goal
follows a hit one, while success, rewards and terminations are shared (env.py). Today's tasks: pen_spin.

Never use a ``None`` default here: a hydra override of a ``None`` default is rejected
(pose_reaching_6d/env_cfg.py says the same).
"""
from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ..pose_reaching_6d.env_cfg import _default_sim_cfg


@configclass
class HandCfg:
    # A registered single-hand spec (pose_reaching_6d/scene_utils/robots). Its URDF is wrapped by the wrist chain.
    robot_spec: str = "wuji2_left_uniform_handonly"
    # 0 = the URDF's own per-joint speed cap (the uniform assets: 5 rad/s).
    velocity_limit: float = 0.0


@configclass
class WristCfg:
    """Six dummy joints between a fixed base and the hand's root body: prismatic x, y, z, then revolute roll (about
    x), pitch (about y), yaw (about z). Axes are world axes at zero roll/pitch; yaw comes last, so it always turns
    about the palm normal.

    The base sits on the table's top surface at the env origin, so ``wrist_z`` is the height of the hand's root
    body above the table. At zero roll/pitch/yaw the palm faces down and the fingers point along world +x.
    """

    # (lower, upper) per joint: metres for x/y/z, radians for roll/pitch/yaw.
    # Narrow on purpose: in-hand manipulation, not the wrist, should do the work.
    x_range: tuple[float, float] = (-0.10, 0.10)
    y_range: tuple[float, float] = (-0.10, 0.10)
    z_range: tuple[float, float] = (0.02, 0.20)
    # Human-like, from the normal anatomical ranges (AAOS). Signs, checked by forward kinematics on the left hand
    # (thumb at -y): roll - = supination (palm turns toward the thumb side), + = pronation; palm-down is already near
    # full pronation, so the range is lopsided and stops with the palm on edge. Pitch + = flexion (fingertips
    # down) 80 deg, - = extension 70 deg. Yaw + = ulnar deviation 30 deg, - = radial deviation 20 deg.
    roll_range: tuple[float, float] = (-1.5708, 0.3491)      # -90 .. +20 deg
    pitch_range: tuple[float, float] = (-1.2217, 1.3963)     # -70 .. +80 deg
    yaw_range: tuple[float, float] = (-0.3491, 0.5236)       # -20 .. +30 deg
    # Pose at reset: (x, y, z, roll, pitch, yaw).
    default_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.15, 0.0, 0.0, 0.0)

    # Implicit PD drives. The robot has gravity off (flatten_robot_usd), so these only fight contacts and inertia.
    prismatic_stiffness: float = 1000.0      # N/m
    prismatic_damping: float = 60.0          # N s/m
    prismatic_effort_limit: float = 100.0    # N
    prismatic_velocity_limit: float = 1.0    # m/s
    revolute_stiffness: float = 50.0         # N m/rad
    revolute_damping: float = 2.0            # N m s/rad
    revolute_effort_limit: float = 20.0      # N m
    revolute_velocity_limit: float = 6.0     # rad/s
    # Mass of each dummy link; PhysX needs a body behind every joint.
    link_mass: float = 0.01


@configclass
class ActionCfg:
    """Wrist: delta position targets. Each step a in [-1, 1] moves an integrated target by a * max_speed * dt
    (clamped to the joint range), and the PD target follows it through an EMA. The EMA runs on the integrated target,
    not on the previous PD target, so the wrist really reaches max_speed (smoothing the previous PD target would
    cap it at ema * max_speed); it lags the command by about dt / ema = 0.17 s.

    Fingers: absolute targets, a in [-1, 1] mapped onto each joint's range, through an EMA (as in-hand training).
    """

    wrist_max_speed_lin: float = 0.2         # m/s, x/y/z
    wrist_max_speed_ang: float = 1.5         # rad/s, roll/pitch/yaw (~86 deg/s)
    wrist_ema: float = 0.1
    hand_ema: float = 0.1
    # Action latency, as in-hand training (pose_reaching_6d actions.py): every step each env applies one of its
    # last action_delay_max + 1 actions, picked uniformly, so 0..action_delay_max steps (0-50 ms at 60 Hz) late.
    # At an episode's first step the queue is filled with that step's action.
    use_action_delay: bool = True
    action_delay_max: int = 3


@configclass
class TableCfg:
    size: tuple[float, float, float] = (0.8, 0.8, 0.04)   # x, y, thickness (m)
    top_height: float = 0.40                              # top surface above the ground (m)
    friction: float = 0.5


@configclass
class PenCfg:
    """The pen: a capsule lying on the table. ``length`` is tip to tip, caps included (the PhysX capsule's
    cylinder is length - diameter long)."""

    length: float = 0.10                     # m
    diameter: float = 0.03                   # m
    mass: float = 0.03                       # kg
    friction: float = 0.5
    # Spawn pose on the table, in the env frame: centre (x, y), and yaw of the pen's tip (its local +x end) from
    # world +x. Each task places the pen at reset itself; this is only where it is built.
    init_xy: tuple[float, float] = (0.0, 0.0)
    init_yaw: float = 0.0                    # rad


@configclass
class ObsCfg:
    """In-hand reorientation's format (observations.py): the actor sees obs_list, the critic state_list."""

    obs_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
        "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
        "keypoints_rel_ee", "keypoints_rel_goal", "object_vel", "palm_extents", "task_onehot",
    )
    state_list: tuple[str, ...] = (
        "joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
        "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
        "keypoints_rel_ee", "keypoints_rel_goal", "object_vel", "palm_extents", "task_onehot",
        "object_keypoints_rel_joint", "palm_keypoints", "ee_pos", "ee_rot", "ee_vel", "object_rot",
        "fingertip_pos_rel_palm", "closest_keypoint_max_dist", "closest_fingertip_dist",
        "progress", "successes", "reward",
    )
    clamp_abs_observations: float = 10.0


@configclass
class TaskCfg:
    """Which tasks the env samples from, one per env at every reset, uniformly. The policy sees the task as a
    one-hot over ``tasks/__init__.py``'s TASK_NAMES (fixed order, so a policy's input never changes meaning when
    the mix does)."""

    names: tuple[str, ...] = ("pen_spin",)


@configclass
class PenSpinCfg:
    """Spin the pen flat on the table, in place. Goals: the pen lying flat with its centre at ``centre_xy``, at
    yaw = previous goal's yaw + direction * step; the direction is drawn per episode. The pen starts at the centre at
    a random yaw and the first goal is one step from it."""

    centre_xy: tuple[float, float] = (0.0, 0.0)   # env frame, on the table
    step_deg: float = 45.0


@configclass
class PenSpinAirCfg:
    """Spin the pen in the air (tasks/pen_spin.py PenSpinAir): it starts flat on the table at a random yaw; the
    first goal is it picked up flat to pen_lift's lift_height (the same height as the held pen) at that yaw, and
    each goal after that is ``step_deg`` further round about the vertical at that height, one direction per
    episode."""

    centre_xy: tuple[float, float] = (0.0, 0.0)
    step_deg: float = 45.0


@configclass
class PenFlipAirCfg:
    """Flip the pen end over end in the air (tasks/pen_flip.py): picked up flat to pen_lift's lift_height at its
    starting yaw (the first goal), then each goal ``step_deg`` further about the horizontal axis across the pen, its
    centre staying put, one direction per episode."""

    centre_xy: tuple[float, float] = (0.0, 0.0)
    step_deg: float = 45.0


@configclass
class PenLiftCfg:
    """Lift the pen flat and hold it (tasks/pen_lift.py). The goal: the pen at its starting yaw, flat, centred at
    ``centre_xy``, ``lift_height`` above where it rests on the table. A goal counts after ``hold_steps``
    consecutive steps within tolerance; the next goal is the same pose."""

    centre_xy: tuple[float, float] = (0.0, 0.0)
    lift_height: float = 0.08                # m
    hold_steps: int = 30                     # 0.5 s at 60 Hz


@configclass
class PerturbCfg:
    """Random pushes and twists on the pen, for tasks that ask for them (pen_lift): from the episode's first hit on
    (so the pickup is learned undisturbed), while the pen is lifted.
    In-hand reorientation's wrench DR with pose reaching's defaults (pose_reaching_6d/obs_utils/actions.py): each
    step a new force fires with the env's probability (log-uniform over ``prob_range``, drawn at reset):
    randn(3) * pen mass * force_scale (N); likewise the torque with torque_scale (N m). With decay 0 each lasts one
    policy step."""

    force_scale: float = 20.0
    torque_scale: float = 2.0
    prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    torque_decay: float = 0.0
    decay_interval: float = 0.08             # s
    # "Lifted": the pen's centre this far above its resting height (m).
    lifted_height: float = 0.02


@configclass
class GoalCfg:
    """Success = every pen keypoint (4 points along the long axis, tip to tail, so the ends are distinct) within
    ``success_tolerance`` of its goal position, for the task's success_steps consecutive steps (pen_spin 1,
    pen_lift its hold_steps). Roll about the long axis is invisible to it, as it is to a capsule."""

    success_tolerance: float = 0.01          # m, max keypoint distance
    max_successes: int = 50                  # goals after which the episode ends
    # Each goal gets this long (s); a hit restarts the clock, as in in-hand reorientation.
    seconds_per_goal: float = 10.0


@configclass
class RewardCfg:
    # Progress: scale * (best keypoint distance so far for this goal - current), when positive (in-hand's
    # keypoint reward).
    keypoint_rew_scale: float = 2000.0
    # Per goal, paid as goal_bonus / success_steps on each step within tolerance (all at once for a one-step goal).
    goal_bonus: float = 1000.0
    # Getting the fingertips to the pen: scale * the mean over fingertips of (closest distance so far - current).
    approach_rew_scale: float = 50.0
    # Charged on the step an episode ends because the pen was lost (termination "away" or "fell"). Without it, a
    # policy that cannot reach a goal learns to swat the pen away and restart, collecting the approach reward every
    # episode (pen_lift, 12-13 Oct: episodes of ~0.6 s from epoch 500 on).
    lost_pen_penalty: float = 100.0
    # L1 on joint velocities.
    wrist_action_penalty: float = 0.03
    hand_action_penalty: float = 0.0003


@configclass
class TerminationCfg:
    # The pen's centre more than this far from the goal's (m): it slid or was knocked away.
    max_goal_distance: float = 0.15
    # Below the table top (m): it fell off.
    fall_below_table: float = 0.05


@configclass
class InHandOlympicsEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 10.0          # set from goal.seconds_per_goal by the env (after any override)
    # Derived in setup_scene (6 wrist joints + the hand's); a stale non-zero value raises.
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    # In-hand reorientation's simulation settings, unchanged (120 Hz TGS, 8 position iterations, its friction and
    # GPU buffer settings).
    sim: SimulationCfg = _default_sim_cfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)

    hand: HandCfg = HandCfg()
    wrist: WristCfg = WristCfg()
    action: ActionCfg = ActionCfg()
    table: TableCfg = TableCfg()
    pen: PenCfg = PenCfg()
    obs: ObsCfg = ObsCfg()
    task: TaskCfg = TaskCfg()
    pen_spin: PenSpinCfg = PenSpinCfg()
    pen_lift: PenLiftCfg = PenLiftCfg()
    pen_spin_air: PenSpinAirCfg = PenSpinAirCfg()
    pen_flip_air: PenFlipAirCfg = PenFlipAirCfg()
    perturb: PerturbCfg = PerturbCfg()
    goal: GoalCfg = GoalCfg()
    reward: RewardCfg = RewardCfg()
    termination: TerminationCfg = TerminationCfg()
    contact_offset: float = 0.002            # as in-hand (PhysicsCfg)
    rest_offset: float = 0.0
