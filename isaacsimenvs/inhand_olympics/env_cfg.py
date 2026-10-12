"""Config for the in-hand olympics env: one hand on a 6-DOF dummy wrist, palm down above a table.

Skeleton stage: the scene (hand + wrist + table + pen) and the action pipeline only. Tasks, objects, observations and
rewards come later.

Never use a ``None`` default here: a hydra override of a ``None`` default is rejected
(pose_reaching_6d/env_cfg.py says the same).
"""
from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


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
    # Reset pose on the table, in the env frame: centre (x, y), and yaw of the pen's long axis from world +x.
    init_xy: tuple[float, float] = (0.0, 0.0)
    init_yaw: float = 0.0                    # rad


@configclass
class InHandOlympicsEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 10.0
    # Derived in setup_scene (6 wrist joints + the hand's); a stale non-zero value raises.
    action_space: int = 0
    observation_space: int = 0
    state_space: int = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(solver_type=1, min_position_iteration_count=8, max_position_iteration_count=8,
                       min_velocity_iteration_count=0, max_velocity_iteration_count=0,
                       bounce_threshold_velocity=0.2),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)

    hand: HandCfg = HandCfg()
    wrist: WristCfg = WristCfg()
    action: ActionCfg = ActionCfg()
    table: TableCfg = TableCfg()
    pen: PenCfg = PenCfg()
    contact_offset: float = 0.002
    rest_offset: float = 0.0
