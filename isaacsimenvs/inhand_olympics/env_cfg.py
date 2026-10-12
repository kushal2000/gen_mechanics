"""Config for the in-hand olympics env: one hand on a 6-DOF dummy wrist, palm down above a table.

Skeleton stage: the scene (hand + wrist + table) and joint-target actions only. Tasks, objects, observations and
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
    x_range: tuple[float, float] = (-0.25, 0.25)
    y_range: tuple[float, float] = (-0.25, 0.25)
    z_range: tuple[float, float] = (0.02, 0.40)
    roll_range: tuple[float, float] = (-1.5708, 1.5708)
    pitch_range: tuple[float, float] = (-1.5708, 1.5708)
    yaw_range: tuple[float, float] = (-3.1416, 3.1416)
    # Pose at reset: (x, y, z, roll, pitch, yaw).
    default_pose: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.20, 0.0, 0.0, 0.0)

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
class TableCfg:
    size: tuple[float, float, float] = (0.8, 0.8, 0.04)   # x, y, thickness (m)
    top_height: float = 0.40                              # top surface above the ground (m)
    friction: float = 0.5


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
    table: TableCfg = TableCfg()
    contact_offset: float = 0.002
    rest_offset: float = 0.0
