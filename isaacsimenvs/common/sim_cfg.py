"""The default simulation config every task shares (from pose_reaching_6d/env_cfg.py, 27 Oct)."""
from __future__ import annotations

from isaaclab.sim import PhysxCfg, SimulationCfg


def default_sim_cfg() -> SimulationCfg:
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


__all__ = ["default_sim_cfg"]
