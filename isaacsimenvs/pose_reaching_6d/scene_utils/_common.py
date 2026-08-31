"""Constants and helpers shared by the three scene modules."""

from __future__ import annotations

import time


# Joint names, regexes, PD-gain tables, the arm home pose, and the palm and
# fingertip body names all used to live here as module constants pinned to the
# left SHARPA hand. They are now fields on the selected RobotSpec
# (isaacsimenvs/pose_reaching_6d/robots/), so the scene follows the configured hand.
#
# HAND_JOINT_FRICTION was dropped rather than moved: it was defined and
# length-asserted but never read -- joint friction reaches PhysX from the URDF's
# <dynamics friction> via UrdfConverter.

_CONTACT_OFFSET = 0.002
_REST_OFFSET = 0.0

# group: "rb" (RigidBodyAPI) or "art" (ArticulationRootAPI).
# attr_name: USD attribute path. vtype_str: matched against pxr.Sdf.ValueTypeNames.
_PHYSICS_SPECS: dict[str, tuple[str, str, str]] = {
    "kinematic_enabled": ("rb", "physics:kinematicEnabled", "Bool"),
    "disable_gravity": ("rb", "physxRigidBody:disableGravity", "Bool"),
    "max_depenetration_velocity": ("rb", "physxRigidBody:maxDepenetrationVelocity", "Float"),
    "rb_solver_position_iterations": ("rb", "physxRigidBody:solverPositionIterationCount", "Int"),
    "rb_solver_velocity_iterations": ("rb", "physxRigidBody:solverVelocityIterationCount", "Int"),
    "articulation_enabled": ("art", "physics:articulationEnabled", "Bool"),
    "enabled_self_collisions": ("art", "physxArticulation:enabledSelfCollisions", "Bool"),
    "solver_position_iterations": ("art", "physxArticulation:solverPositionIterationCount", "Int"),
    "solver_velocity_iterations": ("art", "physxArticulation:solverVelocityIterationCount", "Int"),
}


def _log_scene_step(start_time: float, message: str) -> None:
    print(f"[scene_utils][+{time.perf_counter() - start_time:.2f}s] {message}", flush=True)

def _materialize_env_prims(env) -> None:
    stage = get_current_stage()
    for env_path in env.scene.env_prim_paths:
        if not stage.GetPrimAtPath(env_path).IsValid():
            stage.DefinePrim(env_path, "Xform")
