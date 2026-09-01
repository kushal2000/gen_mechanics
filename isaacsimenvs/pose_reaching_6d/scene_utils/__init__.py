"""Scene construction for the pose-reaching task.

Split three ways because the single module was doing three unrelated jobs:
``usd_conversion`` turns URDFs into baked USD, ``materials`` binds PhysX
material properties to the built stage, and ``assembly`` builds the scene
and assigns a design and object to every env.
"""

from .assembly import (
    build_rigid_object_cfg,
    build_robot_articulation_usd_cfg,
    setup_scene,
    _author_objects_into_envs,
    _build_object_scale_tensor,
    _build_robot_design_tensor,
    _ensure_robot_population,
    finalize_scene,
    resolve_spec,
    _resolve_robot_population,
    RobotPopulation,
    _verify_robot_design_assignment,
)
from .materials import (
    apply_physx_material_properties,
    arm_counts_from,
    shape_layouts_from_record,
)

__all__ = [
    "RobotPopulation",
    "_author_objects_into_envs",
    "_build_object_scale_tensor",
    "_build_robot_design_tensor",
    "_ensure_robot_population",
    "_resolve_robot_population",
    "_verify_robot_design_assignment",
    "apply_physx_material_properties",
    "arm_counts_from",
    "build_rigid_object_cfg",
    "build_robot_articulation_usd_cfg",
    "finalize_scene",
    "resolve_spec",
    "setup_scene",
    "shape_layouts_from_record",
]
