"""Scene construction for the pose-reaching task.

``assembly`` builds the scene and assigns a design and an object to every env,
``materials`` sets PhysX material properties once the sim has started, and the
``author_*`` modules write USD directly instead of converting URDFs.
"""

from .assembly import RobotPopulation, SceneRecord, finalize_scene, setup_scene

__all__ = ["RobotPopulation", "SceneRecord", "finalize_scene", "setup_scene"]
