"""Scene construction for the pose-reaching task.

``assembly`` builds the scene and assigns a design and an object to every env,
``materials`` sets PhysX material properties once the sim has started,
``author_robot`` writes a generated design's hand straight into USD (instead of
converting a URDF), ``multi_articulation`` presents several hands' articulations as
one. The pieces the other tasks share (robot registry, object authoring, URDF -> USD,
flattening) are in ``isaacsimenvs.common``.
"""

from .assembly import SceneRecord, finalize_scene, setup_scene

__all__ = ["SceneRecord", "finalize_scene", "setup_scene"]
