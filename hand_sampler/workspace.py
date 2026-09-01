"""Task geometry shared by every viewer, and the collision-hull swap.

These are facts about the scene a hand is judged in -- where the table top
sits, how big the table actually is, and what geometry PhysX really uses for
contacts -- not facts about any one viewer. They lived in the reachability
viewer, which meant the design-space and mutation viewers had to reach into
``isaacsimenvs`` to draw a table, inverting the dependency rule. Pure numpy and
trimesh, so ``hand_sampler`` stays importable without a simulator.
"""

from __future__ import annotations

import numpy as np

from hand_sampler.paths import resolve as resolve_repo_path


# Task geometry, from ResetCfg. Identical for every hand — that is the point.
GOAL_VOLUME_MINS = (-0.35, -0.2, 0.6)
GOAL_VOLUME_MAXS = (0.35, 0.2, 0.95)
TABLE_Z = 0.38
TABLE_URDF = "assets/urdf/table_narrow.urdf"


def table_extents() -> tuple[float, float, float]:
    """Box dimensions read from the actual table asset.

    This used to be a hardcoded TABLE_SIZE = (1.2, 0.8), which is 2.5x too wide
    in x and 2x in y against the asset's 0.475 x 0.4 x 0.3. An oversized table
    swallows the arm's link_3 and link_4, which looks exactly like the robot
    colliding with the table when nothing is wrong with the robot.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(resolve_repo_path(TABLE_URDF)).getroot()
    for link in root.findall("link"):
        for coll in link.findall("collision"):
            box = coll.find("geometry/box")
            if box is not None:
                return tuple(float(v) for v in box.get("size").split())
    raise RuntimeError(f"no box collision geometry in {TABLE_URDF}")


def _hull_collision_scene(urdf) -> int:
    """Replace each collision mesh with its convex hull, in place.

    Isaac Lab stamps ``approximation="convexHull"`` on every mesh collider
    (verified in the baked USD: 34/34 on SHARPA, 26/26 on Allegro), so PhysX
    never resolves contacts against the triangle meshes the URDF declares -- it
    uses their hulls, with every concavity between phalanges filled in.

    Showing the declared mesh therefore overstates the fidelity of the
    simulation. Hulling the collision scene before ViserUrdf reads it means the
    "collision" view is the geometry that actually decides contacts, and it
    still follows the joint sliders because only the geometry is swapped, not
    the scene graph.

    Generated hands are unaffected -- their capsules and palm box are analytic
    primitives with approximation "None", i.e. simulated exactly as declared.
    """
    import trimesh

    scene = getattr(urdf, "collision_scene", None)
    if scene is None:
        return 0
    n = 0
    for key, geom in list(scene.geometry.items()):
        if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 3:
            try:
                scene.geometry[key] = geom.convex_hull
                n += 1
            except Exception:
                pass
    return n


__all__ = ["GOAL_VOLUME_MINS", "GOAL_VOLUME_MAXS", "TABLE_Z", "TABLE_URDF",
           "table_extents", "_hull_collision_scene"]
