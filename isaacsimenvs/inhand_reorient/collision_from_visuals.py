"""Rewrite a URDF's collision geometry from its visual meshes.

Pure stdlib, like ``urdf_cutter.py``. For every link that has at least one
``<visual>`` with a ``<mesh>``, the link's ``<collision>`` elements are
replaced by one ``<collision>`` per such visual (same origin, same mesh and
scale); the URDF importer then turns each mesh into a convex hull. Links
without a visual mesh keep their collisions.

Why: the drake ``allegro_right`` URDF collides through boxes and 12 mm tip
spheres, while NVIDIA's Allegro USD (Isaac-Repose-Cube-Allegro) collides
through convex hulls of the link meshes. Used by the ``isaaclab_repose``
profile's ``repose.collision_from_visuals`` option only.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = ["collisions_from_visuals"]


def collisions_from_visuals(urdf_in: str | Path, urdf_out: str | Path) -> int:
    """Write the rewritten URDF to ``urdf_out``; return how many links changed."""
    tree = ET.parse(str(urdf_in))
    root = tree.getroot()
    changed = 0
    for link in root.findall("link"):
        visuals = [v for v in link.findall("visual") if v.find("geometry/mesh") is not None]
        if not visuals:
            continue
        for c in link.findall("collision"):
            link.remove(c)
        for v in visuals:
            col = ET.SubElement(link, "collision")
            origin = v.find("origin")
            if origin is not None:
                col.append(copy.deepcopy(origin))
            col.append(copy.deepcopy(v.find("geometry")))
        changed += 1
    tree.write(str(urdf_out))
    return changed
