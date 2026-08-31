"""Task registry for genmech.

Each task subpackage registers itself with gymnasium on import (side effect
in its ``__init__.py``). Importing ``genmech`` (or any child) is enough
to expose all task ids to ``gym.make`` / ``gym.spec``.
"""

from . import pose_reach  # side effect: gym.register("GenMech-PoseReach-Direct-v0", ...)

__all__ = ["pose_reach"]
