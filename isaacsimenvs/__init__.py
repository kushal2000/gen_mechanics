"""Task registry.

One subpackage per task (its env and config, scene construction, observations,
rewards), plus ``common/``: what the tasks share (the robot registry, URDF -> USD,
object authoring, the observation layout). ``common`` imports no task. Each task
registers its own gym ids as an import side effect, so importing this
package exposes every task to ``gym.make`` / ``gym.spec``.

Importing a task pulls in Isaac Lab, whose sub-namespaces only resolve once
``AppLauncher`` has booted Kit -- so import this *after* constructing the
launcher, not at module scope in a script that also parses arguments.
"""

from . import pose_reaching_6d  # noqa: F401  registers GenMech-PoseReach-Direct-v0
from . import inhand_reorient  # noqa: F401  registers GenMech-InHandReorient-Direct-v0
from . import inhand_olympics  # noqa: F401  registers GenMech-InHandOlympics-Direct-v0

__all__ = ["pose_reaching_6d", "inhand_reorient", "inhand_olympics"]
