"""Task registry.

One subpackage per task. Each registers its own gym ids as an import side
effect, so importing this package exposes every task to ``gym.make`` /
``gym.spec``.

Importing a task pulls in Isaac Lab, whose sub-namespaces only resolve once
``AppLauncher`` has booted Kit -- so import this *after* constructing the
launcher, not at module scope in a script that also parses arguments.

Shared across tasks, and deliberately outside any of them:
``authoring`` (design -> USD prims), ``robots`` (the RobotSpec registry) and
``reachability_viewer``.
"""

from . import pose_reaching_6d  # noqa: F401  registers GenMech-PoseReach-Direct-v0

__all__ = ["pose_reaching_6d"]
