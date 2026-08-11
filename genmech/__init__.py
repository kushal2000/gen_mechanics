"""genmech: hand-hardware generalization on 6D pose reaching.

**Importing this package does not import Isaac Sim.** That is deliberate.

Isaac Lab's sub-namespaces (``isaaclab.envs``, ``isaaclab.sim``, ...) only
resolve after ``AppLauncher`` has booted Kit, so anything that imports the task
env transitively requires a Kit boot (~60-120 s, one process per GPU, hangs on
teardown). Several parts of this project are deliberately offline and must not
pay that cost or hold a GPU: the reachability viewer
(``genmech.tools.reachability_viewer``), URDF authoring and geometry
calibration, eval aggregation, and the robot registry itself.

So ``genmech``, ``genmech.robots``, ``genmech.utils``, and ``genmech.tools`` are
all importable without Kit. To register the gym task ids, import the task
package explicitly — *after* constructing ``AppLauncher``:

    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app

    import genmech.tasks  # noqa: F401  registers GenMech-PoseReach-Direct-v0
"""

__all__: list[str] = []
