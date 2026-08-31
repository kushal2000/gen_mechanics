"""hand_sampler: the generated-hand design space.

**Pure Python -- importing this package does not import Isaac Sim.** That is
the point of keeping it separate. Isaac Lab's sub-namespaces only resolve after
``AppLauncher`` has booted Kit (~60-120 s, one process per GPU, hangs on
teardown), and the parts of this project that search over designs are
deliberately offline: sampling a population, mutating it, scoring it against a
frozen value function, and the geometry gates all run on CPU.

The dependency rule for the repo is one line, and this package is the base of
it::

    hand_sampler  <-  isaacsimenvs  <-  coevolution

Nothing here may import from the other two.
"""

__all__: list[str] = []
