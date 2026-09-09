"""hand_sampler: the hand design space, and the robots it is measured against.

**Pure Python -- importing this package does not import Isaac Sim.** That is the
point of keeping it separate. Isaac Lab's sub-namespaces only resolve after
``AppLauncher`` has booted Kit (~60-120 s, one process per GPU, hangs on
teardown), and the parts of this project that search over designs are
deliberately offline: sampling a population, mutating it, scoring it against a
frozen value function, and the collision gates all run on CPU.

The dependency rule for the repo is one line, and this package is the base of
it::

    hand_sampler  <-  isaacsimenvs  <-  coevolution

Nothing here may import from the other two.

The design space, as a grammar::

    design_space      what a hand is, where its parts are  (the noun)
    validate_design   which designs are legal              (cheap, every mutation)
    self_collision    which designs intersect themselves   (expensive, once)
    gen_init_pop      generation 0
    mutate_design     which designs are one step away
    experiments/      where a population of them ends up

``DESIGN.md`` has the reasoning, and code cites it by section number.
``robot_spec`` is the other half: fixed arm+hand descriptions the task reads,
which is what ``isaacsimenvs`` imports. ``params``/``urdf``/``synth_spec``/
``population`` are the superseded HandParams pipeline, kept until the tree grows
its own path to a URDF.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]

ASSETS_DIR: Path = REPO_ROOT / "assets"


# Lives here because every layer needs repo-relative asset paths.
def resolve(relpath: str | Path) -> Path:
    """Resolve a repo-relative path; absolute paths pass through unchanged."""
    p = Path(relpath)
    return p if p.is_absolute() else REPO_ROOT / p


__all__ = ["ASSETS_DIR", "REPO_ROOT", "resolve"]
