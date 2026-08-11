"""Robot hardware registry.

Each entry is a :class:`~genmech.robots.spec.RobotSpec` — a frozen description
of one arm+hand combination. The task reads everything hardware-specific from
the selected spec, so adding a hand is a new module plus a line here.

Selection is ``cfg.assets.robot_spec``; the env resolves it before
``super().__init__`` because Isaac Lab reads ``action_space`` and
``observation_space`` off the configclass, and both are derived from the spec.

Importable without Isaac Sim — nothing here touches isaaclab, so the offline
tools (reachability viewer, URDF authoring, eval suite generation) can use the
registry without booting Kit.
"""

from __future__ import annotations

from genmech.robots.spec import RobotSpec
from genmech.robots.sharpa_iiwa14 import SHARPA_IIWA14


REGISTRY: dict[str, RobotSpec] = {
    spec.name: spec
    for spec in (
        SHARPA_IIWA14,
    )
}


def get_robot_spec(name: str) -> RobotSpec:
    """Look up a spec by name, or fail with the list of valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown robot_spec {name!r}; registered: {sorted(REGISTRY)}"
        ) from None


__all__ = ["RobotSpec", "REGISTRY", "get_robot_spec", "SHARPA_IIWA14"]
