"""Robot hardware registry.

Each entry is a :class:`~hand_sampler.robot_spec.RobotSpec` — a frozen description
of one arm+hand combination. The task reads everything hardware-specific from
the selected spec, so adding a hand is a new module plus a line here.

Selection is ``cfg.assets.robot_spec``; ``setup_scene`` resolves it and derives
``action_space`` and ``observation_space`` from it.

Importable without Isaac Sim — nothing here touches isaaclab, so the offline
tools (reachability viewer, URDF authoring, eval suite generation) can use the
registry without booting Kit.
"""

from __future__ import annotations

from hand_sampler.robot_spec import RobotSpec
from isaacsimenvs.pose_reaching_6d.scene_utils.robots.sharpa_iiwa14 import SHARPA_IIWA14


REGISTRY: dict[str, RobotSpec] = {spec.name: spec for spec in (SHARPA_IIWA14,)}


def get_robot_spec(name: str) -> RobotSpec:
    """Look up a spec by name, or fail with the list of valid names."""
    if name in REGISTRY:
        return REGISTRY[name]
    # A generated population's shared template, by name or by file. Resolved
    # here so everything that asks -- the env AND the network, which
    # interpolates its own copy from the config -- gets the same answer.
    from hand_sampler.robot_spec import is_population_ref, population_from_ref
    if is_population_ref(name):
        return population_from_ref(name).spec
    raise KeyError(
        f"unknown robot_spec {name!r}; registered: {sorted(REGISTRY)}, "
        "a generated population gen_s<seed>_n<count>, or a path to a "
        "population .json written by hand_sampler.population_io")


__all__ = ["RobotSpec", "REGISTRY", "get_robot_spec", "SHARPA_IIWA14"]
