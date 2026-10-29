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

import functools

from hand_sampler.robot_spec import RobotSpec
from isaacsimenvs.common.robots.allegro_handonly import ALLEGRO_HANDONLY
from isaacsimenvs.common.robots.sharpa_handonly import SHARPA_HANDONLY
from isaacsimenvs.common.robots.sharpa_iiwa14 import SHARPA_IIWA14
from isaacsimenvs.common.robots.unified_hands import UNIFIED_LEFT
from isaacsimenvs.common.robots.unified_dynamics_hands import MISSING_FINGER, UNIFORM_LEFT, UNIFORM_RIGHT


REGISTRY: dict[str, RobotSpec] = {
    spec.name: spec
    for spec in (SHARPA_IIWA14, SHARPA_HANDONLY, ALLEGRO_HANDONLY, *UNIFIED_LEFT.values(),
                 *UNIFORM_LEFT.values(), *UNIFORM_RIGHT.values(), *MISSING_FINGER.values())
}


@functools.lru_cache(maxsize=None)
def _spec_from_file(path: str) -> RobotSpec:
    """A RobotSpec from a spec JSON on disk, named by its path (so the network resolves the same layout from the
    same config value). Honours the JSON's ``replace_cylinders_with_capsules`` (capsule-built hands)."""
    import dataclasses
    import json
    from pathlib import Path

    from isaacsimenvs.common.robots.unified_hands import REPO, spec_from_json
    p = Path(path) if Path(path).is_absolute() else REPO / path
    if not p.is_file():
        raise KeyError(f"robot_spec {path!r}: no such spec JSON")
    spec = spec_from_json(p, name=path)
    caps = bool(json.loads(p.read_text()).get("replace_cylinders_with_capsules", False))
    return dataclasses.replace(spec, replace_cylinders_with_capsules=caps)


def get_robot_spec(name: str) -> RobotSpec:
    """Look up a spec by name, or fail with the list of valid names."""
    if name in REGISTRY:
        return REGISTRY[name]
    # Several hands in one scene: a padded template over their specs (robots/multi_hand.py). Before the spec-JSON
    # case: "multi:a.spec.json+b.spec.json" also ends in .spec.json.
    from isaacsimenvs.common.robots.multi_hand import hand_set, is_multi_ref
    if is_multi_ref(name):
        return hand_set(name).template
    # A hand described by its own spec JSON (e.g. a generated primitive hand,
    # experiments/14oct_design_search/primitive_hands.py): a path, absolute or from the repo.
    if name.endswith(".spec.json"):
        return _spec_from_file(name)
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


__all__ = ["ALLEGRO_HANDONLY", "RobotSpec", "REGISTRY", "get_robot_spec",
           "SHARPA_HANDONLY", "SHARPA_IIWA14"]
