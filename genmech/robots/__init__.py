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
from genmech.robots.allegro_iiwa14 import ALLEGRO_IIWA14
from genmech.robots.sharpa_iiwa14 import SHARPA_IIWA14


REGISTRY: dict[str, RobotSpec] = {
    spec.name: spec
    for spec in (
        SHARPA_IIWA14,
        ALLEGRO_IIWA14,
    )
}


def get_robot_spec(name: str) -> RobotSpec:
    """Look up a spec by name, or fail with the list of valid names.

    Procedurally generated hands are not in ``REGISTRY``: there are too many of
    them, and their names encode their parameters, so the spec is synthesised on
    demand instead. ``gen_sharpa_like`` is the measured SHARPA reference vector;
    ``gen_<seed>_<index>`` is that element of ``sample_population(seed)``. The
    URDF is written on first use. See ``genmech/robots/generated/synth_spec.py``.
    """
    if name in REGISTRY:
        return REGISTRY[name]

    # Imported lazily: the generated path pulls in the URDF builder and the
    # object pipeline's inertia helper, and this module must stay importable
    # without Isaac Sim for the offline tools.
    from genmech.robots.generated.synth_spec import (
        get_generated_spec, is_generated_name,
    )

    if is_generated_name(name):
        return get_generated_spec(name)

    raise KeyError(
        f"unknown robot_spec {name!r}; registered: {sorted(REGISTRY)}, "
        f"or a generated name (gen_sharpa_like, gen_<seed>_<index>)"
    )


__all__ = ["RobotSpec", "REGISTRY", "get_robot_spec", "SHARPA_IIWA14", "ALLEGRO_IIWA14"]
