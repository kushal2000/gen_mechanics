"""Self-collision filter pairs for the Allegro hand on a KUKA iiwa14.

Names are POST-``merge_fixed_joints``: the importer collapses
``iiwa14_link_ee -> allegro_mount -> palm_link`` into ``iiwa14_link_7``, exactly
as it collapses SHARPA's mount chain, so the palm appears here as the arm's last
link.

PhysX already auto-filters directly-jointed parent/child pairs, so the entries
that earn their place are the palm-to-finger-base ones: each finger's ``link_0``
sits inside the palm shell and overlaps it geometrically without being its
kinematic child in a way PhysX filters. The per-finger chain is listed anyway to
mirror the SHARPA map's shape.

The arm chain is imported from ``iiwa14_arm`` rather than restated, so it stays
identical across hands (docs/methodology.md §1).

Authored by hand — unlike SHARPA's, which was ported from simtoolreal. The
``strict`` check in ``_apply_self_collision_filters`` is what catches a typo
here: a map that matches nothing would otherwise leave the hand with
self-collisions enabled and no masking, which explodes at reset.
"""

from __future__ import annotations

from hand_sampler.iiwa14_arm import ARM_ADJACENT_LINKS, ARM_TIP_LINK


_FINGERS = ("index", "middle", "ring", "thumb")

# Palm <-> every finger base, both directions.
_HAND_ADJACENT_LINKS: dict[str, list[str]] = {
    ARM_TIP_LINK: ["iiwa14_link_6"] + [f"{f}_link_0" for f in _FINGERS]
                  + [f"{f}_link_1" for f in _FINGERS],
}

for _f in _FINGERS:
    _HAND_ADJACENT_LINKS[f"{_f}_link_0"] = [ARM_TIP_LINK, f"{_f}_link_1"]
    _HAND_ADJACENT_LINKS[f"{_f}_link_1"] = [ARM_TIP_LINK, f"{_f}_link_0", f"{_f}_link_2"]
    _HAND_ADJACENT_LINKS[f"{_f}_link_2"] = [f"{_f}_link_1", f"{_f}_link_3"]
    _HAND_ADJACENT_LINKS[f"{_f}_link_3"] = [f"{_f}_link_2"]

# Neighbouring fingers' bases sit shoulder to shoulder on the palm.
for _a, _b in (("index", "middle"), ("middle", "ring")):
    _HAND_ADJACENT_LINKS[f"{_a}_link_0"].append(f"{_b}_link_0")
    _HAND_ADJACENT_LINKS[f"{_b}_link_0"].append(f"{_a}_link_0")

del _f, _a, _b

ALLEGRO_IIWA14_ADJACENT_LINKS: dict[str, list[str]] = {
    **ARM_ADJACENT_LINKS,
    **_HAND_ADJACENT_LINKS,
}


__all__ = ["ALLEGRO_IIWA14_ADJACENT_LINKS"]
