"""Self-collision filter pairs for the hand-only Allegro.

Names are POST-``merge_fixed_joints``, but for this hand that is a no-op: the
unified URDF is rooted at ``palm_link`` and its only joints are the 16 revolute
ones, so nothing collapses and every name here is a link as written.

That is the one real difference from the arm-mounted version deleted in e4eb91a,
which had the importer collapse ``iiwa14_link_ee -> allegro_mount -> palm_link``
into ``iiwa14_link_7`` and so listed the palm under the arm's last link name.
There is no arm here, so there is no ``ARM_ADJACENT_LINKS`` either.

PhysX already auto-filters directly-jointed parent/child pairs. The entries that
earn their place are palm-to-finger: each finger's ``link_0`` and ``link_1`` sits
inside the palm shell and overlaps the slab collider geometrically. The per-finger
chains are listed anyway, to mirror the shape of the SHARPA map.

``_apply_self_collision_filters`` runs strict and raises if the map matches
nothing, which is what catches a typo here. Without that, a name that matched
nothing would leave the hand with self-collisions fully enabled and no masking,
and it explodes at reset rather than at build.
"""

from __future__ import annotations

PALM_BODY = "palm_link"
_FINGERS = ("index", "middle", "ring", "thumb")

_ADJACENT: dict[str, list[str]] = {
    PALM_BODY: [f"{f}_link_{d}" for f in _FINGERS for d in (0, 1)],
}
for _f in _FINGERS:
    _ADJACENT[f"{_f}_link_0"] = [PALM_BODY, f"{_f}_link_1"]
    _ADJACENT[f"{_f}_link_1"] = [PALM_BODY, f"{_f}_link_0", f"{_f}_link_2"]
    _ADJACENT[f"{_f}_link_2"] = [f"{_f}_link_1", f"{_f}_link_3"]
    _ADJACENT[f"{_f}_link_3"] = [f"{_f}_link_2"]

# Neighbouring fingers sit shoulder to shoulder on the palm: measured, the index,
# middle and ring roots are 45.1 mm apart in the palm's width while a finger base
# mesh spans 19.6 mm, so the bases themselves clear -- but they are close enough
# that a spread of 32 degrees each way brings them into contact, and self-collision
# between two fingers of the same hand at rest is not information the policy needs.
for _a, _b in (("index", "middle"), ("middle", "ring")):
    _ADJACENT[f"{_a}_link_0"].append(f"{_b}_link_0")
    _ADJACENT[f"{_b}_link_0"].append(f"{_a}_link_0")

# The thumb opposes the fingers across the palm, 88 mm proximal of the finger roots
# and 18 mm deeper into the slab. Its base is nowhere near theirs, so it is NOT
# filtered against them: a thumb-to-finger contact is exactly the contact a grasp is
# made of, and filtering it would let the hand close through itself.
del _f, _a, _b

ALLEGRO_HANDONLY_ADJACENT_LINKS: dict[str, list[str]] = dict(_ADJACENT)

__all__ = ["ALLEGRO_HANDONLY_ADJACENT_LINKS"]
