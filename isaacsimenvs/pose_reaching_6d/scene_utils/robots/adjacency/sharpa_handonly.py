"""Self-collision filter pairs for the hand-only SHARPA.

The arm-mounted map, with the arm removed and the palm called by its own name.

``sharpa_iiwa14``'s map lists the palm as ``iiwa14_link_7``, because there the palm
merges into the flange under ``merge_fixed_joints``. The unified hand-only URDF is
rooted at the palm, so it appears as ``left_hand_C_MC`` and there is no arm chain to
merge in. The palm's neighbours are otherwise the same links, derived from the same
source map rather than retyped, so the two cannot drift apart.

``_apply_self_collision_filters`` runs strict and raises if the map matches nothing,
which is documented there as the swap-a-new-hand failure mode.
"""

from __future__ import annotations

from isaacsimenvs.pose_reaching_6d.scene_utils.robots.adjacency.sharpa_iiwa14 import (
    SHARPA_IIWA14_ADJACENT_LINKS,
)

PALM_BODY = "left_hand_C_MC"
_FLANGE = "iiwa14_link_7"


def _hand_only(pairs: dict[str, list[str]]) -> dict[str, list[str]]:
    """Drop the arm links and rename the flange to the palm.

    Derived rather than restated: a hand-only map that was typed out separately would
    be one edit away from disagreeing with the arm-mounted one about which of this
    hand's links overlap, and only one of the two would ever be exercised.
    """
    out: dict[str, list[str]] = {}
    for link, neighbours in pairs.items():
        if link.startswith("iiwa14_") and link != _FLANGE:
            continue                                   # an arm link; there is no arm
        key = PALM_BODY if link == _FLANGE else link
        kept = [PALM_BODY if n == _FLANGE else n
                for n in neighbours if not n.startswith("iiwa14_")]
        if kept:
            out[key] = kept
    return out


SHARPA_HANDONLY_ADJACENT_LINKS: dict[str, list[str]] = _hand_only(SHARPA_IIWA14_ADJACENT_LINKS)

__all__ = ["SHARPA_HANDONLY_ADJACENT_LINKS"]
