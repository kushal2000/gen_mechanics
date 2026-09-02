"""Self-collision filter pairs for the left SHARPA hand on a KUKA iiwa14.

Link pairs listed here have their self-collision *filtered out*. PhysX already
auto-filters directly-jointed parent/child pairs, so what earns its place is the
non-kinematic-neighbour pairs: palm-to-proximal-phalanx and MC-to-PP, which
overlap geometrically at the knuckles without being joined.

Names are POST-``merge_fixed_joints``. That is why the palm appears as
``iiwa14_link_7`` rather than the URDF's ``left_hand_C_MC`` — the importer
collapses link_ee -> sharpa_mount -> left_hand_C_MC into the arm's last link.

The arm chain is imported from ``iiwa14_arm`` rather than restated, so it stays
identical across hands along with the rest of the arm.

Ported from simtoolreal ``isaacgymenvs/tasks/simtoolreal/adjacent_links.py``.
Only its LEFT map is used: simtoolreal merged LEFT+RIGHT defensively, but the
right-hand links match no prim in a left-hand URDF and were silently skipped.
Pairs are listed in both directions, matching the original.
"""

from __future__ import annotations

from hand_sampler.iiwa14_arm import ARM_ADJACENT_LINKS


_HAND_ADJACENT_LINKS: dict[str, list[str]] = {
    "iiwa14_link_7": [
        "iiwa14_link_6",
        "left_thumb_CMC_VL",
        "left_thumb_MC",
        "left_index_MCP_VL",
        "left_index_PP",
        "left_middle_MCP_VL",
        "left_middle_PP",
        "left_ring_MCP_VL",
        "left_ring_PP",
        "left_pinky_MC",
    ],
    "left_index_MCP_VL": ["iiwa14_link_7", "left_index_PP"],
    "left_index_PP": ["iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"],
    "left_index_MP": ["left_index_PP", "left_index_DP"],
    "left_index_DP": ["left_index_MP"],
    "left_middle_MCP_VL": ["iiwa14_link_7", "left_middle_PP"],
    "left_middle_PP": ["iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"],
    "left_middle_MP": ["left_middle_PP", "left_middle_DP"],
    "left_middle_DP": ["left_middle_MP"],
    "left_pinky_MC": ["iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"],
    "left_pinky_MCP_VL": ["left_pinky_MC", "left_pinky_PP"],
    "left_pinky_PP": ["left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"],
    "left_pinky_MP": ["left_pinky_PP", "left_pinky_DP"],
    "left_pinky_DP": ["left_pinky_MP"],
    "left_ring_MCP_VL": ["iiwa14_link_7", "left_ring_PP"],
    "left_ring_PP": ["iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"],
    "left_ring_MP": ["left_ring_PP", "left_ring_DP"],
    "left_ring_DP": ["left_ring_MP"],
    "left_thumb_CMC_VL": ["iiwa14_link_7", "left_thumb_MC"],
    "left_thumb_MC": [
        "iiwa14_link_7",
        "left_thumb_CMC_VL",
        "left_thumb_MCP_VL",
        "left_thumb_PP",
    ],
    "left_thumb_MCP_VL": ["left_thumb_MC", "left_thumb_PP"],
    "left_thumb_PP": ["left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"],
    "left_thumb_DP": ["left_thumb_PP"],
}

# Shared arm chain + this hand's palm/finger pairs.
SHARPA_IIWA14_ADJACENT_LINKS: dict[str, list[str]] = {
    **ARM_ADJACENT_LINKS,
    **_HAND_ADJACENT_LINKS,
}


__all__ = ["SHARPA_IIWA14_ADJACENT_LINKS"]
