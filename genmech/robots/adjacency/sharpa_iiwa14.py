"""Self-collision filter pairs for the left SHARPA hand on a KUKA iiwa14.

Link pairs listed here have their self-collision *filtered out*; PhysX already
auto-filters directly-jointed parent/child pairs, so the value here is the
non-kinematic-neighbor pairs (palm-to-proximal-phalanx, MC-to-PP).

Ported from simtoolreal ``isaacgymenvs/tasks/simtoolreal/adjacent_links.py``,
merging its LEFT and RIGHT maps (the RIGHT entries add only the shared iiwa14
arm chain for a left-hand URDF; right-hand links simply find no prim).
Pairs are listed in both directions, matching the original.
"""

from __future__ import annotations


SHARPA_IIWA14_ADJACENT_LINKS: dict[str, list[str]] = {
    "iiwa14_link_0": ["iiwa14_link_1"],
    "iiwa14_link_1": ["iiwa14_link_0", "iiwa14_link_2"],
    "iiwa14_link_2": ["iiwa14_link_1", "iiwa14_link_3"],
    "iiwa14_link_3": ["iiwa14_link_2", "iiwa14_link_4"],
    "iiwa14_link_4": ["iiwa14_link_3", "iiwa14_link_5"],
    "iiwa14_link_5": ["iiwa14_link_4", "iiwa14_link_6"],
    "iiwa14_link_6": ["iiwa14_link_5", "iiwa14_link_7"],
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
        "right_index_MCP_VL",
        "right_middle_MCP_VL",
        "right_pinky_MC",
        "right_ring_MCP_VL",
        "right_thumb_CMC_VL",
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
    "right_index_MCP_VL": ["iiwa14_link_7", "right_index_PP"],
    "right_index_PP": ["right_index_MCP_VL", "right_index_MP"],
    "right_index_MP": ["right_index_PP", "right_index_DP"],
    "right_index_DP": ["right_index_MP"],
    "right_middle_MCP_VL": ["iiwa14_link_7", "right_middle_PP"],
    "right_middle_PP": ["right_middle_MCP_VL", "right_middle_MP"],
    "right_middle_MP": ["right_middle_PP", "right_middle_DP"],
    "right_middle_DP": ["right_middle_MP"],
    "right_pinky_MC": ["iiwa14_link_7", "right_pinky_MCP_VL"],
    "right_pinky_MCP_VL": ["right_pinky_MC", "right_pinky_PP"],
    "right_pinky_PP": ["right_pinky_MCP_VL", "right_pinky_MP"],
    "right_pinky_MP": ["right_pinky_PP", "right_pinky_DP"],
    "right_pinky_DP": ["right_pinky_MP"],
    "right_ring_MCP_VL": ["iiwa14_link_7", "right_ring_PP"],
    "right_ring_PP": ["right_ring_MCP_VL", "right_ring_MP"],
    "right_ring_MP": ["right_ring_PP", "right_ring_DP"],
    "right_ring_DP": ["right_ring_MP"],
    "right_thumb_CMC_VL": ["iiwa14_link_7", "right_thumb_MC"],
    "right_thumb_MC": ["right_thumb_CMC_VL", "right_thumb_MCP_VL"],
    "right_thumb_MCP_VL": ["right_thumb_MC", "right_thumb_PP"],
    "right_thumb_PP": ["right_thumb_MCP_VL", "right_thumb_DP"],
    "right_thumb_DP": ["right_thumb_PP"],
}


__all__ = ["SHARPA_IIWA14_ADJACENT_LINKS"]
