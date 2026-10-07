# wuji2_left_replica_palm

`hand_sampler.commercial.fit('wuji2')` of `assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.urdf`: 5 fingers, 20 joints, palm 35 x 76 x 88 mm.

How it differs from the vendor's hand:

* bases spread to 50 mm rather than the 35 mm floor: at the floor this hand's LINKS still met
* l_index_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_middle_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_ring_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_pinky_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_index_finger_mcp_flex: base 15 mm from where the vendor puts it, after the grid and the motor floor
* l_index_finger_mcp_flex: base 3 mm off the palm midplane, which a mount cannot express
* l_middle_finger_mcp_flex: base 20 mm from where the vendor puts it, after the grid and the motor floor
* l_middle_finger_mcp_flex: base 8 mm off the palm midplane, which a mount cannot express
* l_ring_finger_mcp_flex: base 29 mm from where the vendor puts it, after the grid and the motor floor
* l_ring_finger_mcp_flex: base 5 mm off the palm midplane, which a mount cannot express
* l_pinky_mcp_flex: base 34 mm from where the vendor puts it, after the grid and the motor floor
* l_thumb_cmc_flex: base 11 mm from where the vendor puts it, after the grid and the motor floor
* l_thumb_cmc_flex: base 5 mm off the palm midplane, which a mount cannot express
