# wuji2_left_replica_limits

`hand_sampler.commercial.fit('wuji2')` of `assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.urdf`: 5 fingers, 20 joints, palm 25 x 100 x 75 mm.

How it differs from the vendor's hand:

* l_index_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_middle_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_ring_finger_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_pinky_mcp_flex: link 0 14.5 -> 20 mm (the 20 mm floor)
* l_index_finger_mcp_flex: base 10 mm from where the vendor puts it, after the grid and the motor floor
* l_index_finger_mcp_flex: base 4 mm off the palm midplane, which a mount cannot express
* l_middle_finger_mcp_flex: base 14 mm from where the vendor puts it, after the grid and the motor floor
* l_middle_finger_mcp_flex: base 1 mm off the palm midplane, which a mount cannot express
* l_ring_finger_mcp_flex: base 6 mm from where the vendor puts it, after the grid and the motor floor
* l_pinky_mcp_flex: base 21 mm from where the vendor puts it, after the grid and the motor floor
* l_pinky_mcp_flex: base 4 mm off the palm midplane, which a mount cannot express
* l_thumb_cmc_flex: base 1 mm from where the vendor puts it, after the grid and the motor floor
* l_thumb_cmc_flex: base 7 mm off the palm midplane, which a mount cannot express
