# midas_replica

`hand_sampler.commercial.fit('midas')` of `assets/urdf/unified_dynamics_commercial_hands/midas/midas.urdf`: 4 fingers, 16 joints, palm 25 x 80 x 70 mm.

How it differs from the vendor's hand:

* index_mcp_abad_joint: base 6 mm from where the vendor puts it, after the grid and the motor floor
* index_mcp_abad_joint: base 9 mm off the palm midplane, which a mount cannot express
* middle_mcp_abad_joint: base 4 mm from where the vendor puts it, after the grid and the motor floor
* middle_mcp_abad_joint: base 9 mm off the palm midplane, which a mount cannot express
* ring_mcp_abad_joint: base 3 mm from where the vendor puts it, after the grid and the motor floor
* ring_mcp_abad_joint: base 9 mm off the palm midplane, which a mount cannot express
* thumb_cmc_roll_joint: base 11 mm off the palm midplane, which a mount cannot express
