# leap_left_replica

`hand_sampler.commercial.fit('leap')` of `assets/urdf/unified_dynamics_commercial_hands/leap/leap_left.urdf`: 4 fingers, 16 joints, palm 25 x 100 x 85 mm.

How it differs from the vendor's hand:

* j1: link 1 15.0 -> 20 mm (the 20 mm floor)
* j5: link 1 15.0 -> 20 mm (the 20 mm floor)
* j9: link 1 15.0 -> 20 mm (the 20 mm floor)
* j12: link 0 15.6 -> 20 mm (the 20 mm floor)
* j12: link 1 15.7 -> 20 mm (the 20 mm floor)
* j1: base 2 mm from where the vendor puts it, after the grid and the motor floor
* j1: base 6 mm off the palm midplane, which a mount cannot express
* j5: base 2 mm from where the vendor puts it, after the grid and the motor floor
* j5: base 6 mm off the palm midplane, which a mount cannot express
* j9: base 1 mm from where the vendor puts it, after the grid and the motor floor
* j9: base 6 mm off the palm midplane, which a mount cannot express
* j12: base 2 mm from where the vendor puts it, after the grid and the motor floor
* j12: base 2 mm off the palm midplane, which a mount cannot express
