# allegro_left_replica

`hand_sampler.commercial.fit('allegro')` of `assets/urdf/unified_dynamics_commercial_hands/allegro/allegro_left.urdf`: 4 fingers, 16 joints, palm 25 x 100 x 85 mm.

How it differs from the vendor's hand:

* joint_0_0: link 0 16.4 -> 20 mm (the 20 mm floor)
* joint_4_0: link 0 16.4 -> 20 mm (the 20 mm floor)
* joint_8_0: link 0 16.4 -> 20 mm (the 20 mm floor)
* joint_0_0: base 3 mm from where the vendor puts it, after the grid and the motor floor
* joint_4_0: base 1 mm from where the vendor puts it, after the grid and the motor floor
* joint_8_0: base 2 mm from where the vendor puts it, after the grid and the motor floor
* joint_12_0: base 3 mm from where the vendor puts it, after the grid and the motor floor
* joint_12_0: base 18 mm off the palm midplane, which a mount cannot express
