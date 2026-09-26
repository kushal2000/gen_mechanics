# E13: representation check (5 mm / 10 deg) and real-hand atlas

15 available hands, 15 PASS, 0 FAIL. Tolerances: 5 mm / 10 deg. Seed 20260926, 64 random configs + the zero config, every original movable joint (including mimic/coupled ones) sampled independently within its own declared limits.

## Per-hand results

| hand | split | available | pass | max pos (mm) | max axis (deg) | max tip (mm) | notes |
|---|---|---|---|---|---|---|---|
| allegro_right | dev | available | PASS | 1.02e-13 | 8.54e-07 | 1.01e-13 | - |
| leap_right | dev | available | PASS | 1.38e-13 | 1.21e-06 | 1.57e-13 | - |
| barrett_bh | dev | available | PASS | 1.14e-13 | 1.21e-06 | 0 | fingertip_undefined: ['finger_1_dist_link', 'finger_2_dist_link', 'finger_3_dist_link'] |
| ability_right | dev | available | PASS | 7.62e-14 | 8.54e-07 | 9.83e-14 | - |
| inspire_right | dev | available | PASS | 7.35e-14 | 8.54e-07 | 9.44e-14 | - |
| dclaw | heldout | available | PASS | 6.94e-14 | 1.21e-06 | 1.05e-13 | - |
| wuji_right | heldout | available | PASS | 8.92e-14 | 1.21e-06 | 9.13e-14 | - |
| xhand_right | heldout | available | PASS | 1.97e-13 | 1.21e-06 | 3.07e-13 | - |
| tesollo_dg5f_right | heldout | available | PASS | 8.8e-14 | 1.21e-06 | 1.17e-13 | - |
| orca_right | heldout | available | PASS | 1.67e-13 | 8.54e-07 | 1.71e-13 | - |
| sharpa_left_on_iiwa14 | heldout | available | PASS | 8.09e-14 | 8.54e-07 | 8.89e-14 | - |
| shadow_right_local | articulated_palm | available | PASS | 8.34e-14 | 8.54e-07 | 8.33e-14 | - |
| svh_right | articulated_palm | available | PASS | 9.98e-14 | 1.21e-06 | 1e-13 | coupling_not_in_structure: ['right_hand_j5', 'right_hand_index_spread', 'right_hand_ring_spread'] |
| arms_skel | articulated_palm | available | PASS | 6.95e-14 | 1.71e-06 | 0 | fingertip_undefined: ['5distph', '4distph', 'distal_thumb', '2distph', '3distph']; coupling_not_in_structure: ['CMC5'] |
| coupled_finger | analytic | available | PASS | 1.55e-14 | 0 | 0 | fingertip_undefined: ['link3'] |

## Atlas highlights (pooled over passing/scored hands)

Hands pooled: 15. Knuckles/palm segments under 5 mm: 32. Couplings kept as `Coupled` phalanx modules: 20. Couplings reported `coupling_not_in_structure`: 4.

## Grammar gap table (real-hand values vs `DEFAULT_DISTRIBUTION`)

| parameter | real min | real median | real max | n | grammar range/choices | frac. outside |
|---|---|---|---|---|---|---|
| digit_count (per hand) | 1 | 5 | 5 | 15 | range (1, 6) | 0.00 |
| palm_body_count (non-root, per hand) | 0 | 0 | 2 | 15 | range (0, 3) | 0.00 |
| phalanx_count (per digit) | 2 | 4 | 5 | 65 | range (1, 6) | 0.00 |
| phalanx link length (m) | 0 | 0.032 | 0.084 | 223 | range (0.015, 0.08), grid 0.005 m | 0.18 |
| palm/root segment length (m) | 0.00964 | 0.0928 | 0.151 | 21 | range (0.02, 0.08) | 0.67 |
| mount_frac | 0.00449 | 0.944 | 1 | 71 | choices (0.0, 0.25, 0.5, 0.75, 1.0) | 0.00 |
| lateral mount offset (mount_offset / bend_offset, m) | 0 | 4.63e-19 | 0.0617 | 229 | choices ((0.0, 0.0),) (mount_offset has no grammar production at all) | 0.28 |
| rest bend angle, continuation phalanges (deg) | 0 | 18.9 | 180 | 158 | choices ((0.0, 0.0, 0.0),) (bend_probability=0.0) | 0.70 |
| axis-to-link angle, phalanges (deg; 90=perpendicular) | 0 | 90 | 180 | 223 | axis drawn on a 15-degree spherical grid (continuous elevation/azimuth choice set, not a perpendicularity constraint) | - |
| palm-joint axis vs root +z (deg) | 6.87 | 32 | 174 | 6 | same 15-degree spherical grid as phalanx axes | - |
| revolute/palm joint limit lo (deg) | -180 | -15 | 15.1 | 209 | choice-set range [-45.0, 0.0] or continuous range (-45.0, 110.0) if limits_continuous | 0.17 |
| revolute/palm joint limit hi (deg) | 0 | 90 | 180 | 209 | choice-set range [20.0, 110.0] or continuous range (-45.0, 110.0) if limits_continuous | 0.20 |
| prismatic joint limit lo (m) | - | - | - | 0 | choice-set range [-0.010, 0.000] | - |
| prismatic joint limit hi (m) | - | - | - | 0 | choice-set range [0.010, 0.030] | - |
| coupling multiplier | -1 | 1.06 | 1.5 | 20 | choices (1.0, 0.5, 0.75, -0.5, -1.0, 1.5) | 0.90 |
| coupling offset (rad) | -0.0454 | 0 | 0.1 | 20 | choices (0.0, 0.1, -0.1, 0.2, -0.2) | 0.20 |

This table is descriptive input for later grammar tuning (widening ranges/choice sets, adding a rest-bend/lateral-offset production with nonzero default probability, etc.) -- no grammar file changed as part of this experiment.
