# E13: representation check (5 mm / 10 deg) and real-hand atlas

15 available hands, 15 PASS, 0 FAIL. Tolerances: 5 mm / 10 deg. Seed 20260926, 64 random configs + the zero config, every original movable joint (including mimic/coupled ones) sampled independently within its own declared limits.

## Per-hand results

| hand | split | available | pass | max pos (mm) | max axis (deg) | max tip (mm) | Pinocchio-export max err (mm/deg) | DOF orig/derived | notes |
|---|---|---|---|---|---|---|---|---|---|
| allegro_right | dev | available | PASS | 8.55e-14 | 8.54e-07 | 8.89e-14 | 5.72e-14 mm / 1.21e-06 deg | 16/16 | - |
| leap_right | dev | available | PASS | 6.55e-14 | 1.21e-06 | 1.06e-13 | 9.54e-14 mm / 1.21e-06 deg | 16/16 | - |
| barrett_bh | dev | available | PASS | 7.47e-14 | 8.54e-07 | - | 6.21e-14 mm / 0 deg | 8/8 | max tip (mm): n/a, no defined fingertip to compare (excluded from pass/fail); fingertip_undefined: ['finger_1_dist_link', 'finger_2_dist_link', 'finger_3_dist_link'] |
| ability_right | dev | available | PASS | 5.78e-14 | 8.54e-07 | 8.33e-14 | 3.94e-14 mm / 0 deg | 6/10 | coupling_as_independent: ['index_q2', 'middle_q2', 'ring_q2', 'pinky_q2'] |
| inspire_right | dev | available | PASS | 6.87e-14 | 1.21e-06 | 8.47e-14 | 1.2e-13 mm / 0 deg | 6/12 | coupling_as_independent: ['thumb_intermediate_joint', 'thumb_distal_joint', 'index_intermediate_joint', 'middle_intermediate_joint', 'ring_intermediate_joint', 'pinky_intermediate_joint'] |
| dclaw | heldout | available | PASS | 1.11e-13 | 1.21e-06 | 1.69e-13 | 1.13e-13 mm / 1.21e-06 deg | 9/9 | - |
| wuji_right | heldout | available | PASS | 6.55e-14 | 1.21e-06 | 9.32e-14 | 6.09e-14 mm / 0 deg | 20/20 | - |
| xhand_right | heldout | available | PASS | 5.93e-14 | 1.21e-06 | 8.33e-14 | 5.61e-14 mm / 0 deg | 12/12 | - |
| tesollo_dg5f_right | heldout | available | PASS | 8.51e-14 | 1.21e-06 | 1.13e-13 | 6.55e-14 mm / 1.21e-06 deg | 20/20 | - |
| orca_right | heldout | available | PASS | 3.53e-14 | 1.21e-06 | 5.05e-14 | 4.71e-14 mm / 1.21e-06 deg | 16/16 | - |
| sharpa_left_on_iiwa14 | heldout | available | PASS | 7.24e-14 | 8.54e-07 | 1.72e-13 | 3.61e-13 mm / 1.21e-06 deg | 22/22 | - |
| shadow_right_local | articulated_palm | available | PASS | 8.4e-14 | 1.21e-06 | 8.56e-14 | 8.58e-14 mm / 1.21e-06 deg | 22/22 | - |
| svh_right | articulated_palm | available | PASS | 8.42e-14 | 1.21e-06 | 6.97e-14 | 6.25e-14 mm / 8.54e-07 deg | 9/20 | coupling_as_independent: ['right_hand_j5', 'right_hand_j3', 'right_hand_j4', 'right_hand_index_spread', 'right_hand_j14', 'right_hand_j15', 'right_hand_ring_spread', 'right_hand_j12', 'right_hand_j16', 'right_hand_j13', 'right_hand_j17'] |
| arms_skel | articulated_palm | available | PASS | 6.25e-14 | 1.71e-06 | - | 5.43e-14 mm / 1.21e-06 deg | 21/22 | max tip (mm): n/a, no defined fingertip to compare (excluded from pass/fail); fingertip_undefined: ['5distph', '4distph', 'distal_thumb', '2distph', '3distph']; coupling_as_independent: ['CMC5'] |
| coupled_finger | analytic | available | PASS | 0 | 0 | - | 1.39e-14 mm / 0 deg | 1/3 | max tip (mm): n/a, no defined fingertip to compare (excluded from pass/fail); fingertip_undefined: ['link3']; coupling_as_independent: ['joint2', 'joint3'] |

## Atlas highlights (pooled over passing/scored hands)

Hands pooled: 14. Knuckles/palm segments under 5 mm: 31. Couplings kept as `Coupled` phalanx modules: 0 (always 0 -- I22 decision: one motor per joint, projection never emits ``Coupled``). Mimic joints emitted as independent joints (`coupling_as_independent`): 22.

## Grammar gap table (real-hand values vs `DEFAULT_DISTRIBUTION`)

| parameter | real min | real median | real max | n | grammar range/choices | frac. outside |
|---|---|---|---|---|---|---|
| digit_count (per hand) | 3 | 5 | 5 | 14 | range (1, 6) | 0.00 |
| palm_body_count (non-root, per hand) | 0 | 0 | 2 | 14 | range (0, 3) | 0.00 |
| phalanx_count (per digit) | 2 | 4 | 5 | 64 | range (1, 6) | 0.00 |
| phalanx link length (m) | 0 | 0.0321 | 0.084 | 220 | range (0.015, 0.08), grid 0.005 m | 0.18 |
| palm/root segment length (m) -- NOTE: depends on the source URDF's own root-link origin placement, not a hand-intrinsic quantity (I22 fix 6) | 0.00964 | 0.0928 | 0.144 | 19 | range (0.02, 0.08) | 0.68 |
| mount_frac | -0.377 | 0.943 | 1 | 69 | choices (0.0, 0.25, 0.5, 0.75, 1.0) | 0.01 |
| lateral mount offset, digit + palm-body mounts only (m) | 6.04e-19 | 0.0225 | 0.0617 | 69 | choices ((0.0, 0.0),) (mount_offset has no grammar production at all); I22 fix 6: continuation phalanges' own bend offset excluded from this row (a different quantity, see the rest-bend rows) | 0.94 |
| physical bend angle, continuation phalanges (deg): angle between consecutive link directions | 0 | 0.868 | 99 | 129 | choices ((0.0, 0.0, 0.0),) (bend_probability=0.0) | 0.57 |
| bend_rpy rotation angle, continuation phalanges (deg) [sanity check vs the row above -- I22 fix 2's gauge should make these close] | 0 | 1.19 | 99 | 156 | choices ((0.0, 0.0, 0.0),) (bend_probability=0.0) | 0.58 |
| axis-to-link angle, phalanges (deg, folded to [0,90]; 90=perpendicular) | 0 | 90 | 90 | 220 | axis drawn on a 15-degree spherical grid (continuous elevation/azimuth choice set, not a perpendicularity constraint) | - |
| palm-joint axis vs root +z (deg, folded to [0,90]) | 5.51 | 27 | 34.8 | 5 | same 15-degree spherical grid as phalanx axes | - |
| DH common-normal length a, consecutive phalanx joint pairs (m) | 0 | 0.0315 | 0.07 | 156 | not modeled: derive() has no per-pair common-normal production (gauge-free geometry, I22 fix 6) | - |
| DH twist alpha, consecutive phalanx joint pairs (deg, folded to [0,90]) | 0 | 8.11 | 90 | 156 | not modeled directly (axes are drawn independently per phalanx, not as a twist relative to the previous joint) | - |
| DH-like offset d, consecutive phalanx joint pairs (m; simplified pairwise projection, see _dh_pair) | -0.046 | 0 | 0.042 | 156 | not modeled directly | - |
| revolute/palm joint limit lo (deg) | -180 | -12 | 15.1 | 225 | choice-set range [-45.0, 0.0] or continuous range (-45.0, 110.0) if limits_continuous | 0.15 |
| revolute/palm joint limit hi (deg) | 0 | 89.4 | 180 | 225 | choice-set range [20.0, 110.0] or continuous range (-45.0, 110.0) if limits_continuous | 0.19 |
| mimic joints per hand (count; I22 decision: coupled/mimic joints are a future item, the projection now emits every one as an independent joint) | 0 | 0 | 11 | 14 | not modeled (0 by construction: derive() productions this experiment samples never emit a Coupled module from real-hand projection) | - |

This table is descriptive input for later grammar tuning (widening ranges/choice sets, adding a rest-bend/lateral-offset production with nonzero default probability, etc.) -- no grammar file changed as part of this experiment.
