# Hand-kinematics grammar: iteration-4 coverage pilot

Generated 2026-09-26T04:10:23.994170+00:00 at git SHA `b194651ee288a119c3a6db59f05c2a7cc6fdda50`, seed 20260925.

For every hand: whether we could import it, whether our forward kinematics matches an independent oracle (Pinocchio) within tolerance, and whether the grammar's own support audit (`hand_sampler.grammar.coverage`) judges the imported model expressible by the grammar's productions and inside the default distribution's sampled ranges.

| id | family | split | availability | movable joints | couplings | digit count | digit count source | fidelity (max pos / max rot) | fidelity pass | topology_expressible | in_support | missing constructs | out-of-support items |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| allegro_right | F1_four_finger_serial | dev | available | 16 | 0 | 4 | root_chains | 5.551e-17 m / 1.888e-16 rad (frozen) | yes | no | no | continuation_pose:joint_13 | limits_not_in_set:joint_0,joint_1,joint_10,joint_11,joint_12,joint_13,joint_14,joint_15,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,joint_8,joint_9 |
| leap_right | F1_four_finger_serial | dev | available | 16 | 0 | 4 | root_chains | 2.010e-16 m / 4.057e-16 rad (frozen) | yes | no | no | continuation_pose:0,10,11,13,14,15,2,3,4,6,7,8 | limits_not_in_set:0,1,10,11,12,13,14,15,2,3,4,5,6,7,8,9 |
| barrett_bh | F4_non_anthropomorphic | dev | available | 8 | 0 | 3 | root_chains | 5.888e-17 m / 5.332e-16 rad (frozen) | yes | no | no | continuation_pose:finger_1_dist_joint,finger_1_med_joint,finger_2_dist_joint,finger_2_med_joint,finger_3_dist_joint | limits_not_in_set:finger_1_dist_joint,finger_1_med_joint,finger_1_prox_joint,finger_2_dist_joint,finger_2_med_joint,finger_2_prox_joint,finger_3_dist_joint,finger_3_med_joint |
| ability_right | F3_underactuated_coupled | dev | available | 10 | 4 | 5 | root_chains | 5.661e-17 m / 4.180e-16 rad (frozen) | yes | no | no | coupling_limits_not_image:index_q2,middle_q2,pinky_q2,ring_q2, continuation_pose:index_q2,middle_q2,pinky_q2,ring_q2,thumb_q2 | limits_not_in_set:index_q1,middle_q1,pinky_q1,ring_q1,thumb_q1,thumb_q2, coupling_params_not_in_set:index_q2:multiplier,middle_q2:multiplier,pinky_q2:multiplier,ring_q2:multiplier |
| inspire_right | F3_underactuated_coupled | dev | available | 12 | 6 | 5 | root_chains | 1.218e-16 m / 4.613e-16 rad (frozen) | yes | no | no | coupling_limits_not_image:index_intermediate_joint,middle_intermediate_joint,pinky_intermediate_joint,ring_intermediate_joint,thumb_distal_joint,thumb_intermediate_joint, continuation_pose:index_intermediate_joint,middle_intermediate_joint,pinky_intermediate_joint,ring_intermediate_joint,thumb_distal_joint,thumb_intermediate_joint,thumb_proximal_pitch_joint | limits_not_in_set:index_proximal_joint,middle_proximal_joint,pinky_proximal_joint,ring_proximal_joint,thumb_proximal_pitch_joint,thumb_proximal_yaw_joint, coupling_params_not_in_set:index_intermediate_joint:multiplier,index_intermediate_joint:offset,middle_intermediate_joint:multiplier,middle_intermediate_joint:offset,pinky_intermediate_joint:multiplier,pinky_intermediate_joint:offset,ring_intermediate_joint:multiplier,ring_intermediate_joint:offset,thumb_distal_joint:multiplier,thumb_intermediate_joint:multiplier |
| dclaw | F4_non_anthropomorphic | heldout | available | 9 | 0 | 3 | root_chains | 1.415e-16 m / 3.947e-16 rad (generated_at_run) | yes | no | no | continuation_pose:joint_f1_1,joint_f1_2,joint_f2_1,joint_f2_2,joint_f3_1,joint_f3_2 | axis_off_grid:joint_f1_0,joint_f1_1,joint_f1_2,joint_f2_0,joint_f2_1,joint_f2_2,joint_f3_0,joint_f3_1,joint_f3_2, limits_not_in_set:joint_f1_0,joint_f1_1,joint_f1_2,joint_f2_0,joint_f2_1,joint_f2_2,joint_f3_0,joint_f3_1,joint_f3_2 |
| wuji_right | F2_five_finger_full | heldout | available | 20 | 0 | 5 | root_chains | 7.076e-17 m / 4.142e-16 rad (generated_at_run) | yes | no | no | continuation_pose:right_finger1_joint2,right_finger1_joint3,right_finger1_joint4,right_finger2_joint2,right_finger2_joint3,right_finger3_joint2,right_finger3_joint3,right_finger4_joint2,right_finger4_joint3,right_finger5_joint2,right_finger5_joint3 | limits_not_in_set:right_finger1_joint1,right_finger1_joint2,right_finger1_joint3,right_finger1_joint4,right_finger2_joint1,right_finger2_joint2,right_finger2_joint3,right_finger2_joint4,right_finger3_joint1,right_finger3_joint2,right_finger3_joint3,right_finger3_joint4,right_finger4_joint1,right_finger4_joint2,right_finger4_joint3,right_finger4_joint4,right_finger5_joint1,right_finger5_joint2,right_finger5_joint3,right_finger5_joint4 |
| xhand_right | F2_five_finger_full | heldout | available | 12 | 0 | 5 | root_chains | 3.925e-17 m / 1.943e-16 rad (generated_at_run) | yes | no | no | continuation_pose:right_hand_thumb_rota_joint1,right_hand_thumb_rota_joint2 | limits_not_in_set:right_hand_index_bend_joint,right_hand_index_joint1,right_hand_index_joint2,right_hand_mid_joint1,right_hand_mid_joint2,right_hand_pinky_joint1,right_hand_pinky_joint2,right_hand_ring_joint1,right_hand_ring_joint2,right_hand_thumb_bend_joint,right_hand_thumb_rota_joint1,right_hand_thumb_rota_joint2 |
| tesollo_dg5f_right | F2_five_finger_full | heldout | available | 20 | 0 | 5 | root_chains | 5.552e-17 m / 3.331e-16 rad (generated_at_run) | yes | no | no | continuation_pose:rj_dg_1_2,rj_dg_1_3,rj_dg_1_4,rj_dg_2_2,rj_dg_3_2,rj_dg_4_2,rj_dg_5_2 | limits_not_in_set:rj_dg_1_1,rj_dg_1_2,rj_dg_1_3,rj_dg_1_4,rj_dg_2_1,rj_dg_2_2,rj_dg_2_3,rj_dg_2_4,rj_dg_3_1,rj_dg_3_2,rj_dg_3_3,rj_dg_3_4,rj_dg_4_1,rj_dg_4_2,rj_dg_4_3,rj_dg_4_4,rj_dg_5_1,rj_dg_5_2,rj_dg_5_3,rj_dg_5_4 |
| orca_right | F6_rolling_contact | heldout | available | 17 | 0 | 1 | root_chains | 1.157e-16 m / 5.222e-16 rad (generated_at_run) | yes | no | no | excess_children:right_palm, fixed_in_digit:right_index_abd_offset,right_index_mcp_offset,right_middle_abd_offset,right_middle_mcp_offset,right_pinky_abd_offset,right_pinky_mcp_offset,right_ring_abd_offset,right_ring_mcp_offset,right_thumb_abd_offset,right_thumb_mcp_offset,right_thumb_pip_offset,right_wrist_offset | axis_off_grid:right_thumb_abd, limits_not_in_set:right_index_abd,right_index_mcp,right_index_pip,right_middle_abd,right_middle_mcp,right_middle_pip,right_pinky_abd,right_pinky_mcp,right_pinky_pip,right_ring_abd,right_ring_mcp,right_ring_pip,right_thumb_abd,right_thumb_dip,right_thumb_mcp,right_thumb_pip,right_wrist |
| sharpa_left_on_iiwa14 | F2_five_finger_full | heldout | available | 22 | 0 | 5 | root_chains | 4.741e-16 m / 7.489e-16 rad (frozen) | yes | no | no | continuation_pose:left_index_DIP,left_index_MCP_AA,left_index_PIP,left_middle_DIP,left_middle_MCP_AA,left_middle_PIP,left_pinky_DIP,left_pinky_MCP_AA,left_pinky_MCP_FE,left_pinky_PIP,left_ring_DIP,left_ring_MCP_AA,left_ring_PIP,left_thumb_CMC_AA,left_thumb_IP,left_thumb_MCP_AA,left_thumb_MCP_FE | limits_not_in_set:left_1_thumb_CMC_FE,left_2_index_MCP_FE,left_3_middle_MCP_FE,left_4_ring_MCP_FE,left_5_pinky_CMC,left_index_DIP,left_index_MCP_AA,left_index_PIP,left_middle_DIP,left_middle_MCP_AA,left_middle_PIP,left_pinky_DIP,left_pinky_MCP_AA,left_pinky_MCP_FE,left_pinky_PIP,left_ring_DIP,left_ring_MCP_AA,left_ring_PIP,left_thumb_CMC_AA,left_thumb_IP,left_thumb_MCP_AA,left_thumb_MCP_FE |
| shadow_right | F2_five_finger_full | excluded | excluded | - | - | - | - | - | - | - | - | - | - |
| svh_right | F3_underactuated_coupled | excluded | excluded | - | - | - | - | - | - | - | - | - | - |
| dex1 | F5_prismatic | excluded | excluded | - | - | - | - | - | - | - | - | - | - |

## Constructs covered by the grammar

Proven reachable by `hand_sampler.grammar.derive.sample_derivation` over `test_grammar_support_audit`'s seeded sample (see `grammar_bench/tests/test_acceptance_grammar.py`), independent of this pilot:

- one_digit
- five_plus_digits
- six_plus_phalanges
- in_digit_branch
- palm_tree
- palm_joint
- two_nonparallel_palm_joints
- nonperpendicular_axis
- coupling_nonzero_offset
- coupling_negative_multiplier
- asymmetric_limits
- continuous_joint
- prismatic_joint
- nonidentity_mount_rotation

## Known limitations

- Digit counting for an imported model (`digit_count_source: root_chains`) counts movable-joint chains leaving the declared root, treating a run of fixed joints as transparent -- so a hand whose root passes through a single wrist joint before fanning out to its fingers (e.g. `orca_right`) counts as 1 digit, not 5; it is not a count of anatomical fingers.
- `topology_expressible` is a necessary, not sufficient, condition: it means the grammar's own productions could build a model with this *topology* (joint types, branching, coupling scope, ...), never that the grammar would ever sample this particular hand's lengths/axes/limits -- that is what `in_support` checks.
- `in_support` is judged against `DEFAULT_DISTRIBUTION` only; a hand out of support under the default ranges/grids/choice-sets might still be in support of some other `Distribution` this module could construct.

## Claims

- Availability, fidelity and coverage above were computed for every hand listed in `grammar_bench/manifest.json` (14 hands); `excluded`-split hands were never imported, and `unavailable` hands were never scored (their reason is reported instead of a result).
- Fidelity, where reported, compares our own `hand_sampler.grammar.fk` against an independent Pinocchio implementation reading the same URDF, either from a frozen reference committed to the benchmark or one generated fresh in this run (`reference: generated_at_run`).
- `topology_expressible`/`in_support` come from `hand_sampler.grammar.coverage`, which checks the imported model against the grammar's own productions (`rules.py`) and the default sampling distribution's ranges/grids (`distributions.py`); named `missing_constructs`/`out_of_support` items are never dropped or relabeled to make a hand look better.
- Held-out-split hands are reported exactly like dev-split hands; none of their results were used to change any grammar rule, sampling range or tolerance.
- No universality claim is made.
