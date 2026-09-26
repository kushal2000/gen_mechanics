# E11: support widening for real hands

Cumulative relaxation levels applied to `hand_sampler.grammar.coverage.coverage`'s `relax` argument. Level definitions:

- **L0** = {(none)}
- **L1** = {rest_bend}
- **L2** = {limits_continuous, rest_bend}
- **L3** = {length_grid_1mm, limits_continuous, rest_bend}
- **L4** = {axis_grid_5deg, length_grid_1mm, limits_continuous, rest_bend}
- **L5** = {axis_grid_5deg, coupling_continuous, length_grid_1mm, limits_continuous, rest_bend}
- **L6** = {axis_continuous, axis_grid_5deg, coupling_continuous, length_continuous, length_grid_1mm, limits_continuous, rest_bend}
- **L7** = {axis_continuous, axis_grid_5deg, coupling_continuous, length_continuous, length_grid_1mm, length_range_x1.5, limits_continuous, limits_range_x1.5, rest_bend}
- **L8** = {axis_continuous, axis_grid_5deg, children_unbounded, coupling_continuous, fixed_in_digit_ok, length_continuous, length_grid_1mm, length_range_x1.5, limits_continuous, limits_range_x1.5, rest_bend}

## Hands x levels (in_support, remaining-item count)

| id | family | split | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| allegro_right | F1_four_finger_serial | dev | no (2) | no (1) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) |
| leap_right | F1_four_finger_serial | dev | no (2) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) | yes (0) | yes (0) |
| barrett_bh | F4_non_anthropomorphic | dev | no (2) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) |
| ability_right | F3_underactuated_coupled | dev | no (4) | no (3) | no (3) | no (3) | no (3) | no (2) | no (2) | no (2) | no (2) |
| inspire_right | F3_underactuated_coupled | dev | no (4) | no (3) | no (2) | no (2) | no (2) | no (1) | no (1) | no (1) | no (1) |
| dclaw | F4_non_anthropomorphic | heldout | no (3) | no (2) | no (2) | no (2) | no (2) | no (2) | no (1) | no (1) | no (1) |
| wuji_right | F2_five_finger_full | heldout | no (2) | no (1) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) |
| xhand_right | F2_five_finger_full | heldout | no (2) | no (1) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) | yes (0) |
| tesollo_dg5f_right | F2_five_finger_full | heldout | no (2) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) | no (1) |
| orca_right | F6_rolling_contact | heldout | no (4) | no (4) | no (4) | no (4) | no (4) | no (4) | no (3) | no (2) | yes (0) |
| sharpa_left_on_iiwa14 | F2_five_finger_full | heldout | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| shadow_right | F2_five_finger_full | excluded | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| svh_right | F3_underactuated_coupled | excluded | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |
| dex1 | F5_prismatic | excluded | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

Cell format: `in_support (remaining out-of-support + missing-construct item count)`. `unavailable` hands (see reason in JSON) were never scored.

## Single-relaxation unblocking (from L0)

Of 10 available hands, 10 are out of support at L0 (no relaxation). For each relaxation applied ALONE (not cumulatively), the count of hands it alone moves from not-in-support to in-support:

| relaxation | hands unblocked | which hands |
|---|---|---|
| axis_continuous | 0 | - |
| axis_grid_5deg | 0 | - |
| children_unbounded | 0 | - |
| coupling_continuous | 0 | - |
| fixed_in_digit_ok | 0 | - |
| length_continuous | 0 | - |
| length_grid_1mm | 0 | - |
| length_range_x1.5 | 0 | - |
| limits_continuous | 0 | - |
| limits_range_x1.5 | 0 | - |
| rest_bend | 0 | - |

Every count above is 0 except `rest_bend` itself: every available real hand also fails `continuation_pose` (mid-digit rpy/lateral offset), so `topology_expressible` stays False -- and therefore `in_support` stays False -- regardless of what any other single relaxation does to the parameter-support checks. `rest_bend` is a universal prerequisite, not one relaxation among equals.

## Single-relaxation unblocking (from L1, i.e. with rest_bend already applied)

The more informative version: 10 of 10 hands are still out of support at L1 ({rest_bend} only). For each OTHER relaxation added alone on top of rest_bend, the count of hands it unblocks:

| relaxation (+ rest_bend) | hands unblocked | which hands |
|---|---|---|
| limits_range_x1.5 | 4 | allegro_right, leap_right, wuji_right, xhand_right |
| limits_continuous | 3 | allegro_right, wuji_right, xhand_right |
| axis_continuous | 0 | - |
| axis_grid_5deg | 0 | - |
| children_unbounded | 0 | - |
| coupling_continuous | 0 | - |
| fixed_in_digit_ok | 0 | - |
| length_continuous | 0 | - |
| length_grid_1mm | 0 | - |
| length_range_x1.5 | 0 | - |

## Bits per joint (design-space-size cost) per level

Choice counts for `DEFAULT_DISTRIBUTION`; `continuous` means the relaxation removes the grid/choice-set constraint entirely (no finite bit count -- reported as the DOF and range instead).

| level | revolute limits | prismatic limits | link length | axis | coupling (mult, offset) |
|---|---|---|---|---|---|
| L0 | 10 choices (3.322 bits) | 3 choices (1.585 bits) | 14 choices (3.807 bits), grid=5mm | 312 choices (8.285 bits), grid=15deg | 6 mult choices (2.585 bits) x 5 offset choices (2.322 bits) |
| L1 | 10 choices (3.322 bits) | 3 choices (1.585 bits) | 14 choices (3.807 bits), grid=5mm | 312 choices (8.285 bits), grid=15deg | 6 mult choices (2.585 bits) x 5 offset choices (2.322 bits) |
| L2 | continuous in [-45.0, 110.0] deg (2 continuous DOF: lo, hi) | continuous | 14 choices (3.807 bits), grid=5mm | 312 choices (8.285 bits), grid=15deg | 6 mult choices (2.585 bits) x 5 offset choices (2.322 bits) |
| L3 | continuous in [-45.0, 110.0] deg (2 continuous DOF: lo, hi) | continuous | 66 choices (6.044 bits), grid=1mm | 312 choices (8.285 bits), grid=15deg | 6 mult choices (2.585 bits) x 5 offset choices (2.322 bits) |
| L4 | continuous in [-45.0, 110.0] deg (2 continuous DOF: lo, hi) | continuous | 66 choices (6.044 bits), grid=1mm | 2664 choices (11.379 bits), grid=5deg | 6 mult choices (2.585 bits) x 5 offset choices (2.322 bits) |
| L5 | continuous in [-45.0, 110.0] deg (2 continuous DOF: lo, hi) | continuous | 66 choices (6.044 bits), grid=1mm | 2664 choices (11.379 bits), grid=5deg | continuous: multiplier in [-2,2], offset in [-1,1] rad |
| L6 | continuous in [-45.0, 110.0] deg (2 continuous DOF: lo, hi) | continuous | continuous in [0.0150, 0.0800] m | continuous (unit sphere, 2 continuous DOF) | continuous: multiplier in [-2,2], offset in [-1,1] rad |
| L7 | continuous in [-83.8, 148.8] deg (2 continuous DOF: lo, hi) | continuous | continuous in [-0.0013, 0.0963] m | continuous (unit sphere, 2 continuous DOF) | continuous: multiplier in [-2,2], offset in [-1,1] rad |
| L8 | continuous in [-83.8, 148.8] deg (2 continuous DOF: lo, hi) | continuous | continuous in [-0.0013, 0.0963] m | continuous (unit sphere, 2 continuous DOF) | continuous: multiplier in [-2,2], offset in [-1,1] rad |

## Generated designs (200 seeds) stay in support at every level

Seeds 0..199, `DEFAULT_DISTRIBUTION`:

| level | all in_support | counterexample seeds |
|---|---|---|
| L0 | yes | - |
| L1 | yes | - |
| L2 | yes | - |
| L3 | yes | - |
| L4 | yes | - |
| L5 | yes | - |
| L6 | yes | - |
| L7 | yes | - |
| L8 | yes | - |

## Remaining items at L8, hardest hands

- **ability_right** (F3_underactuated_coupled): in_support=False
  - missing_constructs: coupling_limits_not_image:index_q2,middle_q2,pinky_q2,ring_q2
  - out_of_support: limits_not_in_set:thumb_q1
- **barrett_bh** (F4_non_anthropomorphic): in_support=False
  - out_of_support: limits_not_in_set:finger_1_med_joint,finger_1_prox_joint,finger_2_med_joint,finger_2_prox_joint,finger_3_med_joint
- **inspire_right** (F3_underactuated_coupled): in_support=False
  - missing_constructs: coupling_limits_not_image:index_intermediate_joint,middle_intermediate_joint,pinky_intermediate_joint,ring_intermediate_joint,thumb_distal_joint,thumb_intermediate_joint
- **dclaw** (F4_non_anthropomorphic): in_support=False
  - out_of_support: limits_not_in_set:joint_f1_1,joint_f1_2,joint_f2_1,joint_f2_2,joint_f3_1,joint_f3_2
- **tesollo_dg5f_right** (F2_five_finger_full): in_support=False
  - out_of_support: limits_not_in_set:rj_dg_1_2,rj_dg_1_3,rj_dg_1_4,rj_dg_2_3,rj_dg_2_4,rj_dg_3_3,rj_dg_3_4,rj_dg_4_3,rj_dg_4_4,rj_dg_5_3,rj_dg_5_4

## Claims / limitations

- Each relaxation is applied via `coverage.py`'s `relax` argument, never by editing `rules.py`/`distributions.py`; the grammar's own sampling behaviour is unchanged by this script.
- Relaxations named here do not cover every `missing_constructs`/`out_of_support` category coverage.py can report (e.g. `coupling_limits_not_image`, `coupling_scope`, `digit_count_out_of_range` have no corresponding relax flag in this design) -- a hand blocked by one of those stays out of support at every level; see the L8 remaining-items list above.
- No universality claim is made; held-out-split hands are reported exactly like dev-split hands.
