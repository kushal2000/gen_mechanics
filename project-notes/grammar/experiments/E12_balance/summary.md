# Experiment: e12_balance

Wall time: 203.725 s

## Params

```json
{
  "drift_seeds": 128,
  "drift_walk_length": 200,
  "histogram_n": 2000,
  "locality_seeds": 200,
  "redundancy_n": 5000,
  "reversibility_n_parents": 500
}
```

## (a) Per-pair neutral drift

### Boundary regime (reduced minimum-size starts; drift expected positive)

#### dist=G_FULL_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | -0.4219 | -1.086 | 0.2109 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_digits | 1.734 | 1.453 | 2.062 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 4.227 | 3.109 | 5.406 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_digits | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_joints | 0.2109 | -0.01562 | 0.4375 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_palm_bodies | 0.2109 | -0.01562 | 0.4375 | 128 | PASS |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 4.312 | 3.461 | 5.156 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 3.82 | 2.648 | 4.977 | 128 | PASS |
| EVOLUTION_pool | delta_digits | 1.945 | 1.672 | 2.211 | 128 | PASS |
| EVOLUTION_pool | delta_palm_bodies | 0.6875 | 0.4766 | 0.8906 | 128 | PASS |

| pair | operator | attempts | applicable | rate |
|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | add_minimal_digit | 12606 | 10770 | 0.8544 |
| add_minimal_digit/remove_digit_minimal | remove_digit_minimal | 12994 | 10548 | 0.8118 |
| insert_phalanx/delete_phalanx | insert_phalanx | 12817 | 12101 | 0.9441 |
| insert_phalanx/delete_phalanx | delete_phalanx | 12783 | 11560 | 0.9043 |
| add_palm_body/remove_palm_body_empty | add_palm_body | 12821 | 8278 | 0.6457 |
| add_palm_body/remove_palm_body_empty | remove_palm_body_empty | 12779 | 8251 | 0.6457 |
| toggle_palm_joint | toggle_palm_joint | 25600 | 21000 | 0.8203 |
| add_branch_digit/remove_branch_digit | add_branch_digit | 12736 | 11971 | 0.9399 |
| add_branch_digit/remove_branch_digit | remove_branch_digit | 12864 | 11382 | 0.8848 |

#### dist=G_NOBRANCH_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | 0.6484 | 0.3047 | 1.016 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_digits | 1.734 | 1.414 | 2.07 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 3.555 | 3.047 | 4.055 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_digits | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_joints | 0.1562 | -0.05469 | 0.3828 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_palm_bodies | 0.1562 | -0.05469 | 0.3828 | 128 | PASS |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 5.367 | 4.664 | 6.055 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 6.812 | 5.883 | 7.703 | 128 | PASS |
| EVOLUTION_pool | delta_digits | 1.734 | 1.438 | 2.023 | 128 | PASS |
| EVOLUTION_pool | delta_palm_bodies | 0.4219 | 0.1951 | 0.6562 | 128 | PASS |

| pair | operator | attempts | applicable | rate |
|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | add_minimal_digit | 12609 | 10816 | 0.8578 |
| add_minimal_digit/remove_digit_minimal | remove_digit_minimal | 12991 | 10594 | 0.8155 |
| insert_phalanx/delete_phalanx | insert_phalanx | 12816 | 11687 | 0.9119 |
| insert_phalanx/delete_phalanx | delete_phalanx | 12784 | 11232 | 0.8786 |
| add_palm_body/remove_palm_body_empty | add_palm_body | 12831 | 8198 | 0.6389 |
| add_palm_body/remove_palm_body_empty | remove_palm_body_empty | 12769 | 8178 | 0.6405 |
| toggle_palm_joint | toggle_palm_joint | 25600 | 21000 | 0.8203 |
| add_branch_digit/remove_branch_digit | add_branch_digit | 12781 | 11824 | 0.9251 |
| add_branch_digit/remove_branch_digit | remove_branch_digit | 12819 | 11137 | 0.8688 |

**PASS/FAIL boundary regime: informational only (drift here is expected positive by construction -- reduced starts have a lopsided growth/shrink applicability ratio); it does not gate (a) overall.**

### Stationary regime (random samples from the distribution; 95% CI must include 0)

#### dist=G_FULL_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | 0.3516 | -0.007812 | 0.7031 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_digits | 0.7969 | 0.5 | 1.094 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 0.25 | -1.625 | 2 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_digits | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_joints | 0.4766 | 0.2812 | 0.6721 | 128 | FAIL |
| add_palm_body/remove_palm_body_empty | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_palm_bodies | 0.4766 | 0.2812 | 0.6721 | 128 | FAIL |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 8.234 | 6.984 | 9.625 | 128 | FAIL |
| add_branch_digit/remove_branch_digit | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 2.195 | 0.6484 | 3.821 | 128 | FAIL |
| EVOLUTION_pool | delta_digits | 0.8359 | 0.5078 | 1.18 | 128 | FAIL |
| EVOLUTION_pool | delta_palm_bodies | 0.6562 | 0.4529 | 0.8672 | 128 | FAIL |

| pair | operator | attempts | applicable | rate |
|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | add_minimal_digit | 12789 | 9019 | 0.7052 |
| add_minimal_digit/remove_digit_minimal | remove_digit_minimal | 12811 | 8917 | 0.696 |
| insert_phalanx/delete_phalanx | insert_phalanx | 12868 | 12273 | 0.9538 |
| insert_phalanx/delete_phalanx | delete_phalanx | 12732 | 12241 | 0.9614 |
| add_palm_body/remove_palm_body_empty | add_palm_body | 12953 | 7403 | 0.5715 |
| add_palm_body/remove_palm_body_empty | remove_palm_body_empty | 12647 | 7342 | 0.5805 |
| toggle_palm_joint | toggle_palm_joint | 25600 | 21000 | 0.8203 |
| add_branch_digit/remove_branch_digit | add_branch_digit | 12718 | 12528 | 0.9851 |
| add_branch_digit/remove_branch_digit | remove_branch_digit | 12882 | 11474 | 0.8907 |

#### dist=G_NOBRANCH_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | 0.07812 | -0.3127 | 0.4609 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_digits | 0.6953 | 0.3748 | 0.9924 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 0.7812 | -0.2969 | 1.883 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_digits | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_joints | 0.4844 | 0.2969 | 0.6719 | 128 | FAIL |
| add_palm_body/remove_palm_body_empty | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body_empty | delta_palm_bodies | 0.4844 | 0.2969 | 0.6719 | 128 | FAIL |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 8.898 | 7.625 | 10.09 | 128 | FAIL |
| add_branch_digit/remove_branch_digit | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 3.336 | 2 | 4.586 | 128 | FAIL |
| EVOLUTION_pool | delta_digits | 0.7656 | 0.4139 | 1.133 | 128 | FAIL |
| EVOLUTION_pool | delta_palm_bodies | 0.6719 | 0.4609 | 0.8828 | 128 | FAIL |

| pair | operator | attempts | applicable | rate |
|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | add_minimal_digit | 12841 | 9576 | 0.7457 |
| add_minimal_digit/remove_digit_minimal | remove_digit_minimal | 12759 | 9487 | 0.7436 |
| insert_phalanx/delete_phalanx | insert_phalanx | 12924 | 12022 | 0.9302 |
| insert_phalanx/delete_phalanx | delete_phalanx | 12676 | 11922 | 0.9405 |
| add_palm_body/remove_palm_body_empty | add_palm_body | 12945 | 7321 | 0.5655 |
| add_palm_body/remove_palm_body_empty | remove_palm_body_empty | 12655 | 7259 | 0.5736 |
| toggle_palm_joint | toggle_palm_joint | 25600 | 21000 | 0.8203 |
| add_branch_digit/remove_branch_digit | add_branch_digit | 12753 | 12509 | 0.9809 |
| add_branch_digit/remove_branch_digit | remove_branch_digit | 12847 | 11370 | 0.885 |

**PASS/FAIL (a) per-pair neutral drift (stationary regime): FAIL**

## (b) Locality bands (per-operator, EVOLUTION_OPERATORS pool, G_FULL)

n_seeds: 200

criteria: every small-step operator's median tip displacement <= 0.005 m; every structural operator's p90 <= 0.06 m.

| operator | kind | median (m) | p90 (m) | n | PASS |
|---|---|---|---|---|---|
| add_minimal_digit | structural | 0.0075 | 0.025 | 162 | PASS |
| remove_digit_minimal | structural | 0.006847 | 0.02 | 114 | PASS |
| insert_phalanx | structural | 0.007639 | 0.02892 | 200 | PASS |
| delete_phalanx | structural | 0.01198 | 0.0419 | 190 | PASS |
| add_palm_body | structural | 0 | 0 | 139 | PASS |
| remove_palm_body_empty | structural | 0 | 0 | 66 | PASS |
| toggle_palm_joint | structural | 0 | 0.02096 | 159 | PASS |
| add_branch_digit | structural | 0.007107 | 0.02658 | 200 | PASS |
| remove_branch_digit | structural | 0.005371 | 0.01 | 76 | PASS |
| step_axis | small-step | 0.001701 | 0.007517 | 200 | PASS |
| step_limits | small-step | 0.0004246 | 0.003611 | 199 | PASS |
| step_mount | small-step | 0.004376 | 0.01629 | 200 | PASS |
| step_coupling | small-step | 0.001499 | 0.01057 | 168 | PASS |
| step_root_length | small-step | 0.0025 | 0.004691 | 200 | PASS |
| step_radius | small-step | 0 | 0 | 200 | PASS |
| step_bend_rpy | small-step | n/a | n/a | 0 | PASS |
| step_bend_offset | small-step | n/a | n/a | 0 | PASS |

small-step violators: none
structural violators: none

**PASS/FAIL (b) locality bands: PASS**

## (c) Reversibility (>= 95% of applicable cases)

| pair | n_parents | n_applicable | n_recovered | rate | PASS |
|---|---|---|---|---|---|
| G_FULL_INS/add_minimal_digit/remove_digit_minimal | 500 | 421 | 421 | 1 | PASS |
| G_FULL_INS/insert_phalanx/delete_phalanx | 500 | 490 | 490 | 1 | PASS |
| G_FULL_INS/add_palm_body/remove_palm_body_empty | 500 | 373 | 373 | 1 | PASS |
| G_FULL_INS/toggle_palm_joint | 500 | 240 | 240 | 1 | PASS |
| G_FULL_INS/add_branch_digit/remove_branch_digit | 500 | 500 | 500 | 1 | PASS |

| pair | n_parents | n_applicable | n_recovered | rate | PASS |
|---|---|---|---|---|---|
| G_NOBRANCH_INS/add_minimal_digit/remove_digit_minimal | 500 | 421 | 421 | 1 | PASS |
| G_NOBRANCH_INS/insert_phalanx/delete_phalanx | 500 | 478 | 478 | 1 | PASS |
| G_NOBRANCH_INS/add_palm_body/remove_palm_body_empty | 500 | 373 | 373 | 1 | PASS |
| G_NOBRANCH_INS/toggle_palm_joint | 500 | 240 | 240 | 1 | PASS |
| G_NOBRANCH_INS/add_branch_digit/remove_branch_digit | 500 | 500 | 500 | 1 | PASS |

**PASS/FAIL (c) reversibility: PASS**

## (d) Prior size histograms (no bound > 20% of mass)

### dist=G_FULL_INS

| metric | min | max | frac_at_min | frac_at_max | PASS |
|---|---|---|---|---|---|
| joints | 1 | 133 | 0.0085 | 0.0005 | PASS |
| digits | 1 | 6 | 0.183 | 0.171 | PASS |
| motors | 1 | 113 | 0.016 | 0.0005 | PASS |

### dist=G_NOBRANCH_INS

| metric | min | max | frac_at_min | frac_at_max | PASS |
|---|---|---|---|---|---|
| joints | 1 | 36 | 0.01 | 0.0005 | PASS |
| digits | 1 | 6 | 0.183 | 0.171 | PASS |
| motors | 1 | 31 | 0.019 | 0.0005 | PASS |

### dist=G_BEND

| metric | min | max | frac_at_min | frac_at_max | PASS |
|---|---|---|---|---|---|
| joints | 1 | 150 | 0.0075 | 0.0005 | PASS |
| digits | 1 | 6 | 0.183 | 0.171 | PASS |
| motors | 1 | 129 | 0.016 | 0.0005 | PASS |

**PASS/FAIL (d) prior size histograms: PASS**

## (e) Redundancy (G_FULL, distinct-hash fraction >= 99%)

n=5000 n_distinct=5000 distinct_fraction=1

**PASS/FAIL (e) redundancy: PASS**

## Overall

- (a) drift: FAIL
- (b) locality: PASS
- (c) reversibility: PASS
- (d) histograms: PASS
- (e) redundancy: PASS
