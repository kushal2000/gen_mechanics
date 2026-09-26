# Experiment: e12_balance

Wall time: 202.621 s

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

## (a) Per-pair neutral drift -- 95% CI must include 0

### dist=G_FULL_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | -0.4219 | -1.086 | 0.2109 | 128 | PASS |
| add_minimal_digit/remove_digit_minimal | delta_digits | 1.484 | 1.156 | 1.844 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 4.227 | 3.109 | 5.406 | 128 | FAIL |
| insert_phalanx/delete_phalanx | delta_digits | 1.375 | 1.047 | 1.727 | 128 | FAIL |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_joints | -0.2734 | -0.5391 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_palm_bodies | -0.2734 | -0.5391 | 0 | 128 | PASS |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 4.211 | 3.344 | 5.062 | 128 | FAIL |
| add_branch_digit/remove_branch_digit | delta_digits | -0.2266 | -0.3672 | -0.1094 | 128 | FAIL |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 3.391 | 2.172 | 4.563 | 128 | FAIL |
| EVOLUTION_pool | delta_digits | 1.797 | 1.438 | 2.141 | 128 | FAIL |
| EVOLUTION_pool | delta_palm_bodies | -0.1484 | -0.3984 | 0.1172 | 128 | PASS |

### dist=G_NOBRANCH_INS

| pair | metric | mean | 95% CI lo | 95% CI hi | n | PASS |
|---|---|---|---|---|---|---|
| add_minimal_digit/remove_digit_minimal | delta_joints | 0.6484 | 0.3047 | 1.016 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_digits | 1.734 | 1.414 | 2.07 | 128 | FAIL |
| add_minimal_digit/remove_digit_minimal | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_joints | 3.555 | 3.047 | 4.055 | 128 | FAIL |
| insert_phalanx/delete_phalanx | delta_digits | 0 | 0 | 0 | 128 | PASS |
| insert_phalanx/delete_phalanx | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_joints | -0.2734 | -0.5391 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_palm_body/remove_palm_body | delta_palm_bodies | -0.2734 | -0.5391 | 0 | 128 | PASS |
| toggle_palm_joint | delta_joints | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_digits | 0 | 0 | 0 | 128 | PASS |
| toggle_palm_joint | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_joints | 5.367 | 4.664 | 6.055 | 128 | FAIL |
| add_branch_digit/remove_branch_digit | delta_digits | 0 | 0 | 0 | 128 | PASS |
| add_branch_digit/remove_branch_digit | delta_palm_bodies | 0 | 0 | 0 | 128 | PASS |
| EVOLUTION_pool | delta_joints | 6.336 | 5.398 | 7.227 | 128 | FAIL |
| EVOLUTION_pool | delta_digits | 1.992 | 1.711 | 2.266 | 128 | FAIL |
| EVOLUTION_pool | delta_palm_bodies | -0.2109 | -0.4766 | 0.0627 | 128 | PASS |

**PASS/FAIL (a) per-pair neutral drift: FAIL**

## (b) Locality bands (E1 machinery, EVOLUTION_OPERATORS pool, G_FULL)

n_seeds: 200

small-step IQR: q1=5.544e-05 median=0.00125 q3=0.0036 (n=1167)

structural IQR: q1=0.002247 median=0.006493 q3=0.01735 p90=0.04207 (n=1399)

worst structural operator by p90: remove_palm_body (0.1947 m)

- PASS/FAIL small-step IQR entirely below structural IQR: FAIL
- PASS/FAIL structural median <= 0.020 m: PASS
- PASS/FAIL no structural operator p90 > 0.060 m: FAIL

**PASS/FAIL (b) locality bands: FAIL**

## (c) Reversibility (>= 95% of applicable cases)

| pair | n_parents | n_applicable | n_recovered | rate | PASS |
|---|---|---|---|---|---|
| G_FULL_INS/add_minimal_digit/remove_digit_minimal | 500 | 421 | 421 | 1 | PASS |
| G_FULL_INS/insert_phalanx/delete_phalanx | 500 | 490 | 490 | 1 | PASS |
| G_FULL_INS/add_palm_body/remove_palm_body | 500 | 373 | 373 | 1 | PASS |
| G_FULL_INS/toggle_palm_joint | 500 | 240 | 240 | 1 | PASS |
| G_FULL_INS/add_branch_digit/remove_branch_digit | 500 | 500 | 500 | 1 | PASS |

| pair | n_parents | n_applicable | n_recovered | rate | PASS |
|---|---|---|---|---|---|
| G_NOBRANCH_INS/add_minimal_digit/remove_digit_minimal | 500 | 421 | 421 | 1 | PASS |
| G_NOBRANCH_INS/insert_phalanx/delete_phalanx | 500 | 478 | 478 | 1 | PASS |
| G_NOBRANCH_INS/add_palm_body/remove_palm_body | 500 | 373 | 373 | 1 | PASS |
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
- (b) locality: FAIL
- (c) reversibility: PASS
- (d) histograms: PASS
- (e) redundancy: PASS
