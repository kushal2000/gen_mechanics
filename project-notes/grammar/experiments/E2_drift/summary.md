# Experiment: e2_drift

Wall time: 11.535 s

## Params

```json
{
  "record_every_after": 10,
  "record_through": 40,
  "walk_length": 400
}
```

n_seeds: 128

Bootstrap CIs are PAIRED across mixtures (same seeds/starts within a dist variant -- see module docstring).

## Start sizes per dist variant (mean over seeds)

| dist | start digit_count mean | start max_phalanx_count mean |
|---|---|---|
| G_FULL | 1.812 | 1.914 |
| G_NOBRANCH | 1.812 | 1.922 |
| G_FULL_INS | 1.812 | 1.914 |
| G_NOBRANCH_INS | 1.812 | 1.922 |

## Final joints (ABSOLUTE, not a delta) at step 400, per mixture x dist (paired 95% CI)

| dist | mixture | final joints mean | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|
| G_FULL | DEFAULT_uniform | 30.76 | 27.13 | 34.62 | 128 |
| G_FULL | UNION_uniform | 31.22 | 27.96 | 34.82 | 128 |
| G_FULL | UNION_weighted | 29.24 | 25.87 | 32.57 | 128 |
| G_FULL | EVOLUTION_uniform | 14.16 | 12.88 | 15.58 | 128 |
| G_FULL | EVOLUTION_weighted | 13.05 | 11.91 | 14.3 | 128 |
| G_NOBRANCH | DEFAULT_uniform | 14.4 | 13.19 | 15.61 | 128 |
| G_NOBRANCH | UNION_uniform | 13.44 | 12.3 | 14.62 | 128 |
| G_NOBRANCH | UNION_weighted | 13.23 | 12.08 | 14.37 | 128 |
| G_NOBRANCH | EVOLUTION_uniform | 13.12 | 11.99 | 14.17 | 128 |
| G_NOBRANCH | EVOLUTION_weighted | 13.16 | 12.18 | 14.15 | 128 |
| G_FULL_INS | DEFAULT_uniform | 8.688 | 7.938 | 9.469 | 128 |
| G_FULL_INS | UNION_uniform | 8.273 | 7.672 | 8.891 | 128 |
| G_FULL_INS | UNION_weighted | 8.07 | 7.492 | 8.68 | 128 |
| G_FULL_INS | EVOLUTION_uniform | 13.66 | 12.55 | 14.91 | 128 |
| G_FULL_INS | EVOLUTION_weighted | 14.03 | 12.77 | 15.35 | 128 |
| G_NOBRANCH_INS | DEFAULT_uniform | 8.641 | 7.938 | 9.375 | 128 |
| G_NOBRANCH_INS | UNION_uniform | 8.336 | 7.727 | 8.914 | 128 |
| G_NOBRANCH_INS | UNION_weighted | 7.789 | 7.117 | 8.445 | 128 |
| G_NOBRANCH_INS | EVOLUTION_uniform | 13.53 | 12.58 | 14.52 | 128 |
| G_NOBRANCH_INS | EVOLUTION_weighted | 12.3 | 11.34 | 13.26 | 128 |

## Difference CI: UNION_weighted minus DEFAULT_uniform, final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| dist | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|
| G_FULL | joints | -1.516 | -6.744 | 3.821 | 128 |
| G_FULL | digits | 0.01562 | -0.4453 | 0.4297 | 128 |
| G_FULL | motors | -1.312 | -5.777 | 3.141 | 128 |
| G_NOBRANCH | joints | -1.164 | -2.719 | 0.4846 | 128 |
| G_NOBRANCH | digits | 0.08594 | -0.3439 | 0.5078 | 128 |
| G_NOBRANCH | motors | -0.7969 | -2.141 | 0.5393 | 128 |
| G_FULL_INS | joints | -0.6172 | -1.547 | 0.3047 | 128 |
| G_FULL_INS | digits | 0.09375 | -0.2893 | 0.5 | 128 |
| G_FULL_INS | motors | -0.75 | -1.555 | 0.07832 | 128 |
| G_NOBRANCH_INS | joints | -0.8516 | -1.797 | 0.1094 | 128 |
| G_NOBRANCH_INS | digits | -0.2656 | -0.6641 | 0.1562 | 128 |
| G_NOBRANCH_INS | motors | -0.6328 | -1.445 | 0.2266 | 128 |

## Difference CI: _INS minus plain, per pool (mixture), final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| pair | mixture | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|---|
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | joints | -22.07 | -25.97 | -18.46 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | digits | -0.05469 | -0.5078 | 0.3906 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | motors | -18.41 | -21.76 | -15.27 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | joints | -22.95 | -26.58 | -19.6 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | digits | -0.3047 | -0.6719 | 0.07031 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | motors | -19.81 | -22.9 | -16.96 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | joints | -21.17 | -24.56 | -17.76 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | digits | 0.02344 | -0.3672 | 0.4143 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | motors | -17.85 | -20.77 | -14.89 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | joints | -0.4922 | -2.234 | 1.234 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | digits | -0.09375 | -0.5156 | 0.2812 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | motors | -0.2812 | -1.922 | 1.406 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | joints | 0.9766 | -0.625 | 2.594 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | digits | -0.01562 | -0.375 | 0.3283 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | motors | 1.094 | -0.4377 | 2.649 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | joints | -5.758 | -7.25 | -4.219 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | digits | -0.01562 | -0.5 | 0.4609 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | motors | -4.656 | -5.946 | -3.336 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | joints | -5.102 | -6.391 | -3.867 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | digits | -0.25 | -0.625 | 0.1174 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | motors | -4.32 | -5.492 | -3.211 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | joints | -5.445 | -6.703 | -4.101 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | digits | -0.3672 | -0.7814 | 0.05469 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | motors | -4.492 | -5.555 | -3.422 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | joints | 0.4141 | -1.031 | 1.875 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | digits | 0.1875 | -0.1797 | 0.5549 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | motors | 0.4062 | -1.039 | 1.852 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | joints | -0.8672 | -2.321 | 0.5625 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | digits | -0.4219 | -0.7971 | -0.04668 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | motors | -0.7422 | -2.125 | 0.6176 | 128 |

## dist=G_FULL mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 16.01 | 12.84 | 19.17 | 128 |
| digits | 1.289 | 1.008 | 1.563 | 128 |
| motors | 13.23 | 10.55 | 15.98 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 21.47 | 17.37 | 25.81 | 128 |
| digits | 1.711 | 1.406 | 2.031 | 128 |
| motors | 17.96 | 14.45 | 21.7 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.848 |
| delete_phalanx | 0.9787 |
| insert_phalanx | 0.9898 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8285 |
| resample_parameter | 1 |

## dist=G_FULL mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.898 | 6.445 | 11.39 | 128 |
| digits | 1.125 | 0.8594 | 1.406 | 128 |
| motors | 7.68 | 5.562 | 9.812 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 21.93 | 18.43 | 25.7 | 128 |
| digits | 1.883 | 1.602 | 2.141 | 128 |
| motors | 18.81 | 15.81 | 22.03 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8192 |
| add_minimal_digit | 0.8204 |
| add_palm_body | 0.7291 |
| delete_phalanx | 0.9603 |
| insert_phalanx | 0.992 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8618 |
| remove_digit_minimal | 0.6635 |
| remove_palm_body | 0.7597 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.8075 |
| step_limits | 0.9976 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7554 |

## dist=G_FULL mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.031 | 6.742 | 11.24 | 128 |
| digits | 0.9922 | 0.7266 | 1.25 | 128 |
| motors | 7.477 | 5.469 | 9.375 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 19.95 | 16.45 | 23.38 | 128 |
| digits | 1.727 | 1.438 | 2.016 | 128 |
| motors | 16.65 | 13.77 | 19.56 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8199 |
| add_minimal_digit | 0.8106 |
| add_palm_body | 0.7373 |
| delete_phalanx | 0.9524 |
| insert_phalanx | 0.9927 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8615 |
| remove_digit_minimal | 0.6784 |
| remove_palm_body | 0.7567 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.7845 |
| step_limits | 1 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7744 |

## dist=G_FULL mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.227 | 0.4764 | 1.961 | 128 |
| digits | 0.75 | 0.5156 | 0.9844 | 128 |
| motors | 1.148 | 0.4453 | 1.844 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.867 | 3.531 | 6.297 | 128 |
| digits | 2.234 | 1.953 | 2.531 | 128 |
| motors | 4.508 | 3.25 | 5.805 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9973 |
| add_minimal_digit | 0.8474 |
| add_palm_body | 0.5527 |
| delete_phalanx | 0.7759 |
| insert_phalanx | 0.9993 |
| remove_branch_digit | 0.726 |
| remove_digit_minimal | 0.7635 |
| remove_palm_body_empty | 0.5139 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2758 |
| step_limits | 0.9993 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.965 |

## dist=G_FULL mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.9766 | 0.3047 | 1.586 | 128 |
| digits | 0.625 | 0.3984 | 0.8672 | 128 |
| motors | 0.7656 | 0.1016 | 1.36 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.766 | 2.414 | 5.094 | 128 |
| digits | 2.008 | 1.742 | 2.274 | 128 |
| motors | 3.547 | 2.336 | 4.742 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9978 |
| add_minimal_digit | 0.8612 |
| add_palm_body | 0.5263 |
| delete_phalanx | 0.7818 |
| insert_phalanx | 0.9974 |
| remove_branch_digit | 0.7091 |
| remove_digit_minimal | 0.7419 |
| remove_palm_body_empty | 0.5031 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2964 |
| step_limits | 0.9991 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9395 |

## dist=G_NOBRANCH mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 7.547 | 6.391 | 8.766 | 128 |
| digits | 1.188 | 0.8984 | 1.484 | 128 |
| motors | 6.367 | 5.351 | 7.453 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.773 | 8.562 | 11.02 | 128 |
| digits | 1.789 | 1.477 | 2.109 | 128 |
| motors | 8.234 | 7.148 | 9.297 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8431 |
| delete_phalanx | 0.9541 |
| insert_phalanx | 0.9602 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8256 |
| resample_parameter | 1 |

## dist=G_NOBRANCH mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 5.469 | 4.554 | 6.477 | 128 |
| digits | 1.156 | 0.8906 | 1.438 | 128 |
| motors | 4.555 | 3.742 | 5.414 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.812 | 7.672 | 10.02 | 128 |
| digits | 2.109 | 1.828 | 2.398 | 128 |
| motors | 7.547 | 6.57 | 8.594 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8369 |
| add_minimal_digit | 0.8378 |
| add_palm_body | 0.7528 |
| delete_phalanx | 0.9408 |
| insert_phalanx | 0.9792 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8518 |
| remove_digit_minimal | 0.6922 |
| remove_palm_body | 0.7556 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.6277 |
| step_limits | 0.9981 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7707 |

## dist=G_NOBRANCH mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.273 | 3.461 | 5.086 | 128 |
| digits | 0.9609 | 0.7031 | 1.211 | 128 |
| motors | 3.75 | 3.047 | 4.469 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.609 | 7.445 | 9.734 | 128 |
| digits | 1.875 | 1.578 | 2.156 | 128 |
| motors | 7.438 | 6.484 | 8.367 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8544 |
| add_minimal_digit | 0.8407 |
| add_palm_body | 0.7599 |
| delete_phalanx | 0.9353 |
| insert_phalanx | 0.967 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.867 |
| remove_digit_minimal | 0.7214 |
| remove_palm_body | 0.7773 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.6167 |
| step_limits | 0.9994 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7417 |

## dist=G_NOBRANCH mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 2.414 | 1.93 | 2.867 | 128 |
| digits | 0.8594 | 0.6092 | 1.125 | 128 |
| motors | 2.227 | 1.766 | 2.672 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.492 | 7.312 | 9.641 | 128 |
| digits | 1.844 | 1.57 | 2.109 | 128 |
| motors | 7.93 | 6.797 | 9.024 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9993 |
| add_minimal_digit | 0.8546 |
| add_palm_body | 0.5074 |
| delete_phalanx | 0.7358 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.6773 |
| remove_digit_minimal | 0.7655 |
| remove_palm_body_empty | 0.4933 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.199 |
| step_limits | 0.998 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.966 |

## dist=G_NOBRANCH mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 2.5 | 2.078 | 2.969 | 128 |
| digits | 0.7891 | 0.5703 | 1.031 | 128 |
| motors | 2.305 | 1.883 | 2.766 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.539 | 7.539 | 9.531 | 128 |
| digits | 2.203 | 1.922 | 2.492 | 128 |
| motors | 7.891 | 6.938 | 8.859 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9992 |
| add_minimal_digit | 0.8411 |
| add_palm_body | 0.5493 |
| delete_phalanx | 0.745 |
| insert_phalanx | 0.9996 |
| remove_branch_digit | 0.7216 |
| remove_digit_minimal | 0.7541 |
| remove_palm_body_empty | 0.5034 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.1934 |
| step_limits | 0.9994 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9642 |

## dist=G_FULL_INS mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.352 | -2.781 | -0.04688 | 128 |
| digits | 1.172 | 0.875 | 1.461 | 128 |
| motors | -1.297 | -2.57 | -0.125 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.6016 | -1.961 | 0.7893 | 128 |
| digits | 1.656 | 1.312 | 1.984 | 128 |
| motors | -0.4531 | -1.703 | 0.7895 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8403 |
| delete_phalanx | 0.896 |
| insert_phalanx | 0.9979 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8203 |
| resample_parameter | 1 |

## dist=G_FULL_INS mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.07 | -2.281 | 0.07031 | 128 |
| digits | 0.9844 | 0.6953 | 1.281 | 128 |
| motors | -0.9688 | -2.055 | 0.01562 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.016 | -2.422 | 0.3283 | 128 |
| digits | 1.578 | 1.32 | 1.836 | 128 |
| motors | -1 | -2.242 | 0.1562 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8505 |
| add_minimal_digit | 0.8561 |
| add_palm_body | 0.7246 |
| delete_phalanx | 0.8484 |
| insert_phalanx | 0.9958 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8204 |
| remove_digit_minimal | 0.7912 |
| remove_palm_body | 0.7581 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.3609 |
| step_limits | 0.9961 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7522 |

## dist=G_FULL_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.336 | -2.32 | -0.398 | 128 |
| digits | 0.7969 | 0.5547 | 1.047 | 128 |
| motors | -1.125 | -2.008 | -0.2734 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.219 | -2.594 | 0.07852 | 128 |
| digits | 1.75 | 1.461 | 2.031 | 128 |
| motors | -1.203 | -2.414 | -0.02344 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8811 |
| add_minimal_digit | 0.8611 |
| add_palm_body | 0.7489 |
| delete_phalanx | 0.8449 |
| insert_phalanx | 0.9989 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8155 |
| remove_digit_minimal | 0.7863 |
| remove_palm_body | 0.7566 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.334 |
| step_limits | 0.9962 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.778 |

## dist=G_FULL_INS mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.508 | 0.6719 | 2.227 | 128 |
| digits | 0.7656 | 0.5312 | 1.008 | 128 |
| motors | 1.359 | 0.5469 | 2.039 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.375 | 2.906 | 5.859 | 128 |
| digits | 2.141 | 1.851 | 2.422 | 128 |
| motors | 4.227 | 2.914 | 5.539 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9987 |
| add_minimal_digit | 0.8395 |
| add_palm_body | 0.5224 |
| delete_phalanx | 0.7815 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7407 |
| remove_digit_minimal | 0.7683 |
| remove_palm_body_empty | 0.4888 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2698 |
| step_limits | 0.9997 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9569 |

## dist=G_FULL_INS mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.7734 | 0.1246 | 1.406 | 128 |
| digits | 0.3984 | 0.1875 | 0.6172 | 128 |
| motors | 0.7422 | 0.125 | 1.328 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.742 | 3.359 | 6.156 | 128 |
| digits | 1.992 | 1.734 | 2.25 | 128 |
| motors | 4.641 | 3.375 | 5.899 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9961 |
| add_minimal_digit | 0.8685 |
| add_palm_body | 0.5493 |
| delete_phalanx | 0.7816 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7165 |
| remove_digit_minimal | 0.7303 |
| remove_palm_body_empty | 0.5047 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.3109 |
| step_limits | 0.9986 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9549 |

## dist=G_NOBRANCH_INS mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.852 | 3.242 | 4.453 | 128 |
| digits | 1.484 | 1.211 | 1.75 | 128 |
| motors | 3.375 | 2.781 | 3.946 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.016 | 3.359 | 4.719 | 128 |
| digits | 1.773 | 1.477 | 2.086 | 128 |
| motors | 3.578 | 2.984 | 4.188 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8441 |
| delete_phalanx | 0.8877 |
| insert_phalanx | 0.9981 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8223 |
| resample_parameter | 1 |

## dist=G_NOBRANCH_INS mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 2.117 | 1.508 | 2.773 | 128 |
| digits | 1.094 | 0.8203 | 1.391 | 128 |
| motors | 1.93 | 1.398 | 2.516 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.711 | 3.054 | 4.352 | 128 |
| digits | 1.859 | 1.554 | 2.18 | 128 |
| motors | 3.227 | 2.617 | 3.82 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8544 |
| add_minimal_digit | 0.8474 |
| add_palm_body | 0.7487 |
| delete_phalanx | 0.8363 |
| insert_phalanx | 0.9984 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8177 |
| remove_digit_minimal | 0.792 |
| remove_palm_body | 0.7541 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.3292 |
| step_limits | 0.9961 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7585 |

## dist=G_NOBRANCH_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.828 | 1.289 | 2.352 | 128 |
| digits | 1.008 | 0.7422 | 1.289 | 128 |
| motors | 1.539 | 1.055 | 2.024 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.164 | 2.453 | 3.852 | 128 |
| digits | 1.508 | 1.195 | 1.821 | 128 |
| motors | 2.945 | 2.289 | 3.578 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8662 |
| add_minimal_digit | 0.8657 |
| add_palm_body | 0.7399 |
| delete_phalanx | 0.8278 |
| insert_phalanx | 0.9963 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.7945 |
| remove_digit_minimal | 0.7837 |
| remove_palm_body | 0.7649 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.3066 |
| step_limits | 0.9959 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7632 |

## dist=G_NOBRANCH_INS mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 2.586 | 2.094 | 3.055 | 128 |
| digits | 0.5859 | 0.3516 | 0.8127 | 128 |
| motors | 2.32 | 1.844 | 2.797 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.906 | 7.953 | 9.945 | 128 |
| digits | 2.031 | 1.75 | 2.32 | 128 |
| motors | 8.336 | 7.398 | 9.367 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9984 |
| add_minimal_digit | 0.8383 |
| add_palm_body | 0.5335 |
| delete_phalanx | 0.7584 |
| insert_phalanx | 0.9997 |
| remove_branch_digit | 0.7081 |
| remove_digit_minimal | 0.735 |
| remove_palm_body_empty | 0.5032 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.208 |
| step_limits | 0.999 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9644 |

## dist=G_NOBRANCH_INS mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 2.305 | 1.883 | 2.734 | 128 |
| digits | 0.6719 | 0.4375 | 0.9141 | 128 |
| motors | 2.055 | 1.625 | 2.484 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 7.672 | 6.664 | 8.688 | 128 |
| digits | 1.781 | 1.5 | 2.078 | 128 |
| motors | 7.148 | 6.211 | 8.125 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9955 |
| add_minimal_digit | 0.8719 |
| add_palm_body | 0.5563 |
| delete_phalanx | 0.7403 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.6635 |
| remove_digit_minimal | 0.7692 |
| remove_palm_body_empty | 0.5246 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.186 |
| step_limits | 0.9986 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.9608 |

## Reading

(I15 fix 6: every number below tagged "Δ" is a DELTA from that (seed, dist)'s shared reduced START -- not an absolute joint/digit/motor count; start sizes are reported separately above.)

- G_FULL start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL/DEFAULT_uniform step 40: Δjoints 16.01 [12.84, 19.17]; Δdigits 1.289 [1.008, 1.563]; Δmotors 13.23 [10.55, 15.98]; distinct_hash_fraction 1.
- G_FULL/DEFAULT_uniform step 400: Δjoints 21.47 [17.37, 25.81]; Δdigits 1.711 [1.406, 2.031]; Δmotors 17.96 [14.45, 21.7]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 40: Δjoints 8.898 [6.445, 11.39]; Δdigits 1.125 [0.8594, 1.406]; Δmotors 7.68 [5.562, 9.812]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 400: Δjoints 21.93 [18.43, 25.7]; Δdigits 1.883 [1.602, 2.141]; Δmotors 18.81 [15.81, 22.03]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 40: Δjoints 9.031 [6.742, 11.24]; Δdigits 0.9922 [0.7266, 1.25]; Δmotors 7.477 [5.469, 9.375]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 400: Δjoints 19.95 [16.45, 23.38]; Δdigits 1.727 [1.438, 2.016]; Δmotors 16.65 [13.77, 19.56]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_uniform step 40: Δjoints 1.227 [0.4764, 1.961]; Δdigits 0.75 [0.5156, 0.9844]; Δmotors 1.148 [0.4453, 1.844]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_uniform step 400: Δjoints 4.867 [3.531, 6.297]; Δdigits 2.234 [1.953, 2.531]; Δmotors 4.508 [3.25, 5.805]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_weighted step 40: Δjoints 0.9766 [0.3047, 1.586]; Δdigits 0.625 [0.3984, 0.8672]; Δmotors 0.7656 [0.1016, 1.36]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_weighted step 400: Δjoints 3.766 [2.414, 5.094]; Δdigits 2.008 [1.742, 2.274]; Δmotors 3.547 [2.336, 4.742]; distinct_hash_fraction 1.
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -1.516 [-6.744, 3.821] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.01562 [-0.4453, 0.4297] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -1.312 [-5.777, 3.141] (n=128).
- G_NOBRANCH start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH/DEFAULT_uniform step 40: Δjoints 7.547 [6.391, 8.766]; Δdigits 1.188 [0.8984, 1.484]; Δmotors 6.367 [5.351, 7.453]; distinct_hash_fraction 1.
- G_NOBRANCH/DEFAULT_uniform step 400: Δjoints 9.773 [8.562, 11.02]; Δdigits 1.789 [1.477, 2.109]; Δmotors 8.234 [7.148, 9.297]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 40: Δjoints 5.469 [4.554, 6.477]; Δdigits 1.156 [0.8906, 1.438]; Δmotors 4.555 [3.742, 5.414]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 400: Δjoints 8.812 [7.672, 10.02]; Δdigits 2.109 [1.828, 2.398]; Δmotors 7.547 [6.57, 8.594]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 40: Δjoints 4.273 [3.461, 5.086]; Δdigits 0.9609 [0.7031, 1.211]; Δmotors 3.75 [3.047, 4.469]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 400: Δjoints 8.609 [7.445, 9.734]; Δdigits 1.875 [1.578, 2.156]; Δmotors 7.438 [6.484, 8.367]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_uniform step 40: Δjoints 2.414 [1.93, 2.867]; Δdigits 0.8594 [0.6092, 1.125]; Δmotors 2.227 [1.766, 2.672]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_uniform step 400: Δjoints 8.492 [7.312, 9.641]; Δdigits 1.844 [1.57, 2.109]; Δmotors 7.93 [6.797, 9.024]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_weighted step 40: Δjoints 2.5 [2.078, 2.969]; Δdigits 0.7891 [0.5703, 1.031]; Δmotors 2.305 [1.883, 2.766]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_weighted step 400: Δjoints 8.539 [7.539, 9.531]; Δdigits 2.203 [1.922, 2.492]; Δmotors 7.891 [6.938, 8.859]; distinct_hash_fraction 1.
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -1.164 [-2.719, 0.4846] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.08594 [-0.3439, 0.5078] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.7969 [-2.141, 0.5393] (n=128).
- G_FULL_INS start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL_INS/DEFAULT_uniform step 40: Δjoints -1.352 [-2.781, -0.04688]; Δdigits 1.172 [0.875, 1.461]; Δmotors -1.297 [-2.57, -0.125]; distinct_hash_fraction 1.
- G_FULL_INS/DEFAULT_uniform step 400: Δjoints -0.6016 [-1.961, 0.7893]; Δdigits 1.656 [1.312, 1.984]; Δmotors -0.4531 [-1.703, 0.7895]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 40: Δjoints -1.07 [-2.281, 0.07031]; Δdigits 0.9844 [0.6953, 1.281]; Δmotors -0.9688 [-2.055, 0.01562]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 400: Δjoints -1.016 [-2.422, 0.3283]; Δdigits 1.578 [1.32, 1.836]; Δmotors -1 [-2.242, 0.1562]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 40: Δjoints -1.336 [-2.32, -0.398]; Δdigits 0.7969 [0.5547, 1.047]; Δmotors -1.125 [-2.008, -0.2734]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 400: Δjoints -1.219 [-2.594, 0.07852]; Δdigits 1.75 [1.461, 2.031]; Δmotors -1.203 [-2.414, -0.02344]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_uniform step 40: Δjoints 1.508 [0.6719, 2.227]; Δdigits 0.7656 [0.5312, 1.008]; Δmotors 1.359 [0.5469, 2.039]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_uniform step 400: Δjoints 4.375 [2.906, 5.859]; Δdigits 2.141 [1.851, 2.422]; Δmotors 4.227 [2.914, 5.539]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_weighted step 40: Δjoints 0.7734 [0.1246, 1.406]; Δdigits 0.3984 [0.1875, 0.6172]; Δmotors 0.7422 [0.125, 1.328]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_weighted step 400: Δjoints 4.742 [3.359, 6.156]; Δdigits 1.992 [1.734, 2.25]; Δmotors 4.641 [3.375, 5.899]; distinct_hash_fraction 1.
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.6172 [-1.547, 0.3047] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.09375 [-0.2893, 0.5] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.75 [-1.555, 0.07832] (n=128).
- G_NOBRANCH_INS start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH_INS/DEFAULT_uniform step 40: Δjoints 3.852 [3.242, 4.453]; Δdigits 1.484 [1.211, 1.75]; Δmotors 3.375 [2.781, 3.946]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/DEFAULT_uniform step 400: Δjoints 4.016 [3.359, 4.719]; Δdigits 1.773 [1.477, 2.086]; Δmotors 3.578 [2.984, 4.188]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 40: Δjoints 2.117 [1.508, 2.773]; Δdigits 1.094 [0.8203, 1.391]; Δmotors 1.93 [1.398, 2.516]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 400: Δjoints 3.711 [3.054, 4.352]; Δdigits 1.859 [1.554, 2.18]; Δmotors 3.227 [2.617, 3.82]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 40: Δjoints 1.828 [1.289, 2.352]; Δdigits 1.008 [0.7422, 1.289]; Δmotors 1.539 [1.055, 2.024]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 400: Δjoints 3.164 [2.453, 3.852]; Δdigits 1.508 [1.195, 1.821]; Δmotors 2.945 [2.289, 3.578]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_uniform step 40: Δjoints 2.586 [2.094, 3.055]; Δdigits 0.5859 [0.3516, 0.8127]; Δmotors 2.32 [1.844, 2.797]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_uniform step 400: Δjoints 8.906 [7.953, 9.945]; Δdigits 2.031 [1.75, 2.32]; Δmotors 8.336 [7.398, 9.367]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_weighted step 40: Δjoints 2.305 [1.883, 2.734]; Δdigits 0.6719 [0.4375, 0.9141]; Δmotors 2.055 [1.625, 2.484]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_weighted step 400: Δjoints 7.672 [6.664, 8.688]; Δdigits 1.781 [1.5, 2.078]; Δmotors 7.148 [6.211, 8.125]; distinct_hash_fraction 1.
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.8516 [-1.797, 0.1094] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits -0.2656 [-0.6641, 0.1562] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.6328 [-1.445, 0.2266] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δjoints -22.07 [-25.97, -18.46] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δdigits -0.05469 [-0.5078, 0.3906] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δmotors -18.41 [-21.76, -15.27] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δjoints -22.95 [-26.58, -19.6] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δdigits -0.3047 [-0.6719, 0.07031] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δmotors -19.81 [-22.9, -16.96] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δjoints -21.17 [-24.56, -17.76] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δdigits 0.02344 [-0.3672, 0.4143] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δmotors -17.85 [-20.77, -14.89] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δjoints -0.4922 [-2.234, 1.234] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δdigits -0.09375 [-0.5156, 0.2812] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δmotors -0.2812 [-1.922, 1.406] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δjoints 0.9766 [-0.625, 2.594] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δdigits -0.01562 [-0.375, 0.3283] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δmotors 1.094 [-0.4377, 2.649] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δjoints -5.758 [-7.25, -4.219] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δdigits -0.01562 [-0.5, 0.4609] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δmotors -4.656 [-5.946, -3.336] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δjoints -5.102 [-6.391, -3.867] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δdigits -0.25 [-0.625, 0.1174] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δmotors -4.32 [-5.492, -3.211] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δjoints -5.445 [-6.703, -4.101] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δdigits -0.3672 [-0.7814, 0.05469] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δmotors -4.492 [-5.555, -3.422] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δjoints 0.4141 [-1.031, 1.875] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δdigits 0.1875 [-0.1797, 0.5549] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δmotors 0.4062 [-1.039, 1.852] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δjoints -0.8672 [-2.321, 0.5625] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δdigits -0.4219 [-0.7971, -0.04668] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δmotors -0.7422 [-2.125, 0.6176] (n=128).
