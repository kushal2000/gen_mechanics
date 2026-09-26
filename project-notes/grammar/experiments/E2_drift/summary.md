# Experiment: e2_drift

Wall time: 13.871 s

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
| G_FULL | UNION_uniform | 31.42 | 28.14 | 34.99 | 128 |
| G_FULL | UNION_weighted | 29.57 | 26.13 | 33.06 | 128 |
| G_FULL | EVOLUTION_uniform | 13 | 11.87 | 14.17 | 128 |
| G_FULL | EVOLUTION_weighted | 13.02 | 11.97 | 14.19 | 128 |
| G_NOBRANCH | DEFAULT_uniform | 14.4 | 13.19 | 15.61 | 128 |
| G_NOBRANCH | UNION_uniform | 13.44 | 12.3 | 14.62 | 128 |
| G_NOBRANCH | UNION_weighted | 13.23 | 12.08 | 14.37 | 128 |
| G_NOBRANCH | EVOLUTION_uniform | 13.41 | 12.23 | 14.64 | 128 |
| G_NOBRANCH | EVOLUTION_weighted | 11.86 | 11.09 | 12.65 | 128 |
| G_FULL_INS | DEFAULT_uniform | 8.688 | 7.938 | 9.469 | 128 |
| G_FULL_INS | UNION_uniform | 8.273 | 7.672 | 8.891 | 128 |
| G_FULL_INS | UNION_weighted | 8.07 | 7.492 | 8.68 | 128 |
| G_FULL_INS | EVOLUTION_uniform | 12.47 | 11.28 | 13.81 | 128 |
| G_FULL_INS | EVOLUTION_weighted | 12.64 | 11.4 | 13.91 | 128 |
| G_NOBRANCH_INS | DEFAULT_uniform | 8.641 | 7.938 | 9.375 | 128 |
| G_NOBRANCH_INS | UNION_uniform | 8.336 | 7.727 | 8.914 | 128 |
| G_NOBRANCH_INS | UNION_weighted | 7.789 | 7.117 | 8.445 | 128 |
| G_NOBRANCH_INS | EVOLUTION_uniform | 12.88 | 11.78 | 13.95 | 128 |
| G_NOBRANCH_INS | EVOLUTION_weighted | 11.23 | 10.39 | 12.08 | 128 |

## Difference CI: UNION_weighted minus DEFAULT_uniform, final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| dist | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|
| G_FULL | joints | -1.188 | -6.415 | 4.103 | 128 |
| G_FULL | digits | -0.007812 | -0.4688 | 0.4297 | 128 |
| G_FULL | motors | -1 | -5.556 | 3.438 | 128 |
| G_NOBRANCH | joints | -1.164 | -2.719 | 0.4846 | 128 |
| G_NOBRANCH | digits | 0.08594 | -0.3439 | 0.5078 | 128 |
| G_NOBRANCH | motors | -0.7969 | -2.141 | 0.5393 | 128 |
| G_FULL_INS | joints | -0.6172 | -1.547 | 0.3047 | 128 |
| G_FULL_INS | digits | 0.1016 | -0.2812 | 0.5078 | 128 |
| G_FULL_INS | motors | -0.7344 | -1.547 | 0.09395 | 128 |
| G_NOBRANCH_INS | joints | -0.8516 | -1.797 | 0.1094 | 128 |
| G_NOBRANCH_INS | digits | -0.2656 | -0.6641 | 0.1562 | 128 |
| G_NOBRANCH_INS | motors | -0.6328 | -1.445 | 0.2266 | 128 |

## Difference CI: _INS minus plain, per pool (mixture), final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| pair | mixture | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|---|
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | joints | -22.07 | -25.97 | -18.46 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | digits | -0.1719 | -0.6562 | 0.2969 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | motors | -18.41 | -21.76 | -15.27 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | joints | -23.15 | -26.83 | -19.81 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | digits | -0.3672 | -0.7344 | 0.0001953 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | motors | -19.95 | -23.02 | -17.09 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | joints | -21.5 | -24.95 | -18.07 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | digits | -0.0625 | -0.4609 | 0.3516 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | motors | -18.15 | -21.17 | -15.15 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | joints | -0.5312 | -2.242 | 1.266 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | digits | -0.05469 | -0.4844 | 0.4219 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_uniform | motors | -0.4688 | -2.102 | 1.266 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | joints | -0.375 | -1.891 | 1.133 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | digits | -0.1094 | -0.5312 | 0.3281 | 128 |
| G_FULL_INS_minus_G_FULL | EVOLUTION_weighted | motors | -0.3672 | -1.789 | 1.094 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | joints | -5.758 | -7.25 | -4.219 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | digits | -0.01562 | -0.5 | 0.4609 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | motors | -4.656 | -5.946 | -3.336 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | joints | -5.102 | -6.391 | -3.867 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | digits | -0.25 | -0.625 | 0.1174 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | motors | -4.32 | -5.492 | -3.211 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | joints | -5.445 | -6.703 | -4.101 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | digits | -0.3672 | -0.7814 | 0.05469 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | motors | -4.492 | -5.555 | -3.422 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | joints | -0.5312 | -2.102 | 1.001 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | digits | -0.3281 | -0.7188 | 0.07031 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_uniform | motors | -0.6016 | -2.078 | 0.8443 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | joints | -0.625 | -1.68 | 0.4688 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | digits | -0.3906 | -0.8049 | 0.03125 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | EVOLUTION_weighted | motors | -0.8672 | -1.859 | 0.1484 | 128 |

## dist=G_FULL mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 16.01 | 12.84 | 19.17 | 128 |
| digits | 1.039 | 0.7029 | 1.375 | 128 |
| motors | 13.23 | 10.55 | 15.98 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 21.47 | 17.37 | 25.81 | 128 |
| digits | 1.477 | 1.094 | 1.867 | 128 |
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
| digits | 0.8828 | 0.5703 | 1.211 | 128 |
| motors | 7.703 | 5.585 | 9.836 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 22.13 | 18.65 | 25.88 | 128 |
| digits | 1.594 | 1.281 | 1.891 | 128 |
| motors | 18.95 | 15.92 | 22.13 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8224 |
| add_minimal_digit | 0.8206 |
| add_palm_body | 0.7283 |
| delete_phalanx | 0.9604 |
| insert_phalanx | 0.9917 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.859 |
| remove_digit_minimal | 0.659 |
| remove_palm_body | 0.7617 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.8076 |
| step_limits | 0.9976 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7549 |

## dist=G_FULL mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.031 | 6.742 | 11.24 | 128 |
| digits | 0.8047 | 0.5076 | 1.078 | 128 |
| motors | 7.477 | 5.469 | 9.375 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 20.28 | 16.84 | 23.8 | 128 |
| digits | 1.469 | 1.125 | 1.812 | 128 |
| motors | 16.96 | 14.05 | 19.95 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8197 |
| add_minimal_digit | 0.8101 |
| add_palm_body | 0.7374 |
| delete_phalanx | 0.9525 |
| insert_phalanx | 0.9927 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8614 |
| remove_digit_minimal | 0.678 |
| remove_palm_body | 0.7563 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.7848 |
| step_limits | 1 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7737 |

## dist=G_FULL mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.9453 | 0.2109 | 1.695 | 128 |
| digits | 0.7188 | 0.4219 | 1.016 | 128 |
| motors | 1.055 | 0.3826 | 1.711 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.711 | 2.187 | 5.211 | 128 |
| digits | 1.93 | 1.586 | 2.258 | 128 |
| motors | 3.922 | 2.531 | 5.313 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.998 |
| add_minimal_digit | 0.8504 |
| add_palm_body | 0.7348 |
| delete_phalanx | 0.7749 |
| insert_phalanx | 0.9997 |
| remove_branch_digit | 0.7415 |
| remove_digit_minimal | 0.7736 |
| remove_palm_body | 0.7644 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2564 |
| step_limits | 0.999 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7808 |

## dist=G_FULL mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.7969 | 0.08594 | 1.469 | 128 |
| digits | 0.7422 | 0.4453 | 1.055 | 128 |
| motors | 0.8047 | 0.1326 | 1.438 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.727 | 2.391 | 5.024 | 128 |
| digits | 2.227 | 1.898 | 2.555 | 128 |
| motors | 3.836 | 2.586 | 5.024 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9991 |
| add_minimal_digit | 0.8485 |
| add_palm_body | 0.7435 |
| delete_phalanx | 0.7626 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7226 |
| remove_digit_minimal | 0.7724 |
| remove_palm_body | 0.7341 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2754 |
| step_limits | 1 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7516 |

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
| joints | 1.867 | 1.398 | 2.328 | 128 |
| digits | 0.7266 | 0.4842 | 0.9609 | 128 |
| motors | 1.805 | 1.359 | 2.258 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.781 | 7.664 | 9.977 | 128 |
| digits | 2.453 | 2.164 | 2.742 | 128 |
| motors | 8.633 | 7.516 | 9.797 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.998 |
| add_minimal_digit | 0.839 |
| add_palm_body | 0.7374 |
| delete_phalanx | 0.736 |
| insert_phalanx | 0.9993 |
| remove_branch_digit | 0.6873 |
| remove_digit_minimal | 0.7337 |
| remove_palm_body | 0.7754 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2259 |
| step_limits | 0.9983 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7627 |

## dist=G_NOBRANCH mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.906 | 1.453 | 2.344 | 128 |
| digits | 0.9297 | 0.6953 | 1.172 | 128 |
| motors | 2.016 | 1.547 | 2.445 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 7.234 | 6.453 | 8.031 | 128 |
| digits | 2.586 | 2.273 | 2.922 | 128 |
| motors | 7.18 | 6.406 | 7.953 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9979 |
| add_minimal_digit | 0.8326 |
| add_palm_body | 0.7365 |
| delete_phalanx | 0.7376 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7 |
| remove_digit_minimal | 0.7742 |
| remove_palm_body | 0.7459 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.165 |
| step_limits | 0.9994 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7472 |

## dist=G_FULL_INS mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.352 | -2.781 | -0.04688 | 128 |
| digits | 0.8906 | 0.5232 | 1.25 | 128 |
| motors | -1.297 | -2.57 | -0.125 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.6016 | -1.961 | 0.7893 | 128 |
| digits | 1.305 | 0.9219 | 1.664 | 128 |
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
| digits | 0.8203 | 0.4766 | 1.164 | 128 |
| motors | -0.9688 | -2.055 | 0.01562 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.016 | -2.422 | 0.3283 | 128 |
| digits | 1.227 | 0.9451 | 1.531 | 128 |
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
| digits | 0.5625 | 0.2812 | 0.8438 | 128 |
| motors | -1.125 | -2.008 | -0.2734 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.219 | -2.594 | 0.07852 | 128 |
| digits | 1.406 | 1.047 | 1.758 | 128 |
| motors | -1.188 | -2.391 | -0.007812 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8839 |
| add_minimal_digit | 0.8607 |
| add_palm_body | 0.75 |
| delete_phalanx | 0.845 |
| insert_phalanx | 0.9989 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.816 |
| remove_digit_minimal | 0.7861 |
| remove_palm_body | 0.7563 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.3352 |
| step_limits | 0.9962 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7743 |

## dist=G_FULL_INS mixture=EVOLUTION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.328 | 0.6172 | 1.961 | 128 |
| digits | 0.75 | 0.4766 | 1.031 | 128 |
| motors | 1.438 | 0.8047 | 2.039 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.18 | 1.672 | 4.805 | 128 |
| digits | 1.875 | 1.5 | 2.242 | 128 |
| motors | 3.453 | 2.062 | 4.898 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.997 |
| add_minimal_digit | 0.8451 |
| add_palm_body | 0.76 |
| delete_phalanx | 0.787 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7211 |
| remove_digit_minimal | 0.7515 |
| remove_palm_body | 0.7582 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2748 |
| step_limits | 1 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7554 |

## dist=G_FULL_INS mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.007812 | -0.6641 | 0.6645 | 128 |
| digits | 0.2969 | 0.03125 | 0.5781 | 128 |
| motors | 0.1719 | -0.4531 | 0.7422 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.352 | 1.828 | 4.789 | 128 |
| digits | 2.117 | 1.781 | 2.453 | 128 |
| motors | 3.469 | 2.102 | 4.797 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9943 |
| add_minimal_digit | 0.8676 |
| add_palm_body | 0.7678 |
| delete_phalanx | 0.7801 |
| insert_phalanx | 0.9978 |
| remove_branch_digit | 0.71 |
| remove_digit_minimal | 0.7314 |
| remove_palm_body | 0.7523 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.2789 |
| step_limits | 0.9986 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7452 |

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
| joints | 2.219 | 1.75 | 2.719 | 128 |
| digits | 0.8516 | 0.6016 | 1.109 | 128 |
| motors | 2.188 | 1.726 | 2.68 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.25 | 7.156 | 9.383 | 128 |
| digits | 2.125 | 1.828 | 2.422 | 128 |
| motors | 8.031 | 7.023 | 9.07 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9955 |
| add_minimal_digit | 0.854 |
| add_palm_body | 0.7457 |
| delete_phalanx | 0.7488 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.7034 |
| remove_digit_minimal | 0.7623 |
| remove_palm_body | 0.7549 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.179 |
| step_limits | 0.999 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7633 |

## dist=G_NOBRANCH_INS mixture=EVOLUTION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.891 | 1.43 | 2.383 | 128 |
| digits | 0.7031 | 0.4609 | 0.9375 | 128 |
| motors | 1.867 | 1.406 | 2.344 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 6.609 | 5.758 | 7.477 | 128 |
| digits | 2.195 | 1.875 | 2.516 | 128 |
| motors | 6.312 | 5.469 | 7.148 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_branch_digit | 0.9965 |
| add_minimal_digit | 0.8806 |
| add_palm_body | 0.7683 |
| delete_phalanx | 0.738 |
| insert_phalanx | 1 |
| remove_branch_digit | 0.68 |
| remove_digit_minimal | 0.7483 |
| remove_palm_body | 0.7496 |
| step_axis | 1 |
| step_bend_offset | 0 |
| step_bend_rpy | 0 |
| step_coupling | 0.1573 |
| step_limits | 0.9997 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7542 |

## Reading

(I15 fix 6: every number below tagged "Δ" is a DELTA from that (seed, dist)'s shared reduced START -- not an absolute joint/digit/motor count; start sizes are reported separately above.)

- G_FULL start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL/DEFAULT_uniform step 40: Δjoints 16.01 [12.84, 19.17]; Δdigits 1.039 [0.7029, 1.375]; Δmotors 13.23 [10.55, 15.98]; distinct_hash_fraction 1.
- G_FULL/DEFAULT_uniform step 400: Δjoints 21.47 [17.37, 25.81]; Δdigits 1.477 [1.094, 1.867]; Δmotors 17.96 [14.45, 21.7]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 40: Δjoints 8.898 [6.445, 11.39]; Δdigits 0.8828 [0.5703, 1.211]; Δmotors 7.703 [5.585, 9.836]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 400: Δjoints 22.13 [18.65, 25.88]; Δdigits 1.594 [1.281, 1.891]; Δmotors 18.95 [15.92, 22.13]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 40: Δjoints 9.031 [6.742, 11.24]; Δdigits 0.8047 [0.5076, 1.078]; Δmotors 7.477 [5.469, 9.375]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 400: Δjoints 20.28 [16.84, 23.8]; Δdigits 1.469 [1.125, 1.812]; Δmotors 16.96 [14.05, 19.95]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_uniform step 40: Δjoints 0.9453 [0.2109, 1.695]; Δdigits 0.7188 [0.4219, 1.016]; Δmotors 1.055 [0.3826, 1.711]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_uniform step 400: Δjoints 3.711 [2.187, 5.211]; Δdigits 1.93 [1.586, 2.258]; Δmotors 3.922 [2.531, 5.313]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_weighted step 40: Δjoints 0.7969 [0.08594, 1.469]; Δdigits 0.7422 [0.4453, 1.055]; Δmotors 0.8047 [0.1326, 1.438]; distinct_hash_fraction 1.
- G_FULL/EVOLUTION_weighted step 400: Δjoints 3.727 [2.391, 5.024]; Δdigits 2.227 [1.898, 2.555]; Δmotors 3.836 [2.586, 5.024]; distinct_hash_fraction 1.
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -1.188 [-6.415, 4.103] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits -0.007812 [-0.4688, 0.4297] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -1 [-5.556, 3.438] (n=128).
- G_NOBRANCH start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH/DEFAULT_uniform step 40: Δjoints 7.547 [6.391, 8.766]; Δdigits 1.188 [0.8984, 1.484]; Δmotors 6.367 [5.351, 7.453]; distinct_hash_fraction 1.
- G_NOBRANCH/DEFAULT_uniform step 400: Δjoints 9.773 [8.562, 11.02]; Δdigits 1.789 [1.477, 2.109]; Δmotors 8.234 [7.148, 9.297]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 40: Δjoints 5.469 [4.554, 6.477]; Δdigits 1.156 [0.8906, 1.438]; Δmotors 4.555 [3.742, 5.414]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 400: Δjoints 8.812 [7.672, 10.02]; Δdigits 2.109 [1.828, 2.398]; Δmotors 7.547 [6.57, 8.594]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 40: Δjoints 4.273 [3.461, 5.086]; Δdigits 0.9609 [0.7031, 1.211]; Δmotors 3.75 [3.047, 4.469]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 400: Δjoints 8.609 [7.445, 9.734]; Δdigits 1.875 [1.578, 2.156]; Δmotors 7.438 [6.484, 8.367]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_uniform step 40: Δjoints 1.867 [1.398, 2.328]; Δdigits 0.7266 [0.4842, 0.9609]; Δmotors 1.805 [1.359, 2.258]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_uniform step 400: Δjoints 8.781 [7.664, 9.977]; Δdigits 2.453 [2.164, 2.742]; Δmotors 8.633 [7.516, 9.797]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_weighted step 40: Δjoints 1.906 [1.453, 2.344]; Δdigits 0.9297 [0.6953, 1.172]; Δmotors 2.016 [1.547, 2.445]; distinct_hash_fraction 1.
- G_NOBRANCH/EVOLUTION_weighted step 400: Δjoints 7.234 [6.453, 8.031]; Δdigits 2.586 [2.273, 2.922]; Δmotors 7.18 [6.406, 7.953]; distinct_hash_fraction 1.
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -1.164 [-2.719, 0.4846] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.08594 [-0.3439, 0.5078] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.7969 [-2.141, 0.5393] (n=128).
- G_FULL_INS start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL_INS/DEFAULT_uniform step 40: Δjoints -1.352 [-2.781, -0.04688]; Δdigits 0.8906 [0.5232, 1.25]; Δmotors -1.297 [-2.57, -0.125]; distinct_hash_fraction 1.
- G_FULL_INS/DEFAULT_uniform step 400: Δjoints -0.6016 [-1.961, 0.7893]; Δdigits 1.305 [0.9219, 1.664]; Δmotors -0.4531 [-1.703, 0.7895]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 40: Δjoints -1.07 [-2.281, 0.07031]; Δdigits 0.8203 [0.4766, 1.164]; Δmotors -0.9688 [-2.055, 0.01562]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 400: Δjoints -1.016 [-2.422, 0.3283]; Δdigits 1.227 [0.9451, 1.531]; Δmotors -1 [-2.242, 0.1562]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 40: Δjoints -1.336 [-2.32, -0.398]; Δdigits 0.5625 [0.2812, 0.8438]; Δmotors -1.125 [-2.008, -0.2734]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 400: Δjoints -1.219 [-2.594, 0.07852]; Δdigits 1.406 [1.047, 1.758]; Δmotors -1.188 [-2.391, -0.007812]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_uniform step 40: Δjoints 1.328 [0.6172, 1.961]; Δdigits 0.75 [0.4766, 1.031]; Δmotors 1.438 [0.8047, 2.039]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_uniform step 400: Δjoints 3.18 [1.672, 4.805]; Δdigits 1.875 [1.5, 2.242]; Δmotors 3.453 [2.062, 4.898]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_weighted step 40: Δjoints 0.007812 [-0.6641, 0.6645]; Δdigits 0.2969 [0.03125, 0.5781]; Δmotors 0.1719 [-0.4531, 0.7422]; distinct_hash_fraction 1.
- G_FULL_INS/EVOLUTION_weighted step 400: Δjoints 3.352 [1.828, 4.789]; Δdigits 2.117 [1.781, 2.453]; Δmotors 3.469 [2.102, 4.797]; distinct_hash_fraction 1.
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.6172 [-1.547, 0.3047] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.1016 [-0.2812, 0.5078] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.7344 [-1.547, 0.09395] (n=128).
- G_NOBRANCH_INS start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH_INS/DEFAULT_uniform step 40: Δjoints 3.852 [3.242, 4.453]; Δdigits 1.484 [1.211, 1.75]; Δmotors 3.375 [2.781, 3.946]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/DEFAULT_uniform step 400: Δjoints 4.016 [3.359, 4.719]; Δdigits 1.773 [1.477, 2.086]; Δmotors 3.578 [2.984, 4.188]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 40: Δjoints 2.117 [1.508, 2.773]; Δdigits 1.094 [0.8203, 1.391]; Δmotors 1.93 [1.398, 2.516]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 400: Δjoints 3.711 [3.054, 4.352]; Δdigits 1.859 [1.554, 2.18]; Δmotors 3.227 [2.617, 3.82]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 40: Δjoints 1.828 [1.289, 2.352]; Δdigits 1.008 [0.7422, 1.289]; Δmotors 1.539 [1.055, 2.024]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 400: Δjoints 3.164 [2.453, 3.852]; Δdigits 1.508 [1.195, 1.821]; Δmotors 2.945 [2.289, 3.578]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_uniform step 40: Δjoints 2.219 [1.75, 2.719]; Δdigits 0.8516 [0.6016, 1.109]; Δmotors 2.188 [1.726, 2.68]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_uniform step 400: Δjoints 8.25 [7.156, 9.383]; Δdigits 2.125 [1.828, 2.422]; Δmotors 8.031 [7.023, 9.07]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_weighted step 40: Δjoints 1.891 [1.43, 2.383]; Δdigits 0.7031 [0.4609, 0.9375]; Δmotors 1.867 [1.406, 2.344]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/EVOLUTION_weighted step 400: Δjoints 6.609 [5.758, 7.477]; Δdigits 2.195 [1.875, 2.516]; Δmotors 6.312 [5.469, 7.148]; distinct_hash_fraction 1.
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.8516 [-1.797, 0.1094] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits -0.2656 [-0.6641, 0.1562] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.6328 [-1.445, 0.2266] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δjoints -22.07 [-25.97, -18.46] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δdigits -0.1719 [-0.6562, 0.2969] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δmotors -18.41 [-21.76, -15.27] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δjoints -23.15 [-26.83, -19.81] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δdigits -0.3672 [-0.7344, 0.0001953] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δmotors -19.95 [-23.02, -17.09] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δjoints -21.5 [-24.95, -18.07] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δdigits -0.0625 [-0.4609, 0.3516] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δmotors -18.15 [-21.17, -15.15] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δjoints -0.5312 [-2.242, 1.266] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δdigits -0.05469 [-0.4844, 0.4219] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_uniform) at step 400: Δmotors -0.4688 [-2.102, 1.266] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δjoints -0.375 [-1.891, 1.133] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δdigits -0.1094 [-0.5312, 0.3281] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=EVOLUTION_weighted) at step 400: Δmotors -0.3672 [-1.789, 1.094] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δjoints -5.758 [-7.25, -4.219] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δdigits -0.01562 [-0.5, 0.4609] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δmotors -4.656 [-5.946, -3.336] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δjoints -5.102 [-6.391, -3.867] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δdigits -0.25 [-0.625, 0.1174] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δmotors -4.32 [-5.492, -3.211] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δjoints -5.445 [-6.703, -4.101] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δdigits -0.3672 [-0.7814, 0.05469] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δmotors -4.492 [-5.555, -3.422] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δjoints -0.5312 [-2.102, 1.001] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δdigits -0.3281 [-0.7188, 0.07031] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_uniform) at step 400: Δmotors -0.6016 [-2.078, 0.8443] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δjoints -0.625 [-1.68, 0.4688] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δdigits -0.3906 [-0.8049, 0.03125] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=EVOLUTION_weighted) at step 400: Δmotors -0.8672 [-1.859, 0.1484] (n=128).
