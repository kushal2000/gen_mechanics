# Experiment: e2_drift

Wall time: 6.609 s

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
| G_FULL | UNION_uniform | 31.27 | 28.09 | 34.66 | 128 |
| G_FULL | UNION_weighted | 32.07 | 28.43 | 35.88 | 128 |
| G_NOBRANCH | DEFAULT_uniform | 14.4 | 13.19 | 15.61 | 128 |
| G_NOBRANCH | UNION_uniform | 14.21 | 13.02 | 15.42 | 128 |
| G_NOBRANCH | UNION_weighted | 14.26 | 13.18 | 15.41 | 128 |
| G_FULL_INS | DEFAULT_uniform | 8.688 | 7.938 | 9.469 | 128 |
| G_FULL_INS | UNION_uniform | 8.742 | 8.039 | 9.414 | 128 |
| G_FULL_INS | UNION_weighted | 8.305 | 7.656 | 8.961 | 128 |
| G_NOBRANCH_INS | DEFAULT_uniform | 8.641 | 7.938 | 9.375 | 128 |
| G_NOBRANCH_INS | UNION_uniform | 8.031 | 7.383 | 8.688 | 128 |
| G_NOBRANCH_INS | UNION_weighted | 7.766 | 7.164 | 8.406 | 128 |

## Difference CI: UNION_weighted minus DEFAULT_uniform, final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| dist | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|
| G_FULL | joints | 1.312 | -3.602 | 6.461 | 128 |
| G_FULL | digits | 0.3984 | -0.03125 | 0.7969 | 128 |
| G_FULL | motors | 1.398 | -2.805 | 5.774 | 128 |
| G_NOBRANCH | joints | -0.1406 | -1.649 | 1.336 | 128 |
| G_NOBRANCH | digits | 0.4531 | 0.0625 | 0.8438 | 128 |
| G_NOBRANCH | motors | 0.02344 | -1.227 | 1.242 | 128 |
| G_FULL_INS | joints | -0.3828 | -1.461 | 0.7502 | 128 |
| G_FULL_INS | digits | 0.1094 | -0.3672 | 0.6172 | 128 |
| G_FULL_INS | motors | -0.4453 | -1.43 | 0.5938 | 128 |
| G_NOBRANCH_INS | joints | -0.875 | -1.805 | 0.07812 | 128 |
| G_NOBRANCH_INS | digits | -0.25 | -0.7031 | 0.1719 | 128 |
| G_NOBRANCH_INS | motors | -0.8203 | -1.649 | 0.05469 | 128 |

## Difference CI: _INS minus plain, per pool (mixture), final joints/digits/motors at step 400 (paired by shared start; I15 fix 6)

| pair | mixture | metric | mean DIFFERENCE | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|---|
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | joints | -22.07 | -25.97 | -18.46 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | digits | -0.1719 | -0.6562 | 0.2969 | 128 |
| G_FULL_INS_minus_G_FULL | DEFAULT_uniform | motors | -18.41 | -21.76 | -15.27 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | joints | -22.52 | -26.02 | -19.18 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | digits | -0.09375 | -0.4689 | 0.2734 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_uniform | motors | -18.92 | -21.88 | -15.98 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | joints | -23.77 | -27.7 | -19.84 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | digits | -0.4609 | -0.8281 | -0.07812 | 128 |
| G_FULL_INS_minus_G_FULL | UNION_weighted | motors | -20.26 | -23.6 | -16.98 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | joints | -5.758 | -7.25 | -4.219 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | digits | -0.01562 | -0.5 | 0.4609 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | motors | -4.656 | -5.946 | -3.336 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | joints | -6.18 | -7.649 | -4.75 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | digits | -0.1875 | -0.6016 | 0.2266 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_uniform | motors | -5.117 | -6.352 | -3.866 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | joints | -6.492 | -7.688 | -5.273 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | digits | -0.7188 | -1.094 | -0.3438 | 128 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | motors | -5.5 | -6.484 | -4.484 | 128 |

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
| joints | 12.83 | 10.02 | 15.84 | 128 |
| digits | 1.062 | 0.7422 | 1.398 | 128 |
| motors | 10.65 | 8.219 | 13.16 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 21.98 | 18.62 | 25.46 | 128 |
| digits | 1.656 | 1.328 | 1.984 | 128 |
| motors | 18.48 | 15.54 | 21.54 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8327 |
| add_minimal_digit | 0.826 |
| add_palm_body | 0.7594 |
| delete_phalanx | 0.9606 |
| insert_phalanx | 0.9947 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8875 |
| remove_digit_minimal | 0.6699 |
| remove_palm_body | 0.7428 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.8245 |
| step_limits | 0.999 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7443 |

## dist=G_FULL mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.289 | 6.171 | 10.6 | 128 |
| digits | 0.9609 | 0.6562 | 1.281 | 128 |
| motors | 6.812 | 5.008 | 8.727 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 22.78 | 19.09 | 26.4 | 128 |
| digits | 1.875 | 1.57 | 2.18 | 128 |
| motors | 19.36 | 16.23 | 22.4 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8235 |
| add_minimal_digit | 0.8124 |
| add_palm_body | 0.7239 |
| delete_phalanx | 0.9474 |
| insert_phalanx | 0.9934 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8651 |
| remove_digit_minimal | 0.6898 |
| remove_palm_body | 0.7737 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.7635 |
| step_limits | 0.9995 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7888 |

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
| joints | 5.375 | 4.469 | 6.359 | 128 |
| digits | 1.266 | 0.9922 | 1.547 | 128 |
| motors | 4.695 | 3.891 | 5.563 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.586 | 8.383 | 10.81 | 128 |
| digits | 2.023 | 1.742 | 2.312 | 128 |
| motors | 8.227 | 7.172 | 9.266 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8313 |
| add_minimal_digit | 0.82 |
| add_palm_body | 0.7593 |
| delete_phalanx | 0.9503 |
| insert_phalanx | 0.9798 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.885 |
| remove_digit_minimal | 0.7088 |
| remove_palm_body | 0.7568 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.6719 |
| step_limits | 0.9993 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7645 |

## dist=G_NOBRANCH mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.375 | 3.648 | 5.141 | 128 |
| digits | 0.875 | 0.625 | 1.133 | 128 |
| motors | 3.43 | 2.766 | 4.141 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.633 | 8.539 | 10.79 | 128 |
| digits | 2.242 | 1.977 | 2.523 | 128 |
| motors | 8.258 | 7.328 | 9.203 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8435 |
| add_minimal_digit | 0.8228 |
| add_palm_body | 0.7087 |
| delete_phalanx | 0.9544 |
| insert_phalanx | 0.974 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8653 |
| remove_digit_minimal | 0.7165 |
| remove_palm_body | 0.7704 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.6659 |
| step_limits | 0.9982 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7958 |

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
| joints | -1.031 | -2.234 | 0.07832 | 128 |
| digits | 0.9375 | 0.617 | 1.242 | 128 |
| motors | -0.8828 | -1.93 | 0.1096 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.5469 | -2 | 0.8205 | 128 |
| digits | 1.562 | 1.227 | 1.891 | 128 |
| motors | -0.4375 | -1.727 | 0.75 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8559 |
| add_minimal_digit | 0.8384 |
| add_palm_body | 0.7394 |
| delete_phalanx | 0.8444 |
| insert_phalanx | 1 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8366 |
| remove_digit_minimal | 0.7949 |
| remove_palm_body | 0.7428 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3448 |
| step_limits | 0.9986 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7516 |

## dist=G_FULL_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.8984 | -2.024 | 0.1406 | 128 |
| digits | 0.5391 | 0.2656 | 0.8203 | 128 |
| motors | -0.8203 | -1.812 | 0.1016 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.9844 | -2.461 | 0.3986 | 128 |
| digits | 1.414 | 1.07 | 1.766 | 128 |
| motors | -0.8984 | -2.172 | 0.3281 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8456 |
| add_minimal_digit | 0.8514 |
| add_palm_body | 0.732 |
| delete_phalanx | 0.8472 |
| insert_phalanx | 0.9989 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8159 |
| remove_digit_minimal | 0.772 |
| remove_palm_body | 0.7408 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3218 |
| step_limits | 0.9947 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7359 |

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
| joints | 2.039 | 1.484 | 2.578 | 128 |
| digits | 0.9297 | 0.6406 | 1.219 | 128 |
| motors | 1.797 | 1.304 | 2.305 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.406 | 2.727 | 4.117 | 128 |
| digits | 1.836 | 1.531 | 2.133 | 128 |
| motors | 3.109 | 2.5 | 3.735 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8584 |
| add_minimal_digit | 0.8514 |
| add_palm_body | 0.756 |
| delete_phalanx | 0.8507 |
| insert_phalanx | 0.9993 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8307 |
| remove_digit_minimal | 0.781 |
| remove_palm_body | 0.7456 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3688 |
| step_limits | 0.9972 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7341 |

## dist=G_NOBRANCH_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.414 | 0.898 | 1.945 | 128 |
| digits | 0.7266 | 0.4688 | 0.9844 | 128 |
| motors | 1.258 | 0.7891 | 1.742 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 3.141 | 2.531 | 3.797 | 128 |
| digits | 1.523 | 1.234 | 1.812 | 128 |
| motors | 2.758 | 2.172 | 3.398 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.851 |
| add_minimal_digit | 0.8594 |
| add_palm_body | 0.7216 |
| delete_phalanx | 0.8211 |
| insert_phalanx | 0.9967 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8002 |
| remove_digit_minimal | 0.7668 |
| remove_palm_body | 0.7306 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3429 |
| step_limits | 0.9972 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7391 |

## Reading

(I15 fix 6: every number below tagged "Δ" is a DELTA from that (seed, dist)'s shared reduced START -- not an absolute joint/digit/motor count; start sizes are reported separately above.)

- G_FULL start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL/DEFAULT_uniform step 40: Δjoints 16.01 [12.84, 19.17]; Δdigits 1.039 [0.7029, 1.375]; Δmotors 13.23 [10.55, 15.98]; distinct_hash_fraction 1.
- G_FULL/DEFAULT_uniform step 400: Δjoints 21.47 [17.37, 25.81]; Δdigits 1.477 [1.094, 1.867]; Δmotors 17.96 [14.45, 21.7]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 40: Δjoints 12.83 [10.02, 15.84]; Δdigits 1.062 [0.7422, 1.398]; Δmotors 10.65 [8.219, 13.16]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 400: Δjoints 21.98 [18.62, 25.46]; Δdigits 1.656 [1.328, 1.984]; Δmotors 18.48 [15.54, 21.54]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 40: Δjoints 8.289 [6.171, 10.6]; Δdigits 0.9609 [0.6562, 1.281]; Δmotors 6.812 [5.008, 8.727]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 400: Δjoints 22.78 [19.09, 26.4]; Δdigits 1.875 [1.57, 2.18]; Δmotors 19.36 [16.23, 22.4]; distinct_hash_fraction 1.
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints 1.312 [-3.602, 6.461] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.3984 [-0.03125, 0.7969] (n=128).
- G_FULL DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors 1.398 [-2.805, 5.774] (n=128).
- G_NOBRANCH start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH/DEFAULT_uniform step 40: Δjoints 7.547 [6.391, 8.766]; Δdigits 1.188 [0.8984, 1.484]; Δmotors 6.367 [5.351, 7.453]; distinct_hash_fraction 1.
- G_NOBRANCH/DEFAULT_uniform step 400: Δjoints 9.773 [8.562, 11.02]; Δdigits 1.789 [1.477, 2.109]; Δmotors 8.234 [7.148, 9.297]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 40: Δjoints 5.375 [4.469, 6.359]; Δdigits 1.266 [0.9922, 1.547]; Δmotors 4.695 [3.891, 5.563]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 400: Δjoints 9.586 [8.383, 10.81]; Δdigits 2.023 [1.742, 2.312]; Δmotors 8.227 [7.172, 9.266]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 40: Δjoints 4.375 [3.648, 5.141]; Δdigits 0.875 [0.625, 1.133]; Δmotors 3.43 [2.766, 4.141]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 400: Δjoints 9.633 [8.539, 10.79]; Δdigits 2.242 [1.977, 2.523]; Δmotors 8.258 [7.328, 9.203]; distinct_hash_fraction 1.
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.1406 [-1.649, 1.336] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.4531 [0.0625, 0.8438] (n=128).
- G_NOBRANCH DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors 0.02344 [-1.227, 1.242] (n=128).
- G_FULL_INS start: digit_count mean 1.812, max_phalanx_count mean 1.914.
- G_FULL_INS/DEFAULT_uniform step 40: Δjoints -1.352 [-2.781, -0.04688]; Δdigits 0.8906 [0.5232, 1.25]; Δmotors -1.297 [-2.57, -0.125]; distinct_hash_fraction 1.
- G_FULL_INS/DEFAULT_uniform step 400: Δjoints -0.6016 [-1.961, 0.7893]; Δdigits 1.305 [0.9219, 1.664]; Δmotors -0.4531 [-1.703, 0.7895]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 40: Δjoints -1.031 [-2.234, 0.07832]; Δdigits 0.9375 [0.617, 1.242]; Δmotors -0.8828 [-1.93, 0.1096]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 400: Δjoints -0.5469 [-2, 0.8205]; Δdigits 1.562 [1.227, 1.891]; Δmotors -0.4375 [-1.727, 0.75]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 40: Δjoints -0.8984 [-2.024, 0.1406]; Δdigits 0.5391 [0.2656, 0.8203]; Δmotors -0.8203 [-1.812, 0.1016]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 400: Δjoints -0.9844 [-2.461, 0.3986]; Δdigits 1.414 [1.07, 1.766]; Δmotors -0.8984 [-2.172, 0.3281]; distinct_hash_fraction 1.
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.3828 [-1.461, 0.7502] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits 0.1094 [-0.3672, 0.6172] (n=128).
- G_FULL_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.4453 [-1.43, 0.5938] (n=128).
- G_NOBRANCH_INS start: digit_count mean 1.812, max_phalanx_count mean 1.922.
- G_NOBRANCH_INS/DEFAULT_uniform step 40: Δjoints 3.852 [3.242, 4.453]; Δdigits 1.484 [1.211, 1.75]; Δmotors 3.375 [2.781, 3.946]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/DEFAULT_uniform step 400: Δjoints 4.016 [3.359, 4.719]; Δdigits 1.773 [1.477, 2.086]; Δmotors 3.578 [2.984, 4.188]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 40: Δjoints 2.039 [1.484, 2.578]; Δdigits 0.9297 [0.6406, 1.219]; Δmotors 1.797 [1.304, 2.305]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 400: Δjoints 3.406 [2.727, 4.117]; Δdigits 1.836 [1.531, 2.133]; Δmotors 3.109 [2.5, 3.735]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 40: Δjoints 1.414 [0.898, 1.945]; Δdigits 0.7266 [0.4688, 0.9844]; Δmotors 1.258 [0.7891, 1.742]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 400: Δjoints 3.141 [2.531, 3.797]; Δdigits 1.523 [1.234, 1.812]; Δmotors 2.758 [2.172, 3.398]; distinct_hash_fraction 1.
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δjoints -0.875 [-1.805, 0.07812] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δdigits -0.25 [-0.7031, 0.1719] (n=128).
- G_NOBRANCH_INS DIFFERENCE (UNION_weighted minus DEFAULT_uniform) at step 400: Δmotors -0.8203 [-1.649, 0.05469] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δjoints -22.07 [-25.97, -18.46] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δdigits -0.1719 [-0.6562, 0.2969] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=DEFAULT_uniform) at step 400: Δmotors -18.41 [-21.76, -15.27] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δjoints -22.52 [-26.02, -19.18] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δdigits -0.09375 [-0.4689, 0.2734] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_uniform) at step 400: Δmotors -18.92 [-21.88, -15.98] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δjoints -23.77 [-27.7, -19.84] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δdigits -0.4609 [-0.8281, -0.07812] (n=128).
- DIFFERENCE G_FULL_INS_minus_G_FULL (mixture=UNION_weighted) at step 400: Δmotors -20.26 [-23.6, -16.98] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δjoints -5.758 [-7.25, -4.219] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δdigits -0.01562 [-0.5, 0.4609] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=DEFAULT_uniform) at step 400: Δmotors -4.656 [-5.946, -3.336] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δjoints -6.18 [-7.649, -4.75] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δdigits -0.1875 [-0.6016, 0.2266] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_uniform) at step 400: Δmotors -5.117 [-6.352, -3.866] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δjoints -6.492 [-7.688, -5.273] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δdigits -0.7188 [-1.094, -0.3438] (n=128).
- DIFFERENCE G_NOBRANCH_INS_minus_G_NOBRANCH (mixture=UNION_weighted) at step 400: Δmotors -5.5 [-6.484, -4.484] (n=128).
