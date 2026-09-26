# Experiment: e2_drift

Wall time: 6.713 s

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

## Final joints at step 400, per mixture x dist (paired 95% CI)

| dist | mixture | final joints mean | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|---|
| G_FULL | DEFAULT_uniform | 32.71 | 29.3 | 36.38 | 128 |
| G_FULL | UNION_uniform | 33.16 | 29.72 | 36.68 | 128 |
| G_FULL | UNION_weighted | 32.53 | 29.29 | 35.84 | 128 |
| G_NOBRANCH | DEFAULT_uniform | 14.4 | 13.19 | 15.61 | 128 |
| G_NOBRANCH | UNION_uniform | 14.64 | 13.45 | 15.88 | 128 |
| G_NOBRANCH | UNION_weighted | 15.7 | 14.58 | 16.89 | 128 |
| G_FULL_INS | DEFAULT_uniform | 8.875 | 8.102 | 9.664 | 128 |
| G_FULL_INS | UNION_uniform | 8.805 | 8.125 | 9.531 | 128 |
| G_FULL_INS | UNION_weighted | 9.133 | 8.445 | 9.836 | 128 |
| G_NOBRANCH_INS | DEFAULT_uniform | 8.641 | 7.938 | 9.375 | 128 |
| G_NOBRANCH_INS | UNION_uniform | 9 | 8.383 | 9.656 | 128 |
| G_NOBRANCH_INS | UNION_weighted | 9.141 | 8.469 | 9.836 | 128 |

## dist=G_FULL mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 15.41 | 12.52 | 18.59 | 128 |
| digits | 1.195 | 0.9141 | 1.477 | 128 |
| motors | 12.7 | 10.19 | 15.42 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 23.67 | 20.12 | 27.45 | 128 |
| digits | 1.648 | 1.367 | 1.938 | 128 |
| motors | 19.86 | 16.84 | 23.21 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8497 |
| delete_phalanx | 0.9712 |
| insert_phalanx | 0.987 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8193 |
| resample_parameter | 1 |

## dist=G_FULL mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 15.27 | 12.14 | 18.75 | 128 |
| digits | 1.594 | 1.312 | 1.875 | 128 |
| motors | 12.7 | 10 | 15.57 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 24.12 | 20.66 | 27.64 | 128 |
| digits | 2.211 | 1.961 | 2.469 | 128 |
| motors | 20.64 | 17.56 | 23.73 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.7797 |
| add_minimal_digit | 0.7835 |
| add_palm_body | 0.7658 |
| delete_phalanx | 0.9677 |
| insert_phalanx | 0.9954 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.9021 |
| remove_digit_minimal | 0.5564 |
| remove_palm_body | 0.7434 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.8522 |
| step_limits | 0.9993 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7378 |

## dist=G_FULL mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 7.516 | 5.593 | 9.492 | 128 |
| digits | 1.055 | 0.7969 | 1.32 | 128 |
| motors | 6.258 | 4.562 | 8.047 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 23.49 | 20.15 | 26.88 | 128 |
| digits | 2.562 | 2.312 | 2.828 | 128 |
| motors | 20.09 | 17.29 | 23.07 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8133 |
| add_minimal_digit | 0.8053 |
| add_palm_body | 0.7131 |
| delete_phalanx | 0.9602 |
| insert_phalanx | 0.9952 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.9111 |
| remove_digit_minimal | 0.6111 |
| remove_palm_body | 0.7722 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.7902 |
| step_limits | 0.9996 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7899 |

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
| joints | 6.344 | 5.344 | 7.383 | 128 |
| digits | 1.633 | 1.351 | 1.938 | 128 |
| motors | 5.594 | 4.727 | 6.485 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 10.02 | 8.757 | 11.33 | 128 |
| digits | 2.305 | 2.016 | 2.594 | 128 |
| motors | 8.43 | 7.39 | 9.492 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.7821 |
| add_minimal_digit | 0.7775 |
| add_palm_body | 0.7513 |
| delete_phalanx | 0.9514 |
| insert_phalanx | 0.9773 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8958 |
| remove_digit_minimal | 0.5841 |
| remove_palm_body | 0.7515 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.6806 |
| step_limits | 0.9996 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7559 |

## dist=G_NOBRANCH mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.188 | 3.398 | 5.047 | 128 |
| digits | 0.7266 | 0.4844 | 0.9844 | 128 |
| motors | 3.406 | 2.742 | 4.117 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 11.07 | 9.93 | 12.28 | 128 |
| digits | 2.703 | 2.43 | 2.984 | 128 |
| motors | 9.508 | 8.539 | 10.52 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.7873 |
| add_minimal_digit | 0.7916 |
| add_palm_body | 0.7301 |
| delete_phalanx | 0.9552 |
| insert_phalanx | 0.9804 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8735 |
| remove_digit_minimal | 0.5802 |
| remove_palm_body | 0.7741 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.6595 |
| step_limits | 0.9992 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7845 |

## dist=G_FULL_INS mixture=DEFAULT_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -1.07 | -2.461 | 0.1875 | 128 |
| digits | 1.312 | 1.008 | 1.625 | 128 |
| motors | -1.164 | -2.391 | -0.04668 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.1641 | -1.453 | 1.087 | 128 |
| digits | 1.664 | 1.336 | 1.984 | 128 |
| motors | -0.1719 | -1.305 | 0.9297 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8427 |
| delete_phalanx | 0.8954 |
| insert_phalanx | 0.9983 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8266 |
| resample_parameter | 1 |

## dist=G_FULL_INS mixture=UNION_uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.4375 | -1.672 | 0.7188 | 128 |
| digits | 1.156 | 0.8906 | 1.43 | 128 |
| motors | -0.375 | -1.414 | 0.6484 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.2344 | -1.57 | 1.102 | 128 |
| digits | 1.922 | 1.617 | 2.242 | 128 |
| motors | -0.2656 | -1.469 | 0.9219 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8222 |
| add_minimal_digit | 0.8058 |
| add_palm_body | 0.7389 |
| delete_phalanx | 0.8906 |
| insert_phalanx | 1 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8652 |
| remove_digit_minimal | 0.6995 |
| remove_palm_body | 0.7436 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3936 |
| step_limits | 0.9993 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7409 |

## dist=G_FULL_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | -0.9531 | -2.047 | 0.1094 | 128 |
| digits | 0.8047 | 0.5469 | 1.078 | 128 |
| motors | -0.8203 | -1.773 | 0.1328 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 0.09375 | -1.297 | 1.453 | 128 |
| digits | 1.867 | 1.562 | 2.172 | 128 |
| motors | -0.07031 | -1.281 | 1.102 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.844 |
| add_minimal_digit | 0.8406 |
| add_palm_body | 0.7465 |
| delete_phalanx | 0.8595 |
| insert_phalanx | 0.9989 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8541 |
| remove_digit_minimal | 0.6828 |
| remove_palm_body | 0.7345 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3737 |
| step_limits | 0.9976 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7424 |

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
| joints | 2.406 | 1.828 | 2.992 | 128 |
| digits | 1.078 | 0.7812 | 1.375 | 128 |
| motors | 2.156 | 1.625 | 2.696 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.375 | 3.703 | 5.063 | 128 |
| digits | 1.969 | 1.688 | 2.281 | 128 |
| motors | 3.961 | 3.352 | 4.602 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8273 |
| add_minimal_digit | 0.8215 |
| add_palm_body | 0.7513 |
| delete_phalanx | 0.894 |
| insert_phalanx | 0.999 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8638 |
| remove_digit_minimal | 0.667 |
| remove_palm_body | 0.7347 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3926 |
| step_limits | 1 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7279 |

## dist=G_NOBRANCH_INS mixture=UNION_weighted

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 1.438 | 0.9219 | 1.984 | 128 |
| digits | 0.7344 | 0.492 | 0.9846 | 128 |
| motors | 1.312 | 0.8672 | 1.758 | 128 |

distinct_hash_fraction_at_40: 1

### Step 400

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.516 | 3.828 | 5.227 | 128 |
| digits | 2.062 | 1.766 | 2.352 | 128 |
| motors | 4.117 | 3.477 | 4.781 | 128 |

distinct_hash_fraction_at_400: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8468 |
| add_minimal_digit | 0.8644 |
| add_palm_body | 0.7344 |
| delete_phalanx | 0.8468 |
| insert_phalanx | 1 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8491 |
| remove_digit_minimal | 0.6887 |
| remove_palm_body | 0.7547 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.3444 |
| step_limits | 0.9965 |
| step_mount | 1 |
| step_radius | 1 |
| step_root_length | 1 |
| toggle_palm_joint | 0.7689 |

## Reading

- G_FULL/DEFAULT_uniform step 40: joints 15.41 [12.52, 18.59]; digits 1.195 [0.9141, 1.477]; motors 12.7 [10.19, 15.42]; distinct_hash_fraction 1.
- G_FULL/DEFAULT_uniform step 400: joints 23.67 [20.12, 27.45]; digits 1.648 [1.367, 1.938]; motors 19.86 [16.84, 23.21]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 40: joints 15.27 [12.14, 18.75]; digits 1.594 [1.312, 1.875]; motors 12.7 [10, 15.57]; distinct_hash_fraction 1.
- G_FULL/UNION_uniform step 400: joints 24.12 [20.66, 27.64]; digits 2.211 [1.961, 2.469]; motors 20.64 [17.56, 23.73]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 40: joints 7.516 [5.593, 9.492]; digits 1.055 [0.7969, 1.32]; motors 6.258 [4.562, 8.047]; distinct_hash_fraction 1.
- G_FULL/UNION_weighted step 400: joints 23.49 [20.15, 26.88]; digits 2.562 [2.312, 2.828]; motors 20.09 [17.29, 23.07]; distinct_hash_fraction 1.
- G_NOBRANCH/DEFAULT_uniform step 40: joints 7.547 [6.391, 8.766]; digits 1.188 [0.8984, 1.484]; motors 6.367 [5.351, 7.453]; distinct_hash_fraction 1.
- G_NOBRANCH/DEFAULT_uniform step 400: joints 9.773 [8.562, 11.02]; digits 1.789 [1.477, 2.109]; motors 8.234 [7.148, 9.297]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 40: joints 6.344 [5.344, 7.383]; digits 1.633 [1.351, 1.938]; motors 5.594 [4.727, 6.485]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_uniform step 400: joints 10.02 [8.757, 11.33]; digits 2.305 [2.016, 2.594]; motors 8.43 [7.39, 9.492]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 40: joints 4.188 [3.398, 5.047]; digits 0.7266 [0.4844, 0.9844]; motors 3.406 [2.742, 4.117]; distinct_hash_fraction 1.
- G_NOBRANCH/UNION_weighted step 400: joints 11.07 [9.93, 12.28]; digits 2.703 [2.43, 2.984]; motors 9.508 [8.539, 10.52]; distinct_hash_fraction 1.
- G_FULL_INS/DEFAULT_uniform step 40: joints -1.07 [-2.461, 0.1875]; digits 1.312 [1.008, 1.625]; motors -1.164 [-2.391, -0.04668]; distinct_hash_fraction 1.
- G_FULL_INS/DEFAULT_uniform step 400: joints -0.1641 [-1.453, 1.087]; digits 1.664 [1.336, 1.984]; motors -0.1719 [-1.305, 0.9297]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 40: joints -0.4375 [-1.672, 0.7188]; digits 1.156 [0.8906, 1.43]; motors -0.375 [-1.414, 0.6484]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_uniform step 400: joints -0.2344 [-1.57, 1.102]; digits 1.922 [1.617, 2.242]; motors -0.2656 [-1.469, 0.9219]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 40: joints -0.9531 [-2.047, 0.1094]; digits 0.8047 [0.5469, 1.078]; motors -0.8203 [-1.773, 0.1328]; distinct_hash_fraction 1.
- G_FULL_INS/UNION_weighted step 400: joints 0.09375 [-1.297, 1.453]; digits 1.867 [1.562, 2.172]; motors -0.07031 [-1.281, 1.102]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/DEFAULT_uniform step 40: joints 3.852 [3.242, 4.453]; digits 1.484 [1.211, 1.75]; motors 3.375 [2.781, 3.946]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/DEFAULT_uniform step 400: joints 4.016 [3.359, 4.719]; digits 1.773 [1.477, 2.086]; motors 3.578 [2.984, 4.188]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 40: joints 2.406 [1.828, 2.992]; digits 1.078 [0.7812, 1.375]; motors 2.156 [1.625, 2.696]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_uniform step 400: joints 4.375 [3.703, 5.063]; digits 1.969 [1.688, 2.281]; motors 3.961 [3.352, 4.602]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 40: joints 1.438 [0.9219, 1.984]; digits 0.7344 [0.492, 0.9846]; motors 1.312 [0.8672, 1.758]; distinct_hash_fraction 1.
- G_NOBRANCH_INS/UNION_weighted step 400: joints 4.516 [3.828, 5.227]; digits 2.062 [1.766, 2.352]; motors 4.117 [3.477, 4.781]; distinct_hash_fraction 1.
