# Experiment: e2_drift

Wall time: 3.365 s

## Params

```json
{
  "record_every_after": 10,
  "record_through": 40,
  "walk_length": 500
}
```

n_seeds: 256

## Mixture: uniform

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 7.125 | 6.336 | 7.945 | 256 |
| digits | 1.102 | 0.8984 | 1.305 | 256 |
| motors | 5.891 | 5.176 | 6.598 | 256 |

distinct_hash_fraction_at_40: 1

### Step 500

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.281 | 8.461 | 10.07 | 256 |
| digits | 1.762 | 1.547 | 1.969 | 256 |
| motors | 7.852 | 7.109 | 8.566 | 256 |

distinct_hash_fraction_at_500: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8453 |
| delete_phalanx | 0.9643 |
| insert_phalanx | 0.9602 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.821 |
| resample_parameter | 1 |

## Mixture: balanced

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.699 | 7.898 | 9.531 | 256 |
| digits | 1.395 | 1.191 | 1.598 | 256 |
| motors | 7.379 | 6.664 | 8.117 | 256 |

distinct_hash_fraction_at_40: 1

### Step 500

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 9.98 | 9.137 | 10.85 | 256 |
| digits | 1.75 | 1.535 | 1.957 | 256 |
| motors | 8.379 | 7.637 | 9.141 | 256 |

distinct_hash_fraction_at_500: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8508 |
| delete_phalanx | 0.9586 |
| insert_phalanx | 0.9615 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.8163 |
| resample_parameter | 1 |

## Mixture: small_heavy

### Step 40

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 4.398 | 3.836 | 5.008 | 256 |
| digits | 0.4219 | 0.2734 | 0.582 | 256 |
| motors | 3.594 | 3.098 | 4.121 | 256 |

distinct_hash_fraction_at_40: 1

### Step 500

| metric | mean delta | 95% CI lo | 95% CI hi | n |
|---|---|---|---|---|
| joints | 8.496 | 7.652 | 9.34 | 256 |
| digits | 1.48 | 1.273 | 1.691 | 256 |
| motors | 7.148 | 6.426 | 7.875 | 256 |

distinct_hash_fraction_at_500: 1

### Acceptance rate per operator

| operator | acceptance_rate |
|---|---|
| add_digit | 0.8797 |
| delete_phalanx | 0.9516 |
| insert_phalanx | 0.9602 |
| perturb_parameter | 1 |
| regrow_subtree | 1 |
| remove_digit | 0.7932 |
| resample_parameter | 1 |
| step_axis | 1 |
| step_coupling | 0.6568 |
| step_length | 1 |
| step_limits | 0.9942 |
| step_mount | 1 |

## Reading

- uniform step 40: joints 7.125 [6.336, 7.945]; digits 1.102 [0.8984, 1.305]; motors 5.891 [5.176, 6.598]; distinct_hash_fraction 1.
- uniform step 500: joints 9.281 [8.461, 10.07]; digits 1.762 [1.547, 1.969]; motors 7.852 [7.109, 8.566]; distinct_hash_fraction 1.
- balanced step 40: joints 8.699 [7.898, 9.531]; digits 1.395 [1.191, 1.598]; motors 7.379 [6.664, 8.117]; distinct_hash_fraction 1.
- balanced step 500: joints 9.98 [9.137, 10.85]; digits 1.75 [1.535, 1.957]; motors 8.379 [7.637, 9.141]; distinct_hash_fraction 1.
- small_heavy step 40: joints 4.398 [3.836, 5.008]; digits 0.4219 [0.2734, 0.582]; motors 3.594 [3.098, 4.121]; distinct_hash_fraction 1.
- small_heavy step 500: joints 8.496 [7.652, 9.34]; digits 1.48 [1.273, 1.691]; motors 7.148 [6.426, 7.875]; distinct_hash_fraction 1.
