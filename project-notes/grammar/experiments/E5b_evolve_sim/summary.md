# Experiment: e5b_evolve_sim

Wall time: 185.049 s

**Note:** Full run: 24 restarts x 54 conditions, mu=16 lam=16 generations=40 (defaults). Timing probe (2 restarts x 54 conditions = 18.2s wall) showed the full run was feasible at 24 restarts without cutting to 16; no restart reduction was needed.

## Params

```json
{
  "envelope": {
    "allow_branches": false,
    "allow_palm_joints": false,
    "max_digits": 5,
    "max_joints_per_digit": 6
  },
  "generations": 40,
  "lam": 16,
  "mu": 16,
  "n_conditions": 54,
  "n_proxy_configs": 16,
  "n_restarts": 24
}
```

n_conditions: 54

## Per-condition (mean over restarts, 95% CI)

| dist | pool | fitness | cost | n | final best raw proxy | 95% CI | final motors | final joints | final digits | target reach rate | gen first reach (median) | clone rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.8932 | [0.797, 0.9621] | 6.19 | 6.674 | 4.096 | 0 | n/a | 0 |
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.6791 | [0.5116, 0.8431] | 2.917 | 3.055 | 2.443 | 0 | n/a | 6.352e-05 |
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | 0.02 | 24 | 0.6588 | [0.4916, 0.8118] | 2.51 | 2.648 | 2.133 | 0 | n/a | 0 |
| G_FULL_INS | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 15.27 | 17.33 | 5 | 1 | 15.5 | 0 |
| G_FULL_INS | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 14.83 | 16.96 | 5 | 1 | 16 | 0 |
| G_FULL_INS | DEFAULT_uniform | target_distance | 0.02 | 24 | 0 | [0, 0] | 14.03 | 16.36 | 5 | 1 | 15 | 0 |
| G_FULL_INS | UNION_weighted_nopalm | antipodal_pinch | 0 | 24 | 0.769 | [0.6084, 0.9007] | 5.234 | 5.617 | 3.612 | 0 | n/a | 0 |
| G_FULL_INS | UNION_weighted_nopalm | antipodal_pinch | 0.01 | 24 | 0.7701 | [0.6071, 0.9023] | 2.354 | 2.411 | 2.182 | 0 | n/a | 0 |
| G_FULL_INS | UNION_weighted_nopalm | antipodal_pinch | 0.02 | 24 | 0.7431 | [0.5751, 0.8946] | 2.247 | 2.333 | 1.958 | 0 | n/a | 0 |
| G_FULL_INS | UNION_weighted_nopalm | target_distance | 0 | 24 | -0.25 | [-0.4167, -0.08333] | 13.15 | 14.65 | 4.846 | 0.75 | 28.5 | 0 |
| G_FULL_INS | UNION_weighted_nopalm | target_distance | 0.01 | 24 | -0.2917 | [-0.4583, -0.125] | 12.87 | 14.72 | 4.938 | 0.7083 | 28 | 0 |
| G_FULL_INS | UNION_weighted_nopalm | target_distance | 0.02 | 24 | -0.375 | [-0.625, -0.1667] | 12.14 | 14.08 | 4.846 | 0.6667 | 27 | 0 |
| G_FULL_INS | UNION_weighted | antipodal_pinch | 0 | 24 | 0.8994 | [0.8008, 0.9728] | 5.073 | 5.24 | 3.901 | 0 | n/a | 0.000127 |
| G_FULL_INS | UNION_weighted | antipodal_pinch | 0.01 | 24 | 0.6198 | [0.4285, 0.7979] | 2.344 | 2.367 | 2.036 | 0 | n/a | 0 |
| G_FULL_INS | UNION_weighted | antipodal_pinch | 0.02 | 24 | 0.5844 | [0.4, 0.7575] | 1.919 | 2.055 | 1.784 | 0 | n/a | 0 |
| G_FULL_INS | UNION_weighted | target_distance | 0 | 24 | -0.2917 | [-0.4583, -0.125] | 12.64 | 14.49 | 4.878 | 0.7083 | 31 | 0 |
| G_FULL_INS | UNION_weighted | target_distance | 0.01 | 24 | -0.3333 | [-0.5833, -0.125] | 13.16 | 14.58 | 4.875 | 0.7083 | 30 | 0 |
| G_FULL_INS | UNION_weighted | target_distance | 0.02 | 24 | -0.25 | [-0.4583, -0.08333] | 12.65 | 14.53 | 4.805 | 0.7917 | 28 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.793 | [0.6595, 0.912] | 5.982 | 6.323 | 3.956 | 0 | n/a | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.7248 | [0.5582, 0.8751] | 3.388 | 3.594 | 2.737 | 0 | n/a | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | 0.02 | 24 | 0.7389 | [0.5769, 0.884] | 2.638 | 2.703 | 2.234 | 0 | n/a | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 15.81 | 17.77 | 5 | 1 | 14.5 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 14.58 | 16.92 | 5 | 1 | 13.5 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | target_distance | 0.02 | 24 | 0 | [0, 0] | 13.63 | 16.47 | 5 | 1 | 15 | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | antipodal_pinch | 0 | 24 | 0.8647 | [0.7431, 0.9601] | 4.982 | 5.099 | 3.617 | 0 | n/a | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | antipodal_pinch | 0.01 | 24 | 0.655 | [0.4514, 0.8212] | 2.518 | 2.544 | 2.115 | 0 | n/a | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | antipodal_pinch | 0.02 | 24 | 0.8025 | [0.6491, 0.9313] | 2.396 | 2.417 | 2.008 | 0 | n/a | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | target_distance | 0 | 24 | -0.2917 | [-0.4583, -0.125] | 12.66 | 14.6 | 4.87 | 0.7083 | 27 | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | target_distance | 0.01 | 24 | -0.25 | [-0.4167, -0.08333] | 12.41 | 14.48 | 4.888 | 0.75 | 27 | 0 |
| G_NOBRANCH_INS | UNION_weighted_nopalm | target_distance | 0.02 | 24 | -0.3333 | [-0.5833, -0.125] | 12.99 | 14.49 | 4.885 | 0.7083 | 31 | 0 |
| G_NOBRANCH_INS | UNION_weighted | antipodal_pinch | 0 | 24 | 0.8667 | [0.7491, 0.9576] | 4.682 | 4.818 | 3.643 | 0 | n/a | 0 |
| G_NOBRANCH_INS | UNION_weighted | antipodal_pinch | 0.01 | 24 | 0.7692 | [0.6172, 0.8997] | 2.523 | 2.529 | 2.268 | 0 | n/a | 0 |
| G_NOBRANCH_INS | UNION_weighted | antipodal_pinch | 0.02 | 24 | 0.6055 | [0.4115, 0.7803] | 2.039 | 2.039 | 1.865 | 0 | n/a | 6.352e-05 |
| G_NOBRANCH_INS | UNION_weighted | target_distance | 0 | 24 | -0.3333 | [-0.5417, -0.1667] | 13.1 | 14.67 | 4.898 | 0.6667 | 26 | 0 |
| G_NOBRANCH_INS | UNION_weighted | target_distance | 0.01 | 24 | -0.1667 | [-0.3333, -0.04167] | 12.46 | 14.61 | 4.891 | 0.8333 | 33.5 | 0 |
| G_NOBRANCH_INS | UNION_weighted | target_distance | 0.02 | 24 | -0.4583 | [-0.7083, -0.25] | 11.68 | 13.93 | 4.719 | 0.5833 | 30 | 0 |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.7019 | [0.5443, 0.8498] | 8.755 | 9.885 | 3.784 | 0 | n/a | 0 |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.43 | [0.2546, 0.5999] | 2.378 | 2.427 | 1.773 | 0 | n/a | 6.352e-05 |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0.02 | 24 | 0.6346 | [0.4525, 0.7924] | 3.073 | 3.245 | 2.273 | 0 | n/a | 0 |
| G_NOBRANCH | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 19.73 | 22.88 | 5 | 1 | 10 | 0 |
| G_NOBRANCH | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 17.27 | 20.76 | 4.997 | 1 | 10.5 | 0 |
| G_NOBRANCH | DEFAULT_uniform | target_distance | 0.02 | 24 | 0 | [0, 0] | 15.94 | 19.4 | 5 | 1 | 11.5 | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0 | 24 | 0.8202 | [0.6829, 0.9376] | 6.378 | 6.802 | 3.755 | 0 | n/a | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0.01 | 24 | 0.6975 | [0.5252, 0.8614] | 2.536 | 2.568 | 2.188 | 0 | n/a | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0.02 | 24 | 0.7293 | [0.5569, 0.8916] | 2.177 | 2.263 | 2.039 | 0 | n/a | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0 | 24 | 0 | [0, 0] | 18.26 | 21.98 | 4.995 | 1 | 24 | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0.01 | 24 | -0.04167 | [-0.125, 0] | 17.4 | 20.14 | 4.966 | 0.9583 | 20 | 0 |
| G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0.02 | 24 | 0 | [0, 0] | 16.83 | 20.31 | 5 | 1 | 23.5 | 0 |
| G_NOBRANCH | UNION_weighted | antipodal_pinch | 0 | 24 | 0.7069 | [0.5329, 0.8664] | 5.911 | 6.562 | 3.518 | 0 | n/a | 0 |
| G_NOBRANCH | UNION_weighted | antipodal_pinch | 0.01 | 24 | 0.533 | [0.3306, 0.7386] | 1.953 | 2.016 | 1.727 | 0 | n/a | 0 |
| G_NOBRANCH | UNION_weighted | antipodal_pinch | 0.02 | 24 | 0.7682 | [0.6012, 0.9151] | 2.193 | 2.195 | 2.008 | 0 | n/a | 6.352e-05 |
| G_NOBRANCH | UNION_weighted | target_distance | 0 | 24 | 0 | [0, 0] | 17.8 | 21.45 | 4.997 | 1 | 21.5 | 0 |
| G_NOBRANCH | UNION_weighted | target_distance | 0.01 | 24 | -0.08333 | [-0.25, 0] | 16.8 | 19.88 | 4.943 | 0.9583 | 24 | 0 |
| G_NOBRANCH | UNION_weighted | target_distance | 0.02 | 24 | 0 | [0, 0] | 16.4 | 19.81 | 4.984 | 1 | 22 | 0 |

## Difference CI: an _INS dist minus its plain counterpart, per pool x fitness x cost

| pair | pool | fitness | cost | mean DIFFERENCE (final best raw proxy) | 95% CI | n |
|---|---|---|---|---|---|---|
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0 | 0.09117 | [-0.1109, 0.2848] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0.01 | 0.2948 | [0.04433, 0.5216] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0.02 | 0.1043 | [-0.1309, 0.349] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | target_distance | 0 | 0 | [0, 0] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | target_distance | 0.01 | 0 | [0, 0] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | DEFAULT_uniform | target_distance | 0.02 | 0 | [0, 0] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | antipodal_pinch | 0 | 0.1598 | [-0.02518, 0.3462] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | antipodal_pinch | 0.01 | 0.2362 | [-0.03866, 0.4855] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | antipodal_pinch | 0.02 | -0.1627 | [-0.4027, 0.0782] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | target_distance | 0 | -0.3333 | [-0.5417, -0.1667] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | target_distance | 0.01 | -0.08333 | [-0.2917, 0.1667] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted | target_distance | 0.02 | -0.4583 | [-0.7083, -0.25] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0 | 0.04446 | [-0.1059, 0.1946] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0.01 | -0.04255 | [-0.3267, 0.2199] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | antipodal_pinch | 0.02 | 0.07316 | [-0.1395, 0.2831] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0 | -0.2917 | [-0.4583, -0.125] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0.01 | -0.2083 | [-0.4167, 0] | 24 |
| G_NOBRANCH_INS_minus_G_NOBRANCH | UNION_weighted_nopalm | target_distance | 0.02 | -0.3333 | [-0.5833, -0.125] | 24 |

## Difference CI: UNION_weighted minus DEFAULT_uniform, per dist x fitness x cost

| dist | fitness | cost | mean DIFFERENCE (final best raw proxy) | 95% CI | n |
|---|---|---|---|---|---|
| G_FULL_INS | antipodal_pinch | 0 | 0.00615 | [-0.1306, 0.1378] | 24 |
| G_FULL_INS | antipodal_pinch | 0.01 | -0.05927 | [-0.3242, 0.1979] | 24 |
| G_FULL_INS | antipodal_pinch | 0.02 | -0.0744 | [-0.2646, 0.1046] | 24 |
| G_FULL_INS | target_distance | 0 | -0.2917 | [-0.4583, -0.125] | 24 |
| G_FULL_INS | target_distance | 0.01 | -0.3333 | [-0.5833, -0.125] | 24 |
| G_FULL_INS | target_distance | 0.02 | -0.25 | [-0.4583, -0.08333] | 24 |
| G_NOBRANCH_INS | antipodal_pinch | 0 | 0.0737 | [-0.0996, 0.2422] | 24 |
| G_NOBRANCH_INS | antipodal_pinch | 0.01 | 0.04447 | [-0.1636, 0.2689] | 24 |
| G_NOBRANCH_INS | antipodal_pinch | 0.02 | -0.1333 | [-0.3809, 0.1177] | 24 |
| G_NOBRANCH_INS | target_distance | 0 | -0.3333 | [-0.5417, -0.1667] | 24 |
| G_NOBRANCH_INS | target_distance | 0.01 | -0.1667 | [-0.3333, -0.04167] | 24 |
| G_NOBRANCH_INS | target_distance | 0.02 | -0.4583 | [-0.7083, -0.25] | 24 |
| G_NOBRANCH | antipodal_pinch | 0 | 0.005019 | [-0.204, 0.2077] | 24 |
| G_NOBRANCH | antipodal_pinch | 0.01 | 0.103 | [-0.1065, 0.3103] | 24 |
| G_NOBRANCH | antipodal_pinch | 0.02 | 0.1336 | [-0.1164, 0.377] | 24 |
| G_NOBRANCH | target_distance | 0 | 0 | [0, 0] | 24 |
| G_NOBRANCH | target_distance | 0.01 | -0.08333 | [-0.25, 0] | 24 |
| G_NOBRANCH | target_distance | 0.02 | 0 | [0, 0] | 24 |

## Rejection / improvement rate per operator, per condition

### G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2349 | 0.2461 | 0 | 0.7539 | 0.1754 |
| delete_phalanx | 2433 | 0 | 0.1804 | 0.8196 | 0.1446 |
| insert_phalanx | 2491 | 0 | 0 | 1 | 0.09248 |
| perturb_parameter | 2445 | 0 | 0 | 1 | 0.09026 |
| regrow_subtree | 2504 | 0 | 0 | 1 | 0.08046 |
| remove_digit | 2457 | 0 | 0.1119 | 0.8881 | 0.1026 |
| resample_parameter | 2357 | 0 | 0 | 1 | 0.03955 |

### G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2845 | 0.0239 | 0 | 0.9761 | 0.1053 |
| delete_phalanx | 2722 | 0 | 0.7355 | 0.2645 | 0.1084 |
| insert_phalanx | 2791 | 0 | 0 | 1 | 0.03909 |
| perturb_parameter | 2738 | 0 | 0 | 1 | 0.08265 |
| regrow_subtree | 2664 | 0 | 0 | 1 | 0.02069 |
| remove_digit | 2762 | 0 | 0.5514 | 0.4486 | 0.07293 |
| resample_parameter | 2814 | 0 | 0 | 1 | 0.02649 |

### G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2744 | 0.005102 | 0 | 0.9949 | 0.1357 |
| delete_phalanx | 2712 | 0 | 0.7341 | 0.2659 | 0.1109 |
| insert_phalanx | 2761 | 0 | 0 | 1 | 0.04131 |
| perturb_parameter | 2791 | 0 | 0 | 1 | 0.09144 |
| regrow_subtree | 2761 | 0 | 0 | 1 | 0.02284 |
| remove_digit | 2825 | 0 | 0.5409 | 0.4591 | 0.06988 |
| resample_parameter | 2683 | 0 | 0 | 1 | 0.01972 |

### G_FULL_INS|DEFAULT_uniform|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2516 | 0.7071 | 0 | 0.2929 | 0.4808 |
| delete_phalanx | 2440 | 0 | 0.01148 | 0.9885 | 0 |
| insert_phalanx | 2552 | 0 | 0 | 1 | 0.07877 |
| perturb_parameter | 2536 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2468 | 0 | 0 | 1 | 0.05502 |
| remove_digit | 2559 | 0 | 0.01915 | 0.9809 | 0 |
| resample_parameter | 2529 | 0 | 0 | 1 | 0.002828 |

### G_FULL_INS|DEFAULT_uniform|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2614 | 0.6959 | 0 | 0.3041 | 0.4478 |
| delete_phalanx | 2446 | 0 | 0.01554 | 0.9845 | 0 |
| insert_phalanx | 2557 | 0 | 0 | 1 | 0.08494 |
| perturb_parameter | 2486 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2490 | 0 | 0 | 1 | 0.04949 |
| remove_digit | 2609 | 0 | 0.01265 | 0.9874 | 0 |
| resample_parameter | 2432 | 0 | 0 | 1 | 0.004639 |

### G_FULL_INS|DEFAULT_uniform|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2558 | 0.6943 | 0 | 0.3057 | 0.4323 |
| delete_phalanx | 2562 | 0 | 0.01444 | 0.9856 | 0 |
| insert_phalanx | 2433 | 0 | 0 | 1 | 0.07056 |
| perturb_parameter | 2459 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2474 | 0 | 0 | 1 | 0.05627 |
| remove_digit | 2555 | 0 | 0.01135 | 0.9886 | 0 |
| resample_parameter | 2545 | 0 | 0 | 1 | 0.003616 |

### G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 651 | 0.07527 | 0 | 0.9247 | 0.125 |
| add_minimal_digit | 705 | 0.07376 | 0 | 0.9262 | 0.1315 |
| delete_phalanx | 734 | 0 | 0.2888 | 0.7112 | 0.05088 |
| insert_phalanx | 704 | 0 | 0 | 1 | 0.06069 |
| perturb_parameter | 1584 | 0 | 0 | 1 | 0.09459 |
| regrow_subtree | 648 | 0 | 0 | 1 | 0.04286 |
| remove_digit | 643 | 0 | 0.1477 | 0.8523 | 0.06716 |
| remove_digit_minimal | 736 | 0 | 0.1522 | 0.8478 | 0.04793 |
| resample_parameter | 1945 | 0 | 0 | 1 | 0.03217 |
| step_axis | 1535 | 0 | 0 | 1 | 0.1071 |
| step_coupling | 1583 | 0 | 0.844 | 0.156 | 0.05394 |
| step_limits | 1507 | 0 | 0.01062 | 0.9894 | 0.104 |
| step_mount | 1545 | 0 | 0 | 1 | 0.0845 |
| step_radius | 1557 | 0 | 0 | 1 | 0 |
| step_root_length | 1539 | 0 | 0 | 1 | 0.1162 |

### G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 723 | 0.0166 | 0 | 0.9834 | 0.1196 |
| add_minimal_digit | 769 | 0.006502 | 0 | 0.9935 | 0.1519 |
| delete_phalanx | 727 | 0 | 0.7235 | 0.2765 | 0.05978 |
| insert_phalanx | 771 | 0 | 0 | 1 | 0.04095 |
| perturb_parameter | 1598 | 0 | 0 | 1 | 0.1242 |
| regrow_subtree | 731 | 0 | 0 | 1 | 0.01541 |
| remove_digit | 671 | 0 | 0.4694 | 0.5306 | 0.03892 |
| remove_digit_minimal | 707 | 0 | 0.4413 | 0.5587 | 0.04145 |
| resample_parameter | 1990 | 0 | 0 | 1 | 0.02899 |
| step_axis | 1632 | 0 | 0 | 1 | 0.1151 |
| step_coupling | 1619 | 0 | 0.9086 | 0.09141 | 0.1049 |
| step_limits | 1609 | 0 | 0.02362 | 0.9764 | 0.1373 |
| step_mount | 1608 | 0 | 0 | 1 | 0.1092 |
| step_radius | 1586 | 0 | 0 | 1 | 0 |
| step_root_length | 1682 | 0 | 0 | 1 | 0.1491 |

### G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 667 | 0 | 0 | 1 | 0.1002 |
| add_minimal_digit | 763 | 0 | 0 | 1 | 0.1095 |
| delete_phalanx | 788 | 0 | 0.7195 | 0.2805 | 0.04348 |
| insert_phalanx | 781 | 0 | 0 | 1 | 0.01695 |
| perturb_parameter | 1596 | 0 | 0 | 1 | 0.1192 |
| regrow_subtree | 747 | 0 | 0 | 1 | 0.01639 |
| remove_digit | 668 | 0 | 0.4985 | 0.5015 | 0.01553 |
| remove_digit_minimal | 767 | 0 | 0.4915 | 0.5085 | 0.01323 |
| resample_parameter | 2022 | 0 | 0 | 1 | 0.03355 |
| step_axis | 1603 | 0 | 0 | 1 | 0.1536 |
| step_coupling | 1744 | 0 | 0.9576 | 0.04243 | 0.1389 |
| step_limits | 1686 | 0 | 0.03321 | 0.9668 | 0.1411 |
| step_mount | 1602 | 0 | 0 | 1 | 0.1039 |
| step_radius | 1668 | 0 | 0 | 1 | 0 |
| step_root_length | 1645 | 0 | 0 | 1 | 0.1453 |

### G_FULL_INS|UNION_weighted_nopalm|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 632 | 0.3956 | 0 | 0.6044 | 0.394 |
| add_minimal_digit | 667 | 0.4153 | 0 | 0.5847 | 0.1551 |
| delete_phalanx | 686 | 0 | 0.0102 | 0.9898 | 0 |
| insert_phalanx | 647 | 0 | 0 | 1 | 0.1543 |
| perturb_parameter | 1554 | 0 | 0 | 1 | 0 |
| regrow_subtree | 656 | 0 | 0 | 1 | 0.08411 |
| remove_digit | 691 | 0 | 0.01592 | 0.9841 | 0 |
| remove_digit_minimal | 685 | 0 | 0.3197 | 0.6803 | 0 |
| resample_parameter | 1917 | 0 | 0 | 1 | 0.004839 |
| step_axis | 1503 | 0 | 0 | 1 | 0 |
| step_coupling | 1430 | 0 | 0.3979 | 0.6021 | 0 |
| step_limits | 1524 | 0 | 0.001312 | 0.9987 | 0 |
| step_mount | 1543 | 0 | 0 | 1 | 0.005305 |
| step_radius | 1479 | 0 | 0 | 1 | 0 |
| step_root_length | 1465 | 0 | 0 | 1 | 0 |

### G_FULL_INS|UNION_weighted_nopalm|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 668 | 0.3772 | 0 | 0.6228 | 0.3995 |
| add_minimal_digit | 732 | 0.418 | 0 | 0.582 | 0.1211 |
| delete_phalanx | 692 | 0 | 0.01156 | 0.9884 | 0 |
| insert_phalanx | 725 | 0 | 0 | 1 | 0.1207 |
| perturb_parameter | 1434 | 0 | 0 | 1 | 0 |
| regrow_subtree | 631 | 0 | 0 | 1 | 0.08838 |
| remove_digit | 670 | 0 | 0.00597 | 0.994 | 0 |
| remove_digit_minimal | 656 | 0 | 0.3155 | 0.6845 | 0 |
| resample_parameter | 1996 | 0 | 0 | 1 | 0.004107 |
| step_axis | 1467 | 0 | 0 | 1 | 0 |
| step_coupling | 1450 | 0 | 0.4303 | 0.5697 | 0 |
| step_limits | 1454 | 0 | 0 | 1 | 0 |
| step_mount | 1519 | 0 | 0 | 1 | 0.008802 |
| step_radius | 1531 | 0 | 0 | 1 | 0 |
| step_root_length | 1520 | 0 | 0 | 1 | 0 |

### G_FULL_INS|UNION_weighted_nopalm|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 644 | 0.4348 | 0 | 0.5652 | 0.4563 |
| add_minimal_digit | 707 | 0.4455 | 0 | 0.5545 | 0.1349 |
| delete_phalanx | 695 | 0 | 0.01583 | 0.9842 | 0 |
| insert_phalanx | 689 | 0 | 0 | 1 | 0.1252 |
| perturb_parameter | 1530 | 0 | 0 | 1 | 0 |
| regrow_subtree | 577 | 0 | 0 | 1 | 0.1011 |
| remove_digit | 653 | 0 | 0.02297 | 0.977 | 0 |
| remove_digit_minimal | 690 | 0 | 0.2725 | 0.7275 | 0 |
| resample_parameter | 1929 | 0 | 0 | 1 | 0.002667 |
| step_axis | 1512 | 0 | 0 | 1 | 0 |
| step_coupling | 1414 | 0 | 0.4378 | 0.5622 | 0 |
| step_limits | 1493 | 0 | 0.0006698 | 0.9993 | 0 |
| step_mount | 1508 | 0 | 0 | 1 | 0.006106 |
| step_radius | 1589 | 0 | 0 | 1 | 0 |
| step_root_length | 1543 | 0 | 0 | 1 | 0 |

### G_FULL_INS|UNION_weighted|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 676 | 0.1169 | 0 | 0.8831 | 0.1504 |
| add_minimal_digit | 692 | 0.1301 | 0 | 0.8699 | 0.1769 |
| add_palm_body | 733 | 1 | 0 | 0 | n/a |
| delete_phalanx | 726 | 0 | 0.292 | 0.708 | 0.09128 |
| insert_phalanx | 701 | 0 | 0 | 1 | 0.06881 |
| perturb_parameter | 1540 | 0 | 0 | 1 | 0.1176 |
| regrow_subtree | 690 | 0 | 0 | 1 | 0.0295 |
| remove_digit | 681 | 0 | 0.1439 | 0.8561 | 0.0613 |
| remove_digit_minimal | 662 | 0 | 0.1133 | 0.8867 | 0.06401 |
| remove_palm_body | 709 | 0 | 1 | 0 | n/a |
| resample_parameter | 1978 | 0 | 0 | 1 | 0.03123 |
| step_axis | 1542 | 0 | 0 | 1 | 0.1139 |
| step_coupling | 1636 | 0 | 0.8894 | 0.1106 | 0.06704 |
| step_limits | 1587 | 0 | 0.006301 | 0.9937 | 0.141 |
| step_mount | 1519 | 0 | 0 | 1 | 0.1016 |
| step_radius | 1597 | 0 | 0 | 1 | 0 |
| step_root_length | 1534 | 0 | 0 | 1 | 0.1698 |
| toggle_palm_joint | 716 | 0 | 1 | 0 | n/a |

### G_FULL_INS|UNION_weighted|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 668 | 0.02994 | 0 | 0.9701 | 0.1139 |
| add_minimal_digit | 709 | 0.05078 | 0 | 0.9492 | 0.1394 |
| add_palm_body | 743 | 1 | 0 | 0 | n/a |
| delete_phalanx | 728 | 0 | 0.7088 | 0.2912 | 0.06316 |
| insert_phalanx | 753 | 0 | 0 | 1 | 0.05034 |
| perturb_parameter | 1614 | 0 | 0 | 1 | 0.1067 |
| regrow_subtree | 689 | 0 | 0 | 1 | 0.03566 |
| remove_digit | 732 | 0 | 0.5068 | 0.4932 | 0.06725 |
| remove_digit_minimal | 743 | 0 | 0.5087 | 0.4913 | 0.06232 |
| remove_palm_body | 772 | 0 | 1 | 0 | n/a |
| resample_parameter | 2138 | 0 | 0 | 1 | 0.02912 |
| step_axis | 1653 | 0 | 0 | 1 | 0.1071 |
| step_coupling | 1664 | 0 | 0.9718 | 0.02825 | 0 |
| step_limits | 1682 | 0 | 0.06302 | 0.937 | 0.1012 |
| step_mount | 1727 | 0 | 0 | 1 | 0.09614 |
| step_radius | 1687 | 0 | 0 | 1 | 0 |
| step_root_length | 1601 | 0 | 0 | 1 | 0.1349 |
| toggle_palm_joint | 706 | 0 | 1 | 0 | n/a |

### G_FULL_INS|UNION_weighted|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 674 | 0.001484 | 0 | 0.9985 | 0.0782 |
| add_minimal_digit | 721 | 0.001387 | 0 | 0.9986 | 0.1034 |
| add_palm_body | 719 | 1 | 0 | 0 | n/a |
| delete_phalanx | 743 | 0 | 0.7026 | 0.2974 | 0.06829 |
| insert_phalanx | 765 | 0 | 0 | 1 | 0.02394 |
| perturb_parameter | 1627 | 0 | 0 | 1 | 0.08485 |
| regrow_subtree | 706 | 0 | 0 | 1 | 0.01587 |
| remove_digit | 668 | 0 | 0.5793 | 0.4207 | 0.04151 |
| remove_digit_minimal | 745 | 0 | 0.5852 | 0.4148 | 0.04498 |
| remove_palm_body | 783 | 0 | 1 | 0 | n/a |
| resample_parameter | 2193 | 0 | 0 | 1 | 0.01915 |
| step_axis | 1601 | 0 | 0 | 1 | 0.1089 |
| step_coupling | 1717 | 0 | 0.9056 | 0.09435 | 0.141 |
| step_limits | 1672 | 0 | 0.04605 | 0.9539 | 0.1062 |
| step_mount | 1646 | 0 | 0 | 1 | 0.08163 |
| step_radius | 1654 | 0 | 0 | 1 | 0 |
| step_root_length | 1591 | 0 | 0 | 1 | 0.146 |
| toggle_palm_joint | 757 | 0 | 1 | 0 | n/a |

### G_FULL_INS|UNION_weighted|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 647 | 0.4498 | 0 | 0.5502 | 0.4256 |
| add_minimal_digit | 642 | 0.4408 | 0 | 0.5592 | 0.137 |
| add_palm_body | 702 | 1 | 0 | 0 | n/a |
| delete_phalanx | 641 | 0 | 0.0234 | 0.9766 | 0 |
| insert_phalanx | 679 | 0 | 0 | 1 | 0.1174 |
| perturb_parameter | 1434 | 0 | 0 | 1 | 0 |
| regrow_subtree | 681 | 0 | 0 | 1 | 0.09531 |
| remove_digit | 635 | 0 | 0.02205 | 0.978 | 0 |
| remove_digit_minimal | 702 | 0 | 0.198 | 0.802 | 0 |
| remove_palm_body | 697 | 0 | 1 | 0 | n/a |
| resample_parameter | 1936 | 0 | 0 | 1 | 0.003168 |
| step_axis | 1487 | 0 | 0 | 1 | 0 |
| step_coupling | 1473 | 0 | 0.3435 | 0.6565 | 0 |
| step_limits | 1514 | 0 | 0 | 1 | 0 |
| step_mount | 1537 | 0 | 0 | 1 | 0.002679 |
| step_radius | 1435 | 0 | 0 | 1 | 0 |
| step_root_length | 1549 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 695 | 0 | 1 | 0 | n/a |

### G_FULL_INS|UNION_weighted|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 648 | 0.3827 | 0 | 0.6173 | 0.4052 |
| add_minimal_digit | 718 | 0.3858 | 0 | 0.6142 | 0.1395 |
| add_palm_body | 703 | 1 | 0 | 0 | n/a |
| delete_phalanx | 682 | 0 | 0.008798 | 0.9912 | 0 |
| insert_phalanx | 670 | 0 | 0 | 1 | 0.1227 |
| perturb_parameter | 1517 | 0 | 0 | 1 | 0 |
| regrow_subtree | 676 | 0 | 0 | 1 | 0.1027 |
| remove_digit | 628 | 0 | 0.004777 | 0.9952 | 0 |
| remove_digit_minimal | 700 | 0 | 0.3043 | 0.6957 | 0 |
| remove_palm_body | 642 | 0 | 1 | 0 | n/a |
| resample_parameter | 1911 | 0 | 0 | 1 | 0.002693 |
| step_axis | 1541 | 0 | 0 | 1 | 0 |
| step_coupling | 1531 | 0 | 0.4847 | 0.5153 | 0 |
| step_limits | 1502 | 0 | 0 | 1 | 0 |
| step_mount | 1526 | 0 | 0 | 1 | 0.00473 |
| step_radius | 1483 | 0 | 0 | 1 | 0 |
| step_root_length | 1500 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 634 | 0 | 1 | 0 | n/a |

### G_FULL_INS|UNION_weighted|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 626 | 0.2939 | 0 | 0.7061 | 0.4215 |
| add_minimal_digit | 682 | 0.3006 | 0 | 0.6994 | 0.07576 |
| add_palm_body | 637 | 1 | 0 | 0 | n/a |
| delete_phalanx | 674 | 0 | 0.02077 | 0.9792 | 0 |
| insert_phalanx | 665 | 0 | 0 | 1 | 0.1068 |
| perturb_parameter | 1487 | 0 | 0 | 1 | 0 |
| regrow_subtree | 647 | 0 | 0 | 1 | 0.0938 |
| remove_digit | 622 | 0 | 0.009646 | 0.9904 | 0 |
| remove_digit_minimal | 737 | 0 | 0.3745 | 0.6255 | 0 |
| remove_palm_body | 668 | 0 | 1 | 0 | n/a |
| resample_parameter | 1953 | 0 | 0 | 1 | 0.002621 |
| step_axis | 1497 | 0 | 0 | 1 | 0 |
| step_coupling | 1547 | 0 | 0.4428 | 0.5572 | 0 |
| step_limits | 1456 | 0 | 0 | 1 | 0 |
| step_mount | 1497 | 0 | 0 | 1 | 0.003415 |
| step_radius | 1504 | 0 | 0 | 1 | 0 |
| step_root_length | 1520 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 685 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2473 | 0.1662 | 0 | 0.8338 | 0.1453 |
| delete_phalanx | 2441 | 0 | 0.2216 | 0.7784 | 0.1483 |
| insert_phalanx | 2377 | 0 | 0 | 1 | 0.07875 |
| perturb_parameter | 2509 | 0 | 0 | 1 | 0.08167 |
| regrow_subtree | 2477 | 0 | 0 | 1 | 0.06485 |
| remove_digit | 2437 | 0 | 0.1379 | 0.8621 | 0.0942 |
| resample_parameter | 2318 | 0 | 0 | 1 | 0.03276 |

### G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2665 | 0.05629 | 0 | 0.9437 | 0.1353 |
| delete_phalanx | 2714 | 0 | 0.6183 | 0.3817 | 0.1267 |
| insert_phalanx | 2710 | 0 | 0 | 1 | 0.06471 |
| perturb_parameter | 2654 | 0 | 0 | 1 | 0.09527 |
| regrow_subtree | 2704 | 0 | 0 | 1 | 0.05051 |
| remove_digit | 2666 | 0 | 0.4681 | 0.5319 | 0.09586 |
| resample_parameter | 2707 | 0 | 0 | 1 | 0.02761 |

### G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2808 | 0.008903 | 0 | 0.9911 | 0.119 |
| delete_phalanx | 2850 | 0 | 0.6979 | 0.3021 | 0.08696 |
| insert_phalanx | 2649 | 0 | 0 | 1 | 0.04196 |
| perturb_parameter | 2747 | 0 | 0 | 1 | 0.07859 |
| regrow_subtree | 2723 | 0 | 0 | 1 | 0.01981 |
| remove_digit | 2777 | 0 | 0.5823 | 0.4177 | 0.06469 |
| resample_parameter | 2821 | 0 | 0 | 1 | 0.02379 |

### G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2471 | 0.692 | 0 | 0.308 | 0.4633 |
| delete_phalanx | 2541 | 0 | 0.02007 | 0.9799 | 0 |
| insert_phalanx | 2545 | 0 | 0 | 1 | 0.08521 |
| perturb_parameter | 2511 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2447 | 0 | 0 | 1 | 0.05307 |
| remove_digit | 2535 | 0 | 0.01696 | 0.983 | 0 |
| resample_parameter | 2498 | 0 | 0 | 1 | 0.003698 |

### G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2535 | 0.6785 | 0 | 0.3215 | 0.4427 |
| delete_phalanx | 2476 | 0 | 0.01575 | 0.9842 | 0 |
| insert_phalanx | 2470 | 0 | 0 | 1 | 0.07495 |
| perturb_parameter | 2552 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2520 | 0 | 0 | 1 | 0.04925 |
| remove_digit | 2428 | 0 | 0.01936 | 0.9806 | 0 |
| resample_parameter | 2569 | 0 | 0 | 1 | 0.006803 |

### G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2513 | 0.6518 | 0 | 0.3482 | 0.4335 |
| delete_phalanx | 2483 | 0 | 0.01571 | 0.9843 | 0 |
| insert_phalanx | 2464 | 0 | 0 | 1 | 0.06664 |
| perturb_parameter | 2430 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2499 | 0 | 0 | 1 | 0.04506 |
| remove_digit | 2541 | 0 | 0.01732 | 0.9827 | 0 |
| resample_parameter | 2535 | 0 | 0 | 1 | 0.003638 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 634 | 0.09937 | 0 | 0.9006 | 0.09515 |
| add_minimal_digit | 730 | 0.1068 | 0 | 0.8932 | 0.1058 |
| delete_phalanx | 686 | 0 | 0.2172 | 0.7828 | 0.09298 |
| insert_phalanx | 701 | 0 | 0 | 1 | 0.04342 |
| perturb_parameter | 1608 | 0 | 0 | 1 | 0.09229 |
| regrow_subtree | 641 | 0 | 0 | 1 | 0.04114 |
| remove_digit | 641 | 0 | 0.1638 | 0.8362 | 0.04175 |
| remove_digit_minimal | 713 | 0 | 0.1304 | 0.8696 | 0.03993 |
| resample_parameter | 1949 | 0 | 0 | 1 | 0.02788 |
| step_axis | 1589 | 0 | 0 | 1 | 0.1183 |
| step_coupling | 1628 | 0 | 0.8704 | 0.1296 | 0.02927 |
| step_limits | 1562 | 0 | 0.007682 | 0.9923 | 0.08667 |
| step_mount | 1555 | 0 | 0 | 1 | 0.08785 |
| step_radius | 1514 | 0 | 0 | 1 | 0 |
| step_root_length | 1510 | 0 | 0 | 1 | 0.1374 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 676 | 0.0429 | 0 | 0.9571 | 0.1005 |
| add_minimal_digit | 739 | 0.01353 | 0 | 0.9865 | 0.1169 |
| delete_phalanx | 708 | 0 | 0.7161 | 0.2839 | 0.09677 |
| insert_phalanx | 775 | 0 | 0 | 1 | 0.03831 |
| perturb_parameter | 1650 | 0 | 0 | 1 | 0.09592 |
| regrow_subtree | 699 | 0 | 0 | 1 | 0.02346 |
| remove_digit | 700 | 0 | 0.5329 | 0.4671 | 0.05178 |
| remove_digit_minimal | 733 | 0 | 0.5307 | 0.4693 | 0.05438 |
| resample_parameter | 2077 | 0 | 0 | 1 | 0.0275 |
| step_axis | 1663 | 0 | 0 | 1 | 0.1046 |
| step_coupling | 1633 | 0 | 0.9596 | 0.04042 | 0 |
| step_limits | 1650 | 0 | 0.04909 | 0.9509 | 0.0944 |
| step_mount | 1731 | 0 | 0 | 1 | 0.08462 |
| step_radius | 1593 | 0 | 0 | 1 | 0.001288 |
| step_root_length | 1673 | 0 | 0 | 1 | 0.1448 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 678 | 0.001475 | 0 | 0.9985 | 0.1337 |
| add_minimal_digit | 732 | 0 | 0 | 1 | 0.1664 |
| delete_phalanx | 768 | 0 | 0.7227 | 0.2773 | 0.05759 |
| insert_phalanx | 785 | 0 | 0 | 1 | 0.03906 |
| perturb_parameter | 1540 | 0 | 0 | 1 | 0.154 |
| regrow_subtree | 711 | 0 | 0 | 1 | 0.02432 |
| remove_digit | 712 | 0 | 0.4719 | 0.5281 | 0.04087 |
| remove_digit_minimal | 703 | 0 | 0.4267 | 0.5733 | 0.02577 |
| resample_parameter | 2018 | 0 | 0 | 1 | 0.03188 |
| step_axis | 1604 | 0 | 0 | 1 | 0.1387 |
| step_coupling | 1606 | 0 | 0.9788 | 0.02117 | 0.03226 |
| step_limits | 1732 | 0 | 0.01963 | 0.9804 | 0.1522 |
| step_mount | 1646 | 0 | 0 | 1 | 0.1276 |
| step_radius | 1617 | 0 | 0 | 1 | 0 |
| step_root_length | 1690 | 0 | 0 | 1 | 0.1609 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 658 | 0.4225 | 0 | 0.5775 | 0.3946 |
| add_minimal_digit | 693 | 0.3911 | 0 | 0.6089 | 0.1086 |
| delete_phalanx | 672 | 0 | 0.00744 | 0.9926 | 0 |
| insert_phalanx | 690 | 0 | 0 | 1 | 0.1604 |
| perturb_parameter | 1486 | 0 | 0 | 1 | 0 |
| regrow_subtree | 656 | 0 | 0 | 1 | 0.1121 |
| remove_digit | 601 | 0 | 0.01331 | 0.9867 | 0 |
| remove_digit_minimal | 678 | 0 | 0.3038 | 0.6962 | 0 |
| resample_parameter | 1934 | 0 | 0 | 1 | 0.003696 |
| step_axis | 1524 | 0 | 0 | 1 | 0 |
| step_coupling | 1465 | 0 | 0.4369 | 0.5631 | 0 |
| step_limits | 1557 | 0 | 0 | 1 | 0 |
| step_mount | 1509 | 0 | 0 | 1 | 0.002035 |
| step_radius | 1493 | 0 | 0 | 1 | 0 |
| step_root_length | 1536 | 0 | 0 | 1 | 0 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 642 | 0.4813 | 0 | 0.5187 | 0.4413 |
| add_minimal_digit | 681 | 0.5037 | 0 | 0.4963 | 0.1166 |
| delete_phalanx | 686 | 0 | 0.01603 | 0.984 | 0 |
| insert_phalanx | 661 | 0 | 0 | 1 | 0.1537 |
| perturb_parameter | 1561 | 0 | 0 | 1 | 0 |
| regrow_subtree | 605 | 0 | 0 | 1 | 0.09612 |
| remove_digit | 616 | 0 | 0.01461 | 0.9854 | 0 |
| remove_digit_minimal | 694 | 0 | 0.2997 | 0.7003 | 0 |
| resample_parameter | 1938 | 0 | 0 | 1 | 0.003188 |
| step_axis | 1552 | 0 | 0 | 1 | 0 |
| step_coupling | 1444 | 0 | 0.3144 | 0.6856 | 0 |
| step_limits | 1520 | 0 | 0.0006579 | 0.9993 | 0 |
| step_mount | 1494 | 0 | 0 | 1 | 0.006849 |
| step_radius | 1493 | 0 | 0 | 1 | 0 |
| step_root_length | 1492 | 0 | 0 | 1 | 0 |

### G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 563 | 0.3819 | 0 | 0.6181 | 0.4324 |
| add_minimal_digit | 656 | 0.3933 | 0 | 0.6067 | 0.125 |
| delete_phalanx | 711 | 0 | 0.02813 | 0.9719 | 0 |
| insert_phalanx | 672 | 0 | 0 | 1 | 0.09816 |
| perturb_parameter | 1525 | 0 | 0 | 1 | 0 |
| regrow_subtree | 668 | 0 | 0 | 1 | 0.1046 |
| remove_digit | 662 | 0 | 0.0136 | 0.9864 | 0 |
| remove_digit_minimal | 653 | 0 | 0.2848 | 0.7152 | 0 |
| resample_parameter | 1880 | 0 | 0 | 1 | 0.004376 |
| step_axis | 1522 | 0 | 0 | 1 | 0 |
| step_coupling | 1548 | 0 | 0.4612 | 0.5388 | 0 |
| step_limits | 1483 | 0 | 0 | 1 | 0 |
| step_mount | 1544 | 0 | 0 | 1 | 0.003296 |
| step_radius | 1515 | 0 | 0 | 1 | 0 |
| step_root_length | 1544 | 0 | 0 | 1 | 0 |

### G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 656 | 0.09146 | 0 | 0.9085 | 0.1667 |
| add_minimal_digit | 701 | 0.097 | 0 | 0.903 | 0.2026 |
| add_palm_body | 714 | 1 | 0 | 0 | n/a |
| delete_phalanx | 711 | 0 | 0.3249 | 0.6751 | 0.1512 |
| insert_phalanx | 722 | 0 | 0 | 1 | 0.06206 |
| perturb_parameter | 1570 | 0 | 0 | 1 | 0.1125 |
| regrow_subtree | 673 | 0 | 0 | 1 | 0.05784 |
| remove_digit | 637 | 0 | 0.1774 | 0.8226 | 0.08268 |
| remove_digit_minimal | 723 | 0 | 0.1895 | 0.8105 | 0.06087 |
| remove_palm_body | 744 | 0 | 1 | 0 | n/a |
| resample_parameter | 2021 | 0 | 0 | 1 | 0.03965 |
| step_axis | 1497 | 0 | 0 | 1 | 0.1467 |
| step_coupling | 1555 | 0 | 0.9093 | 0.09068 | 0 |
| step_limits | 1543 | 0 | 0.007129 | 0.9929 | 0.1235 |
| step_mount | 1636 | 0 | 0 | 1 | 0.1053 |
| step_radius | 1537 | 0 | 0 | 1 | 0.0006671 |
| step_root_length | 1596 | 0 | 0 | 1 | 0.2041 |
| toggle_palm_joint | 737 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 738 | 0.06369 | 0 | 0.9363 | 0.1165 |
| add_minimal_digit | 781 | 0.04994 | 0 | 0.9501 | 0.1221 |
| add_palm_body | 737 | 1 | 0 | 0 | n/a |
| delete_phalanx | 767 | 0 | 0.7536 | 0.2464 | 0.1047 |
| insert_phalanx | 736 | 0 | 0 | 1 | 0.04318 |
| perturb_parameter | 1612 | 0 | 0 | 1 | 0.1301 |
| regrow_subtree | 690 | 0 | 0 | 1 | 0.0462 |
| remove_digit | 679 | 0 | 0.4492 | 0.5508 | 0.03562 |
| remove_digit_minimal | 728 | 0 | 0.4904 | 0.5096 | 0.03056 |
| remove_palm_body | 760 | 0 | 1 | 0 | n/a |
| resample_parameter | 2101 | 0 | 0 | 1 | 0.03268 |
| step_axis | 1606 | 0 | 0 | 1 | 0.1227 |
| step_coupling | 1665 | 0 | 0.9724 | 0.02763 | 0.04762 |
| step_limits | 1681 | 0 | 0.02558 | 0.9744 | 0.1276 |
| step_mount | 1645 | 0 | 0 | 1 | 0.09119 |
| step_radius | 1664 | 0 | 0 | 1 | 0 |
| step_root_length | 1639 | 0 | 0 | 1 | 0.1284 |
| toggle_palm_joint | 762 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 671 | 0.00149 | 0 | 0.9985 | 0.103 |
| add_minimal_digit | 757 | 0.001321 | 0 | 0.9987 | 0.1386 |
| add_palm_body | 769 | 1 | 0 | 0 | n/a |
| delete_phalanx | 707 | 0 | 0.7454 | 0.2546 | 0.1296 |
| insert_phalanx | 750 | 0 | 0 | 1 | 0.02881 |
| perturb_parameter | 1614 | 0 | 0 | 1 | 0.09905 |
| regrow_subtree | 690 | 0 | 0 | 1 | 0.01477 |
| remove_digit | 655 | 0 | 0.5282 | 0.4718 | 0.04746 |
| remove_digit_minimal | 766 | 0 | 0.4883 | 0.5117 | 0.03684 |
| remove_palm_body | 730 | 0 | 1 | 0 | n/a |
| resample_parameter | 2147 | 0 | 0 | 1 | 0.02388 |
| step_axis | 1613 | 0 | 0 | 1 | 0.1203 |
| step_coupling | 1631 | 0 | 0.9577 | 0.04231 | 0 |
| step_limits | 1650 | 0 | 0.01212 | 0.9879 | 0.1451 |
| step_mount | 1635 | 0 | 0 | 1 | 0.1031 |
| step_radius | 1666 | 0 | 0 | 1 | 0 |
| step_root_length | 1622 | 0 | 0 | 1 | 0.1098 |
| toggle_palm_joint | 717 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|UNION_weighted|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 668 | 0.476 | 0 | 0.524 | 0.4217 |
| add_minimal_digit | 686 | 0.4548 | 0 | 0.5452 | 0.1408 |
| add_palm_body | 679 | 1 | 0 | 0 | n/a |
| delete_phalanx | 639 | 0 | 0.01252 | 0.9875 | 0 |
| insert_phalanx | 736 | 0 | 0 | 1 | 0.1181 |
| perturb_parameter | 1483 | 0 | 0 | 1 | 0 |
| regrow_subtree | 612 | 0 | 0 | 1 | 0.1044 |
| remove_digit | 615 | 0 | 0.01301 | 0.987 | 0 |
| remove_digit_minimal | 706 | 0 | 0.2975 | 0.7025 | 0 |
| remove_palm_body | 698 | 0 | 1 | 0 | n/a |
| resample_parameter | 1933 | 0 | 0 | 1 | 0.00426 |
| step_axis | 1507 | 0 | 0 | 1 | 0 |
| step_coupling | 1495 | 0 | 0.3793 | 0.6207 | 0 |
| step_limits | 1510 | 0 | 0 | 1 | 0 |
| step_mount | 1535 | 0 | 0 | 1 | 0.007343 |
| step_radius | 1578 | 0 | 0 | 1 | 0 |
| step_root_length | 1464 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 691 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|UNION_weighted|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 633 | 0.3823 | 0 | 0.6177 | 0.4194 |
| add_minimal_digit | 706 | 0.4731 | 0 | 0.5269 | 0.1671 |
| add_palm_body | 653 | 1 | 0 | 0 | n/a |
| delete_phalanx | 688 | 0 | 0.008721 | 0.9913 | 0 |
| insert_phalanx | 646 | 0 | 0 | 1 | 0.1097 |
| perturb_parameter | 1520 | 0 | 0 | 1 | 0 |
| regrow_subtree | 654 | 0 | 0 | 1 | 0.0989 |
| remove_digit | 628 | 0 | 0.0207 | 0.9793 | 0 |
| remove_digit_minimal | 656 | 0 | 0.2622 | 0.7378 | 0 |
| remove_palm_body | 673 | 0 | 1 | 0 | n/a |
| resample_parameter | 1926 | 0 | 0 | 1 | 0.001599 |
| step_axis | 1503 | 0 | 0 | 1 | 0 |
| step_coupling | 1478 | 0 | 0.27 | 0.73 | 0 |
| step_limits | 1480 | 0 | 0 | 1 | 0 |
| step_mount | 1452 | 0 | 0 | 1 | 0.004261 |
| step_radius | 1391 | 0 | 0 | 1 | 0 |
| step_root_length | 1549 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 691 | 0 | 1 | 0 | n/a |

### G_NOBRANCH_INS|UNION_weighted|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 620 | 0.329 | 0 | 0.671 | 0.394 |
| add_minimal_digit | 665 | 0.3414 | 0 | 0.6586 | 0.1129 |
| add_palm_body | 716 | 1 | 0 | 0 | n/a |
| delete_phalanx | 662 | 0 | 0.0136 | 0.9864 | 0 |
| insert_phalanx | 667 | 0 | 0 | 1 | 0.1111 |
| perturb_parameter | 1438 | 0 | 0 | 1 | 0 |
| regrow_subtree | 637 | 0 | 0 | 1 | 0.094 |
| remove_digit | 615 | 0 | 0.01301 | 0.987 | 0 |
| remove_digit_minimal | 698 | 0 | 0.2951 | 0.7049 | 0 |
| remove_palm_body | 686 | 0 | 1 | 0 | n/a |
| resample_parameter | 1972 | 0 | 0 | 1 | 0.003107 |
| step_axis | 1451 | 0 | 0 | 1 | 0 |
| step_coupling | 1469 | 0 | 0.3717 | 0.6283 | 0 |
| step_limits | 1537 | 0 | 0 | 1 | 0 |
| step_mount | 1529 | 0 | 0 | 1 | 0.005358 |
| step_radius | 1507 | 0 | 0 | 1 | 0 |
| step_root_length | 1477 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 654 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2329 | 0.1456 | 0 | 0.8544 | 0.09917 |
| delete_phalanx | 2476 | 0 | 0.1393 | 0.8607 | 0.08068 |
| insert_phalanx | 2462 | 0 | 0.02681 | 0.9732 | 0.04615 |
| perturb_parameter | 2454 | 0 | 0 | 1 | 0.05511 |
| regrow_subtree | 2387 | 0 | 0 | 1 | 0.0379 |
| remove_digit | 2481 | 0 | 0.1955 | 0.8045 | 0.05339 |
| resample_parameter | 2390 | 0 | 0 | 1 | 0.02341 |

### G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2878 | 0.01425 | 0 | 0.9858 | 0.07353 |
| delete_phalanx | 2905 | 0 | 0.7532 | 0.2468 | 0.1158 |
| insert_phalanx | 2891 | 0 | 0 | 1 | 0.0226 |
| perturb_parameter | 2844 | 0 | 0 | 1 | 0.05148 |
| regrow_subtree | 2838 | 0 | 0 | 1 | 0.01691 |
| remove_digit | 2853 | 0 | 0.7171 | 0.2829 | 0.06992 |
| resample_parameter | 2809 | 0 | 0 | 1 | 0.0116 |

### G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2930 | 0.01468 | 0 | 0.9853 | 0.08389 |
| delete_phalanx | 2879 | 0 | 0.7912 | 0.2088 | 0.1338 |
| insert_phalanx | 2884 | 0 | 0 | 1 | 0.03751 |
| perturb_parameter | 2847 | 0 | 0 | 1 | 0.06509 |
| regrow_subtree | 2833 | 0 | 0 | 1 | 0.01951 |
| remove_digit | 2825 | 0 | 0.6853 | 0.3147 | 0.08974 |
| resample_parameter | 2803 | 0 | 0 | 1 | 0.02656 |

### G_NOBRANCH|DEFAULT_uniform|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2604 | 0.7554 | 0 | 0.2446 | 0.7339 |
| delete_phalanx | 2635 | 0 | 0.009488 | 0.9905 | 0 |
| insert_phalanx | 2495 | 0 | 0.002004 | 0.998 | 0.06025 |
| perturb_parameter | 2464 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2575 | 0 | 0 | 1 | 0.06458 |
| remove_digit | 2463 | 0 | 0.01137 | 0.9886 | 0 |
| resample_parameter | 2533 | 0 | 0 | 1 | 0.002021 |

### G_NOBRANCH|DEFAULT_uniform|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2518 | 0.7538 | 0 | 0.2462 | 0.7433 |
| delete_phalanx | 2583 | 0 | 0.01007 | 0.9899 | 0 |
| insert_phalanx | 2540 | 0 | 0.0003937 | 0.9996 | 0.05189 |
| perturb_parameter | 2501 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2477 | 0 | 0 | 1 | 0.06772 |
| remove_digit | 2545 | 0 | 0.01022 | 0.9898 | 0 |
| resample_parameter | 2531 | 0 | 0 | 1 | 0.003234 |

### G_NOBRANCH|DEFAULT_uniform|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 2463 | 0.7426 | 0 | 0.2574 | 0.7504 |
| delete_phalanx | 2543 | 0 | 0.006685 | 0.9933 | 0 |
| insert_phalanx | 2524 | 0 | 0 | 1 | 0.0624 |
| perturb_parameter | 2505 | 0 | 0 | 1 | 0 |
| regrow_subtree | 2530 | 0 | 0 | 1 | 0.06442 |
| remove_digit | 2531 | 0 | 0.01383 | 0.9862 | 0 |
| resample_parameter | 2529 | 0 | 0 | 1 | 0.003645 |

### G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 681 | 0.1366 | 0 | 0.8634 | 0.1553 |
| add_minimal_digit | 723 | 0.1452 | 0 | 0.8548 | 0.2083 |
| delete_phalanx | 672 | 0 | 0.186 | 0.814 | 0.1065 |
| insert_phalanx | 701 | 0 | 0.00428 | 0.9957 | 0.07259 |
| perturb_parameter | 1541 | 0 | 0 | 1 | 0.08289 |
| regrow_subtree | 731 | 0 | 0 | 1 | 0.06056 |
| remove_digit | 662 | 0 | 0.1858 | 0.8142 | 0.0797 |
| remove_digit_minimal | 686 | 0 | 0.1676 | 0.8324 | 0.06989 |
| resample_parameter | 1926 | 0 | 0 | 1 | 0.03625 |
| step_axis | 1569 | 0 | 0 | 1 | 0.1101 |
| step_coupling | 1544 | 0 | 0.7468 | 0.2532 | 0.04393 |
| step_limits | 1533 | 0 | 0.001305 | 0.9987 | 0.09152 |
| step_mount | 1502 | 0 | 0 | 1 | 0.09271 |
| step_radius | 1483 | 0 | 0 | 1 | 0 |
| step_root_length | 1509 | 0 | 0 | 1 | 0.1616 |

### G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 714 | 0.004202 | 0 | 0.9958 | 0.07801 |
| add_minimal_digit | 714 | 0.004202 | 0 | 0.9958 | 0.1243 |
| delete_phalanx | 762 | 0 | 0.6365 | 0.3635 | 0.05364 |
| insert_phalanx | 766 | 0 | 0 | 1 | 0.02785 |
| perturb_parameter | 1597 | 0 | 0 | 1 | 0.1065 |
| regrow_subtree | 717 | 0 | 0 | 1 | 0.01991 |
| remove_digit | 722 | 0 | 0.4958 | 0.5042 | 0.02609 |
| remove_digit_minimal | 750 | 0 | 0.4893 | 0.5107 | 0.03014 |
| resample_parameter | 2101 | 0 | 0 | 1 | 0.02338 |
| step_axis | 1669 | 0 | 0 | 1 | 0.136 |
| step_coupling | 1646 | 0 | 0.9526 | 0.04739 | 0.1429 |
| step_limits | 1626 | 0 | 0.0492 | 0.9508 | 0.112 |
| step_mount | 1604 | 0 | 0 | 1 | 0.09484 |
| step_radius | 1579 | 0 | 0 | 1 | 0 |
| step_root_length | 1641 | 0 | 0 | 1 | 0.1412 |

### G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 663 | 0.003017 | 0 | 0.997 | 0.1578 |
| add_minimal_digit | 736 | 0.004076 | 0 | 0.9959 | 0.1643 |
| delete_phalanx | 776 | 0 | 0.7062 | 0.2938 | 0.122 |
| insert_phalanx | 711 | 0 | 0 | 1 | 0.04422 |
| perturb_parameter | 1595 | 0 | 0 | 1 | 0.1336 |
| regrow_subtree | 669 | 0 | 0 | 1 | 0.02003 |
| remove_digit | 688 | 0 | 0.4549 | 0.5451 | 0.04945 |
| remove_digit_minimal | 720 | 0 | 0.4542 | 0.5458 | 0.03235 |
| resample_parameter | 2055 | 0 | 0 | 1 | 0.02234 |
| step_axis | 1642 | 0 | 0 | 1 | 0.1421 |
| step_coupling | 1584 | 0 | 0.9009 | 0.09912 | 0.1226 |
| step_limits | 1690 | 0 | 0.03136 | 0.9686 | 0.111 |
| step_mount | 1644 | 0 | 0 | 1 | 0.1083 |
| step_radius | 1651 | 0 | 0 | 1 | 0.0006188 |
| step_root_length | 1593 | 0 | 0 | 1 | 0.1675 |

### G_NOBRANCH|UNION_weighted_nopalm|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 606 | 0.5495 | 0 | 0.4505 | 0.6705 |
| add_minimal_digit | 691 | 0.5384 | 0 | 0.4616 | 0.1815 |
| delete_phalanx | 643 | 0 | 0.01089 | 0.9891 | 0 |
| insert_phalanx | 662 | 0 | 0 | 1 | 0.1168 |
| perturb_parameter | 1535 | 0 | 0 | 1 | 0 |
| regrow_subtree | 613 | 0 | 0 | 1 | 0.15 |
| remove_digit | 679 | 0 | 0.007364 | 0.9926 | 0 |
| remove_digit_minimal | 729 | 0 | 0.439 | 0.561 | 0 |
| resample_parameter | 1904 | 0 | 0 | 1 | 0.003774 |
| step_axis | 1506 | 0 | 0 | 1 | 0 |
| step_coupling | 1474 | 0 | 0.2341 | 0.7659 | 0 |
| step_limits | 1517 | 0 | 0.0006592 | 0.9993 | 0 |
| step_mount | 1533 | 0 | 0 | 1 | 0.006729 |
| step_radius | 1501 | 0 | 0 | 1 | 0 |
| step_root_length | 1534 | 0 | 0 | 1 | 0 |

### G_NOBRANCH|UNION_weighted_nopalm|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 666 | 0.5766 | 0 | 0.4234 | 0.7303 |
| add_minimal_digit | 683 | 0.5871 | 0 | 0.4129 | 0.1679 |
| delete_phalanx | 703 | 0 | 0.01138 | 0.9886 | 0 |
| insert_phalanx | 727 | 0 | 0 | 1 | 0.1179 |
| perturb_parameter | 1501 | 0 | 0 | 1 | 0 |
| regrow_subtree | 606 | 0 | 0 | 1 | 0.1302 |
| remove_digit | 622 | 0 | 0.02251 | 0.9775 | 0 |
| remove_digit_minimal | 689 | 0 | 0.4862 | 0.5138 | 0 |
| resample_parameter | 1939 | 0 | 0 | 1 | 0.005823 |
| step_axis | 1533 | 0 | 0 | 1 | 0 |
| step_coupling | 1517 | 0 | 0.2307 | 0.7693 | 0 |
| step_limits | 1522 | 0 | 0.000657 | 0.9993 | 0 |
| step_mount | 1567 | 0 | 0 | 1 | 0.00915 |
| step_radius | 1493 | 0 | 0 | 1 | 0 |
| step_root_length | 1469 | 0 | 0 | 1 | 0 |

### G_NOBRANCH|UNION_weighted_nopalm|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 600 | 0.5317 | 0 | 0.4683 | 0.7395 |
| add_minimal_digit | 705 | 0.5461 | 0 | 0.4539 | 0.1447 |
| delete_phalanx | 703 | 0 | 0.00569 | 0.9943 | 0 |
| insert_phalanx | 680 | 0 | 0.001471 | 0.9985 | 0.1038 |
| perturb_parameter | 1516 | 0 | 0 | 1 | 0 |
| regrow_subtree | 591 | 0 | 0 | 1 | 0.1406 |
| remove_digit | 642 | 0 | 0.01558 | 0.9844 | 0 |
| remove_digit_minimal | 694 | 0 | 0.464 | 0.536 | 0 |
| resample_parameter | 1882 | 0 | 0 | 1 | 0.004379 |
| step_axis | 1561 | 0 | 0 | 1 | 0 |
| step_coupling | 1521 | 0 | 0.2222 | 0.7778 | 0 |
| step_limits | 1462 | 0 | 0 | 1 | 0 |
| step_mount | 1501 | 0 | 0 | 1 | 0.00813 |
| step_radius | 1537 | 0 | 0 | 1 | 0 |
| step_root_length | 1528 | 0 | 0 | 1 | 0 |

### G_NOBRANCH|UNION_weighted|antipodal_pinch|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 612 | 0.09314 | 0 | 0.9069 | 0.08889 |
| add_minimal_digit | 750 | 0.07733 | 0 | 0.9227 | 0.1317 |
| add_palm_body | 665 | 1 | 0 | 0 | n/a |
| delete_phalanx | 644 | 0 | 0.1196 | 0.8804 | 0.06691 |
| insert_phalanx | 705 | 0 | 0.002837 | 0.9972 | 0.0355 |
| perturb_parameter | 1563 | 0 | 0 | 1 | 0.08568 |
| regrow_subtree | 679 | 0 | 0 | 1 | 0.02703 |
| remove_digit | 651 | 0 | 0.1705 | 0.8295 | 0.0452 |
| remove_digit_minimal | 650 | 0 | 0.1785 | 0.8215 | 0.0249 |
| remove_palm_body | 669 | 0 | 1 | 0 | n/a |
| resample_parameter | 1922 | 0 | 0 | 1 | 0.03137 |
| step_axis | 1500 | 0 | 0 | 1 | 0.07882 |
| step_coupling | 1472 | 0 | 0.6841 | 0.3159 | 0.008621 |
| step_limits | 1541 | 0 | 0.004543 | 0.9955 | 0.08871 |
| step_mount | 1498 | 0 | 0 | 1 | 0.08735 |
| step_radius | 1493 | 0 | 0 | 1 | 0 |
| step_root_length | 1499 | 0 | 0 | 1 | 0.1391 |
| toggle_palm_joint | 663 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|UNION_weighted|antipodal_pinch|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 707 | 0.005658 | 0 | 0.9943 | 0.07613 |
| add_minimal_digit | 700 | 0.001429 | 0 | 0.9986 | 0.1025 |
| add_palm_body | 761 | 1 | 0 | 0 | n/a |
| delete_phalanx | 762 | 0 | 0.7441 | 0.2559 | 0.06395 |
| insert_phalanx | 741 | 0 | 0 | 1 | 0.02897 |
| perturb_parameter | 1619 | 0 | 0 | 1 | 0.1011 |
| regrow_subtree | 668 | 0 | 0 | 1 | 0.006088 |
| remove_digit | 741 | 0 | 0.5762 | 0.4238 | 0.01007 |
| remove_digit_minimal | 787 | 0 | 0.5654 | 0.4346 | 0.03681 |
| remove_palm_body | 750 | 0 | 1 | 0 | n/a |
| resample_parameter | 2134 | 0 | 0 | 1 | 0.02158 |
| step_axis | 1594 | 0 | 0 | 1 | 0.1251 |
| step_coupling | 1686 | 0 | 0.9448 | 0.05516 | 0.1413 |
| step_limits | 1667 | 0 | 0.03779 | 0.9622 | 0.1387 |
| step_mount | 1693 | 0 | 0 | 1 | 0.0862 |
| step_radius | 1701 | 0 | 0 | 1 | 0 |
| step_root_length | 1644 | 0 | 0 | 1 | 0.1037 |
| toggle_palm_joint | 680 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|UNION_weighted|antipodal_pinch|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 653 | 0.001531 | 0 | 0.9985 | 0.1083 |
| add_minimal_digit | 742 | 0.004043 | 0 | 0.996 | 0.1535 |
| add_palm_body | 752 | 1 | 0 | 0 | n/a |
| delete_phalanx | 718 | 0 | 0.7451 | 0.2549 | 0.06509 |
| insert_phalanx | 752 | 0 | 0 | 1 | 0.03989 |
| perturb_parameter | 1653 | 0 | 0 | 1 | 0.1607 |
| regrow_subtree | 718 | 0 | 0 | 1 | 0.02415 |
| remove_digit | 678 | 0 | 0.3687 | 0.6313 | 0.01683 |
| remove_digit_minimal | 729 | 0 | 0.4129 | 0.5871 | 0.03448 |
| remove_palm_body | 746 | 0 | 1 | 0 | n/a |
| resample_parameter | 2059 | 0 | 0 | 1 | 0.03919 |
| step_axis | 1594 | 0 | 0 | 1 | 0.1579 |
| step_coupling | 1645 | 0 | 0.9726 | 0.02736 | 0 |
| step_limits | 1581 | 0 | 0.02593 | 0.9741 | 0.1275 |
| step_mount | 1644 | 0 | 0 | 1 | 0.1228 |
| step_radius | 1637 | 0 | 0 | 1 | 0.0006242 |
| step_root_length | 1671 | 0 | 0 | 1 | 0.1699 |
| toggle_palm_joint | 744 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|UNION_weighted|target_distance|0

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 652 | 0.5353 | 0 | 0.4647 | 0.7385 |
| add_minimal_digit | 688 | 0.5887 | 0 | 0.4113 | 0.2285 |
| add_palm_body | 709 | 1 | 0 | 0 | n/a |
| delete_phalanx | 689 | 0 | 0.008708 | 0.9913 | 0 |
| insert_phalanx | 652 | 0 | 0.003067 | 0.9969 | 0.1055 |
| perturb_parameter | 1508 | 0 | 0 | 1 | 0 |
| regrow_subtree | 623 | 0 | 0 | 1 | 0.1304 |
| remove_digit | 628 | 0 | 0.009554 | 0.9904 | 0 |
| remove_digit_minimal | 728 | 0 | 0.5192 | 0.4808 | 0 |
| remove_palm_body | 706 | 0 | 1 | 0 | n/a |
| resample_parameter | 1872 | 0 | 0 | 1 | 0.006554 |
| step_axis | 1502 | 0 | 0 | 1 | 0 |
| step_coupling | 1511 | 0 | 0.2118 | 0.7882 | 0 |
| step_limits | 1562 | 0 | 0 | 1 | 0 |
| step_mount | 1514 | 0 | 0 | 1 | 0.007437 |
| step_radius | 1609 | 0 | 0 | 1 | 0 |
| step_root_length | 1472 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 693 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|UNION_weighted|target_distance|0.01

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 642 | 0.5312 | 0 | 0.4688 | 0.6747 |
| add_minimal_digit | 698 | 0.5401 | 0 | 0.4599 | 0.1683 |
| add_palm_body | 651 | 1 | 0 | 0 | n/a |
| delete_phalanx | 639 | 0 | 0.01721 | 0.9828 | 0 |
| insert_phalanx | 675 | 0 | 0 | 1 | 0.1157 |
| perturb_parameter | 1501 | 0 | 0 | 1 | 0 |
| regrow_subtree | 645 | 0 | 0 | 1 | 0.1487 |
| remove_digit | 659 | 0 | 0.01821 | 0.9818 | 0 |
| remove_digit_minimal | 704 | 0 | 0.3963 | 0.6037 | 0 |
| remove_palm_body | 669 | 0 | 1 | 0 | n/a |
| resample_parameter | 1921 | 0 | 0 | 1 | 0.003214 |
| step_axis | 1497 | 0 | 0 | 1 | 0 |
| step_coupling | 1507 | 0 | 0.2681 | 0.7319 | 0 |
| step_limits | 1483 | 0 | 0 | 1 | 0 |
| step_mount | 1554 | 0 | 0 | 1 | 0.006592 |
| step_radius | 1503 | 0 | 0 | 1 | 0 |
| step_root_length | 1540 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 673 | 0 | 1 | 0 | n/a |

### G_NOBRANCH|UNION_weighted|target_distance|0.02

| operator | attempts | envelope_rejected_rate | inapplicable_rate | accepted_rate | improvement_rate |
|---|---|---|---|---|---|
| add_digit | 624 | 0.4808 | 0 | 0.5192 | 0.7042 |
| add_minimal_digit | 698 | 0.4799 | 0 | 0.5201 | 0.142 |
| add_palm_body | 717 | 1 | 0 | 0 | n/a |
| delete_phalanx | 646 | 0 | 0.01238 | 0.9876 | 0 |
| insert_phalanx | 674 | 0 | 0.001484 | 0.9985 | 0.09816 |
| perturb_parameter | 1514 | 0 | 0 | 1 | 0 |
| regrow_subtree | 638 | 0 | 0 | 1 | 0.1318 |
| remove_digit | 643 | 0 | 0.02022 | 0.9798 | 0 |
| remove_digit_minimal | 629 | 0 | 0.5119 | 0.4881 | 0 |
| remove_palm_body | 633 | 0 | 1 | 0 | n/a |
| resample_parameter | 1896 | 0 | 0 | 1 | 0.004338 |
| step_axis | 1520 | 0 | 0 | 1 | 0 |
| step_coupling | 1523 | 0 | 0.1865 | 0.8135 | 0 |
| step_limits | 1524 | 0 | 0.0006562 | 0.9993 | 0 |
| step_mount | 1492 | 0 | 0 | 1 | 0.004814 |
| step_radius | 1507 | 0 | 0 | 1 | 0 |
| step_root_length | 1480 | 0 | 0 | 1 | 0 |
| toggle_palm_joint | 745 | 0 | 1 | 0 | n/a |

## Reading

(Numbers only; caveats: `target_distance` fitness's structural-distance function is cached by phenotype_hash within a generation, so two structurally-equivalent but differently-derived individuals share one evaluation -- an approximation, not exact per-derivation scoring. `antipodal_pinch` fitness uses the same n_proxy_configs-sample geometric proxy as E5, still a diagnostic, never a real task objective.)

- G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.8932 [0.797, 0.9621] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.6791 [0.5116, 0.8431] (n=24); target reach rate 0; clone rate 6.352e-05.
- G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.02: final best raw proxy 0.6588 [0.4916, 0.8118] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|target_distance|0.02: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0: final best raw proxy 0.769 [0.6084, 0.9007] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0.01: final best raw proxy 0.7701 [0.6071, 0.9023] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|antipodal_pinch|0.02: final best raw proxy 0.7431 [0.5751, 0.8946] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|target_distance|0: final best raw proxy -0.25 [-0.4167, -0.08333] (n=24); target reach rate 0.75; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|target_distance|0.01: final best raw proxy -0.2917 [-0.4583, -0.125] (n=24); target reach rate 0.7083; clone rate 0.
- G_FULL_INS|UNION_weighted_nopalm|target_distance|0.02: final best raw proxy -0.375 [-0.625, -0.1667] (n=24); target reach rate 0.6667; clone rate 0.
- G_FULL_INS|UNION_weighted|antipodal_pinch|0: final best raw proxy 0.8994 [0.8008, 0.9728] (n=24); target reach rate 0; clone rate 0.000127.
- G_FULL_INS|UNION_weighted|antipodal_pinch|0.01: final best raw proxy 0.6198 [0.4285, 0.7979] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|UNION_weighted|antipodal_pinch|0.02: final best raw proxy 0.5844 [0.4, 0.7575] (n=24); target reach rate 0; clone rate 0.
- G_FULL_INS|UNION_weighted|target_distance|0: final best raw proxy -0.2917 [-0.4583, -0.125] (n=24); target reach rate 0.7083; clone rate 0.
- G_FULL_INS|UNION_weighted|target_distance|0.01: final best raw proxy -0.3333 [-0.5833, -0.125] (n=24); target reach rate 0.7083; clone rate 0.
- G_FULL_INS|UNION_weighted|target_distance|0.02: final best raw proxy -0.25 [-0.4583, -0.08333] (n=24); target reach rate 0.7917; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.793 [0.6595, 0.912] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.7248 [0.5582, 0.8751] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.02: final best raw proxy 0.7389 [0.5769, 0.884] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.02: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0: final best raw proxy 0.8647 [0.7431, 0.9601] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0.01: final best raw proxy 0.655 [0.4514, 0.8212] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|antipodal_pinch|0.02: final best raw proxy 0.8025 [0.6491, 0.9313] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0: final best raw proxy -0.2917 [-0.4583, -0.125] (n=24); target reach rate 0.7083; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0.01: final best raw proxy -0.25 [-0.4167, -0.08333] (n=24); target reach rate 0.75; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted_nopalm|target_distance|0.02: final best raw proxy -0.3333 [-0.5833, -0.125] (n=24); target reach rate 0.7083; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0: final best raw proxy 0.8667 [0.7491, 0.9576] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0.01: final best raw proxy 0.7692 [0.6172, 0.8997] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted|antipodal_pinch|0.02: final best raw proxy 0.6055 [0.4115, 0.7803] (n=24); target reach rate 0; clone rate 6.352e-05.
- G_NOBRANCH_INS|UNION_weighted|target_distance|0: final best raw proxy -0.3333 [-0.5417, -0.1667] (n=24); target reach rate 0.6667; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted|target_distance|0.01: final best raw proxy -0.1667 [-0.3333, -0.04167] (n=24); target reach rate 0.8333; clone rate 0.
- G_NOBRANCH_INS|UNION_weighted|target_distance|0.02: final best raw proxy -0.4583 [-0.7083, -0.25] (n=24); target reach rate 0.5833; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.7019 [0.5443, 0.8498] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.43 [0.2546, 0.5999] (n=24); target reach rate 0; clone rate 6.352e-05.
- G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.02: final best raw proxy 0.6346 [0.4525, 0.7924] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|target_distance|0.02: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0: final best raw proxy 0.8202 [0.6829, 0.9376] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0.01: final best raw proxy 0.6975 [0.5252, 0.8614] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|antipodal_pinch|0.02: final best raw proxy 0.7293 [0.5569, 0.8916] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|target_distance|0.01: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; clone rate 0.
- G_NOBRANCH|UNION_weighted_nopalm|target_distance|0.02: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|UNION_weighted|antipodal_pinch|0: final best raw proxy 0.7069 [0.5329, 0.8664] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|UNION_weighted|antipodal_pinch|0.01: final best raw proxy 0.533 [0.3306, 0.7386] (n=24); target reach rate 0; clone rate 0.
- G_NOBRANCH|UNION_weighted|antipodal_pinch|0.02: final best raw proxy 0.7682 [0.6012, 0.9151] (n=24); target reach rate 0; clone rate 6.352e-05.
- G_NOBRANCH|UNION_weighted|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
- G_NOBRANCH|UNION_weighted|target_distance|0.01: final best raw proxy -0.08333 [-0.25, 0] (n=24); target reach rate 0.9583; clone rate 0.
- G_NOBRANCH|UNION_weighted|target_distance|0.02: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; clone rate 0.
