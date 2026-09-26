# Experiment: e7_schedule

Wall time: 208.289 s

**Note:** timing probe: probe_task=(0, 'G_FULL_INS', 'DEFAULT_uniform', 'antipodal_pinch', '0') took 4.062s; projected full run (1152 tasks over 32 processes) = 146.2s (budget 1800s). No conditions dropped.

## Params

```json
{
  "dropped_pinch_cost_0_01": false,
  "envelope": {
    "allow_branches": false,
    "allow_palm_joints": false,
    "max_digits": 5,
    "max_joints_per_digit": 6
  },
  "generations": 40,
  "lam": 16,
  "mu": 16,
  "n_conditions": 48,
  "n_proxy_configs": 16,
  "n_restarts": 24,
  "projected_full_run_s": 146.23449039459229,
  "timing_probe_s": 4.0620691776275635
}
```

n_conditions: 48

## Per-condition (mean over restarts, 95% CI)

| dist | pool | fitness | cost | n | final best raw proxy | 95% CI | final motors | final joints | final digits | target reach rate | gen first reach (censored, mean) | late-locality mean tip displacement (m) | clone rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.8932 | [0.797, 0.9621] | 6.19 | 6.674 | 4.096 | 0 | 40 | 0.01689 | 0 |
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.6791 | [0.5116, 0.8431] | 2.917 | 3.055 | 2.443 | 0 | 40 | 0.02319 | 6.352e-05 |
| G_FULL_INS | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 15.27 | 17.33 | 5 | 1 | 15.38 | 0.0151 | 0 |
| G_FULL_INS | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 14.83 | 16.96 | 5 | 1 | 16 | 0.01395 | 0 |
| G_FULL_INS | SCHEDULE_linear | antipodal_pinch | 0 | 24 | 0.9807 | [0.9714, 0.9885] | 5.659 | 5.885 | 4.318 | 0 | 40 | 0.006691 | 0 |
| G_FULL_INS | SCHEDULE_linear | antipodal_pinch | 0.01 | 24 | 0.8555 | [0.7256, 0.975] | 3.299 | 3.393 | 2.888 | 0 | 40 | 0.007787 | 0 |
| G_FULL_INS | SCHEDULE_linear | target_distance | 0 | 24 | -0.125 | [-0.25, 0] | 14.09 | 15.94 | 4.951 | 0.875 | 22.75 | 0.006081 | 0 |
| G_FULL_INS | SCHEDULE_linear | target_distance | 0.01 | 24 | -0.1667 | [-0.3333, -0.04167] | 13.33 | 15.5 | 4.956 | 0.8333 | 24.92 | 0.00534 | 0 |
| G_FULL_INS | SCHEDULE_step | antipodal_pinch | 0 | 24 | 0.7207 | [0.5519, 0.8805] | 5.469 | 5.867 | 3.802 | 0 | 40 | 0.007588 | 0 |
| G_FULL_INS | SCHEDULE_step | antipodal_pinch | 0.01 | 24 | 0.731 | [0.5626, 0.8829] | 2.755 | 2.802 | 2.365 | 0 | 40 | 0.009988 | 0 |
| G_FULL_INS | SCHEDULE_step | target_distance | 0 | 24 | -0.04167 | [-0.125, 0] | 13.62 | 15.8 | 4.997 | 0.9583 | 20.33 | 0.005902 | 0 |
| G_FULL_INS | SCHEDULE_step | target_distance | 0.01 | 24 | 0 | [0, 0] | 13.35 | 15.8 | 4.992 | 1 | 16.67 | 0.006279 | 0 |
| G_FULL_INS | UNION_nopalm_weighted | antipodal_pinch | 0 | 24 | 0.8191 | [0.6873, 0.9346] | 5.107 | 5.221 | 3.849 | 0 | 40 | 0.008016 | 0 |
| G_FULL_INS | UNION_nopalm_weighted | antipodal_pinch | 0.01 | 24 | 0.6726 | [0.4947, 0.8322] | 2.406 | 2.49 | 2.104 | 0 | 40 | 0.009004 | 0 |
| G_FULL_INS | UNION_nopalm_weighted | target_distance | 0 | 24 | -0.25 | [-0.4583, -0.08333] | 13.15 | 14.87 | 4.885 | 0.7917 | 30.25 | 0.007322 | 0 |
| G_FULL_INS | UNION_nopalm_weighted | target_distance | 0.01 | 24 | -0.3333 | [-0.5417, -0.1667] | 13.01 | 14.4 | 4.846 | 0.6667 | 32.08 | 0.006462 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.793 | [0.6595, 0.912] | 5.982 | 6.323 | 3.956 | 0 | 40 | 0.0165 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.7248 | [0.5582, 0.8751] | 3.388 | 3.594 | 2.737 | 0 | 40 | 0.02144 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 15.81 | 17.77 | 5 | 1 | 16 | 0.01363 | 0 |
| G_NOBRANCH_INS | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 14.58 | 16.92 | 5 | 1 | 14.17 | 0.01465 | 0 |
| G_NOBRANCH_INS | SCHEDULE_linear | antipodal_pinch | 0 | 24 | 0.976 | [0.9613, 0.9855] | 5.51 | 5.839 | 4.393 | 0 | 40 | 0.004857 | 6.352e-05 |
| G_NOBRANCH_INS | SCHEDULE_linear | antipodal_pinch | 0.01 | 24 | 0.7917 | [0.6319, 0.9319] | 2.742 | 2.841 | 2.404 | 0 | 40 | 0.01028 | 0.000127 |
| G_NOBRANCH_INS | SCHEDULE_linear | target_distance | 0 | 24 | -0.08333 | [-0.2083, 0] | 13.53 | 15.69 | 4.951 | 0.9167 | 22.67 | 0.006249 | 0 |
| G_NOBRANCH_INS | SCHEDULE_linear | target_distance | 0.01 | 24 | -0.04167 | [-0.125, 0] | 14.28 | 16.3 | 4.992 | 0.9583 | 22.12 | 0.006388 | 0 |
| G_NOBRANCH_INS | SCHEDULE_step | antipodal_pinch | 0 | 24 | 0.8696 | [0.7494, 0.958] | 5.43 | 5.753 | 3.919 | 0 | 40 | 0.008438 | 0 |
| G_NOBRANCH_INS | SCHEDULE_step | antipodal_pinch | 0.01 | 24 | 0.7324 | [0.5652, 0.8961] | 2.487 | 2.63 | 2.161 | 0 | 40 | 0.01166 | 0 |
| G_NOBRANCH_INS | SCHEDULE_step | target_distance | 0 | 24 | 0 | [0, 0] | 14.4 | 16.08 | 4.974 | 1 | 16.71 | 0.006021 | 0 |
| G_NOBRANCH_INS | SCHEDULE_step | target_distance | 0.01 | 24 | -0.04167 | [-0.125, 0] | 13.77 | 15.6 | 4.99 | 0.9583 | 15.54 | 0.008005 | 0 |
| G_NOBRANCH_INS | UNION_nopalm_weighted | antipodal_pinch | 0 | 24 | 0.8318 | [0.6914, 0.9499] | 5.073 | 5.318 | 3.68 | 0 | 40 | 0.009602 | 0 |
| G_NOBRANCH_INS | UNION_nopalm_weighted | antipodal_pinch | 0.01 | 24 | 0.5591 | [0.3752, 0.7379] | 2.339 | 2.344 | 2.081 | 0 | 40 | 0.01014 | 0 |
| G_NOBRANCH_INS | UNION_nopalm_weighted | target_distance | 0 | 24 | -0.25 | [-0.4583, -0.08333] | 13.22 | 14.89 | 4.885 | 0.7917 | 30.79 | 0.006758 | 0 |
| G_NOBRANCH_INS | UNION_nopalm_weighted | target_distance | 0.01 | 24 | -0.25 | [-0.4583, -0.08333] | 12.78 | 14.8 | 4.88 | 0.7917 | 30.54 | 0.008011 | 0 |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0 | 24 | 0.7019 | [0.5443, 0.8498] | 8.755 | 9.885 | 3.784 | 0 | 40 | 0.02506 | 0 |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | 0.01 | 24 | 0.43 | [0.2546, 0.5999] | 2.378 | 2.427 | 1.773 | 0 | 40 | 0.04539 | 6.352e-05 |
| G_NOBRANCH | DEFAULT_uniform | target_distance | 0 | 24 | 0 | [0, 0] | 19.73 | 22.88 | 5 | 1 | 10.17 | 0.01776 | 0 |
| G_NOBRANCH | DEFAULT_uniform | target_distance | 0.01 | 24 | 0 | [0, 0] | 17.27 | 20.76 | 4.997 | 1 | 11.21 | 0.01682 | 0 |
| G_NOBRANCH | SCHEDULE_linear | antipodal_pinch | 0 | 24 | 0.913 | [0.8067, 0.9792] | 6.768 | 7.26 | 4.237 | 0 | 40 | 0.01053 | 0 |
| G_NOBRANCH | SCHEDULE_linear | antipodal_pinch | 0.01 | 24 | 0.7818 | [0.6163, 0.9448] | 2.602 | 2.628 | 2.352 | 0 | 40 | 0.01012 | 0 |
| G_NOBRANCH | SCHEDULE_linear | target_distance | 0 | 24 | 0 | [0, 0] | 19.17 | 22.33 | 5 | 1 | 13.54 | 0.008063 | 0 |
| G_NOBRANCH | SCHEDULE_linear | target_distance | 0.01 | 24 | 0 | [0, 0] | 17.16 | 20.23 | 5 | 1 | 15.67 | 0.007028 | 0 |
| G_NOBRANCH | SCHEDULE_step | antipodal_pinch | 0 | 24 | 0.6555 | [0.4763, 0.8234] | 6.888 | 7.495 | 3.456 | 0 | 40 | 0.0172 | 0 |
| G_NOBRANCH | SCHEDULE_step | antipodal_pinch | 0.01 | 24 | 0.5588 | [0.3786, 0.7366] | 2.37 | 2.477 | 2.039 | 0 | 40 | 0.01513 | 0 |
| G_NOBRANCH | SCHEDULE_step | target_distance | 0 | 24 | 0 | [0, 0] | 18.59 | 21.88 | 5 | 1 | 10.75 | 0.008184 | 0 |
| G_NOBRANCH | SCHEDULE_step | target_distance | 0.01 | 24 | 0 | [0, 0] | 17.47 | 20.38 | 5 | 1 | 9.625 | 0.007843 | 0 |
| G_NOBRANCH | UNION_nopalm_weighted | antipodal_pinch | 0 | 24 | 0.7873 | [0.6424, 0.9199] | 5.659 | 6.177 | 3.521 | 0 | 40 | 0.01179 | 0 |
| G_NOBRANCH | UNION_nopalm_weighted | antipodal_pinch | 0.01 | 24 | 0.6402 | [0.4531, 0.8157] | 2.279 | 2.299 | 2.047 | 0 | 40 | 0.01343 | 0 |
| G_NOBRANCH | UNION_nopalm_weighted | target_distance | 0 | 24 | -0.04167 | [-0.125, 0] | 16.98 | 20.21 | 5 | 0.9583 | 21.46 | 0.006197 | 0 |
| G_NOBRANCH | UNION_nopalm_weighted | target_distance | 0.01 | 24 | -0.04167 | [-0.125, 0] | 18.58 | 21.59 | 5 | 0.9583 | 22.21 | 0.00733 | 0 |

## Paired difference: schedule minus static pool, per dist x fitness x cost

| schedule minus pool | dist | fitness | cost | metric | mean DIFFERENCE | 95% CI | n |
|---|---|---|---|---|---|---|---|
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.08746 | [0.02073, 0.1796] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.1765 | [-0.07317, 0.4058] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | final_best_raw_proxy | -0.125 | [-0.25, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | gen_first_reach_censored | 7.375 | [4.042, 11.04] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | target_reach_rate | -0.125 | [-0.25, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | final_best_raw_proxy | -0.1667 | [-0.3333, -0.04167] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | gen_first_reach_censored | 8.917 | [5.625, 12.38] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | target_reach_rate | -0.1667 | [-0.3333, -0.04167] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.1829 | [0.06291, 0.319] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.06698 | [-0.1246, 0.2738] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | final_best_raw_proxy | -0.08333 | [-0.2083, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | gen_first_reach_censored | 6.667 | [2.792, 10.71] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | target_reach_rate | -0.08333 | [-0.2083, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | final_best_raw_proxy | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | gen_first_reach_censored | 7.958 | [4.415, 11.54] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | target_reach_rate | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | final_best_raw_proxy | 0.2112 | [0.008758, 0.3988] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.3518 | [0.1013, 0.5958] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | gen_first_reach_censored | 3.375 | [1.708, 5.084] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | gen_first_reach_censored | 4.458 | [2.667, 6.334] | 24 |
| SCHEDULE_linear_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.1616 | [0.04859, 0.2938] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.1829 | [-0.04635, 0.4153] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | final_best_raw_proxy | 0.125 | [-0.125, 0.375] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | gen_first_reach_censored | -7.5 | [-12.46, -2.042] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | target_reach_rate | 0.08333 | [-0.125, 0.2917] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | final_best_raw_proxy | 0.1667 | [-0.125, 0.4167] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | gen_first_reach_censored | -7.167 | [-12.5, -1.75] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | target_reach_rate | 0.1667 | [-0.125, 0.4167] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.1442 | [0.02745, 0.283] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.2326 | [0.006193, 0.4524] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | final_best_raw_proxy | 0.1667 | [-0.08333, 0.4167] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | gen_first_reach_censored | -8.125 | [-11.79, -4.208] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | target_reach_rate | 0.125 | [-0.08333, 0.3333] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | final_best_raw_proxy | 0.2083 | [0, 0.4583] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | gen_first_reach_censored | -8.417 | [-13.25, -4.083] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | target_reach_rate | 0.1667 | [-0.04167, 0.375] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | final_best_raw_proxy | 0.1257 | [-0.05678, 0.3047] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.1416 | [-0.06114, 0.3389] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | final_best_raw_proxy | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | gen_first_reach_censored | -7.917 | [-10.96, -4.916] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | target_reach_rate | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | final_best_raw_proxy | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | gen_first_reach_censored | -6.542 | [-9.542, -3.749] | 24 |
| SCHEDULE_linear_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | target_reach_rate | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | final_best_raw_proxy | -0.1725 | [-0.3757, 0.01873] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.05195 | [-0.1476, 0.2606] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | final_best_raw_proxy | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | gen_first_reach_censored | 4.958 | [1.292, 8.917] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0 | target_reach_rate | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | gen_first_reach_censored | 0.6667 | [-2.375, 3.792] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_FULL_INS | target_distance | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.07653 | [-0.07448, 0.2245] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.007585 | [-0.1871, 0.2002] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | gen_first_reach_censored | 0.7083 | [-2.042, 3.708] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | final_best_raw_proxy | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | gen_first_reach_censored | 1.375 | [-1.875, 4.875] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH_INS | target_distance | 0.01 | target_reach_rate | -0.04167 | [-0.125, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | final_best_raw_proxy | -0.04634 | [-0.2117, 0.1158] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.1288 | [-0.08672, 0.345] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | gen_first_reach_censored | 0.5833 | [-0.9167, 1.918] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | final_best_raw_proxy | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | gen_first_reach_censored | -1.583 | [-2.917, -0.2083] | 24 |
| SCHEDULE_step_minus_DEFAULT_uniform | G_NOBRANCH | target_distance | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | final_best_raw_proxy | -0.09838 | [-0.2909, 0.09169] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.0584 | [-0.1607, 0.2946] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | final_best_raw_proxy | 0.2083 | [0, 0.4583] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | gen_first_reach_censored | -9.917 | [-14.58, -4.833] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0 | target_reach_rate | 0.1667 | [0, 0.375] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | final_best_raw_proxy | 0.3333 | [0.1667, 0.5417] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | gen_first_reach_censored | -15.42 | [-18.88, -11.79] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_FULL_INS | target_distance | 0.01 | target_reach_rate | 0.3333 | [0.1667, 0.5417] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | final_best_raw_proxy | 0.03775 | [-0.141, 0.2171] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | final_best_raw_proxy | 0.1732 | [-0.1196, 0.4422] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | final_best_raw_proxy | 0.25 | [0.08333, 0.4583] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | gen_first_reach_censored | -14.08 | [-17.38, -10.62] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0 | target_reach_rate | 0.2083 | [0.08333, 0.375] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | final_best_raw_proxy | 0.2083 | [0, 0.4583] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | gen_first_reach_censored | -15 | [-19.25, -10.96] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH_INS | target_distance | 0.01 | target_reach_rate | 0.1667 | [0, 0.375] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | final_best_raw_proxy | -0.1318 | [-0.2932, 0.03567] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | final_best_raw_proxy | -0.0814 | [-0.321, 0.1708] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | gen_first_reach_censored | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | antipodal_pinch | 0.01 | target_reach_rate | 0 | [0, 0] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | final_best_raw_proxy | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | gen_first_reach_censored | -10.71 | [-13.75, -8.083] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0 | target_reach_rate | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | final_best_raw_proxy | 0.04167 | [0, 0.125] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | gen_first_reach_censored | -12.58 | [-15, -10.29] | 24 |
| SCHEDULE_step_minus_UNION_nopalm_weighted | G_NOBRANCH | target_distance | 0.01 | target_reach_rate | 0.04167 | [0, 0.125] | 24 |

## Improvement rate per operator group (structural vs small), by generation third, per condition

| condition | group | third 0 (early) | third 1 (mid) | third 2 (late) |
|---|---|---|---|---|
| G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0 | small | 0.04172 | 0.08328 | 0.07199 |
| G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0 | structural | 0.05252 | 0.1345 | 0.162 |
| G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.01 | small | 0.02163 | 0.08436 | 0.06276 |
| G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.01 | structural | 0.0389 | 0.07125 | 0.07357 |
| G_FULL_INS|DEFAULT_uniform|target_distance|0 | small | 0.004416 | 0 | 0 |
| G_FULL_INS|DEFAULT_uniform|target_distance|0 | structural | 0.1504 | 0.02507 | 0 |
| G_FULL_INS|DEFAULT_uniform|target_distance|0.01 | small | 0.006782 | 0 | 0 |
| G_FULL_INS|DEFAULT_uniform|target_distance|0.01 | structural | 0.1537 | 0.02237 | 0.001766 |
| G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0 | small | 0.07912 | 0.1357 | 0.1032 |
| G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0 | structural | 0.06928 | 0.1574 | 0.1625 |
| G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0.01 | small | 0.06388 | 0.112 | 0.1184 |
| G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0.01 | structural | 0.06661 | 0.1326 | 0.1057 |
| G_FULL_INS|SCHEDULE_linear|target_distance|0 | small | 0.008264 | 0 | 0 |
| G_FULL_INS|SCHEDULE_linear|target_distance|0 | structural | 0.1391 | 0.02336 | 0.005994 |
| G_FULL_INS|SCHEDULE_linear|target_distance|0.01 | small | 0.001717 | 0.001025 | 0 |
| G_FULL_INS|SCHEDULE_linear|target_distance|0.01 | structural | 0.1273 | 0.02615 | 0.00496 |
| G_FULL_INS|SCHEDULE_step|antipodal_pinch|0 | small | 0.03311 | 0.08186 | 0.08962 |
| G_FULL_INS|SCHEDULE_step|antipodal_pinch|0 | structural | 0.03472 | 0.08752 | 0.1094 |
| G_FULL_INS|SCHEDULE_step|antipodal_pinch|0.01 | small | 0.02344 | 0.09827 | 0.1084 |
| G_FULL_INS|SCHEDULE_step|antipodal_pinch|0.01 | structural | 0.03964 | 0.09118 | 0.1059 |
| G_FULL_INS|SCHEDULE_step|target_distance|0 | small | 0.008584 | 0.0002649 | 0 |
| G_FULL_INS|SCHEDULE_step|target_distance|0 | structural | 0.1661 | 0.02712 | 0.005181 |
| G_FULL_INS|SCHEDULE_step|target_distance|0.01 | small | 0.009938 | 0.0005292 | 0 |
| G_FULL_INS|SCHEDULE_step|target_distance|0.01 | structural | 0.1567 | 0.03462 | 0.003254 |
| G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0 | small | 0.06426 | 0.1153 | 0.08308 |
| G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0 | structural | 0.04464 | 0.09004 | 0.09008 |
| G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0.01 | small | 0.0536 | 0.1157 | 0.1015 |
| G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0.01 | structural | 0.04549 | 0.07268 | 0.0425 |
| G_FULL_INS|UNION_nopalm_weighted|target_distance|0 | small | 0.002603 | 0.0002691 | 0.0007684 |
| G_FULL_INS|UNION_nopalm_weighted|target_distance|0 | structural | 0.1741 | 0.04467 | 0.02298 |
| G_FULL_INS|UNION_nopalm_weighted|target_distance|0.01 | small | 0.003393 | 0 | 0.0002525 |
| G_FULL_INS|UNION_nopalm_weighted|target_distance|0.01 | structural | 0.1683 | 0.05714 | 0.02134 |
| G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0 | small | 0.03659 | 0.07224 | 0.06792 |
| G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0 | structural | 0.03936 | 0.1172 | 0.1576 |
| G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.01 | small | 0.03464 | 0.08062 | 0.07326 |
| G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.01 | structural | 0.04424 | 0.1026 | 0.1187 |
| G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0 | small | 0.005664 | 0 | 0 |
| G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0 | structural | 0.159 | 0.01894 | 0.0008934 |
| G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.01 | small | 0.01053 | 0 | 0 |
| G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.01 | structural | 0.1515 | 0.02149 | 0 |
| G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0 | small | 0.07606 | 0.1368 | 0.1189 |
| G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0 | structural | 0.1022 | 0.1863 | 0.173 |
| G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0.01 | small | 0.07302 | 0.1202 | 0.118 |
| G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0.01 | structural | 0.06649 | 0.1358 | 0.08861 |
| G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0 | small | 0.009592 | 0.0006866 | 0 |
| G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0 | structural | 0.1454 | 0.02165 | 0.01113 |
| G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0.01 | small | 0.001845 | 0.0003471 | 0 |
| G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0.01 | structural | 0.1216 | 0.02274 | 0.01129 |
| G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0 | small | 0.02994 | 0.08023 | 0.08807 |
| G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0 | structural | 0.04128 | 0.09276 | 0.1241 |
| G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0.01 | small | 0.03531 | 0.1303 | 0.1306 |
| G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0.01 | structural | 0.04551 | 0.08919 | 0.08216 |
| G_NOBRANCH_INS|SCHEDULE_step|target_distance|0 | small | 0.005992 | 0.0005233 | 0 |
| G_NOBRANCH_INS|SCHEDULE_step|target_distance|0 | structural | 0.1664 | 0.02991 | 0.001071 |
| G_NOBRANCH_INS|SCHEDULE_step|target_distance|0.01 | small | 0.006888 | 0 | 0 |
| G_NOBRANCH_INS|SCHEDULE_step|target_distance|0.01 | structural | 0.1688 | 0.01991 | 0.003405 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0 | small | 0.07335 | 0.1162 | 0.1116 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0 | structural | 0.06161 | 0.1461 | 0.11 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0.01 | small | 0.03371 | 0.07291 | 0.09204 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0.01 | structural | 0.04056 | 0.07028 | 0.07583 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0 | small | 0.003873 | 0.001072 | 0 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0 | structural | 0.1697 | 0.06656 | 0.02624 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0.01 | small | 0.003383 | 0 | 0 |
| G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0.01 | structural | 0.1813 | 0.04691 | 0.02361 |
| G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0 | small | 0.0414 | 0.03251 | 0.04398 |
| G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0 | structural | 0.04071 | 0.0726 | 0.07429 |
| G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.01 | small | 0.0258 | 0.03591 | 0.03359 |
| G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.01 | structural | 0.0224 | 0.0648 | 0.05071 |
| G_NOBRANCH|DEFAULT_uniform|target_distance|0 | small | 0.00309 | 0 | 0 |
| G_NOBRANCH|DEFAULT_uniform|target_distance|0 | structural | 0.1889 | 0.006522 | 0 |
| G_NOBRANCH|DEFAULT_uniform|target_distance|0.01 | small | 0.004997 | 0 | 0 |
| G_NOBRANCH|DEFAULT_uniform|target_distance|0.01 | structural | 0.1791 | 0.008923 | 0 |
| G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0 | small | 0.06524 | 0.1007 | 0.09339 |
| G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0 | structural | 0.05543 | 0.1398 | 0.1654 |
| G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0.01 | small | 0.06387 | 0.112 | 0.1159 |
| G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0.01 | structural | 0.07312 | 0.1195 | 0.09934 |
| G_NOBRANCH|SCHEDULE_linear|target_distance|0 | small | 0.007688 | 0 | 0 |
| G_NOBRANCH|SCHEDULE_linear|target_distance|0 | structural | 0.1658 | 0.01462 | 0 |
| G_NOBRANCH|SCHEDULE_linear|target_distance|0.01 | small | 0.002315 | 0.0003221 | 0 |
| G_NOBRANCH|SCHEDULE_linear|target_distance|0.01 | structural | 0.1538 | 0.02756 | 0.003289 |
| G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0 | small | 0.03168 | 0.08118 | 0.05923 |
| G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0 | structural | 0.03214 | 0.0776 | 0.07509 |
| G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0.01 | small | 0.02159 | 0.0922 | 0.08766 |
| G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0.01 | structural | 0.03206 | 0.08153 | 0.0847 |
| G_NOBRANCH|SCHEDULE_step|target_distance|0 | small | 0.007299 | 0 | 0 |
| G_NOBRANCH|SCHEDULE_step|target_distance|0 | structural | 0.1919 | 0.01724 | 0 |
| G_NOBRANCH|SCHEDULE_step|target_distance|0.01 | small | 0.007233 | 0 | 0 |
| G_NOBRANCH|SCHEDULE_step|target_distance|0.01 | structural | 0.1886 | 0.00448 | 0 |
| G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0 | small | 0.06112 | 0.08277 | 0.07412 |
| G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0 | structural | 0.04118 | 0.06921 | 0.09396 |
| G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0.01 | small | 0.05501 | 0.1041 | 0.1231 |
| G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0.01 | structural | 0.06625 | 0.09768 | 0.1018 |
| G_NOBRANCH|UNION_nopalm_weighted|target_distance|0 | small | 0.004836 | 0.0002523 | 0 |
| G_NOBRANCH|UNION_nopalm_weighted|target_distance|0 | structural | 0.2287 | 0.05928 | 0.01115 |
| G_NOBRANCH|UNION_nopalm_weighted|target_distance|0.01 | small | 0.004118 | 0.0002556 | 0 |
| G_NOBRANCH|UNION_nopalm_weighted|target_distance|0.01 | structural | 0.226 | 0.0519 | 0.007804 |

## Reading

(Numbers only. Caveats: `gen_first_reach_censored` assigns `generations` (40, or the run's own `generations` param) to any restart that never reached the target -- a simple censoring imputation, not a Kaplan-Meier estimator, so its mean is a LOWER BOUND on the true mean generation-to-reach whenever the reach rate is below 1. `late_locality_mean_tip_displacement_m` is a mean over at most 8 sampled non-cloned offspring per restart from the last 10 of 40 generations (<=192 pairs per condition across 24 restarts), joint_identity-aligned exactly as in e1_locality/e5b's I15 fixes -- a diagnostic sample, not an exhaustive count. `improvement_rate_per_group_by_third` buckets `regrow_subtree`/`add_digit`/`remove_digit`/`add_minimal_digit`/`remove_digit_minimal`/`insert_phalanx`/`delete_phalanx` as "structural" and the 6 `step_*` operators + `perturb_parameter` + `resample_parameter` as "small"; a third with 0 non-cloned offspring for a group reports `n/a`.)

- G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.8932 [0.797, 0.9621] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.6791 [0.5116, 0.8431] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 6.352e-05.
- G_FULL_INS|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 15.38; clone rate 0.
- G_FULL_INS|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 16; clone rate 0.
- G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0: final best raw proxy 0.9807 [0.9714, 0.9885] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|SCHEDULE_linear|antipodal_pinch|0.01: final best raw proxy 0.8555 [0.7256, 0.975] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|SCHEDULE_linear|target_distance|0: final best raw proxy -0.125 [-0.25, 0] (n=24); target reach rate 0.875; gen first reach (censored, mean) 22.75; clone rate 0.
- G_FULL_INS|SCHEDULE_linear|target_distance|0.01: final best raw proxy -0.1667 [-0.3333, -0.04167] (n=24); target reach rate 0.8333; gen first reach (censored, mean) 24.92; clone rate 0.
- G_FULL_INS|SCHEDULE_step|antipodal_pinch|0: final best raw proxy 0.7207 [0.5519, 0.8805] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|SCHEDULE_step|antipodal_pinch|0.01: final best raw proxy 0.731 [0.5626, 0.8829] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|SCHEDULE_step|target_distance|0: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; gen first reach (censored, mean) 20.33; clone rate 0.
- G_FULL_INS|SCHEDULE_step|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 16.67; clone rate 0.
- G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0: final best raw proxy 0.8191 [0.6873, 0.9346] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|UNION_nopalm_weighted|antipodal_pinch|0.01: final best raw proxy 0.6726 [0.4947, 0.8322] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_FULL_INS|UNION_nopalm_weighted|target_distance|0: final best raw proxy -0.25 [-0.4583, -0.08333] (n=24); target reach rate 0.7917; gen first reach (censored, mean) 30.25; clone rate 0.
- G_FULL_INS|UNION_nopalm_weighted|target_distance|0.01: final best raw proxy -0.3333 [-0.5417, -0.1667] (n=24); target reach rate 0.6667; gen first reach (censored, mean) 32.08; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.793 [0.6595, 0.912] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.7248 [0.5582, 0.8751] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 16; clone rate 0.
- G_NOBRANCH_INS|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 14.17; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0: final best raw proxy 0.976 [0.9613, 0.9855] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 6.352e-05.
- G_NOBRANCH_INS|SCHEDULE_linear|antipodal_pinch|0.01: final best raw proxy 0.7917 [0.6319, 0.9319] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.000127.
- G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0: final best raw proxy -0.08333 [-0.2083, 0] (n=24); target reach rate 0.9167; gen first reach (censored, mean) 22.67; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_linear|target_distance|0.01: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; gen first reach (censored, mean) 22.12; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0: final best raw proxy 0.8696 [0.7494, 0.958] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_step|antipodal_pinch|0.01: final best raw proxy 0.7324 [0.5652, 0.8961] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_step|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 16.71; clone rate 0.
- G_NOBRANCH_INS|SCHEDULE_step|target_distance|0.01: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; gen first reach (censored, mean) 15.54; clone rate 0.
- G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0: final best raw proxy 0.8318 [0.6914, 0.9499] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|UNION_nopalm_weighted|antipodal_pinch|0.01: final best raw proxy 0.5591 [0.3752, 0.7379] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0: final best raw proxy -0.25 [-0.4583, -0.08333] (n=24); target reach rate 0.7917; gen first reach (censored, mean) 30.79; clone rate 0.
- G_NOBRANCH_INS|UNION_nopalm_weighted|target_distance|0.01: final best raw proxy -0.25 [-0.4583, -0.08333] (n=24); target reach rate 0.7917; gen first reach (censored, mean) 30.54; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0: final best raw proxy 0.7019 [0.5443, 0.8498] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|antipodal_pinch|0.01: final best raw proxy 0.43 [0.2546, 0.5999] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 6.352e-05.
- G_NOBRANCH|DEFAULT_uniform|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 10.17; clone rate 0.
- G_NOBRANCH|DEFAULT_uniform|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 11.21; clone rate 0.
- G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0: final best raw proxy 0.913 [0.8067, 0.9792] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|SCHEDULE_linear|antipodal_pinch|0.01: final best raw proxy 0.7818 [0.6163, 0.9448] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|SCHEDULE_linear|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 13.54; clone rate 0.
- G_NOBRANCH|SCHEDULE_linear|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 15.67; clone rate 0.
- G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0: final best raw proxy 0.6555 [0.4763, 0.8234] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|SCHEDULE_step|antipodal_pinch|0.01: final best raw proxy 0.5588 [0.3786, 0.7366] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|SCHEDULE_step|target_distance|0: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 10.75; clone rate 0.
- G_NOBRANCH|SCHEDULE_step|target_distance|0.01: final best raw proxy 0 [0, 0] (n=24); target reach rate 1; gen first reach (censored, mean) 9.625; clone rate 0.
- G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0: final best raw proxy 0.7873 [0.6424, 0.9199] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|UNION_nopalm_weighted|antipodal_pinch|0.01: final best raw proxy 0.6402 [0.4531, 0.8157] (n=24); target reach rate 0; gen first reach (censored, mean) 40; clone rate 0.
- G_NOBRANCH|UNION_nopalm_weighted|target_distance|0: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; gen first reach (censored, mean) 21.46; clone rate 0.
- G_NOBRANCH|UNION_nopalm_weighted|target_distance|0.01: final best raw proxy -0.04167 [-0.125, 0] (n=24); target reach rate 0.9583; gen first reach (censored, mean) 22.21; clone rate 0.
