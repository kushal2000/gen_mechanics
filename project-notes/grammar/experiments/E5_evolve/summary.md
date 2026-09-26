# Experiment: e5_evolve

Wall time: 689.941 s

## Params

```json
{
  "cost_weight": 0.02,
  "fixed_eval_seed": false,
  "generations": 40,
  "lam": 32,
  "mu": 32,
  "n_conditions": 48,
  "n_proxy_configs": 16,
  "n_restarts": 6
}
```

n_ok: 288  n_failed: 0

## Table 1: final best fitness / final motors, per dist x pool x fitness x cost (mean, 95% bootstrap CI over restarts)

| dist | pool | fitness | cost | final best fitness | final mean motors | gen reaching 90% of final best | diversity @ gen40 |
|---|---|---|---|---|---|---|---|
| G_FULL | DEFAULT_uniform | antipodal_pinch | aware | 0.7776 [0.6312, 0.8791] (n=6) | 6.224 [3.411, 10.34] (n=6) | 22.33 [14, 30.84] (n=6) | 0.849 [0.7448, 0.9375] (n=6) |
| G_FULL | DEFAULT_uniform | antipodal_pinch | none | 0.8584 [0.6061, 0.991] (n=6) | 19.16 [10.3, 28.51] (n=6) | 14.17 [5.825, 23.5] (n=6) | 0.9375 [0.8438, 0.9948] (n=6) |
| G_FULL | DEFAULT_uniform | opposition | aware | 0.7612 [0.4442, 0.932] (n=6) | 2.318 [1.651, 3.151] (n=6) | 9.833 [6.5, 13] (n=6) | 0.7031 [0.4948, 0.8543] (n=6) |
| G_FULL | DEFAULT_uniform | opposition | none | 0.6111 [0.2778, 0.9444] (n=6) | 2.016 [1.354, 2.839] (n=6) | 7.5 [0.3333, 18.5] (n=6) | 0.651 [0.5415, 0.7552] (n=6) |
| G_FULL | DEFAULT_uniform | reach_log | aware | 0.4319 [0.3601, 0.497] (n=6) | 4.161 [3.625, 4.698] (n=6) | 29.83 [22.33, 36.67] (n=6) | 0.901 [0.849, 0.9531] (n=6) |
| G_FULL | DEFAULT_uniform | reach_log | none | 0.3796 [0.2972, 0.4686] (n=6) | 9.328 [6.615, 12.1] (n=6) | 25.17 [22.67, 27.83] (n=6) | 0.9688 [0.9479, 0.9896] (n=6) |
| G_FULL | UNION_weighted | antipodal_pinch | aware | 0.8653 [0.7903, 0.9252] (n=6) | 3.839 [2, 6.203] (n=6) | 11.67 [6.667, 17] (n=6) | 0.7083 [0.5677, 0.8333] (n=6) |
| G_FULL | UNION_weighted | antipodal_pinch | none | 0.8247 [0.4942, 0.9946] (n=6) | 7.188 [3.984, 10.86] (n=6) | 9.667 [2.667, 17.5] (n=6) | 0.8698 [0.8073, 0.9323] (n=6) |
| G_FULL | UNION_weighted | opposition | aware | 0.661 [0.3383, 0.9392] (n=6) | 3.505 [1.667, 6.682] (n=6) | 11.33 [4.5, 19.67] (n=6) | 0.8542 [0.8125, 0.8958] (n=6) |
| G_FULL | UNION_weighted | opposition | none | 1 [1, 1] (n=6) | 3.99 [2.495, 5.974] (n=6) | 15.67 [9.333, 23.67] (n=6) | 0.9062 [0.8802, 0.9271] (n=6) |
| G_FULL | UNION_weighted | reach_log | aware | 0.356 [0.309, 0.3973] (n=6) | 3.094 [2.443, 3.839] (n=6) | 33.33 [28.5, 38.17] (n=6) | 0.9115 [0.8594, 0.9583] (n=6) |
| G_FULL | UNION_weighted | reach_log | none | 0.3117 [0.2246, 0.4201] (n=6) | 8.594 [6.703, 9.984] (n=6) | 25.17 [19.83, 30.67] (n=6) | 0.9167 [0.8594, 0.9635] (n=6) |
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | aware | 0.8631 [0.8265, 0.896] (n=6) | 3.932 [3.156, 4.729] (n=6) | 19.5 [13.83, 26.68] (n=6) | 0.8281 [0.7552, 0.901] (n=6) |
| G_FULL_INS | DEFAULT_uniform | antipodal_pinch | none | 0.7273 [0.3981, 0.9849] (n=6) | 6.578 [3.297, 9.76] (n=6) | 3.833 [1.5, 6.167] (n=6) | 0.8854 [0.7811, 0.9688] (n=6) |
| G_FULL_INS | DEFAULT_uniform | opposition | aware | 0.7765 [0.4538, 0.942] (n=6) | 1.833 [1.5, 2] (n=6) | 4 [1.833, 6.167] (n=6) | 0.6719 [0.5833, 0.7605] (n=6) |
| G_FULL_INS | DEFAULT_uniform | opposition | none | 1 [1, 1] (n=6) | 2.948 [2, 4.479] (n=6) | 9.5 [2.667, 20.33] (n=6) | 0.6875 [0.5521, 0.8229] (n=6) |
| G_FULL_INS | DEFAULT_uniform | reach_log | aware | 0.3565 [0.2957, 0.4174] (n=6) | 3.583 [2.984, 4.042] (n=6) | 22.83 [17.67, 29.67] (n=6) | 0.8698 [0.7969, 0.9323] (n=6) |
| G_FULL_INS | DEFAULT_uniform | reach_log | none | 0.3847 [0.3394, 0.4224] (n=6) | 7.156 [5.755, 8.761] (n=6) | 21.67 [17.83, 26.17] (n=6) | 0.9062 [0.8542, 0.9375] (n=6) |
| G_FULL_INS | UNION_weighted | antipodal_pinch | aware | 0.8383 [0.7419, 0.922] (n=6) | 5.089 [2.255, 8.422] (n=6) | 14 [8.333, 19.17] (n=6) | 0.6406 [0.474, 0.7865] (n=6) |
| G_FULL_INS | UNION_weighted | antipodal_pinch | none | 0.9929 [0.9894, 0.9964] (n=6) | 10.44 [5.667, 16.23] (n=6) | 10 [3.833, 17.17] (n=6) | 0.901 [0.8542, 0.9427] (n=6) |
| G_FULL_INS | UNION_weighted | opposition | aware | 0.9392 [0.9358, 0.9432] (n=6) | 2 [2, 2] (n=6) | 6 [2.333, 9.833] (n=6) | 0.8125 [0.6458, 0.9115] (n=6) |
| G_FULL_INS | UNION_weighted | opposition | none | 0.9111 [0.7333, 1] (n=6) | 4.318 [2, 8.286] (n=6) | 7.5 [1.667, 15] (n=6) | 0.8958 [0.8594, 0.9323] (n=6) |
| G_FULL_INS | UNION_weighted | reach_log | aware | 0.2969 [0.1994, 0.394] (n=6) | 2.932 [2.354, 3.516] (n=6) | 27 [22, 32] (n=6) | 0.8385 [0.7969, 0.875] (n=6) |
| G_FULL_INS | UNION_weighted | reach_log | none | 0.3219 [0.2438, 0.4046] (n=6) | 7.75 [5.688, 9.719] (n=6) | 28.5 [19.66, 36.17] (n=6) | 0.8542 [0.776, 0.9271] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | aware | 0.5684 [0.2647, 0.872] (n=6) | 2.63 [1.552, 3.849] (n=6) | 15.17 [10.33, 19.83] (n=6) | 0.6406 [0.526, 0.7396] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | antipodal_pinch | none | 0.5852 [0.2743, 0.8919] (n=6) | 3.453 [1.974, 4.823] (n=6) | 15.17 [3.667, 26.83] (n=6) | 0.7552 [0.6249, 0.875] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | opposition | aware | 0.7713 [0.4493, 0.94] (n=6) | 2.25 [1.667, 2.76] (n=6) | 8.167 [0.8333, 16] (n=6) | 0.7083 [0.6042, 0.8177] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | opposition | none | 0.8333 [0.5, 1] (n=6) | 2.161 [1.661, 2.661] (n=6) | 0.3333 [0, 0.6667] (n=6) | 0.7344 [0.651, 0.8177] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | reach_log | aware | 0.3823 [0.3283, 0.448] (n=6) | 3.896 [3.521, 4.328] (n=6) | 26.83 [24.33, 29.5] (n=6) | 0.8854 [0.8438, 0.9375] (n=6) |
| G_NOBRANCH | DEFAULT_uniform | reach_log | none | 0.4601 [0.4034, 0.5087] (n=6) | 5.297 [4.891, 5.677] (n=6) | 20.83 [14, 29.33] (n=6) | 0.9167 [0.8646, 0.9635] (n=6) |
| G_NOBRANCH | UNION_weighted | antipodal_pinch | aware | 0.2725 [-0.028, 0.5907] (n=6) | 1.786 [1, 3.005] (n=6) | 25.17 [18.17, 32.83] (n=6) | 0.7865 [0.7031, 0.8854] (n=6) |
| G_NOBRANCH | UNION_weighted | antipodal_pinch | none | 0.3258 [0, 0.658] (n=6) | 1.635 [1, 2.281] (n=6) | 10.33 [0, 22.67] (n=6) | 0.8542 [0.8125, 0.8958] (n=6) |
| G_NOBRANCH | UNION_weighted | opposition | aware | 0.6113 [0.2897, 0.932] (n=6) | 1.667 [1.333, 2] (n=6) | 2.5 [0.5, 4.5] (n=6) | 0.8438 [0.75, 0.9271] (n=6) |
| G_NOBRANCH | UNION_weighted | opposition | none | 1 [1, 1] (n=6) | 2 [2, 2] (n=6) | 5.333 [0.1667, 12] (n=6) | 0.8594 [0.8229, 0.8906] (n=6) |
| G_NOBRANCH | UNION_weighted | reach_log | aware | 0.3404 [0.2597, 0.4275] (n=6) | 3.068 [2.573, 3.557] (n=6) | 29.33 [22.67, 35.17] (n=6) | 0.8125 [0.7396, 0.8906] (n=6) |
| G_NOBRANCH | UNION_weighted | reach_log | none | 0.4581 [0.3527, 0.5539] (n=6) | 5.214 [4.214, 6.271] (n=6) | 35.5 [33.83, 37.33] (n=6) | 0.9167 [0.8802, 0.9531] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | aware | 0.8753 [0.8451, 0.9029] (n=6) | 3.5 [2.677, 4.286] (n=6) | 9.833 [4.167, 17] (n=6) | 0.7083 [0.5469, 0.8438] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | antipodal_pinch | none | 0.8871 [0.7358, 0.9768] (n=6) | 5.495 [3.818, 6.745] (n=6) | 18.5 [11.83, 25.83] (n=6) | 0.901 [0.8229, 0.974] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | opposition | aware | 0.9277 [0.9172, 0.9362] (n=6) | 2.125 [2, 2.312] (n=6) | 0.3333 [0, 0.6667] (n=6) | 0.8385 [0.8021, 0.8646] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | opposition | none | 0.8333 [0.5, 1] (n=6) | 1.833 [1.5, 2] (n=6) | 0.5 [0, 1.167] (n=6) | 0.7188 [0.6094, 0.8229] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | reach_log | aware | 0.2984 [0.2379, 0.3506] (n=6) | 3.307 [2.667, 4.011] (n=6) | 24.17 [19, 29.17] (n=6) | 0.8125 [0.7656, 0.849] (n=6) |
| G_NOBRANCH_INS | DEFAULT_uniform | reach_log | none | 0.4001 [0.3078, 0.4877] (n=6) | 6.443 [5.37, 7.958] (n=6) | 30.33 [26.5, 33.5] (n=6) | 0.8802 [0.8021, 0.9479] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | antipodal_pinch | aware | 0.8943 [0.8532, 0.9247] (n=6) | 2.714 [2.281, 3.146] (n=6) | 15.83 [9, 25.5] (n=6) | 0.7135 [0.5521, 0.8385] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | antipodal_pinch | none | 0.9843 [0.9678, 0.9947] (n=6) | 5.208 [3.972, 6.458] (n=6) | 14.33 [8, 23.83] (n=6) | 0.9427 [0.9062, 0.974] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | opposition | aware | 0.9285 [0.9092, 0.9448] (n=6) | 2.333 [2, 2.667] (n=6) | 0.8333 [0, 2.5] (n=6) | 0.8594 [0.8021, 0.9167] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | opposition | none | 1 [1, 1] (n=6) | 2.167 [2, 2.5] (n=6) | 1.833 [0, 4] (n=6) | 0.8125 [0.7031, 0.9167] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | reach_log | aware | 0.2996 [0.2343, 0.3561] (n=6) | 2.714 [2.359, 3.068] (n=6) | 32.67 [29, 36.17] (n=6) | 0.875 [0.8542, 0.8958] (n=6) |
| G_NOBRANCH_INS | UNION_weighted | reach_log | none | 0.447 [0.3632, 0.5309] (n=6) | 4.151 [3.448, 4.922] (n=6) | 31.67 [28, 35] (n=6) | 0.8281 [0.7812, 0.8646] (n=6) |

## Table 2: construct usage at generation 40, per dist x pool (averaged over fitness/cost, mean +/- 95% bootstrap CI)

| dist | pool | frac palm joint | frac branch | frac coupling | frac prismatic |
|---|---|---|---|---|---|
| G_FULL | DEFAULT_uniform | 0.1398 [0.03989, 0.25] (n=36) | 0.3663 [0.2257, 0.5125] (n=36) | 0.3455 [0.2188, 0.4748] (n=36) | 0.401 [0.2682, 0.5278] (n=36) |
| G_FULL | UNION_weighted | 0.2326 [0.1111, 0.3689] (n=36) | 0.3932 [0.2352, 0.5556] (n=36) | 0.2092 [0.09288, 0.3342] (n=36) | 0.3307 [0.1953, 0.4792] (n=36) |
| G_FULL_INS | DEFAULT_uniform | 0.197 [0.08073, 0.3229] (n=36) | 0.1823 [0.07724, 0.3099] (n=36) | 0.1684 [0.08765, 0.2578] (n=36) | 0.3932 [0.2725, 0.5226] (n=36) |
| G_FULL_INS | UNION_weighted | 0.1927 [0.0816, 0.323] (n=36) | 0.3333 [0.1944, 0.5] (n=36) | 0.1849 [0.06944, 0.309] (n=36) | 0.4366 [0.2778, 0.6007] (n=36) |
| G_NOBRANCH | DEFAULT_uniform | 0.08073 [0.001736, 0.1832] (n=36) | 0 [0, 0] (n=36) | 0.2413 [0.1328, 0.3568] (n=36) | 0.3021 [0.1892, 0.4202] (n=36) |
| G_NOBRANCH | UNION_weighted | 0.09288 [0.02083, 0.1858] (n=36) | 0 [0, 0] (n=36) | 0.1476 [0.04427, 0.2622] (n=36) | 0.1745 [0.07289, 0.2917] (n=36) |
| G_NOBRANCH_INS | DEFAULT_uniform | 0.1693 [0.0599, 0.2856] (n=36) | 0 [0, 0] (n=36) | 0.2144 [0.1094, 0.336] (n=36) | 0.4262 [0.3081, 0.546] (n=36) |
| G_NOBRANCH_INS | UNION_weighted | 0.08854 [0.01994, 0.178] (n=36) | 0 [0, 0] (n=36) | 0.1892 [0.08244, 0.3108] (n=36) | 0.3203 [0.1831, 0.4601] (n=36) |

## Table 3: per-operator survival under UNION_weighted (pooled over dist x fitness x cost x restart; Wilson 95% CI)

| operator | survival rate | 95% CI lo | 95% CI hi | n offspring |
|---|---|---|---|---|
| add_digit | 0.09676 | 0.08936 | 0.1047 | 5705 |
| add_minimal_digit | 0.1352 | 0.1278 | 0.1429 | 7870 |
| add_palm_body | 0.1476 | 0.1398 | 0.1558 | 7513 |
| delete_phalanx | 0.2863 | 0.2726 | 0.3004 | 4083 |
| insert_phalanx | 0.1569 | 0.1489 | 0.1652 | 7656 |
| perturb_parameter | 0.4763 | 0.4633 | 0.4894 | 5599 |
| regrow_subtree | 0.08746 | 0.08035 | 0.09513 | 5614 |
| remove_digit | 0.1793 | 0.1662 | 0.1932 | 3106 |
| remove_digit_minimal | 0.1893 | 0.1775 | 0.2017 | 4015 |
| remove_palm_body | 0.3652 | 0.3487 | 0.382 | 3209 |
| resample_parameter | 0.2325 | 0.227 | 0.2381 | 22200 |
| step_axis | 0.4424 | 0.4356 | 0.4492 | 20393 |
| step_coupling | 0.4907 | 0.4722 | 0.5092 | 2804 |
| step_limits | 0.4222 | 0.4154 | 0.4291 | 19849 |
| step_mount | 0.4065 | 0.3998 | 0.4132 | 20497 |
| step_radius | 0.4766 | 0.4697 | 0.4834 | 20350 |
| step_root_length | 0.4367 | 0.4299 | 0.4435 | 20669 |
| toggle_palm_joint | 0.3143 | 0.2984 | 0.3306 | 3188 |

## Reading

- G_FULL::DEFAULT_uniform::antipodal_pinch::aware: final best fitness 0.7776 [0.6312, 0.8791] (n=6); final mean motors 6.224 [3.411, 10.34] (n=6); gen90 22.33 [14, 30.84] (n=6); diversity@40 0.849 [0.7448, 0.9375] (n=6).
- G_FULL::DEFAULT_uniform::antipodal_pinch::none: final best fitness 0.8584 [0.6061, 0.991] (n=6); final mean motors 19.16 [10.3, 28.51] (n=6); gen90 14.17 [5.825, 23.5] (n=6); diversity@40 0.9375 [0.8438, 0.9948] (n=6).
- G_FULL::DEFAULT_uniform::opposition::aware: final best fitness 0.7612 [0.4442, 0.932] (n=6); final mean motors 2.318 [1.651, 3.151] (n=6); gen90 9.833 [6.5, 13] (n=6); diversity@40 0.7031 [0.4948, 0.8543] (n=6).
- G_FULL::DEFAULT_uniform::opposition::none: final best fitness 0.6111 [0.2778, 0.9444] (n=6); final mean motors 2.016 [1.354, 2.839] (n=6); gen90 7.5 [0.3333, 18.5] (n=6); diversity@40 0.651 [0.5415, 0.7552] (n=6).
- G_FULL::DEFAULT_uniform::reach_log::aware: final best fitness 0.4319 [0.3601, 0.497] (n=6); final mean motors 4.161 [3.625, 4.698] (n=6); gen90 29.83 [22.33, 36.67] (n=6); diversity@40 0.901 [0.849, 0.9531] (n=6).
- G_FULL::DEFAULT_uniform::reach_log::none: final best fitness 0.3796 [0.2972, 0.4686] (n=6); final mean motors 9.328 [6.615, 12.1] (n=6); gen90 25.17 [22.67, 27.83] (n=6); diversity@40 0.9688 [0.9479, 0.9896] (n=6).
- G_FULL::UNION_weighted::antipodal_pinch::aware: final best fitness 0.8653 [0.7903, 0.9252] (n=6); final mean motors 3.839 [2, 6.203] (n=6); gen90 11.67 [6.667, 17] (n=6); diversity@40 0.7083 [0.5677, 0.8333] (n=6).
- G_FULL::UNION_weighted::antipodal_pinch::none: final best fitness 0.8247 [0.4942, 0.9946] (n=6); final mean motors 7.188 [3.984, 10.86] (n=6); gen90 9.667 [2.667, 17.5] (n=6); diversity@40 0.8698 [0.8073, 0.9323] (n=6).
- G_FULL::UNION_weighted::opposition::aware: final best fitness 0.661 [0.3383, 0.9392] (n=6); final mean motors 3.505 [1.667, 6.682] (n=6); gen90 11.33 [4.5, 19.67] (n=6); diversity@40 0.8542 [0.8125, 0.8958] (n=6).
- G_FULL::UNION_weighted::opposition::none: final best fitness 1 [1, 1] (n=6); final mean motors 3.99 [2.495, 5.974] (n=6); gen90 15.67 [9.333, 23.67] (n=6); diversity@40 0.9062 [0.8802, 0.9271] (n=6).
- G_FULL::UNION_weighted::reach_log::aware: final best fitness 0.356 [0.309, 0.3973] (n=6); final mean motors 3.094 [2.443, 3.839] (n=6); gen90 33.33 [28.5, 38.17] (n=6); diversity@40 0.9115 [0.8594, 0.9583] (n=6).
- G_FULL::UNION_weighted::reach_log::none: final best fitness 0.3117 [0.2246, 0.4201] (n=6); final mean motors 8.594 [6.703, 9.984] (n=6); gen90 25.17 [19.83, 30.67] (n=6); diversity@40 0.9167 [0.8594, 0.9635] (n=6).
- G_FULL_INS::DEFAULT_uniform::antipodal_pinch::aware: final best fitness 0.8631 [0.8265, 0.896] (n=6); final mean motors 3.932 [3.156, 4.729] (n=6); gen90 19.5 [13.83, 26.68] (n=6); diversity@40 0.8281 [0.7552, 0.901] (n=6).
- G_FULL_INS::DEFAULT_uniform::antipodal_pinch::none: final best fitness 0.7273 [0.3981, 0.9849] (n=6); final mean motors 6.578 [3.297, 9.76] (n=6); gen90 3.833 [1.5, 6.167] (n=6); diversity@40 0.8854 [0.7811, 0.9688] (n=6).
- G_FULL_INS::DEFAULT_uniform::opposition::aware: final best fitness 0.7765 [0.4538, 0.942] (n=6); final mean motors 1.833 [1.5, 2] (n=6); gen90 4 [1.833, 6.167] (n=6); diversity@40 0.6719 [0.5833, 0.7605] (n=6).
- G_FULL_INS::DEFAULT_uniform::opposition::none: final best fitness 1 [1, 1] (n=6); final mean motors 2.948 [2, 4.479] (n=6); gen90 9.5 [2.667, 20.33] (n=6); diversity@40 0.6875 [0.5521, 0.8229] (n=6).
- G_FULL_INS::DEFAULT_uniform::reach_log::aware: final best fitness 0.3565 [0.2957, 0.4174] (n=6); final mean motors 3.583 [2.984, 4.042] (n=6); gen90 22.83 [17.67, 29.67] (n=6); diversity@40 0.8698 [0.7969, 0.9323] (n=6).
- G_FULL_INS::DEFAULT_uniform::reach_log::none: final best fitness 0.3847 [0.3394, 0.4224] (n=6); final mean motors 7.156 [5.755, 8.761] (n=6); gen90 21.67 [17.83, 26.17] (n=6); diversity@40 0.9062 [0.8542, 0.9375] (n=6).
- G_FULL_INS::UNION_weighted::antipodal_pinch::aware: final best fitness 0.8383 [0.7419, 0.922] (n=6); final mean motors 5.089 [2.255, 8.422] (n=6); gen90 14 [8.333, 19.17] (n=6); diversity@40 0.6406 [0.474, 0.7865] (n=6).
- G_FULL_INS::UNION_weighted::antipodal_pinch::none: final best fitness 0.9929 [0.9894, 0.9964] (n=6); final mean motors 10.44 [5.667, 16.23] (n=6); gen90 10 [3.833, 17.17] (n=6); diversity@40 0.901 [0.8542, 0.9427] (n=6).
- G_FULL_INS::UNION_weighted::opposition::aware: final best fitness 0.9392 [0.9358, 0.9432] (n=6); final mean motors 2 [2, 2] (n=6); gen90 6 [2.333, 9.833] (n=6); diversity@40 0.8125 [0.6458, 0.9115] (n=6).
- G_FULL_INS::UNION_weighted::opposition::none: final best fitness 0.9111 [0.7333, 1] (n=6); final mean motors 4.318 [2, 8.286] (n=6); gen90 7.5 [1.667, 15] (n=6); diversity@40 0.8958 [0.8594, 0.9323] (n=6).
- G_FULL_INS::UNION_weighted::reach_log::aware: final best fitness 0.2969 [0.1994, 0.394] (n=6); final mean motors 2.932 [2.354, 3.516] (n=6); gen90 27 [22, 32] (n=6); diversity@40 0.8385 [0.7969, 0.875] (n=6).
- G_FULL_INS::UNION_weighted::reach_log::none: final best fitness 0.3219 [0.2438, 0.4046] (n=6); final mean motors 7.75 [5.688, 9.719] (n=6); gen90 28.5 [19.66, 36.17] (n=6); diversity@40 0.8542 [0.776, 0.9271] (n=6).
- G_NOBRANCH::DEFAULT_uniform::antipodal_pinch::aware: final best fitness 0.5684 [0.2647, 0.872] (n=6); final mean motors 2.63 [1.552, 3.849] (n=6); gen90 15.17 [10.33, 19.83] (n=6); diversity@40 0.6406 [0.526, 0.7396] (n=6).
- G_NOBRANCH::DEFAULT_uniform::antipodal_pinch::none: final best fitness 0.5852 [0.2743, 0.8919] (n=6); final mean motors 3.453 [1.974, 4.823] (n=6); gen90 15.17 [3.667, 26.83] (n=6); diversity@40 0.7552 [0.6249, 0.875] (n=6).
- G_NOBRANCH::DEFAULT_uniform::opposition::aware: final best fitness 0.7713 [0.4493, 0.94] (n=6); final mean motors 2.25 [1.667, 2.76] (n=6); gen90 8.167 [0.8333, 16] (n=6); diversity@40 0.7083 [0.6042, 0.8177] (n=6).
- G_NOBRANCH::DEFAULT_uniform::opposition::none: final best fitness 0.8333 [0.5, 1] (n=6); final mean motors 2.161 [1.661, 2.661] (n=6); gen90 0.3333 [0, 0.6667] (n=6); diversity@40 0.7344 [0.651, 0.8177] (n=6).
- G_NOBRANCH::DEFAULT_uniform::reach_log::aware: final best fitness 0.3823 [0.3283, 0.448] (n=6); final mean motors 3.896 [3.521, 4.328] (n=6); gen90 26.83 [24.33, 29.5] (n=6); diversity@40 0.8854 [0.8438, 0.9375] (n=6).
- G_NOBRANCH::DEFAULT_uniform::reach_log::none: final best fitness 0.4601 [0.4034, 0.5087] (n=6); final mean motors 5.297 [4.891, 5.677] (n=6); gen90 20.83 [14, 29.33] (n=6); diversity@40 0.9167 [0.8646, 0.9635] (n=6).
- G_NOBRANCH::UNION_weighted::antipodal_pinch::aware: final best fitness 0.2725 [-0.028, 0.5907] (n=6); final mean motors 1.786 [1, 3.005] (n=6); gen90 25.17 [18.17, 32.83] (n=6); diversity@40 0.7865 [0.7031, 0.8854] (n=6).
- G_NOBRANCH::UNION_weighted::antipodal_pinch::none: final best fitness 0.3258 [0, 0.658] (n=6); final mean motors 1.635 [1, 2.281] (n=6); gen90 10.33 [0, 22.67] (n=6); diversity@40 0.8542 [0.8125, 0.8958] (n=6).
- G_NOBRANCH::UNION_weighted::opposition::aware: final best fitness 0.6113 [0.2897, 0.932] (n=6); final mean motors 1.667 [1.333, 2] (n=6); gen90 2.5 [0.5, 4.5] (n=6); diversity@40 0.8438 [0.75, 0.9271] (n=6).
- G_NOBRANCH::UNION_weighted::opposition::none: final best fitness 1 [1, 1] (n=6); final mean motors 2 [2, 2] (n=6); gen90 5.333 [0.1667, 12] (n=6); diversity@40 0.8594 [0.8229, 0.8906] (n=6).
- G_NOBRANCH::UNION_weighted::reach_log::aware: final best fitness 0.3404 [0.2597, 0.4275] (n=6); final mean motors 3.068 [2.573, 3.557] (n=6); gen90 29.33 [22.67, 35.17] (n=6); diversity@40 0.8125 [0.7396, 0.8906] (n=6).
- G_NOBRANCH::UNION_weighted::reach_log::none: final best fitness 0.4581 [0.3527, 0.5539] (n=6); final mean motors 5.214 [4.214, 6.271] (n=6); gen90 35.5 [33.83, 37.33] (n=6); diversity@40 0.9167 [0.8802, 0.9531] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::antipodal_pinch::aware: final best fitness 0.8753 [0.8451, 0.9029] (n=6); final mean motors 3.5 [2.677, 4.286] (n=6); gen90 9.833 [4.167, 17] (n=6); diversity@40 0.7083 [0.5469, 0.8438] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::antipodal_pinch::none: final best fitness 0.8871 [0.7358, 0.9768] (n=6); final mean motors 5.495 [3.818, 6.745] (n=6); gen90 18.5 [11.83, 25.83] (n=6); diversity@40 0.901 [0.8229, 0.974] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::opposition::aware: final best fitness 0.9277 [0.9172, 0.9362] (n=6); final mean motors 2.125 [2, 2.312] (n=6); gen90 0.3333 [0, 0.6667] (n=6); diversity@40 0.8385 [0.8021, 0.8646] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::opposition::none: final best fitness 0.8333 [0.5, 1] (n=6); final mean motors 1.833 [1.5, 2] (n=6); gen90 0.5 [0, 1.167] (n=6); diversity@40 0.7188 [0.6094, 0.8229] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::reach_log::aware: final best fitness 0.2984 [0.2379, 0.3506] (n=6); final mean motors 3.307 [2.667, 4.011] (n=6); gen90 24.17 [19, 29.17] (n=6); diversity@40 0.8125 [0.7656, 0.849] (n=6).
- G_NOBRANCH_INS::DEFAULT_uniform::reach_log::none: final best fitness 0.4001 [0.3078, 0.4877] (n=6); final mean motors 6.443 [5.37, 7.958] (n=6); gen90 30.33 [26.5, 33.5] (n=6); diversity@40 0.8802 [0.8021, 0.9479] (n=6).
- G_NOBRANCH_INS::UNION_weighted::antipodal_pinch::aware: final best fitness 0.8943 [0.8532, 0.9247] (n=6); final mean motors 2.714 [2.281, 3.146] (n=6); gen90 15.83 [9, 25.5] (n=6); diversity@40 0.7135 [0.5521, 0.8385] (n=6).
- G_NOBRANCH_INS::UNION_weighted::antipodal_pinch::none: final best fitness 0.9843 [0.9678, 0.9947] (n=6); final mean motors 5.208 [3.972, 6.458] (n=6); gen90 14.33 [8, 23.83] (n=6); diversity@40 0.9427 [0.9062, 0.974] (n=6).
- G_NOBRANCH_INS::UNION_weighted::opposition::aware: final best fitness 0.9285 [0.9092, 0.9448] (n=6); final mean motors 2.333 [2, 2.667] (n=6); gen90 0.8333 [0, 2.5] (n=6); diversity@40 0.8594 [0.8021, 0.9167] (n=6).
- G_NOBRANCH_INS::UNION_weighted::opposition::none: final best fitness 1 [1, 1] (n=6); final mean motors 2.167 [2, 2.5] (n=6); gen90 1.833 [0, 4] (n=6); diversity@40 0.8125 [0.7031, 0.9167] (n=6).
- G_NOBRANCH_INS::UNION_weighted::reach_log::aware: final best fitness 0.2996 [0.2343, 0.3561] (n=6); final mean motors 2.714 [2.359, 3.068] (n=6); gen90 32.67 [29, 36.17] (n=6); diversity@40 0.875 [0.8542, 0.8958] (n=6).
- G_NOBRANCH_INS::UNION_weighted::reach_log::none: final best fitness 0.447 [0.3632, 0.5309] (n=6); final mean motors 4.151 [3.448, 4.922] (n=6); gen90 31.67 [28, 35] (n=6); diversity@40 0.8281 [0.7812, 0.8646] (n=6).

## Caveats

- ``opposition``/``antipodal_pinch``/``reach_coverage``/``structural_cost`` are cheap geometric DIAGNOSTICS (see proxy.py's own module docstring), never a real manipulation task objective -- a high score under any of these fitnesses says nothing about grasp quality, only about what the grammar's operators can cheaply push a proxy toward.
- Fitness is evaluated with the SAME seed for every individual within one generation but a DIFFERENT seed across generations (`proxy.all_proxies(model, seed=generation_seed, n_configs=16)`), so within-generation comparisons (selection) are apples-to-apples but the fitness trajectory across generations is a noisy estimate, not a monotone objective; mu survivors are re-evaluated under each new generation's seed rather than keeping a stale fitness value (see the module docstring's noise-model note).
- 6 restarts is a small sample for the bootstrap CIs reported here; treat table 1/2 CIs as indicative, not as a claim of tight precision.
