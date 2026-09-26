# Experiment: e1_locality

Wall time: 62.555 s

## Params

```json
{
  "n_configs": 32
}
```

n_seeds: 1000

## Table

| operator | applicability_rate | null_rate | tip_disp median | tip_disp p90 | |joint_count_delta| mean | frac_joints_changed median | frac_joints_changed p90 | motor_delta mean | total_length_delta median |
|---|---|---|---|---|---|---|---|---|---|
| resample_parameter | 1 | 0 | 0.01366 | 0.09666 | 0 | 0.06061 | 0.25 | 0.17 | 0.015 |
| perturb_parameter | 1 | 0 | 0.0008333 | 0.002857 | 0 | 0.02778 | 0.1111 | 0 | 0.005 |
| regrow_subtree | 1 | 0 | 0.03717 | 0.1204 | 5.706 | 0.2727 | 0.75 | 4.844 | 0.155 |
| insert_phalanx | 0.988 | 0 | 0.008571 | 0.0325 | 1 | 0.0625 | 0.25 | 0.8573 | 0.05 |
| delete_phalanx | 0.972 | 0 | 0.01267 | 0.05343 | 1 | 0.07143 | 0.2222 | 0.7582 | 0.045 |
| add_digit | 0.823 | 0 | 0.01 | 0.03 | 6.809 | 0.1746 | 0.5447 | 5.9 | 0.195 |
| remove_digit | 0.854 | 0 | 0.0114 | 0.03 | 7.625 | 0.2 | 0.5876 | 6.534 | 0.21 |
| step_axis | 1 | 0.049 | 0.001714 | 0.009124 | 0 | 0.03846 | 0.1667 | 0 | 0 |
| step_limits | 0.998 | 0 | 0.0003492 | 0.002661 | 0 | 0.04762 | 0.1696 | 0 | 0 |
| step_mount | 1 | 0 | 0.004062 | 0.01952 | 0 | 0.04 | 0.1667 | 0 | 0 |
| step_coupling | 0.841 | 0 | 0.001477 | 0.009036 | 0 | 0.03448 | 0.09091 | 0 | 0 |
| step_root_length | 1 | 0 | 0.0025 | 0.004531 | 0 | 0.09524 | 0.2857 | 0 | 0.005 |
| step_radius | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Reading

- resample_parameter: applicable 1, null 0, fraction_joints_changed median 0.06061 (p90 0.25), tip_displacement_m median 0.01366 (p90 0.09666).
- perturb_parameter: applicable 1, null 0, fraction_joints_changed median 0.02778 (p90 0.1111), tip_displacement_m median 0.0008333 (p90 0.002857).
- regrow_subtree: applicable 1, null 0, fraction_joints_changed median 0.2727 (p90 0.75), tip_displacement_m median 0.03717 (p90 0.1204).
- insert_phalanx: applicable 0.988, null 0, fraction_joints_changed median 0.0625 (p90 0.25), tip_displacement_m median 0.008571 (p90 0.0325).
- delete_phalanx: applicable 0.972, null 0, fraction_joints_changed median 0.07143 (p90 0.2222), tip_displacement_m median 0.01267 (p90 0.05343).
- add_digit: applicable 0.823, null 0, fraction_joints_changed median 0.1746 (p90 0.5447), tip_displacement_m median 0.01 (p90 0.03).
- remove_digit: applicable 0.854, null 0, fraction_joints_changed median 0.2 (p90 0.5876), tip_displacement_m median 0.0114 (p90 0.03).
- step_axis: applicable 1, null 0.049, fraction_joints_changed median 0.03846 (p90 0.1667), tip_displacement_m median 0.001714 (p90 0.009124).
- step_limits: applicable 0.998, null 0, fraction_joints_changed median 0.04762 (p90 0.1696), tip_displacement_m median 0.0003492 (p90 0.002661).
- step_mount: applicable 1, null 0, fraction_joints_changed median 0.04 (p90 0.1667), tip_displacement_m median 0.004062 (p90 0.01952).
- step_coupling: applicable 0.841, null 0, fraction_joints_changed median 0.03448 (p90 0.09091), tip_displacement_m median 0.001477 (p90 0.009036).
- step_root_length: applicable 1, null 0, fraction_joints_changed median 0.09524 (p90 0.2857), tip_displacement_m median 0.0025 (p90 0.004531).
- step_radius: applicable 1, null 0, fraction_joints_changed median 0 (p90 0), tip_displacement_m median 0 (p90 0).

## Footnote: aligned vs. legacy (pre-I14) tip_displacement_m

- insert_phalanx (n=30 seeds): aligned (I14) median 0.004722 m; legacy unaligned (pre-I14, independently-sampled parent/child configs -- see ``_legacy_unaligned_tip_displacement``) median 0.09925 m.
