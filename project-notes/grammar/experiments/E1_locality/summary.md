# Experiment: e1_locality

Wall time: 57.148 s

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
| resample_parameter | 1 | 0 | 0.0244 | 0.1355 | 0 | 0.06061 | 0.25 | 0.17 | 0.015 |
| perturb_parameter | 1 | 0 | 0.0008333 | 0.002857 | 0 | 0.02778 | 0.1111 | 0 | 0.005 |
| regrow_subtree | 1 | 0 | 0.1035 | 0.1659 | 5.706 | 0.2727 | 0.75 | 4.844 | 0.155 |
| insert_phalanx | 0.988 | 0 | 0.106 | 0.1594 | 1 | 0.0625 | 0.25 | 0.8573 | 0.05 |
| delete_phalanx | 0.972 | 0 | 0.09243 | 0.1506 | 1 | 0.06667 | 0.2222 | 0.7531 | 0.045 |
| add_digit | 0.823 | 0 | 0.07729 | 0.1149 | 6.809 | 0.1746 | 0.5447 | 5.9 | 0.195 |
| remove_digit | 0.854 | 0 | 0.07688 | 0.1235 | 7.625 | 0.2 | 0.5876 | 6.534 | 0.21 |
| step_axis | 1 | 0.008 | 0.001714 | 0.009124 | 0 | 0.03846 | 0.1667 | 0 | 0 |
| step_limits | 0.998 | 0 | 0.003854 | 0.02057 | 0 | 0.04762 | 0.1696 | 0 | 0 |
| step_mount | 1 | 0 | 0.004062 | 0.01952 | 0 | 0.04 | 0.1667 | 0 | 0 |
| step_length | 1 | 0 | 0.0008333 | 0.003 | 0 | 0.02632 | 0.1111 | 0 | 0.005 |
| step_coupling | 0.841 | 0 | 0.002859 | 0.02063 | 0 | 0.03448 | 0.09091 | 0 | 0 |

## Reading

- resample_parameter: applicable 1, null 0, fraction_joints_changed median 0.06061 (p90 0.25), tip_displacement_m median 0.0244 (p90 0.1355).
- perturb_parameter: applicable 1, null 0, fraction_joints_changed median 0.02778 (p90 0.1111), tip_displacement_m median 0.0008333 (p90 0.002857).
- regrow_subtree: applicable 1, null 0, fraction_joints_changed median 0.2727 (p90 0.75), tip_displacement_m median 0.1035 (p90 0.1659).
- insert_phalanx: applicable 0.988, null 0, fraction_joints_changed median 0.0625 (p90 0.25), tip_displacement_m median 0.106 (p90 0.1594).
- delete_phalanx: applicable 0.972, null 0, fraction_joints_changed median 0.06667 (p90 0.2222), tip_displacement_m median 0.09243 (p90 0.1506).
- add_digit: applicable 0.823, null 0, fraction_joints_changed median 0.1746 (p90 0.5447), tip_displacement_m median 0.07729 (p90 0.1149).
- remove_digit: applicable 0.854, null 0, fraction_joints_changed median 0.2 (p90 0.5876), tip_displacement_m median 0.07688 (p90 0.1235).
- step_axis: applicable 1, null 0.008, fraction_joints_changed median 0.03846 (p90 0.1667), tip_displacement_m median 0.001714 (p90 0.009124).
- step_limits: applicable 0.998, null 0, fraction_joints_changed median 0.04762 (p90 0.1696), tip_displacement_m median 0.003854 (p90 0.02057).
- step_mount: applicable 1, null 0, fraction_joints_changed median 0.04 (p90 0.1667), tip_displacement_m median 0.004062 (p90 0.01952).
- step_length: applicable 1, null 0, fraction_joints_changed median 0.02632 (p90 0.1111), tip_displacement_m median 0.0008333 (p90 0.003).
- step_coupling: applicable 0.841, null 0, fraction_joints_changed median 0.03448 (p90 0.09091), tip_displacement_m median 0.002859 (p90 0.02063).
