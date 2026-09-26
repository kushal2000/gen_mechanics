# Experiment: e3_reach

Wall time: 2.891 s

## Params

```json
{
  "max_accepted": 200,
  "max_attempts_per_step": 8,
  "safety_cap": 4000
}
```

n_seeds (restarts): 64

## Table: target x operator set

| target | operator_set | success_rate | 95% CI lo | 95% CI hi | median accepted @ success | median proposed @ success | n |
|---|---|---|---|---|---|---|---|
| anthropomorphic_staggered | DEFAULT | 1 | 1 | 1 | 39.5 | 49 | 64 |
| anthropomorphic_staggered | DEFAULT+SMALL | 0.9062 | 0.8281 | 0.9688 | 76.5 | 91.5 | 64 |
| anthropomorphic_staggered | MINIMAL | 0.1406 | 0.0625 | 0.2344 | 165 | 227 | 64 |
| radial_3 | DEFAULT | 0.9688 | 0.9219 | 1 | 48.5 | 64.5 | 64 |
| radial_3 | DEFAULT+SMALL | 0.8125 | 0.7188 | 0.9062 | 64 | 80 | 64 |
| radial_3 | MINIMAL | 0.5156 | 0.3906 | 0.6406 | 113 | 152 | 64 |
| prismatic_gripper | DEFAULT | 1 | 1 | 1 | 29 | 36.5 | 64 |
| prismatic_gripper | DEFAULT+SMALL | 0.9844 | 0.9531 | 1 | 39 | 50 | 64 |
| prismatic_gripper | MINIMAL | 0.6094 | 0.4844 | 0.7344 | 82 | 139 | 64 |
| arch_palm | DEFAULT | 0 | 0 | 0 | n/a | n/a | 64 |
| arch_palm | DEFAULT+SMALL | 0 | 0 | 0 | n/a | n/a | 64 |
| arch_palm | MINIMAL | 1 | 1 | 1 | 28 | 40 | 64 |

## Reading

- anthropomorphic_staggered / DEFAULT: success_rate 1 [1, 1] (n=64); median accepted @ success 39.5; median proposed @ success 49.
- anthropomorphic_staggered / DEFAULT+SMALL: success_rate 0.9062 [0.8281, 0.9688] (n=64); median accepted @ success 76.5; median proposed @ success 91.5.
- anthropomorphic_staggered / MINIMAL: success_rate 0.1406 [0.0625, 0.2344] (n=64); median accepted @ success 165; median proposed @ success 227.
- radial_3 / DEFAULT: success_rate 0.9688 [0.9219, 1] (n=64); median accepted @ success 48.5; median proposed @ success 64.5.
- radial_3 / DEFAULT+SMALL: success_rate 0.8125 [0.7188, 0.9062] (n=64); median accepted @ success 64; median proposed @ success 80.
- radial_3 / MINIMAL: success_rate 0.5156 [0.3906, 0.6406] (n=64); median accepted @ success 113; median proposed @ success 152.
- prismatic_gripper / DEFAULT: success_rate 1 [1, 1] (n=64); median accepted @ success 29; median proposed @ success 36.5.
- prismatic_gripper / DEFAULT+SMALL: success_rate 0.9844 [0.9531, 1] (n=64); median accepted @ success 39; median proposed @ success 50.
- prismatic_gripper / MINIMAL: success_rate 0.6094 [0.4844, 0.7344] (n=64); median accepted @ success 82; median proposed @ success 139.
- arch_palm / DEFAULT: success_rate 0 [0, 0] (n=64); median accepted @ success n/a; median proposed @ success n/a.
- arch_palm / DEFAULT+SMALL: success_rate 0 [0, 0] (n=64); median accepted @ success n/a; median proposed @ success n/a.
- arch_palm / MINIMAL: success_rate 1 [1, 1] (n=64); median accepted @ success 28; median proposed @ success 40.
