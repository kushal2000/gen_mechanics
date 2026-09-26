# Experiment: e3_reach

Wall time: 10.485 s

## Params

```json
{
  "budget": 1500
}
```

n_seeds (restarts): 64

Budget is in PROPOSALS (evaluations), not accepted moves. success_rate uses a Wilson score interval. median_proposed is censored at the budget for restarts that never succeeded (reported '>= budget' via is_lower_bound/censored_fraction when that applies).

## Table: target x operator pool x dist

| target | pool | dist | success_rate | 95% CI lo | 95% CI hi | median proposed (censored) | is_lower_bound | censored_fraction | n |
|---|---|---|---|---|---|---|---|---|---|
| anthropomorphic_staggered | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 49 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 43 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 78.5 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 80.5 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_FULL | 1 | 0.9434 | 1 | 142.5 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 107 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 191.5 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 211 | False | 0 | 64 |
| radial_3 | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 66 | False | 0 | 64 |
| radial_3 | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 59.5 | False | 0 | 64 |
| radial_3 | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 52 | False | 0 | 64 |
| radial_3 | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 63 | False | 0 | 64 |
| radial_3 | UNION | G_FULL | 1 | 0.9434 | 1 | 123.5 | False | 0 | 64 |
| radial_3 | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 110 | False | 0 | 64 |
| radial_3 | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 103 | False | 0 | 64 |
| radial_3 | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 114.5 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 36.5 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 26 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 31.5 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 30 | False | 0 | 64 |
| prismatic_gripper | UNION | G_FULL | 1 | 0.9434 | 1 | 101 | False | 0 | 64 |
| prismatic_gripper | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 71.5 | False | 0 | 64 |
| prismatic_gripper | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 81 | False | 0 | 64 |
| prismatic_gripper | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 82 | False | 0 | 64 |
| arch_palm | DEFAULT | G_FULL | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_NOBRANCH | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_FULL_INS | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_NOBRANCH_INS | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | UNION | G_FULL | 1 | 0.9434 | 1 | 55 | False | 0 | 64 |
| arch_palm | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 46 | False | 0 | 64 |
| arch_palm | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 52 | False | 0 | 64 |
| arch_palm | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 52 | False | 0 | 64 |

## Reading

- anthropomorphic_staggered / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 49 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 43 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 78.5 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 80.5 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 142.5 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 107 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 191.5 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 211 (censored_fraction 0).
- radial_3 / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 66 (censored_fraction 0).
- radial_3 / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 59.5 (censored_fraction 0).
- radial_3 / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 52 (censored_fraction 0).
- radial_3 / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 63 (censored_fraction 0).
- radial_3 / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 123.5 (censored_fraction 0).
- radial_3 / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 110 (censored_fraction 0).
- radial_3 / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 103 (censored_fraction 0).
- radial_3 / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 114.5 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 36.5 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 26 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 31.5 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 30 (censored_fraction 0).
- prismatic_gripper / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 101 (censored_fraction 0).
- prismatic_gripper / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 71.5 (censored_fraction 0).
- prismatic_gripper / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 81 (censored_fraction 0).
- prismatic_gripper / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 82 (censored_fraction 0).
- arch_palm / DEFAULT / G_FULL: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_NOBRANCH: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_FULL_INS: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_NOBRANCH_INS: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 55 (censored_fraction 0).
- arch_palm / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 46 (censored_fraction 0).
- arch_palm / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 52 (censored_fraction 0).
- arch_palm / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 52 (censored_fraction 0).
