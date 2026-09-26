# Experiment: e3_reach

Wall time: 33.536 s

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
| anthropomorphic_staggered | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 47 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 43 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 78.5 | False | 0 | 64 |
| anthropomorphic_staggered | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 80.5 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_FULL | 1 | 0.9434 | 1 | 160 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 129 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 242.5 | False | 0 | 64 |
| anthropomorphic_staggered | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 259 | False | 0 | 64 |
| anthropomorphic_staggered | EVOLUTION_uniform | G_FULL | 0.1406 | 0.07579 | 0.2462 | 1500 | True | 0.8594 | 64 |
| anthropomorphic_staggered | EVOLUTION_uniform | G_NOBRANCH | 0.2031 | 0.1227 | 0.3171 | 1500 | True | 0.7969 | 64 |
| anthropomorphic_staggered | EVOLUTION_uniform | G_FULL_INS | 0.1719 | 0.09878 | 0.2821 | 1500 | True | 0.8281 | 64 |
| anthropomorphic_staggered | EVOLUTION_uniform | G_NOBRANCH_INS | 0.07812 | 0.03383 | 0.1702 | 1500 | True | 0.9219 | 64 |
| radial_3 | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 62 | False | 0 | 64 |
| radial_3 | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 59.5 | False | 0 | 64 |
| radial_3 | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 52 | False | 0 | 64 |
| radial_3 | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 63 | False | 0 | 64 |
| radial_3 | UNION | G_FULL | 1 | 0.9434 | 1 | 122 | False | 0 | 64 |
| radial_3 | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 92 | False | 0 | 64 |
| radial_3 | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 132 | False | 0 | 64 |
| radial_3 | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 142.5 | False | 0 | 64 |
| radial_3 | EVOLUTION_uniform | G_FULL | 0.9219 | 0.8298 | 0.9662 | 611.5 | False | 0.07812 | 64 |
| radial_3 | EVOLUTION_uniform | G_NOBRANCH | 0.7969 | 0.6829 | 0.8773 | 640.5 | False | 0.2031 | 64 |
| radial_3 | EVOLUTION_uniform | G_FULL_INS | 0.7812 | 0.6657 | 0.865 | 729 | False | 0.2188 | 64 |
| radial_3 | EVOLUTION_uniform | G_NOBRANCH_INS | 0.75 | 0.6318 | 0.8399 | 880 | False | 0.25 | 64 |
| prismatic_gripper | DEFAULT | G_FULL | 1 | 0.9434 | 1 | 36 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_NOBRANCH | 1 | 0.9434 | 1 | 26 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_FULL_INS | 1 | 0.9434 | 1 | 31.5 | False | 0 | 64 |
| prismatic_gripper | DEFAULT | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 30 | False | 0 | 64 |
| prismatic_gripper | UNION | G_FULL | 1 | 0.9434 | 1 | 61.5 | False | 0 | 64 |
| prismatic_gripper | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 60 | False | 0 | 64 |
| prismatic_gripper | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 115.5 | False | 0 | 64 |
| prismatic_gripper | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 124.5 | False | 0 | 64 |
| prismatic_gripper | EVOLUTION_uniform | G_FULL | 0.5938 | 0.4715 | 0.7054 | 1000 | False | 0.4062 | 64 |
| prismatic_gripper | EVOLUTION_uniform | G_NOBRANCH | 0.5312 | 0.4107 | 0.6482 | 1346 | False | 0.4688 | 64 |
| prismatic_gripper | EVOLUTION_uniform | G_FULL_INS | 0.7344 | 0.6152 | 0.827 | 814 | False | 0.2656 | 64 |
| prismatic_gripper | EVOLUTION_uniform | G_NOBRANCH_INS | 0.6094 | 0.4869 | 0.7194 | 846.5 | False | 0.3906 | 64 |
| arch_palm | DEFAULT | G_FULL | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_NOBRANCH | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_FULL_INS | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | DEFAULT | G_NOBRANCH_INS | 0 | 3.469e-18 | 0.05662 | 1500 | True | 1 | 64 |
| arch_palm | UNION | G_FULL | 1 | 0.9434 | 1 | 62.5 | False | 0 | 64 |
| arch_palm | UNION | G_NOBRANCH | 1 | 0.9434 | 1 | 64.5 | False | 0 | 64 |
| arch_palm | UNION | G_FULL_INS | 1 | 0.9434 | 1 | 66.5 | False | 0 | 64 |
| arch_palm | UNION | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 66 | False | 0 | 64 |
| arch_palm | EVOLUTION_uniform | G_FULL | 1 | 0.9434 | 1 | 71 | False | 0 | 64 |
| arch_palm | EVOLUTION_uniform | G_NOBRANCH | 1 | 0.9434 | 1 | 75.5 | False | 0 | 64 |
| arch_palm | EVOLUTION_uniform | G_FULL_INS | 1 | 0.9434 | 1 | 81 | False | 0 | 64 |
| arch_palm | EVOLUTION_uniform | G_NOBRANCH_INS | 1 | 0.9434 | 1 | 86.5 | False | 0 | 64 |

## Reading

- anthropomorphic_staggered / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 47 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 43 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 78.5 (censored_fraction 0).
- anthropomorphic_staggered / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 80.5 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 160 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 129 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 242.5 (censored_fraction 0).
- anthropomorphic_staggered / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 259 (censored_fraction 0).
- anthropomorphic_staggered / EVOLUTION_uniform / G_FULL: success_rate 0.1406 [0.07579, 0.2462] (n=64); median proposed >= 1500 (censored_fraction 0.8594).
- anthropomorphic_staggered / EVOLUTION_uniform / G_NOBRANCH: success_rate 0.2031 [0.1227, 0.3171] (n=64); median proposed >= 1500 (censored_fraction 0.7969).
- anthropomorphic_staggered / EVOLUTION_uniform / G_FULL_INS: success_rate 0.1719 [0.09878, 0.2821] (n=64); median proposed >= 1500 (censored_fraction 0.8281).
- anthropomorphic_staggered / EVOLUTION_uniform / G_NOBRANCH_INS: success_rate 0.07812 [0.03383, 0.1702] (n=64); median proposed >= 1500 (censored_fraction 0.9219).
- radial_3 / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 62 (censored_fraction 0).
- radial_3 / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 59.5 (censored_fraction 0).
- radial_3 / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 52 (censored_fraction 0).
- radial_3 / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 63 (censored_fraction 0).
- radial_3 / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 122 (censored_fraction 0).
- radial_3 / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 92 (censored_fraction 0).
- radial_3 / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 132 (censored_fraction 0).
- radial_3 / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 142.5 (censored_fraction 0).
- radial_3 / EVOLUTION_uniform / G_FULL: success_rate 0.9219 [0.8298, 0.9662] (n=64); median proposed 611.5 (censored_fraction 0.07812).
- radial_3 / EVOLUTION_uniform / G_NOBRANCH: success_rate 0.7969 [0.6829, 0.8773] (n=64); median proposed 640.5 (censored_fraction 0.2031).
- radial_3 / EVOLUTION_uniform / G_FULL_INS: success_rate 0.7812 [0.6657, 0.865] (n=64); median proposed 729 (censored_fraction 0.2188).
- radial_3 / EVOLUTION_uniform / G_NOBRANCH_INS: success_rate 0.75 [0.6318, 0.8399] (n=64); median proposed 880 (censored_fraction 0.25).
- prismatic_gripper / DEFAULT / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 36 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 26 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 31.5 (censored_fraction 0).
- prismatic_gripper / DEFAULT / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 30 (censored_fraction 0).
- prismatic_gripper / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 61.5 (censored_fraction 0).
- prismatic_gripper / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 60 (censored_fraction 0).
- prismatic_gripper / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 115.5 (censored_fraction 0).
- prismatic_gripper / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 124.5 (censored_fraction 0).
- prismatic_gripper / EVOLUTION_uniform / G_FULL: success_rate 0.5938 [0.4715, 0.7054] (n=64); median proposed 1000 (censored_fraction 0.4062).
- prismatic_gripper / EVOLUTION_uniform / G_NOBRANCH: success_rate 0.5312 [0.4107, 0.6482] (n=64); median proposed 1346 (censored_fraction 0.4688).
- prismatic_gripper / EVOLUTION_uniform / G_FULL_INS: success_rate 0.7344 [0.6152, 0.827] (n=64); median proposed 814 (censored_fraction 0.2656).
- prismatic_gripper / EVOLUTION_uniform / G_NOBRANCH_INS: success_rate 0.6094 [0.4869, 0.7194] (n=64); median proposed 846.5 (censored_fraction 0.3906).
- arch_palm / DEFAULT / G_FULL: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_NOBRANCH: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_FULL_INS: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / DEFAULT / G_NOBRANCH_INS: success_rate 0 [3.469e-18, 0.05662] (n=64); median proposed >= 1500 (censored_fraction 1).
- arch_palm / UNION / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 62.5 (censored_fraction 0).
- arch_palm / UNION / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 64.5 (censored_fraction 0).
- arch_palm / UNION / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 66.5 (censored_fraction 0).
- arch_palm / UNION / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 66 (censored_fraction 0).
- arch_palm / EVOLUTION_uniform / G_FULL: success_rate 1 [0.9434, 1] (n=64); median proposed 71 (censored_fraction 0).
- arch_palm / EVOLUTION_uniform / G_NOBRANCH: success_rate 1 [0.9434, 1] (n=64); median proposed 75.5 (censored_fraction 0).
- arch_palm / EVOLUTION_uniform / G_FULL_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 81 (censored_fraction 0).
- arch_palm / EVOLUTION_uniform / G_NOBRANCH_INS: success_rate 1 [0.9434, 1] (n=64); median proposed 86.5 (censored_fraction 0).
