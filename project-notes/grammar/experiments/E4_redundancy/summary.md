# Experiment: e4_redundancy

Wall time: 8.757 s

## Params

```json
{
  "n_offspring": 10,
  "n_offspring_parents": 1000
}
```

n_seeds: 10000

## Table: variant x metric

| variant | distinct_hash_fraction (10k seeds) | derivation order differs from canonical order (fraction) | default: null_fraction | default: impossible_fraction | default: pairwise_identical_fraction | small: null_fraction | small: impossible_fraction | small: pairwise_identical_fraction |
|---|---|---|---|---|---|---|---|---|
| G_FULL | 1 | 0.9015 | 0 | 0.0542 | 0.01656 | 0.01043 | 0.026 | 0.04887 |
| G_SERIAL | 1 | 0.72 | 0 | 0.0536 | 0.01787 | 0.009931 | 0.042 | 0.05927 |

## Reading

- G_FULL: distinct_hash_fraction 1 (n=10000); fraction of models whose derivation order differs from canonical order 0.9015 (NOT a redundancy measure -- see e4_redundancy.py's module docstring); default ops null_fraction 0 (hash equal AND vary succeeded), impossible_fraction 0.0542, pairwise_identical_fraction 0.01656 (n_parents=1000); small-step ops null_fraction 0.01043, impossible_fraction 0.026, pairwise_identical_fraction 0.04887 (n_parents=1000).
- G_SERIAL: distinct_hash_fraction 1 (n=10000); fraction of models whose derivation order differs from canonical order 0.72 (NOT a redundancy measure -- see e4_redundancy.py's module docstring); default ops null_fraction 0 (hash equal AND vary succeeded), impossible_fraction 0.0536, pairwise_identical_fraction 0.01787 (n_parents=1000); small-step ops null_fraction 0.009931, impossible_fraction 0.042, pairwise_identical_fraction 0.05927 (n_parents=1000).
