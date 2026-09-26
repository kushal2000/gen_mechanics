# Experiment: e4_redundancy

Wall time: 8.916 s

## Params

```json
{
  "n_offspring": 10,
  "n_offspring_parents": 1000
}
```

n_seeds: 10000

## Table: variant x metric

| variant | distinct_hash_fraction (10k seeds) | canonical_reorder_fraction | default: offspring_null_fraction | default: pairwise_identical_fraction | small: offspring_null_fraction | small: pairwise_identical_fraction |
|---|---|---|---|---|---|---|
| G_FULL | 1 | 0.9015 | 0.0542 | 0.01656 | 0.0366 | 0.0176 |
| G_SERIAL | 1 | 0.72 | 0.0536 | 0.01787 | 0.0528 | 0.02733 |

## Reading

- G_FULL: distinct_hash_fraction 1 (n=10000); canonical_reorder_fraction 0.9015; default ops offspring_null_fraction 0.0542, pairwise_identical_fraction 0.01656 (n_parents=1000); small-step ops offspring_null_fraction 0.0366, pairwise_identical_fraction 0.0176 (n_parents=1000).
- G_SERIAL: distinct_hash_fraction 1 (n=10000); canonical_reorder_fraction 0.72; default ops offspring_null_fraction 0.0536, pairwise_identical_fraction 0.01787 (n_parents=1000); small-step ops offspring_null_fraction 0.0528, pairwise_identical_fraction 0.02733 (n_parents=1000).
