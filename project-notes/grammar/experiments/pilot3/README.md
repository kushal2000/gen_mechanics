# Evolution pilot 3: three grammars, 20 generations, legacy task spec (2026-10-01/02)

## Status
This is a record under the **legacy** task spec, which is being replaced (see LOG "Task-spec study results"). Same setup as pilot 2: 32 designs, 1200 epochs per generation, the population config. Pilot-2 runs were extended to 20 generations, and G_V2S s0-3 are new.
- Cluster: arrays 2511386 and 2511387.
- Local: G_V1 s2.
- G_V3S s2 stopped at generation 12 (local GPU freed for the task port) and is excluded.

## Final generation (generation 19)

| | G_V1 (s0 / s1 / s2) | G_V2S (s0 / s1 / s2 / s3) | G_V3S (s0 / s1) |
|---|---|---|---|
| median elite fitness | 0.32 / 0.36 / 0.20 | 0.56 / 0.29 / 0.24 / 0.48 | 0.50 / 0.47 |
| mean elite fitness | 0.41 / 1.28 / 0.22 | 0.50 / 0.93 / 0.28 / 0.56 | 0.71 / 0.70 |
| QD-score | 4.5 / 16.7 / 2.9 | 5.4 / 12.1 / 3.7 / 7.8 | 8.5 / 8.4 |
| best fitness | 0.87 / 10.5 / 0.41 | 1.10 / 5.08 / 0.48 / 1.96 | 3.35 / 3.39 |
| coverage /30 | 11 / 13 / 13 | 11 / 13 / 13 / 14 | 12 / 12 |
| founders (max share) | 4 / 5 / 8 (mean max share 0.35) | 5 / 8 / 8 / 7 (0.34) | 7 / 7 (0.29) |
| probes allegro / dclaw / sharpa / leap (mean) | .40 / .41 / .30 / .40 | .36 / .44 / .33 / .42 | .40 / .37 / .44 / .41 |

## Reading (few seeds; no significance claimed)
- **Median elite fitness ranks V3S > V2S > V1.** The best two G_V3S seeds sit above every G_V1 seed, and G_V3S has by far the lowest seed variance.
- **G_V1 is high-variance.** One seed found an outlier design at 10.5, which dominates its QD-score.
- **Coverage is similar (11-14 of 30)**, and founder diversity is slightly higher for the structured grammars.
- **Probe hands are similar across grammars** (0.3-0.45), except sharpa, which does best under G_V3S (0.44 vs 0.30).

## Cost
Cluster total through pilot 3: 87.7 GPU-h.
