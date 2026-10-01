# Evolution pilot 2: fixed controller, G_V1 vs G_V3S (2026-09-28)

## Setup
- MAP-Elites over digits x joint bins (30 cells).
- 32 designs per generation: 28 archive and offspring, plus 4 projected probes (allegro_right, dclaw, sharpa_left_on_iiwa14, leap_right).
- 4096 envs, 1200 epochs per generation, 12 generations.
- Population train config (`InHandReorientPopSAPG.yaml`: no SAPG entropy bonus, log-std <= 0); sigma clamped on carry; normaliser count cap 1e6.
- Seeds 0-1 ran on the cluster (array 2378309, RTX A6000, 5.6 h each); seed 2 ran on the local 4090.
- All 72 generations were clean: 1 attempt each, 0 non-finite resets. Action std fell from 0.85 to 0.45 in every run.

## Final generation (per seed; mean)

| | G_V1 s0 / s1 / s2 | mean | G_V3S s0 / s1 / s2 | mean |
|---|---|---|---|---|
| coverage /30 | 11 / 13 / 13 | 12.3 | 11 / 10 / 13 | 11.3 |
| QD-score | 3.51 / 9.30 / 2.20 | 5.00 | 6.63 / 4.31 / 8.35 | 6.43 |
| mean elite fitness | 0.32 / 0.72 / 0.17 | 0.40 | 0.60 / 0.43 / 0.64 | 0.56 |
| **median elite fitness** | 0.24 / 0.35 / 0.16 | 0.25 | 0.42 / 0.44 / 0.39 | **0.42** |
| founders (max share) | 4 (.45) / 6 (.38) / 8 (.23) | 6.0 (.36) | 8 (.27) / 7 (.20) / 7 (.23) | **7.3 (.23)** |
| mean joints | 9.3 / 9.8 / 11.1 | 10.1 | 10.6 / 10.4 / 10.7 | 10.5 |
| probe allegro | .35 / .42 / .24 | .34 | .36 / .45 / .40 | .40 |
| probe dclaw | .27 / .44 / .53 | .41 | .36 / .31 / .35 | .34 |
| probe sharpa | .21 / .24 / .39 | .28 | .37 / .38 / .37 | .37 |
| probe leap | .45 / .48 / .37 | .44 | .41 / .51 / .41 | .44 |

Aggregate goals per episode rose from about 0.03-0.04 to 0.05-0.15 in every run. Time held rose from about 0.4-0.5 s to 0.7-3.2 s.

## Reading (2 arms x 3 seeds; no significance claimed)
- **Consistent across all seeds.** Every G_V3S run has a higher median elite fitness (0.39-0.44) than every G_V1 run (0.16-0.35). G_V3S also keeps more founders with lower dominance (max share 0.20-0.27 vs 0.23-0.45). Its probe results are more uniform: sharpa .37 vs .28, allegro .40 vs .34, dclaw lower .34 vs .41, leap equal.
- **Not separated.** QD-score, best fitness and coverage are dominated by seed variance (G_V1 QD range 2.2-9.3).
- **Compared with pilot 1** (random controller), probes rose from about 0.06 to 0.3-0.5. Zero-shot control of known hands now improves as the population evolves.
- **Competence is still modest**: at most about 0.15 goals per episode in aggregate, with the best designs at about 3 goals per episode.

## Cost
Cluster total through pilot 2: 35.4 GPU-h (cluster hist).
