# Evolution pilot 1: weak-controller baseline (2026-09-28)

## Status
This is a pipeline validation and a weak-controller baseline, **not** a grammar comparison. The shared controller's action std exploded in every run (I41). Final std was 765-4800 on the cluster runs and about 245 in local G_V2S_s0 after generation 6, against 0.7-3.0 in single-hand runs that learn. With actions clipped to [-1, 1], the policy is effectively random, so graded fitness mostly measures passive holding.

## Setup
- MAP-Elites over digits x joint bins (30 cells).
- 64 designs per generation: 60 archive and offspring, plus 4 projected probes (allegro_right, dclaw, sharpa_left_on_iiwa14, leap_right).
- 4096 envs, 900 epochs per generation, 8 generations.
- train_tail fitness; weights carried across generations.
- Code 8b4a91e.
- Runs:
  - cluster array 2371572 (RTX A6000, lab nodes): G_V1 s0/s1 and G_V3S s0/s1, about 2.9 GPU-h each;
  - local 4090: G_V2S s0, whose generation 7 crashed and was scored from one stale window (flagged).

## Final generation

| | G_V1 (s0 / s1) | G_V3S (s0 / s1) | G_V2S (s0) |
|---|---|---|---|
| coverage /30 | 15 / 11 | 13 / 10 | 15 |
| QD-score | 1.64 / 1.61 | 1.41 / 1.11 | 1.59 |
| best fitness | 0.16 / 0.36 | 0.32 / 0.17 | 0.19 |
| founders (max share) | 5 (0.33) / 6 (0.36) | 7 (0.31) / 5 (0.30) | 7 (0.20) |
| mean joints | 10.5 / 11.0 | 9.3 / 10.6 | 10.7 |
| probe dclaw | 0.11 / 0.11 | 0.12 / 0.12 | 0.11 |
| probe allegro / sharpa / leap | ~0.06 / 0.06 / 0.07 | ~0.06 / 0.06 / 0.07 | ~0.06 / 0.06 / 0.07 |

## What it shows
- **The pipeline works on the cluster at scale.** 32 generations ran with no failures, calibrated at about 23 min per generation and 2.9 GPU-h per run. The archive, lineage and probes are logged.
- **Seed variance is as large as any variant gap.** No grammar conclusion can be drawn with a random controller.
- **Founder dominance is moderate** (max share 0.2-0.36). Every archive cell turns over within 8 generations.
- **Probe hands do not improve.** Zero-shot control of known hands is not being learned.

## Next
Fix population training (I41); an Opus worker is on it. Then rerun the pilot with the fixed controller settings.

Total cluster GPU-h used through this pilot: 13.06 of 200.
