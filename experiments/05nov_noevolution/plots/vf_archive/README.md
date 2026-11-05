# Value-net MAP-Elites: round 0 vs round 1 (6 Nov)

Study 7 (`studies/study_vf_archive.py`), plotted by `studies/plot_vf_archive.py`.

- **Policy:** the no-evolution run's iteration 3 (`noevolution_4096/gen_0003/policy.pth`), every design zero-shot with
  100 envs (first episode each, sampled actions, the fixed table-safe reset).
- **Archive:** 32 niches x 128.
  - Round 0 is the 4,096-design population.
  - Round 1: every archive design gets one random valid application of each of the 9 mutation types (~33k children).
    A frozen design value net (one per-finger transformer, trained on the population's 100-env scores; not yet the
    tuned dropout-0.3 version) scores them, and each niche keeps its 128 best by prediction.
- **Ground truth:** every round-1 archive design re-scored with 100 envs.

| file | what |
|---|---|
| `archive_per_niche.png` | per niche: min-max (line) and mean (marker) of round 0 true, round 1 true, round 1 predicted |
| `predicted_vs_true.png` | round 1, per niche (means) and per admitted child (hexbin) |
| `per_niche.csv` | the numbers behind both (mean / max / min per niche and series) |
| `archive_niche_means.png` | the archive as MAP-Elites grids (rows: fingers; columns: motor bins; grey: not a niche), one column per iteration: top, the value net's predicted niche means; bottom, the actual (100-env) niche means. Iterations with 100-env truth so far; one colour scale throughout. `studies/plot_vf_archive_grid.py` |

**Results:**

- **The archive improves for real.** The mean over niches goes 0.26 -> 0.66 true (0.70 predicted). The per-niche best
  goes 0.92 -> 1.00 on average; 24/32 niches improve by > 0.03 and none gets worse.
- **The net ranks well** (r = 0.83 over the 3,530 admitted children) but runs slightly high: means are over-predicted by
  +0.04, and duds are missed (predicted minimum 0.07-0.83 while 21/32 niches still hold a 0-goal design).
- **Round 0's predictions equal its true scores exactly.** The net was trained on those designs and memorises them, so
  they are not shown.
