# Co-evolution analysis (coevolution_v1)

Charts comparing the fixed 1024-hand round-500 population (baseline, chained
across 24 h links) with co-evolution V1 (`assets/populations/coevo_r500_v1`,
40 generations). Scripts live here (tracked); every figure, the curve cache and
the videos land in `debug_outputs/10sep_coevo_analysis/` (not tracked). Run
from the repo root with `.venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/<script>.py`.

| script | writes |
|---|---|
| `curves.py` | shared loader; caches tensorboard curves in `.curve_cache/` keyed on file mtime. Baseline chain job IDs, restart blanking (`WARMUP`), `SEC_PER_EPOCH`, tick formatter. |
| `plot_clean.py` | `coevo_clean.png`, `coevo_clean_return.png`, `coevo_clean_tolerance.png` — paper style (`depthbasedRL/plot_figures/_style.py`) |
| `plot_joints.py` | `coevo_clean_joints.png` — joints per hand vs generation, mean and IQR |
| `plot_coevo.py` | `baseline_vs_coevo_v1.png` — annotated return chart |
| `plot_tolerance.py` | `tolerance_baseline_vs_coevo.png` — annotated curriculum chart |
| `plot_imitation.py` | `imitation_vs_evolution.png` — capsule SHARPA vs co-evolution |
| `lineage.py` | prints ancestry collapse and per-generation diversity (no figure) |
| `lineage_figure.py` | `lineage_ancestry.png`, `lineage_final_top8*.png`, `lineage_rivals_gen6.png` — hand renders along the winning lineage |
| `phylogeny.py` | `phylo_muller.png`, `phylo_genealogy.png`, `phylo_diversity.png` |
| `derl_fig2.py` | `derl_fig2*_coevo_r500_v1.png` — DERL Fig. 2 layout (fitness, founders, radial tree, Muller) |
| `plot_traits.py` | `traits*_coevo_r500_v1.png` — fingers, joints, finger length, palm area per generation |
| `plot_objects.py` | `objects_diverse24.png` — the curated pool to scale |
| `render_policy.py`, `render_g39_178.sh` | `videos/g39_178/*.mp4` — a checkpoint driving one design on the curated objects |

The reward curves blank the first 200 epochs after every process restart:
rl_games refills its 100-game reward average from the first, failed episodes of
a fresh process, so every generation and baseline link opens with a spurious
collapse. Wall-clock hours use 5.5 s/epoch, measured from the baseline chain.

## Findings from V1 (`lineage.py`, `lineage_figure.py`)

- One family: by generation 13 all 1024 hands descend from gen-0 design #521
  (rank 118 of 1024 at gen 0; the lineage doubled every generation 1→512
  survivors). Truncation top-50% with elitism compounds any consistent edge.
- Selection in generations 0–3 was on noise: the policy sat on the ~300-return
  plateau until ~epoch 9000, rank correlation between gen-0 return and the
  baseline's eventual per-design return is 0.21, and the baseline's top-50
  designs were lost at ~35 % per generation. Its best design, #933 (5f 20j,
  return 8,445 at tolerance 0.029), was culled at generation 1 at 272 vs a
  median of 276.
- Mutations are not the bottleneck: nine operators fire at equal rates and
  parent→child score correlation late on is ~0.15.

For a V2: do not select until the policy discriminates (train gen 0 longer, or
start from the baseline checkpoint at its tolerance); lower the pressure
(keep 75–90 %, or tournament); cap survivors per gen-0 lineage or niche by
(fingers, joints). Then cross-evaluate both final policies on both populations
at a fixed tolerance.
