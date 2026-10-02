# 02 Oct: out-of-distribution evals

Does training one policy on 8 hands (the unified run, `experiments/01oct_unified_rl`) make it more robust
than a single-hand policy? Every eval here uses `experiments/01oct_embodiment_niches/eval_niches.py`
(1024 envs x 60 s, greedy actions), whose `--target` runs one hand's policy on another hand and whose
`--override dotted.path=literal` sets any env knob.

| | what | plot |
|---|---|---|
| `results/wuji_compare/` | Wuji v2, nominal + 8 OOD conditions: unified (ep 6400 / 7086 / 7827) vs Wuji-only (ep 1600, nearest in skill; ep 2500) | `plots/wuji_compare/id_vs_ood.png` (`analysis/plot_wuji_compare.py`) |
| `results/push_sweep/` | Wuji v2 under random pushes: force_scale 0-150 at 0.02/step, and push probability 0.005-0.2 at scale 50; unified ep 7827 vs Wuji-only ep 1600 / 3600 (`run_push_sweep.sub`) | `plots/push_sweep_wuji2.png` (`analysis/plot_push_sweep.py`) |
| `results/gen_sharpa_zero_shot/` | zero-shot on gen-SHARPA: unified ep 6400 vs SHARPA-only ep 3600 (palm-frame obs), against gen-SHARPA's own policy | `plots/gen_sharpa_zero_shot.png` (`analysis/plot_gen_sharpa_zero_shot.py`) |

Pushes are the env's own wrench DR (`obs_utils/actions.py: apply_wrench_dr`): per step, with probability p, the
cube gets randn(3) x cube mass x force_scale for one 1/60 s step; no torque. No policy here trained with any.

Unified checkpoints `frozen_ep_<N>.pth` are frozen copies of the run's rolling checkpoint (it is rewritten
while training), kept next to it so every condition evaluates the same weights.

Live viewer with push sliders, several policies side by side on one hand:
`experiments/01oct_unified_rl/viser_compare.py --hand wuji2 --policy "Label=<pth>" ...`
