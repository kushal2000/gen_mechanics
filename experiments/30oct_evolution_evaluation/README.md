# 30 Oct: evaluating the evolution runs

Started 30 Oct 2026 on branch `2026-10-30-evolution_evaluation`. Evaluation of
`experiments/29oct_evolution_from_scratch/`: the baseline (DGM-style archive) and MAP-Elites runs, the RL-only control
(loop and one long run), and the single-hand baselines trained alone (the Claude designer's pick and runner-up, a random
design). Pen repose v2; fitness = goals per episode over 10 envs, first episode each, sampled actions
(`coevolution/eval/eval_hands.py`); the 50-goal cap (`max_successes`) ends an episode.

These scripts only read the runs' outputs (`debug_outputs/evolution/`, the long runs' checkpoints): the running jobs
read `coevolution/` and `experiments/29oct_evolution_from_scratch/` from this checkout, so nothing they use is edited
here until they finish.

## Scripts

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/niche_table.py       # every niche, each run
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/plot_niches.py       # one hand per niche, as mounted
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/mutation_quality.py  # child vs parent, per operator
    bash experiments/30oct_evolution_evaluation/eval_long_runs.sh 10176                   # long runs at a matched checkpoint (GPU)
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/final_archive_eval.py --top-k 2 --envs-per-hand 100 \
        --tag _it73_100env                                                                # final archives, final policies (GPU)
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/top_hands_100env.py  # top hands, both reset modes (GPU)
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/plot_best_hand_comparison.py \
        --summary-tag _it73_100env                                                        # best vs best, one image
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/render_archive_videos.py --run map_elites \
        --top 4 --summary-tag _it73_100env                                                # sim clips (GPU, cameras)
    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/diag_table_contact.py --hands g0058_c00 ...   # stuck fingers

`eval_long_runs.sh`: the long runs keep checkpoints at epochs 636, 2544, 5724, 10176, 15900, 20352 = the ends of
iterations 0, 3, 8, 15, 24, 31 (rl_games keeps `last_*_ep_*` only when epoch/636 is a perfect square, and the last).
Outputs `debug_outputs/evolution/long_runs_eval/<run>_ep<E>.json`.

`plot_niches.py`: the 30 niches as a 5 x 6 grid (fingers x palm cut, finger length), each hand at its home pose (all
joints 0) mounted palm down as in the env, seen from below, one scale. Default: the seed (`plots/niches_seed.png`); with
`--manifest <gen>/manifest.json --fitness <gen>/fitness.json` the best hand per niche of that pass, labelled with goals
per episode (`plots/niches_map_elites_it21.png`: MAP-Elites' iteration 21, elites and challengers).

## Final results (2 Nov: both runs at 74 iterations, ~18.5B steps)

Both runs were extended to 74 iterations; iterations 64-73 used table-safe finger resets (below). The end-of-run
evaluation (`final_archive_eval.py --top-k 2 --envs-per-hand 100 --tag _it73_100env`): per niche the top 2 hands by
their latest loop score, re-evaluated under the run's FINAL policy, 100 envs each (s.e. ~+-2), sampled actions, SAPG
leader, table-safe resets, 50-goal cap. `debug_outputs/evolution/final_archive_eval/*_it73_100env*`.

| final archive | greedy evolution | MAP-Elites |
|---|---|---|
| QD (sum of niche winners' goals/ep) | 260.5 | **554.5 (2.13x)** |
| niches with goals (/30) | 11 | **25** |
| niches >= 20 goals/ep | 10 | **17** |
| best hand | 34.0 (`g0063_c01`: 4 fingers, cut, medium) | **39.3** (`g0067_c12`: 4 fingers, cut, long) |
| the loop's own QD at iteration 73 | 335 | 573 |

- **MAP-Elites: twice the QD, twice the coverage, and the best hand** (+5.3 goals, well outside the error). The loop's QD
  overstated greedy by 29%: its niche holders are scored iterations ago (greedy stops training most of them).
- **Coverage:** MAP-Elites has working designs (25-34 goals/ep) in niches greedy never fills: 5- and 6-finger hands
  without a palm cut, and long fingers for 3-6 fingers. Greedy stays in its 3-5-finger palm-cut family. The 2-finger
  niches are empty in both (MAP-Elites' best 2.0; selection and the grammar's removal rule, see below).
- **MAP-Elites' top hands are long-fingered 4-finger designs** once resets are table-safe: `g0067_c12` 39.3,
  `g0058_c00` 36.7 (4 fingers, no cut, long), `g0062_c11` 36.5, `g0064_c00` 35.9 (3 fingers). Greedy: `g0063_c01` 34.0,
  `g0050_c01` 33.2, `g0065_c04` 32.8.
- Design files: `debug_outputs/evolution/<run>/hands/<id>/` (URDF, palm meshes, design and spec JSON); copies on
  `master` under `assets/urdf/29oct_evolution/<run>/<id>/`.
- Images: `plots/best_hand_it73_100env_g0067_c12_vs_g0063_c01.png` (best vs best, 1:1),
  `plots/hand_g0058_c00_vs_g0058_c07_100env.png` (`g0058_c00` vs greedy's best at iteration 63), niche galleries
  `plots/niches_{map_elites,baseline}_final_it73.png`. Videos (20 s clips under the final policies, top 4 per run, and
  best vs best): `debug_outputs/evolution/videos/final_archive_it73/`.
- Evaluation noise: the loop's 10-env "best hand" runs ~5-10 goals high (one env's goals have s.d. ~18, and the best is
  a maximum over 30-46 noisy scores); single 20 s video clips vary even more. Quote the 100-env numbers.

## The table-safe reset fix (1 Nov)

The olympics reset drew each finger joint uniformly over its whole range with the wrist always 0.15 m above the table,
so long fingers drawn bent downward started inside the table: ~20% of `g0058_c00`'s and `g0061_c04`'s episodes had a
fingertip below the table top and stayed stuck (`diag_table_contact.py`). This understated long-fingered designs in
training, evaluation and selection. The fix (`env.reset.table_safe`, on by default): forward kinematics after the draw,
any env with a fingertip centre below table top + 1 cm is redrawn (up to 10 times, then the home pose). After it: 0%
below the table and no slowdown. At 100 envs under the iteration-63 policy it lifted `g0058_c00` from 25.2 to 37.1 and
`g0061_c04` from 30.3 to 37.4 (`top_hands_100env.py`); in the 10 extended iterations MAP-Elites' loop QD rose from
~489-537 to ~491-586 and greedy's from ~308 to ~309-335.

## Learner experiments (31 Oct - 1 Nov)

MAP-Elites' best hand `g0018_c00` alone, 3,180 epochs, 3 seeds, cap-200 evaluation; each a subfolder with a README.
Settled stack: SAPG, shared action noise, log-std clamp [-5, 2], fixed lr 3e-4, fp16, minibatch 16,384.

| question | result | folder |
|---|---|---|
| adaptive lr? | fixed 3e-4 matches it (95 vs 100); fixed 1e-4 unreliable | `lr_schedule_test/` |
| SAPG or PPO? | same ceiling; PPO slow on 2 of 3 seeds with every noise design | `ppo_vs_sapg/`, `ppo_noise/` |
| action noise | shared 100 / 97 / 103 vs per-slot 89 / 122 / 90; predicted per joint worse (66.5 uncapped, 86.9 with clamp + bounds loss) | `shared_log_std/`, `predicted_log_std*/`, `log_std_cap/` |
| precision | bf16 collapsed; fp32 no better (91) and 1.5x slower | `amp_bf16_test/`, `fp32_test/` |
| minibatch 98k vs 16k | 98k fails for both (1-3 goals/ep) | `minibatch_96k/` |
| the NaN crashes | runaway per-slot log-std on rare joint slots overflowed fp16; fixed by the clamp | `nan_debug/`, `log_std_cap/` |

## Findings so far (30 Oct, runs at iteration ~21 of 32)

- **Single hands trained alone lead at every matched budget.** End of iteration 15: Claude's runner-up `seed_f3_cut_j2`
  47.5 goals/episode (9 of 10 episodes at the cap), its pick `seed_f6_nocut_j2` 40.7; baseline best 33.9, MAP-Elites
  3.3, RL-only 1.3-1.4. This shows training one hand beats splitting training across 30-46, not yet a better design.
- **The baseline collapsed onto one lineage** (all 30 hands from `seed_f3_nocut_j0` by iteration 5; 5 niches trained),
  found a palm cut (iteration 2) then 4-finger variants, and plateaued at a mean of ~21 from iteration 16.
- **MAP-Elites keeps 24 of 30 niches with goals** (every 3- to 6-finger niche); its best rose late (9.2 at iteration 20).
  Both runs converged independently on 4-finger descendants of `seed_f3_nocut_j0` with medium fingers.
- **2-finger niches are empty because of selection, not design:** the RL-only control trains `seed_f2_nocut_j1` to
  0.7-0.9 goals/episode; MAP-Elites replaced that hand at iteration 1 on the reward tie-break (both 0 goals).
- **Mutations:** ~92% of attempts build a valid new hand; ~1 in 4 trained children beats its parent (median 0.8-0.9 of
  the parent). The baseline's zero-shot screen passes ~10% (median 0.20) and rejects structural changes untrained.
- **Restarts cost little:** the RL-only control as one long run matches the loop version (QD 11.4 both at iteration 15).

## Plots and tables (`plots/`, `tables/`)

- `plot_training_curves.py` -> `train_goals_per_episode.png`: training goals/episode vs epochs, RL only vs greedy
  evolution vs MAP-Elites (log y; each restart's dip trimmed until back to 80% of the previous iteration's level, at most
  200 epochs, then a 50-epoch rolling mean). Logs are read with a fast parallel reader and cached in
  `debug_outputs/evolution/curve_cache/`.
- `plot_greedy_vs_single_hands.py` -> `greedy_vs_single_hands.png`: greedy evolution vs RL on Claude's pick vs RL on a
  random hand.
- `plot_niche_tables.py` -> `best_per_niche_tables.png`; `best_per_niche_tables.py` -> `tables/*.md`: the best goals/ep
  each algorithm found per niche.
- `plot_niches.py` -> `niches_seed.png`, `niches_map_elites_it21.png`: one hand per niche, home pose, palm down.
- `viser_evolution_play.py`: any evolution policy (any iteration, or a long run's checkpoint) on any of the seed hands,
  MAP-Elites' map and the baseline's population, live in viser (`--selftest` for a headless check).

## The finger-removal test (`removal_test/`, 30 Oct)

Why MAP-Elites' 2-finger niches stay empty. `build_removals.py` builds every single-finger removal of MAP-Elites' 3-finger
elites: 16 of 18 plain removals break the grammar's mount-at-palm-centre rule (operators.mutate then silently picks
another operator), so each removal also gets a re-centred variant (fingers shifted by whole base steps). Zero-shot under
MAP-Elites' iteration-21 policy the stub-removed hands score ~0; after ONE iteration of fine-tuning on the 24-hand set
(`removal_finetune.sub`) they reach 2.1-5.6 goals/ep (parents 4.1-7.6), and the shift alone costs nothing (controls).
So 2-finger hands one removal from the best elites are good; the grammar rule keeps them from being made. Results in
`debug_outputs/evolution/removal_test/`.

## Solo retraining (`solo_retrain/`, 30 Oct; jobs 851829-851837)

The fair design comparison: greedy evolution's best (`g0029_c02`, iteration 31), MAP-Elites' best (`g0018_c00`,
iteration 30) and the Claude designer's runner-up (`seed_f3_cut_j2`) each trained ALONE from scratch, seeds 100-102,
6,360 epochs, the same policy and learner as every run. Under either run's shared policy the two evolved bests are not
comparable (each policy scores the other run's best near 0). Compare epochs to 1 goal/episode and a final evaluation.

**Result (31 Oct; `eval_solo.sh`: `eval_hands_cap.py`, the episode cap raised from 50 to 200, 20 envs per seed):**

| trained alone, 6,360 epochs | seed 100 | 101 | 102 | mean | episodes at 200 | dropped |
|---|---|---|---|---|---|---|
| Claude's runner-up `seed_f3_cut_j2` | 136.7 | 121.2 | 129.2 | **129.0** | 53% | 43% |
| MAP-Elites' best `g0018_c00` | 90.8 | 135.0 | 120.0 | **115.2** | 43% | 50% |
| greedy's best `g0029_c02` | 72.1 | 86.4 | 74.0 | **77.5** | 13% | 80% |

Claude's blind pick and MAP-Elites' best are not separable with 3 seeds; both beat greedy evolution's best on every seed
(+50-65%), the reverse of the evolution runs' own scores (greedy's best looked 2x MAP-Elites' under its shared policy).
Training curves agree: goals/ep at epoch 1,600, mean of seeds, 32.3 / 26.9 / 14.0; all three reach 1 goal/episode at
~435-475 epochs. References, the single-hand long runs at 8B steps, same evaluation: runner-up 132.7, Claude's pick
86.5, the random hand 14.2.

## To do

- Solo retraining of the final bests (`g0067_c12`, `g0058_c00` vs greedy's `g0063_c01`) with the settled stack and
  table-safe resets: the fair design comparison for the paper (the 30 Oct solo runs used iteration-30 bests).
- Matched long-run evaluations at epochs 15900 and 20352.
- Representative videos: several episodes per hand, keep one near its mean (single clips are noisy).
- Done (2 Nov): the end-of-run re-evaluation of every niche under each run's final policy (above).
