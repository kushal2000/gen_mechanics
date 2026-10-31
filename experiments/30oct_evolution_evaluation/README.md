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

`eval_long_runs.sh`: the long runs keep checkpoints at epochs 636, 2544, 5724, 10176, 15900, 20352 = the ends of
iterations 0, 3, 8, 15, 24, 31 (rl_games keeps `last_*_ep_*` only when epoch/636 is a perfect square, and the last).
Outputs `debug_outputs/evolution/long_runs_eval/<run>_ep<E>.json`.

`plot_niches.py`: the 30 niches as a 5 x 6 grid (fingers x palm cut, finger length), each hand at its home pose (all
joints 0) mounted palm down as in the env, seen from below, one scale. Default: the seed (`plots/niches_seed.png`); with
`--manifest <gen>/manifest.json --fitness <gen>/fitness.json` the best hand per niche of that pass, labelled with goals
per episode (`plots/niches_map_elites_it21.png`: MAP-Elites' iteration 21, elites and challengers).

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

## To do

- Evaluate the solo runs with the 50-goal cap raised (every single hand reaches it).
- At the end of the runs: re-evaluate every niche's best hand under each run's final policy (the archive QD keeps stale
  scores for hands no longer trained).
- Matched long-run evaluations at epochs 15900 and 20352.
