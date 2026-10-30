# 29 Oct: embodiment evolution from scratch -- baseline (archive) vs MAP-Elites

Started 29 Oct 2026 on branch `embodiment_evolution`. Co-design of grammar hands by evolution with an archive, one
shared policy across the population, trained from scratch. Two runs compared: **baseline**, the Darwin Goedel
Machine's archive recipe (Zhang et al. 2026), and **MAP-Elites** over niches of hand (ours). Read with POET (Wang et
al. 2019, Enhanced POET 2020) and OMNI-EPIC (Faldor et al. 2025); the hand is the evolving artifact and, as in DGM,
a *solution* scored on a fixed task (POET's environments are challenges), so DGM is the closest template.

## Shared by both runs

- **Start:** the same 30 hands, one random grammar hand (FEETECH_COARSE_RULES, seed 0) per niche, drawn by
  `make_seed.py` into `assets/urdf/29oct_niche_seed/` (also on master, 57f3a6d); an untrained policy.
- **Niches** (`coevolution/evolution/descriptors.py`): finger count (2-6) x palm cut x mean joints per finger
  (< 2.5 short, 2.5-3.5 medium, > 3.5 long) = 30, every one reachable (a total-joints binning left 4-5 impossible).
- **Policy:** the per-finger transformer, 2 layers, width 128, 4 heads, packed tokens, compiled, fp16 rollouts; the
  14 Oct learner (minibatch 16,384, 2 mini-epochs, lr 1e-4); one template scene of the population (`multi:template:`,
  slot grid pinned at 5 rigid + 2 moving fingers so the network keeps one shape); pen repose v2, finger_target home,
  action delay off, the pre-27-Oct tokens; 24,576 envs.
- **Budget:** 32 generations x 636 epochs (250M steps) = 8B steps per run, one SLURM job per generation
  (`generation.sh`; each job submits the next). Generation 0 trains from scratch; each later one from the last.
- **Fitness** (`coevolution/eval/eval_hands.py`): after a generation's training, each hand gets 10 parallel envs,
  each scored on its FIRST episode (sampled actions); fitness = mean goals per episode, + 0.1 x reward per step as a
  tie-breaker that only ranks hands with no goals.
- **Children:** one grammar mutation of a parent (`hand_sampler.grammar.operators.mutate`), built with
  `hand_sampler/grammar/urdf_export.py` (the 100 random hands' exporter, promoted; rebuilds all 100 byte for byte);
  refused if a duplicate of any hand ever made or if it does not fit the slot grid. 16 per generation.
- **Archive:** every hand ever made, with lineage and fitness history; nothing is deleted.

## The two runs

| | baseline (`baseline.sub`) | MAP-Elites (`map_elites.sub`) |
|---|---|---|
| parents | every hand with a fitness, drawn with weight sigmoid(10 (f_rel - 0.5)) / (1 + its children) | a filled niche chosen uniformly; its elite |
| children in | screened zero-shot with the new checkpoint; the best 8 of 16 admitted, protected one generation | all 16, as challengers |
| replacement | the fittest survivors keep the other 22 places | a challenger, after a generation of training, takes its niche if it beats the elite in the same evaluation pass (no margin) |
| trained per generation | 30 hands | 30 elites + 16 challengers |

**RL-only control** (`rl_only.sub`, strategy `static`; 29 Oct, the user's call): the same 30 seed hands, the same
policy, the same 32 x 636 epochs, and no evolution at all -- no children, the population never changes. It runs
through the same generation loop, so it shares the evolution runs' per-generation restarts (weights carried,
optimizer and adaptive lr reset) and their evaluation: any gain the evolution runs show over it is evolution's.

**Long single runs** (30 Oct, the user's call; `long_*.sub`, wandb group `long_runs`, logs
`debug_outputs/train_logs/29oct_evolution_from_scratch/long_runs/`): the same 8B steps as ONE continuous run, no
per-generation restarts, same policy and learner. Checkpoints kept at the ends of generations 0, 3, 8, 15, 24, 31
(rl_games keeps `last_*_ep_*` only when epoch/636 is a perfect square, plus the last); matched evaluations with
`eval_hands.py` in `debug_outputs/evolution/long_runs_eval/`.

| run | hands | job |
|---|---|---|
| `long_rl_only.sub` | the 30 seed hands (the RL-only control without restarts) | 816867 |
| `long_claude_pick.sub` | `seed_f6_nocut_j2`, a fresh Claude agent's blind pick from the 30 (`claude_pick/README.md`) | 816868 |
| `long_claude_runner_up.sub` | `seed_f3_cut_j2`, that agent's second choice | 816881 |
| `long_random_pick.sub` | `seed_f5_cut_j1`, drawn uniformly at random (`random_pick/README.md`) | 829901 |

Expected (29 Oct, before results): MAP-Elites wins coverage and QD score by construction; the baseline likely leads
on the best hand under the shared policy at 8B (it exploits, and trains fewer hands); the real co-design question --
the winners retrained alone from scratch -- is open.

## Running

    sbatch experiments/29oct_evolution_from_scratch/baseline.sub        # job 806542 (29 Oct)
    sbatch experiments/29oct_evolution_from_scratch/map_elites.sub      # job 806543 (29 Oct)
    sbatch experiments/29oct_evolution_from_scratch/rl_only.sub         # job 811622 (29 Oct)
    .venv_isaacsim/bin/python -m coevolution.evolution.generation status --run-dir debug_outputs/evolution/baseline

- Run directories: `debug_outputs/evolution/{baseline,map_elites,rl_only}/` (`archive.json`, `state.json` with the map and
  per-generation summaries, `gen_NNNN/` manifests, fitness, policy); training logs
  `debug_outputs/train_logs/evolution/<run>/gen_NNNN/`; job logs `debug_outputs/train_logs/29oct_evolution_from_scratch/`.
- wandb `gen_mechanics_evolution_scratch`: one run per generation, `baseline_iter_<N>_<date>_<time>`,
  `map_elites_iter_<N>_<date>_<time>`, `rl_only_iter_<N>_<date>_<time>`.
- Every evolved hand goes onto master after each generation (`push_assets.py`, called by `generation.sh`;
  plumbing on a temporary index, no checkout): `assets/urdf/29oct_evolution/<run>/<id>/` plus the generation
  manifests, so the wandb pose viewer can fetch their meshes.
- `dry_baseline.sub`, `dry_map_elites.sub`: 2-generation rehearsals at 3,072 envs (run 29 Oct; they found the
  zero-floor admission bug, fixed and tested).

## To read off

- per generation: best and mean fitness of the population, niches with a hand scoring goals, QD score (sum of the
  niche bests), the baseline's occupied niches
- the archive trees (lineage, operators, fitness)
- validation: the top hands of each run retrained alone from scratch (3 seeds), against the 100 random hands' best
