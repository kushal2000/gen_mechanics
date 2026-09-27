# Existing co-evolution machinery: what the evolution pilot can reuse

Survey date: 2026-09-27. Read-only.

## What exists
- **Generation loop.** One SLURM job per generation, chained by self-resubmission (`experiments/17sep_coevolution/coevo_gen.sub`): train (`coevolution/train.py`), then select and mutate (`python -m hand_sampler.evolve`), then sbatch the next generation. There is no separate evaluation step; selection ranks training-episode returns.
- **Design format.** Old-sampler `design_space.Hand` trees (`population_io` format 1), stored as `gen_<k>/population.json` plus `selection.json`.
- **Weight carry.** `--checkpoint <newest nn/*.pth> --checkpoint_load_mode weights` does a strict `load_state_dict` plus running observation statistics; the optimizer, epoch and env state start fresh. The curriculum is carried by hand via `env.termination.resume_success_tolerance`. There is no weight surgery, because the padded envelope keeps shapes fixed.
- **Fitness.** `coevolution/design_rewards.py` `DesignRewardWrapper` (`--design_rewards`) banks return and goals per design into `design_rewards_rank<r>.json`, pooled over the whole generation while the policy and curriculum change.
- **Selection.** Truncation only, (mu+lambda) with one child per survivor from 9 old-sampler operators (`hand_sampler/evolve.py`). A bug: child records are not truncated when KEEP > n/2. There is no MAP-Elites, archive, descriptor, novelty or niching code anywhere.
- **Scale.** V1: 1024 designs, 24 envs per design on 2 GPUs, 2000 epochs per generation (about 3 h), 40 generations (about 6 days). V2: 5000 epochs per generation (about 8 h), about 14 days.

## Lessons
- **Selection on noise.** Generations 0-3 selected on noise: the rank correlation between successive fitness estimates was 0.21, and the best baseline design was culled in generation 1. Pooled training returns are a noisy fitness.
- **Lineage collapse.** Every hand descended from one founder by generation 13.
- **SHARPA collapse** (branch `origin/2026-09-23-gen_sharpa_coevolution_analysis`, commit 88d70d6):
  - The capsule SHARPA peaks at 0.40 goals per episode in generation 9, then falls to 0.00 by generation 35 while the population holds about 2.9.
  - After generation 12 no design can reach it by mutation, because palm thickness is never mutated.
- **Literature tension.** Strgar & Kriegman report that carrying weights without resets collapses diversity, which cuts against the meeting's "don't re-initialise". MAP-Elites keeps diversity in the archive regardless, but controller specialisation should be monitored.
- **Etiquette.** Self-resubmitting chains, 128 GB for a 35 GB peak, and CPU selection inside a 2-GPU allocation.

## Pilot design choices that follow
- **One SLURM job per run.** A Python driver loops over generations inside a single main-tier job, with an honest limit (about 6 h for about 20 generations). It resumes from a per-generation state file if restarted. No chains.
- **Fitness from a frozen-policy evaluation** at the end of each generation, with a fixed horizon and graded score, not pooled training returns.
- **Per-design scoring inside the inhand env**, or an adapter that accepts the dict `scene_record`.
- **Reuse** weight carry (`weights` mode) and `resume_success_tolerance`.
- **New MAP-Elites archive** over (digit count, joint count) for grammar derivations. Mutate with `derive.vary(..., operators=EVOLUTION_OPERATORS)`, filter with admission plus rest overlap, and write through `population_file`.
- **Fixed population size**, with total envs >= designs. Keep the 32-slot envelope and obs_list fixed across generations, so strict weight loading works.
