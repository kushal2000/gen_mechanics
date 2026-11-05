# Minibatch 98,304 vs 16,384 (31 Oct)

The key Isaac Gym result of Beukman et al. 2026 ("Preventing Learning Stagnation in PPO by Scaling to 1 Million
Parallel Environments", arXiv 2603.06009, Section 5.1 / Figure 10): with SAPG's own code at 24,576 envs, reverting
SAPG's minibatch of 98,304 (4 x envs) to Isaac Gym's default 16,384 "significantly outperforms the default setting for
both methods, makes PPO more amenable to additional parallelisation, and reduces the performance gap between it and
SAPG". Our runs already use 16,384; this checks the claim on our task.

- `sapg_mb98k_g0018_s10{0,1,2}.sub`: exactly `../lr_schedule_test/lr_adapt_g0018_s10*.sub` (SAPG, 16k) but 98,304.
- `ppo_mb98k_g0018_s10{0,1,2}.sub`: exactly `../ppo_vs_sapg/ppo_g0018_s10*.sub` (PPO, 16k) but 98,304.

MAP-Elites' best hand g0018_c00 alone, 3,180 epochs, per-slot noise, adaptive lr capped at 1e-3 from 1e-4 (as the paper
kept SAPG's adaptive lr). 98,304 is computed as 6 gradient-accumulation chunks of 16,384 (`grad_accum_chunks`: one
optimizer step per minibatch, all loss terms are means, so it equals one 98k minibatch at the 16k memory footprint).
Per epoch: 393,216 transitions = 24 minibatches at 16k, 4 at 98k (x 2 mini-epochs).
