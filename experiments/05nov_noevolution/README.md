# No-evolution training (5 Nov 2026)

One shared per-finger policy trained on a **fixed** population of 4,096 random hands, 128 in each of 32 niches, in
1.024B-step iterations. No mutation and no search: this is the House of Dextra protocol in our setup, and the
baseline each design-search scheme gets compared against. Because every iteration saves its policy and scores every
hand, design search can start from any iteration later.

## The population (`make_seed.py`)

- Rules: `FEETECH_COARSE_NOCUT_RULES`: no palm cuts; a hand is its fingers' arrangement; the mount is derived (the
  clear point in the base servos' hull nearest the palm centre); built hands put the wrist frame at the palm centre.
- Niches: `descriptors.NICHES32`, i.e. finger count (2-6) x total motors in pairs. Every cell is reached by at least 1% of
  its finger count's random hands, and the smallest is 4-5 motors. Hands outside the map are not used.
- 128 random hands per niche (the first distinct ones from a seeded, chunked draw), built under
  `debug_outputs/noevolution/seed_4096/`, listed with their niche in `manifest.json` (slot grid 6 x 5).

## Training (`noevolution.sub` -> `experiments/oct_2026_experiments/29oct_evolution_from_scratch/generation.sh`, strategy `static`)

- **One job per iteration:** train 2,604 epochs (24,576 envs x 16 steps = 1.024B steps) from the last checkpoint,
  keep `gen_NNNN/policy.pth`, evaluate every hand (`eval_hands.py`, 4 envs each, first episode) into
  `gen_NNNN/fitness.json`, then resubmit. The run directory is `debug_outputs/evolution/noevolution_4096/`.
- **Envs:** 6 per hand, dealt round-robin, so each SAPG block (4,096 envs) holds every hand exactly once.
- **Joint token:** state (5) + origin (3) + axis (3) + link vector (3) + limits (2) = 16 numbers per joint, with no
  link box.
- **Learner:** SAPG, one shared log-std clamped to [-5, 2], fixed lr 3e-4, fp16, 16k minibatch, 2 mini-epochs,
  gamma 0.99. Task: pen repose v2 as in the 29 Oct runs.

## Logging (wandb project `gen_mechanics_noevolution`, a run per iteration, group `noev4096`)

- **Section `niche_<cell>/`, one per niche:**
  - min / avg / max over its 128 hands of each hand's mean episode successes, return, length and every reward
    term's episode sum (`env.log`, every 400 steps);
  - the avg of each termination reason;
  - `designs_reporting`;
  - the niche's interactive viewer (its first hand).
- **Sections `niches_successes/` and `niches_return/`:** every niche's avg side by side.
- **The viewer:** links coloured by joint kind (flexion blue, abduction red, roll green; roll tips striped). Meshes are
  embedded in the page, so nothing needs to be pushed anywhere.

```
sbatch experiments/05nov_noevolution/noevolution.sub              # 10 iterations
ROUNDS=1 sbatch experiments/05nov_noevolution/noevolution.sub     # the 1B pilot
```

## Studies on the no-evolution policies (5-6 Nov, `studies/`)

All zero-shot under a no-evolution iteration's policy, 100-env scores as the truth unless stated.

| study | script | question -> answer |
|---|---|---|
| 1 | `study_eval_noise.py` | How noisy is the loop's 4-env eval? Corr 0.84 with 100 envs; the top 8 per niche are inflated ~0.27 (winner's curse). |
| 2 | `study_mutations.py` | What does each one-step mutation do zero-shot? Length edits neutral; kind swaps and finger removal cost most. |
| 3 | `study_value.py`, `eval_values.py` | Critic vs explicit eval for ranking children: the critic over 1 s matches an 8-env eval; a design-only net matches a 4-env eval. |
| 4 | `study_design_value.py`, `eval_training_like.py` | Training-log labels (all SAPG blocks) match 100-env scores (corr 0.96) and make the better value net. |
| 5 | `study_beam.py` | Beam search (width 10, every child simulated) from one top design: 0.935 -> ~1.3 in ~12 rounds, then flat. |
| 6 | `study_vf_beam.py` | Beam search on a value net alone: real but modest gains, wildly over-predicted. |
| 7 | `study_vf_archive.py` (+ `mutate_shard.*`) | MAP-Elites on a frozen value net (32 niches x 128): archive mean 0.26 -> 0.66 true in one round; 0.85 by round 6, while the prediction drifts to 1.20. Plots in `plots/vf_archive/`. |

Design value net: `design_value_net.py` (one per-finger transformer from scratch on design-only tokens; tuned in
`study_vf_hparams.py` / `study_vf_tweaks.py` on population and beam designs: dropout 0.3, lr 3e-4, 15k steps).
Viewers: `viser_play_design.py` (play the policy on a chosen design), `viser_children.py` (a design's children by
mutation), `viser_beam.py` (the beam search), `viser_population.py`; shared drawing in `viser_hands.py`.
`studies/diag_reset_table.py` measures resets that start a finger in the table.
