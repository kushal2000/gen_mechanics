# Joint-transformer hyperparameter sweep (overnight, 29–30 Sep)

**Goal:** make the joint transformer learn in-hand reorientation as fast as possible, changing only
**learner settings and architecture**. The environment and the observation are fixed.

**BASE** (`make_sweep.py`) is the best transformer configuration as of 29 Sep:
trimmed observation (with the `keypoints_rel_ee` fix), no fall penalty, 5° tolerance, γ 0.998,
lr 5e-4 adaptive, 5 mini-epochs, horizon 16, joint transformer d_model 64 / 4 layers / 1 head /
ff_mult 2 / action head [64] / value head [512,256]. Its reference runs (seed 100, other folder):
real SHARPA 664621 (1 goal/episode at ~epoch 630, level with the MLP), Allegro 664619 (5 at 1336),
gen-SHARPA 664620 (not learning).

**Budget:** 2000 epochs per run (`EPOCHS=2000`, `MAX_CONT=1`). **Metric:** goals/episode at matched
epochs, the first epoch reaching 1 goal/episode, and epochs per hour.

**Method:** rounds of one-change-at-a-time variants against BASE, a second baseline seed to size
the noise, then combine the winners and test them on the other hands.

    .venv_isaacsim/bin/python experiments/29sep_transformer_hparam_sweep/make_sweep.py <round>   # write .subs
    .venv_isaacsim/bin/python experiments/29sep_transformer_hparam_sweep/sweep_status.py [hand]  # rank runs

Launcher knobs added for this sweep (defaults = previous behaviour): `MU_HEAD_UNITS`,
`VALUE_HEAD_UNITS`, `HORIZON` in `run_rank.sh`. Logs: `debug_outputs/train_logs/29sep_transformer_hparam_sweep/`;
wandb project [`gen_mechanics_gen_sharpa_tf_sweep`](https://wandb.ai/kk837/gen_mechanics_gen_sharpa_tf_sweep).

## Round 1 — gen-SHARPA, one change each

The user asked to focus on gen-SHARPA: BASE learns on real SHARPA (level with the MLP) and Allegro, but
not on gen-SHARPA (664620: 0.009 goals/episode at epoch ~840, lr pinned at 1e-2, episodes relapsing to
~50 steps; the gen-SHARPA MLP reached 1 at epoch 944). Round 1 was first submitted on real SHARPA and
cancelled within minutes; it runs on gen-SHARPA. The working SHARPA / Allegro references (664621,
664619) were cancelled to free GPUs; their curves stay on disk.

| variant | change |
|---|---|
| base_s2 | seed 200 (noise estimate) |
| l2 / l6 | 2 / 6 layers |
| d128 | d_model 128 — **out of GPU memory** (667090) at minibatch 114688 with 30 tokens |
| d96 | d_model 96 — cancelled before starting, replaced by d128_ck |
| l6 | **out of GPU memory** (667095) |
| l6_ck / d128_ck | 6 layers / d_model 128 with activation checkpointing (`GRAD_CHECKPOINT`, new network flag): identical outputs and gradients (tested), ~40% of the activation memory, ~1.27× per update step. The learner is unchanged. |
| h4 | 4 attention heads |
| ff4 | ff_mult 4 |
| mu256 | action head [256,128] |
| kl008 | adaptive-lr KL threshold 0.008 (more conservative) |
| clip02 | PPO clip 0.2 |
| const1e3 / const2e3 | constant lr 1e-3 / 2e-3 (the adaptive schedule sat at 1.3e-3–3e-3 during take-off; constant 5e-4 was too slow) |

## Log

**~1 h into round 1 (epochs 330–740).** The baseline is strongly seed-dependent on gen-SHARPA:
seed 200 (`base_s2`) is at 0.07 goals/episode at epoch 561 with ~490-step episodes, while seed 100
(664620) never learned in 1,700 epochs. No variant beats seed 200 yet (clip02 0.04, const2e3 0.04,
const1e3 / kl008 0.02); l2 is worst (0.01 at 738, lr pinned at 1e-2, 187-step episodes). Added
`base_s3` (seed 300) to size the noise.

**~2 h into round 1 (epochs 920–1480).** Separating by lr. Learning: base_s2 (seed 200) 0.42 at
epoch 1000 / 0.57 at 1141 (lr settled ~6e-4), const2e3 0.32 / 0.41. Stalled: clip02 and kl008 (their
adaptive lr drifted to 1e-4; ~0.05–0.09), const1e3 (0.05 at 1000), mu256 (0.08 at 918), and outside the
sweep const1e-4 (0.017 at 1527). Dead: l2 (lr pinned 1e-2, 88-step episodes) and the seed-100 baseline
664620 — both cancelled, with 666081 (const1e-4), to free GPUs. gen-SHARPA MLP for reference: 1.24 at
epoch 1000. Round 2 queued: constant lr 3e-3 and 5e-3, adaptive with kl_threshold 0.032.

**h4 is INVALID — PhysX crashed, training continued.** 667135 logged 50 goals per 50-step episode
from epoch ~22. Cause: a PhysX GPU crash about 13 minutes in (`GPU solveBlockUnified fail to launch
kernel`, `Scene state is corrupted. Simulation cannot continue!`, then `getRigidDynamicData: CUDA
error, code 2` ×45,204 — code 2 is out of memory). The run did not stop: the env returned frozen state
(every cube motionless, env 0's 8 m from the palm, keypoint residual ~0.1 m) and a "success" each step.
No other run tonight or today shows the error. The watcher now greps every sweep log for
`Scene state is corrupted` and cancels such a run. Retried as `h4_ck` (with activation checkpointing).

**Round 1 at the 2000-epoch budget (~4 h).** Every variant ran on seed 100 — the seed on which BASE never
learned (664620: ~0 through 2400 epochs) — so learning at all means the change rescued it.

| run (seed) | @800 | @1000 | @1500 | ~2000 | first epoch at 1 goal/ep |
|---|---|---|---|---|---|
| mu256, action head [256,128] (100) | 0.022 | 0.136 | 1.29 | **7.33** | 1449 |
| base_s2 (200) | 0.211 | 0.421 | 1.86 | **6.94** | 1335 |
| const2e3 (100) | 0.217 | 0.320 | 0.78 | **4.30** | 1453 |
| clip02 / const1e3 / kl008 (100) | ~0.03 | ~0.05 | ~0.11 | 0.15–0.19 | — |
| BASE (100) | 0.012 | 0.009 | 0.007 | ~0 | — |
| l6_ck, 6 layers (100) — running | 0.39 at 797 | | | | |

Low final lr stalls (clip02 and kl008 drift the adaptive lr to 1e-4; const1e3). A bigger action head and a
constant 2e-3 both rescue seed 100; 6 layers is fastest early. Round 3 queued: l6_mu256_ck, mu256_s2,
mu512, l6_ck_s2. gen-SHARPA MLP reference: 1 goal/episode at epoch 944, 38.6 at 2000.

**~5 h. Depth wins.** l6_ck (6 layers, seed 100) reached 1 goal/episode at epoch **975** — the
gen-SHARPA MLP did at 944 — and 3.69 at 1264 (@800 0.40, @1000 1.11). d128_ck helps less (0.23 at 800)
and is 1.7x slower per epoch. Too much lr kills learning: const5e3 (0.014 at 1000, 126-step episodes)
and kl032 (lr pinned 1e-2) are dead and cancelled; const3e3 middling (0.18 at 1000). base_s3 (seed 300)
0.14 at 1000: BASE alone is unreliable across seeds. h4_ck slow (0.016 at 600). Round 4 queued:
l8_ck, l6_const2e3_ck.

**mu512 ran out of memory** (682896) — the action head runs on 30 tokens × 114688; retried as mu512_ck. l6_ck at 5.61 goals/episode at epoch 1369.

**~6.5 h. Depth replicates; the bigger action head does not.**

| run | @600 | @800 | @1000 | @1500 | latest |
|---|---|---|---|---|---|
| l6_ck (seed 100) | 0.081 | 0.403 | 1.11 | **8.69** | **28.6 at 1837** |
| gen-SHARPA MLP | 0.176 | 0.514 | 1.24 | 11.76 | 38.6 at 2000 |
| l6_ck_s2 (seed 200) | 0.051 | 0.297 | | | 0.73 at 932 |
| d128_ck | 0.065 | 0.232 | 0.683 | | 3.94 at 1347 |
| h4_ck | 0.016 | 0.093 | 0.343 | | 0.89 at 1194 |
| base_s3 (seed 300) | 0.044 | 0.111 | 0.137 | 0.369 | 0.71 at 1972 |
| l6_mu256_ck | 0.013 | 0.053 | | | 0.07 at 905 — cancelled |
| mu256_s2 (seed 200) | 0.009 | | | | 0.02 at 652 |

6 layers is at MLP pace on gen-SHARPA and helps on seed 200 too (0.30 vs 0.21 at 800). mu256's seed-100
win did not replicate on seed 200 and hurts combined with depth; l6_mu256_ck and the pending mu512_ck
were cancelled. Round 5 queued: l6_ck_s3 (seed 300).

**Ghost joints (checked on the l6_ck checkpoint at epoch 2000).** gen-SHARPA has **21 real joints and 9
ghost slots** (not 22 / 8 as first stated). Masking is correct: 21 valid per env, matching `joint_enabled`;
real actions have exactly zero gradient w.r.t. ghost token inputs; ghosts stay out of the normaliser's
statistics. Ghost joint_pos inputs sit at the ±10 clamp (normalised by a ~0 range) but reach nothing.
**The problem is ghost ACTIONS:** all 30 action dimensions enter the PPO log-prob, the KL behind the
adaptive lr and the SAPG entropy bonus. Ghost noise grew to sigma ~1.6 (real ~0.8), 44% of the entropy,
and in the transformer every action comes from the shared head, so updates for real joints also move ghost
means. New network flag `mask_ghost_actions` (`MASK_GHOST_ACTIONS`): ghost dimensions get constant mean 0 /
log-std 0 with no gradient — they cancel from the ratio and KL. Per env from that env's raw joint_enabled
(mixed populations handled; arm dims never masked); tested. Round 6: ghost_l4, ghost_l4_s3, ghost_l6_ck.

**Round 7 — overfitting check.** The gen-SHARPA winner (6 layers, default learner) on the other two hands:
`r7_sharpa_l6ck_real` vs real-SHARPA 4-layer 664621 (1 goal/ep at 619) and `r7_allegro_l6ck_alg` vs Allegro
4-layer 664619 (1 at 909, 9.0 at 1543). Same learner, seed 100; no ghosts on these hands.

**~8 h. 6 layers replicates at MLP pace.** l6_ck_s2 (seed 200) reached 1 goal/episode at epoch 973
(l6_ck, seed 100: 975; MLP: 944) and 6.39 at 1385; l6_ck finished at 36.3 (MLP 38.6 at 2000). d128_ck
16.4 at 1708 (1 at 1079), h4_ck 6.05 at 1675 (1 at 1211) — both help, less than depth. Dead and cancelled:
l6_const2e3_ck (56-step episodes: constant lr breaks the deeper model) and mu256_s2 (0.04 at 1269).

**~9 h.** 6 layers generalises to **Allegro** (r7: 1.58 at 800 vs 4-layer 0.657; 1 goal/ep at 738 vs 909)
but **not to every gen-SHARPA seed**: l6_ck_s3 (seed 300) 0.008 at 800 — behind even base_s3 (0.111).
Depth made seeds 100 and 200 reliable, not all. Ghost mask, early: ghost_l4_s3 0.110 at 600 vs base_s3
0.044 (2.5x); ghost_l4 (seed 100) 0.006 at 600, lr still pinned — no rescue yet. Finished at 2000: d128_ck
35.4 (l6_ck 36.3), h4_ck 17.3. l8_ck slower than l6 so far (0.023 vs 0.081 at 600).

**~9.5 h. The ghost mask helps; depth and the mask fix different things.** At epoch 1000: ghost_l4_s3 0.834
(1 goal/ep at 1033) vs base_s3 0.137; ghost_l4 (seed 100) 0.067 vs BASE 0.009, its lr off the 1e-2 cap
(2e-3) — consistent with ghost dimensions distorting the KL behind the adaptive lr. Depth alone does not
fix seed 300 (l6_ck_s3 0.058). l8_ck worse than l6 at every checkpoint (0.050 at 800) — cancelled.
Other hands, 6 vs 4 layers: Allegro 3.87 vs 1.63 at 1000 (10.6 at 1325); real SHARPA 0.109 vs 0.088 at
400. l6_ck_s2 finished at 26.8. Round 8: ghost_l6_ck_s3.
