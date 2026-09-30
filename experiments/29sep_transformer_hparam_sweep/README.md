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
