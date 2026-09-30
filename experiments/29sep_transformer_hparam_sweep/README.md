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
| d128 | d_model 128 |
| h4 | 4 attention heads |
| ff4 | ff_mult 4 |
| mu256 | action head [256,128] |
| kl008 | adaptive-lr KL threshold 0.008 (more conservative) |
| clip02 | PPO clip 0.2 |
| const1e3 / const2e3 | constant lr 1e-3 / 2e-3 (the adaptive schedule sat at 1.3e-3–3e-3 during take-off; constant 5e-4 was too slow) |

## Log

(results appended per round)
