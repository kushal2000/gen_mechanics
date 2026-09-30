"""Write one .sub per variant. Every variant is BASE plus the exports it names -- nothing else.

BASE is the best transformer configuration as of 29 Sep: trimmed observation (with the
keypoints_rel_ee fix), no fall penalty, gamma 0.998, lr 5e-4 adaptive, 5 mini-epochs, 5 deg,
joint transformer d_model 64 / 4 layers / 1 head / ff_mult 2 / mu head [64] / value head [512,256].
Every run has a 2000-epoch budget (EPOCHS=2000, MAX_CONT=1).
The environment and observation are fixed across the sweep: only learner and architecture move.

    python experiments/29sep_transformer_hparam_sweep/make_sweep.py <round> [name ...]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOGS = REPO / "debug_outputs/train_logs/29sep_transformer_hparam_sweep"
HANDS = {
    "sharpa": "sharpa_handonly",
    "allegro": "allegro_handonly",
    "gen_sharpa": "handonly:/share/portal/kk837/gen_mechanics/assets/populations/sharpa_capsule.json",
}
OBS_LIST = ("[joint_pos,joint_vel,prev_joint_pos,prev_joint_vel,prev_action_targets,joint_link_bbox,"
            "joint_lower,joint_upper,joint_enabled,keypoints_rel_ee,keypoints_rel_goal,object_vel]")
BASE = {
    "ARCH": "transformer", "OBS_LIST": f'"{OBS_LIST}"',
    "USER_HYDRA": '"agent.params.config.gamma=0.998"',
    "LEARNING_RATE": "0.0005", "MINI_EPOCHS": "5",
    "D_MODEL": "64", "TRANSFORMER_LAYERS": "4", "N_HEADS": "1", "FF_MULT": "2",
    "MU_HEAD_UNITS": '"[64]"', "VALUE_HEAD_UNITS": '"[512,256]"', "HORIZON": "16",
    "GRAD_CHECKPOINT": "false",
    "LR_SCHEDULE": "adaptive",
}
UH = BASE["USER_HYDRA"].strip('"')

# name -> (hand, overrides). Round 1: one change at a time on gen-SHARPA -- the one hand where BASE does
# not learn (664620: adaptive lr pinned at 1e-2, episodes relapsing to ~50 steps) -- plus a second seed
# of the baseline to measure run-to-run noise.
ROUNDS = {
    1: {
        "base_s2":     ("gen_sharpa", {"SEED": "200"}),
        # Seed 100 (664620) never learned; seed 200 was at 0.07 by epoch 561. A third seed sizes the noise.
        "base_s3":     ("gen_sharpa", {"SEED": "300"}),
        "l2":          ("gen_sharpa", {"TRANSFORMER_LAYERS": "2"}),
        "l6":          ("gen_sharpa", {"TRANSFORMER_LAYERS": "6"}),
        # d128 ran out of GPU memory (48 GB) at minibatch 114688 with gen-SHARPA's 30 tokens (667090).
        # Halving the minibatch would double the gradient steps too, so width is tested at 96 instead.
        "d128":        ("gen_sharpa", {"D_MODEL": "128"}),
        "d96":         ("gen_sharpa", {"D_MODEL": "96"}),
        # l6 also ran out of memory (667095). Activation checkpointing recomputes layer activations in
        # the backward pass: identical outputs and gradients (tested), ~40% of the memory, ~1.27x per
        # update step. So depth and width are tested with it on and the learner unchanged.
        "l6_ck":       ("gen_sharpa", {"TRANSFORMER_LAYERS": "6", "GRAD_CHECKPOINT": "true"}),
        "d128_ck":     ("gen_sharpa", {"D_MODEL": "128", "GRAD_CHECKPOINT": "true"}),
        "h4":          ("gen_sharpa", {"N_HEADS": "4"}),
        # h4 (667135) hit a PhysX GPU crash ("Scene state is corrupted", CUDA error 2 = out of memory) ~13 min
        # in; training carried on against a frozen sim and logged 50 "goals" per 50-step episode. Invalid.
        # Retried with activation checkpointing (identical maths, far less training memory).
        "h4_ck":       ("gen_sharpa", {"N_HEADS": "4", "GRAD_CHECKPOINT": "true"}),
        "ff4":         ("gen_sharpa", {"FF_MULT": "4"}),
        "mu256":       ("gen_sharpa", {"MU_HEAD_UNITS": '"[256,128]"'}),
        "kl008":       ("gen_sharpa", {"USER_HYDRA": f'"{UH} agent.params.config.kl_threshold=0.008"'}),
        "clip02":      ("gen_sharpa", {"USER_HYDRA": f'"{UH} agent.params.config.e_clip=0.2"'}),
        # Constant lr at the level the adaptive schedule sat at while the working SHARPA transformer took
        # off (1.3e-3..3e-3); constant 5e-4 (665191, SHARPA) and 1e-4 (666081, gen-SHARPA) were too low.
        "const1e3":    ("gen_sharpa", {"LR_SCHEDULE": "constant", "LEARNING_RATE": "0.001"}),
        "const2e3":    ("gen_sharpa", {"LR_SCHEDULE": "constant", "LEARNING_RATE": "0.002"}),
    },
    # Round 2 (queued at ~epoch 1100 of round 1): the lr direction. Runs whose lr stayed high learned
    # (baseline seed 200 at ~6e-4..2e-3: 0.42 at epoch 1000; constant 2e-3: 0.32), runs whose lr ended low
    # stalled (clip02 / kl008 drifted to 1e-4: ~0.05-0.07; constant 1e-3: 0.05; constant 1e-4: 0.012).
    2: {
        "const3e3":    ("gen_sharpa", {"LR_SCHEDULE": "constant", "LEARNING_RATE": "0.003"}),
        "const5e3":    ("gen_sharpa", {"LR_SCHEDULE": "constant", "LEARNING_RATE": "0.005"}),
        "kl032":       ("gen_sharpa", {"USER_HYDRA": f'"{UH} agent.params.config.kl_threshold=0.032"'}),
    },
    # Round 3 (queued at ~epoch 2000 of round 1). Every round-1 variant used seed 100, the seed on which
    # BASE never learned (664620). Rescued it: mu256 (7.33 at 2000), const2e3 (4.30); l6_ck strongest
    # early (0.39 at ~800, ~2x base seed 200). Combine, and replicate on seed 200.
    3: {
        "l6_mu256_ck": ("gen_sharpa", {"TRANSFORMER_LAYERS": "6", "MU_HEAD_UNITS": '"[256,128]"', "GRAD_CHECKPOINT": "true"}),
        "mu256_s2":    ("gen_sharpa", {"MU_HEAD_UNITS": '"[256,128]"', "SEED": "200"}),
        "mu512":       ("gen_sharpa", {"MU_HEAD_UNITS": '"[512,256]"'}),
        # mu512 ran out of memory (682896): the head runs on all 30 tokens x the minibatch. Checkpointing the
        # layers frees enough; identical maths.
        "mu512_ck":    ("gen_sharpa", {"MU_HEAD_UNITS": '"[512,256]"', "GRAD_CHECKPOINT": "true"}),
        "l6_ck_s2":    ("gen_sharpa", {"TRANSFORMER_LAYERS": "6", "GRAD_CHECKPOINT": "true", "SEED": "200"}),
    },
    # Round 4: depth. l6_ck (seed 100) reached 1 goal/episode at epoch 975 -- the gen-SHARPA MLP did at 944.
    4: {
        "l8_ck":          ("gen_sharpa", {"TRANSFORMER_LAYERS": "8", "GRAD_CHECKPOINT": "true"}),
        "l6_const2e3_ck": ("gen_sharpa", {"TRANSFORMER_LAYERS": "6", "GRAD_CHECKPOINT": "true",
                                          "LR_SCHEDULE": "constant", "LEARNING_RATE": "0.002"}),
    },
}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=sw{rnd}_{name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128000
#SBATCH --time=24:00:00
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@900
#SBATCH --output={logs}/r{rnd}_{name}-%j.log
#SBATCH --error={logs}/r{rnd}_{name}-%j.err

# TRANSFORMER SWEEP, round {rnd}, variant "{name}" on {hand}. Generated by make_sweep.py -- edit
# that, not this. Differs from the sweep BASE only in: {diff}
{exports}
set -euo pipefail
export ROBOT_SPEC="${{ROBOT_SPEC:-{spec}}}"
export STUDY_ID="${{STUDY_ID:-sw{rnd}_{hand}_{name}}}"
export SUCCESS_TOLERANCE_DEG=5
# 2000-epoch budget per run, no chained continuation.
export EPOCHS="${{EPOCHS:-2000}}" MAX_CONT="${{MAX_CONT:-1}}" CONT="${{CONT:-1}}"
export WANDB_PROJECT="${{WANDB_PROJECT:-gen_mechanics_gen_sharpa_tf_sweep}}"
export SCALING_RUN_ROOT="${{SCALING_RUN_ROOT:-{logs}}}"

source {repo}/experiments/28sep_inhand_reorientation/common.sh
run_and_chain "$0"
"""

def write(rnd, name):
    hand, over = ROUNDS[rnd][name]
    env = dict(BASE, **over)
    diff = ", ".join(f"{k}={v}" for k, v in over.items()) or "nothing (baseline)"
    exports = "\n".join(f"export {k}={v}" for k, v in env.items())
    p = HERE / f"r{rnd}_{hand}_{name}.sub"
    p.write_text(TEMPLATE.format(rnd=rnd, name=name, hand=hand, spec=HANDS[hand], diff=diff,
                                 exports=exports, logs=LOGS, repo=REPO))
    return p

if __name__ == "__main__":
    rnd = int(sys.argv[1]); names = sys.argv[2:] or list(ROUNDS[rnd])
    for n in names:
        print(write(rnd, n))
