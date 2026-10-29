"""Write one explicit .sub per run of the MLP-vs-transformer comparison (README.md).

    python3 experiments/29oct_mlp_vs_transformer/make_subs.py      # writes <hand>_<arch>_s<seed>.sub here

Same env for every run: pen repose v2 (palm rotation locked, uniform goals, first goal 3 cm), finger_target=home, the
pre-27-Oct tokens (ordered boxes, no joint_origin). MLP: the olympics trainer's own learner (minibatch 114688, 5
mini-epochs, lr 5e-4). Transformer: OlympicsJointTransformerSAPG.yaml's own (width 128, minibatch 16384, 2
mini-epochs, lr 1e-4; ghost + locked-wrist action mask). Action delay OFF (the user's call for these runs; the
7 Oct training default has it on). 5000 epochs, seed 100.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = "debug_outputs/train_logs/29oct_mlp_vs_transformer"
WANDB_PROJECT = "gen_mechanics_mlp_vs_jt"
GR = "assets/urdf/27oct_grammar_random_hands"
HANDS = {                                   # tag: (robot spec, hand joints, note)
    "cd0": ("assets/urdf/27oct_grammar_claude_hands/claude_design_0/claude_design_0.spec.json", 12,
            "claude_design_0, the design agent's grammar hand"),
    "s025": (f"{GR}/coarse_s0_025/coarse_s0_025.spec.json", 13, "coarse_s0_025, the pick-from-100 agent's choice"),
    "s000": (f"{GR}/coarse_s0_000/coarse_s0_000.spec.json", 20, "coarse_s0_000, the first random hand"),
}
SEEDS = (100,)
OLD_OBS = ("[joint_pos,joint_vel,prev_joint_pos,prev_joint_vel,prev_action_targets,joint_link_bbox,joint_lower,"
           "joint_upper,joint_enabled,keypoints_rel_ee,keypoints_rel_goal,object_vel,palm_extents,task_onehot]")
OLD_STATE = OLD_OBS[:-1] + (",object_keypoints_rel_joint,palm_keypoints,ee_pos,ee_rot,ee_vel,object_rot,"
                            "fingertip_pos_rel_palm,closest_keypoint_max_dist,closest_fingertip_dist,progress,"
                            "successes,reward]")
ENV = ("env.goal.first_goal_tolerance=0.03 env.pen_repose.random_first_goal=True env.pen_repose.goal_sampling=uniform "
       f"env.action.wrist_max_speed_ang=0.0 env.action.finger_target=home env.action.use_action_delay=False "
       f"env.obs.obs_list={OLD_OBS} "
       f"env.obs.state_list={OLD_STATE}")

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=MT_{tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --output=/share/portal/kk837/gen_mechanics/{logs}/%x-%j.log
# MLP vs transformer: {note} ({joints} hand joints), {arch_desc}, seed {seed}. Pen repose v2, finger_target=home,
# action delay off, the pre-27-Oct tokens. Written by experiments/29oct_mlp_vs_transformer/make_subs.py (README.md).
#   sbatch experiments/29oct_mlp_vs_transformer/{tag}.sub
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
export HAND={hand}
export HAND_TAG={hand_tag}
export TASKS=pen_repose
export PEN_SHAPE=box
export ARCH={arch}
export NUM_ENVS=24576 {minibatch}EPOCHS=5000 SEED={seed} GAMMA=0.99
export FINGER_RESET=1.0 KEYPOINT=best_so_far
export EXTRA_HYDRA="{env}"
export RUN_TAG=mt_home
export LOG_GROUP=29oct_mlp_vs_transformer
export WANDB_PROJECT={project} WANDB=1
exec bash experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub
"""


def main() -> None:
    for hand_tag, (hand, joints, note) in HANDS.items():
        for arch in ("mlp", "transformer"):
            for seed in SEEDS:
                tag = f"{hand_tag}_{'jt' if arch == 'transformer' else 'mlp'}_s{seed}"
                text = TEMPLATE.format(
                    tag=tag, logs=LOGS, note=note, joints=joints, seed=seed, hand=hand, hand_tag=hand_tag, arch=arch,
                    minibatch="MINIBATCH=114688 " if arch == "mlp" else "", env=ENV, project=WANDB_PROJECT,
                    arch_desc=("the MLP (minibatch 114688, 5 mini-epochs, lr 5e-4)" if arch == "mlp" else
                               "the joint transformer (width 128, minibatch 16384, 2 mini-epochs, lr 1e-4, masks on)"))
                (HERE / f"{tag}.sub").write_text(text)
                print(f"{tag}.sub")


if __name__ == "__main__":
    main()
