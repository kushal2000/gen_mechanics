"""Write one explicit .sub per transformer-debug arm (README.md has the why).

    python3 experiments/27oct_transformer_debug/make_subs.py <round>      # writes <round>_<arm>.sub here

Every arm is the base below -- repose_coarse_s0_000_jt_d32.sub, the transformer run that stalls (job 697372) -- plus
its own overrides, for EPOCHS epochs on hand coarse_s0_000, seed SEED. The overrides go last in EXTRA_HYDRA, so they
win over the trainer's own (hydra applies repeated overrides in order). Each arm also logs per-update gradient / step /
KL stats (coevolution/utils/update_stats.py) next to its training log.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = "debug_outputs/train_logs/27oct_transformer_debug"
WANDB_PROJECT = "gen_mechanics_jt_debug"

TASK = ("env.goal.first_goal_tolerance=0.03 env.pen_repose.random_first_goal=True env.pen_repose.goal_sampling=uniform "
        "env.action.wrist_max_speed_ang=0.0 env.obs.joint_box=extent")
BASE = {                                  # the stalled run's network / learner (repose_coarse_s0_000_jt_d32.sub)
    "agent.params.network.d_model": "32",
    "agent.params.network.compile_net": "true",
    "agent.params.config.minibatch_size": "114688",
    "agent.params.config.mini_epochs": "2",
    "agent.params.config.learning_rate": "1e-4",
}
OBS_NO_JOINT_ORIGIN = ("[joint_pos,joint_vel,prev_joint_pos,prev_joint_vel,prev_action_targets,joint_link_bbox,"
                       "joint_lower,joint_upper,joint_enabled,keypoints_rel_ee,keypoints_rel_goal,object_vel,"
                       "palm_extents,task_onehot]")
STATE_NO_JOINT_ORIGIN = OBS_NO_JOINT_ORIGIN[:-1] + (",object_keypoints_rel_joint,palm_keypoints,ee_pos,ee_rot,ee_vel,"
                                                    "object_rot,fingertip_pos_rel_palm,closest_keypoint_max_dist,"
                                                    "closest_fingertip_dist,progress,successes,reward]")

ROUNDS: dict[str, dict] = {
    "r1": {
        "epochs": 1500, "seed": 100,
        "arms": {
            # name: (what it tests, overrides)
            "critic1": ("the value loss crowding the shared trunk: critic_coef 4 -> 1",
                        {"agent.params.config.critic_coef": "1"}),
            "maxlr": ("the adaptive lr wandering up (to 4.7e-3): cap it at 2e-4",
                      {"++agent.params.config.max_lr": "2e-4"}),
            "lrconst": ("no adaptive lr at all: a constant 1e-4",
                        {"agent.params.config.lr_schedule": "constant"}),
            "fp32": ("fp16 autocast in attention / LayerNorm: full fp32 (each minibatch's gradient accumulated over 2 "
                     "chunks: the same update, half the activation memory; fp32 alone ran out at epoch ~100, job 707328)",
                     {"agent.params.config.mixed_precision": "false", "++agent.params.config.grad_accum_chunks": "2"}),
            "d64": ("width: the 32-wide global token as the only route from goal to fingers (H1): width 64 "
                    "(minibatch halved to fit: 57344)",
                    {"agent.params.network.d_model": "64", "agent.params.config.minibatch_size": "57344"}),
            "heads4": ("one attention head for 20 joints (H2): 4 heads at width 32",
                       {"agent.params.network.n_heads": "4"}),
            "wristsmall": ("the wrist head absorbing the update (H7): arm head [1024, 512] -> [256, 128]",
                           {"agent.params.network.arm_head_units": "[256,128]"}),
            "mu64": ("each joint's action is one Linear on its token: a per-joint MLP head [64] (the 30 Sep default "
                     "had one)", {"++agent.params.network.mu_head_units": "[64]"}),
            "net30sep": ("the 30 Sep default network: width 64, ff x2, mu head [64], value [512, 256] "
                         "(minibatch 57344 to fit)",
                         {"agent.params.network.d_model": "64", "agent.params.network.ff_mult": "2",
                          "++agent.params.network.mu_head_units": "[64]",
                          "agent.params.network.value_head_units": "[512,256]",
                          "agent.params.config.minibatch_size": "57344"}),
            "oldtokens": ("the 27 Oct tokens: back to ordered boxes, no joint_origin (the env drops to ordered "
                          "boxes when obs_list lacks joint_origin)",
                          {"env.obs.obs_list": OBS_NO_JOINT_ORIGIN, "env.obs.state_list": STATE_NO_JOINT_ORIGIN}),
        },
    },
}

ROUNDS["diag"] = {
    "epochs": 400, "seed": 100,
    "arms": {
        "cancel": ("the shared finger head's per-joint gradients cancelling: the base with compile off, so the "
                   "update stats' hooks see head/cancel and head/token_cos (numerically the same network)",
                   {"agent.params.network.compile_net": "false"}),
    },
}

# Round 2: controls. Each arm may carry a third element, a dict of what it changes beyond hydra overrides:
#   hand / hand_tag / tasks / arch, "base" (the network + learner it starts from: "jt" = BASE above, "jt14oct" = the
#   transformer YAML's own -- width 128, minibatch 16384 -- as the 14 Oct olympics runs had it, "mlp"), "task" (the
#   task overrides instead of TASK) and "minibatch" (the MLP's, through the trainer's MINIBATCH).
WUJI_TI = "wuji2_left_uniform_handonly_only_thumb_index"
OLD_OBS = {"env.obs.obs_list": OBS_NO_JOINT_ORIGIN, "env.obs.state_list": STATE_NO_JOINT_ORIGIN}
ROUNDS["r2"] = {
    "epochs": 1500, "seed": 102,
    "arms": {
        "ctrl_ti": ("POSITIVE CONTROL: the 14 Oct transformer run that learned (job 257159: Wuji thumb + index, pen lift, "
                    "seed 102, width 128, minibatch 16384, wrist rotation free at 1.5 rad/s, the old tokens) on today's "
                    "code. If it does not learn, something regressed since 14 Oct",
                    {**OLD_OBS, "env.action.wrist_max_speed_ang": "1.5"},
                    {"hand": WUJI_TI, "hand_tag": "wuji_ti", "tasks": "pen_lift", "base": "jt14oct", "task": ""}),
        "ti_locked": ("the control with the wrist's rotation locked (the 15 Oct default): did the 14 Oct transformer "
                      "need the wrist to turn the pen?",
                      {**OLD_OBS, "env.action.wrist_max_speed_ang": "0.0"},
                      {"hand": WUJI_TI, "hand_tag": "wuji_ti", "tasks": "pen_lift", "base": "jt14oct", "task": ""}),
        "s000_lift": ("is it repose or the hand: the base transformer on hand coarse_s0_000, pen lift, wrist rotation "
                      "locked", {"env.action.wrist_max_speed_ang": "0.0", "env.obs.joint_box": "extent"},
                      {"tasks": "pen_lift", "base": "jt", "task": ""}),
        "s000_lift_mlp": ("the MLP on the same: hand coarse_s0_000, pen lift, wrist rotation locked",
                          {"env.action.wrist_max_speed_ang": "0.0", "env.obs.joint_box": "extent"},
                          {"tasks": "pen_lift", "base": "mlp", "task": "", "arch": "mlp", "minibatch": "114688"}),
    },
}

# Round 3: the joint count. The 14 Oct transformer setup (width 128, minibatch 16384, the old tokens: verified on
# today's code by r2_ctrl_ti) on hands from few joints to 20, on repose with the wrist locked, against MLPs.
REPOSE_OLD = {**OLD_OBS, "env.action.wrist_max_speed_ang": "0.0"}
GR = "assets/urdf/27oct_grammar_random_hands"
ROUNDS["r3"] = {
    "epochs": 1500, "seed": 100,
    "arms": {
        "ti_repose": ("does the task alone break it: the transformer that learns thumb + index pen lift, on thumb + "
                      "index pen repose", REPOSE_OLD,
                      {"hand": WUJI_TI, "hand_tag": "wuji_ti", "base": "jt14oct", "seed": "102"}),
        "ti_repose_mlp": ("its MLP: thumb + index pen repose", REPOSE_OLD,
                          {"hand": WUJI_TI, "hand_tag": "wuji_ti", "base": "mlp", "arch": "mlp",
                           "minibatch": "114688", "seed": "102"}),
        "cd0": ("12 joints: the 14 Oct transformer on claude_design_0 (its MLP, job 695902, seed 100, the old tokens: "
                "23.9 goals per episode, first 1 at epoch 705)", REPOSE_OLD,
                {"hand": "assets/urdf/27oct_grammar_claude_hands/claude_design_0/claude_design_0.spec.json",
                 "hand_tag": "cd0", "base": "jt14oct"}),
        "s025": ("13 joints: the 14 Oct transformer on coarse_s0_025 (its MLP, job 695903: 26.8 goals per episode, "
                 "first 1 at epoch 848)", REPOSE_OLD,
                 {"hand": f"{GR}/coarse_s0_025/coarse_s0_025.spec.json", "hand_tag": "s025", "base": "jt14oct"}),
        "s000_jt14": ("20 joints: the 14 Oct transformer on coarse_s0_000 (its MLP, job 694301: 10.2 goals; the width-128 "
                      "run 696227 was cancelled at epoch 344)", REPOSE_OLD, {"base": "jt14oct"}),
    },
}

# Round 4: does the fast config (BASE: width 32, minibatch 114688, the new tokens) learn where the 14 Oct setup does,
# and do hand 000's two best round-1 arms catch up given time (resumed from epoch 1500 to 3000)?
ROUNDS["r4"] = {
    "epochs": 1500, "seed": 100,
    "arms": {
        "s025_fast": ("the fast config (BASE) on coarse_s0_025, where the 14 Oct setup learns (r3_s025)", {},
                      {"hand": f"{GR}/coarse_s0_025/coarse_s0_025.spec.json", "hand_tag": "s025"}),
        "cd0_fast": ("the fast config (BASE) on claude_design_0 (r3_cd0)", {},
                     {"hand": "assets/urdf/27oct_grammar_claude_hands/claude_design_0/claude_design_0.spec.json",
                       "hand_tag": "cd0"}),
        "d64_more": ("r1_d64 resumed at epoch 1500 to 3000: does width 64 catch the MLP on hand 000?",
                     {"agent.params.network.d_model": "64", "agent.params.config.minibatch_size": "57344"},
                     {"checkpoint": "debug_outputs/train_logs/27oct_transformer_debug/0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_d64_707455/0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_d64_707455/nn/last_0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_d64_707455_ep_1500_rew__285.38284_.pth", "epochs": "3000"}),
        "net30sep_more": ("r1_net30sep resumed at epoch 1500 to 3000",
                          {"agent.params.network.d_model": "64", "agent.params.network.ff_mult": "2",
                            "++agent.params.network.mu_head_units": "[64]",
                            "agent.params.network.value_head_units": "[512,256]",
                            "agent.params.config.minibatch_size": "57344"},
                          {"checkpoint": "debug_outputs/train_logs/27oct_transformer_debug/0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_net30sep_707456/0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_net30sep_707456/nn/last_0_pen_repose_sapg_transformer_e24576_s100_g0.99_fr1.0_kpbest_so_far_box_s0_000_jd_r1_net30sep_707456_ep_1500_rew__198.98444_.pth", "epochs": "3000"}),
    },
}

# Round 5: which half of the fast config stalls hand 000 -- the width (32 vs 128) or the minibatch (114688 vs 16384:
# 8 vs 48 updates per epoch)? Against r3_s000_jt14 (the 14 Oct setup, which learns). Also: the new tokens in the
# setup that works.
ROUNDS["r5"] = {
    "epochs": 1500, "seed": 100,
    "arms": {
        "d128_bigbatch": ("width 128 (the 14 Oct setup) with the fast config's minibatch 114688, accumulated over 7 "
                          "chunks of 16384 to fit: 8 updates per epoch", {**REPOSE_OLD,
                          "agent.params.config.minibatch_size": "114688",
                          "++agent.params.config.grad_accum_chunks": "7"}, {"base": "jt14oct"}),
        "d32_smallbatch": ("width 32 (the fast config's) with the 14 Oct minibatch 16384: 48 updates per epoch",
                           {**REPOSE_OLD, "agent.params.network.d_model": "32"}, {"base": "jt14oct"}),
        "newtokens": ("the 14 Oct setup with the 27 Oct tokens (extent boxes + joint_origin)",
                      {"env.action.wrist_max_speed_ang": "0.0"}, {"base": "jt14oct"}),
    },
}

# Round 6: are the 27 Oct tokens (extent boxes + joint_origin) worse? On hand 000 both networks were behind with them
# (MLP 705905 vs 694301; r5_newtokens vs r3_s000_jt14), single seeds. More seeds, old vs new, same everything else.
NEW_TOK = {"env.action.wrist_max_speed_ang": "0.0"}          # TASK already sets joint_box=extent; the default obs lists
MLP_META = {"base": "mlp", "arch": "mlp", "minibatch": "114688", "epochs": "2500"}
ROUNDS["r6"] = {
    "epochs": 1500, "seed": 101,
    "arms": {
        **{f"mlp_old_s{sd}": (f"MLP, the old tokens, seed {sd}", REPOSE_OLD, {**MLP_META, "seed": str(sd)})
           for sd in (101, 102)},
        **{f"mlp_new_s{sd}": (f"MLP, the 27 Oct tokens, seed {sd}", NEW_TOK, {**MLP_META, "seed": str(sd)})
           for sd in (101, 102)},
        "jt14_old_s101": ("the 14 Oct transformer setup, the old tokens, seed 101", REPOSE_OLD, {"base": "jt14oct"}),
        "jt14_new_s101": ("the 14 Oct transformer setup, the 27 Oct tokens, seed 101", NEW_TOK, {"base": "jt14oct"}),
    },
}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=JD_{tag}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
#SBATCH --gres=gpu:1
#SBATCH --output=/share/portal/kk837/gen_mechanics/{logs}/%x-%j.log
# Transformer debug, {rnd} arm "{arm}": {why}.
# {base_desc}; this arm changes only:
{changes}
# Written by experiments/27oct_transformer_debug/make_subs.py.
#   sbatch experiments/27oct_transformer_debug/{tag}.sub
set -euo pipefail
cd /share/portal/kk837/gen_mechanics
export GENMECH_UPDATE_STATS={logs}/{tag}.update_stats.jsonl
export HAND={hand}
export HAND_TAG={hand_tag}
export TASKS={tasks}
export PEN_SHAPE=box
export ARCH={arch}
export NUM_ENVS=24576 {minibatch}EPOCHS={epochs} SEED={seed} GAMMA=0.99
export FINGER_RESET=1.0 KEYPOINT=best_so_far
export EXTRA_HYDRA="{extra}"
{checkpoint}export RUN_TAG=jd_{tag}
export LOG_GROUP=27oct_transformer_debug
export WANDB_PROJECT={project} WANDB=1
exec bash experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub
"""


def main() -> None:
    rnd = sys.argv[1]
    spec = ROUNDS[rnd]
    for arm, (why, over, *meta) in spec["arms"].items():
        m = meta[0] if meta else {}
        base = m.get("base", "jt")
        settings = {**(BASE if base == "jt" else {}), **over}
        task = m.get("task", TASK)
        extra = " ".join(([task] if task else []) + [f"{k}={v}" for k, v in settings.items()])
        tag = f"{rnd}_{arm}"
        changes = "\n".join(f"#   {k}={v}" for k, v in over.items()) + "".join(
            f"\n#   {k}: {v}" for k, v in m.items() if k not in ("base", "task"))
        base_desc = {"jt": "The base is repose_coarse_s0_000_jt_d32.sub (hand coarse_s0_000, pen repose v2, width 32, "
                           "minibatch 114688, 2\n# mini-epochs, lr 1e-4 adaptive, compiled, extent tokens)",
                     "jt14oct": "The network and learner are the transformer YAML's own (OlympicsJointTransformerSAPG: "
                                "width 128, 4 layers,\n# minibatch 16384, 2 mini-epochs, lr 1e-4), as the 14 Oct "
                                "olympics transformer runs had them",
                     "mlp": "The MLP with the olympics trainer's own learner (minibatch 114688, 5 mini-epochs, lr 5e-4)"}[base]
        text = TEMPLATE.format(tag=tag, logs=LOGS, rnd=rnd, arm=arm, why=why, changes=changes, epochs=m.get("epochs", spec["epochs"]),
                               seed=m.get("seed", spec["seed"]),
                               checkpoint=(f"export CHECKPOINT={m['checkpoint']} CHECKPOINT_MODE=resume\n"
                                           if "checkpoint" in m else ""), extra=extra, project=WANDB_PROJECT, base_desc=base_desc,
                               hand=m.get("hand", "assets/urdf/27oct_grammar_random_hands/coarse_s0_000/coarse_s0_000.spec.json"),
                               hand_tag=m.get("hand_tag", "s0_000"), tasks=m.get("tasks", "pen_repose"),
                               arch=m.get("arch", "transformer"),
                               minibatch=f"MINIBATCH={m['minibatch']} " if "minibatch" in m else "")
        (HERE / f"{tag}.sub").write_text(text)
        print(f"{tag}.sub")


if __name__ == "__main__":
    main()
