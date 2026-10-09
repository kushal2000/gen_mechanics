"""Reproduce the 6 Oct full-Wuji v2 runs with today's code: is today's setup sound?

8 Oct's runs on the full hand and the multi-hand families had PPO blow up (KL 37 to 2.7e6, learning rate pinned at the
adaptive ceiling 0.01) where 6 Oct's same-config runs trained. Config, sim code, assets, venv and nodes all match
6 Oct; the one deliberate difference is CAPTURE_VIEWER_INTERVAL (250 env steps then, 10000 now). Four runs, each a
VERBATIM copy of a 6 Oct .sub with only the wandb project, log dir, study name and viewer interval changed:

  R1 nodelay_v250    experiments/06oct_wuji_5rads/left_wuji2_full_v5_ema0.1.sub       = 12491 exactly (40.6 goals/ep)
  R2 nodelay_v10000  the same, today's viewer interval
  R3 dact_v250       experiments/06oct_wuji_5rads/left_wuji2_full_v5_ema0.1_dact.sub  = 21667 exactly (35.6 goals/ep)
  R4 dact_v10000     the same, today's viewer interval (= 113950, a second sample)

Plus two MLP runs (ARCH=mlp: SAPG's [1024, 1024, 512, 512] trunk, shared actor-critic), 24576 envs in SAPG's 6
blocks (block size = envs / 6 = 4096), from R4's sub (EMA 0.1 + action delay, 5 deg, the same PPO settings): is the
8 Oct blow-up in the transformer path?

  M1 mlp_e24k_full          the full left hand
  M2 mlp_e24k_index_middle  the left index + middle hand

And one transformer run at 24576 envs with gradient accumulation (rl_games grad_accum_chunks, added 8 Oct): R2's sub
(EMA 0.1, no action delay, viewer 10000), the minibatch KEPT at 114688 -- so 4 minibatches per mini-epoch, twice
R2's optimizer steps per epoch -- each backward pass split into 4 chunks of 28672 (~4.4 GB of activations, measured
in toy_transformer_memory.py, instead of ~17 GB). (A first try doubled the minibatch to 229376 in 8 chunks, 119011;
cancelled at start: the minibatch stays the 12288-env runs'.)

  A1 accum4_e24k            the full left hand
  A2 dact_accum4_e24k       the same with action delay on (from R4's sub)

And one multi-embodiment transformer run from A2's sub: all 52 Wuji v2 finger-drop hands (multi:wuji2_lr_2plus,
2+ fingers, left and right, no link scaling), 24576 envs, accumulation x4, EMA 0.1 + action delay, and 2 mini-epochs
instead of 5 (8 optimizer steps an epoch instead of 20):

  E1 lr_2plus_dact_accum4_e24k_me2
  E2 lr_2to3_dact_accum4_e24k_me2   the same over the 40 hands with 2 or 3 fingers (multi:wuji2_lr_2to3)
  E3, E4  _me5                      E1 and E2 again with 5 mini-epochs: with 2, neither took off (E1 0.07 goals/episode
                                    at epoch 3000; 6 Oct's 52-hand run, 5 mini-epochs and no delay, had 4.3 by then)

And scaled-link variants of the full hand (assets/urdf/unified_dynamics_commercial_hands/make_scaled_links.py:
finger links only, palm and finger mounts unchanged), as M1 (MLP, 24576 envs, EMA 0.1 + action delay):

  S1 len50_mlp_e24k         every finger link 50% shorter
  S3 len75_mlp_e24k         every finger link 25% shorter
  S2 wid120_mlp_e24k        every finger link 20% wider (one new rest contact, ring/pinky proximal, filtered)

And two more finger subsets as M2 (MLP, 24576 envs, EMA 0.1 + action delay, seed 100): between index + middle
and the full hand, does the thumb -- then the pinky -- buy the OOD robustness?

  F1 thumb_index_middle_mlp_e24k   thumb + index + middle (wuji2_left_only_thumb_index_middle)
  F2 no_ring_mlp_e24k              thumb + index + middle + pinky (wuji2_left_no_ring)

And the index + middle specialist for the multi-embodiment robustness test (plots/h4_...): the 6 Oct 52-hand
transformer (31826) is compared on each hand with a specialist trained identically -- R2's sub (= 12491's config:
transformer, EMA 0.1, no action delay, 12288 envs, seed 100) with only the hand changed:

  T1 index_middle_tf6oct_spec   left index + middle (wuji2_left_only_index_middle)

References, episode_final/successes at epoch 1000 / 1500: 12491 0.71 / 1.81, 21667 0.26 / 0.78. Earliest signal:
info/last_lr decays (healthy) or stays at 0.01 (8 Oct's failures) by epoch ~400-800.

    .venv_isaacsim/bin/python experiments/08oct_repro_wuji_full/make_runs.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "experiments/06oct_wuji_5rads"
LOGS_OLD = "debug_outputs/train_logs/06oct_wuji_5rads"
LOGS_NEW = "debug_outputs/train_logs/08oct_repro_wuji_full"
WANDB = "gen_mechanics_repro_08oct"
RUNS = {"nodelay": "left_wuji2_full_v5_ema0.1", "dact": "left_wuji2_full_v5_ema0.1_dact"}
VIEWER = (250, 10000)

# Extra training seeds of the single-hand MLP runs (the originals are seed 100): is the embodiment ranking -- and the
# OOD robustness ranking -- bigger than seed-to-seed variance? len50 (learns nothing) gets one, to rule out bad luck.
SEEDS = {"full": (101, 102), "index_middle": (101, 102), "wid120": (101, 102), "len75": (101, 102), "len50": (101,)}

FINGERS = {"thumb_index_middle": "wuji2_left_uniform_handonly_only_thumb_index_middle",
           "no_ring": "wuji2_left_uniform_handonly_no_ring"}

SCALED = {"len50": "every finger link 50% shorter", "len75": "every finger link 25% shorter",
          "wid120": "every finger link 20% wider"}

if __name__ == "__main__":
    for tag, src in RUNS.items():
        base = (SRC / f"{src}.sub").read_text()
        for v in VIEWER:
            run = f"{src}_repro_v{v}"
            text = base.replace(LOGS_OLD, LOGS_NEW).replace(src, run)
            text = text.replace("gen_mechanics_minimal_embodiment", WANDB)
            text = re.sub(r"(#SBATCH --job-name=)\S+", rf"\g<1>R_{tag}_v{v}", text)
            text = text.replace("set -euo pipefail", f'export CAPTURE_VIEWER_INTERVAL={v}\nset -euo pipefail', 1)
            assert text.count("CAPTURE_VIEWER_INTERVAL") == 1, "the 6 Oct sub sets no viewer interval (common.sh: 250)"
            p = HERE / f"{run}.sub"
            p.write_text(text)
            print(p)
    for src, run, job, last in (
            ("left_wuji2_full_v5_ema0.1_repro_v10000", "left_wuji2_full_v5_ema0.1_accum4_e24k", "R_nodelay_v10000",
             'env.action.hand_moving_average=0.1"'),
            ("left_wuji2_full_v5_ema0.1_dact_repro_v10000", "left_wuji2_full_v5_ema0.1_dact_accum4_e24k",
             "R_dact_v10000", 'env.domain_randomization.use_action_delay=true"')):
        text = (HERE / f"{src}.sub").read_text().replace(src, run)
        text = text.replace(f"#SBATCH --job-name={job}", f"#SBATCH --job-name=A_{'dact_' if 'dact' in run else ''}accum4_e24k")
        assert "export ARCH=transformer\n" in text and last in text
        text = text.replace("export ARCH=transformer\n",
                            "export ARCH=transformer\nexport NUM_ENVS_PER_GPU=24576 GLOBAL_MINIBATCH=114688\n")
        text = text.replace(last, last[:-1] + ' ++agent.params.config.grad_accum_chunks=4"')
        (HERE / f"{run}.sub").write_text(text)
        print(HERE / f"{run}.sub")
    r4 = (HERE / "left_wuji2_full_v5_ema0.1_dact_repro_v10000.sub").read_text()
    for tag, spec in (("full", "wuji2_left_uniform_handonly"),
                      ("index_middle", "wuji2_left_uniform_handonly_only_index_middle")):
        run = f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k"
        text = r4.replace("left_wuji2_full_v5_ema0.1_dact_repro_v10000", run)
        text = text.replace("#SBATCH --job-name=R_dact_v10000", f"#SBATCH --job-name=M_mlp_{tag}")
        assert "export ARCH=transformer\n" in text
        text = text.replace("export ARCH=transformer\n", "export ARCH=mlp\nexport NUM_ENVS_PER_GPU=24576\n")
        text = text.replace('ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly}"', f'ROBOT_SPEC="${{ROBOT_SPEC:-{spec}}}"')
        assert f"ROBOT_SPEC:-{spec}" in text
        p = HERE / f"{run}.sub"
        p.write_text(text)
        print(p)
    m1 = (HERE / "left_wuji2_full_v5_ema0.1_dact_mlp_e24k.sub").read_text()
    for tag in SCALED:
        run = f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k"
        text = m1.replace("left_wuji2_full_v5_ema0.1_dact_mlp_e24k", run)
        text = text.replace("#SBATCH --job-name=M_mlp_full", f"#SBATCH --job-name=S_mlp_{tag}")
        old = 'ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly}"'
        assert old in text
        text = text.replace(old, f'ROBOT_SPEC="${{ROBOT_SPEC:-wuji2_left_uniform_handonly_{tag}}}"')
        (HERE / f"{run}.sub").write_text(text)
        print(HERE / f"{run}.sub")
    a2 = (HERE / "left_wuji2_full_v5_ema0.1_dact_accum4_e24k.sub").read_text()
    run = "wuji2_lr_2plus_v5_ema0.1_dact_accum4_e24k_me2"
    text = a2.replace("left_wuji2_full_v5_ema0.1_dact_accum4_e24k", run)
    text = text.replace("#SBATCH --job-name=A_dact_accum4_e24k", "#SBATCH --job-name=E_multi52_me2")
    for old, new in (('ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly}"', 'ROBOT_SPEC="${ROBOT_SPEC:-multi:wuji2_lr_2plus}"'),
                     ("MINI_EPOCHS=5 ", "MINI_EPOCHS=2 ")):
        assert old in text, old
        text = text.replace(old, new)
    (HERE / f"{run}.sub").write_text(text)
    print(HERE / f"{run}.sub")
    for tag, seeds in SEEDS.items():
        src = HERE / f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k.sub"
        base = src.read_text()
        for seed in seeds:
            run = f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k_seed{seed}"
            text = base.replace(f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k", run)
            text = re.sub(r"(#SBATCH --job-name=)(\S+)", rf"\g<1>\g<2>_s{seed}", text)
            assert "set -euo pipefail" in text
            text = text.replace("set -euo pipefail", f"export SEED={seed}\nset -euo pipefail", 1)
            (HERE / f"{run}.sub").write_text(text)
            print(HERE / f"{run}.sub")
    e1 = (HERE / "wuji2_lr_2plus_v5_ema0.1_dact_accum4_e24k_me2.sub").read_text()
    run = "wuji2_lr_2to3_v5_ema0.1_dact_accum4_e24k_me2"
    text = e1.replace("wuji2_lr_2plus_v5_ema0.1_dact_accum4_e24k_me2", run)
    text = text.replace("#SBATCH --job-name=E_multi52_me2", "#SBATCH --job-name=E_multi2to3_me2")
    old = 'ROBOT_SPEC="${ROBOT_SPEC:-multi:wuji2_lr_2plus}"'
    assert old in text
    text = text.replace(old, 'ROBOT_SPEC="${ROBOT_SPEC:-multi:wuji2_lr_2to3}"')
    (HERE / f"{run}.sub").write_text(text)
    print(HERE / f"{run}.sub")
    for fam in ("2plus", "2to3"):
        src = f"wuji2_lr_{fam}_v5_ema0.1_dact_accum4_e24k_me2"
        run = src[:-len("_me2")] + "_me5"
        text = (HERE / f"{src}.sub").read_text().replace(src, run)
        text = re.sub(r"(#SBATCH --job-name=)(\S+)_me2", r"\g<1>\g<2>_me5", text)
        assert "MINI_EPOCHS=2 " in text
        text = text.replace("MINI_EPOCHS=2 ", "MINI_EPOCHS=5 ")
        (HERE / f"{run}.sub").write_text(text)
        print(HERE / f"{run}.sub")
    m2 = (HERE / "left_wuji2_index_middle_v5_ema0.1_dact_mlp_e24k.sub").read_text()
    for tag, spec in FINGERS.items():
        run = f"left_wuji2_{tag}_v5_ema0.1_dact_mlp_e24k"
        text = m2.replace("left_wuji2_index_middle_v5_ema0.1_dact_mlp_e24k", run)
        text = text.replace("#SBATCH --job-name=M_mlp_index_middle", f"#SBATCH --job-name=F_mlp_{tag}")
        old = 'ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly_only_index_middle}"'
        assert old in text
        text = text.replace(old, f'ROBOT_SPEC="${{ROBOT_SPEC:-{spec}}}"')
        (HERE / f"{run}.sub").write_text(text)
        print(HERE / f"{run}.sub")
    r2 = (HERE / "left_wuji2_full_v5_ema0.1_repro_v10000.sub").read_text()
    run = "left_wuji2_index_middle_v5_ema0.1_tf6oct_spec"
    text = r2.replace("left_wuji2_full_v5_ema0.1_repro_v10000", run)
    text = text.replace("#SBATCH --job-name=R_nodelay_v10000", "#SBATCH --job-name=T_tf_index_middle")
    old = 'ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly}"'
    assert old in text and "use_action_delay" not in text
    text = text.replace(old, 'ROBOT_SPEC="${ROBOT_SPEC:-wuji2_left_uniform_handonly_only_index_middle}"')
    (HERE / f"{run}.sub").write_text(text)
    print(HERE / f"{run}.sub")
