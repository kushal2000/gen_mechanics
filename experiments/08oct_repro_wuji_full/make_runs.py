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
