"""Reproduce the 6 Oct full-Wuji v2 runs with today's code: is today's setup sound?

8 Oct's runs on the full hand and the multi-hand families had PPO blow up (KL 37 to 2.7e6, learning rate pinned at the
adaptive ceiling 0.01) where 6 Oct's same-config runs trained. Config, sim code, assets, venv and nodes all match
6 Oct; the one deliberate difference is CAPTURE_VIEWER_INTERVAL (250 env steps then, 10000 now). Four runs, each a
VERBATIM copy of a 6 Oct .sub with only the wandb project, log dir, study name and viewer interval changed:

  R1 nodelay_v250    experiments/06oct_wuji_5rads/left_wuji2_full_v5_ema0.1.sub       = 12491 exactly (40.6 goals/ep)
  R2 nodelay_v10000  the same, today's viewer interval
  R3 dact_v250       experiments/06oct_wuji_5rads/left_wuji2_full_v5_ema0.1_dact.sub  = 21667 exactly (35.6 goals/ep)
  R4 dact_v10000     the same, today's viewer interval (= 113950, a second sample)

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
