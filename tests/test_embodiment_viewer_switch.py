"""Does switching embodiments in the viewer survive the switch?

The interactive viewer kills its Isaac Sim worker and spawns a new one every
time the design changes, because Kit cannot be re-created in-process. The first
load is exercised by merely starting the viewer. The SWITCH is not, and that is
where the bug was: the Load callback ran on viser's own thread and tore down the
scene graph -- removing the ViserUrdf nodes and nulling the handle -- while the
main thread was already inside ``ViserUrdf.update_cfg`` on those nodes. The
parent died, and the incoming worker then failed with BrokenPipeError trying to
report ready to a process that no longer existed.

Reproducing that by hand means clicking a browser button at the wrong instant.
This does it in one command, and draws 120 frames after each load so a
stale-handle crash has time to fire.

Needs a GPU and boots Kit TWICE (~2-3 min).

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        tests/test_embodiment_viewer_switch.py --run_dir train_dir/.../<population run>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (REPO_ROOT / "train_dir/gen_mechanics/multi_embodiment_control"
               / "mec_population24k_seed0_2026-08-17_15-13-28")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--first", default="gen_0003_00049")
    p.add_argument("--second", default="gen_0003_00000")
    p.add_argument("--port", type=int, default=8092)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / ".hydra" / "config.yaml").is_file():
        print(f"SKIP: no population run at {run_dir}")
        return 0
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = sorted((run_dir / "0_pose_reach_sapg" / "nn").glob("*.pth"))
        if not candidates:
            print(f"SKIP: no checkpoint under {run_dir}")
            return 0
        checkpoint = str(candidates[-1])

    cmd = [
        sys.executable, "-u", "-m", "genmech.eval.embodiments.eval_interactive",
        "--run_dir", str(run_dir), "--checkpoint", checkpoint,
        "--initial_design", args.first, "--selftest", args.second,
        "--design_limit", "8", "--num_envs", "64",
        "--success_tolerance", "0.03", "--port", str(args.port),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        print(f"FAILED: viewer selftest exited {proc.returncode}")
        return 1
    print("embodiment switch test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
