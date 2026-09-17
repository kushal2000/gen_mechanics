"""Print the success tolerance a finished run ended at.

The curriculum is env state (reward_utils/curriculum.py); it is not in the
rl_games checkpoint, so whoever continues a run -- the next generation, the next
24 h link -- has to be told where it got to or it silently restarts at 0.075 on
a different reward scale. Reads the per-design table the DesignRewardWrapper
writes (instant), falls back to tensorboard for runs that predate it (minutes on
NFS), and exits non-zero rather than guess.

    python -m coevolution.loop.final_tolerance <run_dir>
"""
from __future__ import annotations

import glob
import json
import math
import sys
import warnings


def final_tolerance(run_dir: str) -> float | None:
    for f in sorted(glob.glob(f"{run_dir}/rank_0/design_rewards_rank0.json")):
        t = json.load(open(f)).get("success_tolerance")
        if t is not None and not math.isnan(t):
            return float(t)
    warnings.filterwarnings("ignore")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = sorted(glob.glob(f"{run_dir}/rank_0/*/summaries/events*"))
    if events:
        acc = EventAccumulator(events[-1], size_guidance={"scalars": 0})
        acc.Reload()
        scalars = acc.Scalars("current_success_tolerance")
        if scalars:
            return float(scalars[-1].value)
    return None


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    t = final_tolerance(sys.argv[1])
    if t is None:
        print(f"no success tolerance recorded under {sys.argv[1]}", file=sys.stderr)
        raise SystemExit(1)
    print(f"{t:.6f}")


if __name__ == "__main__":
    main()
