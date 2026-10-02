"""Keep every design-scoring window of a training run.

``design_scoring`` rewrites ``per_design_scores_rank<r>.json`` in place every
write window, so only the latest window survives. This CPU-only poller
appends each new version (with a wall-clock ``_t``) to a JSON-lines file
until the training process exits:

    python -m isaacsimenvs.inhand_reorient.tools.poll_design_scores \
        RUN_DIR/per_design_scores_rank0.json RUN_DIR/windows.jsonl TRAIN_PID
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def poll_once(src: Path, out: Path, state: dict) -> bool:
    """Append ``src`` to ``out`` if its mtime changed; return whether it did."""
    try:
        mtime = os.path.getmtime(src)
        if mtime == state.get("mtime"):
            return False
        doc = json.loads(Path(src).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    state["mtime"] = mtime
    doc["_t"] = time.time()
    with open(out, "a") as f:
        f.write(json.dumps(doc) + "\n")
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None) -> int:
    src, out, pid = (argv or sys.argv[1:])[:3]
    state: dict = {}
    while _alive(int(pid)):
        poll_once(Path(src), Path(out), state)
        time.sleep(3)
    poll_once(Path(src), Path(out), state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
