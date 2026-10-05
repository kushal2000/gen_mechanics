"""Learning rate, KL, entropy and policy log-std over a training run, as one
small JSON.

rl_games writes the adaptive learning rate, KL and entropy to its
tensorboard events (``<exp>/summaries``) and the policy parameters to a
periodic checkpoint (``<exp>/nn/last_<name>_ep_<E>_rew_<R>.pth``, every
``save_frequency`` epochs). This CPU tool reads both and writes

    {"scalars": {tag: [[frame, value], ...]},          # every `every`-th point
     "checkpoints": [{"file", "epoch", "frame", "cols",
                      "rows": [{"mean", "min", "max"}, ...],
                      "values": [[log-std per action column], ...]}]}

``rows`` has one entry per log-std row: one for ``fixed_sigma: fixed``, one
per SAPG exploration block for ``coef_cond``. ``cols`` are the action
columns summarised: given (``--columns``), or by default the columns whose
log-std left its initial 0 in some row (the joint-token transformer's ghost
columns get no gradient and stay at exactly 0).

    python -m isaacsimenvs.inhand_reorient.tools.train_trace RUN_DIR OUT_JSON \
        [--columns 0,1,2] [--every 10] [--watch PID] [--period 600]

With ``--watch PID`` it rewrites OUT_JSON every ``--period`` seconds while
PID lives, then once more. Checkpoints already in OUT_JSON are not reloaded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

TAGS = ("info/last_lr", "info/kl", "losses/entropy", "info/epochs", "rewards/step", "episode_lengths/step",
        "performance/step_inference_rl_update_fps")
_CKPT = re.compile(r"^last_.*_ep_(\d+)_rew_.*\.pth$")


def read_scalars(run_dir: Path, every: int = 1) -> dict:
    from tensorboard.backend.event_processing import event_accumulator as ea

    out: dict = {}
    for sdir in sorted(Path(run_dir).glob("*/summaries")):
        acc = ea.EventAccumulator(str(sdir), size_guidance={ea.SCALARS: 0})
        acc.Reload()
        have = set(acc.Tags().get("scalars", []))
        for tag in TAGS:
            if tag in have:
                ev = acc.Scalars(tag)
                out.setdefault(tag, []).extend([[int(e.step), float(e.value)] for e in ev[:: max(int(every), 1)]])
    return out


def checkpoint_logstd(path: Path, columns=None) -> dict:
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    while isinstance(ck, dict) and 0 in ck:  # the vendored rl_games saves {policy index: state}
        ck = ck[0]
    model = ck.get("model", ck)
    sigma = model["a2c_network.sigma"].float()
    rows = sigma.reshape(-1, sigma.shape[-1])
    if columns is None:
        cols = [int(c) for c in torch.nonzero((rows != 0).any(dim=0)).flatten()]
    else:
        cols = [int(c) for c in columns]
    stats = []
    for r in rows:
        v = r[cols] if cols else r
        stats.append({"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())})
    return {"file": Path(path).name, "epoch": int(ck.get("epoch", -1)), "frame": int(ck.get("frame", -1)),
            "cols": cols, "rows": stats, "values": [[round(float(x), 4) for x in r] for r in rows]}


def _checkpoints(run_dir: Path) -> list[Path]:
    found = [p for p in Path(run_dir).glob("*/nn/*.pth") if _CKPT.match(p.name)]
    return sorted(found, key=lambda p: int(_CKPT.match(p.name).group(1)))


def trace(run_dir, columns=None, every: int = 1, previous: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    done = {c["file"]: c for c in (previous or {}).get("checkpoints", []) if c.get("rows")}
    ckpts = []
    for p in _checkpoints(run_dir):
        c = done.get(p.name)
        if c is None:
            try:
                c = checkpoint_logstd(p, columns)
            except Exception as exc:  # a checkpoint being written right now: next pass
                print(f"[train_trace] skip {p.name}: {exc!r}")
                continue
        if c:
            ckpts.append(c)
    ckpts.sort(key=lambda c: c.get("epoch", -1))
    return {"scalars": read_scalars(run_dir, every), "checkpoints": ckpts}


def _write(out: Path, doc: dict) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, out)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--columns", default=None)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--watch", type=int, default=None)
    ap.add_argument("--period", type=float, default=600.0)
    a = ap.parse_args(argv)
    cols = None if not a.columns else [int(c) for c in a.columns.split(",")]
    out = Path(a.out)

    def once():
        prev = json.loads(out.read_text()) if out.exists() else None
        _write(out, trace(a.run_dir, cols, a.every, prev))

    if a.watch is not None:
        while _alive(a.watch):
            try:
                once()
            except Exception as exc:  # keep watching; the final pass reports
                print(f"[train_trace] pass failed: {exc!r}")
            time.sleep(a.period)
    once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
