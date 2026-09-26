"""Multiprocessing experiment runner for the E0 evolvability experiments.

``run_experiment`` maps a per-seed function over a list of seeds with
``multiprocessing.Pool``, and writes ``result.json`` (every raw per-seed
result plus aggregate stats and provenance) and ``summary.md`` (a
human-readable aggregate table) into ``out_dir``.

CLI: ``python3 -m hand_sampler.grammar.experiments.runner --list`` prints
every experiment name registered via ``register`` in this process (which,
for this module, means every experiment registered at import time below --
new experiment modules should ``from .runner import register`` and call it
at their own module level so ``--list`` sees them too).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..rules import GRAMMAR_VERSION

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register(name: str, fn: Callable[..., Any]) -> None:
    """Register ``fn`` (signature ``fn(seed, **params) -> JSON-serializable
    result``) under ``name`` so the CLI ``--list`` (and any future
    experiment driver) can find it."""
    _REGISTRY[name] = fn


def registered_experiments() -> Dict[str, Callable[..., Any]]:
    return dict(_REGISTRY)


def _git_info(repo_dir: Path) -> Dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), capture_output=True, text=True, check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo_dir), capture_output=True, text=True, check=True,
        ).stdout
        dirty = porcelain.strip() != ""
    except Exception:
        sha, dirty = None, None
    return {"sha": sha, "dirty": dirty}


def _worker(task) -> Dict[str, Any]:
    fn, seed, params = task
    try:
        result = fn(seed, **params)
        return {"seed": seed, "ok": True, "result": result, "error": None}
    except Exception as e:  # noqa: BLE001 -- a per-seed failure must not kill the batch
        return {"seed": seed, "ok": False, "result": None, "error": f"{type(e).__name__}: {e}"}


def _numeric(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _aggregate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r["result"] for r in per_seed if r["ok"]]
    agg: Dict[str, Any] = {"n_seeds": len(per_seed), "n_ok": len(ok), "n_failed": len(per_seed) - len(ok)}
    keys = sorted({k for r in ok if isinstance(r, dict) for k, v in r.items() if _numeric(v)})
    for k in keys:
        vals = [float(r[k]) for r in ok if isinstance(r, dict) and _numeric(r.get(k))]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        agg[k] = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "n": len(vals),
        }
    return agg


def _summary_md(name: str, params: Dict[str, Any], aggregate: Dict[str, Any], wall_time_s: float) -> str:
    lines = [
        f"# Experiment: {name}", "",
        f"Wall time: {wall_time_s:.3f} s", "",
        "## Params", "", "```json", json.dumps(params, indent=2, sort_keys=True, default=str), "```", "",
        "## Aggregate", "",
        f"seeds: {aggregate.get('n_seeds')}  ok: {aggregate.get('n_ok')}  failed: {aggregate.get('n_failed')}",
        "",
        "| metric | mean | std | min | max | n |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(aggregate):
        v = aggregate[k]
        if isinstance(v, dict) and "mean" in v:
            lines.append(f"| {k} | {v['mean']:.6g} | {v['std']:.6g} | {v['min']:.6g} | {v['max']:.6g} | {v['n']} |")
    lines.append("")
    return "\n".join(lines)


def run_experiment(name: str, fn: Callable[..., Any], params: Dict[str, Any], seeds: Sequence[int],
                    out_dir: Optional[str] = None, processes: int = 24) -> Dict[str, Any]:
    """Run ``fn(seed, **params)`` over every ``seeds`` entry with a
    ``multiprocessing.Pool`` (size ``min(processes, len(seeds))``, never
    started at all for an empty/singleton ``seeds``), then write
    ``result.json``/``summary.md`` into ``out_dir`` (default
    ``project-notes/grammar/experiments/<name>``, relative to the current
    working directory -- callers doing a real overnight run should invoke
    this from the repo root; tests always pass an explicit ``out_dir``).
    Returns the same dict written to ``result.json``."""
    if out_dir is None:
        out_dir = f"project-notes/grammar/experiments/{name}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    repo_dir = Path(__file__).resolve().parents[3]  # .../gen_mechanics
    git_info = _git_info(repo_dir)

    seeds = list(seeds)
    tasks = [(fn, seed, params) for seed in seeds]
    t0 = time.time()
    if len(tasks) > 1:
        n_proc = max(1, min(processes, len(tasks)))
        with mp.Pool(processes=n_proc) as pool:
            per_seed = pool.map(_worker, tasks)
    else:
        per_seed = [_worker(t) for t in tasks]
    wall_time_s = time.time() - t0

    aggregate = _aggregate(per_seed)
    result: Dict[str, Any] = {
        "name": name,
        "params": params,
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "git_sha": git_info["sha"],
        "git_dirty": git_info["dirty"],
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "grammar_version": GRAMMAR_VERSION,
        "wall_time_s": wall_time_s,
    }
    (out_path / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    (out_path / "summary.md").write_text(_summary_md(name, params, aggregate, wall_time_s))
    return result


def _smoke_experiment(seed: int, n_configs: int = 8) -> Dict[str, float]:
    """Trivial registered experiment: derives one hand at ``seed`` under
    ``G_FULL`` and returns its geometric proxies -- exists purely to smoke
    test the ``run_experiment`` plumbing (registry, multiprocessing,
    result.json/summary.md), not as a real research payload."""
    from ..derive import generate
    from ..proxy import all_proxies

    _, model = generate(seed)
    return all_proxies(model, seed, n_configs=n_configs)


register("smoke", _smoke_experiment)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m hand_sampler.grammar.experiments.runner")
    parser.add_argument("--list", action="store_true", help="print registered experiment names and exit")
    args = parser.parse_args(argv)
    if args.list:
        for name in sorted(_REGISTRY):
            print(name)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
