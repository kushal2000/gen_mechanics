#!/usr/bin/env python3
"""Grammar-solo comparison, step 2b: combine each group's
``split_population.py`` manifest into one flat job table for the solo
training array (one row = one SLURM array task = one design, its own
population file and its group's shared grasp cache).

Pure stdlib. No GPU, no Kit.

    python3 experiments/grammar_solo/make_job_manifest.py \\
        --run-root /data/pulkitag/users/mpeticco/runs/grammar_solo \\
        --group G_V1 --group G_V2S --group G_V3S --group controls \\
        --out /data/pulkitag/users/mpeticco/runs/grammar_solo/job_manifest.csv

Expects, per ``--group`` NAME: ``<run-root>/designs/NAME/manifest.csv``
(written by ``split_population.py``) and ``<run-root>/viable/NAME/grasp_cache.npz``.
Writes ``<out>`` with columns: array_index, group, design_index, source,
sha256, population_json, grasp_cache.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--group", action="append", required=True, dest="groups")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    run_root = Path(a.run_root)
    rows = []
    for group in a.groups:
        manifest = run_root / "designs" / group / "manifest.csv"
        cache = run_root / "viable" / group / "grasp_cache.npz"
        if not manifest.exists():
            raise FileNotFoundError(f"{manifest} missing (run split_population.py for {group!r} first)")
        if not cache.exists():
            raise FileNotFoundError(f"{cache} missing (grasp cache for {group!r} not built yet)")
        with open(manifest) as f:
            for r in csv.DictReader(f):
                idx = r["index"]
                rows.append({
                    "array_index": len(rows),
                    "group": group,
                    "design_index": idx,
                    "source": r["source"],
                    "sha256": r["sha256"],
                    "population_json": str(run_root / "designs" / group / idx / "population.json"),
                    "grasp_cache": str(cache),
                })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["array_index", "group", "design_index", "source", "sha256",
                                          "population_json", "grasp_cache"])
        w.writeheader()
        w.writerows(rows)
    print(f"[make_job_manifest] {len(rows)} job(s) -> {out}")
    for g in a.groups:
        n = sum(1 for r in rows if r["group"] == g)
        print(f"  {g}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
