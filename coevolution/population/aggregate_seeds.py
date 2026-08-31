"""Average population evals across seeds into one fitness column.

    python -m coevolution.population.aggregate_seeds \\
        --runs 'results/population_eval/*_rep_s*' --out results/fitness/nominal.csv

WHY SEEDS RATHER THAN A LONGER RUN. A single 6,000-step eval has a measured
test-retest reliability of 0.51: two runs differing only in --env_seed agree on
just 41% of their top-4,096. Averaging k independent seeds follows
Spearman-Brown exactly (verified at k = 1..7 against 15 replicates), so k = 10
buys reliability 0.91 and 1.79x the selection response of a single run. Seeds
also parallelise, whereas one long run does not.

WHAT AVERAGING CANNOT FIX. Every seed uses the same design->object map
(env i holds object i % pool_size), so the object's contribution is a fixed
per-design offset that no amount of averaging removes. Object eta^2 is 2.5% of
observed variance against 51% true variance, so roughly 5% of the converged
score is the object rather than the design. Rotating the object map across seeds
would marginalise it; that needs the explicit-assignment array.

Reported per design:
    mean_goals   the fitness to select on
    sem          standard error across seeds
    n_seeds      how many runs contributed
and, across the population, a split-half reliability estimate computed by
splitting the seeds in two -- which is the honest number, unlike the
within-run split-half that inflates it (an episode straddling the midpoint
lands in both halves).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

CARRY = ("design", "design_index", "n_active_fingers", "n_active_joints",
         "pool_index", "object_type", "object_shape")


def load_run(d: Path) -> tuple[list[dict], dict]:
    rows = list(csv.DictReader(open(d / "per_design.csv")))
    rows.sort(key=lambda r: int(r["design_index"]))
    meta = json.loads((d / "population_eval.json").read_text())
    return rows, meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", required=True,
                   help="Glob over eval directories, or a comma-separated list")
    p.add_argument("--out", required=True)
    p.add_argument("--column", default="total_goals")
    p.add_argument("--max_seeds", type=int, default=0, help="0 = use all matched")
    a = p.parse_args()

    pats = a.runs.split(",") if "," in a.runs else [a.runs]
    dirs = sorted({Path(d) for pat in pats for d in glob.glob(pat)
                   if (Path(d) / "population_eval.json").is_file()})
    if a.max_seeds:
        dirs = dirs[:a.max_seeds]
    if not dirs:
        raise SystemExit(f"no completed evals matched {a.runs!r}")

    import numpy as np

    vals, seeds, conds = [], [], set()
    base = None
    for d in dirs:
        rows, meta = load_run(d)
        if base is None:
            base = rows
        elif [r["design_index"] for r in rows] != [r["design_index"] for r in base]:
            raise SystemExit(f"{d}: design_index order differs from {dirs[0]}")
        vals.append(np.array([float(r[a.column]) for r in rows]))
        seeds.append(meta["protocol"].get("env_seed"))
        conds.add(meta["protocol"].get("condition"))
    if len(conds) > 1:
        raise SystemExit(f"runs mix conditions {conds}; aggregate one condition at a time")
    if len(set(s for s in seeds if s is not None)) != len([s for s in seeds if s is not None]):
        raise SystemExit(f"duplicate env_seeds {seeds}: those runs are not independent")

    V = np.vstack(vals)                       # (k, n_designs)
    k, n = V.shape
    mean = V.mean(axis=0)
    sem = V.std(axis=0, ddof=1) / np.sqrt(k) if k > 1 else np.zeros(n)

    # Honest reliability: split the SEEDS, not one run's steps.
    rel = None
    if k >= 2:
        h = k // 2
        r = float(np.corrcoef(V[:h].mean(axis=0), V[h:2 * h].mean(axis=0))[0, 1])
        rel = 2 * r / (1 + r)                 # Spearman-Brown up to all k

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*CARRY, "mean_goals", "sem", "n_seeds"])
        w.writeheader()
        for i, row in enumerate(base):
            w.writerow({**{c: row[c] for c in CARRY},
                        "mean_goals": float(mean[i]), "sem": float(sem[i]),
                        "n_seeds": k})
    print(f"[agg] {k} seeds {sorted(s for s in seeds if s is not None)} "
          f"condition={conds.pop()}")
    print(f"[agg] {n} designs  mean {mean.mean():.2f}  sd across designs {mean.std():.2f}")
    if rel is not None:
        print(f"[agg] reliability of the {k}-seed average: {rel:.3f} "
              f"(split-half over seeds, Spearman-Brown)")
    print(f"[agg] wrote {out}")


if __name__ == "__main__":
    main()
