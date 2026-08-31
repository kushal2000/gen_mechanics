"""Assemble a generation-1 population from a fitness ranking.

    python -m hand_sampler.assemble_generation \\
        --fitness results/fitness/parents_nominal.csv --mode mutate \\
        --out assets/urdf/generated/population/generations/gen1_nom_mutate

Two compositions, both exactly 24,576 designs so num_envs == population_count:

    repeat   the top-K elites, each occupying 6 slots      (K*6      = 24,576)
    mutate   each elite plus its five mutant children      (K*(1+5)  = 24,576)

SLOT LAYOUT IS 6j+k, NOT BLOCKED BY OPERATOR. Env i holds object i % 1200, so
laying the population out as [all elites][all palm children][all scale children]
would tie each operator to a contiguous index range. Interleaving by family
instead puts an elite and its five children on six DIFFERENT objects, and puts
the elite of family j at the same index -- and therefore the same object -- in
both the repeat and the mutate arm, which makes those two arms a controlled
comparison rather than two unrelated populations.

NAMES ARE NOT MADE UNIQUE. A design that appears six times keeps its original
name, so population_specs writes the same URDF six times (idempotent) instead of
adding 24,576 files per generation to a directory that already holds 172k.
Nothing downstream keys on the name: load_population returns a list and every
consumer indexes by position. Provenance lives in the per-entry fields below.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from hand_sampler.paths import resolve as resolve_repo_path

OPERATORS = ("palm", "scale", "mounting", "num_joints_up", "num_joints_down")
POP = "assets/urdf/generated/population"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fitness", required=True, help="CSV with design_index and mean_goals")
    p.add_argument("--mode", required=True, choices=("repeat", "mutate"))
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=4096)
    p.add_argument("--parents", default=f"{POP}/seed_0003/manifest.json")
    p.add_argument("--mutants", default=f"{POP}/seed_0003/mutants")
    p.add_argument("--total", type=int, default=24576)
    a = p.parse_args()

    fam = 1 + len(OPERATORS)                      # 6 slots per family
    if a.top * fam != a.total:
        raise SystemExit(f"top {a.top} x {fam} = {a.top*fam} != total {a.total}")

    rows = list(csv.DictReader(open(resolve_repo_path(a.fitness))))
    rows.sort(key=lambda r: -float(r["mean_goals"]))
    elites = [(int(r["design_index"]), float(r["mean_goals"])) for r in rows[:a.top]]
    print(f"[gen] top {a.top}: fitness {elites[-1][1]:.2f}..{elites[0][1]:.2f}")

    par = json.loads(resolve_repo_path(a.parents).read_text())["hands"]
    entries: list[dict | None] = [None] * a.total
    for rank, (d, fit) in enumerate(elites):
        entries[rank * fam] = {**{k: par[d][k] for k in ("name", "params")},
                               "source": "elite", "parent_index": d,
                               "rank": rank, "fitness": fit}

    if a.mode == "repeat":
        for rank, (d, fit) in enumerate(elites):
            for k in range(1, fam):
                entries[rank * fam + k] = dict(entries[rank * fam], source="elite_repeat")
    else:
        for k, op in enumerate(OPERATORS, start=1):
            man = json.loads((resolve_repo_path(a.mutants) / op / "manifest.json").read_text())["hands"]
            nfail = 0
            for rank, (d, fit) in enumerate(elites):
                c = man[d]
                assert c["parent_index"] == d, f"{op}: manifest not index-aligned at {d}"
                nfail += bool(c["mutation_failed"])
                entries[rank * fam + k] = {
                    "name": c["name"], "params": c["params"], "source": op,
                    "parent_index": d, "rank": rank, "fitness": fit,
                    "mutation_failed": c["mutation_failed"],
                    "displacement": c["displacement"],
                }
            print(f"[gen] {op:16s} {a.top} children, {nfail} were mutation_failed "
                  f"(parent copies)")
            del man

    assert all(e is not None for e in entries), "unfilled slots"
    out = resolve_repo_path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({
        "version": 1, "kind": "generation", "generation": 1, "mode": a.mode,
        "fitness_source": str(a.fitness), "top": a.top, "count": len(entries),
        "parents": str(a.parents), "operators": list(OPERATORS),
        "slot_layout": "family j occupies slots 6j..6j+5; slot 6j is the elite",
        "seed": 3, "gate": "inherited from the parent population",
        "align_flexion": True, "hands": entries,
    }))
    n_uniq = len({e["name"] for e in entries})
    print(f"[gen] wrote {out/'manifest.json'}: {len(entries)} slots, "
          f"{n_uniq} distinct designs")


if __name__ == "__main__":
    main()
