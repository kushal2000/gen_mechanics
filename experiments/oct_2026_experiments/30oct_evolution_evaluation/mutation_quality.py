"""How good are the mutations? Every child against its parent under the same policy, per operator.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/mutation_quality.py

map_elites: a child after one iteration of training vs its parent in the same evaluation pass. baseline: the child's
zero-shot screen vs its trained parent (same policy), and admitted children 1 and 2 iterations after birth vs their
parent in the same pass. Pairs where both score 0 goals are left out of better/worse; 10-episode scores, so read the
overall rates, not small per-operator differences.
"""
from __future__ import annotations

import collections

import numpy as np

from coevolution.evolution.archive import Archive


def score(r, g, screen):
    e = [x for x in r.fitness if x["generation"] == g and bool(x["screen"]) == screen]
    return e[-1]["goals_per_episode"] if e else None


def line(name, pairs):
    nz = [(c, q) for _, c, q in pairs if c > 0 or q > 0]
    rel = [c / q for c, q in nz if q > 0]
    print(f"{name:16s} n={len(pairs):3d} | scoring pairs {len(nz):3d}: better {sum(c > q for c, q in nz):3d}, "
          f"worse {sum(c < q for c, q in nz):3d} | median child/parent {np.median(rel) if rel else float('nan'):.2f}")


def report(title, pairs):
    print(f"\n== {title}")
    line("all", pairs)
    by = collections.defaultdict(list)
    for p in pairs:
        by[p[0]].append(p)
    for op, ps in sorted(by.items(), key=lambda kv: -len(kv[1])):
        line("  " + op, ps)


def main() -> None:
    me = Archive.load("debug_outputs/evolution/map_elites/archive.json").records
    report("map_elites: after one iteration of training, same pass as the parent",
           [(r.operator, score(r, r.born + 1, False), score(me[r.parent], r.born + 1, False)) for r in me.values()
            if r.parent and score(r, r.born + 1, False) is not None and score(me[r.parent], r.born + 1, False) is not None])
    b = Archive.load("debug_outputs/evolution/baseline/archive.json").records
    report("baseline: zero-shot screen vs the trained parent, same policy",
           [(r.operator, score(r, r.born, True), score(b[r.parent], r.born, False)) for r in b.values()
            if r.parent and score(r, r.born, True) is not None and score(b[r.parent], r.born, False) is not None])
    for lag in (1, 2):
        report(f"baseline: admitted children {lag} iteration(s) after birth, same pass as the parent",
               [(r.operator, score(r, r.born + lag, False), score(b[r.parent], r.born + lag, False)) for r in b.values()
                if r.parent and score(r, r.born + lag, False) is not None
                and score(b[r.parent], r.born + lag, False) is not None])


if __name__ == "__main__":
    main()
