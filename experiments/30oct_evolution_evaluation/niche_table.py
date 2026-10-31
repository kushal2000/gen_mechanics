"""Every niche's best score in each evolution run, at the latest (or a given) iteration.

    .venv_isaacsim/bin/python experiments/30oct_evolution_evaluation/niche_table.py [--runs baseline map_elites rl_only]

Per niche: map_elites' elite (scored this iteration); for the other runs the archive's best hand per niche with the
iteration its score is from (a hand no longer trained keeps an old score), and how many hands of the current training
population sit in the niche. Goals per episode (10 envs, first episode each; coevolution/eval/eval_hands.py).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from coevolution.evolution.archive import Archive
from coevolution.evolution.generation import _trained, best_per_niche

RUNS = Path("debug_outputs/evolution")
LENGTH = {0: "short", 1: "medium", 2: "long"}           # mean joints per finger < 2.5, 2.5-3.5, > 3.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["baseline", "map_elites", "rl_only"])
    args = ap.parse_args()
    cols = []
    for run in args.runs:
        st = json.loads((RUNS / run / "state.json").read_text())
        arch = Archive.load(RUNS / run / "archive.json")
        g = st["history"][-1]["generation"]
        best = best_per_niche(arch.records.values())
        pop = json.loads((RUNS / run / f"gen_{g + 1:04d}" / "manifest.json").read_text())["hands"]
        counts = collections.Counter(arch[h["id"]].niche for h in pop)
        cols.append((run, g, best, counts))
    head = " | ".join(f"{r} (iteration {g}): best, scored at / training now" for r, g, _, _ in cols)
    print(f"| fingers | palm | finger length | {head} |")
    for f in range(2, 7):
        for cut in ("nocut", "cut"):
            for j in range(3):
                n = f"f{f}_{cut}_j{j}"
                cells = []
                for _, _, best, counts in cols:
                    e = _trained(best[n]) if n in best else None
                    cells.append(f"{e['goals_per_episode']} @ {e['generation']} / {counts.get(n, 0)}" if e else "-")
                print(f"| {f} | {'cut' if cut == 'cut' else 'no cut'} | {LENGTH[j]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
