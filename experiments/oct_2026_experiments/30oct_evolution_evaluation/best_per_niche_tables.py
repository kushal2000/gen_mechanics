"""The best design each evolution run found in each of the 30 niches, as markdown tables.

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/best_per_niche_tables.py

Per run (baseline, map_elites) and niche: of every hand the run ever made in that niche, the one with the highest
goals per episode in any training evaluation (10 envs, first episode each; the policy of that iteration, so scores
from different iterations are under different policies), the iteration of that score, its joints per finger and palm
cut, and how many hands the run made in the niche. Writes tables/best_per_niche_<run>.md.
"""
from __future__ import annotations

import collections
from pathlib import Path

from coevolution.evolution.archive import Archive
from coevolution.evolution.descriptors import summary

RUNS = Path("debug_outputs/evolution")
OUT = Path(__file__).resolve().parent / "tables"
LENGTH = {"j0": "short", "j1": "medium", "j2": "long"}


def table(run: str) -> str:
    arch = Archive.load(RUNS / run / "archive.json")
    by = collections.defaultdict(list)
    for r in arch.records.values():
        by[r.niche].append(r)
    last = max(e["generation"] for r in arch.records.values() for e in r.fitness if not e["screen"])
    rows = [f"**{run}** (iterations 0-{last}): best hand ever found per niche", "",
            "| fingers | palm | finger length | best hand | best goals/episode | at iteration | joints per finger | "
            "hands tried |", "|---|---|---|---|---|---|---|---|"]
    for f in range(2, 7):
        for cut in ("nocut", "cut"):
            for j in ("j0", "j1", "j2"):
                n = f"f{f}_{cut}_{j}"
                best = None
                for r in by.get(n, []):
                    for e in r.fitness:
                        if not e["screen"] and (best is None or e["goals_per_episode"] > best[1]):
                            best = (r, e["goals_per_episode"], e["generation"])
                pal = "cut" if cut == "cut" else "no cut"
                if best is None:
                    rows.append(f"| {f} | {pal} | {LENGTH[j]} | - | - | - | - | {len(by.get(n, []))} |")
                    continue
                r, g, it = best
                jpf = "/".join(str(len(x["joints"])) for x in r.design["fingers"])
                rows.append(f"| {f} | {pal} | {LENGTH[j]} | {r.id} | {g:g} | {it} | {jpf} | {len(by[n])} |")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for run in ("baseline", "map_elites"):
        t = table(run)
        (OUT / f"best_per_niche_{run}.md").write_text(t + "\n")
        print(t, "\n")


if __name__ == "__main__":
    main()
