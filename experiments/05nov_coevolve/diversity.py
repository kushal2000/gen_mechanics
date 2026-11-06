"""How diverse is each campaign run's archive, generation by generation?

  lineages   distinct seed (gen-0) ancestors among the 4,096 designs, and their effective number exp(entropy)
  per niche  median distinct seed ancestors among a niche's 128 designs
  topologies distinct finger layouts ignoring lengths and positions (per finger: joint types and axes), overall and
             the per-niche median
  top share  the share of the archive descended from the 10 most prolific seeds

Ancestry is traced through every search round's children log (parents can be children evicted in a later round,
which are not in designs.json).

    .venv_isaacsim/bin/python experiments/05nov_coevolve/diversity.py [--campaign debug_outputs/coevolve/v1]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HEAD = re.compile(r'^\{"id": "([^"]+)", "parent": "([^"]+)"')


def parents_of(run: Path, reg: dict) -> dict[str, str | None]:
    par = {i: r.get("parent") for i, r in reg.items()}
    for f in sorted(run.glob("gen_*/search/round_*_children.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                m = HEAD.match(line)
                if m:
                    par.setdefault(m.group(1), m.group(2))
    return par


def root(i: str, par: dict, memo: dict) -> str:
    path = []
    while par.get(i):
        if i in memo:
            break
        path.append(i)
        i = par[i]
    r = memo.get(i, i)
    for p in path:
        memo[p] = r
    return r


def topology(design: dict) -> tuple:
    return tuple(sorted(tuple((j["type"], tuple(j.get("axis", ()))) for j in f["joints"]) for f in design["fingers"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="debug_outputs/coevolve/v1")
    args = ap.parse_args()
    camp = REPO / args.campaign
    print(f"{'run':10s} {'gen':>3s} {'lineages':>8s} {'eff.':>6s} {'top10':>6s} {'/niche':>6s} {'topol.':>6s} "
          f"{'/niche':>6s}")
    for run in sorted(p for p in camp.iterdir() if (p / "state.json").exists()):
        reg = json.loads((run / "designs.json").read_text())
        par, memo = parents_of(run, reg), {}
        for gd in sorted(run.glob("gen_[0-9][0-9][0-9][0-9]")):
            if not (gd / "manifest.json").exists():
                continue
            hands = json.loads((gd / "manifest.json").read_text())["hands"]
            if any(h["id"] not in reg for h in hands):
                continue
            roots = [root(h["id"], par, memo) for h in hands]
            c = Counter(roots)
            p = np.array(list(c.values()), float) / len(roots)
            eff = float(np.exp(-(p * np.log(p)).sum()))
            top = sum(n for _, n in c.most_common(10)) / len(roots)
            by_n: dict[str, list] = {}
            for h, r in zip(hands, roots):
                by_n.setdefault(h["niche"], []).append((r, topology(reg[h["id"]]["design"])))
            per_lin = np.median([len({r for r, _ in v}) for v in by_n.values()])
            per_top = np.median([len({t for _, t in v}) for v in by_n.values()])
            topo = len({topology(reg[h["id"]]["design"]) for h in hands})
            print(f"{run.name:10s} {int(gd.name[4:]):3d} {len(c):8d} {eff:6.0f} {top:6.1%} {per_lin:6.0f} {topo:6d} "
                  f"{per_top:6.0f}")


if __name__ == "__main__":
    main()
