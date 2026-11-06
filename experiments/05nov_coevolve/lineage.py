"""Trace a design's lineage back to its seed through every search round's children log, with each ancestor's niche,
the mutation that made it, the value net's prediction and any 100-env scores it got; plus how often the archive's
best designs reached their niche from another one (MAP-Elites stepping stones).

    .venv_isaacsim/bin/python experiments/05nov_coevolve/lineage.py --run debug_outputs/coevolve/v1/k5_margin [--id ID]
        (default: the best 100-env design of the latest scored generation)
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HEAD = re.compile(r'^\{"id": "([^"]+)", "parent": "([^"]+)"')


def scores(run: Path, camp: Path) -> dict[str, list[tuple[str, float]]]:
    """id -> [(label, goals/episode)] over every 100-env eval of this run (and the shared seeds)."""
    out: dict[str, list] = {}
    dirs = [(camp / "hod/gen_0000", "eval100", "seeds, gen-0 policy")]
    for gd in sorted(run.glob("gen_[0-9][0-9][0-9][0-9]")):
        for ed in sorted(gd.glob("eval100*")):
            pg = ed.name.split("_policy_g")[-1] if "_policy_g" in ed.name else gd.name[4:]
            dirs.append((gd, ed.name, f"gen {int(gd.name[4:])} archive, gen-{int(pg)} policy"))
    for gd, sub, label in dirs:
        for f in sorted((gd / sub).glob("c*_fitness.json")):
            man = json.loads(Path(str(f).replace("_fitness", "_manifest")).read_text())
            id_of = {h["spec"]: h["id"] for h in man["hands"]}
            for spec, v in json.loads(f.read_text())["hands"].items():
                out.setdefault(id_of[spec], []).append((label, v["goals_per_episode"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="debug_outputs/coevolve/v1/k5_margin")
    ap.add_argument("--id")
    args = ap.parse_args()
    run = REPO / args.run
    camp = run.parent
    reg = json.loads((run / "designs.json").read_text())
    S = scores(run, camp)
    par = {i: r.get("parent") for i, r in reg.items()}
    logs = sorted(run.glob("gen_*/search/round_*_children.jsonl.gz"))
    where = {}
    for f in logs:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                m = HEAD.match(line)
                if m:
                    par.setdefault(m.group(1), m.group(2))
                    where.setdefault(m.group(1), f)

    # the best design: highest 100-env score in the latest generation with a complete own-policy eval
    if args.id:
        best = args.id
    else:
        g = max(int(gd.name[4:]) for gd in run.glob("gen_*") if len(list((gd / "eval100").glob("c*_fitness.json"))) >= 16)
        ids = [h["id"] for h in json.loads((run / f"gen_{g:04d}/manifest.json").read_text())["hands"]]
        own = f"gen {g} archive, gen-{g} policy"
        best = max(ids, key=lambda i: dict(S.get(i, [])).get(own, -1))
    chain = [best]
    while par.get(chain[-1]):
        chain.append(par[chain[-1]])
    need = set(chain)
    rec = {}
    for f in {where[i] for i in chain if i in where}:
        with gzip.open(f, "rt") as fh:
            for line in fh:
                m = HEAD.match(line)
                if m and m.group(1) in need:
                    rec[m.group(1)] = json.loads(line)
    print(f"[lineage] {run.name}: {best}, {len(chain) - 1} mutations from seed {chain[-1]}\n")
    for i in reversed(chain):
        r = rec.get(i) or reg.get(i) or {}
        d = r.get("design") or reg.get(i, {}).get("design")
        nf = len(d["fingers"]) if d else "?"
        nm = sum(len(f["joints"]) for f in d["fingers"]) if d else "?"
        sc = "; ".join(f"{v:.2f} ({lab})" for lab, v in S.get(i, []))
        print(f"  {i:16s} {r.get('niche') or reg.get(i, {}).get('niche', ''):10s} {nf}f {nm}m  "
              f"{(r.get('operator') or 'seed'):13s} {(r.get('edit') or ''):44.44s} "
              f"pred {r['pred']:.2f}" if "pred" in r else
              f"  {i:16s} {reg.get(i, {}).get('niche', ''):10s} {nf}f {nm}m  {'seed':13s} {'':44s} {'':9s}", end="")
        print(f"   {sc}" if sc else "")

    # stepping stones: for each niche's 10 best designs, did the lineage pass through another niche / finger count?
    g = max(int(gd.name[4:]) for gd in run.glob("gen_*") if len(list((gd / "eval100").glob("c*_fitness.json"))) >= 16)
    man = json.loads((run / f"gen_{g:04d}/manifest.json").read_text())["hands"]
    own = f"gen {g} archive, gen-{g} policy"
    by: dict[str, list] = {}
    for h in man:
        by.setdefault(h["niche"], []).append(h["id"])
    niche_of = {i: r.get("niche") for i, r in reg.items()}

    def niche(i):
        return niche_of.get(i) or (rec.get(i) or {}).get("niche") or i.rsplit("_", 1)[0]
    cross_n, cross_f, total = 0, 0, 0
    for n, ids in by.items():
        for i in sorted(ids, key=lambda i: -dict(S.get(i, [])).get(own, -1))[:10]:
            anc, j = [], i
            while par.get(j):
                j = par[j]
                anc.append(j)
            seed_f = anc[-1].split("_")[0] if anc else n.split("_")[0]
            total += 1
            cross_f += seed_f != n.split("_")[0]
            cross_n += anc[-1].rsplit("_", 1)[0] != n if anc else 0
    print(f"\n[lineage] gen {g}: of each niche's 10 best designs ({total}), {cross_n / total:.0%} descend from a seed in "
          f"another niche and {cross_f / total:.0%} from a seed with a different finger count")


if __name__ == "__main__":
    main()
