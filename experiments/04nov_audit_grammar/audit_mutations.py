"""Audit the 29 Oct evolution runs' hands: are the designs valid, and is every child one recorded mutation of its parent?

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/audit_mutations.py [--runs map_elites baseline] [--workers 12]

Kit-free, CPU only, reads debug_outputs/evolution/<run>/archive.json and writes nothing but its report. Per hand:
  valid      the grammar's rules (the run's rule set) and C1 (`viability.clear`) hold
  canonical  the stored design is its own canonical form (turns and mirror images collapsed)
  key        the stored key is the design's canonical JSON (keys are what de-duplication compares)
  niche      descriptors.niche(design) -- recomputed, against the elites map for map_elites
  lineage    (children only) the child is among the valid applications of its recorded operator to its parent,
             enumerated exhaustively (`operators.iter_valid`, no shuffling). add_finger draws the new finger's joints
             at random, so it is checked backwards: the parent is among the valid remove_finger applications to the
             child.
Writes experiments/04nov_audit_grammar/results/<run>_audit.json (per-hand failures) and prints a summary.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "results"


def _audit(args):
    rec, parent_design, rules_name = args
    from coevolution.evolution import descriptors
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    from hand_sampler.grammar import viability as gvb

    rules = getattr(gh, rules_name)
    hand = gh.hand_from_dict(rec["design"])
    out = {"id": rec["id"], "operator": rec["operator"], "fail": []}
    probs = gh.check(hand, rules)
    if probs:
        out["fail"].append(("rules", probs))
    if not gvb.clear(hand):
        out["fail"].append(("c1", gvb.c1_self_overlap(hand).worst_mm))
    if gops.canonical(hand, rules) != hand:
        out["fail"].append(("canonical", None))
    if design_key(hand, rules) != rec["key"]:
        out["fail"].append(("key", None))
    out["niche"] = descriptors.niche(rec["design"])
    if parent_design is None:          # seeds (and random children) have no parent to check against
        return out
    parent = gh.hand_from_dict(parent_design)
    t0 = time.time()
    if rec["operator"] == "add_finger":
        target = design_key(parent, rules)
        hits = [design_key(c, rules) for _, c in gops.iter_valid(hand, gops.OPERATOR_BY_NAME["remove_finger"], rules)]
        ok = target in hits
        n = len(hits)
    else:
        target = rec["key"]
        hits = [design_key(c, rules) for _, c in gops.iter_valid(parent, gops.OPERATOR_BY_NAME[rec["operator"]], rules)]
        ok = target in hits
        n = len(hits)
    out["lineage_candidates"], out["lineage_s"] = n, round(time.time() - t0, 2)
    if not ok:
        # which operator does explain it, if any (one step)?
        expl = []
        for op in gops.OPERATORS:
            if op.name in (rec["operator"], "add_finger"):
                continue
            if any(design_key(c, rules) == rec["key"] for _, c in gops.iter_valid(parent, op, rules)):
                expl.append(op.name)
        out["fail"].append(("lineage", {"candidates": n, "explained_by": expl}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["map_elites", "baseline"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N hands per run (a quick look)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        rd = REPO / "debug_outputs/evolution" / run
        cfg = json.loads((rd / "config.json").read_text())
        recs = json.loads((rd / "archive.json").read_text())["hands"]
        by_id = {r["id"]: r for r in recs}
        if args.limit:
            recs = recs[:args.limit]
        jobs = [(r, by_id[r["parent"]]["design"] if r.get("parent") else None, cfg["rules"]) for r in recs]
        t0 = time.time()
        with Pool(args.workers) as pool:
            res = pool.map(_audit, jobs, chunksize=4)
        fails = collections.Counter(f[0] for r in res for f in r["fail"])
        by_op = collections.defaultdict(lambda: [0, 0])
        for r in res:
            if r["operator"] not in ("seed", "random"):
                by_op[r["operator"]][0] += 1
                by_op[r["operator"]][1] += any(f[0] == "lineage" for f in r["fail"])
        stored_niche = {r["id"]: None for r in recs}
        state = json.loads((rd / "state.json").read_text())
        map_bad = [(n, i) for n, i in state.get("map", {}).items()
                   if next((x["niche"] for x in res if x["id"] == i), n) != n]
        report = {"run": run, "rules": cfg["rules"], "hands": len(res), "seconds": round(time.time() - t0),
                  "failures": dict(fails), "lineage_by_operator": {k: {"children": v[0], "unexplained": v[1]}
                                                                   for k, v in sorted(by_op.items())},
                  "map_niche_mismatch": map_bad,
                  "failed_hands": [r for r in res if r["fail"]]}
        (OUT / f"{run}_audit.json").write_text(json.dumps(report, indent=1, default=str))
        print(f"[audit] {run}: {len(res)} hands in {report['seconds']} s; failures {dict(fails) or 'none'}; "
              f"map niche mismatches {len(map_bad)}", flush=True)
        for k, v in report["lineage_by_operator"].items():
            print(f"    {k:14s} {v['children']:4d} children, {v['unexplained']} not explained by one application", flush=True)


if __name__ == "__main__":
    main()
