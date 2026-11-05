"""Study 2: what does each kind of one-step mutation do to a good hand's score, zero-shot under the current policy?

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_mutations.py --phase make [--iteration 1]
        [--parents-per-niche 4] [--chunk-hands 3000] [--workers 12]
    sbatch --array=0-<n-1> experiments/05nov_noevolution/studies/study_mutations_eval.sub      (one GPU per chunk)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_mutations.py --phase report

make: per niche, the --parents-per-niche best hands of the iteration's evaluation (gen_NNNN/fitness.json); every valid
one-step child of each (every application of every operator under FEETECH_COARSE_NOCUT_RULES, operators.iter_valid,
one per distinct design), described by operator, level (chain / placement / topology), edit and niche; built (URDF,
spec) under debug_outputs/noevolution/studies/it<N>_mutations/hands/, listed in chunk manifests of --chunk-hands.
eval (--chunk k): that chunk zero-shot under the iteration's policy (eval_hands.py, 4 envs each, sampled actions).
report: each child's score against its parent's (the 100-env score from study 1 where there is one, else the loop's
4-env score), by operator and level: mean change, share better / worse than the parent, niche moves.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
RUN = REPO / "debug_outputs/evolution/noevolution_4096"
RULES = "FEETECH_COARSE_NOCUT_RULES"
FIT = "goals_per_episode"


def out_dir(it: int) -> Path:
    return REPO / "debug_outputs/noevolution/studies" / f"it{it}_mutations"


def _children(job):
    """Every valid one-step child of one parent: (child design, operator, level, edit)."""
    pid, design = job
    from children_of import LEVEL, describe
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    rules = getattr(gh, RULES)
    parent = gh.hand_from_dict(design)
    seen, out = {design_key(parent, rules)}, []
    for op in gops.OPERATORS:
        if op.name not in LEVEL:
            continue
        for mv, child in gops.iter_valid(parent, op, rules):
            k = design_key(child, rules)
            if k in seen:
                continue
            seen.add(k)
            out.append((gh.hand_to_dict(child), op.name, LEVEL[op.name], describe(op.name, mv, parent)))
    return pid, out


def _build(job):
    cid, design, hands_dir = job
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar.urdf_export import build
    row = build(gh.hand_from_dict(design), cid, Path(hands_dir) / cid, rules_name=RULES)
    return cid, row["spec"]


def make(args) -> None:
    from coevolution.evolution.descriptors import niche32
    g = RUN / f"gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    fit = json.loads((g / "fitness.json").read_text())["hands"]
    by = defaultdict(list)
    for h in man["hands"]:
        by[h["niche"]].append(h)
    parents = []
    for n, hs in by.items():
        parents += sorted(hs, key=lambda h: (-fit[h["spec"]][FIT], -fit[h["spec"]].get("reward_per_step", 0.0)))[
            :args.parents_per_niche]
    designs = {}
    for h in parents:
        spec = json.loads((REPO / h["spec"]).read_text())
        designs[h["id"]] = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
    out = out_dir(args.iteration)
    (out / "hands").mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        kids = pool.map(_children, list(designs.items()), chunksize=1)
    rows, jobs = [], []
    for pid, cs in kids:
        for k, (d, op, level, edit) in enumerate(cs):
            cid = f"{pid}__c{k:03d}"
            rows.append({"id": cid, "parent": pid, "parent_niche": next(h["niche"] for h in parents if h["id"] == pid),
                         "operator": op, "level": level, "edit": edit, "niche": niche32(d) or "none", "design": d})
            jobs.append((cid, d, str(out / "hands")))
    print(f"[mutations] {len(parents)} parents, {len(rows)} children "
          f"({ {lv: sum(r['level'] == lv for r in rows) for lv in ('chain', 'placement', 'topology')} })", flush=True)
    with Pool(args.workers) as pool:
        specs = dict(pool.map(_build, jobs, chunksize=16))
    for r in rows:
        r["spec"] = specs[r["id"]]
    chunks = [rows[k:k + args.chunk_hands] for k in range(0, len(rows), args.chunk_hands)]
    for c, part in enumerate(chunks):
        (out / f"chunk{c:02d}_manifest.json").write_text(json.dumps({
            "generation": man["generation"], "grid": man["grid"],
            "hands": [{"id": r["id"], "spec": r["spec"]} for r in part]}))
    (out / "children.json").write_text(json.dumps({"iteration": args.iteration, "parents": [h["id"] for h in parents],
                                                    "chunks": len(chunks), "children": rows}))
    print(f"[mutations] built; {len(chunks)} chunks of <= {args.chunk_hands}: "
          f"sbatch --array=0-{len(chunks) - 1} experiments/05nov_noevolution/studies/study_mutations_eval.sub",
          flush=True)


def evaluate(args) -> None:
    g = RUN / f"gen_{args.iteration:04d}"
    out = out_dir(args.iteration)
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    mp, fo = out / f"chunk{args.chunk:02d}_manifest.json", out / f"chunk{args.chunk:02d}_fitness.json"
    with open(out / f"chunk{args.chunk:02d}.log", "w") as log:
        subprocess.run([sys.executable, str(REPO / "coevolution/eval/eval_hands.py"), "--manifest", str(mp),
                        "--checkpoint", str(g / "policy.pth"), "--train-run", train_run, "--out", str(fo),
                        "--envs-per-hand", str(args.envs), "--seed", "2"], stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
    print(f"[mutations] chunk {args.chunk}: {'done' if fo.exists() else 'FAILED, see its log'}", flush=True)


def report(args) -> None:
    out = out_dir(args.iteration)
    meta = json.loads((out / "children.json").read_text())
    scores = {}
    for c in range(meta["chunks"]):
        f = out / f"chunk{c:02d}_fitness.json"
        if f.exists():
            scores.update(json.loads(f.read_text())["hands"])
    g = RUN / f"gen_{args.iteration:04d}"
    loop = json.loads((g / "fitness.json").read_text())["hands"]
    id_spec = {h["id"]: h["spec"] for h in json.loads((g / "manifest.json").read_text())["hands"]}
    noise = out.parent / f"it{args.iteration}_noise" / "scores.json"
    ref = {r["id"]: r["env100"] for r in json.loads(noise.read_text())} if noise.exists() else {}
    rows = [r for r in meta["children"] if r["spec"] in scores]
    for r in rows:
        r["score"] = scores[r["spec"]][FIT]
        r["parent_score"] = ref.get(r["parent"], loop[id_spec[r["parent"]]][FIT])
        r["delta"] = r["score"] - r["parent_score"]
    print(f"[mutations] iteration {args.iteration}: {len(rows)}/{len(meta['children'])} children scored "
          f"(parents: {'100-env study-1 scores' if ref else 'loop 4-env scores'} where available)")
    print(f"  {'group':24s} {'n':>5s}  child mean  parent mean  delta   better  worse  left niche")
    groups = [("all", rows)] + [(lv, [r for r in rows if r["level"] == lv]) for lv in ("chain", "placement", "topology")]
    groups += [(f"  {op}", [r for r in rows if r["operator"] == op]) for op in sorted({r["operator"] for r in rows})]
    for name, rs in groups:
        if not rs:
            continue
        d = [r["delta"] for r in rs]
        print(f"  {name:24s} {len(rs):5d}  {st.mean(r['score'] for r in rs):10.3f}  "
              f"{st.mean(r['parent_score'] for r in rs):11.3f}  {st.mean(d):+.3f}  {sum(x > 0 for x in d) / len(d):6.0%}"
              f"  {sum(x < 0 for x in d) / len(d):5.0%}  {sum(r['niche'] != r['parent_niche'] for r in rs) / len(rs):9.0%}")
    (out / "report.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "design"} for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("make", "eval", "report"), required=True)
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--parents-per-niche", type=int, default=4)
    ap.add_argument("--chunk-hands", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--envs", type=int, default=4)
    args = ap.parse_args()
    {"make": make, "eval": evaluate, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
