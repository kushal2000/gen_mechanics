"""Study 5: how far can one design be improved by repeated one-step mutation? A brute-force beam search, zero-shot under
one fixed policy.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_beam.py [--start f4_m12-13_040]
        [--iteration 3] [--beam 10] [--envs 100] [--rounds 15] [--patience 3] [--min-gain 0.02] [--same-fingers]

Round 0 scores the start design. Each round then:
  1. every valid one-step child of every design in the beam (operators.iter_valid under FEETECH_COARSE_NOCUT_RULES),
     minus every design seen before (canonical key) and those that do not fit the 6 x 5 slot grid;
  2. every child scored with --envs envs (eval_values.py: the first episode of each env, sampled actions, the SAPG
     leader), the beam re-scored alongside it (each member's score = the mean over all its evaluations, so a lucky
     draw does not stay);
  3. the beam becomes the --beam best of (beam + children): elitist.
It stops once the best has not improved by --min-gain for --patience rounds, or after --rounds. The policy is fixed
(gen_NNNN/policy.pth of the no-evolution run) and the env is its training env, except the reset uses the 6 Nov
table-safe fix (pad clearance 0.02). Each round's evaluations run as a SLURM array (study_beam_eval.sub, one GPU per
256 designs), so this driver only builds designs and waits.

Writes debug_outputs/noevolution/studies/beam_<start>_it<N>/: state.json (every design ever scored: its design, parent,
operator, edit, niche, round, scores; each round's beam and summary), the built hands, the round manifests and logs.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
RUN = REPO / "debug_outputs/evolution/noevolution_4096"
RULES = "FEETECH_COARSE_NOCUT_RULES"
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}
FIT = "goals_per_episode"


def _children(job):
    did, design = job
    from children_of import LEVEL, describe
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    rules = getattr(gh, RULES)
    parent = gh.hand_from_dict(design)
    out = []
    for op in gops.OPERATORS:
        if op.name not in LEVEL:
            continue
        for mv, child in gops.iter_valid(parent, op, rules):
            out.append((design_key(child, rules), gh.hand_to_dict(child), op.name, describe(op.name, mv, parent)))
    return did, out


def _build(job):
    did, design, hands_dir = job
    from coevolution.evolution.variation import fits_grid
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar.urdf_export import build
    row = build(gh.hand_from_dict(design), did, Path(hands_dir) / did, rules_name=RULES)
    return did, row["spec"], fits_grid(row["spec"], GRID)


def _key(design: dict) -> str:
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    return design_key(gh.hand_from_dict(design), getattr(gh, RULES))


def score(r: dict) -> float:
    return st.mean(r["scores"]) if r["scores"] else float("-inf")


def evaluate(out: Path, rnd: int, rows: list[dict], args, gman: dict, train_run: str) -> None:
    """Score rows (each with "id", "spec") with --envs envs, one SLURM array task per chunk; appends to r["scores"]."""
    jobs = []
    for k in range(0, len(rows), args.chunk_hands):
        part = rows[k:k + args.chunk_hands]
        mp = out / "rounds" / f"r{rnd:02d}_c{k // args.chunk_hands:02d}_manifest.json"
        mp.write_text(json.dumps({"generation": gman["generation"], "grid": gman["grid"],
                                  "hands": [{"id": r["id"], "spec": r["spec"]} for r in part]}))
        fo = mp.with_name(mp.name.replace("_manifest.json", "_fitness.json"))
        jobs.append((mp, fo))
    jf = out / "rounds" / f"r{rnd:02d}_jobs.txt"
    ck = RUN / f"gen_{args.iteration:04d}" / "policy.pth"
    jf.write_text("".join(f"{m} {args.envs} {f} {1000 + 10 * rnd + i} {ck} {train_run} {args.table_clearance}\n"
                          for i, (m, f) in enumerate(jobs)))
    todo = [i for i, (_, f) in enumerate(jobs) if not f.exists()]
    if todo:
        cmd = ["sbatch", "--wait", f"--array={','.join(map(str, todo))}", "--export", f"ALL,JOBS={jf}",
               str(REPO / "experiments/05nov_noevolution/studies/study_beam_eval.sub")]
        print(f"[beam] round {rnd}: {len(rows)} designs in {len(jobs)} chunk(s) x {args.envs} envs ...", flush=True)
        t0 = time.time()
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL)
        print(f"[beam] round {rnd}: evaluated in {time.time() - t0:.0f} s", flush=True)
    res = {}
    for m, f in jobs:
        if f.exists():
            res.update(json.loads(f.read_text())["hands"])
        else:
            print(f"[beam] WARNING: {f.name} missing (see its log); those designs stay unscored", flush=True)
    for r in rows:
        if r["spec"] in res:
            r["scores"].append(res[r["spec"]][FIT])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="f4_m12-13_040")
    ap.add_argument("--iteration", type=int, default=3, help="the no-evolution iteration whose policy scores everything")
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--same-fingers", action="store_true", help="only children with the start's finger count")
    ap.add_argument("--table-clearance", type=float, default=0.02)
    ap.add_argument("--chunk-hands", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    from coevolution.evolution.descriptors import niche32
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    out = REPO / "debug_outputs/noevolution/studies" / f"beam_{args.start}_it{args.iteration}"
    (out / "rounds").mkdir(parents=True, exist_ok=True)
    (out / "hands").mkdir(exist_ok=True)
    sf = out / "state.json"
    if sf.exists():
        S = json.loads(sf.read_text())
        print(f"[beam] resuming at round {len(S['rounds'])}", flush=True)
    else:
        h = next(h for h in gman["hands"] if h["id"] == args.start)
        spec = json.loads((REPO / h["spec"]).read_text())
        d = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
        S = {"args": vars(args), "designs": {args.start: {
            "id": args.start, "design": d, "key": _key(d), "parent": None, "operator": None, "edit": None,
            "niche": niche32(d) or "none", "round": 0, "spec": h["spec"], "scores": []}}, "rounds": []}
        evaluate(out, 0, [S["designs"][args.start]], args, gman, train_run)
        S["rounds"].append({"beam": [args.start], "best": args.start, "best_score": score(S["designs"][args.start]),
                            "children": 0})
        sf.write_text(json.dumps(S))
        print(f"[beam] round 0: {args.start} scores {score(S['designs'][args.start]):.3f}", flush=True)
    nf0 = len(S["designs"][args.start]["design"]["fingers"])
    stale = 0
    for rnd in range(len(S["rounds"]), args.rounds + 1):
        D = S["designs"]
        beam = S["rounds"][-1]["beam"]
        seen = {r["key"] for r in D.values()}
        with Pool(args.workers) as pool:
            kids = pool.map(_children, [(b, D[b]["design"]) for b in beam], chunksize=1)
        new, n_raw = {}, 0
        for pid, cs in kids:
            for key, d, op, edit in cs:
                n_raw += 1
                if key in seen or key in new or (args.same_fingers and len(d["fingers"]) != nf0):
                    continue
                new[key] = (pid, d, op, edit)
        rows = []
        for k, (key, (pid, d, op, edit)) in enumerate(new.items()):
            did = f"r{rnd:02d}_{k:04d}"
            rows.append({"id": did, "design": d, "key": key, "parent": pid, "operator": op, "edit": edit,
                         "niche": niche32(d) or "none", "round": rnd, "scores": []})
        with Pool(args.workers) as pool:
            built = pool.map(_build, [(r["id"], r["design"], str(out / "hands")) for r in rows], chunksize=8)
        ok = {did: spec for did, spec, fits in built if fits}
        rows = [dict(r, spec=ok[r["id"]]) for r in rows if r["id"] in ok]
        print(f"[beam] round {rnd}: {n_raw} children of {len(beam)} -> {len(rows)} new designs that fit "
              f"({len(new) - len(rows)} do not fit the grid)", flush=True)
        if not rows:
            break
        evaluate(out, rnd, rows + [D[b] for b in beam], args, gman, train_run)
        for r in rows:
            D[r["id"]] = r
        pool_ids = beam + [r["id"] for r in rows]
        nb = sorted(pool_ids, key=lambda i: -score(D[i]))[:args.beam]
        prev = S["rounds"][-1]["best_score"]
        best = nb[0]
        kid_scores = [score(r) for r in rows if r["scores"]]
        summary = {"beam": nb, "best": best, "best_score": score(D[best]), "children": len(rows),
                   "children_mean": st.mean(kid_scores) if kid_scores else None,
                   "children_max": max(kid_scores) if kid_scores else None,
                   "beam_mean": st.mean(score(D[i]) for i in nb),
                   "new_in_beam": sum(D[i]["round"] == rnd for i in nb),
                   "beat_prev_best": sum(s > prev for s in kid_scores)}
        S["rounds"].append(summary)
        sf.write_text(json.dumps(S))
        b = D[best]
        print(f"[beam] round {rnd}: best {best} {summary['best_score']:.3f} ({b['niche']}, by {b['operator']} from "
              f"{b['parent']}); beam mean {summary['beam_mean']:.3f}; children mean {summary['children_mean']:.3f} "
              f"max {summary['children_max']:.3f}; {summary['new_in_beam']} new in the beam, "
              f"{summary['beat_prev_best']} children beat the previous best", flush=True)
        stale = stale + 1 if summary["best_score"] < prev + args.min_gain else 0
        if stale >= args.patience:
            print(f"[beam] no gain of {args.min_gain} for {args.patience} rounds: stopping", flush=True)
            break
    print("[beam] trajectory of the best: " + " -> ".join(f"{r['best_score']:.3f}" for r in S["rounds"]), flush=True)


if __name__ == "__main__":
    main()
