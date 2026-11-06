"""Study 7: MAP-Elites-style rounds on a learned design value function alone, then ground truth -- does the archive
actually improve, or only in the net's eyes?

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_archive.py --phase search [--rounds 6]
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_archive.py --phase eval     (sbatch --wait)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_archive.py --phase report

The value net: the design value function (design_value_net.py: one per-finger transformer, trained from scratch on
design-only tokens; --model deepsets for study_vf_beam.py's DeepSets ensemble), trained ONCE on the 100-env scores of
iteration --iteration's 4,096 designs (it<N>_env100/, the fixed reset) and frozen.

search  The archive: the 32 niches x 128 of that population, every design ranked by the net's prediction (parents and
        children judged alike). Each round: every archive design gets one random valid application of each of the 9
        mutation types (operators.apply_operator under FEETECH_COARSE_NOCUT_RULES, so mutants keep the relaxed mutation
        rules; children keep their orientation), exact duplicates of anything in the archive dropped; the net scores
        every child; each niche keeps its 128 best of old + new by prediction (a child competes in the niche it lands in;
        one outside the 32 niches is dropped). --rounds rounds. Writes vf_archive/state.json.
eval    Every design that was in any round's archive and has no 100-env score yet: built, scored with 100 envs (the
        same policy and reset as the labels; study_beam_eval.sub, one GPU per 256 designs).
report  Per round: the archive's predicted and true mean, per-niche top-10 and best, how many designs are new.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
import design_value_net as dvn  # noqa: E402
import study_beam as sb  # noqa: E402
import study_vf_beam as svb  # noqa: E402

RUN = REPO / "debug_outputs/evolution/noevolution_4096"
RULES = "FEETECH_COARSE_NOCUT_RULES"
OPS9 = ("length", "split", "merge", "kind", "move_finger", "turn_finger", "tilt_finger", "add_finger", "remove_finger")
PER_NICHE = 128


def out_dir(it: int, tag: str = "") -> Path:
    return REPO / "debug_outputs/noevolution/studies" / (f"it{it}_vf_archive" + (f"_{tag}" if tag else ""))


def env100(it: int) -> dict[str, float]:
    out = {}
    for f in sorted((REPO / "debug_outputs/noevolution/studies" / f"it{it}_env100").glob("c*_fitness.json")):
        out.update({s: v["goals_per_episode"] for s, v in json.loads(f.read_text())["hands"].items()})
    return out


def _mutate_all(job):
    """One random valid application of each of the 9 operators to one design: [(op, child dict, edit)]."""
    design, seed = job
    from children_of import describe
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as gops
    rules = getattr(gh, RULES)
    rng = np.random.default_rng(seed)
    parent = gh.hand_from_dict(design)
    out = []
    for name in OPS9:
        res = gops.apply_operator(parent, name, rng, rules)
        if res is not None:
            child, mv = res
            out.append((name, gh.hand_to_dict(child), describe(name, mv, parent)))
    return out


def _mutate_sharded(d: Path, jobs: list, n: int) -> list:
    """jobs [(parent id, design, seed)] split over n SLURM CPU array tasks (mutate_shard.sub); returns the children of
    every parent, in order (as Pool.map over _mutate_all would)."""
    d.mkdir(parents=True, exist_ok=True)
    parts = [jobs[k::n] for k in range(n)]
    for k, part in enumerate(parts):
        (d / f"shard_{k:03d}_in.json").write_text(json.dumps(part))
        (d / f"shard_{k:03d}_out.json").unlink(missing_ok=True)
    out = lambda k: d / f"shard_{k:03d}_out.json"            # noqa: E731  (written atomically by the shard)
    for attempt in range(6):
        missing = [k for k in range(n) if not out(k).exists()]
        if not missing:
            break
        r = subprocess.run(["sbatch", "--wait", f"--array={','.join(map(str, missing))}", "--export", f"ALL,JOBS={d}",
                            str(REPO / "experiments/05nov_noevolution/studies/mutate_shard.sub")], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        # a busy controller can time out and report a failed submission (or drop --wait) for an array it did queue
        # (5 Nov: the K=50 search died this way): give such shards time to land before resubmitting what is missing
        t0 = time.time()
        while time.time() - t0 < 600 and not all(out(k).exists() for k in missing):
            time.sleep(15)
        if not all(out(k).exists() for k in missing):
            print(f"[mutate] attempt {attempt + 1}: shards {[k for k in missing if not out(k).exists()]} missing "
                  f"({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'no sbatch error'}); resubmitting",
                  flush=True)
    kids = {}
    for k in range(n):
        f = d / f"shard_{k:03d}_out.json"
        if not f.exists():
            raise SystemExit(f"shard {k} wrote nothing: see debug_outputs/train_logs/05nov_noevolution/MUT_shard-*_{k}.log")
        kids.update({pid: [tuple(c) for c in cs] for pid, cs in json.loads(f.read_text())})
    return [kids[pid] for pid, _, _ in jobs]


def search(args) -> None:
    from multiprocessing import Pool
    from coevolution.evolution.descriptors import niche32
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    od = out_dir(args.iteration, args.tag)
    od.mkdir(parents=True, exist_ok=True)
    g = RUN / f"gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    truth = env100(args.iteration)
    if len(truth) < 4000:
        raise SystemExit(f"only {len(truth)} population designs have a 100-env score")
    if args.model == "transformer":                  # the design value function (6 Nov): one per-finger transformer
        hs = [h for h in man["hands"] if h["spec"] in truth]
        net = dvn.fit_transformer([svb.sv._load_design(h["spec"]) for h in hs], [truth[h["spec"]] for h in hs])
        predict = lambda ds: (net(ds), np.zeros(len(ds)))      # noqa: E731  (deterministic: no spread)
        print(f"[vf] fitted the per-finger transformer on {len(hs)} designs (100-env labels)", flush=True)
    else:
        vargs = argparse.Namespace(iteration=args.iteration, labels="env100", members=args.members, trainlike_suffix="")
        predict = svb.fit(vargs)
    D = {}
    for h in man["hands"]:
        d = svb.sv._load_design(h["spec"])
        D[h["id"]] = {"design": d, "key": design_key(gh.hand_from_dict(d)), "niche": h["niche"], "round": 0,
                      "parent": None, "operator": None, "edit": None, "spec": h["spec"], "truth": [truth[h["spec"]]]}
    ids = list(D)
    mu, sd = predict([D[i]["design"] for i in ids])
    for i, m, s in zip(ids, mu, sd):
        D[i]["pred"], D[i]["sd"] = float(m), float(s)
    archive = defaultdict(list)
    for i in ids:
        archive[D[i]["niche"]].append(i)
    for n in archive:
        archive[n] = sorted(archive[n], key=lambda i: -D[i]["pred"])[:PER_NICHE]
    rounds = [{"archive": {n: list(v) for n, v in archive.items()}}]
    print(f"[archive] round 0: {sum(map(len, archive.values()))} designs in {len(archive)} niches; predicted mean "
          f"{np.mean([D[i]['pred'] for v in archive.values() for i in v]):.3f}, true mean "
          f"{np.mean([D[i]['truth'][0] for v in archive.values() for i in v]):.3f}", flush=True)
    for r in range(1, args.n_rounds + 1):
        parents = [i for v in archive.values() for i in v]
        keys = {D[i]["key"] for i in D}
        seeds = [args.seed * 1_000_003 + r * 10_007 + k for k in range(len(parents))]
        t_mut = time.time()
        if args.shards:                               # SLURM CPU array tasks on the default partition
            res = _mutate_sharded(od / "mutate" / f"r{r:02d}", [(p, D[p]["design"], s) for p, s in zip(parents, seeds)],
                                  args.shards)
        else:
            with Pool(args.workers) as pool:
                res = pool.map(_mutate_all, [(D[p]["design"], s) for p, s in zip(parents, seeds)], chunksize=16)
        print(f"[archive] round {r}: mutated {len(parents)} parents in {time.time() - t_mut:.0f} s"
              + (f" ({args.shards} SLURM shards)" if args.shards else f" ({args.workers} local workers)"), flush=True)
        kids, n_raw = [], 0
        for p, cs in zip(parents, res):
            for op, d, edit in cs:
                n_raw += 1
                k = design_key(gh.hand_from_dict(d))
                cell = niche32(d)
                if k in keys or cell is None:
                    continue
                keys.add(k)
                kids.append((p, op, d, edit, k, cell))
        mu, sd = predict([d for _, _, d, _, _, _ in kids])
        new_ids = []
        for j, ((p, op, d, edit, k, cell), m, s) in enumerate(zip(kids, mu, sd)):
            cid = f"a{r:02d}_{j:05d}"
            D[cid] = {"design": d, "key": k, "niche": cell, "round": r, "parent": p, "operator": op, "edit": edit,
                      "pred": float(m), "sd": float(s), "truth": []}
            new_ids.append(cid)
        for cid in new_ids:
            archive[D[cid]["niche"]].append(cid)
        for n in archive:
            archive[n] = sorted(archive[n], key=lambda i: -D[i]["pred"])[:PER_NICHE]
        members = [i for v in archive.values() for i in v]
        new_in = sum(D[i]["round"] == r for i in members)
        by_op = defaultdict(lambda: [0, 0])
        for cid in new_ids:
            by_op[D[cid]["operator"]][0] += 1
        for i in members:
            if D[i]["round"] == r:
                by_op[D[i]["operator"]][1] += 1
        rounds.append({"archive": {n: list(v) for n, v in archive.items()}, "children": len(new_ids), "raw": n_raw,
                       "new_in_archive": new_in, "by_op": {k: v for k, v in by_op.items()}})
        print(f"[archive] round {r}: {n_raw} children, {len(new_ids)} new and in a niche; {new_in} entered the archive; "
              f"predicted mean {np.mean([D[i]['pred'] for i in members]):.3f}; admitted by operator "
              + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_op.items(), key=lambda kv: -kv[1][1])), flush=True)
        (od / "state.json").write_text(json.dumps({"args": vars(args), "designs": D, "rounds": rounds}))
    todo = {i for rd in rounds for v in rd["archive"].values() for i in v if not D[i]["truth"]}
    print(f"[archive] {len(todo)} archive designs over the {args.n_rounds} rounds still need a 100-env score "
          f"(~{len(todo) / 256 * 9 / 60:.1f} GPU-hours)", flush=True)


def evaluate(args) -> None:
    from multiprocessing import Pool
    od = out_dir(args.iteration, args.tag)
    S = json.loads((od / "state.json").read_text())
    D = S["designs"]
    todo = sorted({i for rd in S["rounds"] for v in rd["archive"].values() for i in v if not D[i]["truth"]})
    hands = od / "hands"
    hands.mkdir(exist_ok=True)
    need = [i for i in todo if "spec" not in D[i]]
    with Pool(args.workers) as pool:
        for i, spec, ok in pool.map(sb._build, [(i, D[i]["design"], str(hands)) for i in need], chunksize=8):
            D[i]["spec"] = spec
    rows = [{"id": i, "spec": D[i]["spec"], "scores": []} for i in todo]
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    eargs = argparse.Namespace(chunk_hands=256, iteration=args.iteration, envs=100, table_clearance=0.02)
    (od / "rounds").mkdir(exist_ok=True)
    print(f"[archive] scoring {len(rows)} designs with 100 envs", flush=True)
    sb.evaluate(od, 0, rows, eargs, gman, train_run)
    for r in rows:
        D[r["id"]]["truth"] = r["scores"]
    (od / "state.json").write_text(json.dumps(S))
    report(args)


def eval_round(args) -> None:
    """While the search runs: 100-env scores for round --round's archive designs that lack one, written to
    truth_r<NN>.json (the search keeps rewriting state.json, so the scores live beside it; report merges them)."""
    from multiprocessing import Pool
    od = out_dir(args.iteration, args.tag)
    S = json.loads((od / "state.json").read_text())
    D = S["designs"]
    if args.round >= len(S["rounds"]):
        raise SystemExit(f"round {args.round} is not finished yet ({len(S['rounds']) - 1} done)")
    done = set()                                   # scored already by an earlier eval-round
    for f in od.glob("truth_r*.json"):
        done |= {i for i, v in json.loads(f.read_text()).items() if v["scores"]}
    ids = sorted({i for v in S["rounds"][args.round]["archive"].values() for i in v if not D[i]["truth"]} - done)
    hands = od / "hands"
    hands.mkdir(exist_ok=True)
    with Pool(args.workers) as pool:
        built = dict((i, spec) for i, spec, ok in pool.map(sb._build, [(i, D[i]["design"], str(hands)) for i in ids],
                                                             chunksize=8))
    rows = [{"id": i, "spec": built[i], "scores": []} for i in ids]
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    eargs = argparse.Namespace(chunk_hands=256, iteration=args.iteration, envs=100, table_clearance=0.02)
    (od / "rounds").mkdir(exist_ok=True)
    print(f"[archive] round {args.round}: scoring {len(rows)} archive designs with 100 envs", flush=True)
    sb.evaluate(od, 100 + args.round, rows, eargs, gman, train_run)
    (od / f"truth_r{args.round:02d}.json").write_text(json.dumps(
        {r["id"]: {"spec": r["spec"], "scores": r["scores"]} for r in rows}))
    A = S["rounds"][args.round]["archive"]
    T = {i: D[i]["truth"][0] for i in D if D[i]["truth"]}
    for f in od.glob("truth_r*.json"):
        T.update({i: v["scores"][0] for i, v in json.loads(f.read_text()).items() if v["scores"]})
    mem = [i for v in A.values() for i in v]
    tr = [T[i] for i in mem if i in T]
    pr = [D[i]["pred"] for i in mem]
    r0 = [D[i]["truth"][0] for v in S["rounds"][0]["archive"].values() for i in v]
    print(f"[archive] round {args.round}'s archive: predicted mean {np.mean(pr):.3f}, TRUE mean {np.mean(tr):.3f} "
          f"({len(tr)}/{len(mem)} scored) | round 0: true mean {np.mean(r0):.3f}", flush=True)
    for n in sorted(A):
        t_n = [T[i] for i in A[n] if i in T]
        t_0 = [D[i]["truth"][0] for i in S["rounds"][0]["archive"][n]]
        print(f"    {n:12s} round 0 true mean {np.mean(t_0):.3f} best {max(t_0):.2f} | round {args.round} true mean "
              f"{np.mean(t_n):.3f} best {max(t_n):.2f} (predicted mean {np.mean([D[i]['pred'] for i in A[n]]):.3f})",
              flush=True)


def eval_rounds(args) -> None:
    """100-env scores for every design in the archives of --rounds that has none yet, as one array (all GPUs at once),
    leaving out the archive of --skip-round (scored by its own eval-round job); writes truth_rounds<list>.json."""
    from multiprocessing import Pool
    od = out_dir(args.iteration, args.tag)
    S = json.loads((od / "state.json").read_text())
    D = S["designs"]
    done = {i for i in D if D[i]["truth"]}
    for f in od.glob("truth_r*.json"):
        done |= {i for i, v in json.loads(f.read_text()).items() if v["scores"]}
    skip = set()
    if args.skip_round is not None:
        skip = {i for v in S["rounds"][args.skip_round]["archive"].values() for i in v}
    ids = sorted({i for r in args.rounds for v in S["rounds"][r]["archive"].values() for i in v} - done - skip)
    hands = od / "hands"
    hands.mkdir(exist_ok=True)
    print(f"[archive] rounds {args.rounds}: {len(ids)} designs to score (round {args.skip_round}'s left to its own job)",
          flush=True)
    with Pool(args.workers) as pool:
        built = dict((i, spec) for i, spec, ok in pool.map(sb._build, [(i, D[i]["design"], str(hands)) for i in ids],
                                                             chunksize=8))
    rows = [{"id": i, "spec": built[i], "scores": []} for i in ids]
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    train_run = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/evolution/noevolution_4096" / g.name / "*/")))[0]
    eargs = argparse.Namespace(chunk_hands=256, iteration=args.iteration, envs=100, table_clearance=0.02)
    (od / "rounds").mkdir(exist_ok=True)
    sb.evaluate(od, 200 + min(args.rounds), rows, eargs, gman, train_run)
    tag = "".join(f"{r:02d}" for r in args.rounds)
    (od / f"truth_rounds{tag}.json").write_text(json.dumps(
        {r["id"]: {"spec": r["spec"], "scores": r["scores"]} for r in rows}))
    print(f"[archive] wrote truth_rounds{tag}.json ({sum(bool(r['scores']) for r in rows)} scored)", flush=True)


def report(args) -> None:
    S = json.loads((out_dir(args.iteration, args.tag) / "state.json").read_text())
    D = S["designs"]
    t = lambda i: st.mean(D[i]["truth"]) if D[i]["truth"] else None   # noqa: E731
    print(f"\n[archive] the archive per round (32 niches x 128), the net's prediction against 100-env truth")
    print(f"  {'round':>5s} {'new':>5s} {'pred mean':>9s} {'true mean':>9s} {'scored':>6s} {'true top-10/niche':>17s} "
          f"{'true best/niche':>15s} {'pred best/niche':>15s}")
    base = None
    for r, rd in enumerate(S["rounds"]):
        A = rd["archive"]
        mem = [i for v in A.values() for i in v]
        tr = [t(i) for i in mem if t(i) is not None]
        top10 = [st.mean(sorted([t(i) for i in v if t(i) is not None], reverse=True)[:10]) for v in A.values()
                 if any(t(i) is not None for i in v)]
        best = [max(t(i) for i in v if t(i) is not None) for v in A.values() if any(t(i) is not None for i in v)]
        pbest = [max(D[i]["pred"] for i in v) for v in A.values()]
        row = (r, sum(D[i]["round"] == r for i in mem) if r else 0, np.mean([D[i]["pred"] for i in mem]),
               st.mean(tr) if tr else float("nan"), len(tr), st.mean(top10) if top10 else float("nan"),
               st.mean(best) if best else float("nan"), st.mean(pbest))
        base = base or row
        print(f"  {row[0]:5d} {row[1]:5d} {row[2]:9.3f} {row[3]:9.3f} {row[4]:6d} {row[5]:17.3f} {row[6]:15.3f} "
              f"{row[7]:15.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("search", "eval", "eval-round", "eval-rounds", "report"), required=True)
    ap.add_argument("--rounds", type=int, nargs="+", default=[2, 3, 4, 5], help="eval-rounds: which rounds' archives")
    ap.add_argument("--skip-round", type=int, default=None, help="eval-rounds: leave this round's archive out")
    ap.add_argument("--round", type=int, default=1, help="eval-round: which round's archive")
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--search-rounds", dest="n_rounds", type=int, default=6)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--model", choices=("transformer", "deepsets"), default="transformer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="", help="a separate run: writes it<N>_vf_archive_<tag>/")
    ap.add_argument("--shards", type=int, default=0, help="search: mutate over this many SLURM CPU tasks (0: local)")
    args = ap.parse_args()
    {"search": search, "eval": evaluate, "eval-round": eval_round, "eval-rounds": eval_rounds,
     "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
