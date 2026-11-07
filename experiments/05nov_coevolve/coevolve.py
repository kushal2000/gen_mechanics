"""Co-evolution vs House of Dextra (v1 campaign): one shared policy trained on an archive of 32 niches x 128 designs;
between training iterations the archive changes by K rounds of MAP-Elites search on a design value net fitted to the
iteration's own training logs. House of Dextra (no evolution) is the same driver with K = 0.

    python coevolve.py init  --run <dir> --seeds <seed manifest> [--k 0]           (gen 0 = the seed population)
    python coevolve.py fork  --run <dir> --from <run> --k 5 --rule none|cap|margin (shares <run>'s trained gen 0)
    python coevolve.py next  --run <dir>      (after training gen g: labels -> net -> K rounds -> gen g+1, or a copy)
    python coevolve.py trained --run <dir> --train-run <dir> --checkpoint <pth>    (coevolve.sub, after training)
    python coevolve.py eval  --run <dir> --gen g      (submit the 100-env eval of gen g under its policy; async)
    python coevolve.py fork-all --campaign <dir>       (create and submit every run in <campaign>/forks.json)
    python coevolve.py report --campaign <dir>
One SLURM job per iteration (coevolve.sub): train gen g -> trained -> eval (async) -> next -> resubmit.

Run folder (debug_outputs/coevolve/<campaign>/<run>/):
  config.json  state.json  designs.json (every design that ever entered the archive: design, niche, born, parent,
  operator, edit, spec)  hands/ (their URDFs)
  gen_XXXX/manifest.json       the archive trained in iteration XXXX (each hand's id, spec, niche)
           policy.pth          the policy after that training
           labels.json         id -> [training goals/episode over the last 25% of the iteration, episodes]
           value_net.pt        the design value net fitted on those labels (design_value_net, tuned defaults)
           search/round_RR_children.jsonl.gz   every child of round RR: id, parent, operator, edit, niche, design,
                                               prediction, admitted (in the archive after the round)
           search/round_RR_archive.json         the archive after round RR (niche -> ids) and its predicted mean
           eval100/            the 100-env eval of manifest.json under policy.pth (chunks; report reads them)
Selection rules (each niche keeps 128 by the net's prediction):
  none    the 128 best of old + new
  margin  a newcomer counts its prediction minus 0.05 against the round's incumbents
  cap     at most 64 of a niche's 128 may be new against the archive the iteration started with
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import shutil
import statistics as st
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
PER_NICHE = 128
LABEL_FRACTION = 0.25
MARGIN = 0.05
CAP = 64
SUB = REPO / "experiments/05nov_coevolve/coevolve.sub"
EVAL_SUB = REPO / "experiments/05nov_noevolution/studies/study_beam_eval.sub"


def gen_dir(run: Path, g: int) -> Path:
    return run / f"gen_{g:04d}"


def load(run: Path, name: str):
    return json.loads((run / name).read_text())


def save(run: Path, name: str, obj) -> None:
    tmp = run / f".{name}.tmp"
    tmp.write_text(json.dumps(obj))
    tmp.rename(run / name)


def design_of(spec_rel: str) -> dict:
    s = json.loads((REPO / spec_rel).read_text())
    return s["design"] if isinstance(s["design"], dict) else json.loads(s["design"])


# ---------------------------------------------------------------- creating runs
def init(args) -> None:
    run = Path(args.run)
    if (run / "state.json").exists():
        raise SystemExit(f"{run} already initialised")
    seeds = json.loads((REPO / args.seeds).read_text())
    gen_dir(run, 0).mkdir(parents=True)
    (gen_dir(run, 0) / "manifest.json").write_text(json.dumps({"generation": 0, "grid": seeds["grid"],
                                                               "hands": seeds["hands"]}))
    reg = {h["id"]: {"design": design_of(h["spec"]), "niche": h["niche"], "born": 0, "parent": None, "operator": None,
                     "edit": None, "spec": h["spec"]} for h in seeds["hands"]}
    save(run, "designs.json", reg)
    save(run, "config.json", {"k": args.k, "rule": args.rule, "seeds": args.seeds, "seed": args.seed,
                              "per_niche": PER_NICHE, "label_fraction": LABEL_FRACTION, "margin": MARGIN, "cap": CAP,
                              "shards": args.shards, "forked_from": None})
    save(run, "state.json", {"generation": 0, "trained": False, "checkpoint": None, "train_run": None, "history": []})
    print(f"[coevolve] {run.name}: gen 0 = {len(reg)} seed designs (K {args.k}, rule {args.rule})", flush=True)


def fork(args) -> None:
    run, src = Path(args.run), Path(getattr(args, "src"))
    if (run / "state.json").exists():
        raise SystemExit(f"{run} already initialised")
    S0 = load(src, "state.json")
    g = args.gen
    if not (gen_dir(src, g) / "policy.pth").exists():
        raise SystemExit(f"{src.name} gen {g} is not trained yet")
    gen_dir(run, g).mkdir(parents=True)
    for f in ("manifest.json", "labels.json"):
        if (gen_dir(src, g) / f).exists():
            shutil.copy(gen_dir(src, g) / f, gen_dir(run, g) / f)
    (gen_dir(run, g) / "policy.pth").symlink_to((gen_dir(src, g) / "policy.pth").resolve())
    (gen_dir(run, g) / "FORKED_FROM").write_text(f"{src.resolve()} gen {g}\n")
    shutil.copy(src / "designs.json", run / "designs.json")
    cfg0 = load(src, "config.json")
    save(run, "config.json", {**cfg0, "k": args.k, "rule": args.rule, "forked_from": f"{src.name}/gen_{g:04d}"})
    tr = S0["train_run"] if S0["generation"] == g else S0.get("train_runs", {}).get(str(g))
    save(run, "state.json", {"generation": g, "trained": True, "checkpoint": str(gen_dir(src, g) / "policy.pth"),
                             "train_run": tr, "train_runs": {str(g): tr}, "history": []})
    print(f"[coevolve] {run.name}: forked from {src.name} gen {g} (K {args.k}, rule {args.rule})", flush=True)


def fork_all(args) -> None:
    """Create every run in <campaign>/forks.json from its source's gen and submit its chain (once)."""
    camp = Path(args.campaign)
    spec = json.loads((camp / "forks.json").read_text())
    for f in spec["runs"]:
        run = camp / f["name"]
        if (run / "state.json").exists():
            continue
        ns = argparse.Namespace(run=str(run), src=str(camp / spec["from"]), gen=spec["gen"], k=f["k"], rule=f["rule"])
        fork(ns)
        sub = subprocess.run(["sbatch", f"--job-name=CO_{f['name']}", "--export", f"ALL,RUN={run}", str(SUB)],
                             capture_output=True, text=True)
        print(f"[coevolve] {f['name']}: {sub.stdout.strip()} {sub.stderr.strip()}", flush=True)


# ---------------------------------------------------------------- one iteration
def labels_from_logs(train_run: Path, manifest: dict) -> dict[str, list]:
    """design id -> [goals/episode, episodes] over the last LABEL_FRACTION of the iteration's per-episode dumps."""
    d = train_run / "design_stats"
    dumps = sorted(d.glob("step_*.npz"))
    if not dumps:
        raise SystemExit(f"no per-episode logs in {d} (env.log.design_stats off?)")
    use = dumps[-max(1, math.ceil(LABEL_FRACTION * len(dumps))):]
    hoe = np.load(d / "meta.npz")["hand_of_env"].astype(int)
    hands = json.loads((d / "hands.json").read_text())
    env = np.concatenate([np.load(f)["env"] for f in use]).astype(int)
    goals = np.concatenate([np.load(f)["goals"] for f in use])
    ep = np.bincount(hoe[env], minlength=len(hands))
    gl = np.bincount(hoe[env], goals, minlength=len(hands))
    id_of = {h["spec"]: h["id"] for h in manifest["hands"]}
    out = {id_of[s]: [float(gl[k] / ep[k]), int(ep[k])] for k, s in enumerate(hands) if ep[k] > 0 and s in id_of}
    print(f"[coevolve] labels: {len(use)}/{len(dumps)} dumps, {ep.mean():.0f} episodes per design, {len(out)} designs, "
          f"mean {np.mean([v[0] for v in out.values()]):.3f}", flush=True)
    return out


def select(ids: list[str], pred: dict, rule: str, round_inc: set, iter_inc: set, size: int = PER_NICHE) -> list[str]:
    if rule == "global":                              # no niches: the population's top `size` by prediction
        return sorted(ids, key=lambda i: -pred[i])[:size]
    if rule == "none":
        return sorted(ids, key=lambda i: -pred[i])[:PER_NICHE]
    if rule == "margin":
        return sorted(ids, key=lambda i: -(pred[i] - (0.0 if i in round_inc else MARGIN)))[:PER_NICHE]
    if rule == "cap":
        out, new = [], 0
        for i in sorted(ids, key=lambda i: -pred[i]):
            fresh = i not in iter_inc
            if fresh and new >= CAP:
                continue
            out.append(i)
            new += fresh
            if len(out) == PER_NICHE:
                break
        return out
    raise ValueError(rule)


def trained(args) -> None:
    run = Path(args.run)
    S = load(run, "state.json")
    S.update({"trained": True, "checkpoint": args.checkpoint, "train_run": args.train_run})
    S.setdefault("train_runs", {})[str(S["generation"])] = args.train_run
    save(run, "state.json", S)


def next_gen(args) -> None:
    run = Path(args.run)
    cfg, S, reg = load(run, "config.json"), load(run, "state.json"), load(run, "designs.json")
    g = S["generation"]
    if not S["trained"]:
        raise SystemExit(f"gen {g} is not trained yet")
    t0 = time.time()
    gd = gen_dir(run, g)
    man = json.loads((gd / "manifest.json").read_text())
    lab = labels_from_logs(Path(S["train_run"]), man)
    (gd / "labels.json").write_text(json.dumps(lab))
    nd = gen_dir(run, g + 1)
    nd.mkdir(parents=True, exist_ok=True)
    hist = {"generation": g, "labels_mean": float(np.mean([v[0] for v in lab.values()])), "rounds": []}
    if cfg["k"] == 0:                                 # House of Dextra: the same designs again
        (nd / "manifest.json").write_text(json.dumps({**man, "generation": g + 1}))
    else:
        hist.update(search(run, cfg, reg, man, lab, g, nd))
    S["history"].append({**hist, "propose_s": time.time() - t0})
    S.update({"generation": g + 1, "trained": False, "checkpoint": str(gd / "policy.pth"), "train_run": None})
    save(run, "state.json", S)
    print(f"[coevolve] {run.name}: gen {g + 1} ready ({time.time() - t0:.0f} s)", flush=True)


def search(run: Path, cfg: dict, reg: dict, man: dict, lab: dict, g: int, nd: Path) -> dict:
    import torch
    import design_value_net as dvn
    import study_beam as sb
    import study_vf_archive as sva
    from coevolution.evolution.descriptors import niche32
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    gd = gen_dir(run, g)
    sd = gd / "search"
    sd.mkdir(exist_ok=True)
    ids = [h["id"] for h in man["hands"] if h["id"] in lab]
    t = time.time()
    net = dvn.fit_transformer([reg[i]["design"] for i in ids], [lab[i][0] for i in ids], seed=cfg["seed"] + g)
    torch.save(net.net.state_dict(), gd / "value_net.pt")
    fit_s = time.time() - t
    # rule "global" (the no-MAP-Elites ablation): one pool, the top len(population) by prediction; children must
    # still fall inside the grid (same design space), but niches get no quota and may empty out
    flat = cfg["rule"] == "global"
    bucket = (lambda n: "all") if flat else (lambda n: n)          # noqa: E731
    niche_of = {h["id"]: h["niche"] for h in man["hands"]}
    archive = defaultdict(list)
    for h in man["hands"]:
        archive[bucket(h["niche"])].append(h["id"])
    members = [i for v in archive.values() for i in v]
    pred = dict(zip(members, net([reg[i]["design"] for i in members]).tolist()))
    keys = {design_key(gh.hand_from_dict(r["design"])) for r in reg.values()}
    iter_inc = set(members)
    rounds, kids_all = [], {}
    for r in range(1, cfg["k"] + 1):
        parents = [i for v in archive.values() for i in v]
        seeds = [cfg["seed"] * 1_000_003 + g * 100_003 + r * 10_007 + k for k in range(len(parents))]
        t1 = time.time()
        res = sva._mutate_sharded(sd / f"shards_r{r:02d}",       # parents from earlier rounds are in kids_all only
                                  [(p, (reg.get(p) or kids_all[p])["design"], s) for p, s in zip(parents, seeds)],
                                  cfg["shards"])
        mut_s = time.time() - t1
        shutil.rmtree(sd / f"shards_r{r:02d}", ignore_errors=True)   # every child is logged in round_RR_children
        kids, n_raw, n_dup, n_out = [], 0, 0, 0
        for p, cs in zip(parents, res):
            for op, d, edit in cs:
                n_raw += 1
                k = design_key(gh.hand_from_dict(d))
                cell = niche32(d)
                if k in keys:
                    n_dup += 1
                    continue
                if cell is None:
                    n_out += 1
                    continue
                keys.add(k)
                kids.append((p, op, edit, d, cell))
        P = net([d for *_, d, _ in kids]).tolist() if kids else []
        round_inc = set(parents)
        cids = []
        for j, ((p, op, edit, d, cell), v) in enumerate(zip(kids, P)):
            cid = f"g{g + 1:04d}r{r:02d}_{j:05d}"
            kids_all[cid] = {"design": d, "niche": cell, "born": g + 1, "parent": p, "operator": op, "edit": edit,
                             "spec": None}
            pred[cid] = v
            niche_of[cid] = cell
            archive[bucket(cell)].append(cid)
            cids.append(cid)
        for n in archive:
            archive[n] = select(archive[n], pred, cfg["rule"], round_inc, iter_inc, size=len(man["hands"]))
        inside = {i for v in archive.values() for i in v}
        with gzip.open(sd / f"round_{r:02d}_children.jsonl.gz", "wt") as fh:
            for cid in cids:
                c = kids_all[cid]
                fh.write(json.dumps({"id": cid, "parent": c["parent"], "operator": c["operator"], "edit": c["edit"],
                                     "niche": c["niche"], "pred": pred[cid], "admitted": cid in inside,
                                     "design": c["design"]}) + "\n")
        pm = float(np.mean([pred[i] for i in inside]))
        by_niche = defaultdict(list)                  # logged by real niche for every rule
        for v in archive.values():
            for i in v:
                by_niche[niche_of[i]].append(i)
        (sd / f"round_{r:02d}_archive.json").write_text(json.dumps(
            {"pred_mean": pm, "archive": {n: v for n, v in sorted(by_niche.items())},
             "pred": {i: pred[i] for i in inside}}))
        rounds.append({"round": r, "children_raw": n_raw, "duplicates": n_dup, "outside_niches": n_out,
                       "children": len(cids), "admitted": sum(c in inside for c in cids), "pred_mean": pm,
                       "new_vs_iteration": sum(i not in iter_inc for i in inside), "mutate_s": mut_s})
        print(f"[coevolve] gen {g} round {r}/{cfg['k']}: {len(cids)} children ({rounds[-1]['admitted']} admitted), "
              f"predicted mean {pm:.3f}, {rounds[-1]['new_vs_iteration']} new this iteration ({mut_s:.0f} s mutating)",
              flush=True)
    final = {i for v in archive.values() for i in v}
    new = [i for i in final if i not in reg]
    for i in new:
        reg[i] = kids_all[i]
    t2 = time.time()
    with Pool(4) as pool:
        for i, spec, ok in pool.map(sb._build, [(i, reg[i]["design"], str(run / "hands")) for i in new], chunksize=8):
            reg[i]["spec"] = spec
    (nd / "manifest.json").write_text(json.dumps({"generation": g + 1, "grid": man["grid"], "hands": [
        {"id": i, "spec": reg[i]["spec"], "niche": niche_of[i]}
        for i in sorted(final, key=lambda i: (niche_of[i], i))]}))
    save(run, "designs.json", reg)
    return {"rounds": rounds, "fit_s": fit_s, "build_s": time.time() - t2, "new_designs": len(new)}


# ---------------------------------------------------------------- evaluation
def evaluate(args) -> None:
    run = Path(args.run)
    stop = [f for f in (run / "NO_EVAL", run.parent / "NO_EVAL") if f.exists()]
    if stop and not args.force:                       # 6 Nov: per-generation evals off; training logs are the record
        print(f"[coevolve] {run.name} gen {args.gen}: eval skipped ({stop[0]}; --force to run it)", flush=True)
        return
    S = load(run, "state.json")
    g = args.gen
    gd = gen_dir(run, g)
    pg = g if args.policy_gen is None else args.policy_gen    # --policy-gen g-1: the archive as the search left it,
    pd = gen_dir(run, pg)                                     # scored by the policy whose labels drove the search
    tr = S.get("train_runs", {}).get(str(pg)) or S["train_run"]
    man = json.loads((gd / "manifest.json").read_text())
    ed = gd / ("eval100" if pg == g else f"eval100_policy_g{pg:04d}")
    ed.mkdir(exist_ok=True)
    lines = []
    for k in range(0, len(man["hands"]), 256):
        mp = ed / f"c{k // 256:02d}_manifest.json"
        mp.write_text(json.dumps({"generation": man["generation"], "grid": man["grid"],
                                  "hands": [{"id": h["id"], "spec": h["spec"]} for h in man["hands"][k:k + 256]]}))
        lines.append(f"{mp} {args.envs} {ed / f'c{k // 256:02d}_fitness.json'} {9000 + g * 100 + k // 256} "
                     f"{(pd / 'policy.pth').resolve()} {tr} 0.02\n")
    (ed / "jobs.txt").write_text("".join(lines))
    out = subprocess.run(["sbatch", f"--array=0-{len(lines) - 1}", f"--job-name=EV_{run.name}_g{g}" + (f"_p{pg}" if pg != g else ""), "--export",
                          f"ALL,JOBS={ed / 'jobs.txt'}", str(EVAL_SUB)], capture_output=True, text=True)
    (ed / "submitted.txt").write_text(out.stdout + out.stderr)
    print(f"[coevolve] {run.name} gen {g} (policy gen {pg}): {args.envs}-env eval, {len(lines)} chunks ({out.stdout.strip()})", flush=True)


def report(args) -> None:
    camp = Path(args.campaign)
    for run in sorted(p for p in camp.iterdir() if (p / "state.json").exists()):
        cfg = load(run, "config.json")
        print(f"\n[coevolve] {run.name} (K {cfg['k']}, rule {cfg['rule']})")
        for gd in sorted(run.glob("gen_[0-9][0-9][0-9][0-9]")):
            fs = sorted((gd / "eval100").glob("c*_fitness.json"))
            if not fs:
                continue
            man = json.loads((gd / "manifest.json").read_text())
            T = {}
            for f in fs:
                T.update({s: v["goals_per_episode"] for s, v in json.loads(f.read_text())["hands"].items()})
            by = defaultdict(list)
            for h in man["hands"]:
                if h["spec"] in T:
                    by[h["niche"]].append(T[h["spec"]])
            n = sum(len(v) for v in by.values())
            print(f"  {gd.name}: {n}/{len(man['hands'])} scored; niche means avg {st.mean(st.mean(v) for v in by.values()):.3f},"
                  f" niche best avg {st.mean(max(v) for v in by.values()):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("init", "fork", "fork-all", "next", "trained", "eval", "report"))
    ap.add_argument("--run")
    ap.add_argument("--campaign")
    ap.add_argument("--seeds")
    ap.add_argument("--from", dest="src")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--rule", choices=("none", "cap", "margin", "global"), default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--train-run")
    ap.add_argument("--checkpoint")
    ap.add_argument("--envs", type=int, default=100, help="eval: envs per design")
    ap.add_argument("--policy-gen", type=int, help="eval: score gen --gen's designs with this gen's policy")
    ap.add_argument("--force", action="store_true", help="eval: run even when a NO_EVAL file switches evals off")
    args = ap.parse_args()
    {"init": init, "fork": fork, "fork-all": fork_all, "next": next_gen, "trained": trained, "eval": evaluate,
     "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
