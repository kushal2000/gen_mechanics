"""Study 3: can a learned value predict a child's score as well as running it? The critic and a design-only surrogate
against the explicit 4-env eval, all judged by 100-env scores.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_value.py --phase make [--iteration 1]
    sbatch --array=0-<n-1> experiments/05nov_noevolution/studies/study_value_eval.sub        (jobs.txt, one per task)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_value.py --phase surrogate
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_value.py --phase picks   (then sbatch again)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_value.py --phase report

Children: study 2's (every one-step child of the top 4 hands per niche, it<N>_mutations/children.json).

make writes jobs.txt (manifest, envs per hand, output, seed) for eval_values.py:
    value pass   every child at 4 envs, seed 3, recording the critic (V at the reset state, averaged over 1 s, 4 s,
                 the episode); with study 2's seed-2 run that is two independent 4-env scores per child
    parents      the 128 parents at 100 envs, recording the critic
    truth        8 random children per parent (1,024) at 100 envs, seed 4: the ground truth
surrogate: a design-only value net (DeepSets over fingers, an ensemble of 5) fitted to the iteration's own
    population scores (gen_NNNN/fitness.json, 4,096 hands, 4 envs each), the House-of-Dextra-style screen that needs
    no simulation; predicts every child.
picks: per parent and per predictor, its best child; those not already in the truth set get 100-env jobs (jobs2.txt).
report: for each predictor (4-env eval, 8-env eval, the critic's variants, truncated evals, the surrogate) against the
    100-env truth: correlation over the random children, within-parent rank correlation, whether it calls "better than
    the parent" right, and the true score of the child it would pick per parent.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
RUN = REPO / "debug_outputs/evolution/noevolution_4096"
FIT = "goals_per_episode"
TRUTH_ENVS = 100


def out_dir(it: int) -> Path:
    return REPO / "debug_outputs/noevolution/studies" / f"it{it}_value"


def mut_dir(it: int) -> Path:
    return REPO / "debug_outputs/noevolution/studies" / f"it{it}_mutations"


def _write_manifest(path: Path, gen_man: dict, rows: list[dict]) -> None:
    path.write_text(json.dumps({"generation": gen_man["generation"], "grid": gen_man["grid"],
                                "hands": [{"id": r["id"], "spec": r["spec"]} for r in rows]}))


def make(args) -> None:
    out = out_dir(args.iteration)
    out.mkdir(parents=True, exist_ok=True)
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    meta = json.loads((mut_dir(args.iteration) / "children.json").read_text())
    jobs = []
    for c in range(meta["chunks"]):          # the value pass reuses study 2's chunk manifests
        jobs.append((mut_dir(args.iteration) / f"chunk{c:02d}_manifest.json", 4, out / f"vpass{c:02d}.json", 3))
    parents = [h for h in gman["hands"] if h["id"] in set(meta["parents"])]
    _write_manifest(out / "parents_manifest.json", gman, parents)
    jobs.append((out / "parents_manifest.json", TRUTH_ENVS, out / "parents.json", 4))
    rng = random.Random(0)
    by = defaultdict(list)
    for r in meta["children"]:
        by[r["parent"]].append(r)
    truth = [r for p in meta["parents"] for r in rng.sample(by[p], min(args.per_parent, len(by[p])))]
    for k in range(0, len(truth), args.chunk_hands):
        mp = out / f"truth{k // args.chunk_hands:02d}_manifest.json"
        _write_manifest(mp, gman, truth[k:k + args.chunk_hands])
        jobs.append((mp, TRUTH_ENVS, out / f"truth{k // args.chunk_hands:02d}.json", 4))
    (out / "truth_ids.json").write_text(json.dumps([r["id"] for r in truth]))
    (out / "jobs.txt").write_text("".join(f"{m} {e} {o} {s}\n" for m, e, o, s in jobs))
    print(f"[value] {len(jobs)} jobs ({meta['chunks']} value-pass chunks, parents, {len(truth)} truth children): "
          f"sbatch --array=0-{len(jobs) - 1} experiments/05nov_noevolution/studies/study_value_eval.sub")


# ---------------------------------------------------------------- the design-only surrogate
KINDS = {(0, 0): 0, (90, 0): 1, (0, 90): 2}
SLOTS = 5


def featurise(design: dict):
    """(fingers x 31 tokens, hand vector): base y, z, facing (cos, sin), tilt, joint count, then per joint slot
    present + kind one-hot + length; hand: finger count, motor count."""
    import numpy as np
    toks = []
    for f in design["fingers"]:
        a = np.deg2rad(f["facing"])
        t = [f["y"] / 100, f["z"] / 100, np.cos(a), np.sin(a), f["tilt"] / 90, len(f["joints"]) / SLOTS]
        for s in range(SLOTS):
            if s < len(f["joints"]):
                j = f["joints"][s]
                oh = [0.0, 0.0, 0.0]
                oh[KINDS.get(tuple(j["axis"]), 0)] = 1.0
                t += [1.0] + oh + [j["length"] / 90]
            else:
                t += [0.0] * 5
        toks.append(t)
    motors = sum(len(f["joints"]) for f in design["fingers"])
    return np.asarray(toks, np.float32), np.asarray([len(design["fingers"]) / 6, motors / 25], np.float32)


def _batch(designs):
    import torch
    feats = [featurise(d) for d in designs]
    F = max(len(t) for t, _ in feats)
    X = torch.zeros(len(feats), F, feats[0][0].shape[1])
    M = torch.zeros(len(feats), F)
    H = torch.stack([torch.from_numpy(h) for _, h in feats])
    for i, (t, _) in enumerate(feats):
        X[i, :len(t)] = torch.from_numpy(t)
        M[i, :len(t)] = 1
    return X, M, H


def _net(d_in: int):
    import torch
    from torch import nn

    class DeepSets(nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(256 + 2, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, X, M, H):
            e = self.phi(X) * M[..., None]
            mean = e.sum(1) / M.sum(1, keepdim=True)
            mx = (e - 1e4 * (1 - M[..., None])).max(1).values
            return self.rho(torch.cat([mean, mx, H], -1)).squeeze(-1)
    return DeepSets()


def _load_design(spec_rel: str) -> dict:
    s = json.loads((REPO / spec_rel).read_text())
    return s["design"] if isinstance(s["design"], dict) else json.loads(s["design"])


def surrogate(args) -> None:
    import torch
    out = out_dir(args.iteration)
    g = RUN / f"gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    fit = json.loads((g / "fitness.json").read_text())["hands"]
    designs = [_load_design(h["spec"]) for h in man["hands"]]
    y = torch.tensor([fit[h["spec"]][FIT] for h in man["hands"]])
    X, M, H = _batch(designs)
    meta = json.loads((mut_dir(args.iteration) / "children.json").read_text())
    cX, cM, cH = _batch([r["design"] for r in meta["children"]])
    pX, pM, pH = X, M, H
    torch.manual_seed(0)
    n = len(y)
    perm = torch.randperm(n)
    val, tr = perm[:n // 10], perm[n // 10:]
    preds, ppreds, val_r = [], [], []
    for e in range(5):
        net = _net(X.shape[-1])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        idx = tr[torch.randint(len(tr), (len(tr),))]            # a bootstrap sample per member
        for step in range(3000):
            b = idx[torch.randint(len(idx), (256,))]
            loss = ((net(X[b], M[b], H[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pv = net(X[val], M[val], H[val])
            val_r.append(float(torch.corrcoef(torch.stack([pv, y[val]]))[0, 1]))
            preds.append(net(cX, cM, cH))
            ppreds.append(net(pX, pM, pH))
    pred = torch.stack(preds).mean(0)
    ppred = torch.stack(ppreds).mean(0)
    id_spec = {h["id"]: h["spec"] for h in man["hands"]}
    (out / "surrogate.json").write_text(json.dumps({
        "val_corr_members": val_r,
        "children": {r["id"]: float(p) for r, p in zip(meta["children"], pred)},
        "population": {h["id"]: float(p) for h, p in zip(man["hands"], ppred)},
        "parents": {p: float(ppred[[h["id"] for h in man["hands"]].index(p)]) for p in meta["parents"]
                    if p in id_spec}}))
    print(f"[value] surrogate: held-out correlation with the 4-env population scores, per member "
          f"{', '.join(f'{r:.2f}' for r in val_r)}; wrote {out / 'surrogate.json'}")


# ---------------------------------------------------------------- reading everything back
def _gather(it: int) -> dict:
    out = out_dir(it)
    meta = json.loads((mut_dir(it) / "children.json").read_text())
    rows = {r["id"]: {k: v for k, v in r.items() if k != "design"} for r in meta["children"]}
    spec_id = {r["spec"]: r["id"] for r in rows.values()}

    def load(pattern):
        d = {}
        for f in sorted(glob.glob(str(out / pattern))):
            d.update(json.loads(Path(f).read_text())["hands"])
        return d
    s2 = {}                                   # study 2's seed-2 4-env run
    for c in range(meta["chunks"]):
        f = mut_dir(it) / f"chunk{c:02d}_fitness.json"
        if f.exists():
            s2.update(json.loads(f.read_text())["hands"])
    vp, truth = load("vpass[0-9][0-9].json"), {**load("truth[0-9][0-9].json"), **load("picks*[0-9].json")}
    for spec, cid in spec_id.items():
        r = rows[cid]
        if spec in s2:
            r["eval4_a"] = s2[spec][FIT]
        if spec in vp:
            v = vp[spec]
            r["eval4_b"] = v[FIT]
            for k in ("v0", "v_1s", "v_4s", "v_episode", "goals_1s", "goals_4s"):
                r[k] = v[k]
        if "eval4_a" in r and "eval4_b" in r:
            r["eval8"] = (r["eval4_a"] + r["eval4_b"]) / 2
        if spec in truth:
            r["truth"] = truth[spec][FIT]
            r["truth_v0"] = truth[spec]["v0"]
    sur = out / "surrogate.json"
    if sur.exists():
        s = json.loads(sur.read_text())
        for cid, p in s["children"].items():
            rows[cid]["surrogate"] = p
    g = RUN / f"gen_{it:04d}"
    id_spec = {h["id"]: h["spec"] for h in json.loads((g / "manifest.json").read_text())["hands"]}
    par = json.loads((out / "parents.json").read_text())["hands"] if (out / "parents.json").exists() else {}
    parents = {p: {"truth": par[id_spec[p]][FIT], "v0": par[id_spec[p]]["v0"], "v_1s": par[id_spec[p]]["v_1s"],
                   "v_4s": par[id_spec[p]]["v_4s"], "v_episode": par[id_spec[p]]["v_episode"]}
               for p in meta["parents"] if id_spec[p] in par}
    if sur.exists():
        for p, v in json.loads(sur.read_text())["parents"].items():
            if p in parents:
                parents[p]["surrogate"] = v
    # one rollout, two signals: the 4-env eval and the critic over its first second, each z-scored over the children
    # (the parents scored on the same scale)
    for a, b, key in (("eval4_b", "v_1s", "eval4_v1s"), ("eval4_b", "v_4s", "eval4_v4s")):
        xs = [r for r in rows.values() if a in r and b in r]
        if len(xs) > 10:
            ma, sa = st.mean(r[a] for r in xs), st.pstdev(r[a] for r in xs)
            mb, sb = st.mean(r[b] for r in xs), st.pstdev(r[b] for r in xs)
            for r in xs:
                r[key] = (r[a] - ma) / sa + (r[b] - mb) / sb
            for P in parents.values():
                if b in P:
                    P[key] = (P["truth"] - ma) / sa + (P[b] - mb) / sb
    return {"rows": rows, "parents": parents, "meta": meta}


PREDICTORS = [("eval4_a", "explicit eval, 4 envs"), ("eval8", "explicit eval, 8 envs"),
              ("goals_4s", "truncated eval, goals by 4 s"), ("goals_1s", "truncated eval, goals by 1 s"),
              ("v0", "critic V(reset state)"), ("v_1s", "critic, mean over 1 s"), ("v_4s", "critic, mean over 4 s"),
              ("v_episode", "critic, mean over the episode"), ("surrogate", "design-only value net"),
              ("eval4_v1s", "4-env eval + critic 1 s (one rollout)"),
              ("eval4_v4s", "4-env eval + critic 4 s (one rollout)")]


def picks(args) -> None:
    out = out_dir(args.iteration)
    D = _gather(args.iteration)
    by = defaultdict(list)
    for r in D["rows"].values():
        by[r["parent"]].append(r)
    need = {}
    for p, rs in by.items():
        for key, _ in PREDICTORS:
            cand = [r for r in rs if key in r]
            if cand:
                best = max(cand, key=lambda r: r[key])
                if "truth" not in best:
                    need[best["id"]] = best
    g = RUN / f"gen_{args.iteration:04d}"
    gman = json.loads((g / "manifest.json").read_text())
    rows = list(need.values())
    rnd = 1 + sum(1 for _ in out.glob("jobs*.txt"))          # jobs.txt is round 1; picks rounds are 2, 3, ...
    pre = "picks" if rnd == 2 else f"picks_r{rnd}_"
    jobs = []
    for k in range(0, len(rows), args.chunk_hands):
        mp = out / f"{pre}{k // args.chunk_hands:02d}_manifest.json"
        _write_manifest(mp, gman, rows[k:k + args.chunk_hands])
        jobs.append((mp, TRUTH_ENVS, out / f"{pre}{k // args.chunk_hands:02d}.json", 4))
    (out / f"jobs{rnd}.txt").write_text("".join(f"{m} {e} {o} {s}\n" for m, e, o, s in jobs))
    print(f"[value] {len(rows)} picked children without a 100-env score, {len(jobs)} jobs: "
          f"JOBS={out / f'jobs{rnd}.txt'} sbatch --array=0-{len(jobs) - 1} "
          f"experiments/05nov_noevolution/studies/study_value_eval.sub")


def _rank(x):
    o = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def _spearman(a, b):
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    return st.correlation(_rank(a), _rank(b))


def report(args) -> None:
    out = out_dir(args.iteration)
    D = _gather(args.iteration)
    rows, parents = D["rows"], D["parents"]
    truth_ids = set(json.loads((out / "truth_ids.json").read_text()))
    rnd = [rows[i] for i in truth_ids if "truth" in rows[i]]
    print(f"[value] iteration {args.iteration}: {len(rnd)} random children with a {TRUTH_ENVS}-env truth, "
          f"{len(parents)} parents")
    t_sd = st.pstdev(r["truth"] for r in rnd)
    print(f"  truth: mean {st.mean(r['truth'] for r in rnd):.3f}, sd {t_sd:.3f}\n")
    by = defaultdict(list)
    for r in rnd:
        by[r["parent"]].append(r)
    allby = defaultdict(list)
    for r in rows.values():
        allby[r["parent"]].append(r)
    hdr = (f"  {'predictor':34s} {'pearson':>7s} {'spearman':>8s} {'within-parent':>13s} {'better-than-parent':>18s}"
           f" {'pick: true score':>16s} {'pick > parent':>13s}")
    print(hdr)
    summary = []
    for key, name in PREDICTORS + [("parent", "the parent's own score (no info)")]:
        if key == "parent":
            xs = [(parents[r["parent"]]["truth"], r["truth"]) for r in rnd if r["parent"] in parents]
        else:
            xs = [(r[key], r["truth"]) for r in rnd if key in r]
        if len(xs) < 10:
            continue
        a, b = zip(*xs)
        pear = st.correlation(a, b) if len(set(a)) > 1 else float("nan")
        spr = _spearman(list(a), list(b))
        wp = [s for p, rs in by.items() for s in [_spearman([r[key] for r in rs if key in r],
                                                           [r["truth"] for r in rs if key in r])]
              if key != "parent" and s is not None]
        # better than the parent? the predictor's child vs the predictor's parent (eval predictors: the parent's truth)
        sign = []
        for r in rnd:
            P = parents.get(r["parent"])
            if P is None or key == "parent" or key not in r:
                continue
            pk = P.get(key) if key in P else (P["truth"] if key.startswith(("eval", "goals")) else None)
            if pk is None:
                continue
            sign.append((r[key] > pk) == (r["truth"] > P["truth"]))
        # the child each predictor picks per parent, and its true score
        pk_true, pk_gain = [], []
        if key != "parent":
            for p, rs in allby.items():
                cand = [r for r in rs if key in r]
                if not cand or p not in parents:
                    continue
                best = max(cand, key=lambda r: r[key])
                if "truth" in best:
                    pk_true.append(best["truth"])
                    pk_gain.append(best["truth"] > parents[p]["truth"])
        line = (f"  {name:34s} {pear:7.2f} {spr if spr is not None else float('nan'):8.2f} "
                f"{st.mean(wp) if wp else float('nan'):13.2f} "
                f"{(sum(sign) / len(sign)) if sign else float('nan'):18.0%} "
                f"{(st.mean(pk_true) if pk_true else float('nan')):16.3f} "
                f"{(sum(pk_gain) / len(pk_gain)) if pk_gain else float('nan'):13.0%}")
        print(line + (f"   ({len(pk_true)}/{len(allby)} picks scored)" if key != "parent" and
                      len(pk_true) < len(allby) else ""))
        summary.append({"predictor": key, "name": name, "pearson": pear, "spearman": spr,
                        "within_parent_spearman": st.mean(wp) if wp else None,
                        "better_than_parent_acc": (sum(sign) / len(sign)) if sign else None,
                        "pick_true_mean": st.mean(pk_true) if pk_true else None,
                        "pick_beats_parent": (sum(pk_gain) / len(pk_gain)) if pk_gain else None,
                        "picks_scored": len(pk_true)})
    oracle = [max(r["truth"] for r in rs) for rs in by.values()]
    rand = [st.mean(r["truth"] for r in rs) for rs in by.values()]
    print(f"\n  reference, on the 8 random children per parent: a random pick {st.mean(rand):.3f}, the best of the 8 "
          f"by truth {st.mean(oracle):.3f}; parents {st.mean(p['truth'] for p in parents.values()):.3f}")
    vt = [(r["v0"], r["truth_v0"]) for r in rnd if "v0" in r]
    if vt:
        a, b = zip(*vt)
        print(f"  critic noise: V(reset) from 4 envs vs from 100 envs, correlation {st.correlation(a, b):.2f}")
    (out / "report.json").write_text(json.dumps({"summary": summary, "rows": list(rows.values()),
                                                 "parents": parents}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("make", "surrogate", "picks", "report"), required=True)
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--per-parent", type=int, default=8)
    ap.add_argument("--chunk-hands", type=int, default=256)
    args = ap.parse_args()
    {"make": make, "surrogate": surrogate, "picks": picks, "report": report}[args.phase](args)


if __name__ == "__main__":
    main()
