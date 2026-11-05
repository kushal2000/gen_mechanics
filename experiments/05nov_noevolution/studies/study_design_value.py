"""Study 4: a design value function, V(design) -> goals/episode, trained on each design's TRAINING goals/episode at
the end of an iteration, against the same net trained on the loop's 4-env eval, judged by 100-env scores.

    sbatch experiments/05nov_noevolution/studies/training_like.sub     (the targets; ITERATION=1)
    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_design_value.py [--iteration 1] [--window 0]
    (it writes a picks job file when a net's best child per parent lacks a 100-env score; sbatch it, run again)

Targets per design (the 4,096 hands of gen_NNNN):
    train_all     training goals/episode over its 6 envs (every SAPG block), the episodes finished in the last
                  --window dumps (0: all but the first, whose episodes the staggered start cut short; one dump =
                  1,600 steps = 100 epochs) of the training-like rollout (eval_training_like.py: the checkpoint
                  rolled out as in training, no updates)
    train_leader  the same over its leader-block env only (coefficient 0, the last block: what the evals run)
    eval4         the loop's own post-training eval (gen_NNNN/fitness.json: 4 envs, one episode each)
The truth: 100-env scores under the iteration's policy -- 512 population hands (study 1: top 8 + random 8 per niche)
and the children of study 3 (1,024 random children of the 128 parents, plus each predictor's pick per parent).

1. How good is each target as a measurement? Its correlation with the truth on the 512 hands (and within a niche).
2. A design value net per target (study_value.py's DeepSets over fingers, 5 bootstrap members): on the population,
   5-fold out-of-fold predictions on the 512; on the children (unseen designs), the all-data net: correlation, rank
   within a parent's children, and the true score of the child it picks per parent.
Writes debug_outputs/noevolution/studies/it<N>_design_value/report.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_value as sv  # noqa: E402

RUN = REPO / "debug_outputs/evolution/noevolution_4096"
BLOCK = 4096                 # expl_coef_block_size: SAPG block of env e is e // BLOCK; the leader is the last


def targets(it: int, window: int) -> dict[str, dict[str, float]]:
    """{target name: {hand spec: value}} plus the episode counts behind the training targets."""
    d = REPO / "debug_outputs/noevolution/studies" / f"it{it}_design_value" / "training_like"
    if not (d / "done.json").exists():
        raise SystemExit(f"no finished training-like rollout in {d}: sbatch experiments/05nov_noevolution/studies/"
                         f"training_like.sub")
    dumps = sorted(d.glob("step_*.npz"))
    if not dumps:
        raise SystemExit(f"no dumps yet in {d}")
    use = dumps[-window:] if window else dumps[1:]
    meta = np.load(d / "meta.npz")
    hands = json.loads((d / "hands.json").read_text())
    hoe = meta["hand_of_env"].astype(int)
    n = len(hoe)
    lead = n // BLOCK - 1
    out = {"train_all": {}, "train_leader": {}, "episodes_all": {}, "episodes_leader": {}}
    first = np.load(use[0])
    if "goals" in first.files and "env" in first.files:          # per-episode rows (the logger since 6 Nov)
        env = np.concatenate([np.load(f)["env"] for f in use]).astype(int)
        goals = np.concatenate([np.load(f)["goals"] for f in use])
        ones = np.ones(len(env))
    else:                                                         # per-env sums (the first version)
        env = np.arange(n)
        goals = sum(np.load(f)["goals"] for f in use)
        ones = sum(np.load(f)["episodes"] for f in use)
    for name, mask in (("all", np.ones(len(env), bool)), ("leader", env // BLOCK == lead)):
        ep = np.bincount(hoe[env[mask]], ones[mask], minlength=len(hands))
        gl = np.bincount(hoe[env[mask]], goals[mask], minlength=len(hands))
        for h, spec in enumerate(hands):
            out[f"episodes_{name}"][spec] = float(ep[h])
            if ep[h] > 0:
                out[f"train_{name}"][spec] = float(gl[h] / ep[h])
    fit = json.loads((RUN / f"gen_{it:04d}" / "fitness.json").read_text())["hands"]
    out["eval4"] = {s: v[sv.FIT] for s, v in fit.items()}
    print(f"[design value] {len(use)}/{len(dumps)} dumps of {d} ({len(use) * 1600} steps); episodes per hand: all "
          f"blocks {st.mean(out['episodes_all'].values()):.1f}, leader {st.mean(out['episodes_leader'].values()):.1f}")
    return out


def _corr(a, b):
    return st.correlation(a, b) if len(a) > 2 and len(set(a)) > 1 and len(set(b)) > 1 else float("nan")


def fit_net(X, M, H, y, seed: int, members: int = 5, steps: int = 3000):
    import torch
    nets = []
    g = torch.Generator().manual_seed(seed)
    for e in range(members):
        torch.manual_seed(seed * 100 + e)
        net = sv._net(X.shape[-1]).to(X.device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        idx = torch.randint(len(y), (len(y),), generator=g)      # bootstrap
        for _ in range(steps):
            b = idx[torch.randint(len(idx), (256,), generator=g)].to(X.device)
            loss = ((net(X[b], M[b], H[b]) - y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        nets.append(net.eval())

    def predict(X2, M2, H2):
        with torch.no_grad():
            return torch.stack([n(X2, M2, H2) for n in nets]).mean(0).cpu()
    return predict


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=1)
    ap.add_argument("--window", type=int, default=0, help="last N design_stats dumps (0: all)")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    it = args.iteration
    out = REPO / "debug_outputs/noevolution/studies" / f"it{it}_design_value"
    out.mkdir(parents=True, exist_ok=True)
    T = targets(it, args.window)
    g = RUN / f"gen_{it:04d}"
    man = json.loads((g / "manifest.json").read_text())
    specs = [h["spec"] for h in man["hands"]]
    niche = {h["spec"]: h["niche"] for h in man["hands"]}
    id_spec = {h["id"]: h["spec"] for h in man["hands"]}
    truth_pop = {id_spec[r["id"]]: r["env100"] for r in
                 json.loads((REPO / f"debug_outputs/noevolution/studies/it{it}_noise/scores.json").read_text())}

    # 1. the targets as measurements
    names = [("eval4", "loop eval, 4 envs x 1 episode"), ("train_leader", "training, leader block"),
             ("train_all", "training, all 6 blocks")]
    print(f"\n  1. each target against the 100-env truth on {len(truth_pop)} population hands")
    print(f"  {'target':32s} {'episodes/hand':>13s} {'pearson':>7s} {'spearman':>8s} {'within niche':>12s}")
    res1 = {}
    for key, name in names:
        ks = [s for s in truth_pop if s in T[key]]
        a, b = [T[key][s] for s in ks], [truth_pop[s] for s in ks]
        by = defaultdict(list)
        for s in ks:
            by[niche[s]].append((T[key][s], truth_pop[s]))
        wn = [x for v in by.values() for x in [sv._spearman([p for p, _ in v], [q for _, q in v])] if x is not None]
        eps = 4.0 if key == "eval4" else st.mean(T[f"episodes_{key.split('_')[1]}"][s] for s in ks)
        res1[key] = {"pearson": _corr(a, b), "spearman": sv._spearman(a, b), "within_niche": st.mean(wn)}
        print(f"  {name:32s} {eps:13.1f} {res1[key]['pearson']:7.2f} {res1[key]['spearman']:8.2f} "
              f"{res1[key]['within_niche']:12.2f}")

    # 2. design value nets
    designs = [sv._load_design(s) for s in specs]
    dev = "cuda" if torch.cuda.is_available() else "cpu"     # ~20 s per member on CPU, 90 members
    X, M, H = (t.to(dev) for t in sv._batch(designs))
    meta = json.loads((sv.mut_dir(it) / "children.json").read_text())
    cX, cM, cH = (t.to(dev) for t in sv._batch([r["design"] for r in meta["children"]]))
    D = sv._gather(it)
    rows, parents = D["rows"], D["parents"]
    truth_ids = set(json.loads((sv.out_dir(it) / "truth_ids.json").read_text()))
    rng = np.random.default_rng(0)
    fold = rng.integers(0, args.folds, len(specs))
    preds_children = {}
    print(f"\n  2. a design value net per target (DeepSets, 5 members); population: {args.folds}-fold out-of-fold on the "
          f"{len(truth_pop)} truth hands; children: unseen designs, {len(truth_ids)} random + the picks")
    print(f"  {'net trained on':32s} {'pop pearson':>11s} {'pop within niche':>16s} {'child pearson':>13s} "
          f"{'within parent':>13s} {'pick: true':>10s} {'pick > parent':>13s}")
    res2 = {}
    for key, name in names:
        ok = [i for i, s in enumerate(specs) if s in T[key]]
        y = torch.tensor([T[key].get(s, float("nan")) for s in specs], dtype=torch.float32, device=dev)
        oof = torch.full((len(specs),), float("nan"))
        for f in range(args.folds):
            tr = torch.tensor([i for i in ok if fold[i] != f], device=dev)
            te = torch.tensor([i for i in range(len(specs)) if fold[i] == f], device=dev)
            pr = fit_net(X[tr], M[tr], H[tr], y[tr], seed=f)
            oof[te.cpu()] = pr(X[te], M[te], H[te])
        ks = list(truth_pop)
        a = [float(oof[specs.index(s)]) for s in ks]
        b = [truth_pop[s] for s in ks]
        by = defaultdict(list)
        for s, p in zip(ks, a):
            by[niche[s]].append((p, truth_pop[s]))
        wn = st.mean(x for v in by.values() for x in [sv._spearman([p for p, _ in v], [q for _, q in v])]
                     if x is not None)
        tr = torch.tensor(ok, device=dev)
        pc = fit_net(X[tr], M[tr], H[tr], y[tr], seed=99)(cX, cM, cH)
        pk = f"dv_{key}"
        for r, p in zip(meta["children"], pc.tolist()):
            rows[r["id"]][pk] = p
        preds_children[pk] = {r["id"]: p for r, p in zip(meta["children"], pc.tolist())}
        rnd = [rows[i] for i in truth_ids if "truth" in rows[i]]
        cp = _corr([r[pk] for r in rnd], [r["truth"] for r in rnd])
        byp, allp = defaultdict(list), defaultdict(list)
        for r in rnd:
            byp[r["parent"]].append(r)
        for r in rows.values():
            allp[r["parent"]].append(r)
        wp = st.mean(x for v in byp.values() for x in [sv._spearman([r[pk] for r in v], [r["truth"] for r in v])]
                     if x is not None)
        picks = {p: max(v, key=lambda r: r[pk]) for p, v in allp.items()}
        scored = [(r["truth"], parents[p]["truth"]) for p, r in picks.items() if "truth" in r and p in parents]
        res2[key] = {"pop_pearson": _corr(a, b), "pop_within_niche": wn, "child_pearson": cp, "within_parent": wp,
                     "pick_true": st.mean(t for t, _ in scored) if scored else None,
                     "pick_beats_parent": (sum(t > q for t, q in scored) / len(scored)) if scored else None,
                     "picks_scored": len(scored), "missing": [r["id"] for r in picks.values() if "truth" not in r]}
        print(f"  {name:32s} {res2[key]['pop_pearson']:11.2f} {wn:16.2f} {cp:13.2f} {wp:13.2f} "
              f"{res2[key]['pick_true'] if scored else float('nan'):10.3f} "
              f"{res2[key]['pick_beats_parent'] if scored else float('nan'):13.0%}"
              + (f"   ({len(scored)}/{len(picks)} picks scored)" if len(scored) < len(picks) else ""))
    missing = sorted({i for v in res2.values() for i in v["missing"]})
    if missing:
        gman = json.loads((g / "manifest.json").read_text())
        so = sv.out_dir(it)
        rnd_no = 1 + sum(1 for _ in so.glob("jobs*.txt"))
        mp = so / f"picks_r{rnd_no}_00_manifest.json"
        sv._write_manifest(mp, gman, [rows[i] for i in missing])
        (so / f"jobs{rnd_no}.txt").write_text(f"{mp} {sv.TRUTH_ENVS} {so / f'picks_r{rnd_no}_00.json'} 4\n")
        print(f"\n  {len(missing)} picks lack a 100-env score: JOBS={so / f'jobs{rnd_no}.txt'} sbatch --array=0-0 "
              f"experiments/05nov_noevolution/studies/study_value_eval.sub, then run this again")
    print(f"\n  for comparison (study 3, children): explicit 4-env eval within parent 0.37, pick 0.433; 8-env 0.45, "
          f"0.456; eval + critic 1 s 0.47, 0.470; parents 0.418")
    (out / "report.json").write_text(json.dumps({"window": args.window, "targets": res1, "nets": res2}, indent=1))
    (out / "children_predictions.json").write_text(json.dumps(preds_children))


if __name__ == "__main__":
    main()
