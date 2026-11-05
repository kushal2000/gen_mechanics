"""Which design value net? The per-finger transformer (design_value_net.py, one deterministic net) against the DeepSets
net (study_value.py; one net, and the 5-member ensemble), all on iteration --iteration's 100-env labels, judged on the
same data:

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/study_vf_arch.py [--iteration 3]

  population   5-fold out-of-fold predictions for the 4,096 labelled designs: correlation, rank within each niche
  beam         every design the study-5 beam search scored (100 envs): correlation, rank within each round's children
               -- trained on the population only, and (forward) on the population + the beam's earlier rounds
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import design_value_net as dvn  # noqa: E402
import study_design_value as sdv  # noqa: E402
import study_value as sv  # noqa: E402

RUN = REPO / "debug_outputs/evolution/noevolution_4096"


def deepsets(members: int):
    def fit(designs, labels):
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        X, M, H = (t.to(dev) for t in sv._batch(designs))
        y = torch.tensor(labels, dtype=torch.float32, device=dev)
        nets = [sdv.fit_net(X, M, H, y, seed=s, members=1) for s in range(members)]

        def predict(ds):
            out = []
            for k in range(0, len(ds), 2048):
                Xc, Mc, Hc = (t.to(dev) for t in sv._batch(ds[k:k + 2048]))
                out.append(torch.stack([p(Xc, Mc, Hc) for p in nets]).mean(0).numpy())
            return np.concatenate(out)
        return predict
    return fit


MODELS = {"transformer (1 net)": lambda d, y: dvn.fit_transformer(d, y),
          "DeepSets (1 net)": deepsets(1), "DeepSets (5-net ensemble)": deepsets(5)}


def corr(a, b):
    return st.correlation(list(a), list(b)), sv._spearman(list(a), list(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--start", default="f4_m12-13_126")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    g = RUN / f"gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    lab = {}
    for f in sorted((REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_env100").glob("c*_fitness.json")):
        lab.update({s: v["goals_per_episode"] for s, v in json.loads(f.read_text())["hands"].items()})
    hs = [h for h in man["hands"] if h["spec"] in lab]
    pd = [sv._load_design(h["spec"]) for h in hs]
    py = [lab[h["spec"]] for h in hs]
    niche = [h["niche"] for h in hs]
    S = json.loads((REPO / "debug_outputs/noevolution/studies" / f"beam_{args.start}_it{args.iteration}" /
                    "state.json").read_text())
    B = [(r["design"], st.mean(r["scores"]), r["round"]) for r in S["designs"].values() if r["scores"] and r["round"] > 0]
    fold = np.random.default_rng(0).integers(0, args.folds, len(pd))
    rounds = sorted({r for _, _, r in B})
    results = {}
    for name, fit in MODELS.items():
        t0 = time.time()
        oof = np.zeros(len(pd))
        for k in range(args.folds):
            tr = [i for i in range(len(pd)) if fold[i] != k]
            te = [i for i in range(len(pd)) if fold[i] == k]
            p = fit([pd[i] for i in tr], [py[i] for i in tr])
            oof[te] = p([pd[i] for i in te])
        by = defaultdict(list)
        for i in range(len(pd)):
            by[niche[i]].append((oof[i], py[i]))
        wn = st.mean(x for v in by.values() for x in [sv._spearman([a for a, _ in v], [b for _, b in v])] if x is not None)
        pp, ps = corr(oof, py)
        full = fit(pd, py)
        bp = full([d for d, _, _ in B])
        bpe, bsp = corr(bp, [t for _, t, _ in B])
        within, fwd = [], []
        for r in rounds:
            idx = [k for k, (_, _, rr) in enumerate(B) if rr == r]
            if len(idx) >= 5:
                within.append(sv._spearman([bp[k] for k in idx], [B[k][1] for k in idx]))
                prior = [k for k, (_, _, rr) in enumerate(B) if rr < r]
                f2 = fit(pd + [B[k][0] for k in prior], py + [B[k][1] for k in prior])
                q = f2([B[k][0] for k in idx])
                fwd.append(sv._spearman(list(q), [B[k][1] for k in idx]))
        results[name] = {"pop_pearson": pp, "pop_spearman": ps, "pop_within_niche": wn, "beam_pearson": bpe,
                         "beam_spearman": bsp, "within_round": within, "within_round_forward": fwd,
                         "minutes": (time.time() - t0) / 60}
        print(f"[arch] {name}: done in {results[name]['minutes']:.1f} min", flush=True)
    print(f"\n[arch] iteration {args.iteration}, 100-env labels; population: {args.folds}-fold out-of-fold on "
          f"{len(pd)} designs; beam: {len(B)} designs over rounds {rounds[0]}-{rounds[-1]}")
    print(f"  {'model':28s} {'pop r':>6s} {'pop rank':>8s} {'in niche':>8s} {'beam r':>7s} {'beam rank':>9s} "
          f"{'in round (pop only)':>20s} {'in round (+ earlier beam)':>26s}")
    for name, R in results.items():
        print(f"  {name:28s} {R['pop_pearson']:6.2f} {R['pop_spearman']:8.2f} {R['pop_within_niche']:8.2f} "
              f"{R['beam_pearson']:7.2f} {R['beam_spearman']:9.2f} {st.mean(R['within_round']):20.2f} "
              f"{st.mean(R['within_round_forward']):26.2f}")
        print(f"  {'':28s} per round, pop only: " + " ".join(f"{x:.2f}" for x in R["within_round"])
              + "  |  + earlier beam: " + " ".join(f"{x:.2f}" for x in R["within_round_forward"]))
    out = REPO / "debug_outputs/noevolution/studies" / f"it{args.iteration}_vf_arch.json"
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
