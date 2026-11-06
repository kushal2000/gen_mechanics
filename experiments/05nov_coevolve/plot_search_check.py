"""Does the campaign's design search find better designs? The archives each run's search proposed after gen 0 (K rounds
of value-net search), scored with 100 envs by the gen-0 policy whose training logs labelled the search -- the
campaign's version of study 7 (experiments/05nov_noevolution/plots/vf_archive/).

Columns: the seeds (gen 0), then each run's gen-1 archive. Top row: the value net's predicted niche means; bottom row:
the 100-env truth. The net is trained on training-log goals/episode (all SAPG blocks), which sit on a different scale
from the 100-env eval, so predictions are mapped to the eval's scale with a linear fit of the seeds' training labels
to their 100-env scores (printed); correlations do not depend on it.

    .venv_isaacsim/bin/python experiments/05nov_coevolve/plot_search_check.py [--campaign debug_outputs/coevolve/v1]
        [--runs k1_none k5_none k10_none]

Writes experiments/05nov_coevolve/plots/search_check_gen1.png and prints a per-column table.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
import plot_vf_archive_grid as G  # noqa: E402


def scores(gd: Path, sub: str) -> dict[str, float]:
    """id -> 100-env goals/episode for gen dir gd's eval folder sub (chunks scored so far)."""
    man = json.loads((gd / "manifest.json").read_text())
    id_of = {h["spec"]: h["id"] for h in man["hands"]}
    out = {}
    for f in sorted((gd / sub).glob("c*_fitness.json")):
        for spec, v in json.loads(f.read_text())["hands"].items():
            out[id_of[spec]] = v["goals_per_episode"]
    return out


def archive_of(gd: Path) -> dict[str, list[str]]:
    A: dict[str, list[str]] = {}
    for h in json.loads((gd / "manifest.json").read_text())["hands"]:
        A.setdefault(h["niche"], []).append(h["id"])
    return A


def seed_predictions(run: Path, A: dict) -> dict[str, float]:
    """The run's gen-0 value net on the seeds (not saved by the search, so recomputed from value_net.pt)."""
    import torch
    import design_value_net as dvn
    reg = json.loads((run / "designs.json").read_text())
    net = dvn._net()
    net.load_state_dict(torch.load(run / "gen_0000/value_net.pt", map_location="cpu"))
    net.eval()
    ids = [i for v in A.values() for i in v]
    out = []
    with torch.no_grad():
        for k in range(0, len(ids), 2048):
            X, M, H = dvn.batch([reg[i]["design"] for i in ids[k:k + 2048]])
            out.append(net(X, M, H).numpy())
    return dict(zip(ids, np.concatenate(out).tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="debug_outputs/coevolve/v1")
    ap.add_argument("--runs", nargs="+", default=["k1_none", "k5_none", "k10_none"])
    args = ap.parse_args()
    camp = REPO / args.campaign
    src = json.loads((camp / "forks.json").read_text())["from"]
    seeds_gd = camp / src / "gen_0000"
    A0 = archive_of(seeds_gd)
    T0 = scores(seeds_gd, "eval100")
    lab = {i: v[0] for i, v in json.loads((seeds_gd / "labels.json").read_text()).items()}
    both = [i for i in T0 if i in lab]
    a, b = np.polyfit([lab[i] for i in both], [T0[i] for i in both], 1)
    r = np.corrcoef([lab[i] for i in both], [T0[i] for i in both])[0, 1]
    print(f"[search_check] calibration on {len(both)} seeds: 100-env = {a:.2f} x label + {b:.3f} (corr {r:.2f})")

    cols = [("Seeds", A0, {i: a * p + b for i, p in seed_predictions(camp / args.runs[0], A0).items()}, T0)]
    for name in args.runs:
        run = camp / name
        cfg = json.loads((run / "config.json").read_text())
        gd = run / "gen_0001"
        sd = run / "gen_0000/search"
        P = json.loads((sd / f"round_{cfg['k']:02d}_archive.json").read_text())["pred"]
        cols.append((f"K = {cfg['k']}", archive_of(gd), {i: a * p + b for i, p in P.items()},
                     scores(gd, "eval100_policy_g0000")))

    print(f"{'':8s} {'actual':>7s} {'pred':>7s} {'gap':>7s} {'corr':>5s} {'top10':>6s} {'best':>6s}  scored")
    for title, A, P, T in cols:
        ids = [i for v in A.values() for i in v]
        n = sum(i in T for i in ids)
        if n < len(ids):
            print(f"{title:8s} {'':>7s} {np.mean([P[i] for i in ids]):7.3f} {'':>7s} {'':>5s} {'':>6s} {'':>6s}  "
                  f"{n}/{len(ids)}")
            continue
        act, pr = np.mean([T[i] for i in ids]), np.mean([P[i] for i in ids])
        c = np.corrcoef([T[i] for i in ids], [P[i] for i in ids])[0, 1]
        top = np.mean([np.mean(sorted(T[i] for i in v)[-10:]) for v in A.values()])
        best = np.mean([max(T[i] for i in v) for v in A.values()])
        print(f"{title:8s} {act:7.3f} {pr:7.3f} {pr - act:+7.3f} {c:5.2f} {top:6.3f} {best:6.3f}  {n}/{len(ids)}")

    bins = sorted({G.parse(n)[1] for n in A0})
    G.FS = 7.2
    nc = len(cols)
    fig, axs = plt.subplots(2, nc, figsize=(4.3 * nc + 1.0, 5.6), facecolor=G.SURF, squeeze=False)
    for c, (title, A, P, T) in enumerate(cols):
        G.draw(axs[0, c], A, bins, lambda ids, P=P: st.mean(P[i] for i in ids), f"{title} – Predicted Niche Means")
        G.draw(axs[1, c], A, bins, lambda ids, T=T: st.mean(T[i] for i in ids) if all(i in T for i in ids) else None,
               f"{title} – Actual Niche Means")
        axs[1, c].set_xlabel("motors", color=G.INK2, fontsize=9.5)
        if c:
            for ax in axs[:, c]:
                ax.set_ylabel("")
    fig.subplots_adjust(left=0.03, right=0.955, top=0.93, bottom=0.12, hspace=0.38, wspace=0.08)
    cax = fig.add_axes([0.965, 0.2, 0.006, 0.6])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=G.CMAP, norm=plt.Normalize(0, G.VMAX)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5, labelcolor=G.INK2)
    cb.set_label("goals / episode (100 envs)", color=G.INK2, fontsize=9)
    out = Path(__file__).resolve().parent / "plots"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "search_check_gen1.png", dpi=170, facecolor=G.SURF)
    print(f"[search_check] wrote {out / 'search_check_gen1.png'}")


if __name__ == "__main__":
    main()
