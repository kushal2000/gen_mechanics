"""Candidate MAP-Elites niche cells that capture commercial hands (7 Nov). Every descriptor is computed the same way
from a grammar design's URDF and from a vendor URDF (the joint tokens' own geometry), so each commercial hand maps to
a cell; then the co-evolution population's occupancy of those cells shows what the archive does not cover.

Descriptors (the structure the joint-transformer policy is sensitive to, per the 7 Nov counterfactuals):
  fingers   finger count
  layout    largest angular gap between neighbouring fingers about the palm normal (deg): radial / partial /
            one-sided / extreme
  knuckles  fingers whose first joint's link is short (< 40 mm: a co-located 2-DOF knuckle has 0 mm)
  size      mean finger length, base joint to fingertip pad (mm)

    python experiments/06nov_coevolve_v2/niche_design.py [--run debug_outputs/coevolve/v2/k5_none --gen 4]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv, _a = sys.argv[:1], sys.argv[1:]
import commercial_coverage as cc                         # noqa: E402  (features())

LAYOUT = [0, 180, 240, 300, 361]          # radial / partial / one-sided / extreme
KNUCKLE = [0, 1, 3, 99]                   # none / 1-2 / 3+
SIZE = [0, 130, 170, 999]                 # small (commercial) / medium / large (grammar default)
COMMERCIAL = ("allegro", "leap", "shadow", "dex3", "tesollo", "wuji2", "xhand")


def cell(f):
    b = lambda v, edges: int(np.searchsorted(edges, v, side="right") - 1)     # noqa: E731
    return (f["n"], b(f["gap"], LAYOUT), b(f["short"], KNUCKLE), b(f["len"], SIZE))


def name(c):
    n, l, k, s = c
    return f"f{n}_{['radial', 'partial', 'onesided', 'extreme'][l]}_{['k0', 'k1-2', 'k3+'][k]}_{['small', 'medium', 'large'][s]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="debug_outputs/coevolve/v2/k5_none")
    ap.add_argument("--gen", type=int, default=4)
    ap.add_argument("--out", default="debug_outputs/coevolve/studies/niche_design")
    a = ap.parse_args(_a)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((REPO / a.run / f"gen_{a.gen:04d}/manifest.json").read_text())
    lab = json.loads((REPO / a.run / f"gen_{a.gen:04d}/labels.json").read_text())
    cache = out / f"features_{Path(a.run).name}_g{a.gen}.json"
    feats = json.loads(cache.read_text()) if cache.exists() else {}
    for i, h in enumerate(m["hands"]):
        if h["id"] in feats:
            continue
        try:
            feats[h["id"]] = cc.features(h["spec"])
        except Exception as e:                           # noqa: BLE001
            feats[h["id"]] = None
        if i % 500 == 0:
            cache.write_text(json.dumps(feats))
    cache.write_text(json.dumps(feats))
    occ, goals = Counter(), {}
    for hid, f in feats.items():
        if f:
            c = cell(f)
            occ[c] += 1
            goals.setdefault(c, []).append(lab.get(hid, [0])[0])
    n_cells = len(range(2, 7)) * (len(LAYOUT) - 1) * (len(KNUCKLE) - 1) * (len(SIZE) - 1)
    print(f"{a.run} gen {a.gen}: {sum(occ.values())} designs in {len(occ)} of {n_cells} candidate cells")
    print(f"\n{'commercial hand':10s} {'cell':38s} {'designs there':>13s} {'their mean label':>16s}")
    rows = []
    for h in COMMERCIAL:
        f = cc.features(f"{h}_left_uniform_handonly")
        c = cell(f)
        g = goals.get(c, [])
        rows.append({"hand": h, "cell": name(c), "features": f, "designs": occ.get(c, 0)})
        print(f"{h:10s} {name(c):38s} {occ.get(c, 0):13d} {np.mean(g) if g else float('nan'):16.2f}")
    print("\npopulation by axis:")
    for ax, labels in ((1, ["radial", "partial", "onesided", "extreme"]), (2, ["k0", "k1-2", "k3+"]), (3, ["small", "medium", "large"])):
        cnt = Counter()
        for c, k in occ.items():
            cnt[c[ax]] += k
        print("   " + "  ".join(f"{labels[i]}: {cnt.get(i, 0)}" for i in range(len(labels))))
    (out / "commercial_cells.json").write_text(json.dumps({"layout_edges": LAYOUT, "knuckle_edges": KNUCKLE,
                                                          "size_edges": SIZE, "commercial": rows,
                                                          "occupancy": {name(c): k for c, k in occ.items()}}, indent=1))


if __name__ == "__main__":
    main()
