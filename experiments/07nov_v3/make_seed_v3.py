"""v3 gen 0: 16 designs per v3 MAP-Elites cell, drawn INSIDE each cell (hand_sampler/grammar/v3.random_hand_in_cell:
link geometry only), built (v3.build); each design's cell from its design-level features, and a random --verify
fraction re-checked from the built URDF (descriptors_v3.features, the ground truth: ~5 s a hand), the run failing if
any disagrees; plus the 4,096 training designs, a stratified sample with an equal share from every occupied cell.

    .venv_isaacsim/bin/python experiments/07nov_v3/make_seed_v3.py [--per-cell 16] [--train 4096] [--workers 12]

Phase 1 tries each of the 396 cells a few times (cells with no hand are recorded as infeasible); phase 2 fills each
feasible cell to --per-cell distinct designs (design_key under the v3 rules). Outputs under --out:
  manifest.json        every kept design: {id, spec, niche (its v3 cell)}
  train_manifest.json  the --train training designs (stratified by cell)
  cells.json           per cell: kept count, tries, infeasible flag; URDF-cell mismatches
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}


def _draw(job):
    cell, seed = job
    import numpy as np
    from hand_sampler.grammar import hand as gh, v3
    h = v3.random_hand_in_cell(np.random.default_rng(seed), cell)
    if h is None:
        return cell, seed, None, None
    d = gh.hand_to_dict(h)
    return cell, seed, json.dumps(d, sort_keys=True), d


def _build(job):
    rid, cell, design, out_dir, verify = job
    from hand_sampler.grammar import hand as gh, v3
    from coevolution.evolution.descriptors_v3 import cell as cell_of, features
    try:
        hand = gh.hand_from_dict(design)
        row = v3.build(hand, rid, Path(out_dir) / rid)
        got = cell_of(v3.design_features(hand))
        urdf = cell_of(features(row["spec"])) if verify else None
        return rid, cell, row["spec"], got, urdf, None
    except Exception as e:                               # noqa: BLE001
        return rid, cell, None, None, None, f"{type(e).__name__}: {str(e)[:200]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=16)
    ap.add_argument("--train", type=int, default=4096)
    ap.add_argument("--probe", type=int, default=3, help="phase-1 tries per cell")
    ap.add_argument("--max-tries", type=int, default=48, help="tries per feasible cell in all")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", type=float, default=0.1, help="fraction re-checked from the built URDF")
    ap.add_argument("--out", default="debug_outputs/coevolve/v3/seeds")
    ap.add_argument("--cells", default="debug_outputs/coevolve/v3/feasible_cells.json",
                    help="the niche list (JSON {cells: [...]}); empty string: every analytic cell")
    args = ap.parse_args()
    from coevolution.evolution.descriptors_v3 import analytic_cells
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cells = json.loads((REPO / args.cells).read_text())["cells"] if args.cells else analytic_cells()
    print(f"[seed_v3] {len(cells)} cells from {args.cells or 'descriptors_v3.analytic_cells()'}", flush=True)
    kept = defaultdict(dict)                             # cell -> key -> design
    tries = defaultdict(int)
    t0 = time.time()
    import numpy as np
    vrng = np.random.default_rng(args.seed + 1)
    with Pool(args.workers) as pool:
        jobs = [(c, hash((args.seed, c, k)) & 0x7FFFFFFF) for c in cells for k in range(args.probe)]
        for c, s, key, d in pool.imap_unordered(_draw, jobs, chunksize=4):
            tries[c] += 1
            if d is not None:
                kept[c].setdefault(key, d)
        feasible = [c for c in cells if kept[c]]
        print(f"[seed_v3] phase 1: {len(feasible)} of {len(cells)} cells have a hand ({time.time() - t0:.0f} s)",
              flush=True)
        k = args.probe
        while True:
            todo = [c for c in feasible if len(kept[c]) < args.per_cell and tries[c] < args.max_tries]
            if not todo:
                break
            jobs = [(c, hash((args.seed, c, k + i)) & 0x7FFFFFFF) for c in todo
                    for i in range(args.per_cell - len(kept[c]))]
            k += args.per_cell
            for c, s, key, d in pool.imap_unordered(_draw, jobs, chunksize=4):
                tries[c] += 1
                if d is not None and len(kept[c]) < args.per_cell:
                    kept[c].setdefault(key, d)
            print(f"[seed_v3] phase 2: {sum(min(len(kept[c]), args.per_cell) for c in feasible)} designs "
                  f"({time.time() - t0:.0f} s)", flush=True)
        builds = [(f"{c}_{i:02d}", c, d, str(out / "hands"), bool(vrng.random() < args.verify)) for c in feasible
                  for i, d in enumerate(list(kept[c].values())[: args.per_cell])]
        hands, mismatch, errors, verified, urdf_bad = [], [], [], 0, []
        for rid, c, spec, got, urdf, err in pool.imap_unordered(_build, builds, chunksize=8):
            if err:
                errors.append((rid, err))
                continue
            if urdf is not None:
                verified += 1
                if urdf != c:
                    urdf_bad.append((rid, c, urdf))
            if got != c:
                mismatch.append((rid, c, got))
            else:
                hands.append({"id": rid, "spec": str(Path(spec).relative_to(REPO)) if Path(spec).is_absolute() else spec,
                              "niche": c})
    print(f"[seed_v3] built {len(hands)} designs; {len(mismatch)} in another cell, {len(errors)} build errors; "
          f"URDF-verified {verified}: {len(urdf_bad)} disagree ({time.time() - t0:.0f} s)", flush=True)
    if urdf_bad:
        (out / "urdf_disagree.json").write_text(json.dumps(urdf_bad, indent=1))
        raise SystemExit(f"[seed_v3] {len(urdf_bad)} hands' URDF cell differs from the design-level cell: {urdf_bad[:5]}")
    hands.sort(key=lambda h: h["id"])
    (out / "manifest.json").write_text(json.dumps({"generation": 0, "grid": GRID, "hands": hands}))
    # the training designs: an equal share from every occupied cell, the remainder spread one by one (stratified)
    rng = np.random.default_rng(args.seed)
    by = defaultdict(list)
    for h in hands:
        by[h["niche"]].append(h)
    occ = sorted(by)
    quota = {c: 0 for c in occ}
    left = min(args.train, len(hands))
    while left > 0:
        progressed = False
        for c in rng.permutation(occ):
            if left == 0:
                break
            if quota[c] < len(by[c]):
                quota[c] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    train = [h for c in occ for h in [by[c][i] for i in rng.permutation(len(by[c]))[: quota[c]]]]
    (out / "train_manifest.json").write_text(json.dumps({"generation": 0, "grid": GRID, "hands": train}))
    (out / "cells.json").write_text(json.dumps({
        "cells": {c: {"kept": len(by.get(c, [])), "tries": tries[c], "infeasible": c not in feasible,
                      "train": quota.get(c, 0)} for c in cells},
        "mismatch": mismatch, "errors": errors[:50], "urdf_verified": verified}, indent=1))
    print(f"[seed_v3] {len(occ)} occupied cells, {len(cells) - len(feasible)} infeasible; train {len(train)} "
          f"(per cell {min(quota.values()) if quota else 0}-{max(quota.values()) if quota else 0}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
