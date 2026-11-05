"""The no-evolution population: 128 random hands in each of the 32 niches (4,096), built for the template scene.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/make_seed.py [--per-niche 128] [--seed 0] [--workers 12]

Rules: FEETECH_COARSE_NOCUT_RULES (no palm cuts, the mount derived). Niches: descriptors.NICHES32 (finger count x
total motors in pairs). Random hands are drawn in chunks of --chunk, chunk k from generator (seed, k), in parallel; the
chunks are consumed in order and each niche keeps its first --per-niche distinct hands (canonical key), so the set is a
function of the seed alone. Hands outside the 32 niches are dropped. Each kept hand is built (URDF, spec with the wrist
frame at the palm centre) under debug_outputs/noevolution/seed_<n>/<id>/ and listed in manifest.json there with its
niche and the pinned slot grid (6 rigid finger slots x 5 joints: no cuts, so no moving slots).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RULES = "FEETECH_COARSE_NOCUT_RULES"
GRID = {"n_rigid": 6, "n_moving": 0, "n_slots": 5}


def _draw(job):
    seed, chunk, n = job[:3]
    fingers = job[3] if len(job) > 3 else None       # --by-fingers: this chunk's finger count
    import numpy as np
    from coevolution.evolution.descriptors import niche32
    from coevolution.evolution.variation import design_key
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar import operators as ops
    rules = getattr(gh, RULES)
    if fingers is not None:     # an exact draw given the finger count (random_hand draws it first, uniformly)
        from dataclasses import replace
        rules = replace(rules, min_fingers=fingers, max_fingers=fingers)
    rng = np.random.default_rng([seed, chunk] + ([fingers] if fingers is not None else []))
    out = []
    for _ in range(n):
        try:
            h = ops.random_hand(rng, rules)
        except ops.NoHand:
            continue
        d = gh.hand_to_dict(h)
        cell = niche32(d)
        if cell is not None:
            out.append((cell, design_key(h, rules), d))
    return chunk, out


def _build(job):
    rid, design, out_dir = job
    from coevolution.evolution.variation import fits_grid
    from hand_sampler.grammar import hand as gh
    from hand_sampler.grammar.urdf_export import build
    row = build(gh.hand_from_dict(design), rid, Path(out_dir) / rid, rules_name=RULES)
    return rid, row["spec"], fits_grid(row["spec"], GRID)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-niche", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--max-chunks", type=int, default=400)
    ap.add_argument("--by-fingers", action="store_true",
                    help="draw each chunk at one finger count (2-6 in turn): the same distribution within a finger "
                         "count, 5x the rate into a row's rare cells")
    ap.add_argument("--out", default=None, help="output folder (default debug_outputs/noevolution/seed_<n>)")
    args = ap.parse_args()
    from coevolution.evolution.descriptors import NICHES32
    need = args.per_niche
    kept = {n: [] for n in NICHES32}
    keys = set()
    t0 = time.time()
    with Pool(args.workers) as pool:
        draws = 0
        jobs = ((args.seed, k, args.chunk) + ((2 + k % 5,) if args.by_fingers else ()) for k in range(args.max_chunks))
        for chunk, rows in pool.imap(_draw, jobs):
            draws += args.chunk
            for cell, key, d in rows:
                if len(kept[cell]) < need and key not in keys:
                    keys.add(key)
                    kept[cell].append(d)
            short = {n: need - len(v) for n, v in kept.items() if len(v) < need}
            if chunk % 10 == 0 or not short:
                print(f"[seed] {draws} draws, {time.time() - t0:.0f} s: {len(NICHES32) - len(short)}/32 niches full"
                      + (f"; short {dict(sorted(short.items(), key=lambda kv: -kv[1])[:4])}" if short else ""),
                      flush=True)
            if not short:
                pool.terminate()
                break
    missing = {n: need - len(v) for n, v in kept.items() if len(v) < need}
    if missing:
        raise SystemExit(f"[seed] gave up after {draws} draws; short {missing}")
    out = Path(args.out) if args.out else REPO / "debug_outputs/noevolution" / f"seed_{need * len(NICHES32)}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{n}_{k:03d}", d, str(out)) for n in NICHES32 for k, d in enumerate(kept[n])]
    with Pool(args.workers) as pool:
        built = pool.map(_build, jobs, chunksize=8)
    bad = [rid for rid, _, ok in built if not ok]
    if bad:
        raise SystemExit(f"[seed] {len(bad)} hands do not fit the slot grid {GRID}: {bad[:5]}")
    niche_of = {rid: rid.rsplit("_", 1)[0] for rid, _, _ in built}
    manifest = {"seed": args.seed, "rules": RULES, "grid": GRID, "draws": draws, "per_niche": need,
                "niches": list(NICHES32),
                "hands": [{"id": rid, "spec": spec, "niche": niche_of[rid]} for rid, spec, _ in built]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    counts = collections.Counter(niche_of.values())
    print(f"[seed] {len(built)} hands in {len(counts)} niches ({min(counts.values())}-{max(counts.values())} each) "
          f"from {draws} draws in {time.time() - t0:.0f} s -> {out / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
