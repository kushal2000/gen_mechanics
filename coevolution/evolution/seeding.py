"""The starting population: one random grammar hand per niche (descriptors.niche), drawn with a fixed seed before
generation 0, so every compared method starts from the same hands (29 Oct: the user's call).

Draws ``operators.random_hand`` under the run's rules until every niche has a hand that fits the slot grid (it stops
there), or ``max_draws``; the first hand drawn per niche is kept, so the set is a function of the seed alone.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar.urdf_export import build

from . import descriptors
from .variation import design_key, fits_grid

ALL_NICHES = tuple(f"f{f}_{c}_j{b}" for f in range(2, 7) for c in ("nocut", "cut") for b in range(3))


def one_per_niche(out_dir: Path, rules, grid: dict, seed: int, max_draws: int = 8000, log_every: int = 500) -> dict:
    """Build the seed hands under ``out_dir/<id>/`` and write ``out_dir/manifest.json``; returns the manifest."""
    rng = np.random.default_rng(seed)
    kept: dict[str, dict] = {}
    seen: set[str] = set()
    t0 = time.time()
    for k in range(max_draws):
        hand = gops.random_hand(rng, rules)
        d = gh.hand_to_dict(hand)
        n = descriptors.niche(d)
        if n in kept:
            continue
        key = design_key(hand, rules)
        if key in seen:
            continue
        seen.add(key)
        rid = f"seed_{n}"
        row = build(hand, rid, out_dir / rid)
        if not fits_grid(row["spec"], grid):
            continue
        kept[n] = {"id": rid, "spec": row["spec"], "niche": n, "draw": k, **descriptors.summary(d)}
        print(f"[seeding] draw {k}: {n} ({len(kept)} niches, {time.time() - t0:.0f} s)", flush=True)
        if len(kept) == len(ALL_NICHES):
            break
        if (k + 1) % log_every == 0:
            print(f"[seeding] {k + 1} draws, {len(kept)} niches", flush=True)
    order = [n for n in ALL_NICHES if n in kept]
    manifest = {"seed": seed, "rules": next(name for name in dir(gh) if getattr(gh, name) is rules),
                "grid": grid, "draws": k + 1, "missing": [n for n in ALL_NICHES if n not in kept],
                "hands": [kept[n] for n in order]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


__all__ = ["ALL_NICHES", "one_per_niche"]
