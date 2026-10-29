"""Parent selection: the one thing the compared methods change.

``archive`` (the baseline): the Darwin Goedel Machine's rule over every hand with a fitness -- weight =
sigmoid(lambda * (f_rel - midpoint)) / (1 + children), parents drawn with replacement in proportion. Strong hands
are favoured, hands with many children less so, and every hand keeps a nonzero chance (stepping stones).
``map_elites`` (ours): MAP-Elites -- a filled niche chosen uniformly, its elite the parent (``elites``: niche ->
hand id, the run's map). ``greedy``: always the fittest hand (DGM's "Greedy" ablation). ``random``: no parents --
variation makes a fresh random hand for each child.
"""
from __future__ import annotations

import numpy as np

from .archive import Archive, HandRecord
from .config import EvolutionConfig


def dgm_weights(records: list[HandRecord], cfg: EvolutionConfig) -> np.ndarray:
    f = np.array([r.current() or 0.0 for r in records], dtype=float)
    # relative to the archive's range (the shaping tie-breaker can make values slightly negative)
    lo, hi = (f.min(), f.max()) if len(f) else (0.0, 1.0)
    base = min(lo, 0.0)
    rel = (f - base) / (hi - base) if hi > base else np.ones_like(f)
    s = 1.0 / (1.0 + np.exp(-cfg.sigmoid_lambda * (rel - cfg.sigmoid_midpoint)))
    h = 1.0 / (1.0 + np.array([len(r.children) for r in records], dtype=float))
    return s * h


def select_parents(archive: Archive, cfg: EvolutionConfig, rng: np.random.Generator, n: int,
                   elites: dict | None = None) -> list:
    """``n`` parent ids (with replacement); ``None`` entries mean "a fresh random hand" (strategy random)."""
    if cfg.strategy == "random":
        return [None] * n
    if cfg.strategy == "map_elites":
        ids = sorted(elites.values()) if elites else []
        if not ids:
            raise RuntimeError("map_elites: the map is empty")
        return [ids[int(i)] for i in rng.integers(0, len(ids), size=n)]
    cands = [r for r in archive.records.values() if r.current() is not None]
    if not cands:
        raise RuntimeError("no hand in the archive has a fitness to select on")
    if cfg.strategy == "greedy":
        best = max(cands, key=lambda r: r.current())
        return [best.id] * n
    w = dgm_weights(cands, cfg)
    p = w / w.sum()
    return [cands[int(i)].id for i in rng.choice(len(cands), size=n, replace=True, p=p)]


__all__ = ["dgm_weights", "select_parents"]
