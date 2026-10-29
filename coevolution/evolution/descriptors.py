"""What kind of hand a design is (a grammar design dict, ``hand_to_dict``): its niche cell and a genetic vector.

Niches (MAP-Elites' map): finger count (2-6) x palm cut x mean joints per finger in three bins (< 2.5 short,
2.5-3.5 medium, > 3.5 long) -- 30 cells, all read straight off the design. Per finger, not total: total joints mostly
counts fingers, so bins on it left niches the grammar cannot reach (2 fingers never make 15 joints); per finger, every
cell is reachable -- the rarest holds 17 of 1,500 random hands (29 Oct sample), so one-per-niche seeding fills all 30. The genetic vector (POET's novelty, k-nearest-neighbour distance over the
encoding): finger count, total joints, palm cut, joints per finger sorted (padded to 6), total finger length.
"""
from __future__ import annotations

import numpy as np

JOINT_BINS = (2.5, 3.5)        # mean joints per finger: < 2.5 short, 2.5-3.5 medium, > 3.5 long
MAX_FINGERS = 6


def summary(design: dict) -> dict:
    fingers = design["fingers"]
    per = sorted((len(f["joints"]) for f in fingers), reverse=True)
    return {"fingers": len(fingers), "joints": sum(per) + len(design.get("palm_joints", [])),
            "finger_joints": sum(per), "palm_cut": bool(design.get("palm_joints")), "joints_per_finger": per,
            "length_mm": int(sum(j["length"] for f in fingers for j in f["joints"]))}


def niche(design: dict) -> str:
    s = summary(design)
    per = s["finger_joints"] / s["fingers"]
    b = 0 if per < JOINT_BINS[0] else 1 if per <= JOINT_BINS[1] else 2
    return f"f{s['fingers']}_{'cut' if s['palm_cut'] else 'nocut'}_j{b}"


def genetic_vector(design: dict) -> np.ndarray:
    s = summary(design)
    per = (s["joints_per_finger"] + [0] * MAX_FINGERS)[:MAX_FINGERS]
    return np.array([s["fingers"], s["finger_joints"] / 2.0, 2.0 * s["palm_cut"], *per, s["length_mm"] / 100.0],
                    dtype=float)


def novelty(vec: np.ndarray, others: list[np.ndarray], k: int = 5) -> float:
    """Mean distance to the k nearest of ``others`` (POET: k = 5); inf with nothing to compare against."""
    if not others:
        return float("inf")
    d = np.sort(np.linalg.norm(np.stack(others) - vec, axis=1))
    return float(d[:k].mean())


__all__ = ["JOINT_BINS", "genetic_vector", "niche", "novelty", "summary"]
