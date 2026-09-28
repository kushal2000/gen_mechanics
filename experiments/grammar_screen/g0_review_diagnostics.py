#!/usr/bin/env python3
"""Opus review of G0 (opus-review-g0.md, I39): before/after diagnostics for
review items 1 (surface mounting / cross-host spacing) and 2 (curl-axis +
opposition host-frame fix), plus the "shared hinge plane" claim check
(dropped from variants.py's docs -- see this script's own measurement of
why). Pure grammar/kinematics, no oracle dependency, so it runs before or
after ``[oracle-v2]`` lands; used to produce the numbers in
``project-notes/grammar/experiments/g0/report_v2.md``.

Run with the project's ``.venv_isaacsim`` interpreter (matches
``g0_screen.py``, even though this script itself never imports
``isaacsimenvs``):

    .venv_isaacsim/bin/python experiments/grammar_screen/g0_review_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from hand_sampler.grammar.derive import derive, sample_derivation  # noqa: E402
from hand_sampler.grammar.fk import forward_kinematics  # noqa: E402
from hand_sampler.grammar.variants import G_V1, G_V1S, G_V2S, G_V3, G_V3S  # noqa: E402


def top_level_points(model, derivation):
    tf = forward_kinematics(model, {})
    out = []
    for s in derivation.steps:
        if s.production == "Digit" and s.params.get("top_level"):
            body = f"d{s.params['digit_id']}p1"
            if body in tf:
                out.append((s.params["mount"], tf[body][:3, 3]))
    return out


def min_pairwise_sep_stats(dist, n: int) -> Dict[str, Any]:
    """Review item 1: pairwise 3-D distance (root frame, q=0) between every
    top-level digit's own mount point, over ``n`` seeds -- same-host and
    cross-host pairs both included in the overall stats, same-host broken
    out separately since that is what the OLD (pre-review) mount placement
    could not space at all."""
    seps_same_host: List[float] = []
    seps_all: List[float] = []
    n_collide = 0
    n_hands_multi = 0
    for seed in range(n):
        d = sample_derivation(seed, dist)
        m = derive(d)
        pts = top_level_points(m, d)
        if len(pts) < 2:
            continue
        n_hands_multi += 1
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                sep = float(np.linalg.norm(pts[i][1] - pts[j][1]))
                seps_all.append(sep)
                if pts[i][0] == pts[j][0]:
                    seps_same_host.append(sep)
                if sep < 1e-6:
                    n_collide += 1
    return {
        "n_hands_multi_digit": n_hands_multi,
        "n_pairs": len(seps_all),
        "n_collisions": n_collide,
        "mean_sep_mm": 1000 * float(np.mean(seps_all)) if seps_all else None,
        "median_sep_mm": 1000 * float(np.median(seps_all)) if seps_all else None,
        "p10_sep_mm": 1000 * float(np.percentile(seps_all, 10)) if seps_all else None,
        "mean_same_host_sep_mm": 1000 * float(np.mean(seps_same_host)) if seps_same_host else None,
        "median_same_host_sep_mm": 1000 * float(np.median(seps_same_host)) if seps_same_host else None,
        "frac_meeting_29mm_target": float(np.mean([s >= 0.029 for s in seps_all])) if seps_all else None,
    }


def opposition_diagnostics(dist, n: int) -> Dict[str, Any]:
    """Review item 5: the angle between the LAST top-level digit's own
    root-frame forward direction and the mean root-frame forward direction
    of the earlier digits -- "opposing" means this angle is large (near
    180 deg); ``frac_within_90deg`` is the FAILURE share opus-review-g0.md
    itself reports ("37% of 'opposing' digits point within 90 deg of the
    others"), i.e. digits the opposition prior intended to oppose but that
    ended up on the SAME side as the others instead."""
    angles_all: List[float] = []
    angles_mixed: List[float] = []
    for seed in range(n):
        d = sample_derivation(seed, dist)
        m = derive(d)
        top = [s for s in d.steps if s.production == "Digit" and s.params.get("top_level")]
        if len(top) < 2:
            continue
        tf = forward_kinematics(m, {})
        fwds = [tf[f"d{s.params['digit_id']}p1"][:3, 2] for s in top]
        mean_others = np.mean(fwds[:-1], axis=0)
        norm = float(np.linalg.norm(mean_others))
        if norm < 1e-9:
            continue
        mean_others = mean_others / norm
        dot = float(np.clip(np.dot(fwds[-1], mean_others), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(dot))
        angles_all.append(angle_deg)
        if len({s.params["mount"] for s in top}) >= 2:
            angles_mixed.append(angle_deg)

    def summarize(angles: List[float]) -> Dict[str, Optional[float]]:
        if not angles:
            return {"n": 0, "median_deg": None, "frac_within_90deg": None}
        return {
            "n": len(angles),
            "median_deg": float(np.median(angles)),
            "frac_within_90deg": float(np.mean([a <= 90.0 for a in angles])),
        }

    return {"overall": summarize(angles_all), "mixed_host": summarize(angles_mixed)}


def axis_coherence(dist, n: int) -> Dict[str, Any]:
    """The "shared hinge plane" claim's own operationalization (opus-review
    -g0.md item 5): share of SUCCESSIVE (consecutive phalanx-to-phalanx,
    within one digit) revolute-axis pairs whose angle is within 20 deg of
    each other. Not delivered by V3 or V3s (both measure ~8%, see this
    script's own output) -- reported here, not claimed, per the review's
    "either deliver ... or drop it from the docs"."""
    within20 = 0
    total = 0
    for seed in range(n):
        d = sample_derivation(seed, dist)
        m = derive(d)
        joints_by_name = {j.name: j for j in m.joints}
        digits: Dict[str, List[int]] = {}
        for s in d.steps:
            if s.production == "Phalanx":
                digits.setdefault(s.params["digit_id"], []).append(s.params["p"])
        for digit_id, ps in digits.items():
            ps = sorted(ps)
            for a, b in zip(ps, ps[1:]):
                ja = joints_by_name.get(f"d{digit_id}p{a + 1}_j")
                jb = joints_by_name.get(f"d{digit_id}p{b + 1}_j")
                if ja is None or jb is None or ja.type != "revolute" or jb.type != "revolute":
                    continue
                va, vb = np.array(ja.axis), np.array(jb.axis)
                dot = float(np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1.0, 1.0))
                ang = math.degrees(math.acos(abs(dot)))  # axis sign is not meaningful for a hinge
                total += 1
                if ang <= 20.0:
                    within20 += 1
    return {"n_pairs": total, "frac_within_20deg": (within20 / total) if total else None}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seeds", type=int, default=500)
    p.add_argument("--out", type=Path, default=None, help="optional path to also write JSON results")
    args = p.parse_args(argv)
    n = args.n_seeds

    out: Dict[str, Any] = {}

    print(f"=== Item 1: mount separation (4-5 digits, root 20-80mm, <=2 palm bodies, n={n}) ===")
    base = replace(G_V1, digit_count_range=(4, 5), palm_body_count_range=(0, 2),
                    palm_length_range_m=(0.020, 0.080))
    v1s = replace(base, mount_on_host_surface=True)
    v2s = replace(v1s, mount_min_separation_m=2.0 * 0.012 + 0.005)
    out["mount_separation"] = {}
    for name, dist in [("V1", base), ("V1s", v1s), ("V2s", v2s)]:
        stats = min_pairwise_sep_stats(dist, n)
        out["mount_separation"][name] = stats
        print(f"{name}: {stats}")

    print()
    print(f"=== Item 5: opposition angle, V3 (before) vs V3s (after), n={n} ===")
    out["opposition"] = {}
    for name, dist in [("V3", G_V3), ("V3s", G_V3S)]:
        dist2 = replace(dist, palm_body_count_range=(1, 2), digit_count_range=(2, 5))
        stats = opposition_diagnostics(dist2, n)
        out["opposition"][name] = stats
        print(f"{name}: {stats}")

    print()
    print("=== 'Shared hinge plane' claim check (axis coherence), V3 vs V3s ===")
    out["axis_coherence"] = {}
    for name, dist in [("V3", G_V3), ("V3s", G_V3S)]:
        stats = axis_coherence(dist, n)
        out["axis_coherence"][name] = stats
        print(f"{name}: {stats}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
