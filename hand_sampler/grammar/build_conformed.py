"""Conform every commercial reference hand onto the grammar and write
`hand_sampler/grammar_bench/conformed_hands.json` (the grammar hands, their
scale, palm frames, zero-pose differences and fit errors) plus the fidelity
table. Each hand is scaled up by the smallest factor (to 0.05) at which its
fit is buildable from Feetech servos and keeps every joint's kind: servo
link lengths and no collisions (`conform.conform_buildable`). Errors are in
millimetres at the real hand's size (a 20 mm error at 2 x counts 10 mm), so
fits at different scales compare.

    python3 -m hand_sampler.grammar.build_conformed [--ids a,b] [--out PATH]

Hands whose URDF is not on this machine are listed as unavailable. The file
is data with provenance (git sha, the manifest's sha256 per hand); the viewer
and the simulator's population builder read it instead of re-fitting
(one process per hand; about as long as the slowest hand).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import commercial as C
from . import conform as CF
from . import derive as dv
from .hand import COMMERCIAL_RULES, GRAMMAR_VERSION, Hand, check, hand_from_dict, hand_to_dict

OUT_PATH = C.BENCH_DIR / "conformed_hands.json"
SCHEMA = "conformed_hands/3"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(C.REPO_ROOT), capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def fit_record(hand_id: str, fit: CF.HandFit, real: C.RealHand) -> dict:
    R, o = fit.palm_T[:3, :3], fit.palm_T[:3, 3]
    root_in_palm = R.T @ (np.zeros(3) - o)
    return {
        "id": hand_id,
        "scale": round(float(fit.scale), 6),
        "buildable": bool(CF.buildable(fit)),
        "hand": hand_to_dict(fit.hand),
        "palm_T": np.round(fit.palm_T, 9).tolist(),
        "q_off": np.round(fit.q_off(), 9).tolist(),
        "name_map": fit.name_map,
        "thumbs": [i for i, f in enumerate(real.fingers) if f.thumb],
        "max_joint_mm": round(fit.max_joint_mm, 3),
        "max_axis_deg": round(fit.max_axis_deg, 3),
        "max_tip_mm": round(fit.max_tip_mm, 3),
        "within_target": bool(fit.within_target),
        "fingers": [{"joint_mm": [round(e, 3) for e in f.joint_mm], "axis_deg": [round(e, 3) for e in f.axis_deg],
                     "tip_mm": round(f.tip_mm, 3)} for f in fit.fingers],
        "palm_joint_mm": [round(e, 3) for e in fit.palm_joint_mm],
        "palm_axis_deg": [round(e, 3) for e in fit.palm_axis_deg],
        "section": list(fit.section),
        "dropped_palm_joints": list(fit.dropped_palm_joints),
        "root_out_of_plate_mm": round(float(root_in_palm[0]) * 1e3, 2),
        "notes": fit.notes,
        "violations": check(fit.hand, COMMERCIAL_RULES),
    }


def _fit_one(hid: str) -> Tuple[dict, float]:
    t = time.time()
    real = C.load_real_hand(hid)
    fit = CF.conform_buildable(real)
    return fit_record(hid, fit, real), time.time() - t


def build(ids: Optional[List[str]] = None, verbose: bool = True) -> dict:
    """Every hand fitted, one process per hand (each is independent)."""
    manifest = C.load_manifest()
    ids = ids or C.reference_ids()
    hands, unavailable, todo = [], {}, []
    for hid in ids:
        if C.resolve_path(C.entry(hid), manifest) is None:
            unavailable[hid] = f"local-only:{hid} source not on this machine"
        else:
            todo.append(hid)
    done: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, min(len(todo), os.cpu_count() or 1))) as ex:
        futures = {ex.submit(_fit_one, hid): hid for hid in todo}
        for fut in as_completed(futures):
            hid = futures[fut]
            rec, dt = fut.result()
            rec["source_sha256"] = C.entry(hid).get("sha256")
            done[hid] = rec
            if verbose:
                print(f"{hid:24s} x{rec['scale']:<5g} joint {rec['max_joint_mm']:6.2f} mm  axis {rec['max_axis_deg']:5.1f} deg  "
                      f"tip {rec['max_tip_mm']:6.2f} mm  {'ok' if rec['within_target'] else 'MISS'}  ({dt:.0f} s)",
                      flush=True)
    hands = [done[hid] for hid in todo]
    return {"schema": SCHEMA, "grammar": GRAMMAR_VERSION, "git_sha": _git_sha(), "target_mm": CF.TARGET_MM,
            "target_deg": CF.TARGET_DEG, "hands": hands, "unavailable": unavailable}


def load(path: Path = OUT_PATH) -> Dict[str, dict]:
    """`{hand_id: record}` from the committed file; each record's `hand` is a
    `Hand`."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA or doc.get("grammar") != GRAMMAR_VERSION:
        raise ValueError(f"{path}: written for {doc.get('schema')}/{doc.get('grammar')}; rebuild it with "
                         f"python3 -m hand_sampler.grammar.build_conformed")
    out = {}
    for rec in doc["hands"]:
        rec = dict(rec)
        rec["hand"] = hand_from_dict(rec["hand"])
        rec["palm_T"] = np.asarray(rec["palm_T"], dtype=float)
        rec["q_off"] = np.asarray(rec["q_off"], dtype=float)
        out[rec["id"]] = rec
    return out


def fidelity_table(doc: dict) -> str:
    t_mm, t_deg = doc.get("target_mm", CF.TARGET_MM), doc.get("target_deg", CF.TARGET_DEG)
    rows = [f"| hand | scale | fingers / joints | joint (mm) | axis (deg) | fingertip (mm) | palm cut | "
            f"within {t_mm:g} mm / {t_deg:g} deg |",
            "|---|---|---|---|---|---|---|---|"]
    for rec in doc["hands"]:
        h = hand_from_dict(rec["hand"])
        nj = sum(len(f.joints) for f in h.fingers) + len(h.palm_joints)
        cut = "-"
        if h.palm_joints:
            cut = f"fingers {rec['section']}, axis {rec['palm_axis_deg'][0]:.0f} deg"
            if rec["dropped_palm_joints"]:
                cut += f" ({len(rec['dropped_palm_joints'])} dropped)"
        scale = f"{rec['scale']:g}" + ("" if rec.get("buildable", True) else " (not buildable)")
        rows.append(f"| {rec['id']} | {scale} | {len(h.fingers)} / {nj} | {rec['max_joint_mm']:.1f} | {rec['max_axis_deg']:.1f} | "
                    f"{rec['max_tip_mm']:.1f} | {cut} | {'yes' if rec['within_target'] else '**no**'} |")
    return "\n".join(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ids", default="")
    ap.add_argument("--out", default=str(OUT_PATH))
    a = ap.parse_args(argv)
    doc = build([s for s in a.ids.split(",") if s] or None)
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(fidelity_table(doc))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
