"""The commercial hands as fitted at the scale where Feetech servos build
them (2026-10-14): per hand, the scale, whether it is buildable, the fit
errors (millimetres at the real hand's size), its largest link, finger and
base distance, and the gravity torque its heaviest finger puts on its first
joint when flat, palm up (the simulator's effort limit is the Feetech's
stall torque, `hand.JOINT_EFFORT_NM` = 1.39 N m). Then the
extremes over all hands, and those x 1.1 rounded out to 10 mm: the grammar's
own limits.

    PYTHONPATH=. python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/commercial_scaled.py

Reads `hand_sampler/grammar_bench/conformed_hands.json`; writes
`commercial_scaled.json` next to this file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hand_sampler.grammar import build_conformed as BC
from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import viability as V
from hand_sampler.grammar.hand import COMMERCIAL_RULES, JOINT_EFFORT_NM, check, finger_length_mm


def flat_torque_nm(h, i: int) -> float:
    """Gravity torque about finger i's first joint, the finger at the zero
    pose and the palm up (gravity along -x)."""
    pose = dv.fk(h)
    f = h.fingers[i]
    a = pose.links[i][0][:3, :3] @ dv.joint_axis(h, i, 0)
    base = dv.mount_point_m(f)
    tq = 0.0
    for j in range(len(f.joints)):
        m, com, _ = dv.link_mass_props(h, i, j)
        T = pose.links[i][j]
        r = T[:3, :3] @ com + T[:3, 3] - base
        tq += abs(float(np.cross(r, np.array([-9.81 * m, 0.0, 0.0])) @ a))
    return tq


def main() -> dict:
    rows = {}
    for hid, r in BC.load().items():
        h = r["hand"]
        rows[hid] = {
            "scale": r["scale"], "buildable": r["buildable"], "c1_ok": V.c1_self_overlap(h).ok,
            "in_grammar": not check(h, COMMERCIAL_RULES),
            "max_joint_mm": round(r["max_joint_mm"], 1), "max_axis_deg": round(r["max_axis_deg"], 1),
            "max_tip_mm": round(r["max_tip_mm"], 1), "within_target": r["within_target"],
            "longest_link_mm": max(j.length for f in h.fingers for j in f.joints),
            "longest_finger_mm": max(finger_length_mm(f) for f in h.fingers),
            "farthest_base_mm": round(max(math.hypot(f.y, f.z) for f in h.fingers), 1),
            "max_flat_torque_nm": round(max(flat_torque_nm(h, i) for i in range(len(h.fingers))), 2),
            "note": r["notes"][-1] if r["notes"] else "",
        }
    ext = {k: max(v[k] for v in rows.values()) for k in ("longest_link_mm", "longest_finger_mm", "farthest_base_mm")}
    up = {k: int(math.ceil(v * 1.1 / 10.0) * 10) for k, v in ext.items()}
    out = {"hands": rows, "extremes": ext, "extremes_x1_1_rounded_out": up,
           "buildable": sum(v["buildable"] for v in rows.values()), "of": len(rows),
           "within_target": sum(v["within_target"] for v in rows.values()),
           "effort_nm": JOINT_EFFORT_NM,
           "over_effort": sorted(k for k, v in rows.items() if v["max_flat_torque_nm"] > JOINT_EFFORT_NM)}
    Path(__file__).with_name("commercial_scaled.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    o = main()
    print(f"{'hand':22s} {'scale':>5s} build  joint  axis   tip  link finger  base  torque")
    for hid, v in o["hands"].items():
        print(f"{hid:22s} {v['scale']:5g} {str(v['buildable']):5s} {v['max_joint_mm']:6.1f} {v['max_axis_deg']:5.1f} "
              f"{v['max_tip_mm']:5.1f} {v['longest_link_mm']:5d} {v['longest_finger_mm']:6d} {v['farthest_base_mm']:5.0f} "
              f"{v['max_flat_torque_nm']:7.2f}")
    print("extremes", o["extremes"], "x1.1", o["extremes_x1_1_rounded_out"])
    print("buildable", o["buildable"], "of", o["of"], "; within target", o["within_target"],
          f"; a finger over {o['effort_nm']} N m:", o["over_effort"])
