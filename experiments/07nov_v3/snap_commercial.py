"""Commercial hands rebuilt in the v3 grammar (7 Nov 2026): each vendor hand's structure snapped to what v3 can
express, built with v3 geometry -- the in-distribution proxy of a commercial hand, beside the vendor URDF itself.

Per finger (vendor URDF at q = 0, in its canonical palm frame): the base -> (y, z) relative to the palm centre on the
20 mm grid; the facing (base -> fingertip direction in the plate) and the tilt (out of the plate) on the 30 deg grid
(tilt kept only where v3 allows it); joint kinds flexion / abduction by axis (along the palm normal = abduction); link
lengths to 30 / 40 / 50 mm, a first link under KNUCKLE_SNAP_MM between a flexion and an abduction -> a co-located
knuckle (0 mm); hands with a 5-joint finger (Shadow) or mixed base types after snapping are reported and skipped,
mixed bases resolved to the majority first (kn1dof when tied). The snapped hand must pass v3.valid; its cell is printed
next to the vendor hand's.

    .venv_isaacsim/bin/python experiments/07nov_v3/snap_commercial.py [--out debug_outputs/coevolve/v3/commercial_snapped]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")
KNUCKLE_SNAP_MM = 25.0


def snap(name: str):
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from hand_sampler.grammar import hand as gh, v3
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames
    s = get_robot_spec(f"{name}_left_uniform_handonly")
    pose, child_of = _frames(s)
    kp = np.asarray(s.palm_keypoints)
    e = kp[1:] - kp[:1]
    n_ax, w_ax, l_ax = (e / np.linalg.norm(e, axis=1)[:, None])
    c = np.asarray(s.palm_center_offset)
    ax, lv = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    L = np.linalg.norm(lv, axis=1) * 1000
    fingers, notes = [], []
    for f_i, (joints, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
        if len(joints) > v3.MAX_JOINTS:
            return None, [f"finger {f_i}: {len(joints)} joints (v3 allows {v3.MAX_JOINTS})"]
        base = pose(child_of[s.hand_joint_names[joints[0]]])[:3, 3]
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[f_i]) + Mt[:3, 3]
        d = pad - base
        b = base - c
        y, z = float(b @ w_ax) * 1000, float(b @ l_ax) * 1000
        facing = math.degrees(math.atan2(d @ w_ax, d @ l_ax)) % 360
        tilt = math.degrees(math.atan2(d @ n_ax, math.hypot(d @ w_ax, d @ l_ax)))
        kinds = []
        for j in joints:
            a = pose(child_of[s.hand_joint_names[j]])[:3, :3] @ ax[j]
            kinds.append(v3.ABD if abs(a @ n_ax) > 0.7 else v3.FLEX)
        lens = [float(L[j]) for j in joints]
        snapped = []
        for k, (kd, Lk) in enumerate(zip(kinds, lens)):
            if k == 0 and len(joints) >= 2 and kinds[0] != kinds[1] and Lk < KNUCKLE_SNAP_MM:
                snapped.append(0)
            else:
                snapped.append(int(min(v3.LINKS_MM, key=lambda x: abs(x - Lk))))
        fingers.append({"y": y, "z": z, "facing": facing, "tilt": tilt, "kinds": kinds, "lens": snapped,
                        "raw_lens": [round(x) for x in lens]})
    # one base type for the hand: the majority; a finger off it is converted (knuckle <-> 1-DOF first link)
    def btype(fd):
        if fd["lens"][0] == 0:
            return "knFA" if fd["kinds"][0] == v3.FLEX else "knAF"
        return "kn1dof"
    cnt = Counter(btype(fd) for fd in fingers)
    target = max(cnt, key=lambda t: (cnt[t], t == "kn1dof"))
    if len(cnt) > 1:
        notes.append(f"mixed bases {dict(cnt)} -> all {target}")
    hand_f = []
    for fd in fingers:
        kinds, lens = list(fd["kinds"]), list(fd["lens"])
        if btype(fd) != target:
            if target == "kn1dof":
                lens[0] = int(min(v3.LINKS_MM))
            else:
                kinds[0], kinds[1] = ((v3.FLEX, v3.ABD) if target == "knFA" else (v3.ABD, v3.FLEX))
                lens[0] = 0
        joints = tuple(gh.Joint("hinge", k, Lk) for k, Lk in zip(kinds, lens))
        tilt = int(round(fd["tilt"] / 30) * 30)
        if not (target == "kn1dof" and kinds[0] == v3.ABD):
            tilt = 0
        tilt = max(0, min(v3.MAX_TILT_DEG, tilt))
        hand_f.append(gh.Finger(y=int(round(fd["y"] / v3.BASE_PITCH_MM) * v3.BASE_PITCH_MM),
                                z=int(round(fd["z"] / v3.BASE_PITCH_MM) * v3.BASE_PITCH_MM),
                                facing=int(round(fd["facing"] / 30) * 30) % 360, tilt=tilt, joints=joints))
    hand = v3.centred(gh.Hand(fingers=tuple(hand_f)))
    return hand, notes + [f"raw links {[fd['raw_lens'] for fd in fingers]}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="debug_outputs/coevolve/v3/commercial_snapped")
    a = ap.parse_args()
    from hand_sampler.grammar import v3
    from coevolution.evolution.descriptors_v3 import cell, features
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for h in HANDS:
        hand, notes = snap(h)
        vendor_cell = cell(features(f"{h}_left_uniform_handonly"))
        if hand is None:
            print(f"{h:8s} skipped: {notes}")
            continue
        viol = v3.violations(hand, v3.MUTANT_SLACK_DEG, v3.MUTANT_BASE_DISTANCE_MM)
        ok = not viol and v3.clear(hand)
        row = v3.build(hand, f"{h}_v3", out / f"{h}_v3")
        snapped_cell = cell(features(row["spec"]))
        print(f"{h:8s} vendor {vendor_cell}  ->  v3 {snapped_cell}  valid {ok} {viol[:2] if viol else ''}")
        print(f"          {' | '.join(notes)}")
        print(f"          v3 fingers {[(f.y, f.z, f.facing, f.tilt, [j.length for j in f.joints]) for f in hand.fingers]}")
        manifest.append({"id": f"{h}_v3", "spec": row["spec"], "niche": snapped_cell, "valid": ok,
                         "vendor": f"{h}_left_uniform_handonly", "vendor_cell": vendor_cell})
    (out / "manifest.json").write_text(json.dumps({"generation": 0, "grid": {"n_rigid": 6, "n_moving": 0, "n_slots": 5},
                                                   "hands": manifest}, indent=1))


if __name__ == "__main__":
    main()
