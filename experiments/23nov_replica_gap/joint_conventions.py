"""Replica vs vendor joint conventions (23 Nov 2026). For every finger joint of a vendor hand and of its grammar replica:
the fingertip pad's direction of motion for +q at q = 0, in the hand's canonical palm frame (n = palm normal, w = width,
l = length), and the joint limits. A replica joint whose +q moves the tip the other way from the vendor's (cosine < 0)
means the same action curls the two hands in opposite directions -- a policy cannot transfer across that.

    .venv_isaacsim/bin/python experiments/23nov_replica_gap/joint_conventions.py [--replicas debug_outputs/coevolve/v9/commercial_snapped]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HANDS = ("wuji2", "tesollo", "xhand", "leap", "dex3", "allegro", "shadow")


def joint_motion(spec_name: str):
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames
    s = get_robot_spec(spec_name)
    pose, child_of = _frames(s)
    kp = np.asarray(s.palm_keypoints)
    e = kp[1:] - kp[:1]
    to_c = np.stack(e / np.linalg.norm(e, axis=1)[:, None])            # rows: n, w, l
    ax, _ = joint_axes_and_link_vectors(s.urdf_path, s.hand_joint_names, s.fingertip_body_names, s.fingertip_offsets)
    lo = np.degrees(np.asarray(s.joint_lower, float)) if getattr(s, "joint_lower", None) is not None else None
    hi = np.degrees(np.asarray(s.joint_upper, float)) if getattr(s, "joint_upper", None) is not None else None
    out = []
    for fi, (joints, tip) in enumerate(zip(finger_layout(s).fingers, s.fingertip_body_names)):
        Mt = pose(tip)
        pad = Mt[:3, :3] @ np.asarray(s.fingertip_offsets[fi]) + Mt[:3, 3]
        rows = []
        for jn in joints:
            P = pose(child_of[s.hand_joint_names[jn]])
            a = P[:3, :3] @ ax[jn]
            v = np.cross(a, pad - P[:3, 3])
            v = to_c @ v
            nv = np.linalg.norm(v)
            rows.append({"name": s.hand_joint_names[jn], "v": v / nv if nv > 1e-9 else v,
                         "lim": (None if lo is None else (round(float(lo[jn])), round(float(hi[jn]))))})
        out.append(rows)
    return out, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", default="debug_outputs/coevolve/v9/commercial_snapped")
    ap.add_argument("--tag", default="", help="vendor variant, e.g. canon -> <hand>_left_uniform_handonly_canon")
    a = ap.parse_args()
    for h in HANDS:
        reps = sorted((REPO / a.replicas).glob(f"{h}_*/*.spec.json"))
        if not reps:
            print(f"== {h}: no replica"); continue
        V, sv = joint_motion(f"{h}_left_uniform_handonly" + (f"_{a.tag}" if a.tag else ""))
        R, sr = joint_motion(str(reps[0].relative_to(REPO)))
        print(f"== {h}  (vendor {sum(map(len, V))} joints, replica {sum(map(len, R))})")
        flips = 0
        for fi, (fv, fr) in enumerate(zip(V, R)):
            for j, (jv, jr) in enumerate(zip(fv, fr)):
                cos = float(jv["v"] @ jr["v"])
                flips += cos < 0
                tag = "  FLIPPED" if cos < 0 else ("  off" if cos < 0.7 else "")
                print(f"  f{fi} j{j} {jv['name']:28s} tip dir vendor (n,w,l) {np.round(jv['v'], 2)} replica {np.round(jr['v'], 2)}"
                      f"  cos {cos:+.2f}  limits vendor {jv['lim']} replica {jr['lim']}{tag}")
        print(f"  -> {flips} joints with opposite +q direction")


if __name__ == "__main__":
    main()
