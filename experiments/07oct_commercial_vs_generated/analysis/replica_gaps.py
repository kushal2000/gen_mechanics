"""How far each Wuji v2 replica (fidelity level) is from the vendor hand, in what the policy sees.

For every capture results/spawn/wuji2_replica[_<level>].json against wuji2_commercial.json:
  * link boxes at q = 0: mean distance between paired corners of joint_link_bbox (palm frame, mm) -- the hand's
    shape as the policy reads it;
  * fingertip pads at q = 0 (mm);
  * the cube's spawn centre in the palm frame (mm);
  * at spawn, mean |difference| of each per-joint and global observation term, paired through the joint map.

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/analysis/replica_gaps.py
"""
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import viser_spawn as v  # noqa: E402

TIPS_A = ["l_index_finger_distal", "l_middle_finger_distal", "l_ring_finger_distal", "l_pinky_distal",
          "l_thumb_distal"]
TERMS = ["joint_pos", "joint_link_bbox", "joint_lower", "joint_upper", "keypoints_rel_ee", "palm_extents"]


def pad_local(h):
    h.q = dict(h.spawn_q)
    T0 = h.body_T()
    out = {}
    for b, p, ok in zip(h.d["fingertip_bodies"], h.d["fingertip_pads"], h.d["fingertip_valid"]):
        if ok:
            Ti = np.linalg.inv(T0[b])
            out[b] = Ti[:3, :3] @ np.array(p) + Ti[:3, 3]
    return out


def at_zero(h, local):
    h.q = {j: 0.0 for j in h.q}
    T = h.body_T()
    terms = h.terms()
    tips = {b: ((T[b][:3, :3] @ l + T[b][:3, 3]) - h.c) @ h.R_canon * 1000 for b, l in local.items()}
    h.q = dict(h.spawn_q)
    return terms, tips


def main():
    A = v.Hand("wuji2_commercial")
    la = pad_local(A)
    TA0, tipsA = at_zero(A, la)
    TAs = A.terms()
    cubeA = (np.array(A.d["object"][:3]) - A.c) @ A.R_canon * 1000
    names = ["wuji2_replica"] + [f"wuji2_replica_{lev}" for lev in
                                 ("limits", "axes", "zero", "palm", "links", "mounts", "clone")]
    hdr = f"{'replica':24s} {'box q=0':>8s} {'tips q=0':>9s} {'cube':>6s} | " + " ".join(f"{t[:12]:>12s}" for t in TERMS)
    print("mm except the observation terms (mean |difference| at spawn, observation units)\n" + hdr)
    for n in names:
        if not (v.SPAWN / f"{n}.json").exists():
            print(f"{n:24s} (no capture)")
            continue
        B = v.Hand(n)
        lm = v.label_map(A.name, B.name)
        TB0, tipsB = at_zero(B, pad_local(B))
        p = v.pair(TA0["joint_link_bbox"], TB0["joint_link_bbox"], lm)
        ca = np.array([x for _, x, _, _ in p]).reshape(-1, 3)
        cb = np.array([y for _, _, _, y in p]).reshape(-1, 3)
        box = float(np.linalg.norm(cb - ca, axis=1).mean()) * 1000
        tip = float(np.mean([np.linalg.norm(tipsB[f"f{i}_link4"] - tipsA[TIPS_A[i]]) for i in range(5)]))
        cubeB = (np.array(B.d["object"][:3]) - B.c) @ B.R_canon * 1000
        TBs = B.terms()
        cols = []
        for t in TERMS:
            q = v.pair(TAs[t], TBs[t], lm)
            cols.append(f"{np.mean([abs(y - x) for _, x, _, y in q]):12.3f}" if q else f"{'-':>12s}")
        print(f"{n:24s} {box:8.1f} {tip:9.1f} {np.linalg.norm(cubeB - cubeA):6.1f} | " + " ".join(cols))


if __name__ == "__main__":
    main()
