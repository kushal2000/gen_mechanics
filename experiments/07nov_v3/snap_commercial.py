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
VENDOR_PALM_MM: dict = {}
PARTS = False         # --parts (18 Nov): each finger built from its own vendor finger's real link geometry (v7 parts)
EXACT_AXES = False    # --exact-axes (18 Nov): with --exact, every joint's own vendor axis (not flexion / abduction at the roll)
EXACT = False         # --exact (18 Nov): no rounding -- links, bases, facings, tilts, rolls at the vendor's own values
VENDOR_TAG = ""       # --vendor-tag (23 Nov): snap <hand>_left_uniform_handonly_<tag> (e.g. canon: grammar joint signs)


def snap(name: str):
    from hand_sampler.design_space import joint_axes_and_link_vectors
    from hand_sampler.grammar import hand as gh, v3
    from isaacsimenvs.common.robots import get_robot_spec
    from isaacsimenvs.common.robots.finger_layout import finger_layout
    from coevolution.evolution.descriptors_v3 import _frames, joint_kinds
    s = get_robot_spec(f"{name}_left_uniform_handonly" + (f"_{VENDOR_TAG}" if VENDOR_TAG else ""))
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
        kl = joint_kinds(s, pose, child_of, ax, joints, d)                 # roll-invariant (7 Nov)
        if "R" in kl:
            notes.append(f"finger {f_i}: roll joint(s) {kl} -> flexion")
        kinds = ["flex" if k in ("F", "R") else "abd" for k in kl]
        if v3.VARIANT == "v7" and kl and kl[0] == "R":        # v7: a twist base joint stays a twist (18 Nov)
            kinds[0] = "roll"
        # the finger's roll: its flexion axis (the last joint's) turned about the finger, against an un-rolled
        # finger of the same facing / tilt (canonical palm frame: x normal, y width, z length)
        to_c = np.stack([n_ax, w_ax, l_ax])                              # palm-body -> canonical
        a_last = to_c @ (pose(child_of[s.hand_joint_names[joints[-1]]])[:3, :3] @ ax[joints[-1]])
        fa_s = int(round((math.degrees(math.atan2(d @ w_ax, d @ l_ax)) % 360) / 30) * 30) % 360
        t_s = int(round(math.degrees(math.atan2(d @ n_ax, math.hypot(d @ w_ax, d @ l_ax))) / 30) * 30)
        B0 = v3.finger_frame(gh.Finger(y=0, z=0, facing=fa_s, tilt=max(0, min(v3.MAX_TILT_DEG, t_s))))
        phi = math.degrees(math.atan2(float(a_last @ B0[:, 1]), float(a_last @ B0[:, 2])))
        roll = int(round(phi)) if EXACT else v3.wrap_roll(round(phi / 30) * 30)   # axis direction kept
        lens = [float(L[j]) for j in joints]
        snapped = []
        for k, (kd, Lk) in enumerate(zip(kinds, lens)):
            if EXACT:
                snapped.append(max(1, int(round(Lk))))
            elif (k == 0 and len(joints) >= 2 and kinds[0] != kinds[1] and Lk < KNUCKLE_SNAP_MM
                  and "roll" not in kinds[:2]):            # a twist base keeps a link (knuckles are flex / abd pairs)
                snapped.append(0)
            else:
                snapped.append(int(min(v3.LINKS_MM, key=lambda x: abs(x - Lk))))
        axes_exact = []
        if EXACT_AXES:                                      # each joint's axis in the (exact) finger frame -> (az, el) deg
            Bx = v3.finger_frame(gh.Finger(y=0, z=0, facing=int(round(facing)) % 360, tilt=int(round(tilt))))
            for jn in joints:
                a_c = to_c @ (pose(child_of[s.hand_joint_names[jn]])[:3, :3] @ ax[jn])
                u = Bx.T @ (a_c / np.linalg.norm(a_c))
                axes_exact.append((int(round(math.degrees(math.atan2(u[1], u[2])))),
                                   int(round(math.degrees(math.asin(max(-1.0, min(1.0, u[0]))))))))
        fingers.append({"axes_exact": axes_exact, "y": y, "z": z, "facing": facing, "tilt": tilt, "kinds": kinds, "lens": snapped,
                        "raw_lens": [round(x) for x in lens], "roll": roll, "roll_raw": round(phi)})
    # one base type for the hand: the majority; a finger off it is converted (knuckle <-> 1-DOF first link)
    def btype(fd):
        if fd["lens"][0] == 0:
            return "knFA" if fd["kinds"][0] == "flex" else "knAF"
        return "kn1dof"
    cnt = Counter(btype(fd) for fd in fingers)
    target = max(cnt, key=lambda t: (cnt[t], t == "kn1dof"))
    if len(cnt) > 1:
        notes.append(f"mixed bases {dict(cnt)} -> all {target}")
    hand_f = []
    for fi_, fd in enumerate(fingers):
        part = f"{name.split('_')[0]}/f{fi_}" if PARTS else ""
        kinds, lens = list(fd["kinds"]), list(fd["lens"])
        if btype(fd) != target and not EXACT:
            if target == "kn1dof":
                lens[0] = int(min(v3.LINKS_MM))
            else:
                kinds[0], kinds[1] = (("flex", "abd") if target == "knFA" else ("abd", "flex"))
                lens[0] = 0
        joints = tuple(gh.Joint("hinge", (fd["axes_exact"][k_] if fd["axes_exact"] else v3.axis_for(k, fd["roll"])), Lk)
                       for k_, (k, Lk) in enumerate(zip(kinds, lens)))
        if EXACT:
            hand_f.append(v3.Finger(y=int(round(fd["y"])), z=int(round(fd["z"])), facing=int(round(fd["facing"])) % 360,
                                    tilt=int(round(fd["tilt"])), joints=joints, roll=fd["roll"], part=part))
            continue
        tilt = int(round(fd["tilt"] / 30) * 30)
        if not (target == "kn1dof" and kinds[0] == "abd"):
            tilt = 0
        tilt = max(0, min(v3.MAX_TILT_DEG, tilt))
        hand_f.append(v3.Finger(y=int(round(fd["y"] / v3.BASE_PITCH_MM) * v3.BASE_PITCH_MM),
                                z=int(round(fd["z"] / v3.BASE_PITCH_MM) * v3.BASE_PITCH_MM),
                                facing=int(round(fd["facing"] / 30) * 30) % 360, tilt=tilt, joints=joints,
                                roll=fd["roll"], part=part))
    hand = v3.centred(gh.Hand(fingers=tuple(hand_f)))
    notes.append(f"vendor palm thickness {float(np.linalg.norm(e[0])) * 1000:.0f} mm")
    VENDOR_PALM_MM[name] = float(np.linalg.norm(e[0])) * 1000                # for --match-palm-thickness
    return hand, notes + [f"raw links {[fd['raw_lens'] for fd in fingers]}",
                          f"rolls (raw -> snapped) {[(fd['roll_raw'], fd['roll']) for fd in fingers]}",
                          f"kinds {[''.join(k[0].upper() for k in fd['kinds']) for fd in fingers]}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="default debug_outputs/coevolve/<grammar>/commercial_snapped")
    ap.add_argument("--link-width-mm", type=float, default=0.0,
                    help="build (only) with this link width, tip radius half of it (16 Nov test: vendor fingers are bulkier)")
    ap.add_argument("--parts", action="store_true", help="real link geometry from the v7 parts library (18 Nov)")
    ap.add_argument("--exact-axes", action="store_true", help="with --exact: the vendor's own joint axes (18 Nov)")
    ap.add_argument("--exact", action="store_true", help="no rounding to the grammar's grids (18 Nov fidelity test)")
    ap.add_argument("--vendor-tag", default="", help="snap a vendor variant, e.g. canon (23 Nov: grammar joint signs)")
    ap.add_argument("--match-palm-thickness", action="store_true",
                    help="build each replica with its vendor's palm thickness (15 Nov: vendor palms 34-49 mm, grammar 20)")
    ap.add_argument("--grammar", default="v3", choices=("v3", "v4", "v5", "v6", "v7"),
                    help="v4 (14 Nov): every joint -30..90 deg, each finger's last joint a flexion (v3.configure)")
    a = ap.parse_args()
    from hand_sampler.grammar import hand as gh, v3
    v3.configure(a.grammar)
    global EXACT, PARTS, EXACT_AXES, VENDOR_TAG
    EXACT, PARTS, EXACT_AXES, VENDOR_TAG = a.exact, a.parts, a.exact_axes, a.vendor_tag
    a.out = a.out or f"debug_outputs/coevolve/{a.grammar}/commercial_snapped"
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
        if a.match_palm_thickness:
            v3.PALM_THICKNESS_MM = round(VENDOR_PALM_MM[h])
        if a.link_width_mm:                                # since v6 a per-finger field (v3.Finger.width)
            v3.LINK_WIDTH_MM, v3.TIP_RADIUS_MM = a.link_width_mm, a.link_width_mm / 2.0
            hand = gh.Hand(fingers=tuple(v3.as_v3(f, width=int(a.link_width_mm)) for f in hand.fingers))
        row = v3.build(hand, f"{h}_{a.grammar}", out / f"{h}_{a.grammar}")
        v3.PALM_THICKNESS_MM = 20.0
        v3.LINK_WIDTH_MM, v3.TIP_RADIUS_MM = 20.0, 10.0
        snapped_cell = cell(features(row["spec"]))
        print(f"{h:8s} vendor {vendor_cell}  ->  v3 {snapped_cell}  valid {ok} {viol[:2] if viol else ''}")
        print(f"          {' | '.join(notes)}")
        print(f"          v3 fingers {[(f.y, f.z, f.facing, f.tilt, v3.roll_of(f), [j.length for j in f.joints]) for f in hand.fingers]}")
        manifest.append({"id": f"{h}_{a.grammar}", "spec": row["spec"], "niche": snapped_cell, "valid": ok,
                         "vendor": f"{h}_left_uniform_handonly", "vendor_cell": vendor_cell})
    (out / "manifest.json").write_text(json.dumps({"generation": 0, "grid": {"n_rigid": 6, "n_moving": 0, "n_slots": 5},
                                                   "hands": manifest}, indent=1))


if __name__ == "__main__":
    main()
