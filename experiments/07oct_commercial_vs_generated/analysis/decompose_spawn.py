"""Why does the cube spawn somewhere different relative to the palm for a vendor hand and its replica?

reset.py's object_in_hand placement (in_hand_placement: fingertips), in the canonical palm frame (origin the palm
centre; x the grasp normal, y width, z length):

    in plane:  f * (fingertip-pad centroid), f = in_hand_fingertip_fraction (0.5)
    height:    max(highest pad along x, half palm thickness) + cube half-reach + clearance + drop margin

The cube half-reach depends only on the cube's orientation, which is the same draw for every hand, so the gap
between two hands is entirely the pad centroid (in plane) and the max(...) term (height). The pad centroid splits
further into WHERE THE FINGERS START relative to the palm centre (finger-base centroid: palm outline / centre
definition) and HOW FAR THEY REACH from their bases (finger geometry at the reset pose).

    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/analysis/decompose_spawn.py [wuji2 leap allegro]
"""
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = pathlib.Path(__file__).resolve().parent.parent
SPAWN = HERE / "results/spawn"
REPLICAS = HERE.parents[1] / "assets/urdf/unified_dynamics_commercial_hands_replicas"
STEM = {"wuji2": "wuji2_left", "leap": "leap_left", "allegro": "allegro_left"}


def frame(d: dict):
    """World (env-local) -> canonical palm frame."""
    w, x, y, z = d["palm_frame_canonical_wxyz"]
    Rm = Rot.from_quat([x, y, z, w]).as_matrix()
    c = np.array(d["palm_center"])
    return lambda p: (np.asarray(p, float) - c) @ Rm


def measure(d: dict, first_joints: list[str]) -> dict:
    to = frame(d)
    pads = np.array([to(p) for p, ok in zip(d["fingertip_pads"], d["fingertip_valid"]) if ok])
    child = dict(zip(d["hand_joint_names"], d["joint_link_bodies"]))
    bases = np.array([to(d["bodies"][child[j]][:3]) for j in first_joints])
    cube = to(d["object"][:3])
    kp = np.array(d["palm_keypoints_local"])
    half_t = 0.5 * float(np.linalg.norm(kp[1] - kp[0]))
    r = d["reset"]
    lift_extra = r["in_hand_clearance"] + r["in_hand_drop_margin"]
    surface = max(float(pads[:, 0].max()), half_t)
    half_reach = cube[0] - surface - lift_extra          # the orientation-only term, inferred from the result
    return {"cube": cube, "pads": pads, "pad_centroid": pads.mean(0), "base_centroid": bases.mean(0),
            "highest_pad": float(pads[:, 0].max()), "half_t": half_t, "surface": surface,
            "half_reach": half_reach, "f": r["in_hand_fingertip_fraction"]}


def main(hands):
    print("canonical palm frame, mm: x height along the grasp normal, y width, z length; B - A = replica - vendor\n")
    for h in hands:
        a = json.loads((SPAWN / f"{h}_commercial.json").read_text())
        b = json.loads((SPAWN / f"{h}_replica.json").read_text())
        jmap = json.loads((REPLICAS / h / f"{STEM[h]}_replica.joint_map.json").read_text())
        firsts_a = [k for k, v in jmap.items() if v.endswith("_j0")]
        firsts_b = [jmap[k] for k in firsts_a]
        A, B = measure(a, firsts_a), measure(b, firsts_b)
        mm = lambda v: np.round(np.asarray(v) * 1000, 1)
        f = A["f"]
        print(f"== {h}   cube centre A {mm(A['cube'])}  B {mm(B['cube'])}  gap {mm(B['cube'] - A['cube'])}")
        # in plane (y, z): f * pad centroid = f * (pad - base) + f * base
        reach = f * ((B["pad_centroid"] - B["base_centroid"]) - (A["pad_centroid"] - A["base_centroid"]))[1:]
        start = f * (B["base_centroid"] - A["base_centroid"])[1:]
        pred = f * (B["pad_centroid"] - A["pad_centroid"])[1:]
        print(f"   in plane (y, z) gap {mm((B['cube'] - A['cube'])[1:])}  predicted {mm(pred)}"
              f"  = finger reach {mm(reach)} + where fingers start rel. palm centre {mm(start)}")
        print(f"   height: surface A {mm(A['surface'])} ({'pad' if A['highest_pad'] > A['half_t'] else 'palm face'};"
              f" highest pad {mm(A['highest_pad'])}, half thickness {mm(A['half_t'])})"
              f"  B {mm(B['surface'])} ({'pad' if B['highest_pad'] > B['half_t'] else 'palm face'};"
              f" highest pad {mm(B['highest_pad'])}, half thickness {mm(B['half_t'])})")
        print(f"   height gap {mm(B['cube'][0] - A['cube'][0])} = surface gap {mm(B['surface'] - A['surface'])}"
              f" + cube half-reach gap {mm(B['half_reach'] - A['half_reach'])} (should be 0: same orientation)\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["wuji2", "leap", "allegro"])
