"""An edited COPY of the vendor Wuji v2 (URDF + spec JSON) moved toward the training distribution one factor at a time,
written under debug_outputs/coevolve/studies/wuji_morph/ (NOT the assets tree, whose *_left_*.spec.json files every
training auto-registers). The original is untouched. Factors, applied in this order:

  --scale s        geometry x s, mass x s^3, inertia x s^5 (scale_vendor_hand.scale_urdf)
  --effort e       every finger joint's effort limit (default 0.5 N m, the training hands' and the vendor's own)
  --limits training   flexion joints -30..90 deg, abduction (*_abd) -30..30 deg, the grammar's ranges (URDF + spec)
  --facings a b c d   the index / middle / ring / pinky chains each rotated rigidly about the palm normal through the
                   palm centre so the finger (root joint -> fingertip at q = 0) points at that facing (deg; the
                   token frame's palm plane, facing 0 = along the palm's length, as grammar designs); the thumb stays
  --spread-frac f  only that fraction of each finger's rotation (a halfway hand)
  --recentre       move the palm link's frame (where the wrist attaches) to the palm centre, as every grammar hand's
                   root frame is (urdf_export: derived mount): the vendor root sits 74 mm (x1) from the palm centre, so
                   the pen spawned beside the palm, not under it (7 Nov)
  --lock-abd       the four fingers' MCP abduction joints made fixed (at 0) and dropped from the spec: one abduction
                   finger left (the thumb's CMC), as most training hands; 7 of 4096 gen-7 hands had 3+ (7 Nov)

    python experiments/06nov_coevolve_v2/morph_vendor_hand.py --tag x2_star --scale 2 --facings 21 93 165 237
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
SRC = REPO / "assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.spec.json"
OUT = REPO / "debug_outputs/coevolve/studies/wuji_morph"
FINGERS = ["index_finger", "middle_finger", "ring_finger", "pinky"]


def _T(o) -> np.ndarray:
    T = np.eye(4)
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def palm_frame(spec: dict, s: float):
    """(centre, w, l, n) in palm-body coords, as the tokens' canonical palm frame (normal, width, length)."""
    from isaacsimenvs.common.robots import get_robot_spec
    rs = get_robot_spec(str(SRC.relative_to(REPO)))
    kp = np.asarray(rs.palm_keypoints) * s
    e = kp[1:] - kp[:1]
    n, w, l = (e / np.linalg.norm(e, axis=1)[:, None])
    return np.asarray(rs.palm_center_offset) * s, w, l, n, rs


def facing(d, w, l) -> float:
    return math.degrees(math.atan2(float(d @ w), float(d @ l))) % 360


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--effort", type=float, default=0.5)
    ap.add_argument("--limits", choices=("vendor", "training"), default="vendor")
    ap.add_argument("--facings", type=float, nargs=4, default=None)
    ap.add_argument("--spread-frac", type=float, default=1.0)
    ap.add_argument("--lock-abd", action="store_true")
    ap.add_argument("--recentre", action="store_true")
    args = ap.parse_args()
    from scale_vendor_hand import scale_urdf
    s = args.scale
    spec = json.loads(SRC.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    src_urdf = REPO / spec["urdf"]
    dst_urdf = OUT / f"wuji2_left_{args.tag}.urdf"
    scale_urdf(src_urdf, dst_urdf, s, args.effort)
    # meshes: the vendor URDF's mesh filenames are relative to its own directory -> make them absolute
    tree = ET.parse(dst_urdf)
    root = tree.getroot()
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if fn and not fn.startswith(("/", "package://", "file://")):
            m.set("filename", str((src_urdf.parent / fn).resolve()))
    out = dict(spec)
    out["urdf"] = str(dst_urdf.relative_to(REPO))
    for k in ("palm_centre_in_palm_body", "palm_centre_ours", "palm_extents_ours"):
        if k in out:
            out[k] = [float(v) * s for v in out[k]]

    def sc(v):
        if isinstance(v, dict):
            return {kk: sc(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [sc(x) for x in v]
        return float(v) * s
    for k in ("fingertip_offsets", "fingertip_pos_home_ours", "overlap_at_home_mm"):
        if k in out:
            out[k] = sc(out[k])
    out["effort_limit"] = args.effort
    notes = [f"x{s:g}", f"effort {args.effort:g}"]

    joints = {j.get("name"): j for j in root.findall("joint")}
    if args.limits == "training":
        lim = {}
        for name in spec["hand_joint_names"]:
            lo, hi = (-30.0, 30.0) if name.endswith("_abd") else (-30.0, 90.0)
            lim[name] = [math.radians(lo), math.radians(hi)]
            L = joints[name].find("limit")
            L.set("lower", f"{lim[name][0]:.6f}")
            L.set("upper", f"{lim[name][1]:.6f}")
        out["joint_limits"] = lim
        notes.append("training limits")

    if args.facings is not None:
        c, w, l, n, rs = palm_frame(spec, s)
        palm = spec["palm_body"]
        parent = {j.find("child").get("link"): j for j in root.findall("joint")}

        def pose(link):
            M = np.eye(4)
            while link != palm:
                j = parent[link]
                M = _T(j.find("origin")) @ M
                link = j.find("parent").get("link")
            return M
        tips = dict(zip(spec["fingertip_body_names"], spec["fingertip_offsets"]))
        done = []
        for fname, target in zip(FINGERS, args.facings):
            rootj = next(j for j in root.findall("joint") if j.find("parent").get("link") == palm
                         and fname in j.get("name"))
            tip = next(t for t in tips if fname in t)
            Mt = pose(tip)
            pad = Mt[:3, :3] @ (np.asarray(tips[tip]) * s) + Mt[:3, 3]
            T0 = _T(rootj.find("origin"))
            d = pad - T0[:3, 3]
            d = d - (d @ n) * n
            f0 = facing(d, w, l)
            delta = ((target - f0) % 360) * args.spread_frac  # always away from the thumb (increasing facing)
            R = None
            for sgn in (1, -1):                         # whichever sense of the rotation about n turns it to target
                Rc = Rotation.from_rotvec(sgn * math.radians(delta) * n).as_matrix()
                if abs(((facing(Rc @ d, w, l) - (f0 + delta)) + 180) % 360 - 180) < 1e-3:
                    R = Rc
                    break
            assert R is not None
            T1 = np.eye(4)
            T1[:3, :3] = R @ T0[:3, :3]
            T1[:3, 3] = c + R @ (T0[:3, 3] - c)
            o = rootj.find("origin")
            o.set("xyz", " ".join(f"{v:.9g}" for v in T1[:3, 3]))
            o.set("rpy", " ".join(f"{v:.9g}" for v in Rotation.from_matrix(T1[:3, :3]).as_euler("xyz")))
            done.append(f"{fname} {f0:.0f}->{(f0 + delta) % 360:.0f}")
        notes.append("facings " + ", ".join(done))
        for k in ("fingertip_pos_home_ours", "overlap_at_home_mm"):
            out.pop(k, None)                            # stale after the rotation (informational only)
    if args.lock_abd:
        # 7 Nov: MERGE each locked joint's child link into its parent (not just type="fixed"): with the stub link
        # left in the chain the MCP flexion token's link_vec read 0 mm (it stops at the fixed joint) while the real
        # MCP -> PIP distance is ~100 mm, so the policy saw a base joint that moves nothing.
        locked = [n for n in spec["hand_joint_names"] if n.endswith("_mcp_abd") and "thumb" not in n]
        links = {l.get("name"): l for l in root.findall("link")}
        merged = {}
        for name in locked:
            j = joints[name]
            P, A = j.find("parent").get("link"), j.find("child").get("link")
            O2 = _T(j.find("origin"))
            lp, la = links[P], links[A]
            def put(el, T):
                o = el.find("origin")
                if o is None:
                    o = ET.SubElement(el, "origin")
                o.set("xyz", " ".join(f"{v:.9g}" for v in T[:3, 3]))
                o.set("rpy", " ".join(f"{v:.9g}" for v in Rotation.from_matrix(T[:3, :3]).as_euler("xyz")))
            for tag in ("visual", "collision"):
                for el in la.findall(tag):
                    put(el, O2 @ _T(el.find("origin")))
                    la.remove(el)
                    lp.append(el)
            def inert(el, T0):
                if el is None:
                    return 0.0, np.zeros(3), np.zeros((3, 3))
                T = T0 @ _T(el.find("origin"))
                m = float(el.find("mass").get("value"))
                I = el.find("inertia")
                g = {k: float(I.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                Il = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
                return m, T[:3, 3], T[:3, :3] @ Il @ T[:3, :3].T
            mp, cp, Ip = inert(lp.find("inertial"), np.eye(4))
            ma, ca, Ia = inert(la.find("inertial"), O2)
            M = mp + ma
            c = (mp * cp + ma * ca) / M
            def shift(I, m, d):
                return I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
            Ic = shift(Ip, mp, cp - c) + shift(Ia, ma, ca - c)
            for el in (lp.find("inertial"), la.find("inertial")):
                if el is not None:
                    (lp if el in list(lp) else la).remove(el)
            ie = ET.SubElement(lp, "inertial")
            ET.SubElement(ie, "origin", xyz=" ".join(f"{v:.9g}" for v in c), rpy="0 0 0")
            ET.SubElement(ie, "mass", value=f"{M:.9g}")
            ET.SubElement(ie, "inertia", ixx=f"{Ic[0,0]:.9g}", ixy=f"{Ic[0,1]:.9g}", ixz=f"{Ic[0,2]:.9g}",
                          iyy=f"{Ic[1,1]:.9g}", iyz=f"{Ic[1,2]:.9g}", izz=f"{Ic[2,2]:.9g}")
            for jj in root.findall("joint"):
                if jj.find("parent").get("link") == A:
                    jj.find("parent").set("link", P)
                    o = jj.find("origin")
                    if o is None:
                        o = ET.SubElement(jj, "origin")
                    put(jj, O2 @ _T(o))
            root.remove(j)
            root.remove(la)
            merged[A] = P
        adj = {}
        for k, v in out.get("adjacent_links", {}).items():
            k2 = merged.get(k, k)
            adj.setdefault(k2, set()).update(merged.get(x, x) for x in v)
        out["adjacent_links"] = {k: sorted(x for x in v if x != k) for k, v in adj.items()}
        out["fingertip_body_names"] = [merged.get(b, b) for b in out["fingertip_body_names"]]
        out["hand_joint_names"] = [n for n in out["hand_joint_names"] if n not in locked]
        for k in ("joint_limits", "hand_default_joint_pos"):
            out[k] = {n: v for n, v in out[k].items() if n not in locked}
        notes.append(f"locked {len(locked)} finger abductions (links merged)")
    if args.recentre:
        c = np.asarray(out["palm_centre_in_palm_body"], float)
        palm = spec["palm_body"]
        pl = next(l for l in root.findall("link") if l.get("name") == palm)
        for el in list(pl.findall("visual")) + list(pl.findall("collision")) + list(pl.findall("inertial")):
            o = el.find("origin")
            if o is None:
                o = ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
            o.set("xyz", " ".join(f"{v:.9g}" for v in np.asarray([float(x) for x in o.get("xyz", "0 0 0").split()]) - c))
        for j in root.findall("joint"):
            if j.find("parent").get("link") == palm:
                o = j.find("origin")
                o.set("xyz", " ".join(f"{v:.9g}" for v in np.asarray([float(x) for x in o.get("xyz", "0 0 0").split()]) - c))
        out["palm_centre_in_palm_body"] = [0.0, 0.0, 0.0]
        out["palm_centre_ours"] = [0.0, 0.0, 0.0]
        notes.append(f"recentred (palm frame moved {1000 * np.linalg.norm(c):.0f} mm to the palm centre)")
    tree.write(dst_urdf, xml_declaration=True, encoding="utf-8")
    out["notes_morph"] = "morph_vendor_hand.py copy of wuji2_left: " + "; ".join(notes)
    dst_spec = OUT / f"wuji2_left_{args.tag}.spec.json"
    dst_spec.write_text(json.dumps(out, indent=1))
    print(f"[morph] {dst_spec.relative_to(REPO)}: {out['notes_morph']}")


if __name__ == "__main__":
    main()
