"""A uniformly scaled COPY of a vendor hand (URDF + its onboarding spec JSON), written next to the original, which
is left untouched. Geometry x s: every <origin xyz>, mesh scale, collision primitive and inertial origin; mass x s^3
and inertia x s^5 (same density); joint limits unchanged (angles). The actuator effort is set to --effort (default:
the grammar hands' Feetech 1.39 N m -- at x2 the gravity torques grow ~16x, beyond the uniform 0.5 N m). The spec
JSON's lengths (palm centre, palm extents, fingertip offsets and home positions) are scaled to match.

    python experiments/06nov_coevolve_v2/scale_vendor_hand.py --spec assets/urdf/unified_dynamics_commercial_hands/wuji2/wuji2_left.spec.json --scale 2.0
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _xyz(s: str, k: float) -> str:
    return " ".join(f"{float(v) * k:.9g}" for v in s.split())


def scale_urdf(src: Path, dst: Path, s: float, effort: float) -> None:
    tree = ET.parse(src)
    root = tree.getroot()
    for o in root.iter("origin"):
        if o.get("xyz"):
            o.set("xyz", _xyz(o.get("xyz"), s))
    for m in root.iter("mesh"):
        m.set("scale", _xyz(m.get("scale", "1 1 1"), s))
    for b in root.iter("box"):
        b.set("size", _xyz(b.get("size"), s))
    for c in root.iter("cylinder"):
        c.set("radius", f"{float(c.get('radius')) * s:.9g}")
        c.set("length", f"{float(c.get('length')) * s:.9g}")
    for sp in root.iter("sphere"):
        sp.set("radius", f"{float(sp.get('radius')) * s:.9g}")
    for inert in root.iter("inertial"):
        m = inert.find("mass")
        if m is not None:
            m.set("value", f"{float(m.get('value')) * s ** 3:.9g}")
        I = inert.find("inertia")
        if I is not None:
            for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                if I.get(k) is not None:
                    I.set(k, f"{float(I.get(k)) * s ** 5:.9g}")
    for j in root.iter("joint"):
        lim = j.find("limit")
        if lim is not None and j.get("type") in ("revolute", "prismatic", "continuous"):
            lim.set("effort", f"{effort:g}")
    tree.write(dst, xml_declaration=True, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--effort", type=float, default=1.39)
    args = ap.parse_args()
    s = args.scale
    sp = REPO / args.spec
    d = json.loads(sp.read_text())
    src = (sp.parent / Path(d["urdf"]).name) if not (REPO / d["urdf"]).exists() else REPO / d["urdf"]
    tag = f"x{s:g}".replace(".", "p")
    dst_urdf = src.with_name(f"{src.stem}_{tag}.urdf")
    dst_spec = sp.with_name(f"{src.stem}_{tag}.spec.json")
    scale_urdf(src, dst_urdf, s, args.effort)
    out = dict(d)
    out["urdf"] = str(dst_urdf.relative_to(REPO)) if dst_urdf.is_relative_to(REPO) else str(dst_urdf)
    for k in ("palm_centre_in_palm_body", "palm_centre_ours", "palm_extents_ours"):
        if k in out:
            out[k] = [float(v) * s for v in out[k]]
    def sc(v):                                         # lists, dicts of lists, or numbers: every length x s
        if isinstance(v, dict):
            return {kk: sc(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [sc(x) for x in v]
        return float(v) * s
    for k in ("fingertip_offsets", "fingertip_pos_home_ours", "overlap_at_home_mm"):
        if k in out:
            out[k] = sc(out[k])
    out["effort_limit"] = args.effort
    out["notes_scaled"] = (f"uniform x{s:g} copy of {sp.name} (scale_vendor_hand.py): geometry x{s:g}, mass x{s ** 3:g}, "
                           f"inertia x{s ** 5:g}, effort {args.effort:g} N m; the original is unchanged")
    dst_spec.write_text(json.dumps(out, indent=1))
    print(f"[scale] wrote {dst_urdf.relative_to(REPO)} and {dst_spec.relative_to(REPO)}")


if __name__ == "__main__":
    main()
