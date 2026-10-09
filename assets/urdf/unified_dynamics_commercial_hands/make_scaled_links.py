"""Scaled-link variants of the uniform-dynamics Wuji v2: every FINGER link's length and/or width scaled, palm untouched.

Wuji v2's finger links all run along their own -z axis (each joint inside a finger sits at (0, ~0, -L) in its
parent link's frame; meshes and inertials are in the link frame), so a link is scaled exactly by the affine map
diag(width, width, length) in its own frame:

  * meshes (visual and collision): URDF <mesh scale="w w l">
  * joints INSIDE a finger (parent is not the palm), fixed tip frames included: origin xyz scaled the same way;
    rotations unchanged. Joints whose parent is the palm keep their origin: the palm and where each finger is
    mounted on it do not change.
  * inertials: the same affine map applied to the link's mass distribution, at the link's density -- mass x w^2 l,
    centre of mass scaled, inertia from the scaled second moments (Sigma' = m' / m * S Sigma S, S = diag(w, w, l)),
    exact for the solid the scaled mesh describes.
  * spec JSON: urdf path, fingertip pad offsets (distal-link frame) scaled; joint limits, dynamics, palm frame, palm
    centre and self-collision filters unchanged (the link boxes the policy observes are recomputed from the URDF).

Written next to the full hand as <hand>_<side>_<tag>.{urdf,spec.json}; registered like make_missing_fingers.py's
variants as <hand>_<side>_uniform_handonly_<tag> (scene_utils/robots/unified_dynamics_hands.py). Also reports any
pair of links that overlap at the default pose and is NOT collision-filtered (wider links can touch neighbours).

    python3 assets/urdf/unified_dynamics_commercial_hands/make_scaled_links.py --length 0.5 --tag len50
    python3 assets/urdf/unified_dynamics_commercial_hands/make_scaled_links.py --width 1.2 --tag wid120
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _vec(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split()])


def _fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def scale_urdf(root: ET.Element, palm: str, w: float, l: float) -> None:
    S = np.array([w, w, l])
    for j in root.findall("joint"):
        if j.find("parent").get("link") == palm:
            continue
        o = j.find("origin")
        if o is not None and o.get("xyz"):
            o.set("xyz", _fmt(_vec(o.get("xyz")) * S))
    for link in root.findall("link"):
        if link.get("name") == palm:
            continue
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                o = el.find("origin")
                if o is not None and o.get("xyz"):
                    o.set("xyz", _fmt(_vec(o.get("xyz")) * S))
                    if any(abs(float(a)) > 1e-9 for a in (o.get("rpy") or "0 0 0").split()):
                        raise ValueError(f"{link.get('name')}: rotated {tag} origin; the axis-aligned scale is not exact")
                for mesh in el.iter("mesh"):
                    old = _vec(mesh.get("scale")) if mesh.get("scale") else np.ones(3)
                    mesh.set("scale", _fmt(old * S))
        inertial = link.find("inertial")
        if inertial is None:
            continue
        o, m_el, I_el = inertial.find("origin"), inertial.find("mass"), inertial.find("inertia")
        if o is not None and any(abs(float(a)) > 1e-9 for a in (o.get("rpy") or "0 0 0").split()):
            raise ValueError(f"{link.get('name')}: rotated inertial frame")
        if o is not None:
            o.set("xyz", _fmt(_vec(o.get("xyz") or "0 0 0") * S))
        m = float(m_el.get("value"))
        g = {k: float(I_el.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
        I = np.array([[g["ixx"], g["ixy"], g["ixz"]], [g["ixy"], g["iyy"], g["iyz"]], [g["ixz"], g["iyz"], g["izz"]]])
        sigma = 0.5 * np.trace(I) * np.eye(3) - I                    # second moments about the COM
        m2 = m * w * w * l
        sigma2 = (m2 / m) * np.diag(S) @ sigma @ np.diag(S)
        I2 = np.trace(sigma2) * np.eye(3) - sigma2
        m_el.set("value", f"{m2:.9g}")
        for k, (a, b) in {"ixx": (0, 0), "iyy": (1, 1), "izz": (2, 2), "ixy": (0, 1), "ixz": (0, 2),
                          "iyz": (1, 2)}.items():
            I_el.set(k, f"{I2[a, b]:.9g}")


def _rpy(r) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r[0]), np.sin(r[0]), np.cos(r[1]), np.sin(r[1]), np.cos(r[2]), np.sin(r[2])
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rest_overlaps(urdf: Path, spec: dict, q: dict) -> list[tuple[str, str, float]]:
    """Link pairs that intersect at joint angles q and are not filtered (spec adjacent_links)."""
    import trimesh
    root = ET.parse(urdf).getroot()
    pose = {spec["palm_body"]: np.eye(4)}
    joints = root.findall("joint")
    while len(pose) < len(root.findall("link")):
        grew = False
        for j in joints:
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            if p in pose and c not in pose:
                o = j.find("origin")
                T = np.eye(4)
                T[:3, :3] = _rpy(_vec(o.get("rpy") or "0 0 0"))
                T[:3, 3] = _vec(o.get("xyz") or "0 0 0")
                if j.get("type") in ("revolute", "continuous"):
                    ax = _vec(j.find("axis").get("xyz"))
                    a = float(q.get(j.get("name"), 0.0))
                    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                    R = np.eye(4)
                    R[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
                    T = T @ R
                pose[c] = pose[p] @ T
                grew = True
        if not grew:
            break
    # PhysX collides each link mesh as its convex hull (Isaac's URDF importer default), so test hulls: a vertex of one
    # hull inside the other (all half-space distances <= 0) is an overlap; depth = how far inside the deepest one is.
    from scipy.spatial import ConvexHull
    hulls = {}
    for link in root.findall("link"):
        for col in link.findall("collision"):
            mesh_el = col.find("geometry/mesh")
            if mesh_el is None:
                continue
            mesh = trimesh.load(urdf.parent / mesh_el.get("filename"), force="mesh")
            v = np.asarray(mesh.vertices) * (_vec(mesh_el.get("scale")) if mesh_el.get("scale") else 1.0)
            o = col.find("origin")
            if o is not None and o.get("xyz"):
                v = v + _vec(o.get("xyz"))
            T = pose[link.get("name")]
            v = v @ T[:3, :3].T + T[:3, 3]
            h = ConvexHull(v)
            hulls[link.get("name")] = (v[h.vertices], h.equations)
    filtered = set()
    for a, bs in spec.get("adjacent_links", {}).items():
        for b in bs:
            filtered |= {(a, b), (b, a)}
    names = sorted(hulls)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (a, b) in filtered:
                continue
            depth = 0.0
            for (va, _), (_, eb) in ((hulls[a], hulls[b]), (hulls[b], hulls[a])):
                d = va @ eb[:, :3].T + eb[:, 3]
                inside = (d <= 0).all(axis=1)
                if inside.any():
                    depth = max(depth, float((-d[inside].max(axis=1)).max()))
            if depth > 0:
                out.append((a, b, depth))
    return out


def make(hand: str, side: str, w: float, l: float, tag: str) -> Path:
    base = HERE / hand / f"{hand}_{side}"
    spec = json.loads(Path(f"{base}.spec.json").read_text())
    tree = ET.parse(f"{base}.urdf")
    scale_urdf(tree.getroot(), spec["palm_body"], w, l)
    out = HERE / hand / f"{hand}_{side}_{tag}"
    tree.write(f"{out}.urdf", xml_declaration=True, encoding="utf-8")
    S = np.array([w, w, l])
    spec["urdf"] = str(Path(spec["urdf"]).with_name(f"{hand}_{side}_{tag}.urdf"))
    spec["fingertip_offsets"] = [list(np.asarray(p) * S) for p in spec["fingertip_offsets"]]
    for k in ("fingertip_pos_home_ours", "overlap_at_home_mm"):
        spec.pop(k, None)                                   # describe the full hand; recompute if needed
    spec["scaled_links"] = {"length": l, "width": w, "source": f"{hand}_{side}.urdf",
                            "generator": "make_scaled_links.py"}
    Path(f"{out}.spec.json").write_text(json.dumps(spec, indent=1))
    q0 = spec.get("hand_default_joint_pos", {})
    base_spec = json.loads(Path(f"{base}.spec.json").read_text())
    known = {(a, b) for a, b, _ in rest_overlaps(Path(f"{base}.urdf"), base_spec, q0)}   # the full hand's own (harmless)
    new = [(a, b, d) for a, b, d in rest_overlaps(Path(f"{out}.urdf"), spec, q0) if (a, b) not in known]
    # A pair the scaling made touch at rest would be pushed apart by PhysX every step (a 0.5 mm one stalled the
    # clone's ring MCP, 7 Oct): filter it, and say so in the spec.
    for a, b, _ in new:
        spec["adjacent_links"].setdefault(a, []).append(b)
    spec["scaled_links"]["filtered_rest_overlaps_mm"] = {f"{a}|{b}": round(d * 1000, 2) for a, b, d in new}
    Path(f"{out}.spec.json").write_text(json.dumps(spec, indent=1))
    print(f"{out}.urdf  (length x{l}, width x{w})")
    print(f"  link pairs the scaling made touch at the default pose (now collision-filtered): {len(new)}" +
          "".join(f"\n    {a} -- {b}: {d * 1000:.2f} mm deep" for a, b, d in new))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="wuji2")
    ap.add_argument("--side", default="left")
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    make(a.hand, a.side, a.width, a.length, a.tag)


if __name__ == "__main__":
    main()
