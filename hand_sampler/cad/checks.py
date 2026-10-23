"""Checks on a rendered design (manifold3d, trimesh): what must hold before
anyone prints it.

- every part is watertight and one piece;
- each joint turns over its whole range without its two parts, or the
  servo case and plug in the parent, cutting into each other (sampled
  every `JOINT_STEP_DEG`; across a 0 mm link over both joints);
- every servo case fits its pocket, and its horns stay clear of its parent;
- no two other parts touch at the zero pose;
- how far each part reaches outside its simulator body (the grammar's
  solid; Martin, 2026-10-23: small overhangs are fine).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from hand_sampler.grammar import derive as dv

from .design import Design, ServoSpot, T, inv

JOINT_STEP_DEG = 10.0
TOUCH_MM3 = 1.0               # an intersection smaller than this is a touch, not a clash
OVERHANG_LIMIT_MM = 2.5


def _load(path: Path):
    import manifold3d as m3
    import trimesh

    tm = trimesh.load(str(path), force="mesh")
    mesh = m3.Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                   tri_verts=np.asarray(tm.faces, dtype=np.uint32))
    return tm, m3.Manifold(mesh)


def _xf(man, M: np.ndarray):
    return man.transform(np.asarray(M[:3, :4], dtype=np.float64).tolist())


def _servo_solid(idler_horn: bool):
    """The servo's case and horns (S frame) as one manifold."""
    import manifold3d as m3

    case = m3.Manifold.cube([23.0, 34.0, 20.0]).translate([-11.5, -9.5, -10.0])
    horn = m3.Manifold.cylinder(3.0, 8.0, 8.0, 48).rotate([0.0, 90.0, 0.0]).translate([11.5, 0.0, 0.0])
    out = case + horn
    if idler_horn:
        out = out + m3.Manifold.cylinder(2.4, 8.0, 8.0, 48).rotate([0.0, 90.0, 0.0]).translate([-14.5, 0.0, 0.0])
    return out


def _case_solid():
    import manifold3d as m3

    return m3.Manifold.cube([23.0, 34.0, 20.0]).translate([-11.5, -9.5, -10.0])


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return T(np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]))


def _angles(lo: float, hi: float, step: float = JOINT_STEP_DEG) -> List[float]:
    n = max(1, int(math.ceil((hi - lo) / step)))
    return [lo + (hi - lo) * k / n for k in range(n + 1)]


@dataclass
class Report:
    parts: Dict[str, dict] = field(default_factory=dict)
    clashes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.clashes and all(p["ok"] for p in self.parts.values())

    def summary(self) -> str:
        bad = [n for n, p in self.parts.items() if not p["ok"]]
        lines = [f"{len(self.parts)} parts, {len(bad)} with problems, {len(self.clashes)} clashes"]
        lines += [f"  part {n}: {', '.join(self.parts[n]['problems'])}" for n in bad]
        lines += [f"  clash: {c}" for c in self.clashes]
        lines += [f"  note: {w}" for w in self.warnings]
        return "\n".join(lines)


def check(design: Design, stl_dir: Path) -> Report:
    stl_dir = Path(stl_dir)
    rep = Report()
    solids = {}
    frames = {k: p.frame for k, p in design.parts.items()}
    entries = list(design.parts.items()) + [(("extra", p.name), p) for p in getattr(design, "extras", [])]
    for key, part in entries:
        frames[key] = part.frame
        names = [f"{part.name}_front", f"{part.name}_back"] if part.split_x is not None else [part.name]
        mans = []
        for n in names:
            info = {"ok": True, "problems": []}
            path = stl_dir / f"{n}.stl"
            if not path.exists():
                info.update(ok=False, problems=["not rendered"])
                rep.parts[n] = info
                continue
            tm, man = _load(path)
            info["volume_cm3"] = round(float(man.volume()) / 1000.0, 2)
            info["grams_pla"] = round(float(man.volume()) / 1000.0 * 1.24 * 0.5, 1)   # about half full
            ext = tm.bounds[1] - tm.bounds[0]
            info["extent_mm"] = [round(float(x), 1) for x in ext]
            import manifold3d as m3
            if man.status() != m3.Error.NoError or man.is_empty():
                info["problems"].append("not a closed solid")
            pieces = [p for p in man.decompose() if p.volume() > 1.0]
            if len(pieces) != 1:
                info["problems"].append(f"{len(pieces)} separate pieces")
            info["ok"] = not info["problems"]
            rep.parts[n] = info
            mans.append(man)
        if mans:
            whole = mans[0]
            for m in mans[1:]:
                whole = whole + m
            solids[key] = whole
    _joint_checks(design, solids, rep)
    _zero_pose_checks(design, solids, rep, frames)
    _overhangs(design, solids, rep)
    return rep


HELD_FRACTION = 0.3
"""A servo is held if its parent keeps at least this share of a full
housing's shell around its case."""


def _shell_solid():
    import manifold3d as m3

    outer = m3.Manifold.cube([27.4, 36.2, 24.4]).translate([-13.7, -9.5, -12.2])
    return outer - m3.Manifold.cube([23.4, 34.2, 20.4]).translate([-11.7, -9.6, -10.2])


def _joint_checks(design: Design, solids: Dict, rep: Report) -> None:
    shell = _shell_solid()
    full = shell.volume()
    for sp in design.servos.values():
        P, C = solids.get(sp.parent), solids.get(sp.child)
        if P is None or C is None:
            continue
        name = "/".join(str(k) for k in sp.key)
        held = (P ^ _xf(shell, sp.S_parent)).volume() / full
        if held < HELD_FRACTION:
            rep.clashes.append(f"servo {name}: its housing keeps only {100 * held:.0f}% of a full shell")
        # the servo sits in its parent: its case must not cut the parent
        case_in_parent = _xf(_case_solid(), sp.S_parent)
        v = (P ^ case_in_parent).volume()
        if v > TOUCH_MM3:
            rep.clashes.append(f"servo {name}: its case cuts {sp.parent} ({v:.0f} mm3)")
        # the child (with its horns) turning against the parent and its servo case
        servo_c = _xf(_servo_solid(sp.idler_horn), sp.S_child)
        Pc = P + case_in_parent
        lo, hi = sp.rot_deg
        worst = 0.0
        for a in _angles(lo, hi):
            # the child in the parent's frame: the servo frame turned by a about its X
            M = sp.S_parent @ _rot_x(a) @ inv(sp.S_child)
            v = (Pc ^ _xf(C, M)).volume()
            worst = max(worst, v)
        if worst > TOUCH_MM3:
            rep.clashes.append(f"joint {name}: {sp.child} cuts {sp.parent} or its servo ({worst:.0f} mm3)")
        v = (C ^ _xf(_case_solid(), sp.S_child)).volume()
        if v > TOUCH_MM3:
            rep.clashes.append(f"joint {name}: {sp.child} cuts its servo's case at the zero pose ({v:.0f} mm3)")
        del servo_c


def _zero_pose_checks(design: Design, solids: Dict, rep: Report, frames: Dict) -> None:
    keys = list(solids)
    placed = {k: _xf(solids[k], frames[k]) for k in keys}
    adjacent = {frozenset((("palm", -1), ("extra", "mount"))), frozenset((("extra", "bay"), ("extra", "mount")))}
    for sp in design.servos.values():
        adjacent.add(frozenset((sp.parent, sp.child)))
        if sp.yoke == "crossing":
            _, i, j = sp.key
            adjacent.add(frozenset((sp.parent, ("link", i, j + 1))))
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            if frozenset((keys[a], keys[b])) in adjacent:
                continue
            v = (placed[keys[a]] ^ placed[keys[b]]).volume()
            if v > TOUCH_MM3:
                rep.clashes.append(f"{keys[a]} and {keys[b]} overlap at the zero pose ({v:.0f} mm3)")


def _sim_points_mm(design: Design, key: Tuple) -> Optional[np.ndarray]:
    """The simulator's solid for a part's body, in the body's frame (mm)."""
    hand = design.hand
    if key[0] == "link":
        _, i, j = key
        return dv.link_points_m(hand, i, j) / dv.MM
    g = dv.hand_geometry(hand)
    if key[0] != "palm":
        return None
    # the plate with the base servos (and the hinge servo) in their housings
    pts = [dv.plate_points_m(g.outlines[key[1]]) / dv.MM]
    for i, bs in g.base_boxes.items():
        if (key[1] < 0) == (dv.finger_section(hand, i) < 0):
            pts += [b.grown(dv.mt.WALL_MM).corners() for b in bs]
    if key[1] < 0 and g.hinge_box is not None:
        pts.append(g.hinge_box.grown(dv.mt.WALL_MM).corners())
    return np.concatenate(pts)


def _overhangs(design: Design, solids: Dict, rep: Report) -> None:
    from scipy.spatial import ConvexHull

    for key, man in solids.items():
        if key[0] == "extra":
            continue
        P = _sim_points_mm(design, key)
        if P is None:
            continue
        hull = ConvexHull(P)
        mesh = man.to_mesh()
        V = np.asarray(mesh.vert_properties)[:, :3].astype(float)
        # how far each vertex lies outside the hull: the largest signed plane distance
        d = (V @ hull.equations[:, :3].T + hull.equations[:, 3]).max(axis=1)
        worst = float(max(0.0, d.max()))
        name = design.parts[key].name
        for n in ([f"{name}_front", f"{name}_back"] if design.parts[key].split_x is not None else [name]):
            if n in rep.parts:
                rep.parts[n]["overhang_mm"] = round(worst, 1)
        if worst > OVERHANG_LIMIT_MM:
            rep.warnings.append(f"{name} reaches {worst:.1f} mm outside its simulator body")
