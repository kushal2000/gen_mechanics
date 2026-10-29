"""Checks on a rendered design (manifold3d, trimesh): what must hold before
anyone prints it.

- every part is watertight and one piece;
- each joint turns over its whole range without its two parts, or the
  servo case and plug in the parent, cutting into each other (sampled
  every `JOINT_STEP_DEG`; across a 0 mm link over both joints);
- every servo case fits its housing, which keeps its screw bosses, and its
  horns stay clear of its parent;
- a screwdriver reaches every screw in the order ASSEMBLY.md gives: the
  case screws past the part holding the servo, the arm screws at some joint
  angle or the link slid in on the servo;
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
STOP_FREE_DEG = 10.0          # a base joint must turn this far either way before its finger meets the palm
TOUCH_MM3 = 1.0               # an intersection smaller than this is a touch, not a clash
OVERHANG_LIMIT_MM = 2.5


def _load(path: Path):
    from .render import load

    return load(path)


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
    orders: Dict[Tuple, Optional[str]] = field(default_factory=dict)   # per servo: "servo first" / "link first"

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
        mans = []
        for n in [part.name]:
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


"""A servo screwed into a socket or housing is held if three of the four
screw bosses round its case's far holes are there (`BOSS_SHARE`; a warning
below four) and most of its back wall (`BACK_SHARE`)."""
BOSS_SHARE = 0.7            # a boss counts if this much of it is there; three of four must
BACK_SHARE = 0.4            # under this much back wall, a warning (the screws carry the load)


FAR_YZ = ((22.5, -8.0), (22.5, 8.0))
NEAR_YZ = ((-7.5, -8.0), (-7.5, 8.0))


def _bosses(flange: bool = False):
    """The four screw bosses that hold a servo (servo frame), each less its
    hole: round the case's far holes on both horn faces, or (a flange cup)
    round the four holes of its idler face."""
    import manifold3d as m3

    def boss(x0, y, z):
        c = m3.Manifold.cylinder(2.0, 3.5, 3.5, 24).rotate([0.0, 90.0, 0.0]).translate([x0, y, z])
        return c - m3.Manifold.cylinder(2.2, 1.2, 1.2, 16).rotate([0.0, 90.0, 0.0]).translate([x0 - 0.1, y, z])

    if flange:
        return [boss(-13.6, y, z) for y, z in FAR_YZ + NEAR_YZ]
    return [boss(x0, y, z) for y, z in FAR_YZ for x0 in (11.6, -13.6)]


def _pocket_solid():
    """A palm pocket's outer box (servo frame)."""
    import manifold3d as m3

    return m3.Manifold.cube([27.2, 36.1, 22.6]).translate([-13.6, -9.5, -11.3])


def _back_wall_solid():
    import manifold3d as m3

    return m3.Manifold.cube([23.0, 2.0, 20.0]).translate([-11.5, 24.6, -10.0])


def _bodies(design: Design, solids: Dict) -> Dict:
    """Each rigid body's printed parts as one solid, in the body's frame."""
    return {key: man for key, man in solids.items() if key[0] != "extra"}


def _joint_checks(design: Design, solids: Dict, rep: Report) -> None:
    bodies = _bodies(design, solids)
    extras = None
    for p in getattr(design, "extras", []):
        e = solids.get(("extra", p.name))
        if e is not None:
            e = _xf(e, p.frame)
            extras = e if extras is None else extras + e
    rep.orders = {}
    back = _back_wall_solid()
    for sp in design.servos.values():
        H = solids.get(sp.parent)                              # what its case is screwed to
        P, C = bodies.get(sp.parent), bodies.get(sp.child)     # the bodies turning against each other
        if P is None or C is None or H is None:
            continue
        name = "/".join(str(k) for k in sp.key)
        kept = sum((H ^ _xf(b, sp.S_parent)).volume() >= BOSS_SHARE * b.volume() for b in _bosses(sp.flange))
        w = (H ^ _xf(back, sp.S_parent)).volume() / back.volume()
        if kept < 3:
            rep.clashes.append(f"servo {name}: its housing keeps {kept} of its 4 screw bosses")
        elif kept < 4:
            rep.warnings.append(f"servo {name}: its housing keeps 3 of its 4 screw bosses (use 3 screws)")
        if w < BACK_SHARE and not sp.flange:
            rep.warnings.append(f"servo {name}: its housing keeps {100 * w:.0f}% of its back wall (something "
                                f"turns close to it; the case screws hold it)")
        if sp.parent == ("palm", -1) and extras is not None:
            P = P + extras                        # the bay and the mount ride on the main palm
        # the servo sits in its parent: its case must not cut the parent
        case_in_parent = _xf(_case_solid(), sp.S_parent)
        v = (P ^ case_in_parent).volume()
        if v > TOUCH_MM3:
            rep.clashes.append(f"servo {name}: its case cuts {sp.parent} ({v:.0f} mm3)")
        # the child (with its horns) turning against the parent and its servo case
        Pc = P + case_in_parent
        lo, hi = sp.rot_deg
        worst, hits = 0.0, []
        for a in _angles(lo, hi):
            # the child in the parent's frame: the servo frame turned by a about its X
            M = sp.S_parent @ _rot_x(a) @ inv(sp.S_child)
            v = (Pc ^ _xf(C, M)).volume()
            worst = max(worst, v)
            if v > TOUCH_MM3:
                hits.append(a)
        if worst > TOUCH_MM3 and sp.parent[0] == "palm" and sp.key[0] == "f" and all(abs(a) > STOP_FREE_DEG for a in hits):
            # past its pocket a finger folding onto the palm stops on it, as in the simulator (Martin, 10/29)
            a0 = max([a for a in hits if a < 0.0], default=lo)
            a1 = min([a for a in hits if a > 0.0], default=hi)
            q0, q1 = sorted((a0 * sp.sign, a1 * sp.sign))
            rep.warnings.append(f"joint {name}: its finger stops on the palm past its pocket (free from {q0:.0f} "
                                f"to {q1:.0f} degrees)")
            worst = 0.0
        if worst > TOUCH_MM3 and sp.parent[0] == "palm" and sp.key[0] == "f":
            # the palm keeps other fingers' pockets: meeting them in motion is finger against finger,
            # left to the physics and the policy (a warning), unless it still clashes without them
            others = None
            for o in design.parts[sp.parent].holds:
                # other fingers' housings and the hinge's (kept against every finger's sweep)
                if (o.key[0] == "f" and o.key[1] != sp.key[1]) or o.key == ("hinge",):
                    b = _xf(_pocket_solid(), o.S_parent)
                    others = b if others is None else others + b
            if others is not None:
                Po = Pc - others
                rest = max((Po ^ _xf(C, sp.S_parent @ _rot_x(a) @ inv(sp.S_child))).volume() for a in _angles(lo, hi))
                if rest <= TOUCH_MM3:
                    rep.warnings.append(f"joint {name}: {sp.child} meets another servo's housing in the palm at some "
                                        f"angles (in each other's way, as in the simulator)")
                    worst = 0.0
        if worst > TOUCH_MM3:
            rep.clashes.append(f"joint {name}: {sp.child} cuts {sp.parent} or its servo ({worst:.0f} mm3)")
        v = (C ^ _xf(_case_solid(), sp.S_child)).volume()
        if v > TOUCH_MM3:
            rep.clashes.append(f"joint {name}: {sp.child} cuts its servo's case at the zero pose ({v:.0f} mm3)")
        order = _assembly_order(sp, P, solids.get(sp.child, C))
        rep.orders[sp.key] = order
        if order is None:
            rep.clashes.append(f"joint {name}: no way to put it together: a screwdriver cannot reach the case "
                               f"screws past {sp.parent}, or neither the arm's screws (at any angle) nor sliding the "
                               f"servo in with {sp.child} on it works")


def _assembly_order(sp: ServoSpot, P, C) -> Optional[str]:
    """How a joint goes together: "servo first" (the servo into its housing,
    then the link onto its horns: a screwdriver must reach the arm's screws
    past the parent at some angle) or "link first" (the link onto the servo
    on the bench, then both slid into the housing from the front: the link
    must pass the parent on the way in and leave the case screws reachable).
    Either way a screwdriver must reach the case screws past the parent
    (two of the far four; a flange cup's servo, three of its idler face's
    four, and it goes in along its axis: servo first only). None if it
    cannot or no order works."""
    # the case screws are driven past the part holding the servo, whatever the order
    if sp.flange:
        reach = sum((P ^ _xf(dv_, sp.S_parent)).volume() <= TOUCH_MM3 for dv_ in _case_driver_list(FAR_YZ + NEAR_YZ, (-1,)))
        if reach < 3:
            return None
    elif sum((P ^ _xf(dv_, sp.S_parent)).volume() <= TOUCH_MM3 for dv_ in _case_driver_list()) < 2:
        return None
    lo, hi = sp.rot_deg
    drivers = _drivers(sp)
    for a in (0.0, lo, hi):
        M = sp.S_parent @ _rot_x(a) @ inv(sp.S_child)
        if (P ^ _xf(_xf(drivers, sp.S_child), M)).volume() <= TOUCH_MM3:
            return "servo first"
    if sp.flange:
        return None
    # link first: the child slid in along the servo's back (+Y), from 60 mm in front of where it ends up
    Cs = _xf(C, inv(sp.S_child))                  # the child in the servo's frame
    Ps = _xf(P, inv(sp.S_parent))                 # the parent in the servo's frame
    for t in np.arange(2.0, 61.0, 4.0):
        if (Ps ^ Cs.translate([0.0, -float(t), 0.0])).volume() > TOUCH_MM3:
            return None
    case_drivers = _case_drivers()
    if (Cs ^ case_drivers).volume() > TOUCH_MM3:
        return None
    return "link first"


def _case_driver_list(holes=FAR_YZ, sides=(-1, 1)):
    import manifold3d as m3

    out = []
    for y, z in holes:
        for s in sides:
            c = m3.Manifold.cylinder(25.0, 1.6, 1.6, 16).rotate([0.0, 90.0, 0.0])
            out.append(c.translate([14.0 if s > 0 else -39.0, y, z]))
    return out


def _case_drivers():
    """A screwdriver's shaft (3.2 mm, an M2 driver's) on each of the four
    case screws, 25 mm out along the servo's axis from the side plates
    (servo frame)."""
    import manifold3d as m3

    out = None
    for y, z in ((22.5, -8.0), (22.5, 8.0)):
        for s in (-1, 1):
            c = m3.Manifold.cylinder(25.0, 1.6, 1.6, 16).rotate([0.0, 90.0, 0.0])
            c = c.translate([14.0 if s > 0 else -39.0, y, z])
            out = c if out is None else out + c
    return out


def _drivers(sp: ServoSpot):
    """A screwdriver's shaft (4 mm) on each arm screw, 25 mm out along the
    servo's axis from the arm's outer face (servo frame; as far as the case
    screws' and the library's `SCREWDRIVER_REACH`)."""
    import manifold3d as m3

    out = None
    sides = [(1, (45.0, 135.0, 225.0, 315.0), 17.6)] + ([(-1, (45.0, 135.0, 225.0, 315.0), 16.6)] if sp.yoke == "u" else [])
    for s, holes, x in sides:
        for a in holes:
            c = m3.Manifold.cylinder(25.0, 2.0, 2.0, 16).rotate([0.0, 90.0, 0.0])
            c = c.translate([x + 0.5 if s > 0 else -x - 25.5, 6.0 * math.sin(math.radians(a)), 6.0 * math.cos(math.radians(a))])
            out = c if out is None else out + c
    return out


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
        if name in rep.parts:
            rep.parts[name]["overhang_mm"] = round(worst, 1)
        if worst > OVERHANG_LIMIT_MM:
            rep.warnings.append(f"{name} reaches {worst:.1f} mm outside its simulator body")
