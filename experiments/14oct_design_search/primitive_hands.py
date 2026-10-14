"""Primitive hands: a disc palm with fingers of identical capsule modules, placed anywhere around its rim.

A hand is a ``PrimitiveHand``: the bearings at which its fingers leave the palm's rim, plus the module dimensions
(fixed for now). Everything else is derived from it, so a design is a few numbers in a JSON -- what a sampler, a
mutation or an agent proposes -- and ``build`` turns it into the URDF and spec JSON the envs load
(``isaacsimenvs.pose_reaching_6d.scene_utils.robots.unified_hands.spec_from_json``: the same format as the uniform
Wuji hands, so a primitive hand goes into the olympics env, the joint transformer and the multi-hand scene as is).

Geometry (the palm frame is the "ours" palm frame of the spec: palm_frame is the identity):

    palm      a disc, radius 5 cm, thickness 2 cm; its axis is +x, the grasp normal (the face fingers curl toward)
    finger    leaves the rim radially at mid-thickness, at bearing b in the palm plane (y-z), measured from +z
              toward +y; its base joint sits on the rim
    module    one capsule along the finger, radius 0.5 cm, 3 cm long (caps included), one joint at its base;
              4 modules per finger, collinear at q = 0
    joint     flexion only: the axis is the rim tangent, and q in [0, 90 deg] curls the finger toward +x

Collisions are URDF cylinders (the 2 cm between the cap centres) that the importer turns into capsules
(``replace_cylinders_with_capsules``); the visuals add the end spheres, so the URDF draws as the capsule.

    .venv_isaacsim/bin/python experiments/14oct_design_search/primitive_hands.py --random 8 --seed 0
    ... primitive_hands.py --bearings 0,90,180,270             # one hand from explicit bearings (degrees)
    # -> assets/urdf/generated/14oct_design_search/<name>.{urdf,spec.json,design.json}
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "assets/urdf/generated/14oct_design_search"   # named after this experiment

# Actuation and materials: the uniform-dynamics commercial hands' (unified_dynamics_commercial_hands), so a primitive
# hand trains under the same gains as every Wuji run.
STIFFNESS, DAMPING, ARMATURE = 3.0, 0.0775033248319064, 0.00058
EFFORT_LIMIT, VELOCITY_LIMIT = 0.5, 5.0
DENSITY = 1750.0                       # kg / m^3
FRICTION = 0.5


@dataclass(frozen=True)
class PrimitiveHand:
    """A design: where each finger leaves the rim (degrees, from +z toward +y), and the fixed dimensions."""
    bearings_deg: tuple[float, ...]
    palm_radius: float = 0.05
    palm_thickness: float = 0.02
    module_length: float = 0.03          # capsule, caps included; also the joint-to-joint pitch
    module_radius: float = 0.005
    modules_per_finger: int = 4
    flex_range_deg: tuple[float, float] = (0.0, 90.0)

    # -- identity ------------------------------------------------------------------------------------------
    def gaps_deg(self) -> list[float]:
        """The angles between consecutive fingers around the rim, in order (they sum to 360)."""
        b = sorted(x % 360.0 for x in self.bearings_deg)
        return [(b[(i + 1) % len(b)] - b[i]) % 360.0 or 360.0 for i in range(len(b))]

    def canonical(self) -> "PrimitiveHand":
        """The same hand rotated and mirrored about its axis into a standard pose: the first finger at 0, and of
        every rotation and reflection of the gap sequence the lexicographically smallest. Two designs that differ
        only by how the disc is turned (or flipped) on its mount get the same canonical form, and the same name."""
        g = [round(x, 6) for x in self.gaps_deg()]
        n = len(g)
        cands = [tuple(g[i:] + g[:i]) for i in range(n)] + [tuple((g[::-1])[i:] + (g[::-1])[:i]) for i in range(n)]
        best = min(cands)
        bearings = tuple(round(float(x), 6) for x in np.cumsum((0.0,) + best[:-1]))
        return PrimitiveHand(bearings_deg=bearings, **{k: v for k, v in asdict(self).items() if k != "bearings_deg"})

    @property
    def name(self) -> str:
        """``prim<F>x<M>_g<gaps>``: finger count, modules per finger, and the canonical gaps in whole degrees."""
        g = self.canonical().gaps_deg()
        return f"prim{len(g)}x{self.modules_per_finger}_g" + "_".join(f"{x:03.0f}" for x in g)

    def validate(self, min_gap_deg: float | None = None) -> None:
        """Raise on a design the URDF cannot honestly represent: overlapping finger bases."""
        if len(self.bearings_deg) < 1:
            raise ValueError("a hand needs at least one finger")
        need = min_gap_deg if min_gap_deg is not None else self.min_gap_deg()
        if min(self.gaps_deg()) < need - 1e-6:
            raise ValueError(f"{self.bearings_deg}: fingers {min(self.gaps_deg()):.1f} deg apart, need >= {need:.1f}")

    def min_gap_deg(self) -> float:
        """The smallest angle two finger bases can sit apart without their capsules overlapping at the rim."""
        return math.degrees(2.0 * math.asin(min(1.0, self.module_radius / self.palm_radius)))

    def to_json(self) -> dict:
        return {**asdict(self), "name": self.name}

    @staticmethod
    def from_json(d: dict) -> "PrimitiveHand":
        d = {k: v for k, v in d.items() if k != "name"}
        d["bearings_deg"] = tuple(d["bearings_deg"])
        if "flex_range_deg" in d:
            d["flex_range_deg"] = tuple(d["flex_range_deg"])
        return PrimitiveHand(**d)


# --- sampling -----------------------------------------------------------------------------------------------

def sample(n: int, *, fingers: int = 4, seed: int = 0, min_gap_deg: float = 20.0, grid_deg: float = 5.0,
           max_tries: int = 100000) -> list[PrimitiveHand]:
    """``n`` distinct hands (distinct up to turning and flipping the disc): ``fingers`` bearings drawn uniformly on
    a ``grid_deg`` grid, kept if every pair of neighbours is at least ``min_gap_deg`` apart."""
    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, 360.0, grid_deg)
    out, seen = [], set()
    for _ in range(max_tries):
        h = PrimitiveHand(bearings_deg=tuple(float(x) for x in rng.choice(grid, fingers, replace=False)))
        if min(h.gaps_deg()) < min_gap_deg or h.name in seen:
            continue
        seen.add(h.name)
        out.append(h.canonical())
        if len(out) == n:
            break
    if len(out) < n:
        raise RuntimeError(f"found only {len(out)} distinct hands in {max_tries} draws")
    return out


# --- geometry -----------------------------------------------------------------------------------------------

def finger_frame(bearing_deg: float) -> np.ndarray:
    """Columns: the finger's local x (grasp normal, palm +x), y (flexion axis: the rim tangent), z (outward)."""
    b = math.radians(bearing_deg)
    z = np.array([0.0, math.sin(b), math.cos(b)])          # radially outward, in the palm plane
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)                                        # rotating about +y takes +z toward +x: curls to the palm
    return np.stack([x, y, z], axis=1)


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw (fixed XYZ) of a rotation matrix."""
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        roll, yaw = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    else:
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return roll, pitch, yaw


def _capsule_mass(r: float, length: float) -> tuple[float, np.ndarray]:
    """Mass and inertia (about its centre, axis z) of a capsule ``length`` long overall."""
    h = max(length - 2.0 * r, 0.0)
    m_cyl = DENSITY * math.pi * r * r * h
    m_sph = DENSITY * 4.0 / 3.0 * math.pi * r ** 3
    m = m_cyl + m_sph
    izz = m_cyl * r * r / 2.0 + m_sph * 2.0 * r * r / 5.0
    # The two hemispheres as a sphere split to the cylinder's ends (parallel axis about the centre).
    ixx = (m_cyl * (3 * r * r + h * h) / 12.0 + m_sph * (2.0 * r * r / 5.0 + (h / 2.0 + 3.0 * r / 8.0) ** 2))
    return m, np.diag([ixx, ixx, izz])


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _inertial(link: ET.Element, mass: float, inertia: np.ndarray, xyz=(0.0, 0.0, 0.0)) -> None:
    el = ET.SubElement(link, "inertial")
    ET.SubElement(el, "origin", xyz=_fmt(xyz), rpy="0 0 0")
    ET.SubElement(el, "mass", value=f"{mass:.6g}")
    ET.SubElement(el, "inertia", ixx=f"{inertia[0, 0]:.6g}", ixy="0", ixz="0", iyy=f"{inertia[1, 1]:.6g}",
                  iyz="0", izz=f"{inertia[2, 2]:.6g}")


def _shape(link: ET.Element, kind: str, geometry: tuple[str, dict], xyz, rpy=(0.0, 0.0, 0.0),
           rgba: str | None = None, name: str = "") -> None:
    el = ET.SubElement(link, kind, **({"name": name} if name else {}))
    ET.SubElement(el, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
    g = ET.SubElement(el, "geometry")
    ET.SubElement(g, geometry[0], **{k: f"{v:.6g}" for k, v in geometry[1].items()})
    if rgba and kind == "visual":
        mat = ET.SubElement(el, "material", name=name or rgba)
        ET.SubElement(mat, "color", rgba=rgba)


PALM_RGBA, LINK_RGBA, TIP_RGBA = "0.72 0.76 0.86 1", "0.93 0.93 0.93 1", "0.55 0.78 0.95 1"


def urdf_of(hand: PrimitiveHand) -> ET.Element:
    h = hand
    root = ET.Element("robot", name=h.name)
    palm = ET.SubElement(root, "link", name="palm")
    # The disc: a URDF cylinder's axis is z, turned onto +x (the palm's axis).
    disc = ("cylinder", {"radius": h.palm_radius, "length": h.palm_thickness})
    turn = (0.0, math.pi / 2.0, 0.0)
    _shape(palm, "visual", disc, (0, 0, 0), turn, PALM_RGBA, "palm_disc")   # not "palm": scene nodes take visual names
    _shape(palm, "collision", disc, (0, 0, 0), turn)
    m = DENSITY * math.pi * h.palm_radius ** 2 * h.palm_thickness
    i_ax = m * h.palm_radius ** 2 / 2.0
    i_tr = m * (3 * h.palm_radius ** 2 + h.palm_thickness ** 2) / 12.0
    _inertial(palm, m, np.diag([i_ax, i_tr, i_tr]))

    r, L = h.module_radius, h.module_length
    cyl = ("cylinder", {"radius": r, "length": L - 2.0 * r})
    sph = ("sphere", {"radius": r})
    m_mod, I_mod = _capsule_mass(r, L)
    lo, hi = (math.radians(v) for v in h.flex_range_deg)
    for f, b in enumerate(h.bearings_deg):
        R = finger_frame(b)
        parent = "palm"
        for k in range(h.modules_per_finger):
            name = f"f{f}_l{k}"
            link = ET.SubElement(root, "link", name=name)
            tip = k == h.modules_per_finger - 1
            rgba = TIP_RGBA if tip else LINK_RGBA
            # The capsule spans z in [0, L] of its joint's frame: its cap centres at r and L - r.
            _shape(link, "visual", cyl, (0, 0, L / 2.0), rgba=rgba, name=f"{name}_body")
            _shape(link, "visual", sph, (0, 0, r), rgba=rgba, name=f"{name}_cap0")
            _shape(link, "visual", sph, (0, 0, L - r), rgba=rgba, name=f"{name}_cap1")
            _shape(link, "collision", cyl, (0, 0, L / 2.0))
            _inertial(link, m_mod, I_mod, xyz=(0, 0, L / 2.0))
            joint = ET.SubElement(root, "joint", name=f"f{f}_j{k}", type="revolute")
            ET.SubElement(joint, "parent", link=parent)
            ET.SubElement(joint, "child", link=name)
            if k == 0:      # on the rim, at mid-thickness, oriented so the finger runs along local +z
                ET.SubElement(joint, "origin", xyz=_fmt(R[:, 2] * h.palm_radius), rpy=_fmt(_rpy(R)))
            else:           # one module further along the finger
                ET.SubElement(joint, "origin", xyz=_fmt((0.0, 0.0, L)), rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 1 0")
            ET.SubElement(joint, "limit", lower=f"{lo:.6g}", upper=f"{hi:.6g}", effort=f"{EFFORT_LIMIT:g}",
                          velocity=f"{VELOCITY_LIMIT:g}")
            ET.SubElement(joint, "dynamics", damping="0", friction="0")
            parent = name
    return root


def spec_of(hand: PrimitiveHand, urdf_rel: str) -> dict:
    """The spec JSON ``spec_from_json`` reads. The palm body's frame IS the "ours" palm frame (x the grasp normal,
    the palm in y-z), so palm_frame is the identity and the palm box is centred on the body's origin."""
    h = hand
    F, M = len(h.bearings_deg), h.modules_per_finger
    joints = [f"f{f}_j{k}" for f in range(F) for k in range(M)]
    lo, hi = (math.radians(v) for v in h.flex_range_deg)
    adj: dict[str, list[str]] = {"palm": []}
    for f in range(F):
        for k in range(M):
            name, prev = f"f{f}_l{k}", ("palm" if k == 0 else f"f{f}_l{k - 1}")
            adj.setdefault(name, []).append(prev)
            adj.setdefault(prev, []).append(name)
        # The palm against every link within two joints of it, as the uniform hands (palm_filter_depth 2).
        if M > 1:
            adj["palm"].append(f"f{f}_l1")
            adj[f"f{f}_l1"].append("palm")
    return {
        "hand": h.name, "side": "left", "urdf": urdf_rel, "palm_body": "palm",
        "hand_joint_names": joints,
        "joint_limits": {j: [lo, hi] for j in joints},
        "hand_default_joint_pos": {j: 0.0 for j in joints},
        "fingertip_body_names": [f"f{f}_l{M - 1}" for f in range(F)],
        "fingertip_offsets": [[0.0, 0.0, h.module_length] for _ in range(F)],
        "palm_frame": np.eye(3).tolist(), "palm_frame_det": 1.0,
        "palm_centre_in_palm_body": [0.0, 0.0, 0.0],
        "palm_extents_ours": [h.palm_thickness, 2 * h.palm_radius, 2 * h.palm_radius],
        "palm_centre_ours": [0.0, 0.0, 0.0],
        "adjacent_links": adj, "palm_filter_depth": 2,
        "stiffness": STIFFNESS, "damping": DAMPING, "armature": ARMATURE,
        "effort_limit": EFFORT_LIMIT, "velocity_limit": VELOCITY_LIMIT,
        "link_density_kg_m3": DENSITY, "friction": FRICTION,
        "replace_cylinders_with_capsules": True,
        "design": h.to_json(),
    }


def build(hand: PrimitiveHand, out_dir: pathlib.Path = OUT) -> dict[str, pathlib.Path]:
    """Write ``<name>.urdf``, ``<name>.spec.json`` and ``<name>.design.json`` for ``hand`` (canonicalised)."""
    hand = hand.canonical()
    hand.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{hand.name}.{k}" for k in ("urdf", "spec.json", "design.json")}
    tree = ET.ElementTree(urdf_of(hand))
    ET.indent(tree)
    tree.write(paths["urdf"], encoding="unicode", xml_declaration=False)
    rel = str(paths["urdf"].resolve().relative_to(REPO)) if paths["urdf"].resolve().is_relative_to(REPO) \
        else str(paths["urdf"].resolve())
    paths["spec.json"].write_text(json.dumps(spec_of(hand, rel), indent=1))
    paths["design.json"].write_text(json.dumps(hand.to_json(), indent=1))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=0, help="this many distinct random 4-finger hands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gap", type=float, default=20.0, help="degrees between neighbouring fingers, at least")
    ap.add_argument("--grid", type=float, default=5.0, help="bearings are multiples of this (degrees)")
    ap.add_argument("--bearings", default="", help="one hand: comma-separated bearings in degrees, e.g. 0,90,180,270")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    hands = sample(args.random, seed=args.seed, min_gap_deg=args.min_gap, grid_deg=args.grid) if args.random else []
    if args.bearings:
        hands.append(PrimitiveHand(bearings_deg=tuple(float(x) for x in args.bearings.split(","))))
    if not hands:
        ap.error("give --random N or --bearings")
    for h in hands:
        p = build(h, pathlib.Path(args.out))
        print(f"{h.name}: bearings {list(h.canonical().bearings_deg)} -> {p['urdf']}")


if __name__ == "__main__":
    main()
