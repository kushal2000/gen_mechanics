"""The commercial reference hands (GRAMMAR-LOCK item 19), read from their
URDFs as plain kinematic data for `conform`.

Every hand is an entry of `hand_sampler/grammar_bench/manifest.json`; the
reference set is that file's `reference_set`. Per-hand annotations (data,
never code): `hand_root` (the body where the hand mounts: the wrist centre is
its origin), `palm_joints` (movable joints whose child is a palm section),
`tip_frames` (finger key -> the body at its fingertip), `tip_points` (last
joint -> the fingertip in that joint's child frame, for URDFs without
fingertip bodies), `drop_joints` (loop-closing joints, MIDAS) and
`thumb_joint` (the first joint of the thumb, where its name does not say) and
`tip_extend` (no fingertip in the file: the last bone continues by the length
of the one before it).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters.urdf import load_urdf
from .fk import forward_kinematics, pose_to_matrix
from .kinematics import KinematicModel, MOVABLE_TYPES

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_PATH = BENCH_DIR / "manifest.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def reference_ids() -> List[str]:
    return list(load_manifest()["reference_set"])


def entry(hand_id: str) -> dict:
    for h in load_manifest()["hands"]:
        if h["id"] == hand_id:
            return h
    raise KeyError(f"hand {hand_id!r} not in the manifest")


def resolve_path(h: dict, manifest: Optional[dict] = None) -> Optional[Path]:
    """The URDF to read (the committed fixture first), or None if it is not
    on this machine."""
    manifest = manifest or load_manifest()
    fp = h.get("fixture_path")
    if fp and (BENCH_DIR / fp).is_file():
        return BENCH_DIR / fp
    sp = h.get("source_path")
    if not isinstance(sp, str) or not sp:
        return None
    if sp.startswith("REPO:"):
        p = REPO_ROOT / sp[len("REPO:"):]
    else:
        p = Path(manifest.get("source_root") or "/") / sp
    return p if p.is_file() else None


def mesh_path(h: dict, manifest: Optional[dict] = None) -> Optional[Path]:
    """A copy of the URDF that ships its meshes (for drawing), if any."""
    manifest = manifest or load_manifest()
    sp = h.get("mesh_source_path") or h.get("source_path")
    if not isinstance(sp, str) or not sp:
        return None
    if sp.startswith("REPO:"):
        p = REPO_ROOT / sp[len("REPO:"):]
    else:
        p = Path(manifest.get("source_root") or "/") / sp
    return p if p.is_file() else None


@dataclass
class RealJoint:
    name: str
    type: str                 # revolute / prismatic / continuous
    point: np.ndarray         # (3,) a point of the axis line at the zero pose, hand-root frame (m)
    axis: np.ndarray          # (3,) unit axis at the zero pose
    coupled_to: Optional[str] = None     # the joint it mimics, if any
    ratio: float = 1.0


@dataclass
class RealFinger:
    key: str                  # the chain's last body
    joints: List[RealJoint]
    tip: np.ndarray           # (3,) fingertip, hand-root frame (m)
    palm_joint: Optional[str]  # the palm joint whose section carries it, if any
    thumb: bool


@dataclass
class RealHand:
    id: str
    model: KinematicModel
    fingers: List[RealFinger]
    palm_joints: List[RealJoint]
    notes: List[str] = field(default_factory=list)

    def joint(self, name: str) -> RealJoint:
        for j in self.palm_joints:
            if j.name == name:
                return j
        for f in self.fingers:
            for j in f.joints:
                if j.name == name:
                    return j
        raise KeyError(name)


def _drop_joints(model: KinematicModel, names: Sequence[str]) -> KinematicModel:
    if not names:
        return model
    drop = set(names)
    dropped_bodies = {j.child for j in model.joints if j.name in drop}
    # a dropped joint's child and everything under it go
    changed = True
    while changed:
        changed = False
        for j in model.joints:
            if j.parent in dropped_bodies and j.child not in dropped_bodies:
                dropped_bodies.add(j.child)
                changed = True
    joints = tuple(j for j in model.joints if j.name not in drop and j.child not in dropped_bodies)
    kept = {j.name for j in joints}
    return KinematicModel(name=model.name, root=model.root,
                          bodies=tuple(b for b in model.bodies if b.name not in dropped_bodies),
                          joints=joints,
                          couplings=tuple(c for c in model.couplings if c.dependent in kept and c.source in kept))


def load_real_hand(hand_id: str, path: Optional[Path] = None) -> RealHand:
    """Read one reference hand: its fingers as chains of movable joints from
    the palm out (fixed joints folded in), each joint's axis line at the zero
    pose, and its fingertip, all in the hand-root frame."""
    manifest = load_manifest()
    h = entry(hand_id)
    path = path or resolve_path(h, manifest)
    if path is None:
        raise FileNotFoundError(f"local-only:{hand_id} source not on this machine")
    if h.get("sha256") and path.suffix == ".urdf" and not h.get("fixture_path"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != h["sha256"]:
            raise ValueError(f"{hand_id}: sha256 mismatch ({digest} != {h['sha256']})")
    data = path.read_bytes()
    drop = h.get("drop_joints") or []
    if drop:
        # loop-closing joints give a body two parents; cut them from the XML first
        import xml.etree.ElementTree as ET

        root = ET.fromstring(data)
        for jel in list(root.findall("joint")):
            if jel.attrib.get("name") in drop:
                root.remove(jel)
        data = ET.tostring(root)
    imported = load_urdf(data, hand_root=h.get("hand_root"))
    return real_hand(hand_id, _drop_joints(imported.model, drop), h)


def real_hand(hand_id: str, model: KinematicModel, h: dict) -> RealHand:
    """The fingers of an already loaded `model`, with the manifest-style
    annotations `h` (palm_joints, tip_frames, tip_points, tip_extend,
    thumb_joint)."""
    T0 = forward_kinematics(model, {})
    by_name = {j.name: j for j in model.joints}
    children: Dict[str, List] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)
    coupling = {c.dependent: c for c in model.couplings}

    def real_joint(j) -> RealJoint:
        T = T0[j.parent] @ pose_to_matrix(j.origin)
        a = T[:3, :3] @ np.asarray(j.axis, dtype=float)
        a = a / np.linalg.norm(a)
        c = coupling.get(j.name)
        return RealJoint(j.name, j.type, T[:3, 3].copy(), a, c.source if c else None, c.multiplier if c else 1.0)

    def movable_below(body: str) -> List:
        """Movable joints reached from `body` through fixed joints only."""
        out, stack = [], [body]
        while stack:
            b = stack.pop()
            for j in children.get(b, []):
                if j.type in MOVABLE_TYPES:
                    out.append(j)
                else:
                    stack.append(j.child)
        return out

    def fixed_leaf(body: str) -> np.ndarray:
        best, best_d = T0[body][:3, 3], 0.0
        stack = [body]
        while stack:
            b = stack.pop()
            for j in children.get(b, []):
                if j.type == "fixed":
                    stack.append(j.child)
                    d = float(np.linalg.norm(T0[j.child][:3, 3] - T0[body][:3, 3]))
                    if d > best_d:
                        best, best_d = T0[j.child][:3, 3], d
        return best

    palm_joint_names = list(h.get("palm_joints") or [])
    palm_bodies = {model.root}
    palm_child = {}
    for name in palm_joint_names:
        j = by_name[name]
        palm_child[name] = j.child
    # the palm: the root, palm sections, and bodies fixed to them
    def palm_closure(b: str) -> set:
        out, stack = {b}, [b]
        while stack:
            x = stack.pop()
            for j in children.get(x, []):
                if j.type == "fixed":
                    out.add(j.child)
                    stack.append(j.child)
        return out

    section_of: Dict[str, Optional[str]] = {b: None for b in palm_closure(model.root)}
    for name in palm_joint_names:
        for b in palm_closure(palm_child[name]):
            section_of[b] = name
    tip_frames = h.get("tip_frames") or {}
    tip_points = h.get("tip_points") or {}
    thumb_joint = h.get("thumb_joint")
    fingers: List[RealFinger] = []
    for b, section in sorted(section_of.items()):
        for j0 in children.get(b, []):
            if j0.type not in MOVABLE_TYPES or j0.name in palm_joint_names:
                continue
            chain = [j0]
            while True:
                nxt = movable_below(chain[-1].child)
                if not nxt:
                    break
                if len(nxt) > 1:
                    raise ValueError(f"{hand_id}: finger branches below {chain[-1].child!r}")
                chain.append(nxt[0])
            last = chain[-1]
            key = last.child
            if key in tip_frames:
                tip = T0[tip_frames[key]][:3, 3].copy()
            elif last.name in tip_points:
                tip = (T0[last.child] @ np.append(np.asarray(tip_points[last.name], dtype=float), 1.0))[:3]
            elif h.get("tip_extend") and len(chain) >= 2:
                a, b = real_joint(chain[-2]).point, real_joint(chain[-1]).point
                tip = b + (b - a)
            else:
                tip = fixed_leaf(last.child).copy()
            names = " ".join(j.name.lower() for j in chain)
            thumb = (thumb_joint == j0.name) if thumb_joint else ("thumb" in names)
            fingers.append(RealFinger(key, [real_joint(j) for j in chain], tip, section, thumb))
    fingers.sort(key=lambda f: f.joints[0].name)
    notes = []
    for f in fingers:
        d = float(np.linalg.norm(f.tip - f.joints[-1].point))
        if d < 1e-3:
            notes.append(f"{f.key}: no fingertip beyond the last joint")
    return RealHand(id=hand_id, model=model, fingers=fingers,
                    palm_joints=[real_joint(by_name[n]) for n in palm_joint_names], notes=notes)
