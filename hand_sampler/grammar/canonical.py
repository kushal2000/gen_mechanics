"""Deterministic canonical form and phenotype hash for a ``KinematicModel``.

``canonical_form`` renames every body/joint (bodies -> ``b0, b1, ...``;
joints -> ``j0, j1, ...``) in a fixed DFS order from the root, so two
derivations that build the identical tree via a different derivation-step
order (e.g. list-splicing by ``vary``'s insert/delete-phalanx operators)
produce byte-identical canonical models. Frames and couplings are remapped
to match. ``phenotype_hash`` is the sha256 of that canonical model's JSON
(floats rounded to 1e-9) -- a cheap, exact "same phenotype?" test.

DFS child order at each body: sorted by (joint type, rounded origin xyz,
rounded origin rpy, rounded axis, rounded limits, whether the joint is a
coupling dependent and -- if so -- its (ancestor-hop-count to its coupling
source, multiplier, offset), the child body's own palm flag and radius),
with a final name-independent tie-break on the child's own (recursively
canonicalized) subtree signature. This key never reads any body/joint
NAME, so it is invariant to renaming and to the order derivation steps were
originally applied in.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

from .kinematics import AffineCoupling, Body, Frame, KinematicModel, Joint, Pose

_ROUND = 9


def _r(x: float) -> float:
    return round(float(x), _ROUND)


def _rtuple(xs) -> Tuple[float, ...]:
    return tuple(_r(v) for v in xs)


def _ancestor_paths(model: KinematicModel) -> Dict[str, List[str]]:
    """old body name -> list of OLD joint names from root to that body
    (empty for the root itself)."""
    children: Dict[str, List[Joint]] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)
    paths: Dict[str, List[str]] = {model.root: []}
    stack = [model.root]
    while stack:
        p = stack.pop()
        for j in children.get(p, []):
            paths[j.child] = paths[p] + [j.name]
            stack.append(j.child)
    return paths


def _coupling_hop_info(model: KinematicModel) -> Dict[str, Tuple[int, float, float]]:
    """joint name (dependent) -> (hop_count, multiplier, offset), where
    ``hop_count`` is how many joints up this joint's own ancestor path its
    coupling source sits at (the source is always an ancestor -- see
    ``rules.py``'s coupling-source rule). Uses OLD names only to look up
    positions; the returned ``hop_count`` itself is name-independent."""
    paths = _ancestor_paths(model)
    joint_by_name = {j.name: j for j in model.joints}
    out: Dict[str, Tuple[int, float, float]] = {}
    for c in model.couplings:
        dep_joint = joint_by_name[c.dependent]
        full_path = paths[dep_joint.child]  # ends with c.dependent itself
        idx_j = len(full_path) - 1
        idx_source = full_path.index(c.source)
        out[c.dependent] = (idx_j - idx_source, c.multiplier, c.offset)
    return out


def _sort_repr(key: Any) -> Any:
    """Wrap a heterogeneous key element so every candidate's key becomes
    structurally comparable (Python cannot compare e.g. ``None`` to a
    ``float``): tags each element with a type-ordinal before its value."""
    if key is None:
        return (0,)
    if isinstance(key, bool):
        return (1, key)
    if isinstance(key, (int, float)):
        return (2, key)
    if isinstance(key, str):
        return (3, key)
    if isinstance(key, tuple):
        return (4, tuple(_sort_repr(v) for v in key))
    return (5, repr(key))


def canonical_form(model: KinematicModel) -> KinematicModel:
    children: Dict[str, List[Joint]] = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)
    bodies_by_name = {b.name: b for b in model.bodies}
    frames_by_body: Dict[str, List[Frame]] = {}
    for f in model.frames:
        frames_by_body.setdefault(f.body, []).append(f)
    hop_info = _coupling_hop_info(model)

    signature_cache: Dict[str, Any] = {}
    order_cache: Dict[str, List[Joint]] = {}

    def _signature(body: str) -> Any:
        if body in signature_cache:
            return signature_cache[body]
        b = bodies_by_name[body]
        entries = []
        for j in children.get(body, []):
            child_sig = _signature(j.child)
            hop, mult, off = hop_info.get(j.name, (-1, 0.0, 0.0))
            child_body = bodies_by_name[j.child]
            key = (
                j.type,
                _rtuple(j.origin.xyz),
                _rtuple(j.origin.rpy),
                _rtuple(j.axis),
                _rtuple(j.limits) if j.limits is not None else (),
                j.limits is None,
                j.name in hop_info,
                hop, _r(mult), _r(off),
                bool(child_body.palm),
                _r(child_body.radius) if child_body.radius is not None else None,
                child_sig,
            )
            entries.append((key, j))
        entries.sort(key=lambda e: _sort_repr(e[0]))
        order_cache[body] = [j for _, j in entries]
        own_frame_keys = sorted(
            _rtuple(f.pose.xyz) + _rtuple(f.pose.rpy) for f in frames_by_body.get(body, [])
        )
        sig = (
            bool(b.palm),
            _r(b.radius) if b.radius is not None else None,
            tuple(own_frame_keys),
            tuple(k for k, _ in entries),
        )
        signature_cache[body] = sig
        return sig

    _signature(model.root)  # populates order_cache/signature_cache for every body

    new_bodies: List[Body] = []
    new_joints: List[Joint] = []
    new_frames: List[Frame] = []
    joint_name_map: Dict[str, str] = {}
    body_counter = [1]
    joint_counter = [0]

    def _emit_frames(old_body: str, new_body: str) -> None:
        frs = sorted(
            frames_by_body.get(old_body, []),
            key=lambda f: _rtuple(f.pose.xyz) + _rtuple(f.pose.rpy),
        )
        for k, f in enumerate(frs):
            suffix = "_tip" if f.name == f"{old_body}_tip" else f"_frame{k}"
            new_frames.append(Frame(name=f"{new_body}{suffix}", body=new_body, pose=f.pose))

    def _walk(old_body: str, new_body: str) -> None:
        _emit_frames(old_body, new_body)
        for j in order_cache.get(old_body, []):
            new_joint_name = f"j{joint_counter[0]}"
            joint_counter[0] += 1
            new_child_name = f"b{body_counter[0]}"
            body_counter[0] += 1
            joint_name_map[j.name] = new_joint_name
            child_body = bodies_by_name[j.child]
            new_bodies.append(Body(name=new_child_name, palm=child_body.palm, radius=child_body.radius))
            new_joints.append(Joint(
                name=new_joint_name, type=j.type, parent=new_body, child=new_child_name,
                origin=j.origin, axis=j.axis, limits=j.limits,
            ))
            _walk(j.child, new_child_name)

    root_body = bodies_by_name[model.root]
    new_bodies.append(Body(name="b0", palm=root_body.palm, radius=root_body.radius))
    _walk(model.root, "b0")

    new_couplings = tuple(
        AffineCoupling(
            dependent=joint_name_map[c.dependent], source=joint_name_map[c.source],
            multiplier=c.multiplier, offset=c.offset,
        )
        for c in sorted(model.couplings, key=lambda c: joint_name_map[c.dependent])
    )

    return KinematicModel(
        name="canonical", root="b0", bodies=tuple(new_bodies), joints=tuple(new_joints),
        frames=tuple(new_frames), couplings=new_couplings,
    )


def _model_to_canonical_dict(model: KinematicModel) -> Dict[str, Any]:
    def pose_d(p: Pose) -> Dict[str, Any]:
        return {"xyz": list(_rtuple(p.xyz)), "rpy": list(_rtuple(p.rpy))}

    return {
        "root": model.root,
        "bodies": [
            {"name": b.name, "palm": bool(b.palm), "radius": _r(b.radius) if b.radius is not None else None}
            for b in sorted(model.bodies, key=lambda b: b.name)
        ],
        "joints": [
            {
                "name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
                "origin": pose_d(j.origin), "axis": list(_rtuple(j.axis)),
                "limits": list(_rtuple(j.limits)) if j.limits is not None else None,
            }
            for j in sorted(model.joints, key=lambda j: j.name)
        ],
        "frames": [
            {"name": f.name, "body": f.body, "pose": pose_d(f.pose)}
            for f in sorted(model.frames, key=lambda f: f.name)
        ],
        "couplings": [
            {"dependent": c.dependent, "source": c.source, "multiplier": _r(c.multiplier), "offset": _r(c.offset)}
            for c in sorted(model.couplings, key=lambda c: c.dependent)
        ],
    }


def phenotype_hash(model: KinematicModel) -> str:
    """sha256 hex digest of ``canonical_form(model)``'s JSON, floats rounded
    to 1e-9. Two models with identical derived geometry (however they were
    derived, and regardless of any body/joint renaming) hash identically."""
    canon = canonical_form(model)
    d = _model_to_canonical_dict(canon)
    text = json.dumps(d, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
