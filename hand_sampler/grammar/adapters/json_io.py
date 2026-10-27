"""JSON serialization for ``KinematicModel`` (iteration 2).

Stdlib only. Schema id ``"hand_kinematics/0.1"``. Key order is fixed per
record type (never ``sort_keys``) so ``to_json`` is byte-deterministic for a
given model. Floats round-trip exactly: Python's ``json`` module already
encodes finite floats via ``float.__repr__`` (the shortest string that
round-trips), which is exactly what this module relies on for byte-identical
``to_json(from_json(s)) == s``. ``allow_nan=False`` makes any non-finite
float a hard error instead of emitting non-standard ``NaN``/``Infinity``
tokens.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from ..kinematics import (
    AffineCoupling,
    Approximation,
    Body,
    Frame,
    Joint,
    JointGroup,
    KinematicModel,
    LoopClosure,
    Pose,
)

SCHEMA = "hand_kinematics/0.1"


def _pose_to_dict(pose: Pose) -> Dict[str, Any]:
    return {"xyz": list(pose.xyz), "rpy": list(pose.rpy)}


def _pose_from_dict(d: Dict[str, Any]) -> Pose:
    return Pose(xyz=tuple(d["xyz"]), rpy=tuple(d["rpy"]))


def _body_to_dict(b: Body) -> Dict[str, Any]:
    return {"name": b.name, "palm": b.palm, "radius": b.radius}


def _body_from_dict(d: Dict[str, Any]) -> Body:
    # ``.get`` with defaults: pre-iteration-3 JSON (no "palm"/"radius" keys)
    # still loads, with both fields taking their ``Body`` defaults.
    return Body(name=d["name"], palm=d.get("palm", False), radius=d.get("radius"))


def _frame_to_dict(f: Frame) -> Dict[str, Any]:
    return {"name": f.name, "body": f.body, "pose": _pose_to_dict(f.pose)}


def _frame_from_dict(d: Dict[str, Any]) -> Frame:
    return Frame(name=d["name"], body=d["body"], pose=_pose_from_dict(d["pose"]))


def _joint_to_dict(j: Joint) -> Dict[str, Any]:
    return {
        "name": j.name,
        "type": j.type,
        "parent": j.parent,
        "child": j.child,
        "origin": _pose_to_dict(j.origin),
        "axis": list(j.axis),
        "limits": None if j.limits is None else list(j.limits),
    }


def _joint_from_dict(d: Dict[str, Any]) -> Joint:
    limits: Optional[Tuple[float, float]] = None if d["limits"] is None else tuple(d["limits"])
    return Joint(
        name=d["name"],
        type=d["type"],
        parent=d["parent"],
        child=d["child"],
        origin=_pose_from_dict(d["origin"]),
        axis=tuple(d["axis"]),
        limits=limits,
    )


def _coupling_to_dict(c: AffineCoupling) -> Dict[str, Any]:
    return {"dependent": c.dependent, "source": c.source, "multiplier": c.multiplier, "offset": c.offset}


def _coupling_from_dict(d: Dict[str, Any]) -> AffineCoupling:
    return AffineCoupling(dependent=d["dependent"], source=d["source"], multiplier=d["multiplier"], offset=d["offset"])


def _closure_to_dict(cl: LoopClosure) -> Dict[str, Any]:
    return {
        "name": cl.name,
        "frame_a": cl.frame_a,
        "frame_b": cl.frame_b,
        "kind": cl.kind,
        "assembly_ref": list(cl.assembly_ref),
    }


def _closure_from_dict(d: Dict[str, Any]) -> LoopClosure:
    return LoopClosure(
        name=d["name"],
        frame_a=d["frame_a"],
        frame_b=d["frame_b"],
        kind=d["kind"],
        assembly_ref=tuple(d["assembly_ref"]),
    )


def _group_to_dict(g: JointGroup) -> Dict[str, Any]:
    return {"name": g.name, "kind": g.kind, "joints": list(g.joints)}


def _group_from_dict(d: Dict[str, Any]) -> JointGroup:
    return JointGroup(name=d["name"], kind=d["kind"], joints=tuple(d["joints"]))


def _approx_to_dict(a: Approximation) -> Dict[str, Any]:
    return {"target": a.target, "physical": a.physical, "reduction": a.reduction, "citation": a.citation}


def _approx_from_dict(d: Dict[str, Any]) -> Approximation:
    return Approximation(target=d["target"], physical=d["physical"], reduction=d["reduction"], citation=d["citation"])


def model_to_dict(model: KinematicModel) -> Dict[str, Any]:
    """Convert ``model`` to a plain dict with a fixed, deterministic key order."""
    return {
        "schema": SCHEMA,
        "name": model.name,
        "root": model.root,
        "bodies": [_body_to_dict(b) for b in model.bodies],
        "joints": [_joint_to_dict(j) for j in model.joints],
        "frames": [_frame_to_dict(f) for f in model.frames],
        "couplings": [_coupling_to_dict(c) for c in model.couplings],
        "closures": [_closure_to_dict(cl) for cl in model.closures],
        "groups": [_group_to_dict(g) for g in model.groups],
        "approximations": [_approx_to_dict(a) for a in model.approximations],
        "independent": list(model.independent),
    }


def model_from_dict(d: Dict[str, Any]) -> KinematicModel:
    if d.get("schema") != SCHEMA:
        raise ValueError(f"unsupported schema {d.get('schema')!r}, expected {SCHEMA!r}")
    return KinematicModel(
        name=d["name"],
        root=d["root"],
        bodies=tuple(_body_from_dict(b) for b in d["bodies"]),
        joints=tuple(_joint_from_dict(j) for j in d["joints"]),
        frames=tuple(_frame_from_dict(f) for f in d["frames"]),
        couplings=tuple(_coupling_from_dict(c) for c in d["couplings"]),
        closures=tuple(_closure_from_dict(cl) for cl in d["closures"]),
        groups=tuple(_group_from_dict(g) for g in d["groups"]),
        approximations=tuple(_approx_from_dict(a) for a in d["approximations"]),
        independent=tuple(d["independent"]),
    )


def to_json(model: KinematicModel) -> str:
    return json.dumps(model_to_dict(model), allow_nan=False)


def from_json(text: str) -> KinematicModel:
    return model_from_dict(json.loads(text))
