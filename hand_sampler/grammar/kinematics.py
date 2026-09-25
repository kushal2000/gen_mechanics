"""Frozen, data-only kinematic model for the hand grammar (iteration 1).

All dataclasses here are immutable (``frozen=True``) and hold only plain
Python data (tuples, strings, floats) -- never numpy arrays. Computation
(forward kinematics, sampling) lives in sibling modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

MOVABLE_TYPES = ("revolute", "continuous", "prismatic")
ALL_TYPES = ("fixed",) + MOVABLE_TYPES


class ModelError(Exception):
    """Raised when a model fails validation. ``issues`` lists every problem found."""

    def __init__(self, issues):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues) if self.issues else "model error")


class UnsupportedConstruct(ModelError):
    """Raised when a source file contains a construct this grammar cannot represent
    (e.g. a ``floating``, ``planar`` or ``ball`` URDF joint). Never silently dropped."""


@dataclass(frozen=True)
class Pose:
    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Body:
    name: str


@dataclass(frozen=True)
class Frame:
    name: str
    body: str
    pose: Pose = field(default_factory=Pose)


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: Pose = field(default_factory=Pose)
    axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    limits: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class AffineCoupling:
    dependent: str
    source: str
    multiplier: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class JointGroup:
    name: str
    kind: str
    joints: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Approximation:
    target: str
    physical: str
    reduction: str
    citation: str


@dataclass(frozen=True)
class LoopClosure:
    name: str
    frame_a: str
    frame_b: str
    kind: str
    assembly_ref: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KinematicModel:
    name: str
    root: str
    bodies: Tuple[Body, ...]
    joints: Tuple[Joint, ...]
    frames: Tuple[Frame, ...] = ()
    couplings: Tuple[AffineCoupling, ...] = ()
    closures: Tuple[LoopClosure, ...] = ()
    groups: Tuple[JointGroup, ...] = ()
    approximations: Tuple[Approximation, ...] = ()
    independent: Tuple[str, ...] = ()


def _is_finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate(model: KinematicModel) -> None:
    """Validate ``model``. Raises ``ModelError`` (or ``UnsupportedConstruct``) with a
    non-empty ``issues`` list on failure; returns ``None`` on success."""

    issues = []

    body_names = [b.name for b in model.bodies]
    joint_names = [j.name for j in model.joints]
    frame_names = [f.name for f in model.frames]

    # Unique names within each namespace.
    for label, names in (("body", body_names), ("joint", joint_names), ("frame", frame_names)):
        seen = set()
        for n in names:
            if n in seen:
                issues.append(f"duplicate {label} name: {n!r}")
            seen.add(n)

    body_set = set(body_names)

    # Every joint parent/child is a body.
    for j in model.joints:
        if j.parent not in body_set:
            issues.append(f"joint {j.name!r} parent {j.parent!r} is not a known body")
        if j.child not in body_set:
            issues.append(f"joint {j.name!r} child {j.child!r} is not a known body")
        if j.type not in ALL_TYPES:
            issues.append(f"joint {j.name!r} has unknown type {j.type!r}")

    # Frames reference known bodies.
    for f in model.frames:
        if f.body not in body_set:
            issues.append(f"frame {f.name!r} body {f.body!r} is not a known body")

    # Tree structure: each non-root body has exactly one parent joint; no cycles.
    parent_joint_of = {}
    for j in model.joints:
        if j.child in body_set:
            if j.child in parent_joint_of:
                issues.append(f"body {j.child!r} has more than one parent joint")
            parent_joint_of[j.child] = j.name

    roots = [b for b in body_names if b not in parent_joint_of]
    if len(roots) != 1:
        issues.append(f"expected exactly one root body, found {roots!r}")
    elif model.root not in roots:
        issues.append(f"declared root {model.root!r} does not match tree root {roots!r}")

    # Cycle check via child->parent walk.
    child_to_parent_body = {}
    for j in model.joints:
        if j.child in body_set and j.parent in body_set:
            child_to_parent_body[j.child] = j.parent
    for start in body_names:
        seen = set()
        cur = start
        steps = 0
        while cur in child_to_parent_body:
            if cur in seen:
                issues.append(f"cycle detected in body tree involving {start!r}")
                break
            seen.add(cur)
            cur = child_to_parent_body[cur]
            steps += 1
            if steps > len(body_names) + 1:
                issues.append(f"cycle detected in body tree involving {start!r}")
                break

    # Coupling checks.
    movable = {j.name for j in model.joints if j.type in MOVABLE_TYPES}
    joint_by_name = {j.name: j for j in model.joints}
    dependents = set()
    for c in model.couplings:
        if c.dependent not in movable:
            issues.append(f"coupling dependent {c.dependent!r} is not a movable joint")
        if c.source not in movable:
            issues.append(f"coupling source {c.source!r} is not a movable joint")
        if c.dependent in dependents:
            issues.append(f"joint {c.dependent!r} has more than one coupling")
        dependents.add(c.dependent)
        if not (_is_finite(c.multiplier) and _is_finite(c.offset)):
            issues.append(f"coupling for {c.dependent!r} has non-finite multiplier/offset")

    # No coupling cycles (dependent -> source is a DAG).
    source_of = {c.dependent: c.source for c in model.couplings}
    for start in dependents:
        seen = set()
        cur = start
        steps = 0
        while cur in source_of:
            if cur in seen:
                issues.append(f"coupling cycle detected involving {start!r}")
                break
            seen.add(cur)
            cur = source_of[cur]
            steps += 1
            if steps > len(source_of) + 1:
                issues.append(f"coupling cycle detected involving {start!r}")
                break

    # independent = movable - dependents, computed if empty, checked if given.
    computed_independent = tuple(j for j in joint_names if j in movable and j not in dependents)
    if model.independent:
        if set(model.independent) != set(computed_independent):
            issues.append(
                "declared independent joints "
                f"{sorted(model.independent)!r} do not match movable-minus-dependents "
                f"{sorted(computed_independent)!r}"
            )

    # Limits: ordered, revolute/prismatic require them, continuous must have none.
    for j in model.joints:
        if j.type in ("revolute", "prismatic"):
            if j.limits is None:
                issues.append(f"joint {j.name!r} of type {j.type!r} requires limits")
            else:
                lo, hi = j.limits
                if not (_is_finite(lo) and _is_finite(hi)):
                    issues.append(f"joint {j.name!r} has non-finite limits {j.limits!r}")
                elif lo > hi:
                    issues.append(f"joint {j.name!r} has inverted limits {j.limits!r}")
        elif j.type == "continuous":
            if j.limits is not None:
                issues.append(f"continuous joint {j.name!r} must not declare limits")

    # Finite floats everywhere.
    def check_pose(label, pose):
        for v in tuple(pose.xyz) + tuple(pose.rpy):
            if not _is_finite(v):
                issues.append(f"{label} has non-finite pose value {v!r}")

    for j in model.joints:
        check_pose(f"joint {j.name!r} origin", j.origin)
        for v in j.axis:
            if not _is_finite(v):
                issues.append(f"joint {j.name!r} has non-finite axis {j.axis!r}")
    for f in model.frames:
        check_pose(f"frame {f.name!r} pose", f.pose)

    if issues:
        raise ModelError(issues)
