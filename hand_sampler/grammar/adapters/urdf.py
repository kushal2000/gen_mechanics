"""URDF importer: parses with ``xml.etree.ElementTree`` into a ``KinematicModel``.

Stdlib + numpy only (imported indirectly via ``kinematics``/``coords``, but this
module itself needs neither). Unsupported joint types (``floating``, ``planar``,
``ball``, or anything not in {fixed, revolute, continuous, prismatic}) raise
``UnsupportedConstruct`` naming the joint -- never silently skipped or expanded.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from ..kinematics import AffineCoupling, Body, Joint, KinematicModel, Pose, UnsupportedConstruct, validate

SUPPORTED_JOINT_TYPES = ("fixed", "revolute", "continuous", "prismatic")


@dataclass(frozen=True)
class LossReport:
    visuals: Tuple[str, ...] = ()
    collisions: Tuple[str, ...] = ()
    materials: Tuple[str, ...] = ()
    gazebo_transmission: Tuple[str, ...] = ()
    dropped_above_hand_root: Tuple[str, ...] = ()
    non_unit_axes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportResult:
    model: KinematicModel
    actuation: Dict[str, dict]
    inertial: Dict[str, dict]
    losses: LossReport


def _parse_root(src: Union[str, Path, bytes]) -> ET.Element:
    if isinstance(src, Path):
        data = src.read_bytes()
        return ET.fromstring(data)
    if isinstance(src, bytes):
        return ET.fromstring(src)
    if isinstance(src, str):
        stripped = src.lstrip()
        path = Path(src)
        if not stripped.startswith("<") and path.exists():
            return ET.fromstring(path.read_bytes())
        return ET.fromstring(src.encode("utf-8"))
    raise TypeError(f"unsupported src type for load_urdf: {type(src)!r}")


def _floats(text: Optional[str], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if text is None or text.strip() == "":
        return default
    return tuple(float(v) for v in text.split())


def load_urdf(src: Union[str, Path, bytes], *, hand_root: Optional[str] = None) -> ImportResult:
    root = _parse_root(src)
    if root.tag != "robot":
        raise ValueError(f"expected a <robot> root element, got <{root.tag}>")

    link_els = {el.attrib["name"]: el for el in root.findall("link")}
    link_order = list(link_els.keys())

    loss_visuals: List[str] = []
    loss_collisions: List[str] = []
    loss_materials: List[str] = []
    loss_gazebo_transmission: List[str] = []
    loss_nonunit_axes: List[str] = []

    inertial: Dict[str, dict] = {}
    for lname, lel in link_els.items():
        if lel.find("visual") is not None:
            loss_visuals.append(lname)
            for vis in lel.findall("visual"):
                if vis.find("material") is not None:
                    loss_materials.append(lname)
        if lel.find("collision") is not None:
            loss_collisions.append(lname)
        inertial_el = lel.find("inertial")
        if inertial_el is not None:
            entry = {}
            mass_el = inertial_el.find("mass")
            if mass_el is not None and "value" in mass_el.attrib:
                entry["mass"] = float(mass_el.attrib["value"])
            origin_el = inertial_el.find("origin")
            if origin_el is not None:
                entry["origin_xyz"] = _floats(origin_el.attrib.get("xyz"), (0.0, 0.0, 0.0))
                entry["origin_rpy"] = _floats(origin_el.attrib.get("rpy"), (0.0, 0.0, 0.0))
            inertia_el = inertial_el.find("inertia")
            if inertia_el is not None:
                for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
                    if k in inertia_el.attrib:
                        entry[k] = float(inertia_el.attrib[k])
            if entry:
                inertial[lname] = entry

    for tag in ("gazebo", "transmission"):
        for el in root.findall(tag):
            loss_gazebo_transmission.append(el.attrib.get("name", tag))

    joints: List[Joint] = []
    couplings: List[AffineCoupling] = []
    actuation: Dict[str, dict] = {}

    for jel in root.findall("joint"):
        jname = jel.attrib["name"]
        jtype = jel.attrib["type"]
        if jtype not in SUPPORTED_JOINT_TYPES:
            raise UnsupportedConstruct([f"joint {jname!r} has unsupported type {jtype!r}"])

        parent_el = jel.find("parent")
        child_el = jel.find("child")
        if parent_el is None or child_el is None:
            raise ValueError(f"joint {jname!r} is missing <parent> or <child>")
        parent = parent_el.attrib["link"]
        child = child_el.attrib["link"]

        origin_el = jel.find("origin")
        if origin_el is not None:
            xyz = _floats(origin_el.attrib.get("xyz"), (0.0, 0.0, 0.0))
            rpy = _floats(origin_el.attrib.get("rpy"), (0.0, 0.0, 0.0))
        else:
            xyz, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        origin = Pose(xyz=xyz, rpy=rpy)

        axis_el = jel.find("axis")
        if axis_el is not None:
            axis = _floats(axis_el.attrib.get("xyz"), (1.0, 0.0, 0.0))
        else:
            axis = (1.0, 0.0, 0.0)
        anorm = math.sqrt(sum(v * v for v in axis))
        if anorm != 0.0 and abs(anorm - 1.0) > 1e-9:
            loss_nonunit_axes.append(jname)

        limits = None
        limit_el = jel.find("limit")
        act: dict = {}
        if limit_el is not None:
            lo = limit_el.attrib.get("lower")
            hi = limit_el.attrib.get("upper")
            if lo is not None and hi is not None:
                limits = (float(lo), float(hi))
            if "effort" in limit_el.attrib:
                act["effort"] = float(limit_el.attrib["effort"])
            if "velocity" in limit_el.attrib:
                act["velocity"] = float(limit_el.attrib["velocity"])
        dyn_el = jel.find("dynamics")
        if dyn_el is not None:
            if "damping" in dyn_el.attrib:
                act["damping"] = float(dyn_el.attrib["damping"])
            if "friction" in dyn_el.attrib:
                act["friction"] = float(dyn_el.attrib["friction"])
        if act:
            actuation[jname] = act

        joints.append(
            Joint(name=jname, type=jtype, parent=parent, child=child, origin=origin, axis=axis, limits=limits)
        )

        mimic_el = jel.find("mimic")
        if mimic_el is not None:
            source = mimic_el.attrib["joint"]
            multiplier = float(mimic_el.attrib.get("multiplier", 1.0))
            offset = float(mimic_el.attrib.get("offset", 0.0))
            couplings.append(AffineCoupling(dependent=jname, source=source, multiplier=multiplier, offset=offset))

    dropped_above_hand_root: List[str] = []

    if hand_root is not None:
        if hand_root not in link_els:
            raise ValueError(f"hand_root {hand_root!r} not found among <link> elements")
        children_map: Dict[str, List[Joint]] = {}
        for j in joints:
            children_map.setdefault(j.parent, []).append(j)
        keep_bodies = set()
        keep_joints: List[Joint] = []
        stack = [hand_root]
        while stack:
            b = stack.pop()
            if b in keep_bodies:
                continue
            keep_bodies.add(b)
            for j in children_map.get(b, []):
                keep_joints.append(j)
                stack.append(j.child)
        dropped_above_hand_root = [b for b in link_order if b not in keep_bodies]
        joints = keep_joints
        kept_joint_names = {j.name for j in joints}
        couplings = [c for c in couplings if c.dependent in kept_joint_names and c.source in kept_joint_names]
        kept_link_order = [b for b in link_order if b in keep_bodies]
        model_root = hand_root
    else:
        child_names = {j.child for j in joints}
        root_candidates = [b for b in link_order if b not in child_names]
        if len(root_candidates) == 1:
            model_root = root_candidates[0]
        elif root_candidates:
            model_root = root_candidates[0]
        else:
            model_root = link_order[0] if link_order else ""
        kept_link_order = link_order

    bodies = tuple(Body(name=n) for n in kept_link_order)

    model = KinematicModel(
        name=root.attrib.get("name", "urdf"),
        root=model_root,
        bodies=bodies,
        joints=tuple(joints),
        couplings=tuple(couplings),
    )
    validate(model)

    losses = LossReport(
        visuals=tuple(loss_visuals),
        collisions=tuple(loss_collisions),
        materials=tuple(loss_materials),
        gazebo_transmission=tuple(loss_gazebo_transmission),
        dropped_above_hand_root=tuple(dropped_above_hand_root),
        non_unit_axes=tuple(loss_nonunit_axes),
    )

    return ImportResult(model=model, actuation=actuation, inertial=inertial, losses=losses)
