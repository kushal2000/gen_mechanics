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
    # import: couplings dropped because ``hand_root`` cut their source or
    # dependent joint out of the kept subtree (issue I2). Never silent.
    couplings_cut: Tuple[str, ...] = ()
    # export (to_urdf): LoopClosures cannot be expressed in URDF; every
    # dropped closure is named here (and in an XML comment in the output).
    closures_dropped: Tuple[str, ...] = ()
    # export (to_urdf): joints whose <limit effort velocity> came from the
    # documented placeholder because no actuation overlay entry was given.
    placeholder_actuation: Tuple[str, ...] = ()


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
    cut_couplings: List[str] = []

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
        kept_couplings = []
        cut_couplings = []
        for c in couplings:
            if c.dependent in kept_joint_names and c.source in kept_joint_names:
                kept_couplings.append(c)
            else:
                cut_couplings.append(c.dependent)
        couplings = kept_couplings
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
        couplings_cut=tuple(cut_couplings),
        non_unit_axes=tuple(loss_nonunit_axes),
    )

    return ImportResult(model=model, actuation=actuation, inertial=inertial, losses=losses)


# --- Export -----------------------------------------------------------------

# Documented placeholder for <limit effort="..." velocity="..."> when no
# actuation overlay entry supplies a real value. URDF requires both
# attributes on revolute/prismatic joints; 1.0 is a syntactically valid,
# physically inert placeholder (never 0, which some consumers treat as
# "unactuatable"). Every joint that used it is named in
# ``LossReport.placeholder_actuation``.
PLACEHOLDER_EFFORT = 1.0
PLACEHOLDER_VELOCITY = 1.0

# Fixed joints synthesized to attach a Frame's zero-geometry link get this
# name suffix so they cannot collide with a real joint of the same name as
# the frame.
FRAME_JOINT_SUFFIX = "__frame_fixed"


def _floats_to_attr(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def to_urdf(
    model: KinematicModel,
    *,
    actuation: Optional[Dict[str, dict]] = None,
    inertial: Optional[Dict[str, dict]] = None,
) -> Tuple[str, LossReport]:
    """Serialize ``model`` to URDF text. Returns ``(text, LossReport)``.

    ``actuation``/``inertial`` use the same per-joint/per-body dict schema as
    ``ImportResult.actuation``/``ImportResult.inertial`` (effort, velocity,
    damping, friction / mass, origin_xyz, origin_rpy, ixx..izz).

    Bodies, joints (in model order, with ``<origin>``/``<axis>`` written via
    ``repr`` so re-parsing recovers the exact stored floats), limits and
    ``<mimic>`` tuples round-trip exactly through ``load_urdf``. ``Frame``
    objects are not a URDF concept: each is exported as a zero-geometry
    ``<link>`` named after the frame, attached to its body by a fixed joint
    named ``"<frame>__frame_fixed"``. ``LoopClosure`` objects cannot be
    expressed in URDF at all: they are omitted from the XML, listed in an
    ``<!-- -->`` comment, and named in ``LossReport.closures_dropped``.
    """
    actuation = actuation or {}
    inertial = inertial or {}

    robot = ET.Element("robot", {"name": model.name})

    for b in model.bodies:
        link_el = ET.SubElement(robot, "link", {"name": b.name})
        entry = inertial.get(b.name)
        if entry:
            inertial_el = ET.SubElement(link_el, "inertial")
            if "mass" in entry:
                ET.SubElement(inertial_el, "mass", {"value": repr(float(entry["mass"]))})
            if "origin_xyz" in entry or "origin_rpy" in entry:
                ET.SubElement(
                    inertial_el,
                    "origin",
                    {
                        "xyz": _floats_to_attr(entry.get("origin_xyz", (0.0, 0.0, 0.0))),
                        "rpy": _floats_to_attr(entry.get("origin_rpy", (0.0, 0.0, 0.0))),
                    },
                )
            inertia_keys = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
            if any(k in entry for k in inertia_keys):
                ET.SubElement(
                    inertial_el,
                    "inertia",
                    {k: repr(float(entry[k])) for k in inertia_keys if k in entry},
                )

    for f in model.frames:
        ET.SubElement(robot, "link", {"name": f.name})

    placeholder_actuation: List[str] = []

    for j in model.joints:
        jel = ET.SubElement(robot, "joint", {"name": j.name, "type": j.type})
        ET.SubElement(jel, "parent", {"link": j.parent})
        ET.SubElement(jel, "child", {"link": j.child})
        ET.SubElement(
            jel,
            "origin",
            {"xyz": _floats_to_attr(j.origin.xyz), "rpy": _floats_to_attr(j.origin.rpy)},
        )
        ET.SubElement(jel, "axis", {"xyz": _floats_to_attr(j.axis)})

        act = actuation.get(j.name, {})
        if j.type in ("revolute", "prismatic"):
            lo, hi = j.limits
            has_effort = "effort" in act
            has_velocity = "velocity" in act
            effort = act["effort"] if has_effort else PLACEHOLDER_EFFORT
            velocity = act["velocity"] if has_velocity else PLACEHOLDER_VELOCITY
            if not (has_effort and has_velocity):
                placeholder_actuation.append(j.name)
            ET.SubElement(
                jel,
                "limit",
                {
                    "lower": repr(float(lo)),
                    "upper": repr(float(hi)),
                    "effort": repr(float(effort)),
                    "velocity": repr(float(velocity)),
                },
            )
        elif j.type == "continuous" and ("effort" in act or "velocity" in act):
            has_effort = "effort" in act
            has_velocity = "velocity" in act
            effort = act.get("effort", PLACEHOLDER_EFFORT)
            velocity = act.get("velocity", PLACEHOLDER_VELOCITY)
            if not (has_effort and has_velocity):
                placeholder_actuation.append(j.name)
            ET.SubElement(
                jel,
                "limit",
                {"effort": repr(float(effort)), "velocity": repr(float(velocity))},
            )

        if "damping" in act or "friction" in act:
            dyn_attrib = {}
            if "damping" in act:
                dyn_attrib["damping"] = repr(float(act["damping"]))
            if "friction" in act:
                dyn_attrib["friction"] = repr(float(act["friction"]))
            ET.SubElement(jel, "dynamics", dyn_attrib)

    for f in model.frames:
        jel = ET.SubElement(robot, "joint", {"name": f"{f.name}{FRAME_JOINT_SUFFIX}", "type": "fixed"})
        ET.SubElement(jel, "parent", {"link": f.body})
        ET.SubElement(jel, "child", {"link": f.name})
        ET.SubElement(
            jel,
            "origin",
            {"xyz": _floats_to_attr(f.pose.xyz), "rpy": _floats_to_attr(f.pose.rpy)},
        )

    coupling_by_dependent = {c.dependent: c for c in model.couplings}
    for jel in robot.findall("joint"):
        c = coupling_by_dependent.get(jel.attrib["name"])
        if c is not None:
            ET.SubElement(
                jel,
                "mimic",
                {"joint": c.source, "multiplier": repr(float(c.multiplier)), "offset": repr(float(c.offset))},
            )

    closures_dropped = tuple(cl.name for cl in model.closures)

    xml_text = ET.tostring(robot, encoding="unicode")
    if model.closures:
        comment_lines = []
        for cl in model.closures:
            comment_lines.append(
                f"UNREPRESENTABLE LoopClosure name={cl.name!r} frame_a={cl.frame_a!r} "
                f"frame_b={cl.frame_b!r} kind={cl.kind!r} assembly_ref={list(cl.assembly_ref)!r}"
            )
        comment_block = "\n".join(f"<!-- {line} -->" for line in comment_lines)
        xml_text = xml_text.replace("</robot>", comment_block + "\n</robot>")

    text = '<?xml version="1.0"?>\n' + xml_text

    losses = LossReport(
        closures_dropped=closures_dropped,
        placeholder_actuation=tuple(placeholder_actuation),
    )
    return text, losses
