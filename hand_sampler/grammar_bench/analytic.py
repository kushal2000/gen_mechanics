"""Independent sympy-based closed-form FK for the analytic fixtures.

This module is deliberately independent of the ``hand_sampler.grammar``
implementation: it parses URDF text itself with ``xml.etree.ElementTree`` and
builds Rz*Ry*Rx / Rodrigues symbolically from the decimal strings in the file,
evaluated at 30 digits and rounded to float only at the very end. It must
never import ``hand_sampler.grammar``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Union

import numpy as np
import sympy as sp


def _svec(text, default):
    if text is None or text.strip() == "":
        return [sp.nsimplify(str(v)) for v in default]
    return [sp.nsimplify(v) for v in text.split()]


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = sp.cos(roll), sp.sin(roll)
    cp, sp_ = sp.cos(pitch), sp.sin(pitch)
    cy, sy = sp.cos(yaw), sp.sin(yaw)
    Rz = sp.Matrix([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = sp.Matrix([[cp, 0, sp_], [0, 1, 0], [-sp_, 0, cp]])
    Rx = sp.Matrix([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz * Ry * Rx


def _pose_matrix(xyz, rpy):
    T = sp.eye(4)
    T[:3, :3] = _rpy_matrix(rpy)
    for i in range(3):
        T[i, 3] = xyz[i]
    return T


def _rodrigues(axis, angle):
    n = sp.sqrt(sum(a * a for a in axis))
    ax = [a / n for a in axis]
    x, y, z = ax
    K = sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    I = sp.eye(3)
    return I + sp.sin(angle) * K + (1 - sp.cos(angle)) * (K * K)


class AnalyticJoint:
    __slots__ = ("name", "type", "parent", "child", "xyz", "rpy", "axis", "mimic")

    def __init__(self, name, type_, parent, child, xyz, rpy, axis, mimic):
        self.name = name
        self.type = type_
        self.parent = parent
        self.child = child
        self.xyz = xyz
        self.rpy = rpy
        self.axis = axis
        self.mimic = mimic  # (source, multiplier, offset) or None


class AnalyticModel:
    def __init__(self, name, root, links, joints):
        self.name = name
        self.root = root
        self.links = links
        self.joints = joints  # list[AnalyticJoint]


def parse_urdf(src: Union[str, Path]) -> AnalyticModel:
    path = Path(src)
    text = path.read_bytes() if path.exists() else str(src).encode("utf-8")
    root = ET.fromstring(text)
    links = [el.attrib["name"] for el in root.findall("link")]
    joints = []
    for jel in root.findall("joint"):
        name = jel.attrib["name"]
        jtype = jel.attrib["type"]
        parent = jel.find("parent").attrib["link"]
        child = jel.find("child").attrib["link"]
        origin_el = jel.find("origin")
        if origin_el is not None:
            xyz = _svec(origin_el.attrib.get("xyz"), (0, 0, 0))
            rpy = _svec(origin_el.attrib.get("rpy"), (0, 0, 0))
        else:
            xyz = _svec(None, (0, 0, 0))
            rpy = _svec(None, (0, 0, 0))
        axis_el = jel.find("axis")
        axis = _svec(axis_el.attrib.get("xyz") if axis_el is not None else None, (1, 0, 0))
        mimic_el = jel.find("mimic")
        mimic = None
        if mimic_el is not None:
            source = mimic_el.attrib["joint"]
            mult = sp.nsimplify(mimic_el.attrib.get("multiplier", "1"))
            off = sp.nsimplify(mimic_el.attrib.get("offset", "0"))
            mimic = (source, mult, off)
        joints.append(AnalyticJoint(name, jtype, parent, child, xyz, rpy, axis, mimic))

    child_names = {j.child for j in joints}
    root_candidates = [l for l in links if l not in child_names]
    root_link = root_candidates[0] if root_candidates else links[0]
    return AnalyticModel(root.attrib.get("name", "urdf"), root_link, links, joints)


def _expand_q(model: AnalyticModel, q: Mapping[str, float]) -> Dict[str, sp.Expr]:
    # Runtime configuration values are exact inputs, not URDF constants: use a
    # high-precision Float built from the float's exact binary value (never
    # nsimplify, which would snap the value to a "nicer" nearby symbolic form).
    qs = {name: sp.Float(val, 30) for name, val in q.items()}
    by_name = {j.name: j for j in model.joints}
    remaining = {j.name: j for j in model.joints if j.mimic is not None}
    progress = True
    while remaining and progress:
        progress = False
        for name, j in list(remaining.items()):
            source, mult, off = j.mimic
            if source in qs:
                qs[name] = mult * qs[source] + off
                del remaining[name]
                progress = True
    return qs


def forward_kinematics(model: AnalyticModel, q: Mapping[str, float]) -> Dict[str, np.ndarray]:
    """Return float 4x4 root-frame transforms for every body, evaluated at 30 digits."""
    qs = _expand_q(model, q)
    children = {}
    for j in model.joints:
        children.setdefault(j.parent, []).append(j)

    transforms_sym: Dict[str, sp.Matrix] = {model.root: sp.eye(4)}
    stack = [model.root]
    while stack:
        parent = stack.pop()
        Tp = transforms_sym[parent]
        for j in children.get(parent, []):
            qval = qs.get(j.name, sp.Integer(0))
            origin_T = _pose_matrix(j.xyz, j.rpy)
            motion_T = sp.eye(4)
            if j.type in ("revolute", "continuous"):
                motion_T[:3, :3] = _rodrigues(j.axis, qval)
            elif j.type == "prismatic":
                n = sp.sqrt(sum(a * a for a in j.axis))
                for i in range(3):
                    motion_T[i, 3] = (j.axis[i] / n) * qval
            Tc = Tp * origin_T * motion_T
            transforms_sym[j.child] = Tc
            stack.append(j.child)

    out: Dict[str, np.ndarray] = {}
    for name, Tm in transforms_sym.items():
        Tv = Tm.evalf(30)
        arr = np.zeros((4, 4), dtype=float)
        for i in range(4):
            for k in range(4):
                arr[i, k] = float(Tv[i, k])
        out[name] = arr
    return out
