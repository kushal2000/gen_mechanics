"""Joint-kind colours for a grammar hand's URDF visuals (renders and viewers only; collisions and physics untouched).

Each finger link f<i>_link<j> -- the body joint j moves -- takes its joint's kind colour: flexion blue, abduction
red, roll green (the audit figures' and the Isaac Sim renders' colours, experiments/04nov_audit_grammar); every other
link (the palm, the wrist) is grey. A roll tip's capsule is light green and gets visual-only strips round it, light to
dark green, so its spin shows (a capsule looks the same at every roll angle). Kit-free (xml + numpy + scipy).
"""
from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET

import numpy as np

KIND_RGBA = {(0, 0): "0.16 0.42 0.85 1", (90, 0): "0.86 0.20 0.18 1", (0, 90): "0.15 0.65 0.30 1"}
GREY_RGBA = "0.62 0.62 0.64 1"
ROLL_TIP_RGBA = "0.60 0.92 0.60 1"
STRIPES = 8


def _link_kind(name: str, design: dict):
    """(axis, is the finger's last joint) for a finger link f<i>_link<j>, else None."""
    if not name.startswith("f") or "_link" not in name:
        return None
    try:
        i, j = (int(v) for v in name[1:].split("_link"))
        joints = design["fingers"][i]["joints"]
        return tuple(joints[j]["axis"]), j == len(joints) - 1
    except (ValueError, IndexError, KeyError):
        return None


def _set_colour(vis: ET.Element, rgba: str, name: str) -> None:
    mat = vis.find("material")
    if mat is None:
        mat = ET.SubElement(vis, "material", name=name)
    col = mat.find("color")
    if col is None:
        col = ET.SubElement(mat, "color")
    col.set("rgba", rgba)


def colour_by_joint_kind(root: ET.Element, design: dict, stripes: bool = True, recolour_others: bool = True) -> None:
    """Recolour the URDF tree `root` in place (see the module docstring). `recolour_others` False leaves links that
    are not finger links (palm, wrist) as they are."""
    from scipy.spatial.transform import Rotation
    for link in root.findall("link"):
        kind = _link_kind(link.get("name", ""), design)
        if kind is None:
            if recolour_others:
                for vis in link.findall("visual"):
                    _set_colour(vis, GREY_RGBA, f"{link.get('name')}_grey")
            continue
        axis, last = kind
        roll_tip = last and axis == (0, 90)
        for vis in link.findall("visual"):
            is_tip = vis.find("geometry/cylinder") is not None
            _set_colour(vis, ROLL_TIP_RGBA if (roll_tip and is_tip) else KIND_RGBA.get(axis, GREY_RGBA),
                        f"{link.get('name')}_kind")
        if not (stripes and roll_tip):
            continue
        for vis in list(link.findall("visual")):
            cyl = vis.find("geometry/cylinder")
            if cyl is None:
                continue
            o = vis.find("origin")
            c = np.array([float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
            Rm = Rotation.from_euler("xyz", [float(v) for v in ((o.get("rpy") if o is not None else None)
                                                                 or "0 0 0").split()]).as_matrix()
            r, L = float(cyl.get("radius")), float(cyl.get("length"))
            for k in range(STRIPES):
                th = 2 * math.pi * k / STRIPES
                s = k / (STRIPES - 1)
                Rk = Rm @ Rotation.from_euler("z", th).as_matrix()
                pos = c + Rm @ np.array([(r + 0.0006) * math.cos(th), (r + 0.0006) * math.sin(th), 0.0])
                v = ET.SubElement(link, "visual", name=f"{link.get('name')}_rolltip_stripe{k}")
                with warnings.catch_warnings():      # gimbal lock at 90 deg: scipy returns an equivalent set
                    warnings.simplefilter("ignore", UserWarning)
                    rpy = Rotation.from_matrix(Rk).as_euler("xyz")
                ET.SubElement(v, "origin", xyz=" ".join(f"{x:.9g}" for x in pos), rpy=" ".join(f"{x:.9g}" for x in rpy))
                ET.SubElement(ET.SubElement(v, "geometry"), "box",
                              size=f"0.0012 {2 * r * math.sin(math.pi / STRIPES) * 0.98:.6g} {max(L, 1e-3) * 0.98:.6g}")
                m = ET.SubElement(v, "material", name=f"{link.get('name')}_stripe{k}")
                ET.SubElement(m, "color", rgba=f"{0.62 - 0.57 * s:.3f} {0.95 - 0.60 * s:.3f} {0.62 - 0.52 * s:.3f} 1")


def colour_urdf_text(text: str, design: dict, stripes: bool = True) -> str:
    """`colour_by_joint_kind` on URDF text."""
    root = ET.fromstring(text)
    colour_by_joint_kind(root, design, stripes)
    return ET.tostring(root, encoding="unicode")


__all__ = ["GREY_RGBA", "KIND_RGBA", "colour_by_joint_kind", "colour_urdf_text"]
