"""Write a design as one OpenSCAD file (`hand.scad`) next to a copy of the
library: the genotype as a comment, the build settings to edit, one module
per printed part, an `assembly()` preview and a `part` selector

    openscad -o f0_l0.stl -D 'part="f0_l0"' hand.scad
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List

import numpy as np

from hand_sampler.grammar.hand import hand_to_json

from .design import Design, Part, mm

LIBRARY = Path(__file__).with_name("scad") / "feetech_hand.scad"
STEP_PULL_MM = 35.0           # the assembly pictures pull each link this far off its servo


def settings_block() -> str:
    """The library's settings, copied so they can be edited in the hand file
    (a later assignment wins in OpenSCAD)."""
    text = LIBRARY.read_text()
    block = text.split("// ---------------------------------------------------------------- settings\n", 1)[1]
    block = block.split("\n// ---", 1)[0]
    return block.strip() + "\n"


def printed_names(design: Design) -> List[str]:
    """Every printable part's name."""
    return [p.name for p in design.part_list()]


def hand_scad(design: Design) -> str:
    out = ["// A hand-grammar hand as printable parts (hand_sampler/cad). Generated: edit the",
           "// settings below and re-render, or edit the hand in the viewer and generate again.",
           "// The hand (hand_grammar genotype):"]
    for line in hand_to_json(design.hand).splitlines():
        out.append("//   " + line)
    out += ["", "include <feetech_hand.scad>", "", "// ---- settings (copied from the library; edit here) ----",
            settings_block(), 'part = "assembly";', ""]
    for p in design.part_list():
        out.append(p.scad())
    # the preview: every part where it sits at the zero pose, and the servos
    out.append("module assembly() {")
    for p in design.part_list():
        out.append(f"    color(\"lightsteelblue\", 0.9) {mm(p.frame)} part_{p.name}();")
    for sp in design.servos.values():
        F = design.parts[sp.parent].frame @ sp.S_parent
        out.append(f"    {mm(F)} servo_model({str(sp.idler_horn).lower()});")
    if getattr(design, "bay_frame", None) is not None:
        cy, cz = design.bay_centre
        out.append(f"    {mm(design.bay_frame)} board_model({cy:.3f}, {cz:.3f});")
    out.append("}\n")
    # each part in its own frame with the servos it holds and carries (for looking at it)
    for p in design.part_list():
        out.append(f"module view_{p.name}() {{")
        out.append(f"    color(\"lightsteelblue\") part_{p.name}();")
        for sp in p.holds:
            out.append(f"    {mm(sp.S_parent)} servo_model({str(sp.idler_horn).lower()});")
        if p.carries is not None:
            out.append(f"    {mm(p.carries.S_child)} servo_model({str(p.carries.idler_horn).lower()});")
        out.append("}")
    # one picture per joint: the part holding the servo, the servo with its screws, the next link
    # pulled off the way it goes on (along the servo's front)
    for sp in design.servos.values():
        tag = "_".join(str(k) for k in sp.key)
        parent, child = design.parts[sp.holder or sp.parent], design.parts[sp.child]
        pull = sp.S_parent @ np.array([[1, 0, 0, 0], [0, 1, 0, -STEP_PULL_MM], [0, 0, 1, 0], [0, 0, 0, 1.0]]) \
            @ np.linalg.inv(sp.S_child)
        single = str(sp.yoke != "u").lower()
        out.append(f"module step_{tag}() {{")
        out.append(f"    color(\"lightsteelblue\") part_{parent.name}();")
        out.append(f"    {mm(sp.S_parent)} {{ servo_model({str(sp.idler_horn).lower()}); "
                   f"case_screws({str(sp.full_plates).lower()}); }}")
        out.append(f"    color(\"lightskyblue\", 0.85) {mm(pull)} part_{child.name}();")
        out.append(f"    {mm(sp.S_parent @ np.array([[1, 0, 0, 0], [0, 1, 0, -STEP_PULL_MM], [0, 0, 1, 0], [0, 0, 0, 1.0]]))} "
                   f"arm_screws({single});")
        out.append("}")
    names = printed_names(design)
    out.append('if (part == "assembly") assembly();')
    for sp in design.servos.values():
        tag = "_".join(str(k) for k in sp.key)
        out.append(f'else if (part == "step_{tag}") step_{tag}();')
    for p in design.part_list():
        out.append(f'else if (part == "view_{p.name}") view_{p.name}();')
    for n in names:
        out.append(f'else if (part == "{n}") part_{n}();')
    return "\n".join(out) + "\n"


def write(design: Design, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(LIBRARY, folder / LIBRARY.name)
    path = folder / "hand.scad"
    path.write_text(hand_scad(design))
    return path
