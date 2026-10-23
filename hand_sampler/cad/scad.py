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


def settings_block() -> str:
    """The library's settings, copied so they can be edited in the hand file
    (a later assignment wins in OpenSCAD)."""
    text = LIBRARY.read_text()
    block = text.split("// ---------------------------------------------------------------- settings\n", 1)[1]
    block = block.split("\n// ---", 1)[0]
    return block.strip() + "\n"


def printed_names(design: Design) -> List[str]:
    """Every printable part's name: palm parts come as two halves."""
    out = []
    for p in design.part_list():
        if p.split_x is not None:
            out += [f"{p.name}_front", f"{p.name}_back"]
        else:
            out.append(p.name)
    return out


def _split_modules(p: Part) -> str:
    big = 1000
    return (f"module part_{p.name}_front() intersection() {{ part_{p.name}(); "
            f"translate([{p.split_x}, -{big / 2}, -{big / 2}]) cube([{big}, {big}, {big}]); }}\n"
            f"module part_{p.name}_back() intersection() {{ part_{p.name}(); "
            f"translate([{p.split_x - big}, -{big / 2}, -{big / 2}]) cube([{big}, {big}, {big}]); }}\n")


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
        if p.split_x is not None:
            out.append(_split_modules(p))
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
    names = printed_names(design)
    out.append('if (part == "assembly") assembly();')
    for p in design.part_list():
        out.append(f'else if (part == "view_{p.name}") view_{p.name}();')
    for n in names:
        out.append(f'else if (part == "{n}") part_{n}();')
    for p in design.part_list():
        if p.split_x is not None:
            out.append(f'else if (part == "{p.name}_whole") part_{p.name}();')
    return "\n".join(out) + "\n"


def write(design: Design, folder: Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(LIBRARY, folder / LIBRARY.name)
    path = folder / "hand.scad"
    path.write_text(hand_scad(design))
    return path
