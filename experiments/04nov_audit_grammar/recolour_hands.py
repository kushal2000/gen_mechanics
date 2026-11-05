"""Copies of evolved hands whose visuals are coloured by joint kind, for renders (physics untouched).

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/recolour_hands.py --run map_elites \
        --hands g0067_c12 g0072_c00 g0058_c00 g0062_c11 [--tag rolltip]

Each finger link f<i>_link<j> -- the body joint j moves -- takes its joint's kind colour (flexion blue, abduction red,
roll green, as in the audit figures); the palm body (plate, mount, base servos) is grey; every roll tip's capsule
gets light-to-dark green strips round it so its spin shows (`_roll_tip_stripes`). Only <material><color> of the
visuals change; collisions, inertials, joints and the spec are copied as they are, except the spec's "urdf" path.
Writes debug_outputs/grammar_audit/recoloured/<run>_<tag>/<id>/ (urdf, spec, meshes) and manifest.json there (the
run's final-archive grid), the template manifest render_archive_videos.py takes with --manifest.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIND_RGBA = {(0, 0): "0.16 0.42 0.85 1", (90, 0): "0.86 0.20 0.18 1", (0, 90): "0.15 0.65 0.30 1"}
PALM_RGBA = "0.62 0.62 0.64 1"


STRIPES = 8


def _roll_tip_stripes(tree, design) -> None:
    """Visual-only strips around every roll tip's capsule, light to dark green round its circumference, so its spin
    shows in a render (a capsule is the same shape at every roll angle). Each strip is a thin box on the cylinder's
    surface along its length, in the tip link's frame; no collision, no inertia."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    root = tree.getroot()
    for i, f in enumerate(design["fingers"]):
        if tuple(f["joints"][-1]["axis"]) != (0, 90):
            continue
        link = root.find(f"link[@name='f{i}_link{len(f['joints']) - 1}']")
        for vis in list(link.findall("visual")):
            cyl = vis.find("geometry/cylinder")
            if cyl is None:
                continue
            o = vis.find("origin")
            c = np.array([float(v) for v in o.get("xyz").split()])
            Rm = Rotation.from_euler("xyz", [float(v) for v in o.get("rpy").split()]).as_matrix()
            r, L = float(cyl.get("radius")), float(cyl.get("length"))
            for k in range(STRIPES):
                th = 2 * np.pi * k / STRIPES
                shade = k / (STRIPES - 1)
                rgb = (0.62 - 0.57 * shade, 0.95 - 0.60 * shade, 0.62 - 0.52 * shade)
                Rk = Rm @ Rotation.from_euler("z", th).as_matrix()
                pos = c + Rm @ np.array([(r + 0.0006) * np.cos(th), (r + 0.0006) * np.sin(th), 0.0])
                v = ET.SubElement(link, "visual", name=f"f{i}_rolltip_stripe{k}")
                ET.SubElement(v, "origin", xyz=" ".join(f"{x:.9g}" for x in pos),
                              rpy=" ".join(f"{x:.9g}" for x in Rotation.from_matrix(Rk).as_euler("xyz")))
                g = ET.SubElement(v, "geometry")
                ET.SubElement(g, "box", size=f"{0.0012:.6g} {2 * r * np.sin(np.pi / STRIPES) * 0.98:.6g} {max(L, 1e-3) * 0.98:.6g}")
                m = ET.SubElement(v, "material", name=f"f{i}_rolltip_stripe{k}_mat")
                ET.SubElement(m, "color", rgba=f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} 1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="map_elites")
    ap.add_argument("--hands", nargs="+", required=True)
    ap.add_argument("--tag", default="rolltip")
    ap.add_argument("--summary-tag", default="_it73_100env")
    ap.add_argument("--no-stripes", action="store_true", help="leave roll-tip capsules plain green")
    args = ap.parse_args()
    src_man = json.loads((REPO / "debug_outputs/evolution/final_archive_eval" /
                          f"{args.run}{args.summary_tag}_manifest.json").read_text())
    out = REPO / "debug_outputs/grammar_audit/recoloured" / f"{args.run}_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    hands = []
    for hid in args.hands:
        src = REPO / "debug_outputs/evolution" / args.run / "hands" / hid
        dst = out / hid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        spec = json.loads((dst / f"{hid}.spec.json").read_text())
        design = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
        urdf = dst / f"{hid}.urdf"
        tree = ET.parse(urdf)
        for link in tree.getroot().findall("link"):
            name = link.get("name")
            if name.startswith("palm"):
                rgba = PALM_RGBA
            else:
                i, j = (int(v) for v in name[1:].split("_link"))
                rgba = KIND_RGBA[tuple(design["fingers"][i]["joints"][j]["axis"])]
            for vis in link.findall("visual"):
                for col in vis.iter("color"):
                    col.set("rgba", rgba)
        if not args.no_stripes:
            _roll_tip_stripes(tree, design)
        ET.indent(tree)
        tree.write(urdf, encoding="unicode", xml_declaration=False)
        spec["urdf"] = str(urdf.relative_to(REPO))
        (dst / f"{hid}.spec.json").write_text(json.dumps(spec, indent=1))
        hands.append({"id": hid, "spec": str((dst / f"{hid}.spec.json").relative_to(REPO))})
    (out / "manifest.json").write_text(json.dumps({"generation": src_man["generation"], "grid": src_man["grid"],
                                                    "hands": hands}, indent=1))
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
