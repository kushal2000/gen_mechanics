"""Copies of evolved hands whose visuals are coloured by joint kind, for renders (physics untouched).

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/recolour_hands.py --run map_elites \
        --hands g0067_c12 g0072_c00 g0058_c00 g0062_c11 [--tag rolltip]

Each finger link f<i>_link<j> -- the body joint j moves -- takes its joint's kind colour (flexion blue, abduction red,
roll green, as in the audit figures); the palm body (plate, mount, base servos) is grey; every roll tip's capsule
gets light-to-dark green strips round it so its spin shows (hand_sampler/grammar/visual_colours.py). Only <material><color> of the
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
sys.path.insert(0, str(REPO))
from hand_sampler.grammar.visual_colours import colour_by_joint_kind  # noqa: E402


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
        colour_by_joint_kind(tree.getroot(), design, stripes=not args.no_stripes)
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
