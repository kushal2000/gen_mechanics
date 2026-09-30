"""Import each hand's LEFT vendor URDF + only the meshes it references into <hand>/vendor_left/.

Reproducible provenance: every source is a public repo pinned to a commit, fetched with a sparse
git clone into a scratch dir, and recorded in <hand>/vendor_left/SOURCE.md. The URDF is copied
UNCHANGED except mesh filenames, which are rewritten to local relative paths (meshes/<name>) so
the asset is self-contained -- that fixes ROS package:// paths (Tesollo) and a repo that ships its
meshes somewhere other than where its URDF says (XHAND: URDF says meshes/, files are in assets/).
Repairs (rooting, wrist joints, colliders, ...) belong to each hand's unify step, not here.

    python assets/urdf/unified_commercial_hands/import_vendor_left.py <scratch_dir>
"""
import os, shutil, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
# hand -> (repo, commit, urdf path in repo, {package name: dir in repo}, mesh dir fallback, licence note)
SOURCES = {
    "allegro": ("dexsuite/dex-urdf", "f5e7132f22108164577fea4c25ef99b5cc0e1900",
                "robots/hands/allegro_hand/allegro_hand_left.urdf", {}, None, "dex-urdf: MIT (repo); Allegro BSD (per dex-urdf README; folder LICENSE copied)"),
    "leap":    ("dexsuite/dex-urdf", "f5e7132f22108164577fea4c25ef99b5cc0e1900",
                "robots/hands/leap_hand/leap_hand_left.urdf", {}, None, "dex-urdf: MIT (repo); LEAP Hand MIT (folder LICENSE copied)"),
    "shadow":  ("dexsuite/dex-urdf", "f5e7132f22108164577fea4c25ef99b5cc0e1900",
                "robots/hands/shadow_hand/shadow_hand_left.urdf", {}, None,
                "dex-urdf: MIT (repo); Shadow Dexterous Hand -- dex-urdf's README lists GPL-3.0 for Shadow, but the "
                "LICENSE file in its shadow_hand folder (copied here) is Apache-2.0"),
    "dex3":    ("unitreerobotics/unitree_ros", "5994d4faef0a9cadd3287f8de0199a67eeb2a259",
                "robots/dexterous_hand_description/dex3_1/dex3_1_l.urdf", {}, None, "Unitree official (unitree_ros, BSD-3)"),
    "tesollo": ("tesollodelto/dg5f_ros2", "9e9af1221cc82e4613a6f18210288f7782b53011",
                "dg5f_description/urdf/dg5f_left.urdf", {"dg5f_description": "dg5f_description"}, None,
                "Tesollo official (dg5f_ros2, BSD-3-Clause)"),
    "wuji2":   ("wuji-technology/wuji-description", "c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95",
                "hand2/hand2_beta2/body/urdf/left.urdf", {}, None,
                "Wuji official (wuji-description, MIT); Wuji Hand 2 Beta 2 -- the version wuji-mjlab deploys"),
    "xhand":   ("facebookresearch/spider", "44717007de41cbef7565dff7ff9f4453557a2d3d",
                "spider/assets/robots/xhand/xhand_left.urdf", {}, "spider/assets/robots/xhand/assets",
                "Robot Era XHAND1 SolidWorks-exported URDF, redistributed in Meta's spider, whose repo LICENSE is CC BY-NC (copied); Robot Era's own terms for the model are unknown -- NOT pushed to master (public)"),
}

def resolve(repo_dir: Path, urdf: Path, fn: str, pkgs: dict, fallback: str | None) -> Path:
    if fn.startswith("package://"):
        pkg, rest = fn[len("package://"):].split("/", 1)
        return repo_dir / pkgs[pkg] / rest
    p = (urdf.parent / fn).resolve()
    if not p.exists() and fallback:
        p = repo_dir / fallback / Path(fn).name
    return p

def main(scratch: Path):
    for hand, (repo, sha, rel, pkgs, fallback, lic) in SOURCES.items():
        repo_dir = scratch / repo.replace("/", "_")
        head = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == sha, f"{hand}: {repo_dir} is at {head}, expected {sha}"
        urdf = repo_dir / rel
        dest = HERE / hand / "vendor_left"
        if dest.exists():
            shutil.rmtree(dest)
        (dest / "meshes").mkdir(parents=True)
        tree = ET.parse(urdf); seen = {}
        for m in tree.getroot().iter("mesh"):
            src = resolve(repo_dir, urdf, m.get("filename"), pkgs, fallback)
            assert src.exists(), f"{hand}: missing mesh {m.get('filename')} -> {src}"
            name = src.name
            if name in seen and seen[name] != src:
                name = f"{src.parent.name}_{name}"
            seen[name] = src
            shutil.copy2(src, dest / "meshes" / name)
            m.set("filename", f"meshes/{name}")
        # Licence notices travel with the files (BSD / MIT / Apache require it): the nearest LICENSE*
        # walking up from the URDF's folder to the repo root.
        lic_files = []
        for d in [urdf.parent, *urdf.parent.parents]:
            lic_files = sorted(p for p in d.glob("LICEN[CS]E*") if p.is_file())
            if lic_files or d == repo_dir:
                break
        for lf in lic_files:
            shutil.copy2(lf, dest / lf.name)
        out = dest / f"{hand}_left.urdf"
        tree.write(out, xml_declaration=True, encoding="utf-8")
        (dest / "SOURCE.md").write_text(
            f"# {hand} left — vendor source\n\n- repo: https://github.com/{repo}\n- commit: {sha}\n"
            f"- file: {rel}\n- licence: {lic}\n- licence file(s) copied: {[p.name for p in lic_files] or 'NONE FOUND'}"
            f" (from {lic_files[0].parent.relative_to(repo_dir) if lic_files else '-'})\n"
            f"- imported by: ../../import_vendor_left.py (mesh paths "
            f"rewritten to meshes/<name>; URDF otherwise unchanged; {len(seen)} meshes)\n")
        print(f"{hand:8s} -> {out.relative_to(HERE.parents[2])}  ({len(seen)} meshes)")

if __name__ == "__main__":
    main(Path(sys.argv[1]))
