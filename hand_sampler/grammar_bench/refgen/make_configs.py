"""Generate u-configurations for a URDF, independently of hand_sampler.grammar.

Parses the URDF's own XML (stdlib ElementTree) to find movable joints minus
mimic dependents, and samples uniformly inside each independent joint's
declared limits (continuous: [-2pi, 2pi]), shrunk by 1e-9 relative. Writes
``references/<id>.configs.json``: a JSON list of {joint_name: value} dicts,
64 random (seed 20260925) + 7 extremal (all-lower, all-upper, zero-if-inside,
4 deterministic mixed corners).

Stdlib + numpy only. Must NOT import hand_sampler.grammar.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SEED = 20260925
N_RANDOM = 64
BOX_SHRINK_REL = 1e-9
CONTINUOUS_RANGE = (-2.0 * math.pi, 2.0 * math.pi)


def _joint_info(urdf_path: Path, hand_root: str = None):
    root = ET.fromstring(urdf_path.read_bytes())
    joints = []
    mimic_dependents = set()
    parent_of = {}
    children_of = {}
    for jel in root.findall("joint"):
        name = jel.attrib["name"]
        jtype = jel.attrib["type"]
        parent = jel.find("parent").attrib["link"]
        child = jel.find("child").attrib["link"]
        limit_el = jel.find("limit")
        limits = None
        if limit_el is not None and "lower" in limit_el.attrib and "upper" in limit_el.attrib:
            limits = (float(limit_el.attrib["lower"]), float(limit_el.attrib["upper"]))
        mimic_el = jel.find("mimic")
        if mimic_el is not None:
            mimic_dependents.add(name)
        parent_of[name] = parent
        children_of.setdefault(parent, []).append((name, child))
        joints.append((name, jtype, limits))

    if hand_root is None:
        return joints, mimic_dependents

    # Keep only joints in the subtree rooted at hand_root (mirrors the
    # grammar's own hand_root cut, reimplemented independently here since
    # this script must not import hand_sampler.grammar).
    keep_bodies = set()
    keep_joint_names = set()
    stack = [hand_root]
    while stack:
        b = stack.pop()
        if b in keep_bodies:
            continue
        keep_bodies.add(b)
        for jname, child in children_of.get(b, []):
            keep_joint_names.add(jname)
            stack.append(child)
    joints = [(n, t, l) for (n, t, l) in joints if n in keep_joint_names]
    mimic_dependents = mimic_dependents & keep_joint_names
    return joints, mimic_dependents


def make_box(urdf_path: Path, hand_root: str = None):
    joints, mimic_dependents = _joint_info(urdf_path, hand_root)
    box = {}
    for name, jtype, limits in joints:
        if jtype not in ("revolute", "continuous", "prismatic"):
            continue
        if name in mimic_dependents:
            continue
        if jtype == "continuous":
            box[name] = CONTINUOUS_RANGE
        else:
            box[name] = limits
    return box


def _shrink(lo, hi, rel):
    amt = (hi - lo) * rel
    new_lo, new_hi = lo + amt, hi - amt
    if new_lo > new_hi:
        mid = (lo + hi) / 2.0
        return mid, mid
    return new_lo, new_hi


def make_configs(urdf_path: Path, n_random=N_RANDOM, seed=SEED, hand_root: str = None):
    box = make_box(urdf_path, hand_root)
    names = sorted(box)
    shrunk = {name: _shrink(*box[name], BOX_SHRINK_REL) for name in names}

    rng = np.random.default_rng(seed)
    configs = []
    for _ in range(n_random):
        configs.append({name: float(rng.uniform(*shrunk[name])) for name in names})

    configs.append({name: shrunk[name][0] for name in names})
    configs.append({name: shrunk[name][1] for name in names})
    if all(shrunk[name][0] <= 0.0 <= shrunk[name][1] for name in names):
        configs.append({name: 0.0 for name in names})
    for k in range(4):
        corner = {}
        for idx, name in enumerate(names):
            lo, hi = shrunk[name]
            corner[name] = hi if ((idx + k) % 2 == 0) else lo
        configs.append(corner)
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--id", required=True, help="hand id; output goes to references/<id>.configs.json")
    ap.add_argument("--hand-root", default=None, help="restrict to the subtree rooted at this link")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "references"))
    args = ap.parse_args()

    urdf_path = Path(args.urdf)
    configs = make_configs(urdf_path, hand_root=args.hand_root)
    out_path = Path(args.out_dir) / f"{args.id}.configs.json"
    out_path.write_text(json.dumps(configs, indent=2))
    print(f"wrote {len(configs)} configurations to {out_path}")


if __name__ == "__main__":
    main()
