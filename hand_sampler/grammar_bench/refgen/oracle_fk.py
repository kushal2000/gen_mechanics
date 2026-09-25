"""Independent oracle forward kinematics via Pinocchio.

Standalone script: stdlib + numpy + pinocchio only. Must NOT import
hand_sampler.grammar. Run with the piper conda interpreter, e.g.:

    /home/singularity/anaconda3/envs/piper/bin/python \\
        hand_sampler/grammar_bench/refgen/oracle_fk.py \\
        --urdf <path> --configs <path> --out <path> [--hand-root <link>]

Mimic couplings are expanded by this script's OWN affine implementation,
reading multiplier/offset from the URDF XML directly (never delegating to
hand_sampler.grammar). Body frame poses are looked up by link name via
``model.getFrameId(name, pin.FrameType.BODY)``.

hand_root handling: this script loads the FULL Pinocchio model (it does not
build a reduced/cut model) and, when --hand-root is given, re-expresses every
body's world-frame pose relative to the hand_root body's world-frame pose:
``T_rel = inv(T_hand_root) @ T_body``. This matches the grammar's convention
where the model root (set to hand_root after the cut) defines the frame that
every other transform is reported in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin


def _read_mimic_map(urdf_path: Path):
    root = ET.fromstring(urdf_path.read_bytes())
    mimic = {}
    for jel in root.findall("joint"):
        name = jel.attrib["name"]
        mimic_el = jel.find("mimic")
        if mimic_el is not None:
            source = mimic_el.attrib["joint"]
            mult = float(mimic_el.attrib.get("multiplier", 1.0))
            off = float(mimic_el.attrib.get("offset", 0.0))
            mimic[name] = (source, mult, off)
    return mimic


def _expand_u(u: dict, mimic_map: dict) -> dict:
    values = dict(u)
    remaining = dict(mimic_map)
    progress = True
    while remaining and progress:
        progress = False
        for dep, (source, mult, off) in list(remaining.items()):
            if source in values:
                values[dep] = mult * values[source] + off
                del remaining[dep]
                progress = True
    if remaining:
        raise ValueError(f"unresolved mimic couplings (cycle?): {sorted(remaining)}")
    return values


def _build_q(model: "pin.Model", values: dict) -> np.ndarray:
    q = pin.neutral(model)
    for jid in range(1, model.njoints):  # joint 0 is the universe
        name = model.names[jid]
        if name not in values:
            continue
        joint = model.joints[jid]
        idx_q = joint.idx_q
        nq = joint.nq
        val = float(values[name])
        if nq == 1:
            q[idx_q] = val
        elif nq == 2:
            # continuous joint: (cos, sin)
            q[idx_q] = np.cos(val)
            q[idx_q + 1] = np.sin(val)
        else:
            raise ValueError(f"joint {name!r} has unexpected nq={nq}")
    return q


def run(urdf_path: Path, configs_path: Path, out_path: Path, hand_root: str | None):
    mimic_map = _read_mimic_map(urdf_path)
    configs = json.loads(configs_path.read_text())

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()

    link_names = []
    urdf_root = ET.fromstring(urdf_path.read_bytes())
    for lel in urdf_root.findall("link"):
        link_names.append(lel.attrib["name"])

    frame_ids = {}
    for name in link_names:
        fid = model.getFrameId(name, pin.FrameType.BODY)
        if fid < model.nframes:
            frame_ids[name] = fid

    results = []
    for u in configs:
        values = _expand_u(u, mimic_map)
        q = _build_q(model, values)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        poses = {}
        for name, fid in frame_ids.items():
            T = np.eye(4)
            placement = data.oMf[fid]
            T[:3, :3] = placement.rotation
            T[:3, 3] = placement.translation
            poses[name] = T

        if hand_root is not None:
            if hand_root not in poses:
                raise ValueError(f"hand_root {hand_root!r} not found among link frames")
            T_root_inv = np.linalg.inv(poses[hand_root])
            poses = {name: T_root_inv @ T for name, T in poses.items()}

        results.append({name: T.tolist() for name, T in poses.items()})

    out = {
        "urdf": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "hand_root": hand_root,
        "tool_versions": {"pinocchio": pin.__version__, "numpy": np.__version__},
        "configs": configs,
        "poses": results,
    }
    out_path.write_text(json.dumps(out))
    print(f"wrote {len(results)} configurations x {len(frame_ids)} bodies to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--hand-root", default=None)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(Path(args.urdf), Path(args.configs), Path(args.out), args.hand_root)


if __name__ == "__main__":
    main()
