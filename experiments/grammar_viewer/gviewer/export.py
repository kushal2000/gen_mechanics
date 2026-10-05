"""Export the current design: its derivation as JSON and a URDF with capsule
collision geometry and palm-cell meshes (`adapters/urdf.to_urdf`)."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Optional

from hand_sampler.grammar.derive import Derivation, derivation_to_dict
from hand_sampler.grammar.geometry import build_geometry, write_geometry_meshes
from hand_sampler.grammar.adapters.urdf import to_urdf
from hand_sampler.grammar.kinematics import KinematicModel

from .envload import REPO_ROOT

DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "grammar_viewer"


def safe_stem(label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.")
    return stem[:80] or "design"


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def export_derivation(derivation: Derivation, label: str, out_dir: Path = DEFAULT_OUT_DIR,
                      extra: Optional[Dict] = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_stamp()}_{safe_stem(label)}.derivation.json"
    doc = derivation_to_dict(derivation)
    if extra:
        # Viewer provenance travels beside the derivation, never inside it, so
        # the file still loads with derivation_from_dict.
        path.with_suffix(".meta.json").write_text(json.dumps(extra, indent=1, default=str))
    path.write_text(json.dumps(doc, indent=1))
    return path


def export_urdf(model: KinematicModel, label: str, out_dir: Path = DEFAULT_OUT_DIR,
                with_geometry: bool = True) -> Dict[str, Path]:
    """`<stem>/<stem>.urdf` plus `<stem>/meshes/<body>_cell.obj` per palm
    body. The URDF names `package://<robot>/meshes/...`, so the folder is laid
    out as a package named after the robot (`model.name`)."""
    stem = f"{_stamp()}_{safe_stem(label)}"
    pkg = Path(out_dir) / stem
    pkg.mkdir(parents=True, exist_ok=True)
    geometry = build_geometry(model) if with_geometry else None
    text, losses = to_urdf(model, geometry=geometry)
    urdf_path = pkg / f"{stem}.urdf"
    urdf_path.write_text(text)
    out = {"urdf": urdf_path}
    if geometry is not None and geometry.cells:
        written = write_geometry_meshes(geometry, pkg)  # writes pkg/meshes/<body>_cell.obj
        out["meshes"] = pkg / "meshes"
        out["n_meshes"] = len(written)
    if losses.placeholder_actuation:
        out["placeholder_actuation"] = len(losses.placeholder_actuation)
    return out
