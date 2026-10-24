"""Render a hand file's parts to STL with OpenSCAD (the Manifold backend:
seconds per part).

OpenSCAD is looked for in `$OPENSCAD`, then on PATH, then at
`~/.local/bin/openscad` (the extracted AppImage; README of the viewer).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ISLAND_MM3 = 500.0


class OpenSCADMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def openscad() -> str:
    for cand in (os.environ.get("OPENSCAD"), shutil.which("openscad"),
                 str(Path.home() / ".local" / "bin" / "openscad")):
        if cand and Path(cand).exists():
            return cand
    raise OpenSCADMissing("OpenSCAD not found: install a 2025+ build (see experiments/grammar_viewer/README.md) "
                          "or set $OPENSCAD")


@lru_cache(maxsize=1)
def backend_flags() -> List[str]:
    try:
        text = subprocess.run([openscad(), "--help"], capture_output=True, text=True, timeout=60).stdout + \
            subprocess.run([openscad(), "--help"], capture_output=True, text=True, timeout=60).stderr
    except Exception:
        text = ""
    if "--backend" in text:
        return ["--backend=manifold"]
    return ["--enable=manifold"]


@dataclass
class Rendered:
    name: str
    stl: Path
    ok: bool
    seconds: float
    log: str


def render_part(scad: Path, name: str, out_dir: Path, timeout_s: float = 900.0) -> Rendered:
    scad, out_dir = Path(scad).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stl, off = out_dir / f"{name}.stl", out_dir / f"{name}.off"
    off.unlink(missing_ok=True)
    # OFF, not STL: it keeps OpenSCAD's indexed mesh, so where two faces of the part meet along one
    # edge (four faces on it) nothing is welded wrongly when the solid is read back
    cmd = [openscad(), "-o", str(off), *backend_flags(), "-D", f'part="{name}"', str(scad)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(scad.parent))
        log = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0 and off.exists() and off.stat().st_size > 20
    except subprocess.TimeoutExpired as e:
        log, ok = f"timed out after {timeout_s:.0f} s: {e}", False
    if ok:
        ok = clean(off, stl, island_fraction=0.05 if name.startswith("palm") else 0.0)
        if not ok:
            log += "\nnot a closed solid after cleaning"
    return Rendered(name, stl, ok, time.time() - t0, log)


def load(path: Path):
    """A rendered part as (trimesh, manifold3d solid), read from its OFF file
    (`path` may name the STL next to it) without welding vertices."""
    import manifold3d as m3
    import numpy as np
    import trimesh

    path = Path(path)
    if path.with_suffix(".off").exists():
        path = path.with_suffix(".off")
    tm = trimesh.load(str(path), force="mesh", process=False)
    man = m3.Manifold(m3.Mesh(vert_properties=np.asarray(tm.vertices, dtype=np.float32),
                              tri_verts=np.asarray(tm.faces, dtype=np.uint32)))
    return tm, man


def clean(off: Path, stl: Path, island_fraction: float = 0.0) -> bool:
    """Read the rendered OFF into manifold3d and write it back (OFF for the
    checks, STL for printing). Loose pieces under `ISLAND_MM3`, or (palm
    plates, whose outline can leave a sliver between two fingers' slots)
    under `island_fraction` of the part, are dropped; the checks make sure
    every servo is still held. False if it is not a closed solid."""
    import manifold3d as m3
    import numpy as np
    import trimesh

    _, man = load(off)
    if man.status() != m3.Error.NoError or man.is_empty():
        return False
    pieces = man.decompose()
    if len(pieces) > 1:
        total = sum(q.volume() for q in pieces)
        cut = max(min(ISLAND_MM3, 0.05 * total), island_fraction * total)
        keep = [q for q in pieces if q.volume() >= cut]
        man = keep[0]
        for q in keep[1:]:
            man = man + q
    mesh = man.to_mesh()
    out = trimesh.Trimesh(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), process=False)
    out.export(str(off), file_type="off")
    out.export(str(stl), file_type="stl")
    return man.status() == m3.Error.NoError and not man.is_empty()


def render_all(scad: Path, names: Sequence[str], out_dir: Path, workers: Optional[int] = None) -> Dict[str, Rendered]:
    workers = workers or max(1, min(len(names), (os.cpu_count() or 4) // 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda n: render_part(Path(scad), n, Path(out_dir)), names))
    return {r.name: r for r in results}
