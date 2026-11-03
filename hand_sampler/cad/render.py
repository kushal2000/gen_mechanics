"""Render a hand file's parts to STL with OpenSCAD (the Manifold backend:
seconds per part). Each part's body (its pieces' union) is rendered first and
rounded as a whole (`round_body`, into bodies/ next to the hand file, where
the part imports it; a link's inside corners filleted first, its reliefs
then cut), then the part: the body less what it gives way to.

OpenSCAD is looked for in `$OPENSCAD`, then on PATH, then at
`~/.local/bin/openscad` (the extracted AppImage; README of the viewer).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence


ISLAND_MM3 = 500.0
ROUND_SEGMENTS = 8            # the rounding ball's facets round a circle (its 0.8 mm radius: under 0.1 mm off)
FILLET_MM = 2.0
"""A link's inside corners are filleted this much (Martin, 11/02: the sharp
angles where a horn's arm joins the rest of the link): its solid is closed,
grown by a ball this big and shrunk by it again, before its reliefs are
cut, so its own joint's clearances stay exact. A fillet fills a square
corner 0.41 of its radius out from the pieces, which the part before the
joint does not see (it gives way to the pieces, 1 mm round, `PART_CLEAR`):
at 2 mm that is 0.83 mm, inside that clearance (4 mm cut into it)."""
FILLET_SEGMENTS = 16          # that ball's facets round a circle (under 0.05 mm off at 2 mm)


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


def _lean(man):
    """The solid with what lies within 0.01 mm merged (a pieces' union, less
    a link's reliefs, comes in sliver facets: 19 554 triangles where 510 do,
    on one link, which makes a rounding minutes, not seconds), unless the
    merge changes its volume (on one palm it closed a hole and took 1.6 cm3
    away). A merge that closes a sliver's tunnel and changes the volume by
    a fraction of a mm3 is kept (a fillet's closing leaves such tunnels)."""
    lean = man.simplify(0.01)
    if abs(lean.volume() - man.volume()) <= 1e-4 * man.volume() + 1.0:
        return lean
    return man


def round_body(src: Path, dst: Path, r: float, fillet: float = 0.0, reliefs: Optional[Path] = None) -> bool:
    """A body's outer edges rounded by r: its morphological opening (shrunk
    by a ball of radius r, then grown by it), so faces that run on from one
    piece to the next stay flush. Bits thinner than 2 r go (slivers).
    With `fillet`, the solid's inside corners are first filleted that much
    (its closing: grown by that ball, then shrunk) and `reliefs` cut from
    it; without, inside corners stay sharp. False if nothing is left (or
    the body is not one closed solid)."""
    import manifold3d as m3
    import numpy as np
    import trimesh

    _, man = load(src)
    if man.status() != m3.Error.NoError or man.is_empty():
        return False
    man = _lean(man)
    if fillet > 0.0:
        ball = m3.Manifold.sphere(fillet, FILLET_SEGMENTS)
        man = _lean(man.minkowski_sum(ball).minkowski_difference(ball))
        if reliefs is not None:
            _, cut = load(reliefs)
            man = _lean(man - cut)
    ball = m3.Manifold.sphere(r, ROUND_SEGMENTS)
    core = man.minkowski_difference(ball)
    if core.is_empty():
        return False
    # a filleted link cut by its reliefs leaves the shrunk body thousands of slivers, and growing each back
    # made a million triangles (half a minute to many); merging them first makes it a second
    core = _lean(core)
    # the sum leaves sliver facets on curved faces; merging what lies within 0.01 mm clears them
    out = core.minkowski_sum(ball).simplify(0.01).to_mesh()
    trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=False).export(str(dst))
    return True


def is_link(name: str) -> bool:
    """A finger's link or fingertip (`f<i>_l<j>`, `f<i>_tip`): its inside corners are filleted."""
    return re.fullmatch(r"f\d+_(l\d+|tip)", name) is not None


def make_body(scad: Path, name: str, timeout_s: float = 900.0) -> str:
    """Render part `name`'s pieces and round them into bodies/<name>.stl
    next to the hand file (a process of its own: the rounding holds
    Python's lock); a link's solid and reliefs apart, its inside corners
    filleted between (`FILLET_MM`). '' when done, else what went wrong."""
    from .design import ROUND_MM

    bodies = scad.parent / "bodies"
    bodies.mkdir(exist_ok=True)
    fillet = is_link(name)
    raw = bodies / f"{name}.{'solid' if fillet else 'pieces'}.off"
    cut = bodies / f"{name}.reliefs.off"
    jobs = [(raw, f"solid_{name}" if fillet else f"pieces_{name}")] + ([(cut, f"reliefs_{name}")] if fillet else [])
    try:
        for out, module in jobs:
            cmd = [openscad(), "-o", str(out), *backend_flags(), "-D", f'part="{module}"', str(scad)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(scad.parent))
            log = (r.stdout or "") + (r.stderr or "")
            if out is cut and "top level object is empty" in log.lower():
                continue                                    # a link with no reliefs
            if r.returncode != 0 or not out.exists():
                return f"its {module.split('_')[0]}: " + log.strip()[-300:]
        args = [str(raw), str(bodies / f"{name}.stl"), str(ROUND_MM)]
        if fillet:
            args += [str(FILLET_MM)] + ([str(cut)] if cut.exists() else [])
        r = subprocess.run([sys.executable, "-m", "hand_sampler.cad.render", "round", *args],
                           capture_output=True, text=True, timeout=timeout_s, cwd=str(Path(__file__).resolve().parents[2]))
        if r.returncode != 0:
            return "rounding its body: " + ((r.stdout or "") + (r.stderr or "")).strip()[-300:]
    except subprocess.TimeoutExpired as e:
        return f"its body timed out after {timeout_s:.0f} s: {e}"
    finally:
        raw.unlink(missing_ok=True)
        cut.unlink(missing_ok=True)
    return ""


def render_part(scad: Path, name: str, out_dir: Path, timeout_s: float = 900.0) -> Rendered:
    scad, out_dir = Path(scad).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stl, off = out_dir / f"{name}.stl", out_dir / f"{name}.off"
    off.unlink(missing_ok=True)
    t0 = time.time()
    err = make_body(scad, name, timeout_s)
    if err:
        return Rendered(name, stl, False, time.time() - t0, err)
    # OFF, not STL: it keeps OpenSCAD's indexed mesh, so where two faces of the part meet along one
    # edge (four faces on it) nothing is welded wrongly when the solid is read back
    cmd = [openscad(), "-o", str(off), *backend_flags(), "-D", f'part="{name}"', str(scad)]
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


if __name__ == "__main__":
    # python -m hand_sampler.cad.render round SRC DST R [FILLET [RELIEFS]]
    if 5 <= len(sys.argv) <= 7 and sys.argv[1] == "round":
        fillet = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        reliefs = Path(sys.argv[6]) if len(sys.argv) > 6 else None
        sys.exit(0 if round_body(Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]), fillet, reliefs) else 1)
    sys.exit("usage: python -m hand_sampler.cad.render round SRC DST R [FILLET [RELIEFS]]")
