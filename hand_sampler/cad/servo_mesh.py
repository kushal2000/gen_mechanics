"""The HL-3915 as meshes for drawing (the viewer's printed parts), in the
servo frame S (`servo.py`), millimetres.

From Feetech's STEP (`outputs/build_ref/FEETECH_HL_3915_Servo_Motor.stp`,
not in git) meshed once with gmsh:

    python -m hand_sampler.cad.servo_mesh [STEP]    # needs `pip install gmsh`

which writes `outputs/build_ref/hl3915_body.stl` (case and connector seats)
and `hl3915_horns.stl` (both horns and their centre screws). The STEP's
axis is its z, the driven horn toward +z, the case's back toward -y: S =
(z, -y, x), 8 mm along the axis. Without these files, `load` gives a plain
case box and horn discs.
"""

from __future__ import annotations

import struct
import sys
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

REF = Path(__file__).resolve().parents[2] / "outputs" / "build_ref"
STEP = REF / "FEETECH_HL_3915_Servo_Motor.stp"
GROUPS = {"body": [1, 2, 3, 10, 14], "horns": [8, 22, 21, 23]}   # the STEP's volumes
TO_S = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
SHIFT = np.array([8.0, 0.0, 0.0])


def _write_stl(path: Path, V: np.ndarray, F: np.ndarray, title: str) -> None:
    with open(path, "wb") as fh:
        fh.write(title.encode()[:80].ljust(80))
        fh.write(struct.pack("<I", len(F)))
        for a, b, c in F:
            n = np.cross(V[b] - V[a], V[c] - V[a])
            n = n / (np.linalg.norm(n) + 1e-12)
            fh.write(struct.pack("<12fH", *n, *V[a], *V[b], *V[c], 0))


def mesh_step(step: Path = STEP, out: Path = REF) -> None:
    """Mesh the STEP (about 30,000 triangles in all) into the two STLs
    (each surface as gmsh orients it; `load` turns every body outward)."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(str(step))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 2.5)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10)
    gmsh.model.mesh.generate(2)
    for name, vols in GROUPS.items():
        V, F, n = [], [], 0
        for t in vols:
            for _, s in gmsh.model.getBoundary([(3, t)], oriented=True):
                types, _, nodes = gmsh.model.mesh.getElements(2, abs(s))
                tags, coords, _ = gmsh.model.mesh.getNodes(2, abs(s), includeBoundary=True)
                idx = {int(k): m for m, k in enumerate(tags)}
                for ty, nd in zip(types, nodes):
                    if ty == 2:                       # 3-node triangles
                        tri = np.array([idx[int(k)] for k in nd]).reshape(-1, 3)
                        F.append((tri[:, ::-1] if s < 0 else tri) + n)
                V.append(coords.reshape(-1, 3) @ TO_S.T + SHIFT)
                n += len(V[-1])
        _write_stl(Path(out) / f"hl3915_{name}.stl", np.concatenate(V), np.concatenate(F), f"HL-3915 {name}")
    gmsh.finalize()


def _box(lo, hi):
    import trimesh

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=np.block([[np.eye(3), ((lo + hi) / 2)[:, None]],
                                                                       [np.zeros((1, 3)), np.ones((1, 1))]]))


def _disc_x(r, x0, x1):
    import trimesh

    c = trimesh.creation.cylinder(radius=r, height=x1 - x0, sections=32)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, 0.0, 0.0])
    return c


def _outward(m):
    """Every closed body of `m` wound outward (gmsh's surfaces come either
    way round, which showed the servos' insides), stray triangles dropped."""
    import trimesh

    bodies = [b for b in m.split(only_watertight=False) if len(b.faces) >= 4]
    for b in bodies:
        trimesh.repair.fix_normals(b)
    return trimesh.util.concatenate(bodies)


@lru_cache(maxsize=1)
def load() -> Tuple[object, object, object]:
    """(case, driven horn, idler horn) as trimesh meshes in S (mm)."""
    import trimesh

    body, horns = REF / "hl3915_body.stl", REF / "hl3915_horns.stl"
    if body.exists() and horns.exists():
        b = _outward(trimesh.load(str(body), force="mesh"))
        h = _outward(trimesh.load(str(horns), force="mesh"))
        driven = h.triangles_center[:, 0] > 0.0
        return b, h.submesh([np.nonzero(driven)[0]], append=True), h.submesh([np.nonzero(~driven)[0]], append=True)
    return _box((-11.5, -9.5, -10.0), (11.5, 24.5, 10.0)), _disc_x(8.0, 11.5, 14.5), _disc_x(8.0, -14.5, -12.1)


if __name__ == "__main__":
    mesh_step(Path(sys.argv[1]) if len(sys.argv) > 1 else STEP)
    print("wrote", REF / "hl3915_body.stl", "and", REF / "hl3915_horns.stl")
