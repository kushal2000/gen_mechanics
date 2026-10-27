"""Visual meshes of a commercial hand's original URDF, per link, for the
overlay under the grammar projection.

The URDF is parsed with ElementTree (only `<link>/<visual>` is read) and each
mesh file is resolved with `grammar_bench/refgen/mesh_sections.
resolve_mesh_filename`, the resolver E14 used. Visual geometry is preferred,
collision geometry is the fallback, and box/cylinder/sphere primitives are
built with trimesh. Large meshes are reduced by vertex clustering so the
browser stays responsive: the overlay is a reference, not a measurement.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from hand_sampler.grammar.fk import pose_to_matrix
from hand_sampler.grammar.kinematics import Pose
from hand_sampler.grammar_bench.refgen.mesh_sections import resolve_mesh_filename

FACE_BUDGET_PER_PIECE = 6000
MESH_EXTS = (".stl", ".STL", ".obj", ".OBJ", ".ply", ".dae", ".DAE", ".glb")


@dataclass(frozen=True)
class MeshPiece:
    link: str
    vertices: np.ndarray  # (n, 3) float32, link frame
    faces: np.ndarray     # (m, 3) int32
    source: str           # file name or primitive kind
    original_faces: int


@dataclass
class MeshSet:
    urdf_path: Optional[Path]
    pieces: Dict[str, List[MeshPiece]] = field(default_factory=dict)
    unresolved: List[str] = field(default_factory=list)
    unloadable: List[str] = field(default_factory=list)
    substituted: List[str] = field(default_factory=list)
    n_links_with_geometry: int = 0

    @property
    def n_pieces(self) -> int:
        return sum(len(v) for v in self.pieces.values())

    @property
    def n_faces(self) -> int:
        return sum(len(p.faces) for v in self.pieces.values() for p in v)

    def summary(self) -> str:
        if self.urdf_path is None:
            return "no mesh-bearing copy of this URDF on disk"
        parts = [f"{self.n_pieces} mesh piece(s) on {len(self.pieces)} link(s), {self.n_faces} faces drawn"]
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} file(s) not found")
        if self.unloadable:
            parts.append(f"{len(self.unloadable)} file(s) found but not loadable "
                         f"({', '.join(sorted({Path(u).suffix for u in self.unloadable}))})")
        if self.substituted:
            parts.append(f"{len(self.substituted)} found by file name elsewhere (alignment unverified)")
        return "; ".join(parts)


def _floats(text: Optional[str], default: Tuple[float, ...]) -> Tuple[float, ...]:
    if text is None or not text.strip():
        return default
    return tuple(float(v) for v in text.split())


def _origin(el: Optional[ET.Element]) -> np.ndarray:
    if el is None:
        return np.eye(4)
    return pose_to_matrix(Pose(xyz=_floats(el.attrib.get("xyz"), (0.0, 0.0, 0.0)),
                               rpy=_floats(el.attrib.get("rpy"), (0.0, 0.0, 0.0))))


def cluster_decimate(V: np.ndarray, F: np.ndarray, budget: int = FACE_BUDGET_PER_PIECE) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering reduction: snap vertices to a grid, merge each cell to
    its mean, drop collapsed and duplicate triangles. The grid is coarsened
    until the face count fits `budget`."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if len(F) <= budget or len(V) == 0:
        return V.astype(np.float32), F.astype(np.int32)
    lo, hi = V.min(axis=0), V.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1e-3
    cell = diag / 120.0
    best = (V, F)
    for _ in range(12):
        keys = np.floor((V - lo) / cell).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
        newV = np.zeros((len(uniq), 3))
        for k in range(3):
            newV[:, k] = np.bincount(inv, weights=V[:, k], minlength=len(uniq)) / counts
        newF = inv[F]
        keep = (newF[:, 0] != newF[:, 1]) & (newF[:, 1] != newF[:, 2]) & (newF[:, 0] != newF[:, 2])
        newF = newF[keep]
        if len(newF):
            srt = np.sort(newF, axis=1)
            _, first = np.unique(srt, axis=0, return_index=True)
            newF = newF[np.sort(first)]
        best = (newV, newF)
        if len(newF) <= budget:
            break
        cell *= 1.4
    return best[0].astype(np.float32), best[1].astype(np.int32)


class _StemIndex:
    """File-name lookup under fallback directories (meshes referenced under a
    path that does not exist, but shipped elsewhere in the download)."""

    def __init__(self, dirs: Iterable[Path]):
        self.by_stem: Dict[str, List[Path]] = {}
        for d in dirs:
            if not d or not Path(d).is_dir():
                continue
            for p in Path(d).rglob("*"):
                if p.suffix in MESH_EXTS and p.is_file():
                    self.by_stem.setdefault(p.stem.lower(), []).append(p)

    def find(self, uri: str) -> Optional[Path]:
        stem = Path(uri.split("://")[-1]).stem.lower()
        cands = self.by_stem.get(stem, [])
        # Prefer formats trimesh reads without extra packages.
        for ext in (".stl", ".obj", ".ply", ".glb", ".dae"):
            for c in cands:
                if c.suffix.lower() == ext:
                    return c
        return cands[0] if cands else None


def _load_file(path: Path):
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
        raise ValueError("empty mesh")
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(loaded.faces, dtype=np.int64)


def _primitive(geom: ET.Element):
    import trimesh

    box = geom.find("box")
    if box is not None:
        m = trimesh.creation.box(extents=_floats(box.attrib.get("size"), (0.01, 0.01, 0.01)))
        return np.asarray(m.vertices), np.asarray(m.faces), "box"
    cyl = geom.find("cylinder")
    if cyl is not None:
        m = trimesh.creation.cylinder(radius=float(cyl.attrib.get("radius", 0.005)),
                                      height=float(cyl.attrib.get("length", 0.01)), sections=20)
        return np.asarray(m.vertices), np.asarray(m.faces), "cylinder"
    sph = geom.find("sphere")
    if sph is not None:
        m = trimesh.creation.icosphere(subdivisions=2, radius=float(sph.attrib.get("radius", 0.005)))
        return np.asarray(m.vertices), np.asarray(m.faces), "sphere"
    return None


def load_link_meshes(urdf_path: Optional[Path], links: Optional[Sequence[str]] = None,
                     fallback_dirs: Sequence[Path] = (), budget: int = FACE_BUDGET_PER_PIECE) -> MeshSet:
    """Per link (restricted to `links` when given), its visual geometry in the
    link frame; collision geometry when a link has no loadable visual."""
    ms = MeshSet(urdf_path=urdf_path)
    if urdf_path is None or not Path(urdf_path).is_file():
        ms.urdf_path = None
        return ms
    root = ET.fromstring(Path(urdf_path).read_bytes())
    urdf_dir = Path(urdf_path).parent
    keep = set(links) if links is not None else None
    stem_index: Optional[_StemIndex] = None
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        if keep is not None and name not in keep:
            continue
        got_any = False
        for tag in ("visual", "collision"):
            pieces: List[MeshPiece] = []
            for el in link.findall(tag):
                geom = el.find("geometry")
                if geom is None:
                    continue
                T = _origin(el.find("origin"))
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    uri = mesh_el.attrib.get("filename", "")
                    scale = np.asarray(_floats(mesh_el.attrib.get("scale"), (1.0, 1.0, 1.0)), dtype=float)
                    if scale.size == 1:
                        scale = np.repeat(scale, 3)
                    path = resolve_mesh_filename(uri, urdf_dir)
                    substituted = False
                    if path is None and fallback_dirs:
                        if stem_index is None:
                            stem_index = _StemIndex(fallback_dirs)
                        path = stem_index.find(uri)
                        substituted = path is not None
                    if path is None:
                        if tag == "visual":
                            ms.unresolved.append(uri)
                        continue
                    try:
                        V, F = _load_file(path)
                    except Exception:  # noqa: BLE001 - e.g. .dae without pycollada
                        if tag == "visual":
                            ms.unloadable.append(str(path))
                        continue
                    if substituted and tag == "visual":
                        ms.substituted.append(f"{uri} -> {path}")
                    V = V * scale[None, :]
                    src = Path(path).name
                else:
                    prim = _primitive(geom)
                    if prim is None:
                        continue
                    V, F, src = prim
                n_orig = len(F)
                V = (V @ T[:3, :3].T) + T[:3, 3]
                Vd, Fd = cluster_decimate(V, F, budget)
                pieces.append(MeshPiece(link=name, vertices=Vd, faces=Fd, source=src, original_faces=n_orig))
            if pieces:
                ms.pieces[name] = pieces
                got_any = True
                break  # visual found: do not also draw collision
        if got_any:
            ms.n_links_with_geometry += 1
    return ms
