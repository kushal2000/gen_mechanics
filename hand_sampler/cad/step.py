"""A printed part as a STEP file (ISO 10303-21, AP214), for opening in a CAD
program (Martin, 11/03: the kit had STLs only).

OpenSCAD builds meshes, so the solid is the part's mesh, written as a
closed shell of planar faces (a manifold solid B-rep, as CAD programs write
their own): each patch of coplanar triangles is one face, its holes inner
loops, and every curved surface (a hole, the rounding, a fillet, the
fingertip) stays the facets it is printed from. It opens as a solid that
can be measured, cut and added to; a cylinder is a ring of flat faces, not
a cylinder.

    python -m hand_sampler.cad.step PART.off PART.step
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PLANE_TOL_MM = 5e-5           # a triangle joins a face if its corners lie this close to the face's plane
UNCERTAINTY_MM = 1e-4         # the file's distance accuracy (vertices are float32: about 1e-5 mm at 100 mm)
TOUCH_MM = 0.01               # where the solid touches itself, each side moved this far into its own material
SIMPLIFY_MM = 1e-3
"""A part's mesh is first simplified this far (manifold3d): the booleans
leave edges far shorter than the file's accuracy, which a CAD kernel
collapses into loops it cannot mend (OpenCascade built no solid from four
palms of the commercial fits until these merged away)."""


def _num(x: float, digits: int = 6) -> str:
    """A STEP real: always a decimal point, no needless zeros."""
    s = f"{x:.{digits}f}".rstrip("0")
    return "0." if s in ("-0.", "0.") else s


def _faces(V: np.ndarray, F: np.ndarray) -> List[Tuple[np.ndarray, List[List[int]], int]]:
    """The mesh's faces: (unit normal, loops, piece), each loop a list of
    vertex indices counterclockwise about the normal for the outer loop and
    clockwise for a hole, `piece` the connected piece of the mesh it is on.
    A closed oriented mesh (manifold3d's) in, so every edge borders two
    triangles, once each way."""
    m = len(F)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    N = np.cross(b - a, c - a)
    area = np.linalg.norm(N, axis=1)
    n = N / np.maximum(area, 1e-300)[:, None]
    good = area > 1e-9
    across: Dict[Tuple[int, int], int] = {}
    for t in range(m):
        for k in range(3):
            across[(int(F[t, k]), int(F[t, (k + 1) % 3]))] = t

    def neighbours(t: int):
        for k in range(3):
            yield across[(int(F[t, (k + 1) % 3]), int(F[t, k]))]

    # coplanar patches, grown from the largest triangles (the steadiest normals)
    region = np.full(m, -1)
    planes: List[Tuple[np.ndarray, float]] = []
    for t0 in np.argsort(-area):
        if region[t0] >= 0 or not good[t0]:
            continue
        r, n0 = len(planes), n[t0]
        d0 = float(n0 @ a[t0])
        region[t0], stack = r, [t0]
        while stack:
            for s in neighbours(stack.pop()):
                if region[s] < 0 and (not good[s] or n[s] @ n0 > 0.99) \
                        and np.all(np.abs(V[F[s]] @ n0 - d0) <= PLANE_TOL_MM):
                    region[s] = r
                    stack.append(s)
        planes.append((n0, d0))
    left = list(np.nonzero(region < 0)[0])       # slivers no patch took: each with a neighbour's patch
    while left:
        for t in left:
            took = [region[s] for s in neighbours(t) if region[s] >= 0]
            if took:
                region[t] = took[0]
        if len(left) == len(left := [t for t in left if region[t] < 0]):
            raise ValueError(f"{len(left)} degenerate triangles with no face to join")

    piece = list(range(m))                       # the mesh's connected pieces (over shared edges)

    def root(i: int) -> int:
        while piece[i] != i:
            piece[i] = piece[piece[i]]
            i = piece[i]
        return i

    for t in range(m):
        for s in neighbours(t):
            a, b = root(t), root(s)
            if a != b:
                piece[max(a, b)] = min(a, b)
    members: Dict[int, List[int]] = {}
    for t, r in enumerate(region):
        members.setdefault(int(r), []).append(t)
    out = []
    for r, ts in members.items():
        nxt: Dict[int, List[int]] = {}
        for t in ts:
            for k in range(3):
                u, v = int(F[t, k]), int(F[t, (k + 1) % 3])
                if region[across[(v, u)]] != r:
                    nxt.setdefault(u, []).append(v)
        if any(len(vs) > 1 for vs in nxt.values()):
            # the patch touches itself at a corner: its triangles as faces of their own, each loop simple
            out += [(n[t] if good[t] else planes[r][0], [[int(x) for x in F[t]]], root(t)) for t in ts]
            continue
        loops, seen = [], set()
        for u0 in nxt:
            if u0 in seen:
                continue
            loop, u = [], u0
            while u not in seen:
                seen.add(u)
                loop.append(u)
                u = nxt[u][0]
            loops.append(loop)
        # the outer loop first (it encloses the most area); the normal by Newell's method over it
        areas = [float(_newell(V[lp]) @ planes[r][0]) for lp in loops]
        loops = [loops[i] for i in np.argsort(areas)[::-1]]
        w = _newell(V[loops[0]])
        out.append((w / np.linalg.norm(w), loops, root(ts[0])))
    return out


def _collapse_needles(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """The mesh with its zero-length edges collapsed: a rendered part can keep
    two vertices at one point joined by an edge, between two triangles of no
    area (a needle), which no B-rep edge can be. Each such pair becomes one
    vertex and the triangles that flatten to a line go; the mesh stays
    closed."""
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    short = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) < 1e-9
    if not short.any():
        return F
    keep = np.arange(len(V))

    def root(i: int) -> int:
        while keep[i] != i:
            keep[i] = keep[keep[i]]
            i = keep[i]
        return i

    for u, v in E[short]:
        ru, rv = root(int(u)), root(int(v))
        if ru != rv:
            keep[max(ru, rv)] = min(ru, rv)
    G = np.array([[root(int(i)) for i in t] for t in F])
    return G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 2] != G[:, 0])]


def _apart(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """The vertices with the solid's self-contacts opened: where it touches
    itself at a point or along an edge, manifold3d keeps a vertex for each
    side at one position, and a CAD kernel reads a shell touching itself as
    no solid (OpenCascade did, on six palms of the commercial fits). Each
    such vertex moves `TOUCH_MM` into its own material, against the normal
    of the triangles round it; its triangles then fall out of the flat faces
    they were in and stay faces of their own, exactly flat."""
    used = np.unique(F)
    _, inv, cnt = np.unique(np.round(V[used], 6), axis=0, return_inverse=True, return_counts=True)
    shared = used[cnt[inv.ravel()] > 1]
    if not len(shared):
        return V
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], N)
    ln = np.linalg.norm(vn[shared], axis=1)
    i = shared[ln > 1e-12]
    out = V.copy()
    out[i] -= TOUCH_MM * vn[i] / ln[ln > 1e-12][:, None]
    return out


def _newell(P: np.ndarray) -> np.ndarray:
    """A polygon's area vector."""
    return 0.5 * np.cross(P, np.roll(P, -1, axis=0)).sum(axis=0)


def write(V: np.ndarray, F: np.ndarray, path: Path, name: str) -> Dict[str, float]:
    """Write the closed mesh (V, F) as one solid named `name`; returns
    counts and the largest distance of a face's corner from its plane."""
    V = np.asarray(V, dtype=np.float64)
    F = _collapse_needles(V, np.asarray(F, dtype=np.int64))
    V = _apart(V, F)
    faces = _faces(V, F)
    lines: List[str] = []

    def add(s: str) -> int:
        lines.append(s)
        return len(lines)

    point: Dict[int, int] = {}
    vertex: Dict[int, int] = {}
    edge: Dict[Tuple[int, int], int] = {}

    def pt(i: int) -> int:
        if i not in point:
            x, y, z = V[i]
            point[i] = add(f"CARTESIAN_POINT('',({_num(x)},{_num(y)},{_num(z)}))")
        return point[i]

    def direction(d) -> int:
        return add(f"DIRECTION('',({_num(d[0], 12)},{_num(d[1], 12)},{_num(d[2], 12)}))")

    def vx(i: int) -> int:
        if i not in vertex:
            vertex[i] = add(f"VERTEX_POINT('',#{pt(i)})")
        return vertex[i]

    def oriented(u: int, v: int) -> int:
        lo, hi = min(u, v), max(u, v)
        if (lo, hi) not in edge:
            d = V[hi] - V[lo]
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                raise ValueError(f"a zero-length edge at {V[lo].round(3).tolist()}")
            vec = add(f"VECTOR('',#{direction(d / length)},{_num(length)})")
            line = add(f"LINE('',#{pt(lo)},#{vec})")
            edge[(lo, hi)] = add(f"EDGE_CURVE('',#{vx(lo)},#{vx(hi)},#{line},.T.)")
        return add(f"ORIENTED_EDGE('',*,*,#{edge[(lo, hi)]},{'.T.' if u == lo else '.F.'})")

    face_ids, worst = {}, 0.0
    for nrm, loops, piece in faces:
        h = np.concatenate([V[lp] @ nrm for lp in loops])
        d = (float(h.min()) + float(h.max())) / 2.0              # the plane midway between its farthest corners
        worst = max(worst, (float(h.max()) - float(h.min())) / 2.0)
        P0 = V[loops[0][0]]
        o = P0 + nrm * (d - float(nrm @ P0))
        ref = V[loops[0][1]] - P0
        ref = ref - nrm * float(ref @ nrm)
        ref /= np.linalg.norm(ref)
        bounds = []
        for k, lp in enumerate(loops):
            es = [oriented(lp[i], lp[(i + 1) % len(lp)]) for i in range(len(lp))]
            loop = add("EDGE_LOOP('',(" + ",".join(f"#{e}" for e in es) + "))")
            bounds.append(add(f"{'FACE_OUTER_BOUND' if k == 0 else 'FACE_BOUND'}('',#{loop},.T.)"))
        at = add(f"CARTESIAN_POINT('',({_num(o[0])},{_num(o[1])},{_num(o[2])}))")
        axis = add(f"AXIS2_PLACEMENT_3D('',#{at},#{direction(nrm)},#{direction(ref)})")
        plane = add(f"PLANE('',#{axis})")
        face_ids.setdefault(piece, []).append(
            add("ADVANCED_FACE('',(" + ",".join(f"#{b}" for b in bounds) + f"),#{plane},.T.)"))

    breps = []                                   # a solid for each piece (one, for a part that is whole)
    for k, ids in enumerate(face_ids.values()):
        shell = add("CLOSED_SHELL('',(" + ",".join(f"#{f}" for f in ids) + "))")
        breps.append(add(f"MANIFOLD_SOLID_BREP('{name if len(face_ids) == 1 else f'{name}_{k + 1}'}',#{shell})"))
    mm = add("(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.))")
    rad = add("(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.))")
    sr = add("(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT())")
    unc = add(f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE({UNCERTAINTY_MM:.1E}),#{mm},'distance_accuracy_value',"
              "'confusion accuracy')")
    ctx = add(f"(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{unc}))"
              f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{mm},#{rad},#{sr}))REPRESENTATION_CONTEXT('','3D'))")
    o = add("CARTESIAN_POINT('',(0.,0.,0.))")
    origin = add(f"AXIS2_PLACEMENT_3D('',#{o},#{direction((0.0, 0.0, 1.0))},#{direction((1.0, 0.0, 0.0))})")
    rep = add(f"ADVANCED_BREP_SHAPE_REPRESENTATION('{name}',(#{origin}," + ",".join(f"#{b}" for b in breps)
              + f"),#{ctx})")
    app = add("APPLICATION_CONTEXT('core data for automotive mechanical design processes')")
    add(f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})")
    pc = add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    product = add(f"PRODUCT('{name}','{name}','',(#{pc}))")
    add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))")
    formation = add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    pdc = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    pdef = add(f"PRODUCT_DEFINITION('design','',#{formation},#{pdc})")
    shape = add(f"PRODUCT_DEFINITION_SHAPE('','',#{pdef})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{rep})")

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    head = ["ISO-10303-21;", "HEADER;", f"FILE_DESCRIPTION(('{name}'),'2;1');",
            f"FILE_NAME('{Path(path).name}','{stamp}',(''),(''),'hand_sampler.cad.step','hand_sampler.cad','');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));", "ENDSEC;", "DATA;"]
    with open(path, "w") as fh:
        fh.write("\n".join(head) + "\n")
        fh.write("\n".join(f"#{i}={s};" for i, s in enumerate(lines, 1)) + "\n")
        fh.write("ENDSEC;\nEND-ISO-10303-21;\n")
    return {"faces": sum(len(ids) for ids in face_ids.values()), "solids": len(breps), "triangles": len(F),
            "edges": len(edge), "plane_error_mm": worst}


def write_part(off: Path, path: Path, name: str) -> Dict[str, float]:
    """A rendered part's OFF mesh (`render.load`, its indexing kept),
    simplified by `SIMPLIFY_MM` where that keeps its volume (as
    `render._lean`: a simplification took a 10 mm3 sliver off a palm that
    came out in two pieces), as STEP."""
    from .render import load

    _, man = load(off)
    lean = man.simplify(SIMPLIFY_MM)
    if abs(lean.volume() - man.volume()) <= 1e-4 * man.volume() + 1.0:
        man = lean
    mesh = man.to_mesh()
    return write(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts), path, name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m hand_sampler.cad.step PART.off PART.step")
    print(write_part(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[1]).stem))
