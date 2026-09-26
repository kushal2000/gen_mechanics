"""Iteration-7 (start of M2) acceptance tests for the geometry overlay
(``hand_sampler/grammar/geometry.py``): ``GeometrySpec`` construction over
200 seeds (nearest-spine palm-cell assignment, iteration 7c), the
convex-hull building blocks in isolation, mass properties against closed
forms, and URDF export with/without geometry.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from urdf_parser_py.urdf import URDF

from hand_sampler.grammar import geometry as geo
from hand_sampler.grammar.adapters.urdf import to_urdf
from hand_sampler.grammar.derive import generate
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.kinematics import Body, Frame, Joint, KinematicModel, Pose

N_SEEDS = 200


# ---------------------------------------------------------------------------
# 1. GeometrySpec over 200 seeds (nearest-spine palm-cell assignment)
# ---------------------------------------------------------------------------


def _cell_root_frame(spec, transforms, name: str):
    """A cell's ``(vertices, faces)``, transformed from its own body-local
    frame into the shared root frame (q=0)."""
    cell = spec.cells[name]
    T = transforms[name]
    Rm, t = T[:3, :3], T[:3, 3]
    verts = np.array([Rm @ np.array(v) + t for v in cell.vertices])
    return verts, cell.faces


def _inside_mask(points: np.ndarray, verts: np.ndarray, faces, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask: which rows of ``points`` lie inside/on polytope
    ``(verts, faces)`` within ``tol``, vectorized over every face. ``tol``
    defaults to 1e-6 m, matching the convexity-check tolerance above (same
    root cause: a convex hull built from thousands of dense-sample points,
    not exact to 1e-9 -- see that comment)."""
    inside = np.ones(len(points), dtype=bool)
    for face in faces:
        nvec, a = geo._face_outward_normal(verts, face)
        if nvec is None:
            continue
        d = float(np.dot(nvec, a))
        inside &= (points @ nvec - d) <= tol
    return inside


def test_geometry_over_200_seeds():
    total_palm = 0
    total_cells = 0
    min_cell_vol = None
    mount_outside_count = 0
    spine_outside_count = 0
    coverage_outside_count = 0
    total_overlap_pairs = 0
    total_possible_pairs = 0
    overlap_fracs = []
    total_deficit_rel = []

    for seed in range(N_SEEDS):
        _derivation, model = generate(seed)
        spec = geo.build_geometry(model)
        transforms = forward_kinematics(model, {})
        palm_names = [b.name for b in model.bodies if b.palm]
        palm_set = set(palm_names)
        tip_len = {f.body: float(f.pose.xyz[2]) for f in model.frames if f.name == f"{f.body}_tip"}

        total_palm += len(palm_names)
        total_cells += len(spec.cells)

        # 1. Exactly one cell per palm body.
        assert len(spec.cells) == len(palm_names), (
            f"seed {seed}: cell count {len(spec.cells)} != palm body count {len(palm_names)}"
        )

        hull_v = np.array(spec.hull_vertices)
        hull_vol = geo.polytope_volume(hull_v, spec.hull_faces) if spec.hull_vertices else 0.0
        rel_scale = max(abs(hull_vol), 1e-15)

        # 2. Every cell volume > 0, and convex (every vertex satisfies every
        # face's own half-space). Tolerance is 1e-6 m, not 1e-9: a cell's
        # convex hull is now built from thousands of dense-sample points
        # (not ~30-150 capsule points as before), and the from-scratch
        # incremental hull builder's visibility test (a single dot product
        # per candidate face, no re-fitting) accumulates float64 error on
        # inputs this size -- the worst observed violation over 200 seeds
        # is 4.3e-7 m (seed 179, 'palm2'), comfortably inside 1e-6 and still
        # tiny relative to any cell's own physical extent (~0.01-0.1 m).
        cell_root = {}
        for name, cell in spec.cells.items():
            verts, faces = _cell_root_frame(spec, transforms, name)
            cell_root[name] = (verts, faces)
            vol = geo.polytope_volume(verts, faces)
            if min_cell_vol is None or vol < min_cell_vol:
                min_cell_vol = vol
            assert vol > 0.0, f"seed {seed}: nonpositive cell volume for {name!r}: {vol!r}"
            for face in faces:
                nvec, a = geo._face_outward_normal(verts, face)
                if nvec is None:
                    continue
                d = float(np.dot(nvec, a))
                viol = float(np.max(verts @ nvec - d))
                assert viol <= 1e-6, f"seed {seed}: non-convex cell {name!r} (violation {viol!r})"

        # 3. Every digit mount point (origin of a joint whose parent is a
        # palm body and whose child is a non-palm body) lies on/inside its
        # palm body's cell.
        for j in model.joints:
            if j.parent in palm_set and j.child not in palm_set:
                verts, faces = cell_root[j.parent]
                T = transforms[j.parent]
                mount_pt = T[:3, :3] @ np.array(j.origin.xyz, dtype=float) + T[:3, 3]
                inside = bool(_inside_mask(mount_pt[None, :], verts, faces)[0])
                if not inside:
                    mount_outside_count += 1
                assert inside, f"seed {seed}: digit mount {j.name!r} lies outside cell {j.parent!r}"

        # 4. Every palm body's own spine (9 sampled points) lies on/inside
        # its own cell (structurally guaranteed by construction -- a body's
        # own capsule points, including its start/end, are always unioned
        # into its own cell -- checked here as a real regression guard).
        for name in palm_names:
            verts, faces = cell_root[name]
            T = transforms[name]
            L = tip_len.get(name, 0.0)
            local_pts = np.array([[0.0, 0.0, t * L] for t in np.linspace(0.0, 1.0, 9)])
            world_pts = (T[:3, :3] @ local_pts.T).T + T[:3, 3]
            mask = _inside_mask(world_pts, verts, faces)
            if not mask.all():
                spine_outside_count += 1
            assert mask.all(), f"seed {seed}: {name!r}'s own spine leaves its own cell"

        # 5. Coverage: every dense hull sample point lands inside at least
        # one cell.
        dense = geo._dense_hull_samples(hull_v, spec.hull_faces, target_count=2000) if len(hull_v) else hull_v
        if len(dense):
            covered = np.zeros(len(dense), dtype=bool)
            for name, (verts, faces) in cell_root.items():
                covered |= _inside_mask(dense, verts, faces)
            n_uncovered = int((~covered).sum())
            if n_uncovered:
                coverage_outside_count += n_uncovered
            assert n_uncovered == 0, f"seed {seed}: {n_uncovered} hull sample point(s) not covered by any cell"

        # 6. Every actually-overlapping pair is recorded in adjacent_cells,
        # and gather stats: overlap fraction, union-volume deficit
        # (reported, not asserted) via first-order inclusion-exclusion.
        names = sorted(cell_root)
        cell_vols = {n: geo.polytope_volume(*cell_root[n]) for n in names}
        sum_overlap = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                total_possible_pairs += 1
                b1, b2 = names[i], names[j]
                v1, f1 = cell_root[b1]
                v2, f2 = cell_root[b2]
                ov = geo._overlap_volume(v1, f1, v2, f2)
                rel_ov = ov / rel_scale
                if rel_ov > 1e-9:
                    total_overlap_pairs += 1
                    overlap_fracs.append(rel_ov)
                    assert frozenset((b1, b2)) in spec.adjacent_cells, (
                        f"seed {seed}: overlapping pair {b1!r}/{b2!r} not in adjacent_cells"
                    )
                    sum_overlap += ov
        incl_excl_estimate = sum(cell_vols.values()) - sum_overlap
        deficit_rel = (hull_vol - incl_excl_estimate) / rel_scale
        total_deficit_rel.append(deficit_rel)

    mean_overlap_frac = float(np.mean(overlap_fracs)) if overlap_fracs else 0.0
    mean_pairs_per_hand = total_overlap_pairs / N_SEEDS
    mean_deficit_rel = float(np.mean(total_deficit_rel))

    print(
        f"mean palm bodies/hand={total_palm / N_SEEDS:.2f} mean cells/hand={total_cells / N_SEEDS:.2f} "
        f"min cell volume={min_cell_vol!r} "
        f"mean pairwise overlap fraction={mean_overlap_frac:.6f} "
        f"mean overlapping pairs/hand={mean_pairs_per_hand:.3f} (of {total_possible_pairs / N_SEEDS:.2f} possible) "
        f"mean union-volume deficit (rel, hull - incl/excl estimate)={mean_deficit_rel:.6f} "
        f"mount-outside count={mount_outside_count} spine-outside count={spine_outside_count} "
        f"coverage-outside point count={coverage_outside_count}"
    )


# ---------------------------------------------------------------------------
# 2b. Regression (re-expressed for the nearest-spine contract, iteration
# 7c): the ORIGINAL plane-carving rule sliced off the parent's own
# remaining segment at a branch point (18/200 cells had zero volume,
# 104/200 hands had a digit mount outside its nominal cell); the whole
# plane-carving approach was later replaced entirely by nearest-spine
# assignment (see module docstring), under which a body's own spine/mounts
# are contained in its own cell BY CONSTRUCTION (its own capsule points are
# always unioned in) -- this fixture (root spine along z; a palm child
# mounts at frac 0.25 and branches off along x; a digit mounts on the root
# at frac 1.0, i.e. exactly at the root's own tip) is kept as a concrete
# regression guard for that guarantee.
# ---------------------------------------------------------------------------


def _synthetic_branch_model() -> KinematicModel:
    root_len = 0.04
    child_len = 0.03
    root = Body(name="root", palm=True, radius=0.01)
    child = Body(name="palm_child", palm=True, radius=0.01)
    digit = Body(name="d1p1", palm=False, radius=0.005)
    branch_j = Joint(
        name="palm_child_j", type="revolute", parent="root", child="palm_child",
        origin=Pose(xyz=(0.0, 0.0, 0.25 * root_len), rpy=(0.0, math.pi / 2.0, 0.0)),
        axis=(0.0, 1.0, 0.0), limits=(-0.5, 0.5),
    )
    digit_j = Joint(
        name="d1p1_j", type="revolute", parent="root", child="d1p1",
        origin=Pose(xyz=(0.0, 0.0, 1.0 * root_len), rpy=(0.0, 0.0, 0.0)),
        axis=(1.0, 0.0, 0.0), limits=(-0.5, 0.5),
    )
    frames = (
        Frame(name="root_tip", body="root", pose=Pose(xyz=(0.0, 0.0, root_len))),
        Frame(name="palm_child_tip", body="palm_child", pose=Pose(xyz=(0.0, 0.0, child_len))),
        Frame(name="d1p1_tip", body="d1p1", pose=Pose(xyz=(0.0, 0.0, 0.02))),
    )
    return KinematicModel(
        name="synthetic_branch", root="root", bodies=(root, child, digit),
        joints=(branch_j, digit_j), frames=frames,
    )


def _point_in_cell(cell, point) -> bool:
    verts = np.array(cell.vertices)
    for face in cell.faces:
        nvec, a = geo._face_outward_normal(verts, face)
        if nvec is None:
            continue
        d = float(np.dot(nvec, a))
        if float(np.dot(nvec, np.asarray(point))) - d > 1e-9:
            return False
    return True


def test_branch_at_root_mounts_and_spines_contained_in_own_cells():
    model = _synthetic_branch_model()
    spec = geo.build_geometry(model)

    root_cell = spec.cells["root"]
    child_cell = spec.cells["palm_child"]

    # The digit mount (root frame == root's own local frame, root is the
    # model's own root) and the root's own tip both lie in the root cell.
    digit_mount_local = np.array(model.joints[1].origin.xyz)  # joint parent==root
    assert _point_in_cell(root_cell, digit_mount_local)
    root_tip_local = np.array([0.0, 0.0, 0.04])
    assert _point_in_cell(root_cell, root_tip_local)

    # The child's own spine (its local (0,0,0)-(0,0,L)) lies in its own cell.
    child_len = 0.03
    for t in np.linspace(0.0, 1.0, 9):
        p = np.array([0.0, 0.0, t * child_len])
        assert _point_in_cell(child_cell, p), f"palm_child spine point {p} left its own cell"


# ---------------------------------------------------------------------------
# 3. Convex hull
# ---------------------------------------------------------------------------


def test_convex_hull_of_box_corners_has_box_volume():
    lo, hi = -0.5, 0.5
    corners = np.array([[x, y, z] for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)])
    verts, faces = geo.convex_hull_3d(corners)
    vol = geo.polytope_volume(verts, faces)
    assert vol == pytest.approx(1.0, rel=1e-9)


def test_convex_hull_contains_all_input_points():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.0, size=(60, 3))
    verts, faces = geo.convex_hull_3d(pts)
    for p in pts:
        inside = True
        for face in faces:
            nvec, a = geo._face_outward_normal(verts, face)
            if nvec is None:
                continue
            d = float(np.dot(nvec, a))
            if float(np.dot(nvec, p)) - d > 1e-9:
                inside = False
                break
        assert inside


# ---------------------------------------------------------------------------
# 4. Mass properties
# ---------------------------------------------------------------------------


def test_capsule_mass_reduces_to_sphere_at_l_zero():
    r = 0.01
    density = 1200.0
    cap = geo.Capsule(body="x", start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=r)
    mp = geo.capsule_mass_props(cap, density)
    m_expected = density * (4.0 / 3.0) * math.pi * r ** 3
    i_expected = (2.0 / 5.0) * m_expected * r * r
    assert mp.mass == pytest.approx(m_expected, rel=1e-12)
    I = np.array(mp.inertia)
    assert np.allclose(I, i_expected * np.eye(3), rtol=1e-9, atol=1e-15)


def test_capsule_mass_cylinder_limit():
    # Long, thin capsule: the cylinder's own transverse term should dominate
    # (hemisphere-cap corrections are a small, bounded fraction of it).
    r = 0.001
    L = 1.0
    density = 1000.0
    cap = geo.Capsule(body="x", start=(0.0, 0.0, 0.0), end=(0.0, 0.0, L), radius=r)
    mp = geo.capsule_mass_props(cap, density)
    I = np.array(mp.inertia)

    m_cyl = density * math.pi * r * r * L
    i_cyl_transverse = m_cyl * (3 * r * r + L * L) / 12.0
    # I about body ORIGIN (one end), not about COM -- shift the pure-cylinder
    # reference the same way (cylinder's own COM is at its own midpoint too).
    com = np.array(mp.com)
    r_vec = com  # COM relative to origin (start)
    i_cyl_origin = i_cyl_transverse + m_cyl * float(np.dot(r_vec, r_vec))

    assert I[0, 0] == pytest.approx(i_cyl_origin, rel=0.05)
    assert I[1, 1] == pytest.approx(i_cyl_origin, rel=0.05)


def test_cell_mass_matches_box_formula():
    lo, hi = -0.5, 0.5
    corners = {
        (lo, lo, lo): 0, (hi, lo, lo): 1, (hi, hi, lo): 2, (lo, hi, lo): 3,
        (lo, lo, hi): 4, (hi, lo, hi): 5, (hi, hi, hi): 6, (lo, hi, hi): 7,
    }
    verts = tuple(corners.keys())
    faces = (
        (0, 3, 2, 1),  # bottom (z=lo), outward normal -z
        (4, 5, 6, 7),  # top (z=hi), outward normal +z
        (0, 1, 5, 4),  # y=lo face, outward -y
        (2, 3, 7, 6),  # y=hi face, outward +y
        (0, 4, 7, 3),  # x=lo face, outward -x
        (1, 2, 6, 5),  # x=hi face, outward +x
    )
    cell = geo.Cell(body="box", vertices=verts, faces=faces)
    vol = geo.polytope_volume(np.array(cell.vertices), cell.faces)
    assert vol == pytest.approx(1.0, rel=1e-12)

    density = 7.0
    mp = geo.cell_mass_props(cell, density)
    assert mp.mass == pytest.approx(density * 1.0, rel=1e-12)
    assert np.allclose(mp.com, (0.0, 0.0, 0.0), atol=1e-12)
    expected = density * (1.0 + 1.0) / 12.0  # unit cube about its own center
    I = np.array(mp.inertia)
    assert np.allclose(I, expected * np.eye(3), rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------------------
# 5. URDF export
# ---------------------------------------------------------------------------


def test_export_without_geometry_unchanged():
    _derivation, model = generate(3)
    text_plain, _losses = to_urdf(model)
    text_again, _losses2 = to_urdf(model, geometry=None)
    assert text_plain == text_again
    # Sanity: no collision element anywhere without geometry.
    assert "<collision>" not in text_plain


def test_export_with_geometry_parses_and_references_meshes():
    _derivation, model = generate(3)
    spec = geo.build_geometry(model)
    text, _losses = to_urdf(model, geometry=spec)

    robot = URDF.from_xml_string(text)
    assert robot is not None

    n_mesh = text.count("<mesh ")
    assert n_mesh == len(spec.cells)
    assert n_mesh > 0

    n_capsule_bodies = len(spec.capsules)
    assert text.count("<cylinder ") == n_capsule_bodies
    assert text.count("<sphere ") == 2 * n_capsule_bodies

    with tempfile.TemporaryDirectory() as tmp:
        written = geo.write_geometry_meshes(spec, tmp)
        assert len(written) == len(spec.cells)
        for name, rel_path in written.items():
            p = Path(tmp) / rel_path
            assert p.is_file()
            assert f"package://{model.name}/{rel_path}" in text
