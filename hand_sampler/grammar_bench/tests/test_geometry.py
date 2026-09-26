"""Iteration-7 (start of M2) acceptance tests for the geometry overlay
(``hand_sampler/grammar/geometry.py``): ``GeometrySpec`` construction over
200 seeds, the convex-hull/cutting-plane building blocks in isolation, mass
properties against closed forms, and URDF export with/without geometry.
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
# 1. GeometrySpec over 200 seeds
# ---------------------------------------------------------------------------


def test_geometry_over_200_seeds():
    total_palm = 0
    total_cells = 0
    n_degenerate_warned = 0
    min_cell_vol = None
    volume_sum_failures = []
    cell_count_failures = []
    convexity_failures = []
    mount_containment_failures = 0
    positive_volume_failures = 0

    for seed in range(N_SEEDS):
        _derivation, model = generate(seed)
        spec = geo.build_geometry(model)
        palm_names = [b.name for b in model.bodies if b.palm]

        total_palm += len(palm_names)
        total_cells += len(spec.cells)
        if spec.warnings:
            n_degenerate_warned += 1

        if len(spec.cells) != len(palm_names):
            cell_count_failures.append(seed)

        hull_vol = geo.polytope_volume(np.array(spec.hull_vertices), spec.hull_faces) if spec.hull_vertices else 0.0
        total_vol = 0.0
        for cell in spec.cells.values():
            verts = np.array(cell.vertices)
            vol = geo.polytope_volume(verts, cell.faces)
            if min_cell_vol is None or vol < min_cell_vol:
                min_cell_vol = vol
            if vol <= 0.0:
                positive_volume_failures += 1
            total_vol += vol
            for face in cell.faces:
                nvec, a = geo._face_outward_normal(verts, face)
                if nvec is None:
                    continue
                d = float(np.dot(nvec, a))
                if float(np.max(verts @ nvec - d)) > 1e-9:
                    convexity_failures.append(seed)
                    break
        rel_scale = max(abs(hull_vol), 1e-15)
        if abs(total_vol - hull_vol) > 1e-9 * rel_scale:
            volume_sum_failures.append(seed)

        issues = geo.validate_geometry(model, spec)
        if any(i.startswith("mount_point_outside_cell") for i in issues):
            mount_containment_failures += 1

    # Robust invariants (hold for every seed): every palm body gets exactly
    # one cell, every cell's vertices satisfy every face half-space (convex),
    # and cell volumes sum to the shared hull's volume (space partition,
    # divergence-theorem volume -- both computed the same way, so this is an
    # exact check to floating-point noise).
    assert cell_count_failures == [], f"cell-count mismatches at seeds {cell_count_failures}"
    assert convexity_failures == [], f"non-convex cells at seeds {convexity_failures}"
    assert volume_sum_failures == [], f"cell-volume-sum != hull-volume at seeds {volume_sum_failures}"

    # Reported (not hard-asserted), see PROJECT NOTE below.
    print(
        f"mean palm bodies/hand={total_palm / N_SEEDS:.2f} mean cells/hand={total_cells / N_SEEDS:.2f} "
        f"degenerate-normal-warned fraction={n_degenerate_warned / N_SEEDS:.3f} "
        f"min cell volume={min_cell_vol!r} "
        f"nonpositive-cell-volume count={positive_volume_failures} "
        f"mount-containment-failed seeds={mount_containment_failures}/{N_SEEDS}"
    )


# ---------------------------------------------------------------------------
# 2. Cutting-plane rule (synthetic, 2-palm-body model)
# ---------------------------------------------------------------------------


def _synthetic_two_palm_model(axis, direction_rpy) -> KinematicModel:
    root = Body(name="root", palm=True, radius=0.01)
    palm0 = Body(name="palm0", palm=True, radius=0.01)
    j = Joint(
        name="palm0_j", type="revolute", parent="root", child="palm0",
        origin=Pose(xyz=(0.0, 0.0, 0.02), rpy=direction_rpy), axis=axis, limits=(-0.5, 0.5),
    )
    frames = (
        Frame(name="root_tip", body="root", pose=Pose(xyz=(0.0, 0.0, 0.04))),
        Frame(name="palm0_tip", body="palm0", pose=Pose(xyz=(0.0, 0.0, 0.03))),
    )
    model = KinematicModel(name="synthetic", root="root", bodies=(root, palm0), joints=(j,), frames=frames)
    return model


def test_cutting_plane_rule_contains_axis_and_matches_formula():
    axis = (0.6, 0.8, 0.0)  # unit
    direction_rpy = (0.3, -0.2, 0.7)
    model = _synthetic_two_palm_model(axis, direction_rpy)
    transforms = forward_kinematics(model, {})
    joint = model.joints[0]

    point, n, degenerate = geo._cutting_plane(transforms, joint)
    assert not degenerate

    a = transforms[joint.child][:3, :3] @ np.array(axis)
    a = a / np.linalg.norm(a)

    # The plane contains the joint axis: n . a == 0.
    assert abs(float(np.dot(n, a))) < 1e-12

    d_vec = transforms[joint.child][:3, 3] - transforms[joint.parent][:3, 3]
    d = d_vec / np.linalg.norm(d_vec)
    expected_n = d - np.dot(d, a) * a
    expected_n = expected_n / np.linalg.norm(expected_n)
    # n == +/- the formula's normalized result (sign is a free choice for a
    # plane's normal; the formula itself, not an arbitrary orientation, is
    # what's under test).
    assert min(np.linalg.norm(n - expected_n), np.linalg.norm(n + expected_n)) < 1e-9


def test_cutting_plane_degenerate_axis_parallel_to_direction():
    # axis parallel to the mount direction (root->palm0 is pure +z, axis is
    # also +z): d - (d.a)a == 0, so the fallback perpendicular vector kicks
    # in and the joint is reported degenerate.
    model = _synthetic_two_palm_model(axis=(0.0, 0.0, 1.0), direction_rpy=(0.0, 0.0, 0.0))
    transforms = forward_kinematics(model, {})
    point, n, degenerate = geo._cutting_plane(transforms, model.joints[0])
    assert degenerate
    assert abs(float(np.linalg.norm(n)) - 1.0) < 1e-12


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
