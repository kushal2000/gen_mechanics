"""Representation-check plan, item 4: acceptance tests for
``hand_sampler.grammar.experiments.e14_cross_section`` and
``hand_sampler.grammar_bench.refgen.mesh_sections``.

Two kinds of test, matching the module split (mesh-shape-fitting math needs
only numpy; mesh I/O needs ``trimesh``/``yourdfpy``, absent on the system
Python this suite normally runs under):

1. Pure-numpy synthetic geometry: a circle section fits the capsule (circle)
   model exactly; a rounded-rectangle section fits the rounded-rectangle
   template closely (to the coordinate-descent fit's own convergence
   tolerance -- see ``mesh_sections.fit_rounded_rect``'s docstring); the
   rounded-rect SDF degenerates to the circle SDF at ``w == h == 2r``;
   ``resolve_mesh_filename``'s package://-URI and relative-path resolution
   against a synthetic directory tree (no mesh library needed at all).
2. A real-hand smoke test (SHARPA, whose STL collision meshes ship in this
   repo, not ``~/karma/...``) through the full ``run_e14`` pipeline --
   skipped with a ``local-only:`` reason on the system Python that has no
   ``trimesh``/``yourdfpy`` (matching the plan's "skip with local-only: when
   meshes are absent" -- absent from THIS interpreter is treated the same
   as absent from disk, since either way no section can be produced here).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hand_sampler.grammar.experiments import e14_cross_section as e14
from hand_sampler.grammar_bench.refgen import mesh_sections as ms

# ---------------------------------------------------------------------------
# 1a. A circle section fits the capsule (circle) model exactly.
# ---------------------------------------------------------------------------


def _sample_circle(r: float, n: int = 256) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def test_circle_section_fits_capsule_exactly():
    r_true = 12.34
    xy = _sample_circle(r_true)

    eq = ms.fit_circle_equal_area(xy)
    assert eq.radius == pytest.approx(r_true, rel=1e-3)
    assert eq.mean_err < 0.01
    assert eq.max_err < 0.01
    assert eq.area_ratio == pytest.approx(1.0, rel=1e-9)

    p95 = ms.fit_circle_p95(xy)
    assert p95.radius == pytest.approx(r_true, rel=1e-3)
    assert p95.mean_err < 0.01
    assert p95.max_err < 0.01


# ---------------------------------------------------------------------------
# 1b. A rounded-rectangle section fits the rounded-rectangle template
# closely (a coordinate-descent fit, not a closed-form solve -- see
# ``fit_rounded_rect``'s own docstring on why "closely" rather than "to
# machine precision" is the honest claim here).
# ---------------------------------------------------------------------------


def _sample_rounded_rect(w: float, h: float, r: float, n_per_corner: int = 100) -> np.ndarray:
    bx, by = w / 2.0 - r, h / 2.0 - r
    corners = [(bx, by), (-bx, by), (-bx, -by), (bx, -by)]
    pts = []
    for i, (cx, cy) in enumerate(corners):
        ang0 = i * math.pi / 2.0
        for t in np.linspace(0.0, math.pi / 2.0, n_per_corner, endpoint=False):
            ang = ang0 + t
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return np.asarray(pts, dtype=float)


def test_rounded_rect_section_fits_template_closely():
    w_true, h_true, r_true = 20.0, 10.0, 3.0
    xy = _sample_rounded_rect(w_true, h_true, r_true)

    fit = ms.fit_rounded_rect(xy)
    assert fit.w == pytest.approx(w_true, rel=0.02)
    assert fit.h == pytest.approx(h_true, rel=0.03)
    assert fit.r == pytest.approx(r_true, rel=0.1)
    assert fit.mean_err < 0.05
    assert fit.max_err < 0.15


def test_rounded_rect_sdf_degenerates_to_circle_when_w_eq_h_eq_2r():
    r = 7.5
    xy = np.array([[0.3, 0.0], [0.0, 0.3], [5.0, 6.0], [-4.0, 2.0], [1.0, 1.0]])
    rr = ms.sdf_rounded_rect(xy, w=2 * r, h=2 * r, r=r)
    circ = ms.sdf_circle(xy, r)
    np.testing.assert_allclose(rr, circ, atol=1e-9)


# ---------------------------------------------------------------------------
# 1c. Mesh-filename resolution (pure pathlib; needs no mesh library).
# ---------------------------------------------------------------------------


def test_resolve_mesh_filename_package_and_relative(tmp_path):
    hand_dir = tmp_path / "SomeHand"
    (hand_dir / "meshes").mkdir(parents=True)
    mesh_file = hand_dir / "meshes" / "link.obj"
    mesh_file.write_text("# empty obj\n")
    urdf_dir = hand_dir / "urdf"
    urdf_dir.mkdir()

    # package://<pkg>/<...>/meshes/link.obj -- the real prefix (pkg name and
    # any intermediate directories that do not exist on disk here) must be
    # stripped by suffix search, exactly as Allegro's
    # "package://drake/manipulation/models/allegro_hand_description/meshes/
    # base_link.obj" needs (see e14_cross_section.py's module docstring).
    found = ms.resolve_mesh_filename(
        "package://some_pkg/manipulation/models/some_hand/meshes/link.obj", urdf_dir,
    )
    assert found == mesh_file

    # Plain relative path, resolved from the URDF's own directory upward
    # (Wuji's "../meshes/right/x.STL" pattern).
    found_rel = ms.resolve_mesh_filename("../meshes/link.obj", urdf_dir)
    assert found_rel == mesh_file

    # Unresolvable URI -> None (never a false match).
    assert ms.resolve_mesh_filename("package://nope/nothing/here.obj", urdf_dir) is None


# ---------------------------------------------------------------------------
# 2. Real-hand smoke test through the full run_e14 pipeline (SHARPA: STL
# collision meshes ship in this repo). local-only skip if trimesh/yourdfpy
# is not importable on this interpreter (the system Python this suite
# normally runs under has neither -- see module docstring).
# ---------------------------------------------------------------------------


def test_e14_sharpa_smoke_pipeline():
    try:
        import trimesh  # noqa: F401
        import yourdfpy  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"local-only:trimesh/yourdfpy not importable on this interpreter ({exc})")

    out = e14.run_e14(20260926, hand_ids=["sharpa_left_on_iiwa14"])
    per_hand = out["_per_hand"]
    assert len(per_hand) == 1
    hand = per_hand[0]
    if hand["availability"] != "available":
        pytest.skip(f"local-only:sharpa_left_on_iiwa14 {hand.get('reason')}")

    assert hand["n_links"] > 0
    assert hand["n_sections"] > 0
    assert out["n_sections_total"] == hand["n_sections"]

    n_checked = 0
    for link in hand["links"]:
        for sec in link["sections"]:
            if sec.get("skipped"):
                continue
            n_checked += 1
            assert sec["n_boundary_points"] >= 3
            for block in ("circle_equal_area", "circle_p95", "rounded_rect"):
                for k, v in sec[block].items():
                    if v is not None:
                        assert math.isfinite(v), f"{block}.{k} is not finite: {v}"
            assert sec["circle_equal_area"]["radius_mm"] > 0
            assert sec["rounded_rect"]["w_mm"] > 0 and sec["rounded_rect"]["h_mm"] > 0
    assert n_checked > 0

    ts = out["_template_search"]
    assert ts["n_sections"] == n_checked
    for key in ("best_template", "per_link_radius_capsule", "per_hand_radius_capsule", "global_radius_capsule"):
        assert key in ts
    bt = ts["best_template"]
    assert 0.0 <= bt["roundedness_r_over_h"] <= 0.5
    for tol_key, frac in bt["fractions_within_tol"].items():
        assert 0.0 <= frac <= 1.0
