"""Render primitives built from a model (no browser)."""

import math

import numpy as np
import pytest

from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.grammar.fk import forward_kinematics
from hand_sampler.grammar.geometry import build_geometry
from hand_sampler.grammar.variants import NAMED_DISTRIBUTIONS

from gviewer import model as gm
from gviewer.envload import load_env_modules


def _model(variant="G_V3S", seed=8):
    return derive(sample_derivation(seed, NAMED_DISTRIBUTIONS[variant]))


def test_primitives_follow_fk():
    m = _model()
    view = gm.ModelView.build(m)
    rng = np.random.default_rng(0)
    u = gm.random_u(view.ranges, rng)
    prims = view.primitives(u)
    T = forward_kinematics(m, gm.expand_q(m, u))
    assert set(prims.bodies) == {b.name for b in m.bodies}
    for name, b in prims.bodies.items():
        assert np.allclose(b.T, T[name])
        assert b.radius == pytest.approx(view.radius)
    movable = [j for j in m.joints if j.type != "fixed"]
    assert [jp.name for jp in prims.joints] == [j.name for j in movable]
    for jp, j in zip(prims.joints, movable):
        a = T[j.child][:3, :3] @ np.asarray(j.axis)
        assert np.allclose(jp.position, T[j.child][:3, 3])
        assert np.allclose(jp.axis, a / np.linalg.norm(a))
    assert [t.frame for t in prims.tips] == view.tips
    for t in prims.tips:
        assert np.allclose(t.position, T[t.frame][:3, 3])
    lo, hi = prims.bounds()
    assert np.all(hi >= lo)


def test_slider_ranges_and_poses():
    m = _model("G_FULL", 1)  # has a continuous joint
    view = gm.ModelView.build(m)
    types = {r.type for r in view.ranges}
    assert "continuous" in types
    for r in view.ranges:
        assert r.lo <= r.hi
        if r.type == "continuous":
            assert (r.lo, r.hi) == gm.CONTINUOUS_SLIDER_RANGE
    zero = gm.clamp_u(view.ranges, {})
    full = gm.curl_u(view.ranges, 1.0)
    assert all(full[r.name] == pytest.approx(r.hi) for r in view.ranges)
    assert all(view.ranges[i].lo <= zero[r.name] <= view.ranges[i].hi for i, r in enumerate(view.ranges))
    big = gm.clamp_u(view.ranges, {r.name: 100.0 for r in view.ranges})
    assert all(big[r.name] == pytest.approx(r.hi) for r in view.ranges)


def test_structure_counts():
    m = _model("G_FULL", 0)  # 6 digits with branches, 3 palm bodies
    st = gm.structure(m)
    assert len(st.palm_bodies) == 1 + sum(1 for b in m.bodies if b.palm and b.name != "root")
    assert len(st.top_level_digits) == 6
    assert st.branch_digits, "seed 0 of G_FULL has branch digits"
    covered = {b for d in st.digits for b in d.bodies}
    assert covered == {b.name for b in m.bodies if not b.palm}
    assert st.n_movable == sum(1 for j in m.joints if j.type != "fixed")


@pytest.mark.parametrize("seed", [0, 4])
def test_palm_cells_match_build_geometry(seed):
    m = _model("G_FULL", seed)
    ref = build_geometry(m).cells
    cells = gm.palm_cells(m)
    assert set(cells) == set(ref)
    for name, c in cells.items():
        assert np.allclose(c.vertices, np.asarray(ref[name].vertices), atol=1e-12)
        assert c.faces.shape[1] == 3 and c.faces.max() < len(c.vertices)


def test_capsule_conventions():
    ge = load_env_modules().grammar_envelope
    for L, r in ((0.04, 0.01), (0.015, 0.012), (0.0001, 0.01)):
        assert gm.capsule_core(L, r, "simulator") == pytest.approx(ge.capsule_core_endpoints_local(max(L, 1e-6), r),
                                                                   abs=1e-9)
        assert gm.capsule_core(L, r, "grammar") == (0.0, L)
        V, F = gm.capsule_mesh_local(L, r, "simulator")
        assert V[:, 2].min() == pytest.approx(min(0.0, L / 2 - r), abs=1e-6)
        assert V[:, 2].max() == pytest.approx(max(L, L / 2 + r), abs=1e-6)
        assert np.hypot(V[:, 0], V[:, 1]).max() <= r + 1e-7
        assert F.max() < len(V)
        Vg, _ = gm.capsule_mesh_local(L, r, "grammar")
        assert Vg[:, 2].min() == pytest.approx(-r, abs=1e-6) and Vg[:, 2].max() == pytest.approx(L + r, abs=1e-6)


def test_quaternion_roundtrip():
    rng = np.random.default_rng(1)
    from hand_sampler.grammar.fk import rpy_to_matrix
    from viewer_full import quat_to_mat  # the viewer's inverse
    for _ in range(50):
        R = rpy_to_matrix(rng.uniform(-math.pi, math.pi, 3))
        q = gm.mat_to_wxyz(R)
        assert np.allclose(quat_to_mat(q), R, atol=1e-9)


def test_joint_classes_allegro():
    from gviewer import commercial as com
    ch = com.load_commercial("allegro_right", with_fidelity=False)
    view = gm.ModelView.build(ch.derived)
    kinds = [view.joint_classes[f"d{d}p{p}_j"].kind for d in (1, 2, 3) for p in (2, 3, 4)]
    assert kinds == ["flexion"] * 9  # index/middle/ring flexion joints
    # the first joint of each finger rotates about the finger's own long axis at rest
    assert {view.joint_classes[f"d{d}p1_j"].kind for d in (1, 2, 3)} == {"twist"}
