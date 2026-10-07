"""Commercial hands: projection, fidelity readout, envelope fit, mesh overlay."""

import numpy as np
import pytest

from hand_sampler.grammar.experiments import e13_representation as e13
from hand_sampler.grammar.fk import forward_kinematics

from gviewer import commercial as com
from gviewer import meshes as gmesh

HANDS = ("allegro_right", "sharpa_left_on_iiwa14", "svh_right")


def _available(hand_id):
    try:
        return com.hand_entry(hand_id).availability == "available"
    except KeyError:
        return False


@pytest.fixture(scope="module")
def loaded():
    return {h: com.load_commercial(h) for h in HANDS if _available(h)}


@pytest.mark.parametrize("hand_id", HANDS)
def test_fidelity_equals_e13_check_hand(hand_id, loaded, monkeypatch):
    if hand_id not in loaded:
        pytest.skip(f"{hand_id} not on this machine")
    ch = loaded[hand_id]
    assert ch.ok, ch.error
    # E13's Pinocchio export cross-check spawns another interpreter; the viewer
    # does not run it, so compare everything else.
    monkeypatch.setattr(e13, "_pinocchio_export_check", lambda *a, **k: None)
    e = ch.entry
    ref = e13.check_hand(e.id, e.kin_path, e.hand_root, e.palm_joints, e.tip_frames)
    f = ch.fidelity
    assert f["max_pos_mm"] == pytest.approx(ref["max_pos_mm"], abs=1e-12)
    assert f["max_axis_deg"] == pytest.approx(ref["max_axis_deg"], abs=1e-12)
    if ref["max_tip_mm"] is None:
        assert f["max_tip_mm"] is None
    else:
        assert f["max_tip_mm"] == pytest.approx(ref["max_tip_mm"], abs=1e-12)
    assert f["n_tip_compared"] == ref["n_tip_compared"]
    assert f["joint_count_conserved"] == ref["joint_count_conserved"]
    assert f["passed"] == ref["passed"]


def test_envelope_fit_and_readout(loaded):
    if "allegro_right" in loaded:
        ch = loaded["allegro_right"]
        assert ch.fit == (True, ())
        assert ch.capsule_radius_m == pytest.approx(0.01)
        lines = ch.approximations()
        assert any("fixed joint(s) merged" in line for line in lines)
        assert ch.sha_ok is True
    if "svh_right" in loaded:
        ch = loaded["svh_right"]
        assert ch.fit == (True, ())          # two fingers on one palm joint fit the 36-slot envelope
        assert any("made independent motors" in line for line in ch.approximations())
    if "sharpa_left_on_iiwa14" in loaded:
        ch = loaded["sharpa_left_on_iiwa14"]
        assert ch.fit[0]
        assert any("articulated palm bodies" in line for line in ch.approximations())


@pytest.mark.parametrize("hand_id", HANDS)
def test_original_links_in_derived_frame(hand_id, loaded):
    """The mesh overlay poses: original links, driven through name_map, land
    on the projected bodies (the same identity E13 checks)."""
    if hand_id not in loaded:
        pytest.skip(f"{hand_id} not on this machine")
    ch = loaded[hand_id]
    rng = np.random.default_rng(3)
    u = {}
    for j in ch.derived.joints:
        if j.type == "revolute":
            u[j.name] = float(rng.uniform(*j.limits))
    poses = ch.orig_link_poses(u)
    T_der = forward_kinematics(ch.derived, u)
    for orig_joint, der_joint in ch.projection.name_map.items():
        if not der_joint.endswith("_j"):
            continue
        child = next(j.child for j in ch.imported.model.joints if j.name == orig_joint)
        assert np.allclose(poses[child][:3, 3], T_der[der_joint[:-2]][:3, 3], atol=1e-9)


def test_mesh_overlay_allegro():
    if not _available("allegro_right"):
        pytest.skip("allegro not on this machine")
    e = com.hand_entry("allegro_right")
    ch = com.load_commercial("allegro_right", with_fidelity=False)
    ms = gmesh.load_link_meshes(e.mesh_path, [b.name for b in ch.imported.model.bodies])
    assert ms.urdf_path is not None
    assert not ms.unresolved and not ms.unloadable
    assert len(ms.pieces) >= 17
    for pieces in ms.pieces.values():
        for p in pieces:
            assert p.vertices.dtype == np.float32 and p.faces.max() < len(p.vertices)


def test_cluster_decimate_reduces():
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=5, radius=0.02)
    V, F = gmesh.cluster_decimate(m.vertices, m.faces, budget=2000)
    assert len(F) <= 2000 and len(F) > 100
    assert np.abs(np.linalg.norm(V, axis=1) - 0.02).max() < 0.003
