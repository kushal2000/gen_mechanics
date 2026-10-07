"""CPU tests for ``scene/projected_hands.py``: a projected commercial hand's
URDF joint -> envelope slot map, its URDF-equivalent placement (the pose,
spawn point and default joints its single-hand URDF asset uses), and the
mount-spanning palm hull for grammar designs. Kit-free."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf
from isaacsimenvs.inhand_reorient.scene import projected_hands as ph

REPO = Path(__file__).resolve().parents[3]
URDF = REPO / "hand_sampler/grammar_bench/fixtures/real/allegro_right/allegro_hand_description_right.urdf"
POSES = REPO / "isaacsimenvs/inhand_reorient/repose_hand_poses.json"


@pytest.fixture(scope="module")
def allegro():
    entry, status, reason = pf.projected_entry("allegro_right")
    assert status == "admitted", reason
    from hand_sampler.grammar.derive import derivation_from_dict, derive

    return ge.canonicalize(derive(derivation_from_dict(entry.derivation_dict)), source=entry.source)


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_hand_id_of_source():
    assert ph.hand_id_of("projected:allegro_right") == "allegro_right"
    assert ph.hand_id_of("arch:gen0-founder-000004") is None


def test_urdf_joints_map_onto_envelope_slots(allegro):
    m = ph.joint_slot_map(allegro, "allegro_right")
    assert len(m) == 16
    assert m["joint_0"] == ge.finger_slot(0, 0) and m["joint_3"] == ge.finger_slot(0, 3)  # index: finger 0
    assert m["joint_12"] == ge.finger_slot(3, 0) and m["joint_15"] == ge.finger_slot(3, 3)  # thumb: finger 3
    for name, slot in m.items():  # limits agree with the URDF
        lo, hi = allegro.slot_limits[slot]
        urdf = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
        assert lo == pytest.approx(float(urdf[name].find("limit").get("lower")), abs=1e-6)
        assert hi == pytest.approx(float(urdf[name].find("limit").get("upper")), abs=1e-6)


def _urdf_mounts_in_root():
    """Finger d==0 joint origins in the URDF root (hand_root) frame."""
    joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
    palm = np.array([float(v) for v in joints["root_to_base"].find("origin").get("xyz").split()])
    return {n: palm + np.array([float(v) for v in joints[n].find("origin").get("xyz").split()])
            for n in ("joint_0", "joint_4", "joint_8", "joint_12")}


def test_urdf_equivalent_placement_puts_the_mounts_where_the_urdf_hand_has_them(allegro):
    entry = json.loads(POSES.read_text())["allegro_right"]
    pl = ph.urdf_equivalent_placement(allegro, "allegro_right", entry)
    R_g = _quat_to_mat(pl["base_rot_wxyz"])
    R_u = _quat_to_mat(entry["base_rot"])
    t_g, t_u = np.asarray(pl["base_pos"]), np.asarray(entry["base_pos"])
    T0 = ge.authored_fk(allegro, np.zeros(ge.N_SLOTS))
    slots = ph.joint_slot_map(allegro, "allegro_right")
    for name, p_u in _urdf_mounts_in_root().items():
        world_u = t_u + R_u @ p_u
        world_g = t_g + R_g @ T0[slots[name]][:3, 3]
        assert np.linalg.norm(world_u - world_g) < 2e-3, (name, world_u, world_g)
    # spawn point: same world point
    s_u = t_u + R_u @ np.asarray(entry["spawn_offset_local"])
    s_g = t_g + R_g @ np.asarray(pl["spawn_offset"])
    assert np.linalg.norm(s_u - s_g) < 1e-6  # quaternion round-off
    # default joints: the entry's, on their slots; thumb rotation 0.28
    assert pl["default_q"][slots["joint_12"]] == pytest.approx(0.28, abs=2e-3)
    assert pl["default_q"][slots["joint_1"]] == pytest.approx(0.0, abs=2e-3)


def test_canonical_grasp_pose_maps_horas_pose_onto_slots(allegro):
    from isaacsimenvs.inhand_reorient import grasp_cache as gc

    q = ph.slot_values(allegro, "allegro_right", gc.CANONICAL_GRASP_POSES["allegro_right"])
    slots = ph.joint_slot_map(allegro, "allegro_right")
    assert q[slots["joint_13"]] == pytest.approx(1.163)
    assert q[slots["joint_1"]] == pytest.approx(1.244)
    assert len(q) == 16


def test_mount_hull_spans_the_finger_mounts_without_reaching_past_them(allegro):
    pts = ph.palm_hull_points(allegro, radius=0.01)
    assert pts.shape[1] == 3 and len(pts) <= 64
    T0 = ge.authored_fk(allegro, np.zeros(ge.N_SLOTS))
    mounts = np.stack([T0[b][:3, 3] for b in ge.FINGER_BASE_SLOTS if allegro.slot_valid[b]])
    assert pts[:, 1].min() <= mounts[:, 1].min() - 0.0099 and pts[:, 1].max() >= mounts[:, 1].max() + 0.0099
    # nothing past the mount height along the root axis, where the second phalanges start
    assert pts[:, 2].max() <= max(allegro.root_length_m, mounts[:, 2].max()) + 1e-9
    # thickness about the palm normal (root x): +-r around the root axis and the mounts
    assert pts[:, 0].max() == pytest.approx(mounts[:, 0].max() + 0.01)


def test_population_geometry_options_are_off_by_default():
    a = yaml.safe_load((REPO / "coevolution/cfg/task/InHandReorient.yaml").read_text())["anyrotate"]
    assert a["population_palm_collider"] == "capsule"
    assert a["population_capsule_radius"] < 0
    assert a["population_projected_pose"] is False
    assert a["grasp_projected_canonical"] is False


def test_signature_records_population_geometry_options_only_when_on():
    from types import SimpleNamespace

    from isaacsimenvs.inhand_reorient import grasp_cache as gc

    base = dict(object_shape="box", box_size=0.0525, capsule_radius=0.0295, capsule_width=0.006,
                static_friction=1.0, dynamic_friction=1.0, sim_dt=1 / 120, decimation=6, hand_disable_gravity=False,
                hand_stiffness=3.0, hand_damping=0.1, hand_effort_limit=0.5, object_contact_offset=0.002,
                object_rest_offset=0.0)
    off = gc.object_signature(SimpleNamespace(**base, population_palm_collider="capsule",
                                              population_capsule_radius=-1.0, population_projected_pose=False))
    on = gc.object_signature(SimpleNamespace(**base, population_palm_collider="mount_hull",
                                             population_capsule_radius=0.012, population_projected_pose=True))
    assert not any(k.startswith("population_") for k in off)
    assert on["population_palm_collider"] == "mount_hull" and on["population_capsule_radius"] == 0.012
    assert on["population_projected_pose"] is True


def test_palm_filter_slots_are_each_fingers_first_two_real_links(allegro):
    """mount_hull_filtered: the palm hull ignores each finger's first two
    real links (they start inside or next to it at rest)."""
    slots = ph.palm_filter_slots(allegro)
    assert slots == sorted(slots)
    assert {ge.finger_slot(f, d) for f in range(4) for d in (0, 1)} == set(slots)  # four fingers x (d0, d1)
    assert all(allegro.slot_valid[s] for s in slots)
