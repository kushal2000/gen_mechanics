"""A hand's URDF placed on the shared template tree: every link lands in its slot at the URDF's own pose, ghosts fill
the rest, nothing is lost. Kit-free, over the 100 random grammar hands."""
from __future__ import annotations

import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from isaacsimenvs.common.robots import get_robot_spec
from isaacsimenvs.common.robots.finger_layout import finger_layout, grid_for
from isaacsimenvs.common.robots.multi_hand import PALM, slot_link
from isaacsimenvs.common.robots.template_robot import (GHOST_LIMITS, WRIST_JOINTS, WRIST_LINKS, load_urdf_model,
                                                       template_robot)

ROOT = Path(__file__).resolve().parents[3]
GRAMMAR = ROOT / "assets/urdf/27oct_grammar_random_hands"
WRIST = types.SimpleNamespace(x_range=(-0.1, 0.1), y_range=(-0.1, 0.1), z_range=(0.02, 0.2), roll_range=(-1.5, 0.3),
                              pitch_range=(-1.2, 1.4), yaw_range=(-0.35, 0.52), prismatic_effort_limit=100.0,
                              prismatic_velocity_limit=1.0, revolute_effort_limit=20.0, revolute_velocity_limit=6.0,
                              link_mass=0.01)
MOUNT = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])    # a proper rotation, not the identity


def _hands():
    return sorted(p for p in GRAMMAR.iterdir() if p.is_dir())


def _spec(d):
    return get_robot_spec(str(d / f"{d.name}.spec.json"))


def test_urdf_model_keeps_every_link_collider_and_mass():
    d = GRAMMAR / "coarse_s0_000"
    model = load_urdf_model(str(d / f"{d.name}.urdf"), capsules=True)
    root = ET.parse(d / f"{d.name}.urdf").getroot()
    assert set(model.links) == {l.get("name") for l in root.findall("link")}
    for l in root.findall("link"):
        link = model.links[l.get("name")]
        assert len(link.colliders) == len(l.findall("collision"))
        assert link.mass == pytest.approx(float(l.find("inertial/mass").get("value")))
    assert model.root == "palm" and all(j.type == "revolute" for j in model.joints)
    mesh = next(c for c in model.links["palm"].colliders if c.kind == "mesh")
    assert mesh.points.shape[1] == 3 and len(mesh.faces) >= 4


def test_every_grammar_hand_fits_the_set_template_at_its_own_poses():
    specs = [_spec(d) for d in _hands()]
    lays = [finger_layout(s) for s in specs]
    grid = grid_for(lays)
    assert (grid.n_rigid, grid.n_moving) == (5, 2)
    SECTION = slot_link(grid.palm_slot)
    for spec, lay in zip(specs, lays):
        lay = lay.with_grid(grid)
        rob = template_robot(spec, lay, WRIST, MOUNT)
        names = [b.name for b in rob.bodies]
        assert names[:6] == list(WRIST_LINKS) and names[6:8] == [PALM, SECTION]
        assert len(rob.bodies) == 8 + grid.n_fingers * grid.n_slots and len(rob.joints) == len(rob.bodies) - 1
        assert [j.name for j in rob.joints[:6]] == list(WRIST_JOINTS)
        by_name = {b.name: b for b in rob.bodies}
        for j in rob.joints:                                   # the tree the joints describe is the bodies' tree
            assert by_name[j.body1].parent == j.body0 and by_name[j.body1].joint == j.name
        model = load_urdf_model(spec.urdf_path, True)
        fk = model.fk()
        mount = np.eye(4)
        mount[:3, :3] = MOUNT
        for link, body in rob.body_of_link.items():            # every URDF link, at the URDF's pose under the mount
            b = by_name[body]
            assert b.real and b.urdf_link == link
            assert np.allclose(b.pose, mount @ fk[link], atol=1e-9)
            assert b.mass == pytest.approx(model.links[link].mass)
            assert len(b.colliders) == len(model.links[link].colliders)
        assert set(rob.body_of_link) == set(model.links)
        ghosts = [b for b in rob.bodies if not b.real]
        assert all(not b.colliders and b.mass < 1e-5 for b in ghosts)
        ghost_joints = [j for j in rob.joints if not j.real]
        assert len(ghost_joints) == len(ghosts) and all((j.lower, j.upper) == GHOST_LIMITS for j in ghost_joints)
        real_hand_joints = [j for j in rob.joints[6:] if j.real]
        assert len(real_hand_joints) == spec.num_hand_joints
        # the fingertip body is the finger's last slot's link
        for f, tip in enumerate(spec.fingertip_body_names):
            assert rob.body_of_link[tip] == slot_link(lay.tip_slot_of_finger[f])
        # a cut hand: a real section, its moving fingers hanging off it; otherwise a ghost section
        if lay.palm:
            assert by_name[SECTION].real
            for f, moving in enumerate(lay.moving):
                base = by_name[slot_link(lay.slot_of_joint[lay.fingers[f][0]])]
                assert (base.parent == SECTION) == moving
        else:
            assert not by_name[SECTION].real
        # adjacent links never collide: the spec's pairs, in template names
        assert rob.filtered_pairs[PALM] and all(n in by_name for v in rob.filtered_pairs.values() for n in v)
