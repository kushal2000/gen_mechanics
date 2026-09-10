"""The viewing URDF must be the articulation the simulator authored.

The browser viewer plays back ``joint_names_canonical`` and throws on the first
name its URDF does not carry, so a hand-only, design-sized URDF fails before it
draws anything -- which is how a population run logged 21 viewers to WandB that
all read `Joint "iiwa14_joint_1" not found in URDF.`
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from coevolution import pose_viewer
from hand_sampler import build, design_space, gen_init_pop
from hand_sampler import robot_param_constants as rpc
from hand_sampler.robot_spec import population_spec


@pytest.fixture(scope="module")
def pop():
    return population_spec(gen_init_pop.seed_population(0, 24), name="gen_s0_n24")


def _viewing_root(hand, tmp_path) -> ET.Element:
    return ET.parse(build.urdf_for_viewing(hand, tmp_path / "d.urdf")).getroot()


def test_envelope_is_complete(pop, tmp_path):
    """Every slot the template names, ghost or not, is a joint in the URDF."""
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    want = {f"f{f}_j{d}" for f in range(F) for d in range(D)}
    for i, hand in enumerate(pop.hands):
        names = {j.get("name") for j in _viewing_root(hand, tmp_path).findall("joint")}
        assert names == want, f"design {i}"


def test_forward_kinematics_matches_authored_bodies(pop, tmp_path):
    """At rest the URDF chain lands each body where ``author_hand`` puts it.

    The joint origin is the child's frame in the PARENT's; publishing the
    palm-frame pose instead left every link past the first at the wrong angle.
    """
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER
    mount = build.flange_to_palm()
    for i, hand in enumerate(pop.hands):
        origins = {}
        for joint in _viewing_root(hand, tmp_path).findall("joint"):
            o = joint.find("origin")
            m = np.eye(4)
            m[:3, :3] = design_space.rpy_to_mat(tuple(float(v) for v in o.get("rpy").split()))
            m[:3, 3] = [float(v) for v in o.get("xyz").split()]
            origins[joint.get("name")] = m

        frames = build.link_frames(hand)
        for f in range(F):
            acc = np.eye(4)
            for d in range(D):
                acc = acc @ origins[f"f{f}_j{d}"]
                assert np.allclose(acc, mount @ frames[(f, d)], atol=1e-9), \
                    f"design {i}, slot f{f}_j{d}"


def test_ghost_slots_carry_no_geometry(pop, tmp_path):
    """A ghost has a joint so the name resolves, and nothing to draw."""
    for i, hand in enumerate(pop.hands):
        root = _viewing_root(hand, tmp_path)
        links = {link.get("name"): link for link in root.findall("link")}
        for f in range(design_space.MAX_FINGERS):
            finger = hand.fingers[f] if f < hand.n_fingers else None
            for d in range(design_space.MAX_JOINTS_PER_FINGER):
                real = finger is not None and d < finger.n_joints
                drawn = links[f"f{f}_link{d}"].findall("visual")
                assert bool(drawn) is real, f"design {i}, slot f{f}_link{d}"


def test_graft_covers_the_canonical_joint_order(pop, tmp_path):
    """Grafted onto the arm, the URDF answers every name the viewer replays."""
    for i, hand in enumerate(pop.hands):
        text = build.urdf_for_viewing(hand, tmp_path / "d.urdf").read_text(encoding="utf-8")
        root = ET.fromstring(pose_viewer._graft_hand_onto_arm(text, pose_viewer.ARM_URDF_PATH))

        names = {j.get("name") for j in root.findall("joint")}
        assert set(pop.spec.joint_names_canonical) <= names, f"design {i}"

        # One tree, one root: a stray SHARPA link would ride along as a second.
        links = [link.get("name") for link in root.findall("link")]
        children = {j.find("child").get("link") for j in root.findall("joint")}
        assert len(links) == len(set(links))
        roots = [n for n in links if n not in children]
        assert len(roots) == 1 and roots[0].startswith(rpc.ARM_NAME), roots
