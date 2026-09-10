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


def test_links_are_drawn_as_the_capsules_they_are(pop, tmp_path):
    """The drawn union spans joint to next joint, caps included.

    URDF has no capsule, so the collider goes in as a cylinder plus two spheres.
    One bare cylinder of the full length draws flat ends over caps that are half
    a median link -- wrong exactly at the tip that does the touching.
    """
    radius = rpc.GEN_LINK_RADIUS_M
    for i, hand in enumerate(pop.hands):
        links = {l.get("name"): l for l in _viewing_root(hand, tmp_path).findall("link")}
        for f, finger in enumerate(hand.fingers):
            for d, seg in enumerate(finger.segments):
                spans = []
                for visual in links[f"f{f}_link{d}"].findall("visual"):
                    along = float(visual.find("origin").get("xyz").split()[0])
                    cylinder = visual.find("geometry/cylinder")
                    half = (float(cylinder.get("length")) / 2.0 if cylinder is not None
                            else float(visual.find("geometry/sphere").get("radius")))
                    spans.append((along - half, along + half))
                assert len(spans) == 3, f"design {i}, f{f}_link{d}: not a capsule"
                assert min(lo for lo, _ in spans) == pytest.approx(0.0, abs=1e-12)
                assert max(hi for _, hi in spans) == pytest.approx(seg.length, abs=1e-12)
                barrel = [hi - lo for lo, hi in spans if hi - lo != 2 * radius]
                assert barrel == pytest.approx(
                    [rpc.cylinder_part(seg.length, radius)]), f"design {i}, f{f}_link{d}"


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


# --- what the SIMULATOR is authored with ------------------------------------
# ``author_hand`` needs Kit, so these check the frames it writes rather than the
# prims. That gap is why an authored hand ran with half its hinges on the wrong
# axis and a third of its joints disagreeing with their own bodies at rest.

def _bodies_in_flange_frame(hand):
    """Where ``author_hand`` puts each body, in ``iiwa14_link_7``'s frame."""
    frames, mount = build.link_frames(hand), build.flange_to_palm()
    return {slot: mount @ m for slot, m in frames.items()}


def _embed(rot, translation=None) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    if translation is not None:
        out[:3, 3] = np.asarray(translation).reshape(3)
    return out


def test_joint_frames_coincide_at_rest(pop):
    """Both halves of a joint land in the same place once the bodies are placed.

    They did not: ``localRot0`` carried the palm-frame pose rather than the
    parent-relative one, so PhysX -- not ``link_frames`` -- decided where a
    finger's second link sat, by as much as 2 m.
    """
    for i, hand in enumerate(pop.hands):
        bodies = _bodies_in_flange_frame(hand)
        for (f, d), (frame0, frame1) in build.joint_local_frames(hand).items():
            body0 = bodies[(f, d - 1)] if d else np.eye(4)   # d == 0 hangs off the flange
            assert np.allclose(body0 @ frame0, bodies[(f, d)] @ frame1, atol=1e-12), \
                f"design {i}, slot f{f}_j{d}"


def test_hinge_is_the_axis_the_design_asks_for(pop):
    """The token axis, rotated into the joint frame, is ``axis_of``."""
    z = np.array([0.0, 0.0, 1.0])
    for i, hand in enumerate(pop.hands):
        frames = build.link_frames(hand)
        joints = build.joint_local_frames(hand)
        for f, finger in enumerate(hand.fingers):
            want = design_space.joint_axes(finger, hand.palm)
            for d in range(finger.n_joints):
                _, frame1 = joints[(f, d)]
                # frame1 sits in the child body, so take it out to the palm frame
                # and compare with what design_space says the hinge is.
                got = frames[(f, d)][:3, :3] @ frame1[:3, :3] @ z
                assert np.allclose(got, want[d], atol=1e-9), f"design {i}, slot f{f}_j{d}"


def test_hinge_to_z_spans_the_whole_axis_space(pop):
    """Including the poles: mutation walks theta continuously, not in steps."""
    z = np.array([0.0, 0.0, 1.0])
    for theta in np.linspace(0.0, np.pi, 23):
        for phi in np.linspace(1e-12, np.pi, 13):
            axis = design_space.axis_of(design_space.Joint(float(theta), float(phi)))
            rot = build.hinge_to_z(axis)
            assert np.allclose(rot @ z, axis, atol=1e-9)
            assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-12)
            assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_the_viewer_draws_the_same_hinge_as_the_simulator(pop, tmp_path):
    """One design, two consumers: the URDF axis and the USD frame must agree."""
    z = np.array([0.0, 0.0, 1.0])
    for i, hand in enumerate(pop.hands):
        joints = build.joint_local_frames(hand)
        root = _viewing_root(hand, tmp_path)
        for joint in root.findall("joint"):
            f, d = (int(part) for part in joint.get("name")[1:].split("_j"))
            urdf = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
            assert np.allclose(urdf, joints[(f, d)][1][:3, :3] @ z, atol=1e-9), \
                f"design {i}, slot f{f}_j{d}"


def test_the_template_tip_body_is_the_fingertip(pop):
    """Slot ``D-1`` must land on the real tip, for every real finger.

    ``reset.py`` gives a generated design a zero pad offset because "its capsule
    tip IS the pad". Ghosts stacked at their parent's BASE instead, leaving that
    body one whole link short and ``fingertip_pos_rel_palm`` wrong by 30-50 mm
    on every finger of every design.
    """
    last = design_space.MAX_JOINTS_PER_FINGER - 1
    for i, hand in enumerate(pop.hands):
        frames = build.link_frames(hand)
        for f, finger in enumerate(hand.fingers):
            assert np.allclose(frames[(f, last)][:3, 3],
                               design_space.fingertip(finger, hand.palm),
                               atol=1e-12), f"design {i}, finger {f}"


def test_a_ghost_finger_still_collapses_to_the_palm(pop):
    """Which is what ``fingertip_valid`` exists to mask; stepping out to a tip
    that does not exist must not move it."""
    for i, hand in enumerate(pop.hands):
        frames = build.link_frames(hand)
        for f in range(hand.n_fingers, design_space.MAX_FINGERS):
            for d in range(design_space.MAX_JOINTS_PER_FINGER):
                assert np.allclose(frames[(f, d)], np.eye(4)), f"design {i}, f{f}_link{d}"


def test_every_link_is_filtered_against_the_links_it_touches(pop):
    """Consecutive capsules are built to touch, so the pairs must go in.

    The fixed robot gets this from ``_apply_self_collision_filters`` editing its
    converted USD; an authored design has no file to edit, and had no filters
    at all.
    """
    root, palm = "/World/envs/env_0/Robot", "/World/envs/env_0/Robot/arm/iiwa14_link_7"
    pairs = build.filtered_pairs(root, palm)
    F, D = design_space.MAX_FINGERS, design_space.MAX_JOINTS_PER_FINGER

    for f in range(F):
        assert palm in pairs[f"f{f}_link0"], f"f{f}_link0 not filtered against the palm"
        assert f"{root}/f{f}_link0" in pairs[rpc.ARM_TIP_LINK]
        for d in range(1, D):
            assert f"{root}/f{f}_link{d - 1}" in pairs[f"f{f}_link{d}"]
            assert f"{root}/f{f}_link{d}" in pairs[f"f{f}_link{d - 1}"]

    # Only real prim paths: a target that resolves to nothing filters nothing.
    known = {palm} | {f"{root}/f{f}_link{d}" for f in range(F) for d in range(D)}
    for link, targets in pairs.items():
        assert set(targets) <= known, link
        assert len(targets) == len(set(targets)), f"{link} lists a pair twice"


def test_fingers_are_not_filtered_against_each_other(pop):
    """Only a joint's own two bodies are excluded. Two fingers closing on the
    same object must still collide, or a design gets a grasp for free."""
    root, palm = "/World/envs/env_0/Robot", "/World/envs/env_0/Robot/arm/iiwa14_link_7"
    pairs = build.filtered_pairs(root, palm)
    for f in range(design_space.MAX_FINGERS):
        for other in range(design_space.MAX_FINGERS):
            if other == f:
                continue
            for d in range(design_space.MAX_JOINTS_PER_FINGER):
                assert f"{root}/f{other}_link{d}" not in pairs[f"f{f}_link0"]
