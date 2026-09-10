"""The palm the simulator never had.

``hand.palm`` decided where fingers mounted and nothing else: no collider, so an
object passed through it; no mass, though it outweighs every finger link put
together; and ``palm_center_offset`` aimed the policy at a centre with no body.
``GEN_PALM_DENSITY_KG_M3`` was defined for exactly this and referenced nowhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from hand_sampler import build, design_space, gen_init_pop
from hand_sampler import robot_param_constants as rpc


@pytest.fixture(scope="module")
def hands():
    return gen_init_pop.seed_population(0, 64)


def _hand_with_palm(palm, hands):
    from dataclasses import replace
    return replace(hands[0], palm=palm)


def test_sharpas_palm_weighs_what_sharpa_weighs(hands):
    """The density is back-derived from SHARPA, so this is the anchor."""
    mass, _ = build.palm_mass_props(_hand_with_palm(design_space.Palm(*rpc.PALM_EXTENTS_M), hands))
    assert mass == pytest.approx(rpc.PALM_MASS_KG, rel=1e-12)


def test_a_bigger_palm_weighs_more(hands):
    small, _ = build.palm_mass_props(_hand_with_palm(design_space.Palm(0.020, 0.050, 0.050), hands))
    big, _ = build.palm_mass_props(_hand_with_palm(design_space.Palm(0.025, 0.070, 0.060), hands))
    assert big > small > 0.0


def test_the_palm_is_most_of_the_hands_mass(hands):
    """Not a detail: leaving it out removed the majority of the hand.

    Per design it ranges from 38% to 80% of the total, so the floor is what to
    pin -- it outweighs every link put together in 82% of the population, but
    a small palm on long fingers does not, and asserting that per design is
    simply false.
    """
    fractions = []
    for hand in hands:
        palm, _ = build.palm_mass_props(hand)
        links = sum(build.link_mass_props(s.length)[0]
                    for f in hand.fingers for s in f.segments)
        fractions.append(palm / (palm + links))
    assert min(fractions) > 0.3, f"lightest palm is only {min(fractions):.1%} of its hand"
    assert sorted(fractions)[len(fractions) // 2] > 0.5


def test_the_palm_box_sits_where_palm_center_offset_says(hands):
    """One transform, two consumers: the observation walks link_7 -> palm centre
    through ``palm_center_offset``, and the collider must be at the same place."""
    for i, hand in enumerate(hands):
        _, pose = build.palm_box(hand)
        assert np.allclose(pose[:3, 3], build.palm_center_offset(hand), atol=1e-12), i


def test_the_palm_box_is_the_designs_own_extents(hands):
    for hand in hands:
        extents, _ = build.palm_box(hand)
        assert extents == hand.palm.extents


# --- merging into link_7 -----------------------------------------------------

def _box(mass, extents, centre, rot=np.eye(3)):
    tx, ty, tz = extents
    return (mass, np.asarray(centre, float),
            np.diag([mass * (ty**2 + tz**2) / 12.0,
                     mass * (tx**2 + tz**2) / 12.0,
                     mass * (tx**2 + ty**2) / 12.0]), rot)


def test_merging_one_body_changes_nothing():
    part = _box(2.0, (0.1, 0.2, 0.3), (0.4, -0.5, 0.6))
    mass, com, inertia = build.merge_rigid_bodies([part])
    assert mass == pytest.approx(part[0])
    assert np.allclose(com, part[1])
    assert np.allclose(inertia, part[2])


def test_two_half_boxes_merge_into_the_whole_box():
    """Analytic: two 0.1-long halves end to end are one 0.2-long box."""
    whole = _box(4.0, (0.2, 0.2, 0.3), (0.0, 0.0, 0.0))
    halves = [_box(2.0, (0.1, 0.2, 0.3), (-0.05, 0.0, 0.0)),
              _box(2.0, (0.1, 0.2, 0.3), (+0.05, 0.0, 0.0))]
    mass, com, inertia = build.merge_rigid_bodies(halves)
    assert mass == pytest.approx(whole[0])
    assert np.allclose(com, whole[1], atol=1e-15)
    assert np.allclose(inertia, whole[2], atol=1e-15)


def test_merging_does_not_depend_on_order():
    parts = [_box(0.7, (0.05, 0.06, 0.06), (0.0, 0.0, 0.03)),
             _box(2.3, (0.09, 0.09, 0.12), (0.01, -0.02, -0.04))]
    a = build.merge_rigid_bodies(parts)
    b = build.merge_rigid_bodies(parts[::-1])
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-15)


def test_a_rotated_part_is_rotated_into_the_shared_frame():
    """A box turned 90 deg about z is the same box with x and y swapped."""
    rot = design_space.rpy_to_mat((0.0, 0.0, np.pi / 2))
    turned = build.merge_rigid_bodies([_box(3.0, (0.1, 0.3, 0.2), (0.0, 0.0, 0.0), rot)])[2]
    swapped = _box(3.0, (0.3, 0.1, 0.2), (0.0, 0.0, 0.0))[2]
    assert np.allclose(turned, swapped, atol=1e-15)


def test_principal_axes_reconstruct_the_inertia():
    """USD stores a diagonal plus a rotation, so the pair must rebuild the tensor."""
    _, _, inertia = build.merge_rigid_bodies(
        [_box(0.7, (0.025, 0.06, 0.06), (0.0, 0.0, 0.03),
              design_space.rpy_to_mat((0.2, -0.4, 1.1))),
         _box(2.3, (0.09, 0.09, 0.12), (0.01, -0.02, -0.04))])
    diag, quat = build.principal_axes(inertia)
    rot = design_space.rpy_to_mat(design_space.mat_to_rpy(
        _quat_to_mat(quat)))
    assert np.allclose(rot @ np.diag(diag) @ rot.T, inertia, atol=1e-12)
    assert np.all(diag > 0.0)


def _quat_to_mat(quat):
    from scipy.spatial.transform import Rotation
    w, x, y, z = quat
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def test_the_merged_body_is_heavier_and_offset_toward_the_palm(hands):
    """A sanity check on the real numbers: link_7 plus a palm out along +z."""
    link7 = (2.0, np.zeros(3), np.diag([0.01, 0.01, 0.01]), np.eye(3))
    for i, hand in enumerate(hands):
        mass, pose = build.palm_mass_props(hand)[0], build.palm_box(hand)[1]
        palm = (mass, pose[:3, 3], build.palm_mass_props(hand)[1], pose[:3, :3])
        total, com, _ = build.merge_rigid_bodies([link7, palm])
        assert total == pytest.approx(link7[0] + mass)
        assert com[2] > 0.0, f"design {i}: palm is out along +z, so the centre moves that way"


def test_the_viewer_draws_the_palm_on_the_flange(hands, tmp_path):
    """The graft used to drop it: the hand's flange link was skipped whole
    because the arm already declared one, taking the palm with it."""
    import xml.etree.ElementTree as ET

    from coevolution import pose_viewer

    for i, hand in enumerate(hands[:8]):
        text = build.urdf_for_viewing(hand, tmp_path / "d.urdf").read_text(encoding="utf-8")
        root = ET.fromstring(pose_viewer._graft_hand_onto_arm(text, pose_viewer.ARM_URDF_PATH))
        flange = [l for l in root.findall("link")
                  if l.get("name") == rpc.ARM_TIP_LINK][0]
        boxes = flange.findall("visual/geometry/box")
        assert len(boxes) == 1, f"design {i}: palm missing from the flange"
        assert [float(v) for v in boxes[0].get("size").split()] == list(hand.palm.extents)
        # and the arm's own flange mesh survived the merge
        assert flange.findall("visual/geometry/mesh"), f"design {i}: arm geometry lost"

        links = [l.get("name") for l in root.findall("link")]
        assert len(links) == len(set(links)), f"design {i}: merged link duplicated"
