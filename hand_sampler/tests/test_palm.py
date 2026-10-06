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


def test_a_palm_the_area_of_sharpas_weighs_what_sharpa_weighs(hands):
    """The density is back-derived from SHARPA, so this is the anchor.

    Taken on AREA now rather than on a bounding box: a radial palm is a rounded
    hull, and the rectangle round it is a third bigger. A palm of SHARPA's own
    footprint still weighs what SHARPA's does.
    """
    tx, ty, tz = rpc.PALM_EXTENTS_M
    hand = _hand_with_palm(design_space.Palm(tx), hands)
    mass = rpc.GEN_PALM_DENSITY_KG_M3 * tx * (ty * tz)
    assert mass == pytest.approx(rpc.PALM_MASS_KG, rel=1e-12)
    # and the real one scales with the outline it actually has
    got, _ = build.palm_mass_props(hand)
    assert got == pytest.approx(
        rpc.GEN_PALM_DENSITY_KG_M3 * tx * design_space.palm_area(hand), rel=1e-9)


def test_a_bigger_palm_weighs_more(hands):
    small, _ = build.palm_mass_props(_hand_with_palm(design_space.Palm(0.020), hands))
    big, _ = build.palm_mass_props(_hand_with_palm(design_space.Palm(0.025), hands))
    assert big > small > 0.0


def test_the_palm_is_most_of_the_hands_mass(hands):
    """Not a detail: leaving it out removed the majority of the hand.

    Per design it runs 33% to 61% of the total, so the floor is what to pin: a
    small palm on long fingers does not outweigh its own links, and asserting
    that per design is simply false.

    It used to run 38% to 80%, and outweigh every link put together in 82% of
    the population rather than 16%. The palm did not lose a job, it lost 20 mm
    of plate outboard of every base -- PALM_RIM replaced growing the hull by
    PALM_MIN_RADIUS, so the plate stops at the fingers instead of reaching a
    whole first link past them.
    """
    fractions = []
    for hand in hands:
        palm, _ = build.palm_mass_props(hand)
        links = sum(build.link_mass_props(s.length)[0]
                    for f in hand.fingers for s in f.segments)
        fractions.append(palm / (palm + links))
    assert min(fractions) > 0.3, f"lightest palm is only {min(fractions):.1%} of its hand"
    assert sorted(fractions)[len(fractions) // 2] > 0.4


def test_the_palm_box_sits_where_palm_center_offset_says(hands):
    """One transform, two consumers: the observation walks link_7 -> palm centre
    through ``palm_center_offset``, and the collider must be at the same place."""
    for i, hand in enumerate(hands):
        _, pose = build.palm_box(hand)
        assert np.allclose(pose[:3, 3], build.palm_center_offset(hand), atol=1e-12), i


def test_the_palm_box_is_the_designs_own_extents(hands):
    for hand in hands:
        extents, _ = build.palm_box(hand)
        assert extents == design_space.palm_extents(hand)


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
        # the palm is a convex hull MESH now, not a box
        meshes = [m for m in flange.findall("visual/geometry/mesh")
                  if "_palm.obj" in (m.get("filename") or "")]
        assert len(meshes) == 1, f"design {i}: palm missing from the flange"
        # and the arm's own flange mesh survived the merge
        assert flange.findall("visual/geometry/mesh"), f"design {i}: arm geometry lost"

        links = [l.get("name") for l in root.findall("link")]
        assert len(links) == len(set(links)), f"design {i}: merged link duplicated"


# --- palm keypoints: the slab as the policy reads it -------------------------

def _box_from_keypoints(kp):
    """``(centre, edge vectors)`` of the box four keypoints describe."""
    kp = np.asarray(kp, float)
    edges = kp[1:] - kp[0]
    return kp[0] + 0.5 * edges.sum(axis=0), edges


def test_palm_keypoints_are_the_authored_slab(hands):
    """Same encoding as a link box -- a corner and its three neighbours -- and
    the box they span is the collider ``palm_box`` authors, edge for edge."""
    for i, hand in enumerate(hands):
        kp = build.palm_keypoints_of(hand)
        assert kp.shape == (4, 3), i
        centre, edges = _box_from_keypoints(kp)
        extents, pose = build.palm_box(hand)
        assert np.allclose(centre, pose[:3, 3], atol=1e-6), i
        # Each edge runs along one of the palm's axes as rotated into link_7.
        assert np.allclose(np.linalg.norm(edges, axis=1), extents, atol=1e-6), i
        assert np.allclose(edges / np.linalg.norm(edges, axis=1, keepdims=True), pose[:3, :3].T, atol=1e-6), i


def test_palm_keypoints_are_in_the_end_effector_frame_not_the_palms(hands):
    """Measured from the end effector, whatever the design.

    There is no wrist FACE to anchor to any more: a radial palm is centred on
    its own origin rather than growing forward from the wrist, so what is fixed
    is the flange stack the palm hangs off, and the keypoints straddle the
    palm's own centre by half its derived depth.
    """
    z_flange = rpc.LINK7_TO_FLANGE_Z_M + rpc.FLANGE_TO_PALM_Z_M
    for hand in hands:
        kp = build.palm_keypoints_of(hand)
        depth = design_space.palm_extents(hand)[2]
        centre_z = design_space.palm_center(hand)[2]
        assert kp[0, 2] == pytest.approx(z_flange + centre_z - depth / 2.0, abs=1e-6)
        assert kp[3, 2] == pytest.approx(z_flange + centre_z + depth / 2.0, abs=1e-6)


def test_palm_keypoints_move_with_the_mounts(hands):
    """A palm IS where its fingers start, so its keypoints move when they do.

    The old property -- two designs with the same palm reading identically --
    has no referent: there is no palm to share. Identical mounts read
    identically, and moving one moves the palm.
    """
    from dataclasses import replace
    a = hands[0]
    same = replace(a, fingers=tuple(replace(f, segments=f.segments)
                                    for f in a.fingers))
    assert np.allclose(build.palm_keypoints_of(a), build.palm_keypoints_of(same))
    # a palm has no length of its own; it changes when a MOUNT moves
    moved = replace(a, fingers=(replace(a.fingers[0],
                                        mount=replace(a.fingers[0].mount,
                                                      radius=a.fingers[0].mount.radius
                                                      + 0.010)),)
                    + a.fingers[1:])
    assert not np.allclose(build.palm_keypoints_of(moved),
                           build.palm_keypoints_of(a)), \
        "moving a mount should have moved the palm"


def test_sharpas_palm_keypoints_sit_at_its_measured_box():
    """SHARPA's box is measured in left_hand_C_MC; through the same URDF chain
    (link_7 -> flange 0.045 -> mount +15 deg -> palm +0.05, -90 deg) it lands
    ~14 cm out along the flange axis, and its edges are the measured extents."""
    kp = build.palm_keypoints(rpc.PALM_BOX_CENTER_M, rpc.PALM_EXTENTS_M)
    centre, edges = _box_from_keypoints(kp)
    assert centre[2] == pytest.approx(0.095 + rpc.PALM_BOX_CENTER_M[2], abs=1e-6)
    assert np.allclose(np.linalg.norm(edges, axis=1), rpc.PALM_EXTENTS_M, atol=1e-9)


def test_no_mount_on_the_grid_can_put_the_PALM_in_the_arm():
    """The plate itself must fit wherever a mount is allowed to go.

    A FINGER that reaches the arm is refused and the operator tries elsewhere,
    so the grammar can afford mounts whose fingers sometimes do not fit. The
    PLATE has no such retry: it follows from the mount, so a mount whose own rim
    lands in the arm would make every hand using that site illegal.

    Three constants decide it and they agree only by arithmetic --
    WRIST_STANDOFF slides the centre forward, MAX_MOUNT_RADIUS says how far back
    the ring reaches, PALM_RIM adds to it -- and they come to exactly the 50 mm
    the arm sits behind the frame. So the outer ring is TANGENT to the arm at
    the wrist bearing, and this sweep is what says so: measured on the outline
    that gets built, not on the algebra.
    """
    face = design_space.ARM_FACE_Z
    q, a = design_space.PALM_QUANTUM, design_space.ANGLE_QUANTUM
    n_r = int(round((design_space.MAX_MOUNT_RADIUS
                     - design_space.PALM_MIN_RADIUS) / q))
    worst, where = float("inf"), None
    for i in range(n_r + 1):
        r = design_space.PALM_MIN_RADIUS + i * q
        for k in range(int(round(2.0 * np.pi / a))):
            bearing = k * a
            hand = design_space.Hand(
                design_space.Palm(design_space.PALM_THICKNESS),
                (design_space.Finger(design_space.Mount(r, bearing, bearing),
                                     (design_space.Segment(
                                         design_space.Joint(design_space.FLEXION),
                                         design_space.MIN_LINK_LENGTH),)),))
            rear = float(design_space.palm_outline(hand)[:, 1].min())
            if rear < worst:
                worst, where = rear, (r, bearing)
    assert worst >= face - 1e-9, (
        f"a mount at {where[0]*1000:.0f} mm, {np.degrees(where[1]):.0f} deg puts "
        f"the palm at z = {worst*1000:.1f} mm, behind the arm's face at "
        f"{face*1000:.0f} mm")
    # Tangent, not merely clear: the three constants add up exactly, and this is
    # the assertion that fails if one of them moves on its own.
    assert worst == pytest.approx(face, abs=2e-4), (
        f"the outer ring clears the arm by {(worst - face)*1000:.1f} mm -- no "
        f"longer tangent, so MAX_MOUNT_RADIUS, WRIST_STANDOFF and PALM_RIM have "
        f"stopped adding up and nothing else says so")
