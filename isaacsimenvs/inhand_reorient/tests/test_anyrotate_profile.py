"""anyrotate profile terms against hand-computed cases (AnyRotate, CoRL 2024,
App. B and F). Kit-free."""

from __future__ import annotations

import math

import pytest
import torch

from isaacsimenvs.inhand_reorient import anyrotate_profile as ar
from isaacsimenvs.inhand_reorient.repose_profile import quat_from_angle_axis

IDENT = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def _rot(angle, axis):
    return quat_from_angle_axis(torch.tensor([float(angle)]), torch.tensor([axis], dtype=torch.float32))


def test_observation_widths_match_the_papers_95_plus_privileged():
    obs = sum(ar.anyrotate_field_width(f, 16, 4) for f in ar.ANYROTATE_OBS_FIELDS)
    priv = sum(ar.anyrotate_field_width(f, 16, 4) for f in ar.ANYROTATE_PRIV_FIELDS)
    assert obs == 95  # App. G: "the full touch N = 95"
    assert priv == 3 + 4 + 3 + 2 + 3 + 1 + 3 + 3 + 4  # Table 3


def test_six_keypoints_five_cm_along_the_principal_axes():
    k = ar.keypoint_offsets(0.05)
    assert k.shape == (6, 3)
    torch.testing.assert_close(k.norm(dim=-1), torch.full((6,), 0.05))
    assert (k.abs().max(dim=-1).values == 0.05).all()


def test_keypoint_distance_hand_cases():
    off = ar.keypoint_offsets(0.05)
    p = torch.zeros(1, 3)
    assert float(ar.keypoint_distance(p, IDENT, p, IDENT, off)) == pytest.approx(0.0, abs=1e-7)
    shift = torch.tensor([[0.0, 0.03, 0.04]])
    assert float(ar.keypoint_distance(p, IDENT, shift, IDENT, off)) == pytest.approx(0.05, abs=1e-6)
    # 90 deg about z: the four x/y keypoints move 0.05*sqrt(2), the z ones stay.
    q = _rot(math.pi / 2, [0.0, 0.0, 1.0])
    expected = 4 * 0.05 * math.sqrt(2) / 6
    assert float(ar.keypoint_distance(p, q, p, IDENT, off)) == pytest.approx(expected, abs=1e-6)


def test_keypoint_kernel_reward():
    assert float(ar.keypoint_reward(torch.tensor(0.0), 50.0, 2.0)) == pytest.approx(0.25)
    assert float(ar.keypoint_reward(torch.tensor(0.02), 50.0, 2.0)) == pytest.approx(
        1.0 / (math.e + 2.0 + 1.0 / math.e), rel=1e-6)


def test_rotation_about_axis_and_its_clip():
    q1 = _rot(0.02, [0.0, 0.0, 1.0])
    z, x = torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    assert float(ar.rotation_about_axis(IDENT, q1, z)) == pytest.approx(0.02, abs=1e-6)
    assert float(ar.rotation_about_axis(IDENT, q1, -z)) == pytest.approx(-0.02, abs=1e-6)
    assert float(ar.rotation_about_axis(IDENT, q1, x)) == pytest.approx(0.0, abs=1e-6)
    # relative to a non-identity start: the increment, not the absolute angle
    q0 = _rot(1.0, [0.0, 0.0, 1.0])
    assert float(ar.rotation_about_axis(q0, _rot(1.03, [0.0, 0.0, 1.0]), z)) == pytest.approx(0.03, abs=1e-5)
    assert float(ar.rotation_reward(torch.tensor(0.1), 0.025)) == pytest.approx(0.025)
    assert float(ar.rotation_reward(torch.tensor(-0.1), 0.025)) == pytest.approx(-0.025)


def test_quat_to_rotvec():
    torch.testing.assert_close(ar.quat_to_rotvec(IDENT), torch.zeros(1, 3))
    torch.testing.assert_close(ar.quat_to_rotvec(_rot(math.pi / 2, [0.0, 1.0, 0.0])),
                               torch.tensor([[0.0, math.pi / 2, 0.0]]), atol=1e-6, rtol=0)
    # a negative-w representation of the same rotation
    torch.testing.assert_close(ar.quat_to_rotvec(-_rot(0.4, [1.0, 0.0, 0.0])),
                               torch.tensor([[0.4, 0.0, 0.0]]), atol=1e-6, rtol=0)


def test_next_goal_rotates_the_current_orientation_by_the_increment():
    g = ar.next_goal(IDENT, torch.tensor([[0.0, 0.0, 1.0]]), math.radians(30))
    torch.testing.assert_close(g, torch.tensor([[math.cos(math.radians(15)), 0.0, 0.0, math.sin(math.radians(15))]]))
    q0 = _rot(0.5, [1.0, 0.0, 0.0])
    g2 = ar.next_goal(q0, torch.tensor([[1.0, 0.0, 0.0]]), 0.3)
    torch.testing.assert_close(g2, _rot(0.8, [1.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_goal_reached_metrics():
    kp, rot = torch.tensor([0.01, 0.2]), torch.tensor([0.1, 0.3])
    assert ar.goal_reached(kp, rot, 0.15, "rotation_rad").tolist() == [True, False]
    assert ar.goal_reached(kp, rot, 0.15, "kp_dist_m").tolist() == [True, False]
    with pytest.raises(ValueError):
        ar.goal_reached(kp, rot, 0.15, "degrees")


def test_contact_rewards():
    tips = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    gc, bc = ar.contact_rewards(tips, torch.tensor([0, 2, 1]))
    assert gc.tolist() == [1.0, 0.0, 1.0] and bc.tolist() == [0.0, 1.0, 1.0]


def test_stability_penalties():
    w = torch.tensor([[0.3, 0.4, 0.0], [0.6, 0.8, 0.0]])  # |w| = 0.5, 1.0
    assert ar.angular_velocity_penalty(w, 0.6).tolist() == pytest.approx([0.0, -0.4])
    q, q0 = torch.tensor([[0.3, 0.4, 9.0]]), torch.zeros(1, 3)
    mask = torch.tensor([[True, True, False]])
    assert float(ar.pose_penalty(q, q0, mask)) == pytest.approx(-0.5)
    tau, dq = torch.tensor([[1.0, 2.0, 50.0]]), torch.tensor([[0.1, -0.3, 1.0]])
    assert float(ar.work_penalty(tau, dq, mask)) == pytest.approx(-0.5)
    assert float(ar.torque_penalty(torch.tensor([[3.0, 4.0, 7.0]]), mask)) == pytest.approx(-5.0)


def test_axis_deviation_and_terminations():
    z = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    w = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.01, 0.0, 0.0]])
    dev, valid = ar.axis_deviation(w, z, 0.5)
    assert dev.tolist() == pytest.approx([math.pi / 2, 0.0, math.pi / 2], abs=1e-6)
    assert valid.tolist() == [True, True, False]
    dropped, off = ar.terminations(torch.tensor([0.2, 0.05, 0.05]), 0.1, dev, valid, math.radians(45))
    assert dropped.tolist() == [True, False, False] and off.tolist() == [True, False, False]


def test_reward_curriculum_lambda():
    assert ar.reward_curriculum_lambda(0.5, 1.0, 2.0) == 0.0
    assert ar.reward_curriculum_lambda(1.5, 1.0, 2.0) == pytest.approx(0.5)
    assert ar.reward_curriculum_lambda(3.0, 1.0, 2.0) == 1.0


def test_combine_rewards_hand_computed():
    weights = dict(kp=1.0, rot=5.0, goal=10.0, gc=0.1, bc=0.2, omega=0.5, pose=0.5, work=0.1, torque=0.05,
                   penalty=50.0)
    t = {k: torch.tensor([v]) for k, v in dict(kp=0.2, rot=0.01, goal=1.0, gc=1.0, bc=1.0, omega=-0.4, pose=-1.0,
                                               work=-2.0, torque=-3.0, penalty=0.0).items()}
    rotation = 0.2 + 0.05 + 10.0
    shaped = (0.1 - 0.2) + (-0.2 - 0.5 - 0.2 - 0.15)
    assert float(ar.combine_rewards(t, weights, 0.0)) == pytest.approx(rotation)
    assert float(ar.combine_rewards(t, weights, 0.5)) == pytest.approx(rotation + 0.5 * shaped)
    t["penalty"] = torch.tensor([1.0])
    assert float(ar.combine_rewards(t, weights, 1.0)) == pytest.approx(rotation + shaped - 50.0)


def test_relative_joint_targets():
    lo, hi = torch.tensor([[-1.0, -1.0]]), torch.tensor([[1.0, 0.01]])
    target, a = ar.relative_joint_targets(torch.zeros(1, 2), torch.tensor([[1.0, 2.0]]), torch.zeros(1, 2),
                                          0.5, 0.026, lo, hi)
    torch.testing.assert_close(a, torch.tensor([[0.026, 0.026]]))  # clipped to [-1, 1] then scaled
    torch.testing.assert_close(target, torch.tensor([[0.013, 0.01]]))  # eta 0.5; upper limit holds
    target2, _ = ar.relative_joint_targets(target, torch.tensor([[-1.0, 0.0]]), torch.tensor([[1.0, 1.0]]),
                                           0.5, 0.026, lo, hi)
    torch.testing.assert_close(target2, torch.tensor([[0.013, 0.01]]))  # -0.013 + 0.013 = 0; capped at 0.01


def test_simulated_tactile():
    force = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.2], [0.0, 0.0, 20.0]]])
    direction = torch.tensor([[[0.0, math.sin(0.2), math.cos(0.2)], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]])
    c, pose, f, fs = ar.simulated_tactile(force, direction, torch.zeros_like(force), alpha=0.5, threshold=0.25,
                                          beta_f=0.6, f_max=5.0, beta_p=0.6, p_max=0.53)
    assert c.tolist() == [[True, False, True]]
    assert f.tolist()[0] == pytest.approx([0.3, 0.0, 3.0])  # 0.6 * min(0.5|0.1|10, 5), masked by c
    assert pose[0, 0].tolist() == pytest.approx([0.6 * 0.2, 0.0], abs=1e-6)
    assert pose[0, 2].tolist() == pytest.approx([0.0, 0.6 * 0.53], abs=1e-6)  # saturated at P_max
    assert (pose[0, 1] == 0).all()


def test_sample_axes():
    g = torch.Generator().manual_seed(0)
    s = ar.sample_axes(1000, "sphere", generator=g)
    torch.testing.assert_close(s.norm(dim=-1), torch.ones(1000))
    assert s.mean(dim=0).abs().max() < 0.1
    p = ar.sample_axes(64, "principal", generator=g)
    assert ((p.abs() == 1.0).sum(dim=-1) == 1).all()
    assert (ar.sample_axes(3, "z") == torch.tensor([0.0, 0.0, 1.0])).all()


def test_graded_rotation_fitness():
    f = ar.graded_rotation_fitness(torch.tensor([2 * math.pi, -1.0, 0.0]), torch.tensor([30.0, 15.0, 0.0]), 30.0)
    assert f.tolist() == pytest.approx([1.25, 0.125, 0.0])


def test_axis_tilt_is_zero_for_rotation_about_the_axis_and_grows_with_tumbling():
    k = torch.tensor([[0.0, 0.0, 1.0]])
    q0 = _rot(0.7, [1.0, 1.0, 0.0])
    axis_obj = ar.axis_in_object_frame(q0, k)
    # any further rotation about k keeps the object's k-aligned axis on k
    q1 = ar.next_goal(q0, k, 1.3)
    assert float(ar.axis_tilt(q1, axis_obj, k)) == pytest.approx(0.0, abs=1e-5)
    # a 60 degree tumble about x tilts it by 60 degrees
    q2 = ar.next_goal(q0, torch.tensor([[1.0, 0.0, 0.0]]), math.radians(60))
    assert float(ar.axis_tilt(q2, axis_obj, k)) == pytest.approx(math.radians(60), abs=1e-5)


def test_world_up_in_the_palm_frame_is_the_palm_normal_of_a_palm_up_hand():
    """The paper's z axis is the palm normal. For allegro_right under
    NVIDIA's pose (repose_hand_poses.json) the palm body's +z runs along the
    fingers (horizontal) and its +x points up, so "z" must be world up
    expressed in the palm frame, not the palm frame's own +z."""
    ident = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(ar.world_up_in_palm(ident), torch.tensor([[0.0, 0.0, 1.0]]))
    nvidia_allegro = torch.tensor([[0.478318, 0.522387, -0.524732, 0.472208]])
    up = ar.world_up_in_palm(nvidia_allegro)
    assert torch.allclose(up, torch.tensor([[0.995, 0.004, -0.096]]), atol=2e-3)
    assert up.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_distal_slots_are_the_last_real_slot_of_each_finger():
    """Population fingertips: the fingertip body f{f}_tip has no collider, so
    a fingertip contact is a contact on the finger's last real link. Finger
    slot f is slots 6 f (its carrier, never counted) to 6 f + 5."""
    valid = torch.zeros(2, 36, dtype=torch.bool)
    valid[0, 1:4] = True      # finger 0: joints 0-2 real -> distal 2
    valid[0, 7:12] = True     # finger 1: all 5 real -> distal 4
    valid[0, 12] = True       # a carrier slot (finger 2's) does not count
    valid[1, 25:26] = True    # finger 4: one joint -> distal 0
    d = ar.distal_slots(valid)
    assert d.tolist() == [[2, 4, -1, -1, -1, -1], [-1, -1, -1, -1, 0, -1]]


def test_tip_sources_index_the_distal_link_sensor_and_drop_it_from_the_nontip_mask():
    tip_names = [f"f{f}_tip" for f in range(6)]
    nontip_names = ["root"] + [f"f{f}_c" for f in range(6)] + [f"f{f}_link{d}" for f in range(6) for d in range(5)]
    distal = torch.tensor([[2, 4, -1, -1, -1, -1], [-1, -1, -1, -1, 0, -1]])
    src, nontip = ar.tip_sources(distal, tip_names, nontip_names)
    all_names = tip_names + nontip_names
    assert [all_names[i] for i in src[0, :2].tolist()] == ["f0_link2", "f1_link4"]
    assert all_names[src[1, 4].item()] == "f4_link0"
    assert not nontip[0, all_names.index("f0_link2")] and not nontip[0, all_names.index("f1_link4")]
    assert nontip[0, all_names.index("f0_link1")] and nontip[0, all_names.index("root")]
    assert not nontip[1, all_names.index("f4_link0")] and nontip[1, all_names.index("f0_link2")]
    # the fingertip bodies never count as non-tip bodies (no collider)
    assert not nontip[:, :6].any()


def test_timer_goal_advance_runs_ahead_of_a_stalled_object():
    """A/B option (not in the paper): with goal_advance=timer, a goal not
    reached within the interval advances by the increment from the previous
    goal, so a stalled object falls behind and r_kp decays."""
    assert ar.goal_timer_due(torch.tensor([0, 29, 30, 31]), 1.5, 0.05).tolist() == [False, False, True, True]
    axis = torch.tensor([[0.0, 0.0, 1.0]])
    goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    for _ in range(3):
        goal = ar.next_goal(goal, axis, math.radians(30.0))
    obj = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert ar.rotation_about_axis(obj, goal, axis).item() == pytest.approx(math.radians(90.0), abs=1e-5)
    offs = ar.keypoint_offsets(0.05)
    lag = ar.keypoint_distance(torch.zeros(1, 3), obj, torch.zeros(1, 3), goal, offs)
    one = ar.keypoint_distance(torch.zeros(1, 3), obj, torch.zeros(1, 3), ar.next_goal(obj, axis, math.radians(30.0)), offs)
    assert ar.keypoint_reward(lag, 50.0, 2.0) < ar.keypoint_reward(one, 50.0, 2.0)


def test_gravity_promotion_by_goals_or_by_rotations():
    """Dexsuite promotes on success (>= 1 goal); the hora profile has no
    goals, so promotion can count rotations about k per episode instead."""
    goals = torch.tensor([0.0, 1.0, 3.0])
    rotations = torch.tensor([0.5, 0.0, 0.1])
    assert ar.gravity_promoted(goals, rotations, "goals", 1, 0.25).tolist() == [False, True, True]
    assert ar.gravity_promoted(goals, rotations, "rotations", 1, 0.25).tolist() == [True, False, False]
    with pytest.raises(ValueError):
        ar.gravity_promoted(goals, rotations, "other", 1, 0.25)
