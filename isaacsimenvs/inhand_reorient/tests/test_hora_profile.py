"""The ``hora`` task profile's arithmetic against HORA's released code (H. Qi
et al., CoRL 2022; hora/tasks/allegro_hand_hora.py, configs/task/
AllegroHandHora.yaml). Kit-free."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import yaml
from pathlib import Path

from isaacsimenvs.inhand_reorient import anyrotate_profile as ar
from isaacsimenvs.inhand_reorient import hora_profile as hp
from isaacsimenvs.inhand_reorient.repose_profile import TASK_PROFILES, quat_from_angle_axis

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_hora_is_a_registered_profile_in_the_anyrotate_family():
    assert "hora" in TASK_PROFILES
    assert ar.is_anyrotate(SimpleNamespace(task_profile="hora"))
    assert hp.is_hora(SimpleNamespace(task_profile="hora"))
    assert not hp.is_hora(SimpleNamespace(task_profile="anyrotate"))
    assert not hp.is_hora(SimpleNamespace(task_profile="legacy"))


def test_rotate_reward_is_the_clipped_angular_velocity_about_the_axis():
    """compute_reward: angdiff / (controlFrequencyInv x dt), dotted with the
    axis, clipped to [angvelClipMin, angvelClipMax] = [-0.5, 0.5]."""
    axis = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    angles = torch.tensor([0.01, 0.05, -0.2])
    q1 = quat_from_angle_axis(angles, axis)
    r = hp.rotate_reward(q0, q1, axis, dt=0.05, lo=-0.5, hi=0.5)
    assert r.tolist() == pytest.approx([0.2, 0.5, -0.5], abs=1e-5)


def test_penalties_match_hora():
    p0 = torch.tensor([[0.0, 0.0, 0.0]])
    p1 = torch.tensor([[0.01, -0.02, 0.0]])
    # object_linvel_penalty = L1 norm of the finite-difference velocity
    assert hp.linvel_penalty(p0, p1, dt=0.05).item() == pytest.approx(0.6)
    q = torch.tensor([[0.1, 0.2]])
    q_init = torch.tensor([[0.0, 0.0]])
    assert hp.pose_diff_penalty(q, q_init).item() == pytest.approx(0.05)  # sum of squares
    tau = torch.tensor([[0.5, -0.5]])
    assert hp.torque_penalty(tau).item() == pytest.approx(0.5)  # sum of squares
    dq = torch.tensor([[1.0, 1.0]])
    assert hp.work_penalty(tau, dq).item() == pytest.approx(0.0)  # (sum tau dq)^2
    assert hp.work_penalty(tau, torch.tensor([[2.0, 0.0]])).item() == pytest.approx(1.0)
    mask = torch.tensor([[True, False]])
    assert hp.pose_diff_penalty(q, q_init, mask).item() == pytest.approx(0.01)


def test_reward_combines_with_hora_scales():
    terms = dict(rotate=torch.tensor([0.5]), linvel=torch.tensor([1.0]), pose=torch.tensor([1.0]),
                 torque=torch.tensor([1.0]), work=torch.tensor([1.0]))
    scales = dict(rotate=1.0, linvel=-0.3, pose=-0.3, torque=-0.1, work=-2.0)
    assert hp.combine_reward(terms, scales).item() == pytest.approx(0.5 - 0.3 - 0.3 - 0.1 - 2.0)


def test_unscale_and_history():
    lo, hi = torch.tensor([[0.0, -1.0]]), torch.tensor([[2.0, 1.0]])
    assert hp.unscale(torch.tensor([[1.0, 1.0]]), lo, hi).tolist() == [[0.0, 1.0]]
    # a locked (ghost) joint does not divide by ~0
    assert torch.isfinite(hp.unscale(torch.tensor([[0.0]]), torch.tensor([[0.0]]), torch.tensor([[1e-8]]))).all()
    hist = torch.zeros(1, 3, 2)
    hist = hp.push_history(hist, torch.tensor([[1.0, 1.0]]))
    hist = hp.push_history(hist, torch.tensor([[2.0, 2.0]]))
    assert hist[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    filled = hp.fill_history(torch.zeros(2, 3, 2), torch.tensor([0]), torch.tensor([[5.0, 6.0]]))
    assert filled[0].tolist() == [[5.0, 6.0]] * 3 and filled[1].abs().sum() == 0


def test_observation_widths():
    assert ar.anyrotate_field_width("hora_proprio_hist", 16, 4) == 96  # 3 x (16 joints + 16 targets)
    assert ar.anyrotate_field_width("hora_priv", 16, 4) == 9  # obj pos, scale, mass, friction, CoM


def test_drop_is_a_fall_below_the_start_height():
    """check_termination: object z < reset_height_threshold (0.645 m, objects
    start at 0.65-0.66 m): here a fall of more than drop_dz below the
    episode's start height."""
    z0 = torch.tensor([0.60, 0.60])
    z = torch.tensor([0.59, 0.58])
    assert hp.dropped(z, z0, 0.015).tolist() == [False, True]


def test_task_yaml_carries_horas_numbers():
    h = yaml.safe_load((REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())["hora"]
    assert h["sim_dt"] == pytest.approx(1 / 120) and h["decimation"] == 6  # controlFrequencyInv 6: 20 Hz
    assert math.ceil(h["episode_length_s"] / (h["sim_dt"] * h["decimation"])) == 400  # episodeLength
    assert h["action_scale"] == pytest.approx(1 / 24)
    assert (h["angvel_clip_min"], h["angvel_clip_max"]) == (-0.5, 0.5)
    assert (h["rotate_reward_scale"], h["obj_linvel_penalty_scale"], h["pose_diff_penalty_scale"],
            h["torque_penalty_scale"], h["work_penalty_scale"]) == (1.0, -0.3, -0.3, -0.1, -2.0)
    assert h["joint_noise_scale"] == 0.02
