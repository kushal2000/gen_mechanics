"""The ``hora`` task profile's arithmetic against HORA's released code (H. Qi
et al., CoRL 2022; hora/tasks/allegro_hand_hora.py, configs/task/
AllegroHandHora.yaml). Kit-free."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
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


# --------------------------------------------------------------------------
# Shared controllers across hands (2026-10-02)
# --------------------------------------------------------------------------


def test_per_design_reward_scale_tracks_each_designs_spread():
    """Running per-design second moment of the per-step reward; each env's
    reward divided by its design's RMS (floored), so a design with rewards in
    the hundreds and one in the tenths contribute on the same scale."""
    st = hp.DesignRewardScale(n_designs=2, decay=0.0, floor=0.05)
    d = torch.tensor([0, 0, 1, 1])
    r = torch.tensor([100.0, -100.0, 0.1, -0.1])
    out = st.normalize(r, d)
    assert out.tolist() == pytest.approx([1.0, -1.0, 1.0, -1.0], rel=1e-5)
    # floor: a design with near-zero rewards is not blown up
    st2 = hp.DesignRewardScale(n_designs=1, decay=0.0, floor=0.05)
    assert st2.normalize(torch.tensor([0.001, -0.001]), torch.tensor([0, 0])).abs().max() <= 0.001 / 0.05 + 1e-6
    # decay keeps a running estimate
    st3 = hp.DesignRewardScale(n_designs=1, decay=0.9, floor=1e-3)
    st3.normalize(torch.tensor([1.0]), torch.tensor([0]))
    st3.normalize(torch.tensor([3.0]), torch.tensor([0]))
    assert st3.rms[0].item() == pytest.approx(math.sqrt(0.9 * 1.0 + 0.1 * 9.0), rel=1e-5)


def test_morphology_table_per_slot():
    """Per slot: validity, joint axis and origin in the root (palm) frame at
    q = 0, link length, limits (10 values); ghost slots all zero."""
    from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
    from isaacsimenvs.inhand_reorient.scene import population_file as pf
    from hand_sampler.grammar.derive import derivation_from_dict, derive

    entry, status, _ = pf.projected_entry("allegro_right")
    design = ge.canonicalize(derive(derivation_from_dict(entry.derivation_dict)), source=entry.source)
    tab = hp.morphology_table(design)
    assert tab.shape == (36, hp.MORPH_PER_SLOT) and hp.MORPH_PER_SLOT == 10
    T0 = ge.authored_fk(design, np.zeros(ge.N_SLOTS))
    s = ge.finger_slot(0, 1)  # first finger, second joint
    assert tab[s, 0] == 1.0
    assert np.allclose(tab[s, 1:4], T0[s][:3, :3] @ design.slot_axis[s])
    assert np.allclose(tab[s, 4:7], T0[s][:3, 3])
    assert tab[s, 7] == pytest.approx(design.slot_length[s])
    assert tuple(tab[s, 8:10]) == pytest.approx(tuple(design.slot_limits[s]))
    assert np.all(tab[~design.slot_valid] == 0.0)


def test_morph_observation_width():
    # per joint: 10 static + canonical pose; per fingertip: position + mask; 3 hand scalars
    assert ar.anyrotate_field_width("hora_morph", 36, 6) == 36 * 11 + 6 * 4 + 3
    assert ar.anyrotate_field_width("hora_morph", 16, 4) == 16 * 11 + 4 * 4 + 3


def test_opposition_poses_put_one_finger_in_horas_thumb_pose():
    """One canonical pose per valid finger as the opposing digit: that finger
    at HORA's thumb fractions, the others at its finger fractions."""
    from isaacsimenvs.inhand_reorient import grasp_cache as gc

    valid = np.zeros(36, dtype=bool)
    limits = np.zeros((36, 2))
    for f in (0, 3):
        valid[f * 6 + 1: f * 6 + 5] = True   # finger slot f's joints 0-3 (slot 6 f is its carrier)
        limits[f * 6 + 1: f * 6 + 5] = [0.0, 1.0]
    default = np.zeros(36)
    poses = gc.opposition_poses(valid, limits, default)
    assert sorted(poses) == [0, 3]
    assert np.allclose(poses[3][19:23], gc.HORA_THUMB_PROFILE[:4])
    assert np.allclose(poses[3][1:5], gc.HORA_LIKE_PROFILE[:4])
    assert np.allclose(poses[0][1:5], gc.HORA_THUMB_PROFILE[:4])
    # HORA's allegro thumb as range fractions: (0.74, 1.0, 0.62, 0.01)
    assert gc.HORA_THUMB_PROFILE[:4] == pytest.approx((0.74, 1.0, 0.62, 0.01), abs=0.01)


def test_hora_block_defaults_keep_the_port():
    h = yaml.safe_load((REPO_ROOT / "coevolution/cfg/task/InHandReorient.yaml").read_text())["hora"]
    assert h["morph_obs"] is False and h["per_design_reward_norm"] is False
    assert h["disjoint_slots"] is False


def test_disjoint_slot_map_gives_each_design_its_own_policy_columns():
    valid = np.zeros((3, 8), dtype=bool)
    valid[0, [0, 1, 2, 3]] = True  # the first design keeps its columns
    valid[1, [0, 1]] = True        # overlaps design 0: moved to free columns
    valid[2, [3, 7]] = True        # 3 is taken, 7 is free
    pol = hp.disjoint_slot_map(valid)  # (D, J): the policy column of each phys column
    assert pol.shape == (3, 8)
    for d in range(3):
        assert sorted(pol[d].tolist()) == list(range(8))  # a permutation
    assert pol[0, [0, 1, 2, 3]].tolist() == [0, 1, 2, 3]
    assert pol[2, 7] == 7
    real_cols = [set(pol[d, valid[d]].tolist()) for d in range(3)]
    assert sum(len(c) for c in real_cols) == len(set().union(*real_cols)) == 8  # disjoint


def test_disjoint_slot_map_needs_enough_columns():
    valid = np.ones((2, 4), dtype=bool)
    with pytest.raises(ValueError):
        hp.disjoint_slot_map(valid)


def test_policy_column_remap_round_trip():
    valid = np.zeros((2, 5), dtype=bool)
    valid[0, [0, 1]] = True
    valid[1, [0, 2]] = True
    pol = torch.as_tensor(hp.disjoint_slot_map(valid))
    design_idx = torch.tensor([0, 1, 1])
    pol_of_phys = pol[design_idx]
    phys_of_pol = torch.argsort(pol_of_phys, dim=1)
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)  # phys order
    xp = hp.to_policy_columns(x, phys_of_pol)
    assert torch.equal(hp.to_phys_columns(xp, pol_of_phys), x)
    assert xp[1, int(pol_of_phys[1, 2])] == x[1, 2]
