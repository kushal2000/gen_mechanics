"""CPU tests for ``grasp_cache.py`` (the anyrotate stable-grasp cache after
HORA, CoRL 2022, and AnyRotate App. C): canonical pose mapping, keys,
storage round trip, candidate sampling, acceptance tests and the
reset-time table. Kit-free."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaacsimenvs.inhand_reorient import grasp_cache as gc


def _set(n, joint_names=("a", "b", "c"), source="s", offset=0.0):
    j = len(joint_names)
    q = np.arange(n * j, dtype=np.float32).reshape(n, j) + offset
    return gc.GraspSet(q, q + 0.5, np.tile([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0], (n, 1)), np.ones((n, 4)),
                       joint_names, source, {"candidates": 10 * n})


# --------------------------------------------------------------------------
# HORA's canonical pose on the drake allegro_right joints
# --------------------------------------------------------------------------


def test_hora_canonical_pose_maps_hora_order_onto_drake_names():
    pose = gc.HORA_CANONICAL_ALLEGRO
    assert sorted(pose) == sorted(f"joint_{i}" for i in range(16))
    # HORA order: index 0-3, thumb 4-7, middle 8-11, ring 12-15.
    assert [pose[f"joint_{i}"] for i in range(4)] == [0.082, 1.244, 0.265, 0.298]  # index
    assert [pose[f"joint_{i}"] for i in range(4, 8)] == [0.005, 1.096, 0.080, 0.150]  # middle
    assert [pose[f"joint_{i}"] for i in range(8, 12)] == [0.029, 1.337, 0.285, 0.317]  # ring
    assert [pose[f"joint_{i}"] for i in range(12, 16)] == [1.104, 1.163, 0.953, -0.138]  # thumb
    # The thumb's second joint sits on the drake joint_13 upper limit (1.163):
    # evidence that HORA lists the thumb second.
    assert pose["joint_13"] == 1.163
    assert gc.CANONICAL_GRASP_POSES["allegro_right"] is pose


def test_hora_canonical_pose_is_inside_the_drake_allegro_limits():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    urdf = (Path(__file__).resolve().parents[3] / "hand_sampler/grammar_bench/fixtures/real/allegro_right/"
            "allegro_hand_description_right.urdf")
    limits = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
              for j in ET.parse(urdf).getroot().findall("joint") if j.find("limit") is not None}
    for name, v in gc.HORA_CANONICAL_ALLEGRO.items():
        lo, hi = limits[name]
        assert lo <= v <= hi + 1e-9, (name, v, lo, hi)


# --------------------------------------------------------------------------
# Keys, sidecar, signature
# --------------------------------------------------------------------------


def test_keys_and_calibration_sha():
    assert gc.design_key("abc") == "design:abc"
    with pytest.raises(ValueError):
        gc.design_key("")
    sha = gc.hand_calibration_sha("allegro_right", {"base_rot": [1, 0, 0, 0]}, {"base_pos": [0, 0, 0.5]}, True)
    assert len(sha) == 64
    assert sha == gc.hand_calibration_sha("allegro_right", {"base_rot": [1, 0, 0, 0]}, {"base_pos": [0, 0, 0.5]},
                                          True)
    assert sha != gc.hand_calibration_sha("allegro_right", {"base_rot": [1, 0, 0, 0]}, {"base_pos": [0, 0, 0.5]},
                                          False)
    assert sha != gc.hand_calibration_sha("allegro_right", {"base_rot": [0, 1, 0, 0]}, {"base_pos": [0, 0, 0.5]},
                                          True)
    assert gc.hand_key("allegro_right", sha) == f"hand:allegro_right:{sha[:16]}"


def test_sidecar_sits_next_to_the_population_file():
    assert str(gc.sidecar_path("out/gen_3/population.json")) == "out/gen_3/population.grasps.npz"


def test_object_signature_and_mismatches():
    from types import SimpleNamespace

    a = SimpleNamespace(object_shape="box", box_size=0.0525, capsule_radius=0.0295, capsule_width=0.006,
                        static_friction=10.0, dynamic_friction=10.0, sim_dt=1 / 60, decimation=3,
                        hand_disable_gravity=False, hand_stiffness=3.0, hand_damping=0.1, hand_effort_limit=0.5,
                        object_contact_offset=0.002, object_rest_offset=0.0)
    sig = gc.object_signature(a)
    assert sig["object_shape"] == "box" and sig["box_size"] == 0.0525
    assert gc.signature_mismatches(sig, dict(sig)) == []
    other = dict(sig, box_size=0.06)
    assert gc.signature_mismatches(sig, other) == ["box_size: cache 0.0525 vs env 0.06"]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_save_load_round_trip_keeps_arrays_names_stats_and_empty_sets(tmp_path):
    sets = {"design:x": _set(3, source="x"), "design:y": gc.GraspSet.empty(("a", "b", "c"), "y", {"candidates": 7})}
    path = gc.save_cache(tmp_path / "c.grasps.npz", sets, {"object_signature": {"box_size": 0.05}})
    loaded, doc = gc.load_cache(path)
    assert doc["schema"] == gc.CACHE_SCHEMA and doc["keys"] == ["design:x", "design:y"]
    assert doc["object_signature"] == {"box_size": 0.05}
    assert loaded["design:x"].n == 3 and loaded["design:y"].n == 0
    np.testing.assert_array_equal(loaded["design:x"].q, sets["design:x"].q)
    np.testing.assert_array_equal(loaded["design:x"].q_target, sets["design:x"].q_target)
    np.testing.assert_array_equal(loaded["design:x"].obj_pose, sets["design:x"].obj_pose)
    assert loaded["design:x"].joint_names == ("a", "b", "c") and loaded["design:x"].source == "x"
    assert loaded["design:y"].stats == {"candidates": 7}
    assert not list(tmp_path.glob("*.tmp*"))  # atomic write leaves no temp file


def test_load_rejects_another_schema(tmp_path):
    import json

    path = tmp_path / "bad.npz"
    np.savez(path, meta=np.array(json.dumps({"schema": "other/9", "keys": [], "entries": {}})))
    with pytest.raises(ValueError, match="schema"):
        gc.load_cache(path)


def test_merge_keeps_existing_entries_unless_overwrite():
    base = {"k1": _set(2, offset=0.0)}
    new = {"k1": _set(5, offset=100.0), "k2": _set(1)}
    merged = gc.merge_sets(base, new)
    assert merged["k1"].n == 2 and merged["k2"].n == 1
    assert gc.merge_sets(base, new, overwrite=True)["k1"].n == 5


def test_grasp_set_validates_shapes_and_truncates():
    s = _set(4)
    assert s.truncated(2).n == 2 and s.truncated(10).n == 4
    with pytest.raises(ValueError):
        gc.GraspSet(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros((2, 7)), np.zeros((2, 4)), ("a", "b", "c"))


def test_cache_summary_counts_viable_designs():
    summ = gc.cache_summary({"a": _set(4), "b": gc.GraspSet.empty(("a", "b", "c")), "c": _set(2)})
    assert summ["n_keys"] == 3 and summ["n_viable"] == 2 and summ["grasps_total"] == 6
    assert summ["viable_frac"] == pytest.approx(2 / 3) and summ["grasps_per_viable_median"] == 3.0


# --------------------------------------------------------------------------
# Candidate sampling and acceptance
# --------------------------------------------------------------------------


def test_canonical_candidates_stay_within_noise_and_limits():
    g = torch.Generator().manual_seed(0)
    n, j = 2000, 4
    canonical = torch.tensor([0.5, 0.0, 1.0, 0.2]).expand(n, j)
    lower = torch.tensor([-1.0, -0.1, 0.0, 0.0]).expand(n, j)
    upper = torch.tensor([1.0, 0.1, 1.5, 1e-8]).expand(n, j)
    q = gc.sample_joint_candidates(canonical, lower, upper, None, noise=0.3, curl_frac=0.0, generator=g)
    assert (q >= lower).all() and (q <= upper).all()
    assert ((q - canonical).abs() <= 0.3 + 1e-6)[:, [0, 2]].all()
    assert q[:, 0].std() > 0.1  # spread: U(-0.3, 0.3) has std 0.17


def test_curl_candidates_cover_the_range_and_ghost_joints_keep_canonical():
    g = torch.Generator().manual_seed(1)
    n, j = 4000, 3
    canonical = torch.zeros(n, j)
    lower = torch.tensor([0.0, -1.0, 0.0]).expand(n, j)
    upper = torch.tensor([2.0, 1.0, 1e-8]).expand(n, j)
    valid = torch.tensor([True, True, False]).expand(n, j)
    q = gc.sample_joint_candidates(canonical, lower, upper, valid, noise=0.3, curl_frac=1.0,
                                   curl_range=(0.15, 0.85), curl_noise=0.1, generator=g)
    assert (q[:, 2] == 0.0).all()
    frac = (q[:, 0] - 0.0) / 2.0
    assert frac.min() >= 0.15 - 0.05 - 1e-6 and frac.max() <= 0.85 + 0.05 + 1e-6
    # every joint of a candidate curls by the same fraction (up to the noise)
    frac1 = (q[:, 1] + 1.0) / 2.0
    assert ((frac - frac1).abs() <= 0.1 + 1e-6).all()


def test_mixture_share_follows_curl_frac():
    g = torch.Generator().manual_seed(2)
    n = 10000
    canonical = torch.full((n, 1), 5.0)
    lower, upper = torch.full((n, 1), 0.0), torch.full((n, 1), 10.0)
    q = gc.sample_joint_candidates(canonical, lower, upper, None, noise=0.01, curl_frac=0.3,
                                   curl_range=(0.0, 0.2), curl_noise=0.0, generator=g)
    curled = (q[:, 0] <= 2.0 + 1e-6).float().mean().item()
    assert curled == pytest.approx(0.3, abs=0.02)


def test_random_quats_are_unit_with_nonnegative_w():
    q = gc.random_quats(5000, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(q.norm(dim=-1), torch.ones(5000), atol=1e-5)
    assert (q[:, 0] >= 0).all()
    assert q[:, 1:].mean(dim=0).abs().max() < 0.03  # no preferred axis


def _ok_inputs(n=1):
    return dict(max_disp=torch.full((n,), 0.005), lin_speed=torch.full((n,), 0.01), ang_speed=torch.full((n,), 0.1),
                tip_contacts=torch.full((n,), 3), nontip_contacts=torch.full((n,), 1),
                max_tip_dist=torch.full((n,), 0.06), mean_tip_dist=torch.full((n,), 0.07))


@pytest.mark.parametrize("field,value", [
    ("max_disp", 0.021), ("lin_speed", 0.06), ("ang_speed", 0.6), ("tip_contacts", 1), ("max_tip_dist", 0.11)])
def test_each_default_test_rejects(field, value):
    th = gc.StabilityThresholds()
    assert gc.stable_mask(**_ok_inputs(), th=th).item()
    bad = _ok_inputs()
    bad[field] = torch.tensor([value])
    assert not gc.stable_mask(**bad, th=th).item()


def test_optional_anyrotate_tests_are_off_by_default_and_reject_when_on():
    base = gc.StabilityThresholds()
    assert base.max_nontip_contacts == -1 and base.max_mean_tip_dist_m < 0 and base.min_tip_contacts == 2
    assert gc.stable_mask(**_ok_inputs(), th=base).item()  # 1 non-tip contact, mean 0.07: fine for HORA
    anyrotate = gc.StabilityThresholds(max_nontip_contacts=0, max_mean_tip_dist_m=0.05, min_tip_contacts=3)
    assert not gc.stable_mask(**_ok_inputs(), th=anyrotate).item()
    ok = _ok_inputs()
    ok.update(nontip_contacts=torch.tensor([0]), mean_tip_dist=torch.tensor([0.04]))
    assert gc.stable_mask(**ok, th=anyrotate).item()


def test_nonfinite_candidates_are_rejected():
    assert not gc.stable_mask(**_ok_inputs(), finite=torch.tensor([False])).item()


# --------------------------------------------------------------------------
# Reset-time table
# --------------------------------------------------------------------------


def test_build_table_reorders_joints_by_name_and_skips_empty_designs():
    s0 = _set(2, joint_names=("b", "a"))
    s2 = _set(3, joint_names=("a", "b"), offset=100.0)
    t = gc.build_table([s0, None, s2, gc.GraspSet.empty(("a", "b"))], ["a", "b"])
    assert t.counts.tolist() == [2, 0, 3, 0] and t.offsets.tolist() == [0, 2, 2, 5]
    assert t.has_grasp.tolist() == [True, False, True, False]
    np.testing.assert_array_equal(t.q[:2].numpy(), s0.q[:, [1, 0]])
    np.testing.assert_array_equal(t.q_target[2:].numpy(), s2.q_target)
    assert t.obj_pose.shape == (5, 7)


def test_build_table_caps_per_design_and_rejects_missing_joints():
    t = gc.build_table([_set(5, joint_names=("a",))], ["a"], max_per_design=2)
    assert t.counts.tolist() == [2]
    with pytest.raises(ValueError, match="lacks joints"):
        gc.build_table([_set(2, joint_names=("a",))], ["a", "z"])


def test_table_sample_draws_each_designs_own_rows_uniformly():
    t = gc.build_table([_set(2, joint_names=("a",)), None, _set(4, joint_names=("a",))], ["a"])
    design_idx = torch.tensor([0, 1, 2] * 4000)
    rows, ok = t.sample(design_idx, generator=torch.Generator().manual_seed(4))
    assert ok.tolist()[:3] == [True, False, True]
    r0, r2 = rows[design_idx == 0], rows[design_idx == 2]
    assert set(r0.tolist()) == {0, 1} and set(r2.tolist()) == {2, 3, 4, 5}
    assert (rows[design_idx == 1] == 0).all()
    counts = torch.bincount(r2 - 2, minlength=4).float() / r2.numel()
    assert torch.allclose(counts, torch.full((4,), 0.25), atol=0.03)


def test_empty_table_samples_nothing():
    t = gc.build_table([None, None], ["a"])
    rows, ok = t.sample(torch.tensor([0, 1]))
    assert not ok.any() and t.q.shape == (0, 1)


def test_prune_keeps_only_the_given_keys():
    sets = {"a": _set(1), "b": _set(2), "c": _set(3)}
    assert list(gc.prune_sets(sets, ["c", "a", "z"])) == ["a", "c"]


def test_table_with_no_grasp_at_all_reports_no_design_viable():
    """Every design non-viable (an empty table): sampling must not index it."""
    t = gc.build_table([gc.GraspSet.empty(("a",)), None], ["a"])
    rows, ok = t.sample(torch.tensor([0, 1, 0]))
    assert not ok.any() and t.total == 0


def test_signature_records_the_population_actuator_override_only_when_on():
    """Caches made before the option existed keep loading (no key when off);
    a cache made with the override does not load without it."""
    from types import SimpleNamespace

    base = dict(object_shape="box", box_size=0.0525, capsule_radius=0.0295, capsule_width=0.006,
                static_friction=1.0, dynamic_friction=1.0, sim_dt=1 / 120, decimation=6, hand_disable_gravity=False,
                hand_stiffness=3.0, hand_damping=0.1, hand_effort_limit=0.5, object_contact_offset=0.002,
                object_rest_offset=0.0)
    off = gc.object_signature(SimpleNamespace(**base, population_hand_actuator=False))
    on = gc.object_signature(SimpleNamespace(**base, population_hand_actuator=True))
    assert "population_hand_actuator" not in off and on["population_hand_actuator"] is True
    assert gc.signature_mismatches(off, on)


def test_optional_joint_speed_bound_rejects_a_vibrating_hand():
    """A stable grasp should also leave the hand at rest: the peak real-joint
    speed over the hold, bounded by max_joint_speed (off by default)."""
    base = gc.StabilityThresholds()
    assert base.max_joint_speed < 0
    ok = _ok_inputs()
    assert gc.stable_mask(**ok, joint_speed=torch.tensor([50.0]), th=base).item()
    th = gc.StabilityThresholds(max_joint_speed=5.0)
    assert not gc.stable_mask(**ok, joint_speed=torch.tensor([50.0]), th=th).item()
    assert gc.stable_mask(**ok, joint_speed=torch.tensor([1.0]), th=th).item()
    assert gc.stable_mask(**ok, th=th).item()  # not measured: no test
