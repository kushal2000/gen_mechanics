"""Acceptance tests for E7 (operator-mixture schedules; see
project-notes/grammar/experiments/E7_schedule/summary.md for the real run).

Two sections: (1) the two schedule weight functions sum to 1 at every
generation and match their documented endpoints; (2) a tiny run into a
tmp dir writes the required keys deterministically (same pattern as
test_i15.py's E5b tiny-run test).
"""

from __future__ import annotations

import json

import pytest

from hand_sampler.grammar.experiments import runner  # noqa: F401 -- import first (e2_drift<->runner cycle guard).
from hand_sampler.grammar.experiments import e7_schedule


# ---------------------------------------------------------------------------
# Schedule weight functions: sum to 1 at every generation; match endpoints.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("generations", [3, 5, 40])
def test_schedule_linear_sums_to_one_every_generation(generations):
    for g0 in range(generations):
        mix = e7_schedule.schedule_linear_mixture(g0, generations)
        assert abs(sum(mix.values()) - 1.0) < 1e-9, (generations, g0, sum(mix.values()))


@pytest.mark.parametrize("generations", [3, 5, 40])
def test_schedule_step_sums_to_one_every_generation(generations):
    for g0 in range(generations):
        mix = e7_schedule.schedule_step_mixture(g0, generations)
        assert abs(sum(mix.values()) - 1.0) < 1e-9, (generations, g0, sum(mix.values()))


def test_schedule_linear_matches_documented_endpoints():
    generations = 40
    s0 = e7_schedule._linear_structural_share(0, generations)
    s_last = e7_schedule._linear_structural_share(generations - 1, generations)
    assert abs(s0 - 0.8) < 1e-12
    assert abs(s_last - 0.2) < 1e-12

    mix0 = e7_schedule.schedule_linear_mixture(0, generations)
    mix_last = e7_schedule.schedule_linear_mixture(generations - 1, generations)
    coarse0 = sum(mix0[op] for op in e7_schedule._COARSE_GROUP)
    minimal0 = sum(mix0[op] for op in e7_schedule._MINIMAL_NOPALM_GROUP)
    resample0 = sum(mix0[op] for op in e7_schedule._RESAMPLE_GROUP)
    small0 = sum(mix0[op] for op in e7_schedule._SMALL_GROUP)
    assert abs((coarse0 + minimal0) - 0.8) < 1e-9
    assert abs(resample0 - 0.10) < 1e-9
    assert abs(small0 - 0.10) < 1e-9

    coarse_last = sum(mix_last[op] for op in e7_schedule._COARSE_GROUP)
    minimal_last = sum(mix_last[op] for op in e7_schedule._MINIMAL_NOPALM_GROUP)
    resample_last = sum(mix_last[op] for op in e7_schedule._RESAMPLE_GROUP)
    small_last = sum(mix_last[op] for op in e7_schedule._SMALL_GROUP)
    assert abs((coarse_last + minimal_last) - 0.2) < 1e-9
    assert abs(resample_last - 0.10) < 1e-9
    assert abs(small_last - 0.70) < 1e-9


def test_schedule_step_endpoints_match_static_pools_exactly():
    generations = 40
    mix0 = e7_schedule.schedule_step_mixture(0, generations)
    mix_last = e7_schedule.schedule_step_mixture(generations - 1, generations)
    assert mix0 == e7_schedule.POOLS_STATIC["DEFAULT_uniform"]
    assert mix_last == e7_schedule.POOLS_STATIC["UNION_nopalm_weighted"]


def test_schedule_step_endpoints_match_static_pools_small_generations():
    # generations=3: boundary = clamp(round(3*15/40), 1, generations-1) = 1,
    # so g0=0 is still DEFAULT_uniform and g0=generations-1=2 is still
    # UNION_nopalm_weighted -- the endpoint property must hold at any
    # generations>1, not only the real run's 40.
    generations = 3
    mix0 = e7_schedule.schedule_step_mixture(0, generations)
    mix_last = e7_schedule.schedule_step_mixture(generations - 1, generations)
    assert mix0 == e7_schedule.POOLS_STATIC["DEFAULT_uniform"]
    assert mix_last == e7_schedule.POOLS_STATIC["UNION_nopalm_weighted"]


def test_pool_operator_universe_default_uniform_is_subset_of_schedule_universe():
    default_ops = set(e7_schedule.pool_operator_universe("DEFAULT_uniform"))
    sched_ops = set(e7_schedule.pool_operator_universe("SCHEDULE_linear"))
    assert default_ops <= sched_ops
    assert set(e7_schedule.pool_operator_universe("SCHEDULE_step")) == sched_ops
    assert set(e7_schedule.pool_operator_universe("UNION_nopalm_weighted")) == sched_ops


# ---------------------------------------------------------------------------
# Tiny run: required keys, determinism.
# ---------------------------------------------------------------------------


def test_e7_tiny_run_result_json_keys_and_determinism(tmp_path):
    conditions = e7_schedule.all_conditions()[:2]
    kwargs = dict(
        restarts=[0, 1], conditions=conditions, mu=4, lam=4, generations=3, n_proxy_configs=4,
        processes=1, allow_dirty=True,
    )
    r1 = e7_schedule.run(out_dir=str(tmp_path / "run1"), **kwargs)
    r2 = e7_schedule.run(out_dir=str(tmp_path / "run2"), **kwargs)
    assert r1["per_seed"] == r2["per_seed"]
    assert r1["aggregate"] == r2["aggregate"]

    assert r1["aggregate"]["n_conditions"] == len(conditions)
    for key, c in r1["aggregate"]["per_condition"].items():
        for req in (
            "n_restarts", "final_best_raw_proxy", "final_mean_motors", "final_mean_joints",
            "final_mean_digits", "target_reach_rate", "gen_first_reach_censored",
            "late_locality_mean_tip_displacement_m", "late_locality_n_pairs_total", "clone_rate",
            "rejection_rate_per_operator", "improvement_rate_per_group_by_third",
        ):
            assert req in c, f"{key} missing {req!r}"
        for group in ("structural", "small"):
            assert group in c["improvement_rate_per_group_by_third"]
            assert len(c["improvement_rate_per_group_by_third"][group]) == 3

    for row in r1["per_seed"]:
        assert row["ok"], row["error"]
        res = row["result"]
        for req in (
            "trajectory", "op_stats", "op_improve_third", "n_clone_events", "n_mutate_calls",
            "final_best_raw_proxy", "reached_target", "sigma", "late_locality_n_pairs",
            "late_locality_mean_tip_displacement_m",
        ):
            assert req in res, f"missing {req!r} in per-restart result"
        assert len(res["trajectory"]) == kwargs["generations"] + 1

    assert (tmp_path / "run1" / "result.json").exists()
    assert (tmp_path / "run1" / "summary.md").exists()
    json.loads((tmp_path / "run1" / "result.json").read_text())


def test_e7_paired_diff_keys_present_for_schedule_minus_pool(tmp_path):
    # Include one schedule and both static pools under the same (dist,
    # fitness, cost) so the paired-difference block is non-empty.
    dist_name, _pool, fitness_name, cost_name = e7_schedule.all_conditions()[0]
    conditions = [
        (dist_name, "DEFAULT_uniform", fitness_name, cost_name),
        (dist_name, "UNION_nopalm_weighted", fitness_name, cost_name),
        (dist_name, "SCHEDULE_linear", fitness_name, cost_name),
    ]
    result = e7_schedule.run(
        out_dir=str(tmp_path / "run"), restarts=[0, 1], conditions=conditions,
        mu=4, lam=4, generations=3, n_proxy_configs=4, processes=1, allow_dirty=True,
    )
    diffs = result["aggregate"]["diff_schedule_minus_pool"]
    key = f"SCHEDULE_linear_minus_DEFAULT_uniform__{dist_name}__{fitness_name}__{cost_name}"
    assert key in diffs
    for metric in ("final_best_raw_proxy", "gen_first_reach_censored", "target_reach_rate"):
        assert metric in diffs[key]
        assert diffs[key][metric]["n"] == 2
