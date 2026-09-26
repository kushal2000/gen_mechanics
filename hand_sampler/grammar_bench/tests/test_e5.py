"""E5 (proxy-fitness evolution) acceptance tests: a tiny run into a tmp dir
has the required ``result.json`` keys with finite values, is deterministic
across two runs, and -- under a FIXED evaluation seed (removing the
across-generation fitness noise described in ``e5_evolve.py``'s module
docstring) -- has a non-decreasing best-fitness trajectory, as
(mu+lambda) truncation selection with implicit elitism guarantees whenever
fitness is a deterministic function of phenotype.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from hand_sampler.grammar.experiments.e5_evolve import (
    DIST_VARIANTS,
    POOLS,
    all_conditions,
    e5_evolve_seed,
    run,
)
from hand_sampler.grammar.experiments.runner import registered_experiments

_TINY_CONDITIONS = all_conditions()[:2]

_REQUIRED_TRAJECTORY_KEYS = (
    "generation", "best_fitness", "mean_fitness", "mean_motors", "mean_digits",
    "mean_palm_bodies", "mean_joints", "frac_palm_joint", "frac_branch",
    "frac_coupling", "frac_prismatic", "distinct_hash_fraction",
)
_REQUIRED_RESULT_KEYS = (
    "restart_seed", "dist", "pool", "fitness", "cost", "trajectory", "operator_survival",
    "final_best_fitness", "gen90", "final_mean_motors", "final_mean_joints",
    "construct_usage_gen40", "diversity_gen40",
)


def test_e5_registered():
    assert "e5_evolve" in registered_experiments()


def _run_tiny(out_dir: Path, **kwargs):
    return run(
        out_dir=str(out_dir), restarts=[0], conditions=_TINY_CONDITIONS,
        mu=4, lam=4, generations=3, n_proxy_configs=4, processes=1, allow_dirty=True, **kwargs,
    )


def _assert_finite(x, path: str):
    assert isinstance(x, (int, float)), f"{path} is not numeric: {x!r}"
    assert math.isfinite(float(x)), f"{path} is not finite: {x!r}"


def test_e5_tiny_run_result_json_keys_and_finiteness(tmp_path):
    result = _run_tiny(tmp_path / "run1")
    assert result["aggregate"]["n_failed"] == 0
    assert result["aggregate"]["n_ok"] == len(_TINY_CONDITIONS)

    result_json_path = tmp_path / "run1" / "result.json"
    summary_path = tmp_path / "run1" / "summary.md"
    assert result_json_path.exists()
    assert summary_path.exists()
    on_disk = json.loads(result_json_path.read_text())
    assert on_disk["name"] == "e5_evolve"

    for entry in result["per_seed"]:
        assert entry["ok"], entry["error"]
        r = entry["result"]
        for key in _REQUIRED_RESULT_KEYS:
            assert key in r, f"missing key {key!r} in per-seed result"
        assert r["dist"] in DIST_VARIANTS
        assert r["pool"] in POOLS
        assert len(r["trajectory"]) == 4  # generations 0..3 inclusive
        for row in r["trajectory"]:
            for key in _REQUIRED_TRAJECTORY_KEYS:
                assert key in row, f"missing trajectory key {key!r}"
                _assert_finite(row[key], f"trajectory.{key}")
        _assert_finite(r["final_best_fitness"], "final_best_fitness")
        _assert_finite(r["gen90"], "gen90")
        _assert_finite(r["final_mean_motors"], "final_mean_motors")
        _assert_finite(r["final_mean_joints"], "final_mean_joints")
        _assert_finite(r["diversity_gen40"], "diversity_gen40")
        for key, v in r["construct_usage_gen40"].items():
            _assert_finite(v, f"construct_usage_gen40.{key}")
        for op, d in r["operator_survival"].items():
            _assert_finite(d["offspring"], f"operator_survival.{op}.offspring")
            _assert_finite(d["survived"], f"operator_survival.{op}.survived")
            assert d["survived"] <= d["offspring"]

    # aggregate tables are present and keyed as the summary/report expect
    assert "table1" in result["aggregate"]
    assert "table2" in result["aggregate"]
    assert "table3_union_weighted_operator_survival" in result["aggregate"]


def test_e5_deterministic_across_two_runs(tmp_path):
    result1 = _run_tiny(tmp_path / "runA")
    result2 = _run_tiny(tmp_path / "runB")

    def _strip(res):
        return {
            "per_seed": [
                {"seed": list(e["seed"]), "ok": e["ok"], "result": e["result"]} for e in res["per_seed"]
            ],
            "aggregate": res["aggregate"],
        }

    assert _strip(result1) == _strip(result2)


def test_e5_best_fitness_non_decreasing_under_fixed_eval_seed(tmp_path):
    for task in [(0, dist, pool, "opposition", "none") for dist in DIST_VARIANTS for pool in POOLS]:
        result = e5_evolve_seed(
            task, mu=4, lam=4, generations=6, n_proxy_configs=4, fixed_eval_seed=True,
        )
        best_values = [row["best_fitness"] for row in result["trajectory"]]
        for a, b in zip(best_values, best_values[1:]):
            assert b >= a - 1e-12, (
                f"best_fitness decreased under fixed_eval_seed for task {task}: {best_values}"
            )

