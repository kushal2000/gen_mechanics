"""Fast smoke tests for E1 (locality) and E2 (neutral drift): run each
experiment's core per-seed/driver function on a handful of seeds at tiny
sizes into a tmp dir, and check ``result.json``'s required aggregate keys
are present and finite, and that two runs are byte-identical (determinism).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from hand_sampler.grammar.experiments import e1_locality, e2_drift
from hand_sampler.grammar.experiments.runner import registered_experiments

SEEDS = [0, 1, 2, 3]


def _all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_e1_e2_registered():
    fns = registered_experiments()
    assert "e1_locality" in fns
    assert "e2_drift" in fns
    assert fns["e1_locality"] is e1_locality.e1_locality_seed
    assert fns["e2_drift"] is e2_drift.e2_drift_seed


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------


def test_e1_locality_result_json_keys_and_finite():
    with tempfile.TemporaryDirectory() as td:
        out_dir = str(Path(td) / "e1_out")
        result = e1_locality.run(out_dir=out_dir, seeds=SEEDS, n_configs=4, processes=2)
        result_path = Path(out_dir) / "result.json"
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        agg = loaded["aggregate"]
        assert agg["n_seeds"] == len(SEEDS)
        for op in e1_locality.ALL_OPERATORS:
            assert op in agg, f"missing operator {op!r} in aggregate"
            op_agg = agg[op]
            for key in ("applicability_rate", "null_rate"):
                assert key in op_agg
            for key in e1_locality._CONTINUOUS_KEYS:
                assert key in op_agg
                assert set(op_agg[key].keys()) == {"median", "p90", "mean", "n"}
        assert _all_finite(agg)
        assert (Path(out_dir) / "summary.md").exists()


def test_e1_locality_deterministic():
    with tempfile.TemporaryDirectory() as td:
        out1 = str(Path(td) / "run1")
        out2 = str(Path(td) / "run2")
        r1 = e1_locality.run(out_dir=out1, seeds=SEEDS, n_configs=4, processes=2)
        r2 = e1_locality.run(out_dir=out2, seeds=SEEDS, n_configs=4, processes=2)
        assert r1["per_seed"] == r2["per_seed"]
        assert r1["aggregate"] == r2["aggregate"]


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------


def test_e2_drift_result_json_keys_and_finite():
    with tempfile.TemporaryDirectory() as td:
        out_dir = str(Path(td) / "e2_out")
        result = e2_drift.run(
            out_dir=out_dir, seeds=SEEDS, walk_length=8, record_through=4,
            record_every_after=2, processes=2,
        )
        result_path = Path(out_dir) / "result.json"
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        agg = loaded["aggregate"]
        assert agg["n_seeds"] == len(SEEDS)
        for mixture_name in e2_drift.MIXTURES:
            assert mixture_name in agg
            m = agg[mixture_name]
            for metric in ("joints", "digits", "motors"):
                for label in ("4", "8"):
                    key = f"delta_{metric}_at_{label}"
                    assert key in m, f"missing {key!r} in mixture {mixture_name!r}"
                    b = m[key]
                    assert set(b.keys()) == {"mean", "ci_lo", "ci_hi", "n"}
                    assert b["n"] == len(SEEDS)
            assert f"distinct_hash_fraction_at_4" in m
            assert f"distinct_hash_fraction_at_8" in m
            assert "acceptance_rate" in m
        assert _all_finite(agg)
        assert (Path(out_dir) / "summary.md").exists()


def test_e2_drift_deterministic():
    with tempfile.TemporaryDirectory() as td:
        out1 = str(Path(td) / "run1")
        out2 = str(Path(td) / "run2")
        r1 = e2_drift.run(out_dir=out1, seeds=SEEDS, walk_length=8, record_through=4,
                           record_every_after=2, processes=2)
        r2 = e2_drift.run(out_dir=out2, seeds=SEEDS, walk_length=8, record_through=4,
                           record_every_after=2, processes=2)
        assert r1["per_seed"] == r2["per_seed"]
        assert r1["aggregate"] == r2["aggregate"]


def test_e2_start_reduction_respects_bounds():
    for seed in range(8):
        derivation, digits, max_phalanx, reduced_fully = e2_drift.build_start(seed)
        if reduced_fully:
            assert digits <= 2
            assert max_phalanx <= 2
