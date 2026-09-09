"""The experiment driver's one load-bearing property: staging is invisible."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

SCALE = ["--parents", "8", "--children", "4", "--seed", "5", "--every", "999"]


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "hand_sampler.experiments.run", *args],
                          capture_output=True, text=True)


def test_a_staged_run_is_the_same_chain_as_one_long_run(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    assert run("--out", str(one), "--gens", "12", *SCALE).returncode == 0
    assert run("--out", str(two), "--gens", "5", *SCALE).returncode == 0
    assert run("--out", str(two), "--gens", "7", *SCALE).returncode == 0

    a = (one / "stats.jsonl").read_text().splitlines()
    b = (two / "stats.jsonl").read_text().splitlines()
    assert len(a) == len(b) == 12
    assert a == b, "resuming produced a different chain -- the RNG state is not restored"


def test_resuming_with_a_different_shape_is_refused(tmp_path):
    """Otherwise two different experiments end up spliced into one statistics file, with..."""
    out = tmp_path / "r"
    assert run("--out", str(out), "--gens", "3", *SCALE).returncode == 0
    bad = run("--out", str(out), "--gens", "3", "--parents", "16", "--children", "4",
              "--seed", "5", "--every", "999")
    assert bad.returncode != 0
    assert "was created with" in bad.stderr


@pytest.mark.parametrize("mode", ("min_joints", "max_joints"))
def test_selection_moves_joint_count_the_way_it_is_pointed(tmp_path, mode):
    """The arms are a yardstick for how fast selection CAN move a statistic, so a run whose..."""
    out = tmp_path / mode
    assert run("--out", str(out), "--gens", "25", "--mode", mode, *SCALE).returncode == 0
    rows = [json.loads(l) for l in (out / "stats.jsonl").read_text().splitlines()]
    first, last = rows[0]["n_joints"], rows[-1]["n_joints"]
    if mode == "max_joints":
        assert last > first
    else:
        assert last <= first
