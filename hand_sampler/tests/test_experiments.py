"""The experiment driver's one load-bearing property: staging is invisible."""

from __future__ import annotations

import json
import subprocess
import sys


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


def test_selection_moves_joint_count_the_way_it_is_pointed(tmp_path):
    """The arms are a yardstick for how fast selection CAN move a statistic.

    What is asserted is the SEPARATION between them, not that each arm moves in
    its own direction. The min arm has nowhere to go: generation 0 already sits
    at about 2.2 joints and the smallest legal hand is 2 -- MIN_FINGERS fingers
    of one joint each -- so across seeds it reads 2.12 to 2.41 and which side of
    its start it lands on is noise. The max arm runs to 9.8-14.8 in the same 25
    generations, and that gap is the real measurement.
    """
    ends = {}
    for mode in ("min_joints", "max_joints"):
        out = tmp_path / mode
        assert run("--out", str(out), "--gens", "25", "--mode", mode, *SCALE).returncode == 0
        rows = [json.loads(l) for l in (out / "stats.jsonl").read_text().splitlines()]
        ends[mode] = (rows[0]["n_joints"], rows[-1]["n_joints"])

    (_, lo_last), (hi_first, hi_last) = ends["min_joints"], ends["max_joints"]
    assert hi_last > hi_first + 3.0, f"max arm barely moved: {ends['max_joints']}"
    assert lo_last < hi_last / 2.0, f"the arms did not separate: {ends}"
    assert lo_last < 3.0, f"min arm left the joint floor: {ends['min_joints']}"
