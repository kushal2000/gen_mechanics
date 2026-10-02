"""CPU tests for viable-only population assembly (``evolution/viable.py``,
the driver's ``--viable-only``): the CPU pre-filter, filling N viable designs
from searched batches, probe handling, reuse of known results, and the
per-role viability rates. The grasp search (a Kit launch) is faked."""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from isaacsimenvs.inhand_reorient.evolution import archive as arch
from isaacsimenvs.inhand_reorient.evolution import driver as drv
from isaacsimenvs.inhand_reorient.evolution import viable as vb


# --------------------------------------------------------------------------
# Pre-filter
# --------------------------------------------------------------------------


def _report(admitted=True, digits=2, reach=1):
    return {"admitted": admitted, "reasons": [] if admitted else ["x"], "digit_count": digits,
            "fingertips_reachable": reach}


def test_prefilter_needs_two_digits_for_two_tip_contacts_and_one_reachable_tip():
    ok, why = vb.prefilter_report(_report())
    assert ok and why == ""
    assert vb.prefilter_report(_report(digits=1)) == (False, "digits<2")
    assert vb.prefilter_report(_report(reach=0)) == (False, "reach<1")
    assert vb.prefilter_report(_report(admitted=False)) == (False, "not_admitted")
    assert vb.prefilter_report({"admitted": False, "reasons": ["s"], "digit_count": None,
                                "fingertips_reachable": None}) == (False, "not_admitted")
    # min_tip_contacts 3 needs three digits
    assert vb.prefilter_report(_report(digits=2), min_tip_contacts=3) == (False, "digits<3")


# --------------------------------------------------------------------------
# Filling N viable designs (fake proposals and search)
# --------------------------------------------------------------------------


def _proposal(i, role="offspring", sha=None):
    meta = drv.DesignMeta(design_id=f"d{i}", founder_id=f"f{i}", parent_id=None, generation_born=1,
                          digit_count=2, joint_count=6, role=role, source=f"arch:d{i}")
    return vb.Proposal(entry=None, meta=meta, sha256=sha or f"sha{i}", prefilter_ok=True, prefilter_reason="")


class _Proposer:
    """Proposals 0, 1, 2, ...; those in ``rejected`` fail the pre-filter."""

    def __init__(self, rejected=(), role_of=None):
        self.i = 0
        self.rejected = set(rejected)
        self.role_of = role_of or (lambda i: "offspring")

    def __call__(self):
        i = self.i
        self.i += 1
        p = _proposal(i, self.role_of(i))
        if i in self.rejected:
            p.prefilter_ok, p.prefilter_reason = False, "digits<2"
        return p


class _Search:
    """Viable iff the proposal index is in ``viable``; records batches."""

    def __init__(self, viable, secs=10.0):
        self.viable = set(viable)
        self.batches = []
        self.secs = secs

    def __call__(self, batch):
        self.batches.append([p.meta.design_id for p in batch])
        return {p.sha256: (5 if int(p.meta.design_id.lstrip("d")) in self.viable else 0) for p in batch}, self.secs


def test_fill_searches_batches_until_the_target_is_filled():
    search = _Search(viable={1, 4, 6, 9, 12})
    res = vb.fill_viable(target=4, forced=[], propose=_Proposer(), search=search, known={}, batch_size=5,
                         max_batches=5)
    assert [p.meta.design_id for p in res.chosen] == ["d1", "d4", "d6", "d9"]
    assert search.batches == [["d0", "d1", "d2", "d3", "d4"], ["d5", "d6", "d7", "d8", "d9"]]
    s = res.stats
    assert s.batches == 2 and s.search_s == 20.0 and s.searched == 10
    assert s.searched_by_role["offspring"] == 10 and s.viable_by_role["offspring"] == 4
    assert {r["design_id"] for r in s.non_viable} == {"d0", "d2", "d3", "d5", "d7", "d8"}
    assert all(r["reason"] == "no_stable_grasp" for r in s.non_viable)
    assert s.unused_viable == []  # d9 filled the last slot


def test_extra_viable_designs_beyond_the_target_are_recorded_unused():
    res = vb.fill_viable(target=1, forced=[], propose=_Proposer(), search=_Search(viable={0, 1, 2}),
                         known={}, batch_size=3, max_batches=3)
    assert [p.meta.design_id for p in res.chosen] == ["d0"]
    assert res.stats.unused_viable == ["d1", "d2"]


def test_prefilter_rejects_are_never_searched_and_count_as_non_viable():
    search = _Search(viable={1, 3})
    res = vb.fill_viable(target=2, forced=[], propose=_Proposer(rejected={0, 2}), search=search, known={},
                         batch_size=2, max_batches=3)
    assert search.batches == [["d1", "d3"]]
    assert [p.meta.design_id for p in res.chosen] == ["d1", "d3"]
    assert res.stats.prefilter_rejected_by_role["offspring"] == 2
    assert {r["design_id"]: r["reason"] for r in res.stats.non_viable} == {"d0": "prefilter:digits<2",
                                                                           "d2": "prefilter:digits<2"}
    # drawn = searched + rejected; the rate counts every drawn design
    assert res.stats.drawn_by_role["offspring"] == 4
    assert res.stats.viability_rate("offspring") == pytest.approx(2 / 4)


def test_known_results_are_reused_without_search():
    search = _Search(viable={2})
    known = {"sha0": 7, "sha1": 0}
    res = vb.fill_viable(target=2, forced=[], propose=_Proposer(), search=search, known=known, batch_size=1,
                         max_batches=2)
    assert [p.meta.design_id for p in res.chosen] == ["d0", "d2"]
    assert search.batches == [["d2"]]
    assert {r["design_id"]: r["reason"] for r in res.stats.non_viable} == {"d1": "known_non_viable"}
    assert known["sha2"] == 5  # the search result is remembered


def test_batch_cap_leaves_the_population_short_and_says_so():
    res = vb.fill_viable(target=5, forced=[], propose=_Proposer(), search=_Search(viable={0}), known={},
                         batch_size=3, max_batches=2)
    assert [p.meta.design_id for p in res.chosen] == ["d0"]
    assert res.stats.batches == 2 and res.stats.short_by == 4


def test_forced_probes_go_in_the_first_batch_and_count_toward_the_target():
    probes = [_proposal(100, "probe", "shaP0"), _proposal(101, "probe", "shaP1")]
    search = _Search(viable={100, 0, 1, 2})
    res = vb.fill_viable(target=3, forced=probes, propose=_Proposer(), search=search, known={}, batch_size=4,
                         max_batches=3)
    assert search.batches[0][:2] == ["d100", "d101"]
    assert [p.meta.design_id for p in res.forced_viable] == ["d100"]
    assert [p.meta.design_id for p in res.forced_non_viable] == ["d101"]
    assert [p.meta.design_id for p in res.chosen] == ["d0", "d1"]  # 3 = 1 probe + 2 new
    assert res.stats.searched_by_role["probe"] == 2 and res.stats.viable_by_role["probe"] == 1


def test_known_probes_need_no_search():
    probes = [_proposal(100, "probe", "shaP0"), _proposal(101, "probe", "shaP1")]
    search = _Search(viable={0})
    res = vb.fill_viable(target=2, forced=probes, propose=_Proposer(), search=search,
                         known={"shaP0": 3, "shaP1": 0}, batch_size=1, max_batches=2)
    assert search.batches == [["d0"]]
    assert [p.meta.design_id for p in res.forced_viable] == ["d100"]
    assert [p.meta.design_id for p in res.forced_non_viable] == ["d101"]


def test_rates_and_log_row():
    roles = {0: "offspring", 1: "offspring", 2: "immigrant", 3: "offspring"}
    res = vb.fill_viable(target=10, forced=[], propose=_Proposer(role_of=lambda i: roles.get(i, "immigrant")),
                         search=_Search(viable={0, 2}), known={}, batch_size=4, max_batches=1)
    row = res.stats.log_row()
    assert row["offspring_viability_rate"] == pytest.approx(1 / 3)
    assert row["immigrant_viability_rate"] == pytest.approx(1.0)
    assert row["candidates_searched"] == 4 and row["grasp_search_s"] == 10.0 and row["batches"] == 1
    assert row["viability_rate"] == pytest.approx(2 / 4)
    assert row["founder_viability_rate"] is None  # no founder drawn


# --------------------------------------------------------------------------
# Driver integration (real grammar sampling, fake search)
# --------------------------------------------------------------------------


def test_viable_generation_keeps_elites_drops_non_viable_probes_and_fills_new_designs():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    a = arch.Archive()
    minter = drv.IdMinter()
    known = {}
    calls = []

    def search(batch):
        calls.append(len(batch))
        # probes: only allegro viable; new designs: every second one
        out = {}
        for j, p in enumerate(batch):
            if p.meta.role == "probe":
                out[p.sha256] = 9 if p.meta.hand_id == "allegro_right" else 0
            else:
                out[p.sha256] = 4 if j % 2 == 0 else 0
        return out, 1.0

    plan, report = vb.build_viable_generation(
        0, 8, ["allegro_right", "dclaw"], dist, a, rng, minter, known=known, search=search, batch_size=6,
        max_batches=4)
    roles = [m.role for m in plan.metas]
    assert len(plan.entries) == len(plan.metas) == 8
    assert roles.count("probe") == 1 and plan.metas[0].hand_id == "allegro_right"
    assert roles.count("founder") == 7
    assert report["probes_dropped"] == ["probe:dclaw"]
    assert report["n_designs"] == 8 and report["short_by"] == 0
    assert all(k in report for k in ("offspring_viability_rate", "candidates_searched", "grasp_search_s",
                                     "non_viable"))
    assert len({e.source for e in plan.entries}) == 8
    # every chosen design is known viable; the probe results are remembered
    assert all(known[e.sha256] > 0 for e in plan.entries)


def test_viable_generation_after_gen_0_reuses_elites_and_known_probes():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(1)
    minter = drv.IdMinter()
    known = {}

    def search(batch):
        return {p.sha256: 3 for p in batch}, 1.0

    plan0, _ = vb.build_viable_generation(0, 5, ["allegro_right"], dist, arch.Archive(), rng, minter,
                                          known=known, search=search, batch_size=8, max_batches=2)
    a = arch.Archive()
    cands = [arch.Candidate(
        design_id=m.design_id, derivation_dict=e.derivation_dict, sha256=e.sha256, founder_id=m.founder_id,
        parent_id=None, generation_born=0, digit_count=m.digit_count, joint_count=m.joint_count, fitness=1.0,
        episodes=20, low_confidence=False, source=m.source) for e, m in zip(plan0.entries, plan0.metas)
        if m.role != "probe"]
    a.update_generation(cands, 0)
    searched = []

    def search1(batch):
        searched.extend(p.meta.role for p in batch)
        return {p.sha256: 3 for p in batch}, 1.0

    plan1, report1 = vb.build_viable_generation(1, 5, ["allegro_right"], dist, a, rng, minter, known=known,
                                                search=search1, batch_size=8, max_batches=2)
    roles = [m.role for m in plan1.metas]
    assert roles.count("elite") == len(a.elites()) and roles.count("probe") == 1
    assert "probe" not in searched and "elite" not in searched  # known: no search
    assert len(plan1.entries) == 5


def test_viable_only_flag_needs_the_grasp_cache_and_adds_csv_columns():
    base = ["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x", "--task-profile", "hora",
            "--agent-entry-point", drv.ANYROTATE_POP_AGENT_ENTRY_POINT]
    with pytest.raises(SystemExit):
        drv.parse_args(base + ["--viable-only"])
    args = drv.parse_args(base + ["--grasp-cache", "--viable-only"])
    assert args.viable_only and drv._resolved_config(args)["viable_only"] is True
    assert "viable_only" not in drv._resolved_config(drv.parse_args(base + ["--grasp-cache"]))
    cols = drv.csv_columns(args)
    assert cols[: len(drv.CSV_COLUMNS)] == drv.CSV_COLUMNS
    assert {"offspring_viability_rate", "candidates_searched", "grasp_search_s", "n_non_viable"} <= set(cols)
    assert drv.csv_columns(drv.parse_args(base + ["--grasp-cache"])) == drv.CSV_COLUMNS


def test_state_round_trips_the_known_viability_map(tmp_path):
    a = arch.Archive()
    rng = np.random.default_rng(0)
    path = tmp_path / "state.json"
    drv.save_state(path, archive=a, driver_rng=rng, generation_completed=0, last_checkpoint=None,
                   prev_tolerance=0.0, minter=drv.IdMinter(), config={}, known_viability={"abc": 3, "def": 0})
    assert drv.load_state(path)["known_viability"] == {"abc": 3, "def": 0}


def test_founders_cycle_through_several_variants():
    """A mixed population (e.g. G_V3S and G_V1 founders): founders are drawn
    round-robin over the distributions."""
    dists = {"G_V3S": drv.resolve_variant("G_V3S"), "G_V1": drv.resolve_variant("G_V1")}
    rng = np.random.default_rng(0)
    seen = []

    def search(batch):
        seen.extend(p.meta.design_id for p in batch)
        return {p.sha256: 1 for p in batch}, 0.0

    plan, report = vb.build_viable_generation(0, 6, [], dists, arch.Archive(), rng, drv.IdMinter(), known={},
                                              search=search, batch_size=6, max_batches=1)
    assert len(plan.entries) == 6
    # every draw (pre-filter rejects included) takes the next variant
    assert report["founder_variants"] == [("G_V3S", "G_V1")[i % 2] for i in range(report["candidates_drawn"])]


def test_kit_cache_path_follows_the_job_environment(monkeypatch, tmp_path):
    """On the cluster each job sets a node-local OMNI_KIT_CACHE_PATH; the
    driver's subprocesses keep it. Unset: /tmp/$USER/ov_cache as before."""
    monkeypatch.setenv("OMNI_KIT_CACHE_PATH", str(tmp_path / "kit"))
    assert drv._write_env(Path("/tmp/x/ov_cache"))["OMNI_KIT_CACHE_PATH"] == str(tmp_path / "kit")
    monkeypatch.delenv("OMNI_KIT_CACHE_PATH")
    assert drv._write_env(Path("/tmp/x/ov_cache"))["OMNI_KIT_CACHE_PATH"] == "/tmp/x/ov_cache"


def test_design_score_poller_appends_each_new_window(tmp_path):
    import json
    import os

    from isaacsimenvs.inhand_reorient.tools import poll_design_scores as poll

    src, out = tmp_path / "scores.json", tmp_path / "windows.jsonl"
    state = {}
    assert poll.poll_once(src, out, state) is False  # nothing yet
    src.write_text(json.dumps({"steps": 1, "designs": {}}))
    assert poll.poll_once(src, out, state) is True
    assert poll.poll_once(src, out, state) is False  # unchanged
    src.write_text(json.dumps({"steps": 2, "designs": {}}))
    os.utime(src, (state["mtime"] + 5, state["mtime"] + 5))
    assert poll.poll_once(src, out, state) is True
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["steps"] for r in rows] == [1, 2] and all("_t" in r for r in rows)


def test_grasp_search_command_bounds_the_peak_joint_speed(tmp_path):
    """--viable-max-joint-speed (default 5 rad/s): a design whose joints sit at
    the 10 rad/s velocity limit while holding is not viable (2026-10-02:
    founders 373, 420, 101 dominated population training)."""
    base = ["--variant", "G_V3S", "--generations", "1", "--run-dir", str(tmp_path), "--task-profile", "hora",
            "--agent-entry-point", drv.ANYROTATE_POP_AGENT_ENTRY_POINT, "--grasp-cache", "--viable-only"]
    args = drv.parse_args(base)
    assert args.viable_max_joint_speed == 5.0
    cmd = drv.grasp_search_cmd(args, tmp_path / "p.json", tmp_path / "c.npz", tmp_path / "s")
    assert "env.anyrotate.grasp_max_joint_speed=5.0" in cmd
    assert "--task-profile" in cmd and cmd[cmd.index("--task-profile") + 1] == "hora"
    assert cmd[:3] == ["timeout", "-k", "30"]
    off = drv.parse_args(base + ["--viable-max-joint-speed", "-1"])
    assert "env.anyrotate.grasp_max_joint_speed=-1.0" in drv.grasp_search_cmd(off, tmp_path / "p.json",
                                                                             tmp_path / "c.npz", tmp_path / "s")


def test_surplus_viable_designs_become_spares_drawn_first_next_generation():
    """Viable designs beyond the target are returned as spares; the next
    generation proposes them first (known viable: no search)."""
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(2)
    known = {}

    def search_all(batch):
        return {p.sha256: 2 for p in batch}, 1.0

    plan0, report0 = vb.build_viable_generation(0, 3, [], dist, arch.Archive(), rng, drv.IdMinter(), known=known,
                                                search=search_all, batch_size=6, max_batches=1)
    spares = report0["spares"]
    assert len(plan0.entries) == 3 and len(spares) >= 1
    assert {"derivation_dict", "design_id", "role", "founder_id", "parent_id", "generation_born", "digit_count",
            "joint_count"} <= set(spares[0])
    searched = []

    def search1(batch):
        searched.extend(p.meta.design_id for p in batch)
        return {p.sha256: 2 for p in batch}, 1.0

    plan1, report1 = vb.build_viable_generation(1, 2, [], dist, arch.Archive(), rng, drv.IdMinter(1000), known=known,
                                                search=search1, batch_size=4, max_batches=1, spares=spares)
    ids = [m.design_id for m in plan1.metas]
    assert ids[: min(2, len(spares))] == [s["design_id"] for s in spares][: min(2, len(spares))]
    assert not set(ids) & set(searched)  # spares were not searched again
    assert report1["spares_used"] == min(2, len(spares))


def test_spare_pool_is_capped_and_round_trips_through_state(tmp_path):
    spares = [{"design_id": f"s{i}"} for i in range(100)]
    assert len(vb.cap_spares(spares, 64)) == 64 and vb.cap_spares(spares, 64)[0]["design_id"] == "s36"
    a = arch.Archive()
    path = tmp_path / "state.json"
    drv.save_state(path, archive=a, driver_rng=np.random.default_rng(0), generation_completed=0, last_checkpoint=None,
                   prev_tolerance=0.0, minter=drv.IdMinter(), config={}, known_viability={}, spares=spares[:2])
    assert drv.load_state(path)["spares"] == spares[:2]


def test_known_designs_do_not_count_in_this_generations_viability_rate():
    """A spare or a previously searched design was judged earlier: the rate
    counts only designs pre-filtered or searched this generation."""
    search = _Search(viable={2})
    res = vb.fill_viable(target=2, forced=[], propose=_Proposer(), search=search, known={"sha0": 7, "sha1": 0},
                         batch_size=1, max_batches=2)
    assert res.stats.viability_rate("offspring") == pytest.approx(1.0)  # d2 only: searched, viable
