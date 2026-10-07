"""CPU tests for `isaacsimenvs/inhand_reorient/evolution/archive.py` (the
MAP-Elites archive: insertion, replacement, descriptors, founder tracking,
state round-trip). No isaaclab/Kit import anywhere in the module under
test, so plain `python3`/pytest is enough:

    env -u PYTHONPATH .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_evolution_archive.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from isaacsimenvs.inhand_reorient.evolution import archive as arc


def _cand(design_id, digit_count, joint_count, fitness, *, founder_id=None, parent_id=None,
          generation_born=0, episodes=100, low_confidence=False, source=""):
    return arc.Candidate(
        design_id=design_id,
        derivation_dict={"seed": hash(design_id) % 1000, "steps": []},
        sha256=f"sha-{design_id}",
        founder_id=founder_id or design_id,
        parent_id=parent_id,
        generation_born=generation_born,
        digit_count=digit_count,
        joint_count=joint_count,
        fitness=fitness,
        episodes=episodes,
        low_confidence=low_confidence,
        source=source,
    )


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------


def test_descriptor_cell_count_is_36():
    assert arc.N_CELLS == 36
    assert len(arc.all_cells()) == 36


@pytest.mark.parametrize("joint_count,expected_bin", [
    (1, 0), (5, 0), (6, 1), (10, 1), (11, 2), (15, 2), (16, 3), (20, 3),
    (21, 4), (25, 4), (26, 5), (32, 5), (36, 5),
])
def test_joint_count_bin_edges(joint_count, expected_bin):
    assert arc.joint_count_bin(joint_count) == expected_bin


def test_joint_count_bin_clamps_out_of_range():
    assert arc.joint_count_bin(0) == 0
    assert arc.joint_count_bin(-5) == 0
    assert arc.joint_count_bin(1000) == arc.N_JOINT_BINS - 1


@pytest.mark.parametrize("digit_count,expected_bin", [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5)])
def test_digit_bin_edges(digit_count, expected_bin):
    assert arc.digit_bin(digit_count) == expected_bin


def test_digit_bin_clamps_out_of_range():
    assert arc.digit_bin(0) == 0
    assert arc.digit_bin(99) == arc.N_DIGIT_BINS - 1


def test_descriptor_combines_both_bins():
    assert arc.descriptor(3, 12) == (2, 2)


def test_cell_label_is_stable_and_readable():
    assert arc.cell_label((0, 0)) == "d1_j1-5"
    assert arc.cell_label((5, 5)) == "d6_j26-36"


# --------------------------------------------------------------------------
# Insertion / replacement
# --------------------------------------------------------------------------


def test_empty_archive_has_no_coverage():
    a = arc.Archive()
    assert a.coverage() == 0
    assert a.qd_score() == 0.0
    assert a.best_fitness() is None
    assert a.mean_fitness() is None
    assert a.median_fitness() is None
    assert a.max_founder_share() is None


def test_single_insertion_creates_one_elite():
    a = arc.Archive()
    changed = a.update_generation([_cand("d0", 2, 8, 1.5)], generation=0)
    assert changed == [arc.descriptor(2, 8)]
    assert a.coverage() == 1
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "d0"
    assert elite.fitness == 1.5
    assert elite.generation_born == 0
    assert elite.fitness_history == [{"generation": 0, "fitness": 1.5, "episodes": 100, "low_confidence": False}]


def test_higher_fitness_candidate_replaces_lower_in_same_cell():
    """Realistic driver usage: the incumbent is ALWAYS resubmitted
    (re-evaluated) alongside any challenger in the same generation's
    batch -- see the archive module's own corollary about un-resubmitted
    incumbents having no protection at all."""
    a = arc.Archive()
    a.update_generation([_cand("d0", 2, 8, 1.0)], generation=0)
    changed = a.update_generation(
        [_cand("d0", 2, 8, 1.0), _cand("d1", 2, 8, 2.0)], generation=1,
    )
    assert changed == [arc.descriptor(2, 8)]
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "d1"
    assert elite.generation_born == 1  # a NEW design took the cell
    assert elite.fitness_history == [{"generation": 1, "fitness": 2.0, "episodes": 100, "low_confidence": False}]


def test_lower_fitness_candidate_does_not_replace_higher_in_same_cell():
    a = arc.Archive()
    a.update_generation([_cand("d0", 2, 8, 2.0)], generation=0)
    changed = a.update_generation(
        [_cand("d0", 2, 8, 2.0), _cand("d1", 2, 8, 1.0)], generation=1,
    )
    assert changed == []
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "d0"  # unchanged
    assert elite.generation_born == 0


def test_an_incumbent_not_resubmitted_offers_no_protection():
    """Documents the archive's own corollary: this module is a pure batch
    argmax over whatever candidates it is given THIS call -- if the caller
    omits the incumbent from the batch, any submitted challenger wins its
    cell unconditionally, even with a lower fitness than the incumbent's
    last recorded value. Real driver usage never hits this (every elite is
    always part of every generation's population), but the contract is
    exercised here directly since archive.py enforces nothing about it."""
    a = arc.Archive()
    a.update_generation([_cand("d0", 2, 8, 100.0)], generation=0)
    changed = a.update_generation([_cand("d1", 2, 8, 0.01)], generation=1)
    assert changed == [arc.descriptor(2, 8)]
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "d1"
    assert elite.fitness == pytest.approx(0.01)


def test_reevaluated_incumbent_updates_fitness_even_if_lower_than_before():
    """The archive's own docstring policy: a re-evaluated elite's fitness
    is always refreshed to this generation's measurement (a batch argmax
    over the generation's OWN candidates), never compared against its
    stale previous-generation value -- an elite that regresses under a new
    shared controller is recorded as regressed, not silently kept at its
    old high-water mark."""
    a = arc.Archive()
    a.update_generation([_cand("d0", 2, 8, 5.0)], generation=0)
    # Same design_id "d0" resubmitted with a WORSE fitness, no competing
    # offspring in this cell this generation.
    changed = a.update_generation([_cand("d0", 2, 8, 1.0, generation_born=0)], generation=1)
    assert changed == []  # same design, not a new occupant
    elite = a.get(arc.descriptor(2, 8))
    assert elite.fitness == 1.0
    assert elite.generation_born == 0  # generation_born is NOT reset for the same design
    assert [h["generation"] for h in elite.fitness_history] == [0, 1]
    assert [h["fitness"] for h in elite.fitness_history] == [5.0, 1.0]


def test_batch_argmax_within_one_generation_picks_the_best_candidate_for_the_cell():
    a = arc.Archive()
    changed = a.update_generation(
        [_cand("a", 2, 8, 1.0), _cand("b", 2, 8, 3.0), _cand("c", 2, 8, 2.0)], generation=0,
    )
    assert changed == [arc.descriptor(2, 8)]
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "b"
    assert elite.fitness == 3.0


def test_different_cells_are_independent():
    a = arc.Archive()
    a.update_generation([_cand("a", 1, 3, 1.0), _cand("b", 4, 20, 9.0)], generation=0)
    assert a.coverage() == 2
    assert a.get(arc.descriptor(1, 3)).design_id == "a"
    assert a.get(arc.descriptor(4, 20)).design_id == "b"


def test_summary_and_qd_score_and_stats():
    a = arc.Archive()
    a.update_generation([_cand("a", 1, 3, 1.0), _cand("b", 4, 20, 3.0)], generation=0)
    assert a.qd_score() == pytest.approx(4.0)
    assert a.best_fitness() == pytest.approx(3.0)
    assert a.mean_fitness() == pytest.approx(2.0)
    assert a.median_fitness() == pytest.approx(2.0)
    summary = a.summary()
    assert summary["coverage"] == 2
    assert summary["n_cells_total"] == 36
    assert set(summary["cells"].keys()) == {arc.cell_label(arc.descriptor(1, 3)), arc.cell_label(arc.descriptor(4, 20))}


# --------------------------------------------------------------------------
# Founder tracking
# --------------------------------------------------------------------------


def test_founder_counts_and_max_share():
    a = arc.Archive()
    a.update_generation([
        _cand("a", 1, 1, 1.0, founder_id="F1"),
        _cand("b", 2, 1, 1.0, founder_id="F1"),
        _cand("c", 3, 1, 1.0, founder_id="F2"),
    ], generation=0)
    counts = a.founder_counts()
    assert counts == {"F1": 2, "F2": 1}
    assert a.n_distinct_founders() == 2
    assert a.max_founder_share() == pytest.approx(2.0 / 3.0)


def test_founder_id_propagates_when_offspring_takes_over_a_cell():
    """A cell's occupant changing design (offspring beats its own parent)
    still carries the SAME founder_id forward if the offspring's own
    Candidate says so -- archive.py trusts the caller's founder_id, it
    never recomputes lineage itself."""
    a = arc.Archive()
    a.update_generation([_cand("parent", 2, 8, 1.0, founder_id="F1")], generation=0)
    a.update_generation([_cand("child", 2, 8, 2.0, founder_id="F1", parent_id="parent")], generation=1)
    elite = a.get(arc.descriptor(2, 8))
    assert elite.design_id == "child"
    assert elite.founder_id == "F1"
    assert elite.parent_id == "parent"


# --------------------------------------------------------------------------
# sample_parent
# --------------------------------------------------------------------------


def test_sample_parent_raises_on_empty_archive():
    a = arc.Archive()
    with pytest.raises(ValueError):
        a.sample_parent(np.random.default_rng(0))


def test_sample_parent_is_uniform_over_elites():
    a = arc.Archive()
    a.update_generation([_cand("a", 1, 1, 1.0), _cand("b", 2, 1, 1.0), _cand("c", 3, 1, 1.0)], generation=0)
    rng = np.random.default_rng(0)
    picks = [a.sample_parent(rng).design_id for _ in range(600)]
    counts = {k: picks.count(k) for k in ("a", "b", "c")}
    # 600 draws over 3 equally-likely elites: expect ~200 each; a generous
    # tolerance keeps this test from being flaky while still catching a
    # badly broken (e.g. always-first) sampler.
    for k in ("a", "b", "c"):
        assert 120 < counts[k] < 320, counts


# --------------------------------------------------------------------------
# State round-trip / resume
# --------------------------------------------------------------------------


def test_to_dict_from_dict_round_trips():
    a = arc.Archive()
    a.update_generation([_cand("a", 1, 3, 1.0, founder_id="F1")], generation=0)
    a.update_generation([_cand("a", 1, 3, 1.5, founder_id="F1")], generation=1)
    doc = a.to_dict()
    b = arc.Archive.from_dict(doc)
    assert b.coverage() == a.coverage()
    elite_a = a.get(arc.descriptor(1, 3))
    elite_b = b.get(arc.descriptor(1, 3))
    assert elite_a.design_id == elite_b.design_id
    assert elite_a.fitness_history == elite_b.fitness_history
    assert elite_a.derivation_dict == elite_b.derivation_dict


def test_from_dict_rejects_unknown_schema():
    with pytest.raises(ValueError):
        arc.Archive.from_dict({"schema": "not_a_real_schema", "cells": []})


def test_save_load_round_trips_through_disk(tmp_path):
    a = arc.Archive()
    a.update_generation([_cand("a", 3, 14, 2.5, founder_id="F9", parent_id="p", generation_born=2)], generation=2)
    path = tmp_path / "state" / "archive.json"
    a.save(path)
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()  # atomic: no leftover temp file
    b = arc.Archive.load(path)
    assert b.coverage() == 1
    elite = b.get(arc.descriptor(3, 14))
    assert elite.design_id == "a"
    assert elite.founder_id == "F9"
    assert elite.parent_id == "p"
    assert elite.generation_born == 2


def test_copy_is_a_deep_copy_independent_of_the_original():
    a = arc.Archive()
    a.update_generation([_cand("a", 1, 1, 1.0)], generation=0)
    b = a.copy()
    b.update_generation([_cand("b", 1, 1, 5.0)], generation=1)
    assert a.get(arc.descriptor(1, 1)).design_id == "a"
    assert b.get(arc.descriptor(1, 1)).design_id == "b"


def test_resume_after_several_generations_matches_uninterrupted_run(tmp_path):
    """Simulates the driver's own resume path: run N generations, save
    after each; loading the saved state after generation k and continuing
    must match running straight through, generation by generation."""
    rng = np.random.default_rng(7)

    def gen_candidates(gen):
        return [
            _cand(f"g{gen}-{i}", (i % 5) + 1, (i * 3) % 32 + 1, float(rng.integers(0, 100)),
                  founder_id=f"F{i % 4}", generation_born=gen)
            for i in range(10)
        ]

    straight = arc.Archive()
    for gen in range(5):
        straight.update_generation(gen_candidates(gen), gen)

    # Redo with an independent rng stream is wrong (would diverge); instead
    # replay the exact same candidate generation function by re-seeding.
    rng2 = np.random.default_rng(7)

    def gen_candidates2(gen):
        return [
            _cand(f"g{gen}-{i}", (i % 5) + 1, (i * 3) % 32 + 1, float(rng2.integers(0, 100)),
                  founder_id=f"F{i % 4}", generation_born=gen)
            for i in range(10)
        ]

    resumable = arc.Archive()
    path = tmp_path / "archive.json"
    for gen in range(3):
        resumable.update_generation(gen_candidates2(gen), gen)
        resumable.save(path)

    resumed = arc.Archive.load(path)
    for gen in range(3, 5):
        resumed.update_generation(gen_candidates2(gen), gen)

    assert resumed.to_dict() == straight.to_dict()
