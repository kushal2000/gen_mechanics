"""Mutation history round trip, diff summary, sources."""

import json
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar.derive import derivation_to_dict, derive, sample_derivation

from gviewer import history as hist
from gviewer import sources as src
from gviewer.envload import REPO_ROOT


def _root():
    return src.sample("G_V3S", 8)[0]


def test_history_roundtrip():
    rng = np.random.default_rng(0)
    dist = src.distribution("G_V3S")
    h = hist.History()
    root = _root()
    h.reset(root, "G_V3S seed 8")
    ops = ["insert_phalanx", "step_mount", "add_minimal_digit", "step_axis"]
    for op in ops:
        res = src.mutate(h.current.derivation, dist, rng, op)
        assert res.child is not None, res.message
        h.push(res.child, op)
    assert [e.operator for e in h.entries] == [None] + ops
    assert h.current.derivation.lineage[-len(ops):] == tuple((op, 8) for op in ops)

    h.back()
    h.back()
    assert h.cursor == 2 and h.can_forward()
    h.forward()
    assert h.cursor == 3

    doc = json.loads(json.dumps(h.to_dict()))
    h2 = hist.History.from_dict(doc)
    assert h2.cursor == h.cursor
    assert [e.label for e in h2.entries] == [e.label for e in h.entries]
    for a, b in zip(h.entries, h2.entries):
        assert derivation_to_dict(a.derivation) == derivation_to_dict(b.derivation)
        derive(b.derivation)  # still derives

    # mutating from an earlier entry drops the later ones
    res = src.mutate(h.current.derivation, dist, rng, "step_limits")
    h.push(res.child, "step_limits")
    assert len(h.entries) == 5 and h.cursor == 4 and not h.can_forward()
    lines = h.lineage_lines()
    assert lines[-1].startswith("**>**") and "step_limits" in lines[-1]


def test_diff_summary_insert_phalanx():
    rng = np.random.default_rng(1)
    root = _root()
    res = src.mutate(root, src.distribution("G_V3S"), rng, "insert_phalanx")
    d = hist.diff(root, res.child)
    assert d.joints[1] == d.joints[0] + 1
    assert d.digits[0] == d.digits[1]
    assert sum(d.phalanges[1]) == sum(d.phalanges[0]) + 1
    assert any(a.startswith("Phalanx") for a in d.added)
    assert any("phalanx_count" in c for c in d.changed)
    assert d.lines()


def test_diff_summary_parameter_change():
    rng = np.random.default_rng(2)
    root = _root()
    res = src.mutate(root, src.distribution("G_V3S"), rng, "step_radius")
    d = hist.diff(root, res.child)
    assert d.joints[0] == d.joints[1] and not d.added and not d.removed
    assert any("capsule_radius_m" in c and "mm" in c for c in d.changed)


def test_variation_impossible_is_a_message():
    root = _root()  # G_V3S: no branch digits
    res = src.mutate(root, src.distribution("G_V3S"), np.random.default_rng(0), "remove_branch_digit")
    assert res.child is None and "VariationImpossible" in res.message


def test_mutate_until_viable():
    res = src.mutate_until_viable(_root(), src.distribution("G_V3S"), np.random.default_rng(5), max_tries=64)
    assert res.child is not None, res.message
    assert res.report["admitted"] and res.report["fingertips_reachable"] >= 2


def test_parse_documents(tmp_path):
    d0 = sample_derivation(3, src.distribution("G_V1"))
    single = src.parse_document(derivation_to_dict(d0), "x.json")
    assert single.kind == "derivation" and len(single.entries) == 1
    lst = src.parse_document([derivation_to_dict(d0), {"derivation": derivation_to_dict(d0)}], "l.json")
    assert len(lst.entries) == 2
    bad = src.parse_document({"foo": 1}, "bad.json")
    assert not bad.entries and bad.errors

    pop = REPO_ROOT / "outputs" / "viable" / "mixed32" / "population.json"
    if pop.is_file():
        lf = src.load_file(str(pop))
        assert lf.kind.startswith("population") and len(lf.entries) == 32 and not lf.errors
    states = sorted((REPO_ROOT / "outputs" / "evolution_pilot").glob("*/*/state.json"))
    if states:
        lf = src.load_file(str(states[0]))
        assert lf.kind.startswith("driver state") and lf.entries and not lf.errors
        assert lf.variant in src.NAMED_DISTRIBUTIONS
        derive(lf.entries[0].derivation)
