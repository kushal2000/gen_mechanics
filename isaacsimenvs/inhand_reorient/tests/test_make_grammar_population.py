"""CPU test for `make_grammar_population.py`'s 16-design test population
composition (8 G_SERIAL, 4 carrier, 4 projected commercial hands). Run with
the isaacsim venv (see test_grammar_envelope.py's docstring):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_make_grammar_population.py -q
"""

from __future__ import annotations

from hand_sampler.grammar.derive import derive

from isaacsimenvs.inhand_reorient import make_grammar_population as mkpop
from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf


def test_build_test16_composition():
    entries = mkpop.build_test16()
    assert len(entries) == 16
    sources = [e.source for e in entries]
    assert sum(s.startswith("sampled:G_SERIAL:") for s in sources) == 8
    assert sum(s.startswith("sampled:DEFAULT_CARRIER:") for s in sources) == 4
    assert sum(s.startswith("projected:") for s in sources) == 4
    for hand_id in mkpop.DEFAULT_COMMERCIAL_HANDS:
        assert f"projected:{hand_id}" in sources


def test_carrier_entries_actually_use_a_real_carrier():
    entries = mkpop.collect_carrier_entries(4)
    for e in entries:
        model = derive(pf.derivation_from_dict(e.derivation_dict))
        design = ge.canonicalize(model, source=e.source)
        assert design.slot_valid[ge.PC0_SLOT] or design.slot_valid[ge.PC1_SLOT]


def test_serial_entries_never_use_a_carrier():
    entries = mkpop.collect_serial_entries(8)
    for e in entries:
        model = derive(pf.derivation_from_dict(e.derivation_dict))
        design = ge.canonicalize(model, source=e.source)
        assert not design.slot_valid[ge.PC0_SLOT]
        assert not design.slot_valid[ge.PC1_SLOT]


def test_write_test16_round_trips(tmp_path):
    entries = mkpop.build_test16()
    path = tmp_path / "test16.json"
    pf.write_population(path, entries)
    loaded = pf.load_population(path)
    assert len(loaded) == 16


def test_sampled_entries_have_no_bad_rest_overlap():
    """A rest-pose self-penetration (design note risk 4) was found, via a
    Kit smoke, to cause a violent (100+ rad in 1-2 steps) joint blowup --
    `ge.admit`'s default `check_overlap=True` (which every sampled-entry
    collector in make_grammar_population.py now relies on, review item 4)
    must reject any design whose rest_overlap_pairs penetration exceeds
    MAX_REST_PENETRATION_M -- confirmed here directly on every collected
    entry, not merely assumed from admission."""
    for entries in (mkpop.collect_serial_entries(8), mkpop.collect_carrier_entries(4)):
        for e in entries:
            model = derive(pf.derivation_from_dict(e.derivation_dict))
            design = ge.canonicalize(model, source=e.source)
            pairs = ge.rest_overlap_pairs(design)
            bad = [p for p in pairs if p[2] > ge.MAX_REST_PENETRATION_M]
            assert not bad, f"{e.source} has a bad rest overlap: {bad}"


def test_admit_rejects_a_known_bad_rest_overlap_design():
    """G_SERIAL seed 7 (traced by a Kit smoke to a 5.6-9.6 mm rest overlap
    between f0/f1/f2/f4, all root-mounted with no palm spread) is the
    concrete case the overlap filter exists for -- confirm it is
    STRUCTURALLY valid (`_admit_structural`, the envelope-shape check alone)
    but rejected by the full `ge.admit` gate (review item 4: the overlap
    check is now part of `admit` itself, so this isn't a vacuously-always-
    true check, and every sampled-entry path -- including `--variant
    sampled_only` -- shares it)."""
    from hand_sampler.grammar.derive import sample_derivation

    from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge2
    from isaacsimenvs.inhand_reorient.scene import population_file as pf2

    dist = pf2._variant_distribution("G_SERIAL")
    derivation = sample_derivation(7, dist)
    model = derive(derivation)
    assert ge2._admit_structural(model).ok
    result = ge2.admit(model)
    assert not result.ok
    assert any("rest-overlap" in r for r in result.reasons)
