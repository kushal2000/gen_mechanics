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
