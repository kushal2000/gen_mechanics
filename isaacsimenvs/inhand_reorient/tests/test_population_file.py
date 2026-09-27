"""CPU tests for `isaacsimenvs/inhand_reorient/scene/population_file.py`
(design note section 5, item 4: population round-trip and tamper
detection). Run with the isaacsim venv (see test_grammar_envelope.py's
docstring for why):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_population_file.py -q
"""

from __future__ import annotations

import json

import pytest

from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf


def _small_population(tmp_path):
    entries, rejections = pf.sampled_entries("G_SERIAL", range(30))
    entries = entries[:6]
    projected, _status, _reason = pf.projected_entry("allegro_right")
    assert projected is not None
    entries.append(projected)
    path = tmp_path / "pop.json"
    doc = pf.write_population(path, entries)
    return path, doc, entries


def test_sampled_entries_rejection_counts_sum_correctly():
    entries, rejections = pf.sampled_entries("DEFAULT", range(100))
    assert len(entries) + sum(rejections.values()) >= 100  # a rejection can hit >1 bucket
    assert len(entries) > 0


def test_write_then_load_round_trips(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    assert doc["schema"] == pf.POPULATION_SCHEMA
    assert doc["population_sha256"]
    loaded = pf.load_population(path)
    assert len(loaded) == len(entries)
    for design, entry in zip(loaded, entries):
        assert isinstance(design, ge.EnvelopeDesign)
        assert design.source == entry.source
        assert design.sha256 == entry.sha256  # threaded through for Part C's per-design score reports


def test_tampered_design_derivation_is_rejected(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    # Flip a seed in the first design's derivation -- sha256 must catch it
    # even though population_sha256 (computed over the per-design hashes,
    # which we do NOT touch) would otherwise still "match".
    raw["designs"][0]["derivation"]["seed"] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        pf.load_population(path)


def test_tampered_population_sha256_is_rejected(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["population_sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="population_sha256 mismatch"):
        pf.load_population(path)


def test_tampered_entry_sha256_is_rejected(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["designs"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        pf.load_population(path)


def test_wrong_schema_is_rejected(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["schema"] = "not-the-real-schema/0.0"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        pf.load_population(path)


def test_canonical_json_hash_is_stable_under_key_reordering():
    a = pf._canonical_json_bytes({"b": 1, "a": 2})
    b = pf._canonical_json_bytes({"a": 2, "b": 1})
    assert a == b
    assert pf._sha256_hex(a) == pf._sha256_hex(b)


# --------------------------------------------------------------------------
# Review item 5: provenance (grammar_version/envelope pinning, derived
# digest independent of the raw derivation hash).
# --------------------------------------------------------------------------


def test_load_population_checks_grammar_version(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["grammar_version"] = "not-the-real-version"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="grammar_version"):
        pf.load_population(path)


def test_load_population_checks_envelope(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["envelope"] = "not-the-real-envelope"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="envelope"):
        pf.load_population(path)


def test_load_population_checks_derived_digest(tmp_path):
    """A tampered/missing `derived_sha256` is fatal even though the raw
    derivation `sha256` (which does NOT cover canonicalize/palm_up output)
    still matches -- review item 5's "no digest covers the derived tables"
    gap."""
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["designs"][0]["derived_sha256"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="derived-table sha256 mismatch"):
        pf.load_population(path)


def test_load_population_requires_derived_digest_present(tmp_path):
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    del raw["designs"][0]["derived_sha256"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="no derived_sha256"):
        pf.load_population(path)


def test_projected_entry_is_marked_exempt_with_filtered_pairs():
    entry, status, reason = pf.projected_entry("sharpa_left_on_iiwa14")
    assert status == "admitted", (status, reason)
    assert entry.exempt_overlap is True
    assert entry.filtered_pairs


def test_sampled_entry_is_never_exempt():
    entries, _rejections = pf.sampled_entries("G_SERIAL", range(30))
    assert entries
    for e in entries:
        assert e.exempt_overlap is False
        assert e.filtered_pairs == ()
