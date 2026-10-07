"""CPU tests for `isaacsimenvs/inhand_reorient/scene/population_file.py`
(design note section 5, item 4: population round-trip and tamper
detection). Run with the isaacsim venv (see test_grammar_envelope.py's
docstring for why):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_population_file.py -q
"""

from __future__ import annotations

import json

import numpy as np
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


# --------------------------------------------------------------------------
# oracle-v2 (opus-review-g0.md items 1/3/4): schema bump + derived_sha256
# now also covers palm_up's own derived fields.
# --------------------------------------------------------------------------


def test_old_pre_oracle_v2_schema_is_rejected(tmp_path):
    """A population file built under the OLD (pre-fix) oracle carries the
    old schema tag and must fail LOUDLY, not silently re-validate under the
    new geometry/spawn/reach code."""
    path, doc, entries = _small_population(tmp_path)
    raw = json.loads(path.read_text())
    raw["schema"] = "grammar_population/0.2"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        pf.load_population(path)


def test_derived_digest_depends_on_palm_up_output(monkeypatch):
    """Review item 5's own gap this fix closes: `_derived_digest` used to
    hash only `canonicalize`'s tables, so a `palm_up`-only change (exactly
    what the spawn-point/reach fix makes) left it unchanged. Confirm the
    digest actually moves when `palm_up`'s output does, by patching it to
    perturb `spawn_offset`."""
    entries, _rejections = pf.sampled_entries("G_SERIAL", range(30))
    assert entries
    from hand_sampler.grammar.derive import derive as _derive

    model = _derive(pf.derivation_from_dict(entries[0].derivation_dict))
    design = ge.canonicalize(model, source=entries[0].source)
    digest_before = pf._derived_digest(design)

    real_palm_up = ge.palm_up

    def _perturbed(design_arg, *args, **kwargs):
        result = real_palm_up(design_arg, *args, **kwargs)
        result.spawn_offset = result.spawn_offset + np.array([0.001, 0.0, 0.0])
        return result

    monkeypatch.setattr(pf.ge, "palm_up", _perturbed)
    digest_after = pf._derived_digest(design)
    assert digest_before != digest_after


def test_write_population_marks_projected_hands_exempt_at_both_poses(tmp_path):
    """`make_entry`'s `filtered_pairs` for an exempt (projected commercial
    hand) design must come from the UNION of q=0 and default_q overlap
    (opus review G0 item 3 applied to the authoring-time exemption, not just
    admission) -- a regression against `mark_filtered_pairs` silently going
    back to q=0-only."""
    entry, status, reason = pf.projected_entry("sharpa_left_on_iiwa14")
    assert status == "admitted", (status, reason)
    model = pf.derive(pf.derivation_from_dict(entry.derivation_dict))
    design = ge.canonicalize(model, source=entry.source)
    default_q = ge.palm_up(design, n_sweep=0).default_q
    q0_pairs = {(i, j) for i, j, pen in ge.rest_overlap_pairs(design) if pen > 0.0}
    default_pairs = {(i, j) for i, j, pen in ge.rest_overlap_pairs(design, q=default_q) if pen > 0.0}
    # plus the pairs joined through its 0 mm bones, which canonicalize records for every design
    assert set(entry.filtered_pairs) == q0_pairs | default_pairs | set(ge.short_bone_pairs(design))
