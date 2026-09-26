"""Representation-check plan, item 3: acceptance tests for
``hand_sampler.grammar.experiments.e13_representation``.

1. Every ``manifest.json`` entry newly added/annotated for this item
   (``svh_right``, ``shadow_right_local``, ``arms_skel``, and SHARPA's added
   ``palm_joints``) resolves to a real file on this machine, with every
   named ``hand_root``/``palm_joints`` joint actually present in the parsed
   model (local-only skip if the file itself is absent).
2. The E13 core check + atlas, run via ``runner.run_experiment`` into a tmp
   dir on 3 known-committed hands, writes ``result.json``/``summary.md``
   with the required keys.
3. Every available hand (the full manifest's scored splits, plus the
   analytic ``coupled_finger`` fixture) passes at 5 mm / 10 degrees.
4. The atlas contains the grammar-gap table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.experiments import e13_representation as e13
from hand_sampler.grammar.experiments.runner import run_experiment
from hand_sampler.grammar.kinematics import MOVABLE_TYPES

MANIFEST = json.loads(e13.MANIFEST_PATH.read_text())

# Manifest ids this item's plan specifically added/annotated (local-only,
# absolute-path, commit_allowed=false entries) or that are otherwise
# exercised below. ``ARMS_LOCAL_IDS`` are checked for hand_root/palm_joints
# name-existence but are not asserted to PASS on machines lacking the file.
NEW_LOCAL_IDS = ["svh_right", "shadow_right_local", "arms_skel"]


def _manifest_hand(hand_id: str) -> dict:
    return next(h for h in MANIFEST["hands"] if h["id"] == hand_id)


# ---------------------------------------------------------------------------
# 1. Manifest entries resolve; hand_root/palm_joints/tip_frames names exist.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hand_id", NEW_LOCAL_IDS + ["sharpa_left_on_iiwa14"])
def test_new_manifest_entries_resolve_and_joint_names_exist(hand_id):
    hand = _manifest_hand(hand_id)
    manifest = MANIFEST
    path, availability, reason = e13._resolve_hand(hand, manifest)
    if availability != "available":
        pytest.skip(f"local-only:{hand_id} {reason}")

    imported = load_urdf(path, hand_root=hand.get("hand_root"))
    model = imported.model
    joint_names = {j.name for j in model.joints}
    link_names = {b.name for b in model.bodies}

    if hand.get("hand_root") is not None:
        assert hand["hand_root"] in link_names or hand["hand_root"] == model.root

    for jn in hand.get("palm_joints") or ():
        assert jn in joint_names, f"{hand_id}: palm_joints entry {jn!r} not found among joints"

    for body_name in (hand.get("tip_frames") or {}).keys():
        assert body_name in link_names, f"{hand_id}: tip_frames key {body_name!r} not a link in the model"


def test_sharpa_palm_joint_is_pinky_cmc_off_the_hand_root():
    hand = _manifest_hand("sharpa_left_on_iiwa14")
    assert hand.get("palm_joints") == ["left_5_pinky_CMC"]
    path, availability, _reason = e13._resolve_hand(hand, MANIFEST)
    if availability != "available":
        pytest.skip("local-only:sharpa_left_on_iiwa14 source not present on this machine")
    model = load_urdf(path, hand_root=hand["hand_root"]).model
    j = next(jj for jj in model.joints if jj.name == "left_5_pinky_CMC")
    assert j.type in MOVABLE_TYPES
    assert j.parent == hand["hand_root"]


# ---------------------------------------------------------------------------
# 2. Core check + atlas, via run_experiment, into a tmp dir, 3 hands.
# ---------------------------------------------------------------------------


def test_e13_core_via_run_experiment_writes_required_keys(tmp_path):
    out_dir = tmp_path / "E13_representation"
    result = run_experiment(
        "e13_representation", e13.run_e13,
        params={"n_configs": 8, "hand_ids": ["allegro_right", "leap_right", "coupled_finger"]},
        seeds=[0], out_dir=str(out_dir), allow_dirty=True,
    )
    assert (out_dir / "result.json").is_file()
    assert (out_dir / "summary.md").is_file()

    per_seed_result = result["per_seed"][0]["result"]
    for key in ("n_hands_total", "n_available", "n_pass", "n_fail",
                "max_pos_mm_all_hands", "max_axis_deg_all_hands", "max_tip_mm_all_hands",
                "_per_hand", "_atlas"):
        assert key in per_seed_result, f"missing required result key {key!r}"

    per_hand = per_seed_result["_per_hand"]
    assert per_seed_result["n_hands_total"] == 3
    assert {h["id"] for h in per_hand} == {"allegro_right", "leap_right", "coupled_finger"}
    for h in per_hand:
        assert h["availability"] == "available"
        assert h["passed"] is True
        for key in ("max_pos_mm", "max_axis_deg", "max_tip_mm", "joint_count_conserved", "report", "atlas"):
            assert key in h

    atlas = per_seed_result["_atlas"]
    assert atlas["n_hands_pooled"] == 3
    assert "grammar_gap_table" in atlas
    assert len(atlas["grammar_gap_table"]) > 0

    # The on-disk result.json round-trips the same structure (this is how
    # ``e13_representation.main`` gets its rich content -- ``run_experiment``
    # writes the full per-seed ``result`` dict, not just numeric aggregates).
    on_disk = json.loads((out_dir / "result.json").read_text())
    assert on_disk["per_seed"][0]["result"]["n_hands_total"] == 3


# ---------------------------------------------------------------------------
# 3. Every available hand passes at 5 mm / 10 deg.
# ---------------------------------------------------------------------------


def test_every_available_hand_passes_at_5mm_10deg():
    out = e13.run_e13(e13.SEED, n_configs=e13.N_RANDOM_CONFIGS)
    per_hand = out["_per_hand"]
    available = [h for h in per_hand if h["availability"] == "available"]
    assert len(available) >= 12, "expected most manifest hands (+coupled_finger) to resolve on this machine"

    failures = [(h["id"], h.get("max_pos_mm"), h.get("max_axis_deg"), h.get("max_tip_mm"))
                for h in available if not h.get("passed")]
    assert not failures, f"hands failing the 5mm/10deg representation check: {failures}"

    for h in available:
        assert h["max_pos_mm"] <= 5.0
        assert h["max_axis_deg"] <= 10.0
        assert h["max_tip_mm"] <= 5.0
        assert h["joint_count_conserved"] is True

    # The three new articulated-palm hands, when present on this machine,
    # must be among the scored/passing hands (not silently skipped by a
    # split filter).
    ids_present = {h["id"] for h in available}
    for hand_id in NEW_LOCAL_IDS:
        if hand_id in ids_present:
            assert any(h["id"] == hand_id and h["passed"] for h in available)


# ---------------------------------------------------------------------------
# 4. Atlas contains the grammar-gap table.
# ---------------------------------------------------------------------------


def test_atlas_grammar_gap_table_structure():
    out = e13.run_e13(e13.SEED, n_configs=8)
    gap = out["_atlas"]["grammar_gap_table"]
    assert len(gap) > 0
    required_params = {
        "digit_count (per hand)", "phalanx link length (m)", "mount_frac",
        "rest bend angle, continuation phalanges (deg)", "coupling multiplier",
    }
    present_params = {row["parameter"] for row in gap}
    assert required_params <= present_params

    for row in gap:
        assert {"parameter", "real", "grammar", "frac_outside"} <= set(row)
        assert {"min", "median", "max", "n"} <= set(row["real"])
