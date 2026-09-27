"""I33: ``resolve_hand_urdf`` must prefer the manifest's mesh-complete
source copy (``source_root``/``source_path``) over the in-repo, kinematics-
only ``fixture_path`` copy -- xhand_right/wuji_right/tesollo_dg5f_right have
collision geometry that is 100% mesh (no box/sphere fallback, unlike
allegro_right/dclaw), so the fixture copy (same URDF text, no ``meshes/``
directory beside it) silently loses every link's collision.

Needs numpy (``design_space.joint_link_boxes`` via ``build_hand_only_spec``);
run with ``.venv_isaacsim``'s python, same as ``test_hand_only_spec.py``:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_resolve_hand_urdf.py -q

Assumes this dev machine's ``~/karma`` checkout is present (the manifest's
``source_root``) -- same assumption ``hand_sampler/grammar_bench/evaluate.py``
and ``e13_representation.py`` already make for their own source-resolution
tests.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaacsimenvs.inhand_reorient.hand_only import (
    SOURCE_ROOT_ENV_VAR, build_hand_only_spec, manifest_entry, resolve_hand_urdf,
)
from isaacsimenvs.inhand_reorient.urdf_cutter import cut_urdf_to_hand

MESH_ONLY_HANDS = ("xhand_right", "wuji_right", "tesollo_dg5f_right")
BOX_COLLISION_HANDS = ("allegro_right", "dclaw")

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAMMAR_BENCH_DIR = REPO_ROOT / "hand_sampler" / "grammar_bench"
MANIFEST_SOURCE_ROOT = Path(
    "/home/singularity/karma/karma-data/all_urdfs/full_models_as_downloaded"
)


def _source_root_present() -> bool:
    return MANIFEST_SOURCE_ROOT.is_dir()


needs_source_root = pytest.mark.skipif(
    not _source_root_present(), reason=f"manifest source_root not on this machine: {MANIFEST_SOURCE_ROOT}"
)


# --------------------------------------------------------------------------
# resolve_hand_urdf: prefers the sha-verified source copy
# --------------------------------------------------------------------------


@needs_source_root
@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_mesh_only_hands_resolve_to_the_source_copy_by_default(hand_id, monkeypatch):
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    resolved = resolve_hand_urdf(hand_id)
    assert resolved.mesh_source == "source"
    assert resolved.urdf_path.is_file()
    entry = manifest_entry(hand_id)
    assert resolved.urdf_path == (MANIFEST_SOURCE_ROOT / entry["source_path"]).resolve()
    # The source copy sits next to its real meshes/ directory (unlike the
    # fixture, whose meshes/ was never committed).
    assert (resolved.urdf_path.parent.parent / "meshes").is_dir() or \
        any(p.name.lower() == "meshes" for p in resolved.urdf_path.parents[:3])


@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_falls_back_to_fixture_and_warns_when_source_root_is_bogus(hand_id, monkeypatch, capsys):
    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, "/nonexistent/nowhere")
    resolved = resolve_hand_urdf(hand_id)
    assert resolved.mesh_source == "fixture"
    entry = manifest_entry(hand_id)
    assert resolved.urdf_path == (GRAMMAR_BENCH_DIR / entry["fixture_path"]).resolve()
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "collision geometry may be missing" in captured.err


@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_falls_back_to_fixture_and_warns_on_sha256_mismatch(hand_id, monkeypatch, capsys, tmp_path):
    entry = manifest_entry(hand_id)
    source_path = entry["source_path"]
    bogus_root = tmp_path / "bogus_source_root"
    bogus_file = bogus_root / source_path
    bogus_file.parent.mkdir(parents=True, exist_ok=True)
    bogus_file.write_text("<robot name='not_the_real_urdf'></robot>")

    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, str(bogus_root))
    resolved = resolve_hand_urdf(hand_id)
    assert resolved.mesh_source == "fixture"
    captured = capsys.readouterr()
    assert "sha256" in captured.err
    assert "WARNING" in captured.err


@needs_source_root
@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_source_root_env_override_takes_priority_over_the_manifest_value(hand_id, monkeypatch):
    # Pointing the override at the SAME real source_root must resolve
    # identically to the no-override (manifest-default) case.
    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, str(MANIFEST_SOURCE_ROOT))
    resolved = resolve_hand_urdf(hand_id)
    assert resolved.mesh_source == "source"
    assert resolved.urdf_path == (MANIFEST_SOURCE_ROOT / manifest_entry(hand_id)["source_path"]).resolve()


def test_sharpa_is_unaffected_source_is_the_repo_asset(monkeypatch):
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    resolved = resolve_hand_urdf("sharpa")
    assert resolved.mesh_source == "repo"


def test_a_hand_unavailable_everywhere_still_raises_a_clear_error(tmp_path):
    # shadow_right: split "excluded", fixture_path None, source_path "Shadow"
    # (a directory, not a file) -- unavailable under source OR fixture
    # resolution regardless of this machine's source_root.
    with pytest.raises(ValueError, match="fixture_path"):
        resolve_hand_urdf("shadow_right")


@needs_source_root
def test_a_hand_with_no_fixture_but_an_available_source_copy_now_resolves(monkeypatch):
    # inspire_right: commit_allowed False (never committed to the repo, so
    # fixture_path is None), but its source copy is present locally --
    # unlike the license gate on committing it, using it for local-only
    # training is not restricted (grammar_bench.evaluate._resolve_hand and
    # scene/population_file._resolve_hand already resolve it the same way).
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    resolved = resolve_hand_urdf("inspire_right")
    assert resolved.mesh_source == "source"
    assert resolved.urdf_path.is_file()


# --------------------------------------------------------------------------
# collision_geometry_lost: per-link loud-warning trigger
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_fixture_copy_loses_all_collision_geometry(hand_id, monkeypatch):
    # Force the fixture path directly (bypassing resolve_hand_urdf) to prove
    # the ORIGINAL bug this issue fixes: cutting the fixture copy silently
    # drops every kept link's collision geometry, because none of these
    # hands have a box/sphere fallback the way allegro_right/dclaw do.
    entry = manifest_entry(hand_id)
    fixture_path = (GRAMMAR_BENCH_DIR / entry["fixture_path"]).resolve()
    cut = cut_urdf_to_hand(fixture_path, entry["hand_root"])
    assert len(cut.collision_geometry_lost) > 0
    # Every kept link that had collision in the source URDF lost it (100%
    # mesh collision, confirmed by inspection of the fixture files).
    source_root_el = ET.parse(fixture_path).getroot()
    had_collision = {
        l.get("name") for l in source_root_el.findall("link")
        if l.get("name") in cut.kept_links and l.findall("collision")
    }
    assert set(cut.collision_geometry_lost) == had_collision


@needs_source_root
@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_source_copy_keeps_all_collision_geometry(hand_id, monkeypatch):
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    resolved = resolve_hand_urdf(hand_id)
    assert resolved.mesh_source == "source"
    cut = cut_urdf_to_hand(resolved.urdf_path, resolved.hand_root)
    assert cut.collision_geometry_lost == ()
    assert cut.unresolved_meshes == ()


@pytest.mark.parametrize("hand_id", BOX_COLLISION_HANDS)
def test_box_collision_hands_never_lose_collision_from_the_fixture(hand_id):
    # allegro_right/dclaw: collision is boxes/spheres with a mesh collision
    # only as an EXTRA shape on some links, so an unresolved mesh does not
    # empty out the link's collision -- the original bug report's "no
    # meshes anywhere" claim did not apply to these two.
    entry = manifest_entry(hand_id)
    fixture_path = (GRAMMAR_BENCH_DIR / entry["fixture_path"]).resolve()
    cut = cut_urdf_to_hand(fixture_path, entry["hand_root"])
    assert cut.collision_geometry_lost == ()


def test_collision_lost_is_empty_when_a_link_keeps_a_box_fallback(tmp_path):
    # Synthetic regression: a link with BOTH a mesh collision (unresolvable)
    # and a box collision must NOT be reported lost.
    urdf = tmp_path / "synthetic.urdf"
    urdf.write_text("""<?xml version="1.0"?>
<robot name="synthetic">
  <link name="root">
    <collision>
      <geometry><mesh filename="meshes/does_not_exist.obj"/></geometry>
    </collision>
    <collision>
      <geometry><box size="0.01 0.01 0.01"/></geometry>
    </collision>
  </link>
  <link name="mesh_only_child">
    <collision>
      <geometry><mesh filename="meshes/also_missing.obj"/></geometry>
    </collision>
  </link>
  <link name="no_collision_child"/>
  <joint name="j1" type="revolute">
    <parent link="root"/><child link="mesh_only_child"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="mesh_only_child"/><child link="no_collision_child"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""")
    cut = cut_urdf_to_hand(urdf, "root")
    assert cut.collision_geometry_lost == ("mesh_only_child",)
    assert "meshes/does_not_exist.obj" in cut.unresolved_meshes
    assert "meshes/also_missing.obj" in cut.unresolved_meshes


# --------------------------------------------------------------------------
# End to end: build_hand_only_spec warns loudly on the fixture fallback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_build_hand_only_spec_warns_when_forced_onto_the_fixture(hand_id, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv(SOURCE_ROOT_ENV_VAR, "/nonexistent/nowhere")
    spec, cut = build_hand_only_spec(hand_id, out_dir=tmp_path)
    assert len(cut.collision_geometry_lost) > 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "had collision geometry" in captured.err
    assert str(len(cut.collision_geometry_lost)) in captured.err


@needs_source_root
@pytest.mark.parametrize("hand_id", MESH_ONLY_HANDS)
def test_build_hand_only_spec_is_clean_with_the_source_copy(hand_id, monkeypatch, capsys, tmp_path):
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    spec, cut = build_hand_only_spec(hand_id, out_dir=tmp_path)
    assert cut.collision_geometry_lost == ()
    captured = capsys.readouterr()
    assert "collision geometry" not in captured.err
    spec.validate()


# --------------------------------------------------------------------------
# fingertip_body_names correctness (I34, 2026-09-27): xhand_right's URDF
# branches a dead-end fixed-joint "back" stub off each MID-CHAIN link
# (right_hand_*_rota_link1) as well as off the true tip's own parent
# (right_hand_*_rota_link2/*back_link2), so the pre-I34 naive leaf merge
# (leaf_link_names + merged_body_name) returned 11 names for a 5-fingered
# hand -- including right_hand_link, the PALM ROOT itself (via the ee_link/
# back_link dead ends off it) -- instead of the 5 real distal tips.
# merged_terminal_names (see test_urdf_cutter.py) fixes this generally;
# these are the end-to-end regression against the real hand.
# --------------------------------------------------------------------------

XHAND_RIGHT_TRUE_TIPS = frozenset({
    "right_hand_thumb_rota_link2", "right_hand_index_rota_link2",
    "right_hand_mid_link2", "right_hand_ring_link2", "right_hand_pinky_link2",
})


@needs_source_root
def test_xhand_right_fingertips_are_the_5_real_distal_tips_not_knuckles_or_palm(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    spec, _cut = build_hand_only_spec("xhand_right", out_dir=tmp_path)
    assert set(spec.fingertip_body_names) == XHAND_RIGHT_TRUE_TIPS
    assert len(spec.fingertip_body_names) == 5
    assert spec.palm_body_name not in spec.fingertip_body_names


@needs_source_root
@pytest.mark.parametrize("hand_id", ("wuji_right", "tesollo_dg5f_right"))
def test_other_mesh_only_hands_fingertips_unchanged_by_the_terminus_filter(
    hand_id, monkeypatch, tmp_path
):
    # wuji_right/tesollo_dg5f_right have no dead-end branching (each finger's
    # leaf chain ends in exactly one real tip link) -- the I34 filter must be
    # a no-op for them: still 5 fingertips, one per finger.
    monkeypatch.delenv(SOURCE_ROOT_ENV_VAR, raising=False)
    spec, _cut = build_hand_only_spec(hand_id, out_dir=tmp_path)
    assert len(spec.fingertip_body_names) == 5
    assert spec.palm_body_name not in spec.fingertip_body_names
