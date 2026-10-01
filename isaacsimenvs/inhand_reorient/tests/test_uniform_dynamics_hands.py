"""Uniform-dynamics commercial hands: same kinematics as the vendor-dynamics specs, gen-SHARPA's actuation.

Kit-free, loading the robot modules by path as test_unified_hands.py does.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG = "isaacsimenvs.pose_reaching_6d.scene_utils.robots"


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    import isaacsimenvs.pose_reaching_6d  # noqa: F401
    for n in ("isaacsimenvs.pose_reaching_6d.scene_utils", PKG, PKG + ".adjacency"):
        if n not in sys.modules:
            m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
    _load(PKG + ".adjacency.sharpa_iiwa14", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/sharpa_iiwa14.py")
    _load(PKG + ".adjacency.sharpa_handonly", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/adjacency/sharpa_handonly.py")
    _load(PKG + ".sharpa_handonly", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/sharpa_handonly.py")
    unified = _load(PKG + ".unified_hands", "isaacsimenvs/pose_reaching_6d/scene_utils/robots/unified_hands.py")
    uniform = _load(PKG + ".unified_dynamics_hands",
                    "isaacsimenvs/pose_reaching_6d/scene_utils/robots/unified_dynamics_hands.py")
    return unified, uniform


def test_every_hand_has_a_uniform_spec(mods):
    unified, uniform = mods
    assert set(uniform.UNIFORM_LEFT) == set(unified.UNIFIED_LEFT) | {"sharpa"}
    for h, s in uniform.UNIFORM_LEFT.items():
        assert s.name == f"{h}_left_uniform_handonly"
        s.validate()


def test_uniform_gains_and_limits(mods):
    _, uniform = mods
    rpc = _load("make_uniform", "assets/urdf/unified_dynamics_commercial_hands/make_uniform.py")
    for h, s in uniform.UNIFORM_LEFT.items():
        assert set(s.hand_stiffness.values()) == {rpc.STIFFNESS}, h
        assert np.allclose(list(s.hand_damping.values()), rpc.DAMPING), h
        assert np.allclose(list(s.hand_armature.values()), rpc.ARMATURE), h
        root = ET.parse(ROOT / s.urdf_path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        for n in s.hand_joint_names:
            lim = joints[n].find("limit")
            assert float(lim.get("effort")) == rpc.EFFORT, (h, n)
            assert float(lim.get("velocity")) == rpc.VELOCITY, (h, n)


def test_same_hand_under_the_canonical_map(mods):
    """Every link lands exactly where it did: vendor q -> canonical q = sign * q - offset."""
    import json
    import yourdfpy
    unified, uniform = mods
    cmap = json.loads((ROOT / "assets/urdf/unified_dynamics_commercial_hands/canonical_map.json").read_text())
    src = {h: ROOT / "assets/urdf/unified_commercial_hands" / h / f"{h}_left.urdf" for h in unified.UNIFIED_LEFT}
    src["sharpa"] = ROOT / "assets/urdf/unified_commercial_hands/sharpa/sharpa.urdf"
    rng = np.random.default_rng(0)
    for h, path in src.items():
        a = yourdfpy.URDF.load(str(path), load_meshes=False)
        b = yourdfpy.URDF.load(str(ROOT / uniform.UNIFORM_LEFT[h].urdf_path), load_meshes=False)
        names = [n for n, j in a.joint_map.items() if j.type == "revolute"]
        for _ in range(5):
            qa = {n: rng.uniform(a.joint_map[n].limit.lower, a.joint_map[n].limit.upper) for n in names}
            qb = {n: cmap[h][n]["sign"] * v - cmap[h][n]["offset"] for n, v in qa.items()}
            for n in qb:      # and the mapped angle is inside the new limits
                assert b.joint_map[n].limit.lower - 1e-6 <= qb[n] <= b.joint_map[n].limit.upper + 1e-6, (h, n)
            a.update_cfg(qa); b.update_cfg(qb)
            for link in a.link_map:
                assert np.allclose(a.get_transform(link), b.get_transform(link), atol=1e-6), (h, link)


def test_canonical_home_is_zero_and_same_tokens_otherwise(mods):
    unified, uniform = mods
    pairs = [(unified.UNIFIED_LEFT[h], uniform.UNIFORM_LEFT[h]) for h in unified.UNIFIED_LEFT]
    pairs.append((sys.modules[PKG + ".sharpa_handonly"].SHARPA_HANDONLY, uniform.UNIFORM_LEFT["sharpa"]))
    for a, b in pairs:
        assert set(b.hand_default_joint_pos.values()) == {0.0}, b.name
        assert a.hand_joint_names == b.hand_joint_names and a.fingertip_body_names == b.fingertip_body_names
        assert np.allclose(a.base_pos, b.base_pos) and np.allclose(a.base_rot, b.base_rot), a.name
        assert a.adjacent_links == b.adjacent_links, a.name
