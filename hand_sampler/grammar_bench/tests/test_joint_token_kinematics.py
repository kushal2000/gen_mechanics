"""The joint tokens' explicit kinematics (2026-11-04): each joint's axis and link vector, read from the URDF in its
child link's frame (`design_space.joint_axes_and_link_vectors`), checked against forward kinematics of the built URDF
(yourdfpy, independent of the grammar's own FK) at random poses: a link vector rotated by its link's pose lands on
the next joint or the fingertip, and turning a joint by +dq moves its fingertip by (axis x (tip - joint)) dq."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.design_space import joint_axes_and_link_vectors
from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as ops
from hand_sampler.grammar.urdf_export import build

yourdfpy = pytest.importorskip("yourdfpy")


@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(3)
    R = gh.FEETECH_COARSE_NOCUT_RULES
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(4):
            h = ops.random_hand(rng, R)
            row = build(h, f"h{k}", Path(tmp) / f"h{k}", rules_name="FEETECH_COARSE_NOCUT_RULES")
            spec = json.loads(Path(row["spec"]).read_text())
            u = yourdfpy.URDF.load(spec["urdf"], load_meshes=False, build_collision_scene_graph=False)
            out.append((h, spec, u))
        yield out


def _tip(u, spec, i, cfg):
    T = u.get_transform(frame_to=spec["fingertip_body_names"][i], frame_from="palm")
    return T[:3, :3] @ np.asarray(spec["fingertip_offsets"][i]) + T[:3, 3]


def test_link_vectors_reach_the_next_joint_or_the_tip(built):
    rng = np.random.default_rng(0)
    for h, spec, u in built:
        names = spec["hand_joint_names"]
        axes, vecs = joint_axes_and_link_vectors(spec["urdf"], names, spec["fingertip_body_names"],
                                                 spec["fingertip_offsets"])
        assert np.allclose(np.linalg.norm(axes, axis=1), 1.0)
        for _ in range(3):
            lo = np.array([spec["joint_limits"][n][0] for n in u.actuated_joint_names])
            hi = np.array([spec["joint_limits"][n][1] for n in u.actuated_joint_names])
            cfg = rng.uniform(lo, hi)
            u.update_cfg(cfg)
            for k, n in enumerate(names):
                i, j = (int(x) for x in n[1:].split("_j"))
                child = u.joint_map[n].child
                T = u.get_transform(frame_to=child, frame_from="palm")
                end = T[:3, :3] @ vecs[k] + T[:3, 3]
                if j + 1 < len(h.fingers[i].joints):
                    nxt = u.get_transform(frame_to=u.joint_map[f"f{i}_j{j + 1}"].child, frame_from="palm")[:3, 3]
                    assert np.allclose(end, nxt, atol=1e-9), n
                else:
                    assert np.allclose(end, _tip(u, spec, i, cfg), atol=1e-9), n
                L = h.fingers[i].joints[j].length
                assert L == 0 or np.linalg.norm(vecs[k]) > 0, n


def test_the_axis_says_which_way_the_tip_moves(built):
    rng = np.random.default_rng(1)
    d = 1e-6
    for h, spec, u in built:
        names = spec["hand_joint_names"]
        axes, _ = joint_axes_and_link_vectors(spec["urdf"], names, spec["fingertip_body_names"],
                                              spec["fingertip_offsets"])
        order = list(u.actuated_joint_names)
        lo = np.array([spec["joint_limits"][n][0] for n in order])
        hi = np.array([spec["joint_limits"][n][1] for n in order])
        cfg = rng.uniform(lo * 0.8, hi * 0.8)
        u.update_cfg(cfg)
        for k, n in enumerate(names):
            i = int(n[1:].split("_j")[0])
            T = u.get_transform(frame_to=u.joint_map[n].child, frame_from="palm")
            w, p = T[:3, :3] @ axes[k], T[:3, 3]
            tip0 = _tip(u, spec, i, cfg)
            c2 = cfg.copy()
            c2[order.index(n)] += d
            u.update_cfg(c2)
            moved = (_tip(u, spec, i, c2) - tip0) / d
            u.update_cfg(cfg)
            assert np.allclose(moved, np.cross(w, tip0 - p), atol=1e-5), n
