"""The wandb pose viewer draws a capsule-simulated cylinder as the capsule (its end spheres). Kit-free."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from hand_sampler.grammar import derive as dv
from hand_sampler.grammar import hand as gh
from isaacsimenvs.inhand_olympics.pose_viewer import _cylinders_as_capsules

ROOT = Path(__file__).resolve().parents[3]
HAND = ROOT / "assets/urdf/27oct_grammar_random_hands/coarse_s0_000"


def test_each_fingertip_gets_the_grammar_capsules_end_spheres():
    text = (HAND / "coarse_s0_000.urdf").read_text()
    out = ET.fromstring(_cylinders_as_capsules(text))
    hand = gh.hand_from_dict(json.loads((HAND / "coarse_s0_000.design.json").read_text()))
    links = {l.get("name"): l for l in out.findall("link")}
    for i, f in enumerate(hand.fingers):
        tip = dv.link_shape_mm(f, len(f.joints) - 1)                 # the grammar's own capsule, link frame, mm
        spheres = [v for v in links[f"f{i}_link{len(f.joints) - 1}"].findall("visual")
                   if v.find("geometry/sphere") is not None]
        assert len(spheres) == 2
        got = sorted((np.array(v.find("origin").get("xyz").split(), float) for v in spheres), key=lambda p: p[0])
        want = sorted((tip.p0 * dv.MM, tip.p1 * dv.MM), key=lambda p: p[0])
        assert np.allclose(got, want, atol=1e-7)
        assert all(abs(float(v.find("geometry/sphere").get("radius")) - tip.radius * dv.MM) < 1e-9 for v in spheres)
    # nothing else gains spheres
    n_cyl = sum(1 for v in ET.fromstring(text).iter("visual") if v.find("geometry/cylinder") is not None)
    assert sum(1 for v in out.iter("visual") if v.find("geometry/sphere") is not None) == 2 * n_cyl
