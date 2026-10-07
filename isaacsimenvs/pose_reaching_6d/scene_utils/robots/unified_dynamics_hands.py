"""The commercial hands with UNIFORM dynamics: same kinematics and geometry, gen-SHARPA's actuation.

URDFs and spec JSONs from ``assets/urdf/unified_dynamics_commercial_hands/make_uniform.py``: every joint
0.5 N.m / 10 rad/s, stiffness 3.0, damping 0.078, armature 0.00058, link
inertials from collision hulls at 1750 kg/m^3. Registered as ``<hand>_left_uniform_handonly``. Friction is not in the spec: uniform runs set
every contact to 0.5 in the env config.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from hand_sampler.design_space import joint_link_boxes
from isaacsimenvs.pose_reaching_6d.scene_utils.robots.sharpa_handonly import SHARPA_HANDONLY
from isaacsimenvs.pose_reaching_6d.scene_utils.robots.unified_hands import LEFT_HANDS, REPO, spec_from_json

UNIFORM_DIR = REPO / "assets/urdf/unified_dynamics_commercial_hands"


def _sharpa_uniform():
    s = json.loads((UNIFORM_DIR / "allegro/allegro_left.spec.json").read_text())   # the shared gains
    urdf = str((UNIFORM_DIR / "sharpa/sharpa_left.urdf").relative_to(REPO))
    joints = SHARPA_HANDONLY.hand_joint_names
    bodies, boxes, valid, scale = joint_link_boxes(str(REPO / urdf), joints)
    # Same palm rule as make_uniform.py's spec JSONs: palm vs every link within two joints is filtered.
    import importlib.util
    sp = importlib.util.spec_from_file_location("make_uniform", UNIFORM_DIR / "make_uniform.py")
    mu = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mu)
    palm, near = mu.palm_near_links(REPO / urdf)
    adj = {k: list(v) for k, v in SHARPA_HANDONLY.adjacent_links.items()}
    for link in near:
        if link not in adj.setdefault(palm, []):
            adj[palm].append(link)
        if palm not in adj.setdefault(link, []):
            adj[link].append(palm)
    return dataclasses.replace(
        SHARPA_HANDONLY, name="sharpa_left_uniform_handonly", urdf_path=urdf,
        hand_stiffness={n: float(s["stiffness"]) for n in joints},
        hand_damping={n: float(s["damping"]) for n in joints},
        hand_armature={n: float(s["armature"]) for n in joints},
        joint_link_bodies=tuple(bodies), joint_link_boxes=boxes, joint_geometry_valid=valid,
        hand_scale=float(scale),
        hand_default_joint_pos={n: 0.0 for n in joints},       # canonical: home is 0 on every joint
        adjacent_links=adj,
        notes="SHARPA left hand with uniform dynamics (unified_dynamics_commercial_hands/make_uniform.py).")


UNIFORM_LEFT = {h: spec_from_json(UNIFORM_DIR / h / f"{h}_left.spec.json", name=f"{h}_left_uniform_handonly")
                for h in LEFT_HANDS}
UNIFORM_LEFT["sharpa"] = _sharpa_uniform()

# Right hands with uniform dynamics (any <hand>/<hand>_right.spec.json; so far only Wuji v2, from the vendor's
# own right URDF): registered as <hand>_right_uniform_handonly. Kept apart from UNIFORM_LEFT so that
# "multi:uniform" (multi_hand.UNIFORM_HANDS, left hands) means what it always has.
UNIFORM_RIGHT = {}
for _p in sorted(UNIFORM_DIR.glob("*/*_right.spec.json")):
    _hand = _p.name[: -len("_right.spec.json")]
    UNIFORM_RIGHT[_hand] = spec_from_json(_p, name=f"{_hand}_right_uniform_handonly")

# Reduced variants (make_missing_fingers.py): the full hand minus some fingers, everything else identical --
# <hand>_<side>_<tag>.spec.json next to the full hand, tag no_<finger> or only_<a>_<b>... Registered as
# <hand>_<side>_uniform_handonly_<tag>.
MISSING_FINGER = {}
for _side in ("left", "right"):
    for _p in sorted(UNIFORM_DIR.glob(f"*/*_{_side}_*.spec.json")):
        _hand, _tag = _p.name[: -len(".spec.json")].split(f"_{_side}_", 1)
        _name = f"{_hand}_{_side}_uniform_handonly_{_tag}"
        MISSING_FINGER[_name] = spec_from_json(_p, name=_name)

__all__ = ["MISSING_FINGER", "UNIFORM_LEFT", "UNIFORM_RIGHT"]
