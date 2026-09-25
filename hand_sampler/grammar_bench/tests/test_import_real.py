"""Cross-check our URDF import + FK against the Pinocchio oracle for the
committed dev-set real hands."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coords import independent_joints
from hand_sampler.grammar_bench.tolerances import ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = BENCH_DIR / "references"

DEV_HANDS = {
    "allegro_right": "fixtures/real/allegro_right/allegro_hand_description_right.urdf",
    "leap_right": "fixtures/real/leap_right/leap_hand_right.urdf",
    "barrett_bh": "fixtures/real/barrett_bh/bhand_model.urdf",
}

MOVABLE_TYPES = ("revolute", "continuous", "prismatic")


def _independent_dof_from_xml(urdf_path: Path) -> int:
    """Independent ElementTree count: movable joints minus mimic dependents,
    computed without going through hand_sampler.grammar."""
    root = ET.fromstring(urdf_path.read_bytes())
    n_movable = 0
    n_mimic = 0
    for jel in root.findall("joint"):
        if jel.attrib["type"] in MOVABLE_TYPES:
            n_movable += 1
            if jel.find("mimic") is not None:
                n_mimic += 1
    return n_movable - n_mimic


@pytest.mark.parametrize("hand_id", sorted(DEV_HANDS))
def test_import_and_fk_match_oracle(hand_id):
    urdf_path = BENCH_DIR / DEV_HANDS[hand_id]
    ref_path = REFERENCES_DIR / f"{hand_id}.json"
    if not ref_path.is_file():
        pytest.fail(f"missing oracle reference for committed hand {hand_id!r}: {ref_path}")

    reference = json.loads(ref_path.read_text())
    result = load_urdf(urdf_path)
    model = result.model

    # DoF check: independent joint count equals an independent ElementTree count
    # of movable joints minus mimic dependents.
    expected_dof = _independent_dof_from_xml(urdf_path)
    assert len(independent_joints(model)) == expected_dof

    max_pos = 0.0
    max_rot = 0.0
    n_compared = 0
    for u, oracle_poses in zip(reference["configs"], reference["poses"]):
        ours = fk.forward_kinematics(model, u)
        for body, T_oracle_list in oracle_poses.items():
            if body not in ours:
                continue
            T_oracle = np.array(T_oracle_list)
            pos_err = fk.position_error(ours[body], T_oracle)
            rot_err = fk.rotation_error(ours[body], T_oracle)
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n_compared += 1
            assert pos_err <= ORACLE_POS_M, f"{hand_id} body={body} pos_err={pos_err}"
            assert rot_err <= ORACLE_ROT_RAD, f"{hand_id} body={body} rot_err={rot_err}"

    assert n_compared > 0
    print(f"{hand_id}: {n_compared} body-pose comparisons, max_pos={max_pos:.3e}, max_rot={max_rot:.3e}")
