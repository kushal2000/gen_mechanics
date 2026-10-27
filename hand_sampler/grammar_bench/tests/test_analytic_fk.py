"""Analytic (sympy-derived) cross-checks for the hand-kinematics grammar.

``hand_sampler.grammar_bench.analytic`` is an independent reference: it parses
URDF text itself and never imports ``hand_sampler.grammar``.
"""

from pathlib import Path

import pytest

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coords import (
    admissible_box,
    check_limits,
    independent_joints,
    q_from_u,
    sample_configurations,
)
from hand_sampler.grammar.kinematics import UnsupportedConstruct
from hand_sampler.grammar_bench import analytic
from hand_sampler.grammar_bench.tolerances import ANALYTIC_POS_M, ANALYTIC_ROT_RAD, N_EXTREMAL, N_RANDOM

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "analytic"
SEED = 20260925
N_RANDOM_FOR_20 = N_RANDOM  # + N_EXTREMAL extremal = N_RANDOM + N_EXTREMAL configurations per fixture


def _configs_and_models(urdf_name):
    path = FIXTURES_DIR / urdf_name
    result = load_urdf(path)
    model = result.model
    ref_model = analytic.parse_urdf(path)
    configs = sample_configurations(model, N_RANDOM_FOR_20, SEED)
    expected = N_RANDOM + N_EXTREMAL
    assert len(configs) == expected, f"expected {expected} configurations, got {len(configs)}"
    return model, ref_model, configs


@pytest.mark.parametrize("urdf_name", ["offaxis_tree.urdf", "coupled_finger.urdf"])
def test_fk_matches_sympy_reference(urdf_name):
    model, ref_model, configs = _configs_and_models(urdf_name)
    max_pos = 0.0
    max_rot = 0.0
    n_compared = 0
    for u in configs:
        q = q_from_u(model, u)
        # Limits must be respected exactly by every sampled configuration.
        assert check_limits(model, q) == []
        ours = fk.forward_kinematics(model, q)
        ref = analytic.forward_kinematics(ref_model, q)
        assert set(ours) == set(ref)
        for body in ours:
            pos_err = fk.position_error(ours[body], ref[body])
            rot_err = fk.rotation_error(ours[body], ref[body])
            max_pos = max(max_pos, pos_err)
            max_rot = max(max_rot, rot_err)
            n_compared += 1
            assert pos_err <= ANALYTIC_POS_M, f"{urdf_name} body={body} pos_err={pos_err}"
            assert rot_err <= ANALYTIC_ROT_RAD, f"{urdf_name} body={body} rot_err={rot_err}"
    print(f"{urdf_name}: {n_compared} body comparisons, max_pos={max_pos:.3e}, max_rot={max_rot:.3e}")


def test_coupled_finger_couplings_expand_correctly():
    path = FIXTURES_DIR / "coupled_finger.urdf"
    model = load_urdf(path).model
    q = q_from_u(model, {"joint1": 0.3})
    # A coupled joint's value is exactly ``multiplier * q_source + offset``
    # (coords.py's own affine expansion), computed here with the identical
    # float arithmetic -- no float-noise tolerance needed or wanted.
    assert q["joint1"] == 0.3
    assert q["joint2"] == 1.5 * 0.3 + 0.1
    assert q["joint3"] == -1.0 * 0.3 + 0.0


def test_coupled_finger_limit_conflict_hand_computed():
    path = FIXTURES_DIR / "coupled_finger.urdf"
    model = load_urdf(path).model
    box, conflicts = admissible_box(model)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.dependent == "joint3"
    # joint3's coupling (coupled_finger.urdf): multiplier=-1.0, offset=0.0,
    # sourced from joint1 whose own limits are (-1.0, 1.0) -- the image is
    # exactly (multiplier * lo + offset, multiplier * hi + offset), swapped
    # since multiplier < 0, computed here with the identical float
    # arithmetic ``admissible_box`` uses, so this holds with ``==``, not a
    # tolerance.
    src_lo, src_hi = -1.0, 1.0
    multiplier, offset = -1.0, 0.0
    img_a, img_b = multiplier * src_lo + offset, multiplier * src_hi + offset
    expected_image = (min(img_a, img_b), max(img_a, img_b))
    assert c.image == expected_image
    # joint3's own declared limits in the fixture.
    expected_limit = (-0.5, 0.8)
    assert c.limit == expected_limit
    excess_lo = expected_limit[0] - expected_image[0]
    excess_hi = expected_image[1] - expected_limit[1]
    assert c.excess == excess_lo + excess_hi


def test_unsupported_construct_raised_for_floating_joint():
    urdf_text = """<?xml version="1.0"?>
<robot name="bad">
  <link name="a"/>
  <link name="b"/>
  <joint name="j" type="floating">
    <parent link="a"/>
    <child link="b"/>
  </joint>
</robot>"""
    with pytest.raises(UnsupportedConstruct) as excinfo:
        load_urdf(urdf_text)
    assert "j" in excinfo.value.issues[0]


@pytest.mark.parametrize("urdf_name", ["offaxis_tree.urdf", "coupled_finger.urdf"])
def test_sample_configurations_respect_declared_limits(urdf_name):
    path = FIXTURES_DIR / urdf_name
    model = load_urdf(path).model
    configs = sample_configurations(model, N_RANDOM_FOR_20, SEED)
    for u in configs:
        assert set(u) == set(independent_joints(model))
        q = q_from_u(model, u)
        assert check_limits(model, q) == []
