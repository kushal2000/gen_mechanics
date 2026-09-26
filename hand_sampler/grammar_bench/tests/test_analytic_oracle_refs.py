"""Pinocchio-vs-sympy cross-check for the two analytic fixtures (iteration 6).

The frozen oracle references under ``grammar_bench/references/offaxis_tree.json``
and ``coupled_finger.json`` were generated with Pinocchio reading the fixture
URDFs directly (see ``refgen/oracle_fk.py``); ``grammar_bench.analytic`` is an
independent, sympy-based closed-form FK that also parses the URDF text itself
(never importing ``hand_sampler.grammar``). Comparing the two, at the exact
configurations recorded in the frozen reference, is the actual
Pinocchio-vs-sympy cross-check other project notes have referred to -- this
module is what makes it a real, executable test rather than a claim. Also
checked here: our own ``hand_sampler.grammar.fk`` against the same frozen
Pinocchio reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar import fk
from hand_sampler.grammar.adapters.urdf import load_urdf
from hand_sampler.grammar.coords import q_from_u
from hand_sampler.grammar_bench import analytic
from hand_sampler.grammar_bench.tolerances import ANALYTIC_POS_M, ANALYTIC_ROT_RAD, ORACLE_POS_M, ORACLE_ROT_RAD

BENCH_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = BENCH_DIR / "fixtures" / "analytic"
REFERENCES_DIR = BENCH_DIR / "references"

FIXTURE_IDS = ("offaxis_tree", "coupled_finger")


@pytest.mark.parametrize("fixture_id", FIXTURE_IDS)
def test_frozen_pinocchio_reference_matches_sympy_and_our_fk(fixture_id):
    urdf_path = FIXTURES_DIR / f"{fixture_id}.urdf"
    ref_path = REFERENCES_DIR / f"{fixture_id}.json"
    assert ref_path.is_file(), f"missing frozen reference {ref_path}"
    reference = json.loads(ref_path.read_text())

    model = load_urdf(urdf_path).model
    ref_model = analytic.parse_urdf(urdf_path)

    configs = reference["configs"]
    ref_poses = reference["poses"]
    assert len(configs) == len(ref_poses) > 0

    max_pos_sym, max_rot_sym = 0.0, 0.0
    max_pos_ours, max_rot_ours = 0.0, 0.0
    n_compared = 0
    n_bodies = None
    for u, T_pin_by_body in zip(configs, ref_poses):
        q = q_from_u(model, u)
        sym = analytic.forward_kinematics(ref_model, q)
        ours = fk.forward_kinematics(model, q)

        assert set(sym) == set(T_pin_by_body), (
            f"{fixture_id}: sympy/pinocchio body-name mismatch, "
            f"sympy-only={set(sym) - set(T_pin_by_body)} pinocchio-only={set(T_pin_by_body) - set(sym)}"
        )
        assert set(ours) == set(T_pin_by_body), (
            f"{fixture_id}: ours/pinocchio body-name mismatch, "
            f"ours-only={set(ours) - set(T_pin_by_body)} pinocchio-only={set(T_pin_by_body) - set(ours)}"
        )
        n_bodies = len(T_pin_by_body)

        for body, T_list in T_pin_by_body.items():
            T_pin = np.array(T_list)

            pos_err_sym = fk.position_error(sym[body], T_pin)
            rot_err_sym = fk.rotation_error(sym[body], T_pin)
            max_pos_sym = max(max_pos_sym, pos_err_sym)
            max_rot_sym = max(max_rot_sym, rot_err_sym)
            assert pos_err_sym <= ANALYTIC_POS_M, f"{fixture_id} body={body} sympy-vs-pinocchio pos_err={pos_err_sym}"
            assert rot_err_sym <= ANALYTIC_ROT_RAD, f"{fixture_id} body={body} sympy-vs-pinocchio rot_err={rot_err_sym}"

            pos_err_ours = fk.position_error(ours[body], T_pin)
            rot_err_ours = fk.rotation_error(ours[body], T_pin)
            max_pos_ours = max(max_pos_ours, pos_err_ours)
            max_rot_ours = max(max_rot_ours, rot_err_ours)
            assert pos_err_ours <= ORACLE_POS_M, f"{fixture_id} body={body} ours-vs-pinocchio pos_err={pos_err_ours}"
            assert rot_err_ours <= ORACLE_ROT_RAD, f"{fixture_id} body={body} ours-vs-pinocchio rot_err={rot_err_ours}"

            n_compared += 1

    assert n_bodies is not None
    assert n_compared == len(configs) * n_bodies
    print(
        f"{fixture_id}: {n_compared} comparisons, sympy-vs-pinocchio max_pos={max_pos_sym:.3e} "
        f"max_rot={max_rot_sym:.3e}, ours-vs-pinocchio max_pos={max_pos_ours:.3e} max_rot={max_rot_ours:.3e}"
    )
