"""Fake-env CPU tests for the population-path env wiring (opus-review-
phase2.md's "test adequacy" list: shuffled joint names; 3 designs over 7
envs; masks, defaults and spawn rows; zero penalty on ghost columns;
fingertip FK equal to the real tip; no spawn-triggered termination).

These do NOT import ``obs_utils``/``reward_utils``/``reset_utils``/
``scene_utils``/``env_cfg``/``env`` -- every one of those imports
``isaaclab.*`` at module scope, and (per ``test_config_sanity.py``'s own
docstring, confirmed by directly trying it) ``isaaclab/__init__.py``
unconditionally bootstraps Kit on import, crashing plain pytest collection.
Instead, this module reproduces the exact GATHER/PERMUTE arithmetic those
modules perform against ``env.hand_tables``/``env.scene_record`` (a
one-line ``table[design_idx][:, perm]`` in every case) directly against
plain torch tensors built from a real ``GrammarPopulation`` -- the same
data, the same indexing, just without an ``env`` object or any isaaclab
import standing in the way. See ``drop_detection.py``'s and
``goal_curriculum.py``'s docstrings for the same Kit-avoidance discipline.

Run with the isaacsim venv (see test_grammar_envelope.py's docstring):

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_population_env_wiring.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from hand_sampler.grammar.derive import derive, sample_derivation
from hand_sampler.robot_spec import design_index

from isaacsimenvs.inhand_reorient.drop_detection import object_below_palm
from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge
from isaacsimenvs.inhand_reorient.scene import population_file as pf

N_ENVS = 7
DROP_DISTANCE_M = 0.24  # matches env_cfg.ResetCfg.drop_distance_m's default


def _three_designs() -> ge.GrammarPopulation:
    """3 admitted designs: 2 sampled (G_SERIAL) + 1 projected commercial
    hand, canonicalized into one `GrammarPopulation` -- exactly what
    `author_grammar.setup_grammar_robot` builds from a population file."""
    dist = pf._variant_distribution("G_SERIAL")
    designs = []
    for seed in range(50):
        derivation = sample_derivation(seed, dist)
        model = derive(derivation)
        if ge.admit(model).ok:
            designs.append(ge.canonicalize(model, source=f"sampled:G_SERIAL:{seed}"))
        if len(designs) == 2:
            break
    assert len(designs) == 2, "need 2 admitted G_SERIAL designs for this test"

    entry, status, reason = pf.projected_entry("dclaw")
    assert status == "admitted", (status, reason)
    model = derive(pf.derivation_from_dict(entry.derivation_dict))
    designs.append(ge.canonicalize(model, source=entry.source))

    return ge.build_population(designs, n_sweep=10)


def _shuffled_phys_names(seed: int = 0) -> tuple[list[str], list[int]]:
    """A fake articulation-view joint order that DIFFERS from
    `ge.SLOT_NAMES` (the "shuffled joint names" case) -- and the permutation
    `scene_utils._resolve_population_joint_permutation` would resolve for
    it: `perm[col] = SLOT_NAMES.index(phys_names[col])`, so
    `table[..., perm]` reindexes a SLOT_NAMES-ordered row into phys-column
    order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(ge.N_SLOTS)
    phys_names = [ge.SLOT_NAMES[i] for i in order]
    assert phys_names != list(ge.SLOT_NAMES), "shuffle produced the identity permutation; reseed"
    perm = [ge.SLOT_NAMES.index(n) for n in phys_names]
    return phys_names, perm


def test_masks_defaults_and_spawn_rows_gather_correctly_under_a_shuffle():
    population = _three_designs()
    phys_names, perm = _shuffled_phys_names()
    perm_t = torch.as_tensor(perm, dtype=torch.long)

    design_idx = design_index(N_ENVS, population.n_designs)  # (7,) in {0,1,2}
    design_idx_t = torch.as_tensor(design_idx, dtype=torch.long)
    assert design_idx_t.shape == (N_ENVS,)
    assert set(design_idx.tolist()) <= {0, 1, 2}

    joint_valid = torch.as_tensor(population.joint_valid, dtype=torch.bool)
    default_pos = torch.as_tensor(population.default_joint_pos, dtype=torch.float32)
    joint_limits = torch.as_tensor(population.joint_limits, dtype=torch.float32)
    spawn_offset = torch.as_tensor(population.spawn_offset, dtype=torch.float32)
    fingertip_valid = torch.as_tensor(population.fingertip_valid, dtype=torch.bool)

    # -- masks: every env's row, in PHYS-COLUMN order, must equal that env's
    # OWN design's SLOT_NAMES-order row reindexed by `perm` -- checked
    # per-env, per-column, against the SLOT_NAMES-indexed source directly
    # (not merely "shapes match").
    valid_rows = joint_valid[design_idx_t][:, perm_t]
    assert valid_rows.shape == (N_ENVS, ge.N_SLOTS)
    for env_id in range(N_ENVS):
        d = int(design_idx[env_id])
        for col, name in enumerate(phys_names):
            slot = ge.SLOT_NAMES.index(name)
            assert bool(valid_rows[env_id, col]) == bool(population.joint_valid[d, slot])

    # -- defaults: same gather, and every valid joint's default sits inside
    # ITS OWN (phys-column-order) limits.
    default_rows = default_pos[design_idx_t][:, perm_t]
    limits_rows = joint_limits[design_idx_t][:, perm_t]
    assert default_rows.shape == (N_ENVS, ge.N_SLOTS)
    lo, hi = limits_rows[..., 0], limits_rows[..., 1]
    assert torch.all(default_rows >= lo - 1e-9)
    assert torch.all(default_rows <= hi + 1e-9)

    # -- spawn rows: no permutation involved (spawn_offset is (n,3), not
    # per-joint), just the design_idx gather -- each env's row must equal
    # ITS OWN design's spawn_offset exactly.
    spawn_rows = spawn_offset[design_idx_t]
    assert spawn_rows.shape == (N_ENVS, 3)
    for env_id in range(N_ENVS):
        d = int(design_idx[env_id])
        assert torch.allclose(spawn_rows[env_id], torch.as_tensor(population.spawn_offset[d], dtype=torch.float32))

    # -- fingertip validity rows: same gather pattern once more (obs_utils.
    # _fingertip_valid_mask), no per-joint permutation (fingertip_valid is
    # per-FINGER, not per-joint-slot).
    fingertip_rows = fingertip_valid[design_idx_t]
    assert fingertip_rows.shape == (N_ENVS, ge.N_FINGERS)
    for env_id in range(N_ENVS):
        d = int(design_idx[env_id])
        assert torch.equal(fingertip_rows[env_id], torch.as_tensor(population.fingertip_valid[d]))


def test_zero_penalty_on_ghost_columns():
    """Review item 9: whatever value a ghost column's RAW action holds, the
    action-penalty arithmetic (`(actions**2 * joint_mask).sum(-1)`,
    reward_utils.compute_rewards) must not see it -- and, separately
    (obs_utils.pre_physics_step's own fix), zeroing the action AT THE
    SOURCE (`actions * joint_mask`) must reproduce the SAME penalty as
    zeroing only inside the penalty sum, i.e. the two are equivalent, not
    merely "the penalty happens to come out the same by cancellation"."""
    population = _three_designs()
    phys_names, perm = _shuffled_phys_names(seed=1)
    perm_t = torch.as_tensor(perm, dtype=torch.long)
    design_idx = design_index(N_ENVS, population.n_designs)
    design_idx_t = torch.as_tensor(design_idx, dtype=torch.long)

    joint_valid = torch.as_tensor(population.joint_valid, dtype=torch.bool)
    joint_mask = joint_valid[design_idx_t][:, perm_t]
    assert joint_mask.any(dim=-1).all(), "every env must have at least one real joint"
    assert not joint_mask.all(), "this population must have at least one ghost column somewhere"

    rng = np.random.default_rng(2)
    actions = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(N_ENVS, ge.N_SLOTS)), dtype=torch.float32)
    # Inflate every GHOST column to a large, obviously-penalizable value --
    # if masking is broken, this would dominate the penalty.
    ghost = ~joint_mask
    inflated = torch.where(ghost, torch.full_like(actions, 1000.0), actions)

    penalty_mask_in_sum = ((inflated ** 2) * joint_mask).sum(dim=-1)
    zeroed_actions = inflated * joint_mask  # obs_utils.pre_physics_step's own fix
    penalty_after_source_zero = (zeroed_actions ** 2).sum(dim=-1)
    assert torch.allclose(penalty_mask_in_sum, penalty_after_source_zero)

    # And it must equal the REAL-joints-only penalty computed directly from
    # the UNinflated actions (proving the ghost inflation truly contributed
    # nothing, not just "the two masking styles agree with each other").
    real_only_penalty = ((actions ** 2) * joint_mask).sum(dim=-1)
    assert torch.allclose(penalty_mask_in_sum, real_only_penalty)


def test_fingertip_fk_equals_the_real_tip_per_env():
    """`population.fingertip_offsets[d]` (used by `palm_up`/graded scoring)
    must equal the ANALYTIC tip position from `authored_fk` for that design
    -- and, transitively through the `design_idx` gather, so must each
    env's own row."""
    population = _three_designs()
    design_idx = design_index(N_ENVS, population.n_designs)
    design_idx_t = torch.as_tensor(design_idx, dtype=torch.long)
    fingertip_offsets_per_env = torch.as_tensor(population.fingertip_offsets, dtype=torch.float32)[design_idx_t]
    fingertip_valid_per_env = torch.as_tensor(population.fingertip_valid, dtype=torch.bool)[design_idx_t]
    assert fingertip_offsets_per_env.shape == (N_ENVS, ge.N_FINGERS, 3)

    checked = 0
    for env_id in range(N_ENVS):
        d = int(design_idx[env_id])
        design = population.designs[d]
        pu = population.palm_up_results[d]
        T_mid = ge.authored_fk(design, pu.default_q)
        for f in range(ge.N_FINGERS):
            if not fingertip_valid_per_env[env_id, f]:
                continue
            base = f * ge.N_JOINTS_PER_FINGER
            used = [base + j for j in range(ge.N_JOINTS_PER_FINGER) if design.slot_valid[base + j]]
            last = max(used)
            tip = (T_mid[last] @ np.array([0.0, 0.0, float(design.slot_length[last]), 1.0]))[:3]
            row = fingertip_offsets_per_env[env_id, f].numpy()
            assert np.allclose(tip, row, atol=1e-6), (env_id, d, f)
            checked += 1
    assert checked > 0


def test_no_spawn_triggered_termination_per_env():
    """The object's spawn point, at reset, must never itself trip the
    below-palm drop check (`drop_detection.object_below_palm`) -- otherwise
    every episode would drop-terminate on step 0/1 regardless of policy
    behavior. Checked per env (through the `design_idx` gather, matching
    `reset_utils.reset_env_state`'s own `_object_spawn_offset` /
    `object.write_root_state_to_sim` -> `compute_terminations`'s
    `object_below_palm` call chain), using `spawn_height_above_palm_m`
    (world z after `base_rot`, the SAME quantity `admit`'s
    `check_spawn_height` gate already enforces at admission time -- this
    test re-derives it independently, through the population's OWN tables
    plus `design_idx`, rather than re-running `admit`)."""
    population = _three_designs()
    design_idx = design_index(N_ENVS, population.n_designs)

    for env_id in range(N_ENVS):
        d = int(design_idx[env_id])
        design = population.designs[d]
        pu = population.palm_up_results[d]
        height_above_palm = ge.spawn_height_above_palm_m(design, pu)
        # obj_pos_w_z - palm_pos_w_z == height_above_palm at the moment of
        # spawn (palm sits at its own body origin; the object is placed
        # `height_above_palm` above it along world z, by construction of
        # `spawn_offset`/`base_rot`).
        dropped = object_below_palm(
            torch.tensor([height_above_palm]), torch.tensor([0.0]), DROP_DISTANCE_M,
        )
        assert not bool(dropped[0]), (
            f"env {env_id} design {d} ({design.source}): spawn point "
            f"{height_above_palm * 1000:.1f} mm above palm trips the drop check"
        )
