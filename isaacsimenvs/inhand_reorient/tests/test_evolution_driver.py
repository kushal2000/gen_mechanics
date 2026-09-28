"""CPU tests for `isaacsimenvs/inhand_reorient/evolution/driver.py`'s
non-subprocess pieces: variant resolution, gen-0/offspring/immigrant
population assembly, train_tail fitness, checkpoint discovery, and the
state.json save/load/resume contract. Nothing here boots Kit or launches a
subprocess -- `run_training_subprocess`/`run_generation`/`main` (which do)
are exercised by the tiny end-to-end GPU run instead (see the worker
report), not by this file.

    env -u PYTHONPATH .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_evolution_driver.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaacsimenvs.inhand_reorient.evolution import archive as arch
from isaacsimenvs.inhand_reorient.evolution import driver as drv
from isaacsimenvs.inhand_reorient.scene import population_file as pf


# --------------------------------------------------------------------------
# Variant resolution
# --------------------------------------------------------------------------


def test_resolve_variant_finds_named_distributions():
    dist = drv.resolve_variant("G_SERIAL")
    assert dist.palm_body_count_range == (0, 0)


def test_resolve_variant_finds_g0_screen_variants():
    dist = drv.resolve_variant("V1")  # G0_SCREEN_VARIANTS' own alias for G_V1
    assert dist.digit_count_range == (1, 5)


def test_resolve_variant_unwraps_tuple_entries(monkeypatch):
    """Some registered variants (e.g. `variants.G_FULL_SMALL`) are `(dist,
    operators)` pairs meant for `vary`'s default operator pool -- the
    pilot always uses EVOLUTION_OPERATORS regardless (plan-rl-grammar-
    tuning.md), so `resolve_variant` must unwrap the Distribution half
    rather than handing a raw tuple to the rest of the driver."""
    from hand_sampler.grammar import variants as grammar_variants

    real_dist = drv.resolve_variant("G_SERIAL")
    fake_registry = dict(grammar_variants.NAMED_DISTRIBUTIONS)
    fake_registry["_FAKE_PAIR"] = (real_dist, ("resample_parameter",))
    monkeypatch.setattr(grammar_variants, "NAMED_DISTRIBUTIONS", fake_registry)
    dist = drv.resolve_variant("_FAKE_PAIR")
    assert dist is real_dist


def test_resolve_variant_raises_with_a_helpful_message_for_unknown_names():
    with pytest.raises(ValueError, match="unknown --variant"):
        drv.resolve_variant("NOT_A_REAL_VARIANT")


# --------------------------------------------------------------------------
# sample_new_founder / try_offspring
# --------------------------------------------------------------------------


def test_sample_new_founder_returns_an_admitted_design():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    derivation, design = drv.sample_new_founder(dist, rng)
    model = None
    from hand_sampler.grammar.derive import derive
    from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge

    model = derive(derivation)
    assert ge.admit(model).ok
    digit_count, joint_count = drv._design_counts(design)
    assert 1 <= digit_count <= 5
    assert 1 <= joint_count <= 32


def test_try_offspring_returns_an_admitted_mutant_or_none():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(1)
    parent_derivation, _parent_design = drv.sample_new_founder(dist, rng)
    result = drv.try_offspring(parent_derivation, dist, rng, max_retries=16)
    # Either it succeeds (and is admitted) or politely gives up (None) --
    # both are valid outcomes; what matters is it never raises.
    if result is not None:
        from hand_sampler.grammar.derive import derive
        from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge

        candidate_derivation, _design = result
        assert ge.admit(derive(candidate_derivation)).ok


# --------------------------------------------------------------------------
# build_generation_population
# --------------------------------------------------------------------------


def _archive_with_one_elite(dist, rng):
    a = arch.Archive()
    derivation, design = drv.sample_new_founder(dist, rng)
    digit_count, joint_count = drv._design_counts(design)
    from hand_sampler.grammar.derive import derivation_to_dict

    cand = arch.Candidate(
        design_id="seed-elite", derivation_dict=derivation_to_dict(derivation), sha256="deadbeef",
        founder_id="seed-elite", parent_id=None, generation_born=0, digit_count=digit_count,
        joint_count=joint_count, fitness=1.0, episodes=50, low_confidence=False,
    )
    a.update_generation([cand], generation=0)
    return a


def test_generation_0_population_is_all_founders_plus_probes():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    a = arch.Archive()
    minter = drv.IdMinter()
    plan = drv.build_generation_population(0, 12, list(drv.DEFAULT_PROBES), dist, a, rng, minter)
    assert len(plan.entries) == 12
    roles = [m.role for m in plan.metas]
    assert roles.count("probe") == len(drv.DEFAULT_PROBES)
    assert roles.count("founder") == 12 - len(drv.DEFAULT_PROBES)
    assert set(roles) == {"probe", "founder"}
    # every entry's source is unique (design_scoring matches per-design rows by it)
    sources = [e.source for e in plan.entries]
    assert len(sources) == len(set(sources))


def test_later_generation_carries_elites_and_fills_with_offspring_or_immigrants():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    a = _archive_with_one_elite(dist, rng)
    minter = drv.IdMinter()
    plan = drv.build_generation_population(1, 12, list(drv.DEFAULT_PROBES), dist, a, rng, minter)
    roles = [m.role for m in plan.metas]
    assert roles.count("elite") == 1
    assert roles.count("probe") == len(drv.DEFAULT_PROBES)
    assert roles.count("offspring") + roles.count("immigrant") == 12 - 1 - len(drv.DEFAULT_PROBES)
    # the elite's own founder_id/parent_id/generation_born pass through unchanged
    elite_meta = next(m for m in plan.metas if m.role == "elite")
    assert elite_meta.design_id == "seed-elite"
    assert elite_meta.founder_id == "seed-elite"
    assert elite_meta.generation_born == 0
    # offspring propagate the parent's founder_id and point parent_id at it
    for m in plan.metas:
        if m.role == "offspring":
            assert m.founder_id == "seed-elite"
            assert m.parent_id == "seed-elite"
            assert m.generation_born == 1


def test_population_entries_round_trip_through_write_and_load_population(tmp_path):
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(3)
    a = _archive_with_one_elite(dist, rng)
    minter = drv.IdMinter()
    plan = drv.build_generation_population(1, 10, list(drv.DEFAULT_PROBES), dist, a, rng, minter)
    path = tmp_path / "pop.json"
    doc = pf.write_population(path, plan.entries)
    assert doc["population_sha256"]
    loaded = pf.load_population(path)
    assert len(loaded) == len(plan.entries)


def test_probes_never_appear_as_offspring_parents_or_founders():
    """Probes are logged every generation but never enter the archive, so
    they can never become a `sample_parent` candidate either."""
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    a = _archive_with_one_elite(dist, rng)
    minter = drv.IdMinter()
    plan = drv.build_generation_population(1, 12, list(drv.DEFAULT_PROBES), dist, a, rng, minter)
    parent_ids = {m.parent_id for m in plan.metas if m.parent_id is not None}
    assert not any(p.startswith("probe:") for p in parent_ids)


def test_ids_minted_by_the_same_minter_never_collide_across_generations():
    dist = drv.resolve_variant("G_V1")
    rng = np.random.default_rng(0)
    a = arch.Archive()
    minter = drv.IdMinter()
    all_ids = []
    for gen in range(3):
        plan = drv.build_generation_population(gen, 10, list(drv.DEFAULT_PROBES), dist, a, rng, minter)
        all_ids.extend(m.design_id for m in plan.metas if m.role != "probe" and m.role != "elite")
    assert len(all_ids) == len(set(all_ids))


# --------------------------------------------------------------------------
# build_train_cmd
# --------------------------------------------------------------------------


def test_build_train_cmd_omits_checkpoint_and_tolerance_for_generation_zero():
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=512, max_epochs=30,
        hydra_run_dir=Path("/tmp/run/gen_0/train"), checkpoint=None, resume_success_tolerance=None,
        horizon_length=16,
    )
    joined = " ".join(cmd)
    assert "--checkpoint" not in cmd
    assert "checkpoint_load_mode" not in joined
    assert "resume_success_tolerance" not in joined
    assert "env.scene.num_envs=512" in cmd
    assert "agent.params.config.max_epochs=30" in cmd
    assert "hydra.run.dir=/tmp/run/gen_0/train" in cmd


def test_build_train_cmd_includes_checkpoint_and_tolerance_from_generation_1():
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=4096, max_epochs=200,
        hydra_run_dir=Path("/tmp/run/gen_1/train"), checkpoint=Path("/tmp/ckpt.pth"),
        resume_success_tolerance=0.37, horizon_length=16,
    )
    assert "--checkpoint" in cmd
    idx = cmd.index("--checkpoint")
    assert cmd[idx + 1] == "/tmp/ckpt.pth"
    assert "--checkpoint_load_mode" in cmd
    assert cmd[cmd.index("--checkpoint_load_mode") + 1] == "weights"
    assert "env.termination.resume_success_tolerance=0.37" in cmd


def test_build_train_cmd_scales_minibatch_and_block_size_for_small_num_envs():
    """coevolution/cfg/train/InHandReorientSAPG.yaml's own constraint:
    expl_coef_block_size must divide num_envs, and minibatch_size must
    divide num_envs * horizon_length -- both violated by the yaml's
    defaults (4096, 16384) at a small --num-envs like 512."""
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=512, max_epochs=30,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
    )
    assert "agent.params.config.expl_coef_block_size=512" in cmd
    assert "agent.params.config.minibatch_size=8192" in cmd  # 512 * 16, <= the 16384 cap


def test_build_train_cmd_keeps_default_minibatch_at_the_yaml_scale():
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=4096, max_epochs=200,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
    )
    assert "agent.params.config.minibatch_size=16384" in cmd
    assert "agent.params.config.expl_coef_block_size=4096" in cmd


# --------------------------------------------------------------------------
# compute_train_tail_fitness
# --------------------------------------------------------------------------


def _window(steps, elapsed_s, tol, rows):
    return {"steps": steps, "elapsed_s": elapsed_s, "success_tolerance": tol, "designs": rows}


def test_train_tail_fitness_uses_only_the_last_fraction_of_windows():
    windows = [
        _window(100 * (i + 1), 2.0 * (i + 1), 0.4, {
            "0": {"source": "a", "episodes": 10, "graded_fitness": float(i), "envs_per_design": 8},
        })
        for i in range(10)
    ]
    fit = drv.compute_train_tail_fitness(windows, tail_frac=0.3, min_episodes=1)
    # ceil(0.3 * 10) = 3 tail windows: i = 7, 8, 9 -> mean fitness (7+8+9)/3 = 8.0
    assert fit["a"].fitness == pytest.approx(8.0)
    assert fit["a"].episodes == 30


def test_train_tail_fitness_weights_by_episode_count():
    windows = [
        _window(1, 1.0, 0.4, {"0": {"source": "a", "episodes": 90, "graded_fitness": 1.0, "envs_per_design": 8}}),
        _window(2, 2.0, 0.4, {"0": {"source": "a", "episodes": 10, "graded_fitness": 10.0, "envs_per_design": 8}}),
    ]
    fit = drv.compute_train_tail_fitness(windows, tail_frac=1.0, min_episodes=1)
    # (90*1.0 + 10*10.0) / 100 = 1.9, not the unweighted mean 5.5
    assert fit["a"].fitness == pytest.approx(1.9)


def test_train_tail_fitness_flags_low_confidence_below_the_episode_threshold():
    windows = [_window(1, 1.0, 0.4, {"0": {"source": "a", "episodes": 3, "graded_fitness": 1.0, "envs_per_design": 8}})]
    fit = drv.compute_train_tail_fitness(windows, tail_frac=1.0, min_episodes=10)
    assert fit["a"].low_confidence is True


def test_train_tail_fitness_reports_zero_episode_designs_as_low_confidence_zero():
    windows = [_window(1, 1.0, 0.4, {
        "0": {"source": "a", "episodes": 5, "graded_fitness": 1.0, "envs_per_design": 8},
        "1": {"source": "b", "episodes": 0, "graded_fitness": 0.0, "envs_per_design": 8},
    })]
    fit = drv.compute_train_tail_fitness(windows, tail_frac=1.0, min_episodes=1)
    assert fit["b"].fitness == 0.0
    assert fit["b"].episodes == 0
    assert fit["b"].low_confidence is True


def test_train_tail_fitness_of_empty_windows_is_empty():
    assert drv.compute_train_tail_fitness([], tail_frac=0.3, min_episodes=1) == {}


# --------------------------------------------------------------------------
# find_last_checkpoint
# --------------------------------------------------------------------------


def test_find_last_checkpoint_prefers_last_model_pth(tmp_path):
    """rl_games' vendored agent writes `<experiment_dir>/last/model.pth`
    every 3 epochs, always in place -- exactly what the plan's own
    `--checkpoint <prev gen last/model.pth>` names; prefer it over the
    epoch-tagged nn/ files even when both exist."""
    (tmp_path / "last").mkdir()
    (tmp_path / "last" / "model.pth").write_text("x")
    nn = tmp_path / "nn"
    nn.mkdir()
    (nn / "last_run_ep_30_rew__2.0_.pth").write_text("x")
    got = drv.find_last_checkpoint(tmp_path)
    assert got == tmp_path / "last" / "model.pth"


def test_find_last_checkpoint_picks_the_highest_epoch(tmp_path):
    nn = tmp_path / "nn"
    nn.mkdir()
    (nn / "last_run_ep_3_rew__1.0_.pth").write_text("x")
    (nn / "last_run_ep_30_rew__2.0_.pth").write_text("x")
    (nn / "last_run_ep_9_rew__9.0_.pth").write_text("x")
    got = drv.find_last_checkpoint(tmp_path)
    assert got.name == "last_run_ep_30_rew__2.0_.pth"


def test_find_last_checkpoint_falls_back_to_the_best_checkpoint(tmp_path):
    nn = tmp_path / "nn"
    nn.mkdir()
    (nn / "0_inhand_reorient_sapg.pth").write_text("x")
    got = drv.find_last_checkpoint(tmp_path)
    assert got.name == "0_inhand_reorient_sapg.pth"


def test_find_last_checkpoint_returns_none_when_nothing_was_saved(tmp_path):
    assert drv.find_last_checkpoint(tmp_path) is None


# --------------------------------------------------------------------------
# state.json save/load + resume contract
# --------------------------------------------------------------------------


def test_save_state_then_load_state_round_trips(tmp_path):
    a = _archive_with_one_elite(drv.resolve_variant("G_V1"), np.random.default_rng(0))
    rng = np.random.default_rng(5)
    minter = drv.IdMinter(7)
    path = tmp_path / "state.json"
    config = {"variant": "G_V1", "seed": 5}
    drv.save_state(
        path, archive=a, driver_rng=rng, generation_completed=2, last_checkpoint="/tmp/ckpt.pth",
        prev_tolerance=0.33, minter=minter, config=config,
    )
    doc = drv.load_state(path)
    assert doc["generation_completed"] == 2
    assert doc["last_checkpoint"] == "/tmp/ckpt.pth"
    assert doc["prev_tolerance"] == pytest.approx(0.33)
    assert doc["next_uid"] == 7
    assert doc["config"] == config
    assert doc["config_hash"] == drv._config_hash(config)
    restored_archive = arch.Archive.from_dict(doc["archive"])
    assert restored_archive.coverage() == a.coverage()


def test_load_state_rejects_unknown_schema(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"schema": "not_it"}')
    with pytest.raises(ValueError):
        drv.load_state(path)


def test_resumed_driver_rng_continues_the_exact_same_stream():
    """The resume contract main() relies on: saving and restoring
    `driver_rng.bit_generator.state` must continue the identical random
    stream, not just produce *some* valid state."""
    rng = np.random.default_rng(42)
    state = rng.bit_generator.state
    straight_through = [rng.integers(0, 1_000_000) for _ in range(5)]

    resumed_rng = np.random.default_rng(0)  # deliberately different seed
    resumed_rng.bit_generator.state = state
    resumed = [resumed_rng.integers(0, 1_000_000) for _ in range(5)]
    assert straight_through == resumed
