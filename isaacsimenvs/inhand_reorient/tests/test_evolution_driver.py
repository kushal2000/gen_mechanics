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


def test_repo_root_points_at_the_actual_gen_mechanics_checkout():
    """Regression test for a real bug found during the tiny end-to-end
    verification run: REPO_ROOT was off by one parents[] level (pointed
    one directory ABOVE the gen_mechanics checkout), so every generation's
    `timeout ... <train_python> coevolution/train.py` command silently
    failed with exit 127 ("no such file or directory") for BOTH the
    python interpreter and the script path, and the driver degraded
    (correctly, but uselessly) to an all-zero-fitness archive every
    generation without ever raising."""
    assert drv.REPO_ROOT.name == "gen_mechanics"
    assert (drv.REPO_ROOT / "coevolution" / "train.py").is_file()
    assert drv.TRAIN_PY == drv.REPO_ROOT / "coevolution" / "train.py"
    assert drv.TRAIN_PY.is_file()


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
    # the asymmetric-critic block's own hardcoded 16384 must scale down too
    # (train_central_value's num_minibatches = batch_size // this value
    # divides by zero otherwise -- see build_train_cmd's own comment).
    assert "agent.params.config.central_value_config.minibatch_size=8192" in cmd


def test_build_train_cmd_keeps_default_minibatch_at_the_yaml_scale():
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=4096, max_epochs=200,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
    )
    assert "agent.params.config.minibatch_size=16384" in cmd
    assert "agent.params.config.expl_coef_block_size=4096" in cmd


def test_build_train_cmd_defaults_to_the_legacy_profile_and_sapg():
    """In-flight runs (started before the isaaclab_repose profile existed)
    must resume with the spec they started on, although the env's own
    default profile is now isaaclab_repose."""
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=4096, max_epochs=200,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
    )
    assert "env.task_profile=legacy" in cmd
    assert cmd[cmd.index("--agent") + 1] == "rl_games_sapg_cfg_entry_point"
    assert "agent.params.config.expl_coef_block_size=4096" in cmd


def test_build_train_cmd_repose_profile_with_the_ppo_agent_skips_sapg_overrides():
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=4096, max_epochs=200,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
        agent_entry_point=drv.REPOSE_POP_AGENT_ENTRY_POINT, task_profile="isaaclab_repose",
    )
    joined = " ".join(cmd)
    assert "env.task_profile=isaaclab_repose" in cmd
    assert cmd[cmd.index("--agent") + 1] == "rl_games_repose_pop_ppo_cfg_entry_point"
    assert "expl_coef_block_size" not in joined
    assert "central_value_config" not in joined
    assert "agent.params.config.minibatch_size=32768" in cmd  # NVIDIA's minibatch
    small = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=512, max_epochs=30,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
        agent_entry_point=drv.REPOSE_POP_AGENT_ENTRY_POINT, task_profile="isaaclab_repose",
    )
    assert "agent.params.config.minibatch_size=8192" in small  # 512 * 16 < 32768


def test_anyrotate_profile_uses_its_own_ppo_horizon():
    """InHandAnyRotatePopPPO.yaml rolls out 8 steps (AnyRotate Table 5): the
    minibatch and the scoring-window cadence must use 8, not the SAPG 16."""
    assert "anyrotate" in drv.TASK_PROFILES
    assert drv.agent_horizon(drv.ANYROTATE_POP_AGENT_ENTRY_POINT, 16) == 8
    assert drv.agent_horizon("rl_games_sapg_pop_cfg_entry_point", 16) == 16
    cmd = drv.build_train_cmd(
        train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=256, max_epochs=30,
        hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
        agent_entry_point=drv.ANYROTATE_POP_AGENT_ENTRY_POINT, task_profile="anyrotate",
    )
    assert "env.task_profile=anyrotate" in cmd
    assert "agent.params.config.minibatch_size=2048" in cmd  # 256 envs x 8 steps
    assert "expl_coef_block_size" not in " ".join(cmd)
    args = drv.parse_args(["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x",
                           "--task-profile", "anyrotate", "--agent-entry-point", drv.ANYROTATE_POP_AGENT_ENTRY_POINT])
    assert drv._resolved_config(args)["task_profile"] == "anyrotate"
    assert drv.parse_args(["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x"]).task_profile == "legacy"


def test_build_train_cmd_rejects_an_unknown_profile():
    with pytest.raises(ValueError):
        drv.build_train_cmd(
            train_python="python3", population_path=Path("/tmp/pop.json"), num_envs=512, max_epochs=30,
            hydra_run_dir=Path("/tmp/run"), checkpoint=None, resume_success_tolerance=None,
            horizon_length=16, task_profile="isaaclab",
        )


def test_task_profile_flag_defaults_to_legacy_and_keeps_the_old_config_hash():
    base = ["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x"]
    legacy = drv.parse_args(base)
    assert legacy.task_profile == "legacy"
    assert legacy.agent_entry_point == "rl_games_sapg_cfg_entry_point"
    assert "task_profile" not in drv._resolved_config(legacy)
    repose = drv.parse_args(base + ["--task-profile", "isaaclab_repose",
                                    "--agent-entry-point", drv.REPOSE_POP_AGENT_ENTRY_POINT])
    assert drv._resolved_config(repose)["task_profile"] == "isaaclab_repose"
    assert drv._config_hash(drv._resolved_config(repose)) != drv._config_hash(drv._resolved_config(legacy))


def test_repose_agent_entry_points_are_registered():
    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  registers the task

    kwargs = gym.spec(drv.TASK_ID).kwargs
    for key, name in (("rl_games_repose_pop_ppo_cfg_entry_point", "InHandReposeIsaacLabPopPPO.yaml"),
                      ("rl_games_repose_ppo_cfg_entry_point", "InHandReposeIsaacLabPPO.yaml")):
        assert Path(kwargs[key]).name == name and Path(kwargs[key]).is_file()
    # train.py's default --agent stays unregistered for this task, as before.
    assert "rl_games_cfg_entry_point" not in kwargs
    assert Path(kwargs["rl_games_sapg_cfg_entry_point"]).name == "InHandReorientSAPG.yaml"


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


def test_find_last_checkpoint_finds_rl_games_own_nested_experiment_dir(tmp_path):
    """rl_games nests its own experiment_dir one level below the Hydra run
    dir we pass as train_dir (`<train_dir>/<experiment_name>/...`) --
    reproduced against a real training run while investigating the eval
    fitness player mismatch (see the README)."""
    nested = tmp_path / "0_inhand_reorient_sapg" / "last"
    nested.mkdir(parents=True)
    (nested / "model.pth").write_text("x")
    got = drv.find_last_checkpoint(tmp_path)
    assert got == nested / "model.pth"


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


# --------------------------------------------------------------------------
# --grasp-cache (anyrotate stable-grasp cache, generated inside the training launch)
# --------------------------------------------------------------------------

_AR = ["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x", "--task-profile", "anyrotate",
       "--agent-entry-point", drv.ANYROTATE_POP_AGENT_ENTRY_POINT]


def test_grasp_cache_flag_is_off_by_default_and_keeps_the_config_hash():
    off = drv.parse_args(_AR)
    assert off.grasp_cache is False
    assert "grasp_cache" not in drv._resolved_config(off)
    on = drv.parse_args(_AR + ["--grasp-cache"])
    assert on.grasp_cache is True and drv._resolved_config(on)["grasp_cache"] is True
    assert drv._config_hash(drv._resolved_config(on)) != drv._config_hash(drv._resolved_config(off))


def test_grasp_cache_needs_the_anyrotate_profile():
    with pytest.raises(SystemExit):
        drv.parse_args(["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x", "--grasp-cache"])


def test_grasp_cache_overrides_point_the_env_at_the_run_cache_and_generate_missing_designs(tmp_path):
    args = drv.parse_args(_AR + ["--grasp-cache"])
    path = drv.grasp_cache_path(args, tmp_path)
    assert path == tmp_path / "grasp_cache.npz"
    # prune: the saved cache keeps only this generation's designs (every elite is
    # resubmitted each generation, so nothing still needed is dropped).
    assert drv.grasp_cache_overrides(path) == [f"env.anyrotate.grasp_cache={path}",
                                               "env.anyrotate.grasp_cache_generate=true",
                                               "env.anyrotate.grasp_cache_prune=true"]
    custom = drv.parse_args(_AR + ["--grasp-cache", "--grasp-cache-path", str(tmp_path / "shared.npz")])
    assert drv.grasp_cache_path(custom, tmp_path) == tmp_path / "shared.npz"


def test_grasp_report_row_maps_sources_to_design_ids(tmp_path):
    import json

    report = {"cache": "c.npz", "designs": 3, "reused": 1, "generated": 2, "gen_s": 41.5,
              "grasps_per_design": {"arch:a": 1000, "arch:b": 0, "projected:allegro_right": 640},
              "viable": 2, "non_viable": ["arch:b"], "generation": {"rounds": 6, "s_per_design": 20.7}}
    (tmp_path / "grasp_cache_report.json").write_text(json.dumps(report))
    loaded = drv.read_grasp_report(tmp_path)
    row = drv.grasp_report_row(loaded, {"arch:a": "g1-0001", "arch:b": "g1-0002"})
    assert row["gen_s"] == 41.5 and row["generated"] == 2 and row["reused"] == 1 and row["rounds"] == 6
    assert row["viable"] == 2 and row["viable_frac"] == pytest.approx(2 / 3)
    assert row["non_viable"] == ["g1-0002"]
    assert row["grasps_by_design"] == {"g1-0001": 1000, "g1-0002": 0, "projected:allegro_right": 640}
    assert drv.read_grasp_report(tmp_path / "missing") is None
    assert drv.grasp_report_row(None, {}) is None


def test_hora_profile_is_accepted_with_the_grasp_cache():
    args = drv.parse_args(["--variant", "G_V3S", "--generations", "1", "--run-dir", "/tmp/x", "--task-profile", "hora",
                           "--agent-entry-point", drv.ANYROTATE_POP_AGENT_ENTRY_POINT, "--grasp-cache"])
    assert args.task_profile == "hora" and args.grasp_cache
    cmd = drv.build_train_cmd(
        train_python="python", population_path=Path("/tmp/p.json"), num_envs=4096, max_epochs=10,
        hydra_run_dir=Path("/tmp/r"), checkpoint=None, resume_success_tolerance=None, horizon_length=16,
        agent_entry_point=drv.ANYROTATE_POP_AGENT_ENTRY_POINT, task_profile="hora")
    assert "env.task_profile=hora" in cmd
