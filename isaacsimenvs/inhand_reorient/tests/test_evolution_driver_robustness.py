"""CPU tests for the evolution driver's robustness layer (I41): checkpoint
health checks and sigma handling on carry, the retry-once policy for a
failed `train.py`, the clean stop that leaves state.json at the last good
generation, and the per-generation training-health log.

`run_training_subprocess` is replaced by a fake that writes what a real
generation leaves behind (an rl_games checkpoint, design-scoring windows),
so nothing here boots Kit.

    env -u PYTHONPATH .venv_isaacsim/bin/python3 -m pytest \
        isaacsimenvs/inhand_reorient/tests/test_evolution_driver_robustness.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from isaacsimenvs.inhand_reorient.evolution import archive as arch
from isaacsimenvs.inhand_reorient.evolution import driver as drv
from isaacsimenvs.inhand_reorient.scene import population_file as pf

EXPERIMENT = "0_inhand_reorient_sapg"


def _fake_checkpoint(path: Path, sigma=None, nan_in: str | None = None, scale: float = 1024.0) -> Path:
    model = {
        "a2c_network.sigma": torch.zeros(1, 4) if sigma is None else torch.as_tensor(sigma, dtype=torch.float32),
        "a2c_network.mu.weight": torch.ones(4, 8),
        "running_mean_std.running_mean": torch.zeros(8, dtype=torch.float64),
        "running_mean_std.count": torch.tensor(5.0, dtype=torch.float64),
    }
    if nan_in is not None:
        model[nan_in] = model[nan_in].clone()
        model[nan_in].view(-1)[0] = float("nan")
    ck = {0: {
        "model": model, "epoch": 12, "frame": 100,
        "assymetric_vf_nets": {"critic.weight": torch.ones(2, 2)},
        "scaler": {"scale": scale, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 2000,
                   "_growth_tracker": 7},
    }}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return path


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)[0]


# --------------------------------------------------------------------------
# checkpoint_stats
# --------------------------------------------------------------------------


def test_checkpoint_stats_reports_sigma_std_and_grad_scale(tmp_path):
    p = _fake_checkpoint(tmp_path / "ck.pth", sigma=[[0.0, 1.0, -1.0, 2.0]], scale=512.0)
    st = drv.checkpoint_stats(p)
    assert st["finite"] is True
    assert st["nonfinite_tensors"] == []
    assert st["sigma"]["mean"] == pytest.approx(0.5)
    assert st["sigma"]["min"] == pytest.approx(-1.0)
    assert st["sigma"]["max"] == pytest.approx(2.0)
    stds = [math.exp(v) for v in (0.0, 1.0, -1.0, 2.0)]
    assert st["std"]["mean"] == pytest.approx(sum(stds) / 4, rel=1e-5)
    assert st["std"]["max"] == pytest.approx(math.exp(2.0), rel=1e-5)
    assert st["grad_scale"] == pytest.approx(512.0)
    assert st["epoch"] == 12


def test_checkpoint_stats_names_every_non_finite_tensor(tmp_path):
    p = _fake_checkpoint(tmp_path / "ck.pth", nan_in="a2c_network.mu.weight")
    st = drv.checkpoint_stats(p)
    assert st["finite"] is False
    assert st["nonfinite_tensors"] == ["model/a2c_network.mu.weight"]


# --------------------------------------------------------------------------
# prepare_carry_checkpoint
# --------------------------------------------------------------------------


def test_carry_reset_sets_every_sigma_entry_and_leaves_other_weights(tmp_path):
    src = _fake_checkpoint(tmp_path / "src.pth", sigma=[[3.0, 4.0, 5.0, 6.0]])
    dst = tmp_path / "gen_1" / "carry_checkpoint.pth"
    st = drv.prepare_carry_checkpoint(src, dst, sigma_mode="reset", sigma_reset_value=0.0)
    out = _load(dst)
    assert torch.equal(out["model"]["a2c_network.sigma"], torch.zeros(1, 4))
    assert torch.equal(out["model"]["a2c_network.mu.weight"], torch.ones(4, 8))
    assert st["std"]["mean"] == pytest.approx(1.0)


def test_carry_clamp_bounds_sigma_from_above_only(tmp_path):
    src = _fake_checkpoint(tmp_path / "src.pth", sigma=[[-2.0, 0.0, 1.0, 6.0]])
    dst = tmp_path / "carry.pth"
    drv.prepare_carry_checkpoint(src, dst, sigma_mode="clamp", sigma_clamp_max=0.5)
    assert _load(dst)["model"]["a2c_network.sigma"].tolist() == [[-2.0, 0.0, 0.5, 0.5]]


def test_carry_keep_leaves_sigma_unchanged(tmp_path):
    src = _fake_checkpoint(tmp_path / "src.pth", sigma=[[3.0, 4.0, 5.0, 6.0]])
    dst = tmp_path / "carry.pth"
    drv.prepare_carry_checkpoint(src, dst, sigma_mode="keep")
    assert _load(dst)["model"]["a2c_network.sigma"].tolist() == [[3.0, 4.0, 5.0, 6.0]]


def test_carry_resets_a_collapsed_grad_scaler_by_default(tmp_path):
    """The pilot's gen-6 checkpoint carried scale 0.0 into gen 7 -- a scaler
    that can never step again (rl_games' weights-mode load restores it)."""
    src = _fake_checkpoint(tmp_path / "src.pth", scale=0.0)
    dst = tmp_path / "carry.pth"
    drv.prepare_carry_checkpoint(src, dst, sigma_mode="keep")
    scaler = _load(dst)["scaler"]
    assert scaler["scale"] == drv.FRESH_GRAD_SCALER["scale"]
    assert scaler["_growth_tracker"] == 0

    drv.prepare_carry_checkpoint(src, dst, sigma_mode="keep", reset_grad_scaler=False)
    assert _load(dst)["scaler"]["scale"] == 0.0


def test_carry_caps_the_running_normalizer_counts_when_asked(tmp_path):
    """A normalizer that has seen 1e8 samples barely moves for a new
    generation's designs (a joint slot no earlier design used keeps its ~0
    variance and saturates the new design's inputs at +-5)."""
    src = _fake_checkpoint(tmp_path / "src.pth")
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck[0]["model"]["running_mean_std.count"] = torch.tensor(4e8, dtype=torch.float64)
    ck[0]["model"]["value_mean_std.count"] = torch.tensor(10.0, dtype=torch.float64)
    ck[0]["assymetric_vf_nets"]["model.running_mean_std.count"] = torch.tensor(9e8, dtype=torch.float64)
    torch.save(ck, src)
    dst = tmp_path / "carry.pth"
    drv.prepare_carry_checkpoint(src, dst, sigma_mode="keep", norm_count_cap=1e6)
    out = torch.load(dst, map_location="cpu", weights_only=False)[0]
    assert float(out["model"]["running_mean_std.count"]) == 1e6
    assert float(out["model"]["value_mean_std.count"]) == 10.0  # below the cap: untouched
    assert float(out["assymetric_vf_nets"]["model.running_mean_std.count"]) == 1e6
    assert out["model"]["running_mean_std.count"].dtype == torch.float64

    drv.prepare_carry_checkpoint(src, dst, sigma_mode="keep")  # default: no cap
    out = torch.load(dst, map_location="cpu", weights_only=False)[0]
    assert float(out["model"]["running_mean_std.count"]) == 4e8


def test_carry_refuses_a_non_finite_checkpoint(tmp_path):
    src = _fake_checkpoint(tmp_path / "src.pth", nan_in="a2c_network.sigma")
    with pytest.raises(drv.NonFiniteCheckpoint, match="a2c_network.sigma"):
        drv.prepare_carry_checkpoint(src, tmp_path / "carry.pth", sigma_mode="reset")
    assert not (tmp_path / "carry.pth").exists()


def test_carry_rejects_an_unknown_mode(tmp_path):
    src = _fake_checkpoint(tmp_path / "src.pth")
    with pytest.raises(ValueError):
        drv.prepare_carry_checkpoint(src, tmp_path / "carry.pth", sigma_mode="bogus")


# --------------------------------------------------------------------------
# window / TensorBoard aggregates
# --------------------------------------------------------------------------


def _row(source, episodes, goals_per_episode, fitness, held=1.0, nonfinite_total=0):
    return {"source": source, "episodes": episodes, "goals_per_episode": goals_per_episode,
            "graded_fitness": fitness, "time_held_mean_s": held, "envs_per_design": 4,
            "nonfinite_resets": 0, "nonfinite_resets_total": nonfinite_total}


def test_window_metrics_aggregates_episode_weighted_over_the_tail():
    windows = [
        {"designs": {"0": _row("a", 10, 0.0, 0.1), "1": _row("b", 10, 0.0, 0.1)}},
        {"designs": {"0": _row("a", 30, 1.0, 1.0, held=2.0), "1": _row("b", 10, 3.0, 3.0, held=6.0)}},
    ]
    m = drv.window_metrics(windows, tail_frac=0.5)
    assert m["tail"]["episodes"] == 40
    assert m["tail"]["goals_per_episode"] == pytest.approx((30 * 1.0 + 10 * 3.0) / 40)
    assert m["tail"]["time_held_mean_s"] == pytest.approx((30 * 2.0 + 10 * 6.0) / 40)
    assert m["head"]["goals_per_episode"] == pytest.approx(0.0)


def test_window_metrics_of_no_windows_is_empty():
    assert drv.window_metrics([], tail_frac=0.3) == {}


def test_nonfinite_by_source_takes_each_designs_running_total():
    windows = [
        {"designs": {"0": _row("a", 5, 0, 0, nonfinite_total=1), "1": _row("b", 5, 0, 0)}},
        {"designs": {"0": _row("a", 5, 0, 0, nonfinite_total=4), "1": _row("b", 5, 0, 0)}},
    ]
    assert drv.nonfinite_by_source(windows) == {"a": 4}


def test_training_scalars_reads_head_and_tail_means_from_tensorboard(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    summaries = tmp_path / EXPERIMENT / "summaries"
    w = SummaryWriter(str(summaries))
    for i in range(20):
        w.add_scalar("successes", float(i), i)
        w.add_scalar("rot_error_mean", 2.0 - 0.1 * i, i)
    w.close()
    m = drv.training_scalars(tmp_path, frac=0.1)
    assert m["successes"]["head"] == pytest.approx(0.5)   # mean(0, 1)
    assert m["successes"]["tail"] == pytest.approx(18.5)  # mean(18, 19)
    assert m["rot_error_mean"]["tail"] == pytest.approx(2.0 - 1.85)
    assert m["successes"]["n"] == 20


def test_training_scalars_counts_non_finite_losses(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    w = SummaryWriter(str(tmp_path / EXPERIMENT / "summaries"))
    for i in range(10):
        w.add_scalar("losses/a_loss", float("nan") if i >= 7 else 0.1, i)
    w.close()
    m = drv.training_scalars(tmp_path)
    assert m["losses/a_loss"]["n"] == 10
    assert m["losses/a_loss"]["n_nonfinite"] == 3
    assert m["losses/a_loss"]["tail"] == pytest.approx(0.1)  # finite points only


def test_training_scalars_without_summaries_is_empty(tmp_path):
    assert drv.training_scalars(tmp_path) == {}


# --------------------------------------------------------------------------
# run_generation: retry / stop policy
# --------------------------------------------------------------------------


def _cmd_value(cmd, prefix):
    for tok in cmd:
        if tok.startswith(prefix):
            return tok[len(prefix):]
    return None


class FakeTrainer:
    """Stands in for `run_training_subprocess`. `outcomes` is one entry per
    call: `(returncode, fitness, checkpoint_kind)` with checkpoint_kind in
    {"ok", "nan", None}."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, cmd, *, cwd, extra_env, log_path, score_path, timeout_s, poll_interval_s=3.0):
        rc, fitness, ck_kind = self.outcomes.pop(0)
        train_dir = Path(_cmd_value(cmd, "hydra.run.dir="))
        carried = cmd[cmd.index("--checkpoint") + 1] if "--checkpoint" in cmd else None
        self.calls.append({"cmd": list(cmd), "train_dir": train_dir, "checkpoint": carried})
        log_path.write_text(f"fake train.py, exit {rc}\n")
        if ck_kind is not None:
            _fake_checkpoint(train_dir / EXPERIMENT / "last" / "model.pth", sigma=[[1.0, 1.0, 1.0, 1.0]],
                             nan_in="a2c_network.sigma" if ck_kind == "nan" else None)
        entries = pf.load_population(Path(_cmd_value(cmd, "env.assets.hand_population=")))
        window = {"steps": 100, "elapsed_s": 10.0, "success_tolerance": 0.4, "episodes": 10 * len(entries),
                  "designs": {str(i): _row(e.source, 10, fitness, fitness) for i, e in enumerate(entries)}}
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_path.write_text(json.dumps(window))
        return drv.TrainRunResult(windows=[window], returncode=rc, boot_s=1.0, train_s=2.0, fps=100.0)


def _args(tmp_path, *extra):
    return drv.parse_args([
        "--rules", "evolution", "--generations", "2", "--designs", "3", "--probes", "",
        "--num-envs", "64", "--epochs-per-gen", "6", "--run-dir", str(tmp_path / "run"), *extra,
    ])


def _run_gen(tmp_path, args, archive, generation=0, last_checkpoint=None):
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return drv.run_generation(
        generation, args, run_dir, drv.resolve_rules(args.rules), archive, np.random.default_rng(0),
        drv.IdMinter(), last_checkpoint, 0.4, [],
    )


def test_a_failed_attempt_is_retried_once_and_only_the_retry_is_scored(tmp_path, monkeypatch):
    fake = FakeTrainer([(1, 99.0, "ok"), (0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    args = _args(tmp_path)
    a = arch.Archive()
    ckpt, _tol = _run_gen(tmp_path, args, a)
    assert len(fake.calls) == 2
    assert {e.fitness for e in a.elites()} == {0.5}  # the failed attempt's 99.0 never reached the archive
    gen_dir = Path(args.run_dir) / "gen_0"
    assert (gen_dir / "train_failed_0").is_dir()
    assert (gen_dir / "train_failed_0.log").is_file()
    assert ckpt == gen_dir / "train" / EXPERIMENT / "last" / "model.pth"
    row = json.loads((Path(args.run_dir) / "generations.jsonl").read_text().splitlines()[-1])
    assert [att["returncode"] for att in row["attempts"]] == [1, 0]
    assert row["returncode"] == 0


def test_the_retry_uses_a_different_seed(tmp_path, monkeypatch):
    fake = FakeTrainer([(1, 0.0, None), (0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    _run_gen(tmp_path, _args(tmp_path), arch.Archive())
    seeds = [_cmd_value(c["cmd"], "agent.params.seed=") for c in fake.calls]
    assert seeds[0] != seeds[1]


def test_two_failed_attempts_raise_and_leave_the_archive_untouched(tmp_path, monkeypatch):
    fake = FakeTrainer([(1, 99.0, "ok"), (137, 99.0, None)])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    a = arch.Archive()
    with pytest.raises(drv.GenerationFailed, match="generation 0"):
        _run_gen(tmp_path, _args(tmp_path), a)
    assert a.coverage() == 0
    assert not (Path(tmp_path / "run") / "generations.jsonl").exists()


def test_a_non_finite_trained_checkpoint_counts_as_a_failed_attempt(tmp_path, monkeypatch):
    fake = FakeTrainer([(0, 0.3, "nan"), (0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    a = arch.Archive()
    _run_gen(tmp_path, _args(tmp_path), a)
    assert len(fake.calls) == 2
    assert {e.fitness for e in a.elites()} == {0.5}


def test_a_missing_checkpoint_counts_as_a_failed_attempt(tmp_path, monkeypatch):
    fake = FakeTrainer([(0, 0.3, None), (0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    _run_gen(tmp_path, _args(tmp_path), arch.Archive())
    assert len(fake.calls) == 2


def test_retry_restarts_from_the_same_carried_checkpoint(tmp_path, monkeypatch):
    fake = FakeTrainer([(1, 0.0, None), (0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    prev = _fake_checkpoint(tmp_path / "prev" / "model.pth", sigma=[[5.0, 5.0, 5.0, 5.0]])
    _run_gen(tmp_path, _args(tmp_path, "--sigma-on-carry", "reset"), arch.Archive(), generation=1,
             last_checkpoint=prev)
    carried = [c["checkpoint"] for c in fake.calls]
    assert carried[0] == carried[1]
    assert _load(Path(carried[0]))["model"]["a2c_network.sigma"].tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert _load(prev)["model"]["a2c_network.sigma"].tolist() == [[5.0, 5.0, 5.0, 5.0]]  # source untouched


def test_generation_row_logs_sigma_window_and_nonfinite_health(tmp_path, monkeypatch):
    fake = FakeTrainer([(0, 0.5, "ok")])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    prev = _fake_checkpoint(tmp_path / "prev" / "model.pth", sigma=[[5.0, 5.0, 5.0, 5.0]])
    _run_gen(tmp_path, _args(tmp_path, "--sigma-on-carry", "reset"), arch.Archive(), generation=1,
             last_checkpoint=prev)
    row = json.loads((Path(tmp_path / "run") / "generations.jsonl").read_text().splitlines()[-1])
    assert row["sigma_carried_in"]["std"]["mean"] == pytest.approx(1.0)
    assert row["sigma_trained"]["std"]["mean"] == pytest.approx(math.e, rel=1e-5)
    assert row["sigma_trained"]["finite"] is True
    assert row["window_metrics"]["tail"]["goals_per_episode"] == pytest.approx(0.5)
    assert row["nonfinite_by_design"] == {}
    assert "train_metrics" in row
    csv_text = (Path(tmp_path / "run") / "generations.csv").read_text()
    assert "std_mean" in csv_text.splitlines()[0]


def test_main_stops_cleanly_with_state_at_the_last_good_generation(tmp_path, monkeypatch, capsys):
    fake = FakeTrainer([(0, 0.5, "ok"), (1, 0.0, None), (1, 0.0, None)])
    monkeypatch.setattr(drv, "run_training_subprocess", fake)
    run_dir = tmp_path / "run"
    rc = drv.main([
        "--rules", "evolution", "--generations", "3", "--designs", "3", "--probes", "",
        "--num-envs", "64", "--epochs-per-gen", "6", "--run-dir", str(run_dir),
    ])
    assert rc != 0
    state = drv.load_state(run_dir / "state.json")
    assert state["generation_completed"] == 0
    assert state["last_checkpoint"].endswith("gen_0/train/0_inhand_reorient_sapg/last/model.pth")
    assert len((run_dir / "generations.jsonl").read_text().splitlines()) == 1
    assert "generation 1 failed" in capsys.readouterr().out
