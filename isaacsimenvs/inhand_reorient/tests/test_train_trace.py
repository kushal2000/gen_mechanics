"""CPU tests for ``tools/train_trace.py``: learning rate, KL and entropy from
rl_games' tensorboard events, and the policy log-std from each periodic
checkpoint, into one small JSON per run."""

from __future__ import annotations

import json

import torch


def _fake_run(tmp_path, sigma_by_epoch, n_scalar=30):
    from tensorboardX import SummaryWriter

    exp = tmp_path / "0_x"
    (exp / "nn").mkdir(parents=True)
    w = SummaryWriter(str(exp / "summaries"))
    for i in range(n_scalar):
        w.add_scalar("info/last_lr", 1e-4 * (1 + i), i * 100)
        w.add_scalar("info/kl", 0.01, i * 100)
        w.add_scalar("losses/entropy", 3.0 - 0.1 * i, i * 100)
        w.add_scalar("other/ignored", 1.0, i * 100)
    w.close()
    for ep, sigma in sigma_by_epoch.items():
        torch.save({"model": {"a2c_network.sigma": sigma, "a2c_network.w": torch.ones(3)}, "epoch": ep,
                    "frame": ep * 1000}, exp / "nn" / f"last_0_x_ep_{ep}_rew_1.5.pth")
    torch.save({"model": {"a2c_network.sigma": sigma}, "epoch": 999, "frame": 1}, exp / "nn" / "0_x.pth")  # best: skipped
    return tmp_path


def test_scalars_and_logstd_of_real_columns(tmp_path):
    from isaacsimenvs.inhand_reorient.tools import train_trace as tt

    s200 = torch.zeros(8)
    s200[:3] = torch.tensor([-0.5, -1.0, -1.5])
    s400 = torch.zeros(8)
    s400[:3] = torch.tensor([-1.0, -2.0, -3.0])
    run = _fake_run(tmp_path, {200: s200, 400: s400})
    doc = tt.trace(run, every=10)
    lr = doc["scalars"]["info/last_lr"]
    assert lr[0][0] == 0 and abs(lr[0][1] - 1e-4) < 1e-9 and len(lr) == 3 and abs(lr[-1][1] - 2.1e-3) < 1e-8
    assert "other/ignored" not in doc["scalars"]
    eps = [c["epoch"] for c in doc["checkpoints"]]
    assert eps == [200, 400]  # sorted by epoch, the best-model file skipped
    c = doc["checkpoints"][1]
    assert c["frame"] == 400000 and c["cols"] == [0, 1, 2]
    assert abs(c["rows"][0]["mean"] + 2.0) < 1e-6 and c["rows"][0]["min"] == -3.0 and c["rows"][0]["max"] == -1.0


def test_explicit_columns_and_sapg_rows(tmp_path):
    from isaacsimenvs.inhand_reorient.tools import train_trace as tt

    s = torch.zeros(6, 8)
    s[:, :4] = -torch.arange(1, 7, dtype=torch.float32).unsqueeze(1)
    run = _fake_run(tmp_path, {200: s})
    doc = tt.trace(run, columns=[0, 1])
    rows = doc["checkpoints"][0]["rows"]
    assert len(rows) == 6 and rows[5]["mean"] == -6.0
    assert doc["checkpoints"][0]["values"][5] == [-6.0] * 4 + [0.0] * 4  # every column, for other column sets
    assert doc["checkpoints"][0]["cols"] == [0, 1]


def test_main_writes_json_and_reuses_checkpoints(tmp_path, monkeypatch):
    from isaacsimenvs.inhand_reorient.tools import train_trace as tt

    s = torch.zeros(4)
    s[0] = -0.3
    run = _fake_run(tmp_path, {200: s})
    out = tmp_path / "trace.json"
    assert tt.main([str(run), str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["checkpoints"][0]["epoch"] == 200
    calls = []
    monkeypatch.setattr(tt, "checkpoint_logstd", lambda *a, **k: calls.append(a) or {})
    assert tt.main([str(run), str(out)]) == 0  # already traced: not reloaded
    assert calls == []
