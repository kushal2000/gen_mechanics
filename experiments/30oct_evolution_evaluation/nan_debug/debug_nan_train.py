"""coevolution/train.py with a NaN tripwire on the rollout forward (31 Oct NaN crashes).

The crashes: "normal expects all elements of std >= 0.0" in the rollout's distr.sample(). The joint transformer returns
sigma = mu * 0 + sigma (coevolution/networks/joint_transformer.py), so a NaN action mean makes the std NaN; the
optimizer step cannot write NaN into the sigma parameter (GradScaler skips non-finite steps). This wraps
rl_games' A2CBase.get_action_values: at the first failure (or the first non-finite mu) it re-runs the same batch through
the network in fp16 (as the rollout does, rollout_autocast) and in fp32, with a forward hook on every module, and writes
  $NAN_DEBUG_OUT/report.json   obs finiteness, non-finite params/buffers, the largest weights, per-module max |out| and
                               non-finite counts in firing order for fp16 and fp32, the failing rows (envs) and hands
  $NAN_DEBUG_OUT/batch.pt      the failing batch's raw observations (for offline replays)
then re-raises. The decisive fp16/fp32 test uses the network as trained; the per-module probe re-runs it uncompiled on the dense
attention path. Otherwise train.py unchanged:
    python experiments/30oct_evolution_evaluation/nan_debug/debug_nan_train.py <train.py's arguments>
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get("NAN_DEBUG_OUT", str(REPO / "debug_outputs/evolution/nan_debug")))
_done = {"fired": False}


def _forward(algo, processed, fp16: bool, probe: bool):
    """The network's (mu, value) on the batch; returns (mu, value, per-module rows). probe=False: exactly as trained
    (compiled, packed tokens) -- the decisive fp16-vs-fp32 test. probe=True: the uncompiled dense-attention path on the
    same weights, with a forward hook on every module (hooks do not fire inside a compiled forward)."""
    import torch
    model, net = algo.model, algo.model.a2c_network
    rows: list = []
    hooks = []
    if not probe:
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=fp16):
            obs = model.norm_obs(processed) if hasattr(model, "norm_obs") else processed
            out = net({"obs": obs, "is_train": False, "prev_actions": None, "rnn_states": None})
        return out[0], out[2], rows
    saved = getattr(net, "packed_tokens", None)

    def hook(name):
        def h(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point():
                f = t.detach().float()
                bad = int((~torch.isfinite(f)).sum())
                rows.append({"module": name, "dtype": str(t.dtype), "nonfinite": bad,
                             "absmax": float(torch.nan_to_num(f, nan=0.0, posinf=3e38, neginf=-3e38).abs().max())})
        return h

    for name, m in net.named_modules():
        hooks.append(m.register_forward_hook(hook(name or "<net>")))
    try:
        if saved is not None:
            net.packed_tokens = False
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=fp16):
            obs = model.norm_obs(processed) if hasattr(model, "norm_obs") else processed
            out = net.forward({"obs": obs, "is_train": False, "prev_actions": None, "rnn_states": None})
    finally:
        for h in hooks:
            h.remove()
        if saved is not None:
            net.packed_tokens = saved
    mu, value = out[0], out[2]
    return mu, value, rows


def _hands(algo, idx):
    for path in ("vec_env.env.unwrapped", "vec_env.env", "vec_env"):
        o = algo
        try:
            for a in path.split("."):
                o = getattr(o, a)
            hoe = getattr(o, "hand_of_env", None)
            if hoe is not None:
                hs = getattr(o, "hand_set", None)
                names = [s.name for s in hs.specs] if hs is not None else None
                return [{"env": int(i), "hand": int(hoe[i]), "spec": names[int(hoe[i])] if names else None} for i in idx]
        except Exception:
            continue
    return None


def _diagnose(algo, obs, why: str) -> None:
    import torch
    if _done["fired"]:
        return
    _done["fired"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    raw = obs["obs"]
    rep = {"why": why, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": getattr(algo, "epoch_num", None),
           "frame": getattr(algo, "frame", None), "last_lr": getattr(algo, "last_lr", None),
           "obs_shape": list(raw.shape)}
    fin = torch.isfinite(raw).all(dim=-1)
    rep["obs_nonfinite_rows"] = int((~fin).sum())
    rep["obs_nonfinite_cols"] = torch.nonzero(~torch.isfinite(raw).all(dim=0)).flatten().tolist()[:50]
    rep["obs_absmax_finite_rows"] = float(raw[fin].abs().max()) if bool(fin.any()) else None
    model = algo.model
    rep["nonfinite_params"] = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    rep["nonfinite_buffers"] = [n for n, b in model.named_buffers() if b.is_floating_point() and not torch.isfinite(b).all()]
    rep["largest_params"] = sorted(((float(p.detach().abs().max()), n) for n, p in model.named_parameters()), reverse=True)[:10]
    processed = algo._preproc_obs(raw)
    for tag, fp16, probe in (("fp16", True, False), ("fp32", False, False), ("probe_fp16", True, True),
                             ("probe_fp32", False, True)):
        try:
            mu, value, rows = _forward(algo, processed, fp16, probe)
            bad = ~torch.isfinite(mu).all(dim=-1)
            rep[tag] = {"mu_nonfinite_rows": int(bad.sum()), "value_nonfinite": int((~torch.isfinite(value)).sum()),
                        "first_nonfinite_module": next((r for r in rows if r["nonfinite"]), None),
                        "largest_outputs": sorted(rows, key=lambda r: -r["absmax"])[:12]}
            if tag == "fp16":
                idx = torch.nonzero(bad).flatten().tolist()
                rep["fp16"]["bad_rows"] = idx[:64]
                rep["fp16"]["bad_hands"] = _hands(algo, idx[:64])
                rep["fp16"]["bad_rows_obs_finite"] = bool(fin[bad].all()) if bool(bad.any()) else None
            if tag == "probe_fp16":
                rep[tag]["modules_in_order"] = rows
        except Exception as exc:          # keep whatever was collected
            rep[tag] = {"error": repr(exc)}
    torch.save({"obs": raw.detach().cpu()}, OUT / "batch.pt")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(f"[nan_debug] wrote {OUT / 'report.json'}: obs non-finite rows {rep['obs_nonfinite_rows']}; "
          f"fp16 mu non-finite rows {rep.get('fp16', {}).get('mu_nonfinite_rows')}, "
          f"fp32 {rep.get('fp32', {}).get('mu_nonfinite_rows')}; first fp16 non-finite module "
          f"{(rep.get('fp16', {}).get('first_nonfinite_module') or {}).get('module')}", flush=True)


def install() -> None:
    import torch
    from rl_games.common import a2c_common
    orig = a2c_common.A2CBase.get_action_values

    def wrapped(self, obs, rnn_states=None):
        try:
            res = orig(self, obs, rnn_states)
        except RuntimeError as exc:
            if "std" in str(exc):
                _diagnose(self, obs, f"exception: {exc}")
            raise
        mus = res.get("mus")
        if mus is not None and not bool(torch.isfinite(mus).all()):
            _diagnose(self, obs, "non-finite mus (no exception)")
        return res

    a2c_common.A2CBase.get_action_values = wrapped
    print("[nan_debug] tripwire on A2CBase.get_action_values", flush=True)
    _install_step_tripwire(a2c_common)


_ring: list = []
_step_done = {"fired": False}


def _install_step_tripwire(a2c_common) -> None:
    """rl_games' single-GPU trancate_gradients_and_step, inline (unscale_, clip_grad_norm_, scaler.step, update), with a
    record per step (found_inf, the pre-clip norm, the scale before/after, non-finite grads) in a 200-step ring; the first
    step after which any parameter is non-finite writes step_report.json (the ring, the bad parameters, their grads and
    Adam state)."""
    import torch
    from torch import nn
    orig = a2c_common.A2CBase.trancate_gradients_and_step

    def step(self):
        if self.multi_gpu or not self.truncate_grads:
            return orig(self)
        named = [(n, p) for n, p in self.model.named_parameters() if p.grad is not None]
        scale0 = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
        self.scaler.unscale_(self.optimizer)
        found = None
        try:
            found = float(sum(v.sum() for v in self.scaler._found_inf_per_device(self.optimizer).values()))
        except Exception:
            pass
        grads_bad = [n for n, p in named if not bool(torch.isfinite(p.grad).all())]
        norm = float(nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm))
        grads_bad_after_clip = [n for n, p in named if not bool(torch.isfinite(p.grad).all())]
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale1 = float(self.scaler.get_scale()) if self.scaler.is_enabled() else 1.0
        rec = {"epoch": getattr(self, "epoch_num", None), "lr": self.last_lr, "scale_before": scale0, "scale_after": scale1,
               "skipped": scale1 < scale0, "found_inf": found, "pre_clip_norm": norm, "grads_nonfinite": grads_bad[:6],
               "grads_nonfinite_after_clip": grads_bad_after_clip[:6]}
        _ring.append(rec)
        del _ring[:-200]
        if not _step_done["fired"]:
            bad = [n for n, p in self.model.named_parameters() if not bool(torch.isfinite(p).all())]
            if bad:
                _step_done["fired"] = True
                st = {}
                for n, p in self.model.named_parameters():
                    if n in bad[:8]:
                        s = self.optimizer.state.get(p, {})
                        st[n] = {k: {"finite": bool(torch.isfinite(v).all()), "absmax": float(torch.nan_to_num(v.float(), nan=0).abs().max())}
                                 for k, v in s.items() if torch.is_tensor(v) and v.numel() > 1}
                        st[n]["grad_finite"] = bool(torch.isfinite(p.grad).all()) if p.grad is not None else None
                OUT.mkdir(parents=True, exist_ok=True)
                (OUT / "step_report.json").write_text(json.dumps(
                    {"bad_params": bad, "n_bad": len(bad), "this_step": rec, "adam_state": st, "last_steps": _ring[-60:]},
                    indent=1, default=str))
                print(f"[nan_debug] parameters went non-finite at epoch {rec['epoch']}: {len(bad)} tensors; found_inf "
                      f"{found}, pre-clip norm {norm}, scale {scale0} -> {scale1}, skipped {rec['skipped']}; wrote "
                      f"{OUT / 'step_report.json'}", flush=True)
        return None

    a2c_common.A2CBase.trancate_gradients_and_step = step
    print("[nan_debug] tripwire on A2CBase.trancate_gradients_and_step", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "third_party/rl_games"))
    install()
    sys.argv[0] = str(REPO / "coevolution/train.py")
    runpy.run_path(str(REPO / "coevolution/train.py"), run_name="__main__")
