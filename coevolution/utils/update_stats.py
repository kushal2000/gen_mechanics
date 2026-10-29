"""Per-update PPO diagnostics: where the gradient and the policy change go, by part of the network.

Opt in with ``GENMECH_UPDATE_STATS=<path.jsonl>``. Nothing changes otherwise. Written to test whether a shared per-joint
action head (coevolution/networks/joint_transformer.py: every hand joint's mean from one Linear on its own token)
amplifies each update relative to an MLP, whose output layer has a separate row per action.

Every minibatch step records, then averages over the epoch (one JSON line per epoch, also to the run's writer):

  grad/<group>        gradient norm of each parameter group, unscaled (the AMP scale divided out), BEFORE clipping
  grad/total, clip_frac   the global norm rl_games clips at ``grad_norm``, and how often it does
  step/<group>        how far each group's parameters moved in the step (L2 of the parameter delta)
  kl/hand, kl/wrist   KL(old || new) per action dimension, mean over the batch and over that kind's dimensions;
                      kl/*_mean is the mean-shift part ((mu1 - mu0)^2 / 2 sigma1^2), the rest is the sigma change
  kl/total            the summed KL rl_games' adaptive schedule sees
  head/cancel         the joint transformer only: each hand joint's own contribution g_j to the SHARED action head's
                      gradient (from the head's input tokens and the gradient at its outputs), then
                      ||sum_j g_j|| / sum_j ||g_j||: 1 = every joint pushes the head the same way, near 0 = they cancel
  head/token_cos      the joint transformer only: mean cosine between different joints' tokens entering the head
                      (1 = collapsed: the head cannot tell joints apart). Needs compile_net off (hooks).

Groups: ``hand_head`` (the transformer's shared mu_head; the MLP's mu rows for hand actions), ``wrist_head`` (the
transformer's arm_head; the MLP's mu rows for the wrist), ``value_head``, ``sigma``, ``trunk`` (everything else). The
first ``n_arm`` action dimensions are the wrist (policy order: arm, then hand).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
from rl_games.common.algo_observer import AlgoObserver


def _group(name: str) -> str:
    if ".mu_head." in name:
        return "hand_head"
    if ".arm_head." in name:
        return "wrist_head"
    if ".value_head." in name or ".value." in name:
        return "value_head"
    if name.endswith(".sigma") or ".sigma." in name:
        return "sigma"
    if ".mu." in name:
        return "mlp_mu"                # split by rows below
    return "trunk"


class UpdateStatsObserver(AlgoObserver):
    def __init__(self, path: str, n_arm: int = 6):
        super().__init__()
        self.path, self.n_arm = path, int(n_arm)
        self.acc: dict[str, list[float]] = defaultdict(list)

    def after_init(self, algo):
        from rl_games.algos_torch import torch_ext
        self.algo = algo
        self.params = [(n, p) for n, p in algo.model.named_parameters() if p.requires_grad]
        orig_step = algo.trancate_gradients_and_step
        orig_kl = torch_ext.policy_kl
        obs = self

        def step_with_stats():
            with torch.no_grad():
                obs._grad_stats()
                before = [p.detach().clone() for _, p in obs.params]
            out = orig_step()
            with torch.no_grad():
                obs._step_stats(before)
            return out

        def kl_with_stats(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce=True):
            with torch.no_grad():
                mean_part = (p1_mu - p0_mu) ** 2 / (2.0 * (p1_sigma ** 2 + 1e-5))
                full = (torch.log(p1_sigma / p0_sigma + 1e-5)
                        + (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5)) - 0.5)
                a = obs.n_arm
                for kind, sl in (("wrist", slice(0, a)), ("hand", slice(a, None))):
                    obs.acc[f"kl/{kind}"].append(float(full[:, sl].mean()))
                    obs.acc[f"kl/{kind}_mean"].append(float(mean_part[:, sl].mean()))
                obs.acc["kl/total"].append(float(full.sum(-1).mean()))
            return orig_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, reduce)

        algo.trancate_gradients_and_step = step_with_stats
        torch_ext.policy_kl = kl_with_stats
        self._hook_shared_head(algo)
        print(f"[update_stats] logging per-update gradient / step / KL stats to {self.path}", flush=True)

    def _hook_shared_head(self, algo) -> None:
        net = getattr(algo.model, "a2c_network", None)
        head = getattr(net, "mu_head", None)
        if head is None:
            return
        first = head if isinstance(head, torch.nn.Linear) else next(m for m in head if isinstance(m, torch.nn.Linear))
        if not isinstance(head, torch.nn.Linear):
            return                         # a per-joint MLP head: g_j is not a single product; skip
        obs = self
        store = {}

        def fwd(_m, inp, _out):
            if torch.is_grad_enabled():
                store["x"] = inp[0].detach()               # (B, J, d): each joint's token entering the head

        def bwd(_m, _gin, gout):
            x = store.pop("x", None)
            if x is None:
                return
            with torch.no_grad():
                g = gout[0].detach().float()                # (B, J, 1): dLoss / d mu_joint
                xf = x.float()
                gj = torch.einsum("bjo,bjd->jd", g, xf)     # each joint's contribution to the weight gradient
                gj = torch.cat([gj, g.sum(0)], dim=1)       # ... and to the bias
                obs.acc["head/cancel"].append(float(gj.sum(0).norm() / gj.norm(dim=1).sum().clamp(min=1e-12)))
                xs = xf[: min(4096, xf.shape[0])]
                xn = xs / xs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                c = torch.einsum("bjd,bkd->bjk", xn, xn)
                J = c.shape[1]
                obs.acc["head/token_cos"].append(float((c.sum((1, 2)) - J).mean() / (J * (J - 1))))

        first.register_forward_hook(fwd)
        first.register_full_backward_hook(bwd)
        print("[update_stats] shared action head hooked: head/cancel, head/token_cos", flush=True)

    def _split(self, name: str, t: torch.Tensor) -> dict[str, torch.Tensor]:
        g = _group(name)
        if g != "mlp_mu":
            return {g: t}
        a = self.n_arm                     # the MLP's output layer: rows are actions (arm first)
        return {"wrist_head": t[:a], "hand_head": t[a:]}

    def _grad_stats(self) -> None:
        scale = float(self.algo.scaler.get_scale()) if self.algo.scaler.is_enabled() else 1.0
        sq: dict[str, float] = defaultdict(float)
        finite = True
        for n, p in self.params:
            if p.grad is None:
                continue
            for g, t in self._split(n, p.grad.detach().float() / scale).items():
                v = float((t * t).sum())
                finite &= v == v and v != float("inf")
                sq[g] += v
        total = sum(sq.values()) ** 0.5
        for g, v in sq.items():
            self.acc[f"grad/{g}"].append(v ** 0.5)
        self.acc["grad/total"].append(total)
        self.acc["clip_frac"].append(float(total > float(self.algo.grad_norm)))
        self.acc["nonfinite_frac"].append(float(not finite))

    def _step_stats(self, before) -> None:
        sq: dict[str, float] = defaultdict(float)
        for (n, p), b in zip(self.params, before):
            for g, t in self._split(n, (p.detach() - b).float()).items():
                sq[g] += float((t * t).sum())
        for g, v in sq.items():
            self.acc[f"step/{g}"].append(v ** 0.5)
        self.acc["lr"].append(float(self.algo.last_lr))

    def after_print_stats(self, frame, epoch_num, total_time):
        if not self.acc:
            return
        row = {"epoch": int(epoch_num), "frame": int(frame),
               **{k: sum(v) / len(v) for k, v in self.acc.items()}}
        self.acc.clear()
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")
        writer = getattr(self.algo, "writer", None)
        if writer is not None:
            for k, v in row.items():
                if k not in ("epoch", "frame"):
                    writer.add_scalar(f"update_stats/{k}", v, frame)


def maybe_observer():
    path = os.environ.get("GENMECH_UPDATE_STATS", "")
    return UpdateStatsObserver(path) if path else None


__all__ = ["UpdateStatsObserver", "maybe_observer"]
