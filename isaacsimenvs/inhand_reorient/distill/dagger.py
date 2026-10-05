"""Kit-free core of the GET-Zero-style distillation (``run.py`` drives it).

GET-Zero (A. Patel and S. Song, 2024), Sec. III-IV: one PPO expert per
embodiment (proprioception in, delta joint targets out), embodiments whose
expert does not complete a full turn dropped, then one embodiment-aware
transformer trained by behaviour cloning with an L2 loss on the experts'
actions (plus a self-modeling head supervised with each joint's forward-
kinematics pose), evaluated zero-shot on held-out embodiments.

Adaptations here:

- Student: the team's joint-token transformer (``token_policy.py``: one
  token per articulation column with its link box, limits and enabled flag,
  masked attention, valid-token normalisation); no graph attention bias.
  Each token already carries its link's palm-frame box at the current q, so
  GET-Zero's self-modeling target is an input and that head is left out.
- Data: DAgger rather than offline BC. Rollouts start with the experts
  (beta = 1, GET-Zero's demonstrations); then each env takes the student's
  action with probability 1 - beta per step (``beta_at``,
  ``mix_actions``), and every visited state is labelled by its own design's
  expert (``ExpertBank``). All labels go to one aggregated dataset
  (``AggregatedDataset``, a ring buffer once full).
- Loss: MSE between the student's action means and the experts'
  deterministic, clipped means, averaged over each sample's real joints
  (``masked_action_mse``); ghost columns carry no loss.
- Evaluation: deterministic actions, every env's first episode after a full
  reset (``FirstEpisodeTracker``), so episodes of every length count once.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from torch import nn

__all__ = [
    "AggregatedDataset", "ExpertBank", "FirstEpisodeTracker", "MeanPolicy", "add_sums", "append_jsonl", "beta_at",
    "build_rlg_model", "checkpoint_weights", "load_checkpoint", "load_expert", "masked_action_mse", "mix_actions",
    "per_design_mean", "resolve_experts", "save_checkpoint", "seed_everything", "summarise", "valid_from_tokens",
]

TWO_PI = 2.0 * math.pi


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA): the student's initial
    weights, the action mix and the minibatch draws."""
    import random

    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))


# --------------------------------------------------------------------------
# DAgger schedule and action mix
# --------------------------------------------------------------------------


def beta_at(it: int, warmup_iters: int, decay_iters: int, beta_min: float = 0.0) -> float:
    """Probability that an env acts with its expert at iteration ``it``: 1
    for ``warmup_iters`` iterations (expert rollouts), then linear to
    ``beta_min`` over ``decay_iters`` iterations, then ``beta_min``."""
    if it < warmup_iters:
        return 1.0
    if decay_iters <= 0:
        return float(beta_min)
    frac = min(1.0, (it - warmup_iters) / float(decay_iters))
    return float(1.0 - frac * (1.0 - beta_min))


def mix_actions(expert: torch.Tensor, student: torch.Tensor, beta: float, generator=None):
    """Per env and step: the expert's action with probability ``beta``,
    else the student's. Returns ``(actions, used_expert)``."""
    n = expert.shape[0]
    if beta >= 1.0:
        return expert, torch.ones(n, dtype=torch.bool, device=expert.device)
    if beta <= 0.0:
        return student, torch.zeros(n, dtype=torch.bool, device=expert.device)
    use = torch.rand(n, device=expert.device, generator=generator) < beta
    return torch.where(use.unsqueeze(-1), expert, student), use


# --------------------------------------------------------------------------
# Loss and dataset
# --------------------------------------------------------------------------


def valid_from_tokens(obs: torch.Tensor, n_tokens: int, token_dim: int, enabled_col: int) -> torch.Tensor:
    """``(n, n_tokens)`` bool: the RAW enabled column of each token (> 0.5)."""
    return obs[:, : n_tokens * token_dim].reshape(obs.shape[0], n_tokens, token_dim)[..., enabled_col] > 0.5


def masked_action_mse(mu: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    """Squared error over each sample's real joints, averaged per sample and
    then over the batch. Returns ``(loss, per_sample)``."""
    v = valid.to(mu.dtype)
    per = ((mu - target) ** 2 * v).sum(dim=-1) / v.sum(dim=-1).clamp(min=1.0)
    return per.mean(), per


def per_design_mean(values: torch.Tensor, design: torch.Tensor, n_designs: int):
    """Mean of ``values`` per design index, and the counts."""
    s = torch.zeros(n_designs, device=values.device, dtype=values.dtype).index_add_(0, design, values)
    c = torch.zeros(n_designs, device=values.device, dtype=values.dtype).index_add_(
        0, design, torch.ones_like(values))
    return s / c.clamp(min=1.0), c


class AggregatedDataset:
    """DAgger's aggregated dataset of (student observation, expert label,
    design), a ring buffer once ``capacity`` rows are stored."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device, obs_dtype=torch.float32):
        self.capacity = int(capacity)
        self.obs = torch.zeros(self.capacity, obs_dim, dtype=obs_dtype, device=device)
        self.target = torch.zeros(self.capacity, act_dim, device=device)
        self.design = torch.zeros(self.capacity, dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0
        self.total = 0

    def __len__(self) -> int:
        return self.size

    def add(self, obs: torch.Tensor, target: torch.Tensor, design: torch.Tensor) -> None:
        n = obs.shape[0]
        self.total += n
        if n > self.capacity:
            obs, target, design = obs[-self.capacity:], target[-self.capacity:], design[-self.capacity:]
            n = self.capacity
        idx = (self.ptr + torch.arange(n, device=self.obs.device)) % self.capacity
        self.obs[idx] = obs.to(self.obs.dtype)
        self.target[idx] = target.to(self.target.dtype)
        self.design[idx] = design.long()
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch: int, generator=None):
        if self.size == 0:
            raise RuntimeError("AggregatedDataset.sample on an empty dataset")
        idx = torch.randint(0, self.size, (int(batch),), device=self.obs.device, generator=generator)
        return self.obs[idx].float(), self.target[idx], self.design[idx]


# --------------------------------------------------------------------------
# Experts (frozen rl_games models)
# --------------------------------------------------------------------------


def build_rlg_model(params: dict, obs_dim: int, actions_num: int):
    """The rl_games model (``params['model']``, ``params['network']``) as
    the rl_games agent builds it, with the config's normaliser switches."""
    from rl_games.algos_torch import model_builder

    cfg = params.get("config", {})
    model = model_builder.ModelBuilder().load(params)
    return model.build({
        "actions_num": int(actions_num), "input_shape": (int(obs_dim),), "num_seqs": 1, "value_size": 1,
        "normalize_value": bool(cfg.get("normalize_value", False)),
        "normalize_input": bool(cfg.get("normalize_input", False)),
    })


def checkpoint_weights(path) -> dict:
    """The weights dict of an rl_games checkpoint (``{rank: weights}`` as
    the vendored rl_games saves it, or a plain weights dict)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and 0 in ckpt:
        ckpt = ckpt[0]
    return ckpt


class MeanPolicy(nn.Module):
    """Deterministic action of an rl_games continuous model as its player
    computes it: observations clipped (the env wrapper's
    ``clip_observations``), the model's own input normaliser, the network's
    mean, clipped to the action bound."""

    def __init__(self, model: nn.Module, obs_clip: float | None = None, action_clip: float = 1.0):
        super().__init__()
        self.model = model
        self.obs_clip = None if obs_clip is None else float(obs_clip)
        self.action_clip = float(action_clip)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_clip is not None:
            obs = obs.clamp(-self.obs_clip, self.obs_clip)
        mu = self.model.a2c_network({"obs": self.model.norm_obs(obs)})[0]
        return mu.clamp(-self.action_clip, self.action_clip)


def load_expert(params: dict, path, obs_dim: int, actions_num: int, device="cpu") -> MeanPolicy:
    """A frozen expert in eval mode (its input normaliser does not update)."""
    model = build_rlg_model(params, obs_dim, actions_num)
    model.load_state_dict(checkpoint_weights(path)["model"])
    clip = params.get("env", {}).get("clip_observations")
    pol = MeanPolicy(model, obs_clip=clip).to(device).eval()
    for p in pol.parameters():
        p.requires_grad_(False)
    return pol


def resolve_experts(sources: list[str], mapping: dict) -> list:
    """Each population design's expert checkpoint (None if it has none)."""
    return [mapping.get(s) for s in sources]


class ExpertBank:
    """Routes every env to its own design's expert. ``expert_of_design[d]``
    indexes ``experts`` (None: that design's envs get zero actions)."""

    def __init__(self, experts: list, design_of_env: torch.Tensor, expert_of_design: list):
        self.experts = list(experts)
        self.design_of_env = design_of_env.long()
        self.groups = []
        for k in range(len(self.experts)):
            designs = [d for d, e in enumerate(expert_of_design) if e == k]
            rows = torch.nonzero(torch.isin(self.design_of_env, torch.tensor(designs, device=design_of_env.device,
                                                                            dtype=torch.long)),
                                 as_tuple=False).squeeze(-1)
            if rows.numel():
                self.groups.append((k, rows))
        self.missing = [d for d, e in enumerate(expert_of_design) if e is None]

    @torch.no_grad()
    def act(self, teacher_obs: torch.Tensor, act_dim: int) -> torch.Tensor:
        out = torch.zeros(teacher_obs.shape[0], act_dim, device=teacher_obs.device)
        for k, rows in self.groups:
            out[rows] = self.experts[k](teacher_obs[rows]).to(out.dtype)
        return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


class FirstEpisodeTracker:
    """The first episode of every env after a full reset: holding time and
    rotation about the axis, banked when the env is first done."""

    def __init__(self, design_of_env: torch.Tensor, n_designs: int):
        self.design = design_of_env.long()
        self.n_designs = int(n_designs)
        n = self.design.shape[0]
        self.done = torch.zeros(n, dtype=torch.bool, device=self.design.device)
        self.ttt = torch.zeros(n, device=self.design.device)
        self.rot = torch.zeros(n, device=self.design.device)

    def update(self, done: torch.Tensor, ttt_s: torch.Tensor, rotation_rad: torch.Tensor) -> None:
        new = done.bool() & ~self.done
        self.ttt = torch.where(new, ttt_s.float(), self.ttt)
        self.rot = torch.where(new, rotation_rad.float(), self.rot)
        self.done |= new

    @property
    def all_done(self) -> bool:
        return bool(self.done.all())

    def sums(self, episode_max_s: float) -> dict:
        """Per-design sums over the envs that finished: episodes, holding
        time, rotation (clamped at 0 per episode, as ``design_scoring``),
        signed rotation, time-outs."""
        d = self.design[self.done]
        z = lambda: torch.zeros(self.n_designs, device=self.design.device)
        ttt, rot = self.ttt[self.done], self.rot[self.done]
        return {
            "episodes": z().index_add_(0, d, torch.ones_like(ttt)),
            "ttt_sum": z().index_add_(0, d, ttt),
            "rot_sum": z().index_add_(0, d, rot.clamp(min=0.0)),
            "rot_signed_sum": z().index_add_(0, d, rot),
            "timeouts": z().index_add_(0, d, (ttt >= episode_max_s - 1e-3).float()),
        }


def summarise(sums: dict, sources: list[str]) -> dict:
    """Per design: episodes, mean holding time (s), mean rotations per
    episode (clamped and signed), rad/s while holding, time-out share."""
    out = {}
    for i, s in enumerate(sources):
        n = float(sums["episodes"][i])
        if n <= 0:
            out[s] = {"episodes": 0}
            continue
        ttt = float(sums["ttt_sum"][i])
        out[s] = {
            "episodes": int(n),
            "ttt_mean_s": ttt / n,
            "rotations_mean": float(sums["rot_sum"][i]) / n / TWO_PI,
            "rotations_signed_mean": float(sums["rot_signed_sum"][i]) / n / TWO_PI,
            "rad_per_s": float(sums["rot_sum"][i]) / max(ttt, 1e-6),
            "timeout_frac": float(sums["timeouts"][i]) / n,
        }
    return out


def add_sums(a: dict | None, b: dict) -> dict:
    return dict(b) if a is None else {k: a[k] + b[k] for k in b}


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


def save_checkpoint(path, model: nn.Module, optimizer=None, state: dict | None = None) -> None:
    """``{0: weights}`` as the vendored rl_games saves it (its player and
    ``--checkpoint_load_mode weights`` read ``model``), plus the
    distillation state under ``distill``; written atomically."""
    state = dict(state or {})
    payload = {0: {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(state.get("iteration", 0)),
        "frame": int(state.get("samples", 0)),
        "distill": state,
    }}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(tmp, path)


def load_checkpoint(path, model: nn.Module, optimizer=None) -> dict:
    """Restore the model (and optimizer); return the distillation state."""
    w = checkpoint_weights(path)
    model.load_state_dict(w["model"])
    if optimizer is not None and w.get("optimizer") is not None:
        optimizer.load_state_dict(w["optimizer"])
    return dict(w.get("distill") or {})


def append_jsonl(path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
