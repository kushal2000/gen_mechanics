"""rl_games actor-critic with a bounded, state-independent policy log-std.

Registered as ``inhand_actor_critic``: rl_games' stock ``actor_critic``
(``A2CBuilder``), plus two optional keys under ``network.space.continuous``:

    logstd_max: 0.0     # project the log-std parameter to <= this before every forward
    logstd_min: -5.0    # ... and to >= this (omit either to leave that side open)

Why (I41): the population controller's log-std parameter (``a2c_network.
sigma``, ``fixed_sigma: coef_cond``) drifted from 0 to a mean of 5.5 (max
6.75) over seven evolution-pilot generations (geometric-mean std ~230).
Actions are clipped to [-1, 1] before they reach the env, so any std well
above 1 only makes the executed actions bang-bang: it adds no exploration, it removes the reward's grip on
the std (the executed action distribution stops changing as sigma grows),
and a large std shrinks the policy KL per update, which pushes the adaptive
learning rate to its 1e-2 ceiling. Projecting the parameter (rather than
clamping only the forward output) keeps its gradient alive inside the range:
a hard clamp on the output alone would leave a parameter that overshot the
bound with zero gradient, stuck there.

A third optional key, ``ghost_mask_tail: N``, reads the last N observation
values as each action slot's sign (+1 a real joint of this env's hand, -1 a
ghost slot; ``hora.ghost_action_mask`` appends them) and holds every slot
whose (normalised) sign is < 0 at mean 0 and log-std 0 with no gradient.
With several hands on one controller, a slot that is real for one hand and
a ghost for another otherwise feeds the second hand's samples into the
first hand's update as pure noise, and perturbs the PPO ratio; with one
hand, ghost slots are ghosts for every env and nothing changes. A slot that
reads exactly 0 after rl_games' input normalisation is the same for every
env and counts as real.

A network-level key, ``per_design_nets: K`` (under ``network``), holds K
independent actor-critics (each its own trunk, mean head, log-std and
value head) in this one model, and routes each sample to the one its
design selects. The design is read from a K-wide one-hot just before the
ghost-sign tail (``hora.design_id_obs``): after rl_games' input
normalisation each column is positive for its own design's envs and
negative elsewhere, so the argmax still picks the design. rl_games sees one
model, one batch, one optimiser and one adaptive learning rate; only the
weights are separate. Design 0 is this network's own layers, so with K <= 1
the parameters are the stock ones.

``per_design_heads: K`` (under ``network``) is the middle ground: one shared
trunk, and per design its own mean head, log-std and value head (same
routing; shared-trunk MLPs with ``fixed_sigma: fixed`` only).

``per_design_input_norm: K`` (under ``network``) normalises the observation
body (everything before the design one-hot) again with each design's own
running mean and variance (``GroupRunningNorm``, updated in training mode
like rl_games' input normaliser) before the network, shared or per design.
rl_games' input normaliser pools all designs, so a hand whose joints are
ghosts for half the batch sees its own inputs shifted and compressed; the
second, per-design normalisation is affine on top of it and undoes that.

Only the ``nn.Parameter`` log-std of ``fixed_sigma: fixed | coef_cond`` is
bounded; a state-dependent sigma head is left alone. Checkpoints are
interchangeable with the stock ``actor_critic`` (same parameters, same
names), so a stock checkpoint loads here and vice versa.
"""

from __future__ import annotations

import torch
from torch import nn
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder

__all__ = ["NETWORK_NAME", "BoundedSigmaA2CBuilder"]

NETWORK_NAME = "inhand_actor_critic"


class GroupRunningNorm(nn.Module):
    """Running mean and variance per group (``k`` groups of ``dim`` values),
    updated from each training-mode batch (the parallel update rl_games'
    ``RunningMeanStd`` uses); output (x - mean[g]) / sqrt(var[g] + eps),
    clipped to +-clip."""

    def __init__(self, k: int, dim: int, eps: float = 1e-5, clip: float = 5.0):
        super().__init__()
        self.eps, self.clip = float(eps), float(clip)
        self.register_buffer("mean", torch.zeros(k, dim))
        self.register_buffer("var", torch.ones(k, dim))
        self.register_buffer("count", torch.full((k,), 1e-4))

    @torch.no_grad()
    def _update(self, x: torch.Tensor, g: torch.Tensor) -> None:
        k = self.mean.shape[0]
        n = torch.zeros(k, device=x.device).index_add_(0, g, torch.ones(len(g), device=x.device))
        has = n > 0
        nc = n.clamp(min=1).unsqueeze(-1)
        bmean = torch.zeros_like(self.mean).index_add_(0, g, x) / nc
        bvar = torch.zeros_like(self.var).index_add_(0, g, (x - bmean[g]) ** 2) / nc
        tot = (self.count + n).unsqueeze(-1)
        delta = bmean - self.mean
        mean = self.mean + delta * n.unsqueeze(-1) / tot
        m2 = self.var * self.count.unsqueeze(-1) + bvar * n.unsqueeze(-1) + delta ** 2 * (self.count * n).unsqueeze(-1) / tot
        h = has.unsqueeze(-1)
        self.mean.copy_(torch.where(h, mean, self.mean))
        self.var.copy_(torch.where(h, m2 / tot, self.var))
        self.count.copy_(torch.where(has, self.count + n, self.count))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g = g.long()
        if self.training:
            self._update(x, g)
        return ((x - self.mean[g]) / torch.sqrt(self.var[g] + self.eps)).clamp(-self.clip, self.clip)


class _DesignNet(A2CBuilder.Network):
    """One design's actor-critic inside ``per_design_nets``."""

    def own_parameters(self):
        return list(self.parameters())


class BoundedSigmaA2CBuilder(A2CBuilder):
    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            continuous = params.get("space", {}).get("continuous", {}) or {}
            lo = continuous.get("logstd_min")
            hi = continuous.get("logstd_max")
            self.logstd_min = None if lo is None else float(lo)
            self.logstd_max = None if hi is None else float(hi)
            self.ghost_mask_tail = int(continuous.get("ghost_mask_tail") or 0)
            self.per_design_nets = int(params.get("per_design_nets") or 0)
            self.per_design_input_norm = int(params.get("per_design_input_norm") or 0)
            self.input_norm = None
            if self.per_design_input_norm > 0:
                if self.per_design_nets > 1 and self.per_design_nets != self.per_design_input_norm:
                    raise ValueError("per_design_input_norm and per_design_nets read the same one-hot: set both to K")
                n_obs = int(kwargs["input_shape"][0])
                self.input_norm = GroupRunningNorm(
                    self.per_design_input_norm, n_obs - self.ghost_mask_tail - self.per_design_input_norm)
            self.per_design_heads = int(params.get("per_design_heads") or 0)
            self.mu_heads = self.value_heads = self.sigma_heads = None
            if self.per_design_heads > 1:
                if self.separate or self.is_rnn() or self.fixed_sigma != "fixed" or self.per_design_nets > 1:
                    raise NotImplementedError("per_design_heads: shared-trunk MLP, fixed sigma, no per_design_nets")
                k = self.per_design_heads
                self.mu_heads = nn.ModuleList(
                    [nn.Linear(self.mu.in_features, self.mu.out_features) for _ in range(k - 1)])
                self.value_heads = nn.ModuleList(
                    [nn.Linear(self.value.in_features, self.value.out_features) for _ in range(k - 1)])
                self.sigma_heads = nn.ParameterList(
                    [nn.Parameter(self.sigma.detach().clone()) for _ in range(k - 1)])
                self._trunk_out = None
                self.actor_mlp.register_forward_hook(self._keep_trunk_out)
            self.design_nets = None
            if self.per_design_nets > 1:
                if self.is_rnn():
                    raise NotImplementedError("per_design_nets: MLP networks only")
                self.design_nets = nn.ModuleList(
                    [_DesignNet(params, **kwargs) for _ in range(self.per_design_nets - 1)])

        def _keep_trunk_out(self, _module, _inp, out) -> None:
            self._trunk_out = out

        def _design_of(self, obs: torch.Tensor, k: int) -> torch.Tensor:
            end = obs.shape[1] - self.ghost_mask_tail
            return obs[:, end - k:end].argmax(dim=1)

        @staticmethod
        def _select(outs: list, design: torch.Tensor) -> torch.Tensor:
            x = torch.stack(outs, dim=0)
            idx = design.view(1, -1, *([1] * (x.dim() - 2))).expand(1, *x.shape[1:])
            return x.gather(0, idx)[0]

        def _heads_forward(self, obs_dict):
            mu0, sigma0, value0, states = A2CBuilder.Network.forward(self, obs_dict)
            out = self._trunk_out
            design = self._design_of(obs_dict["obs"], self.per_design_heads)
            mus = [mu0] + [self.mu_act(h(out)) for h in self.mu_heads]
            sigmas = [sigma0] + [mu0 * 0.0 + self.sigma_act(p) for p in self.sigma_heads]
            values = [value0] + [self.value_act(h(out)) for h in self.value_heads]
            return self._select(mus, design), self._select(sigmas, design), self._select(values, design), states

        def design_subnets(self) -> list:
            return [self] + list(self.design_nets or [])

        def own_parameters(self) -> list:
            extra = {id(p) for p in self.design_nets.parameters()} if self.design_nets is not None else set()
            return [p for p in self.parameters() if id(p) not in extra]

        def design_normalised_obs(self, obs: torch.Tensor):
            """(body normalised per design, tail unchanged): the tail is the
            design one-hot and the slot signs."""
            k = self.per_design_input_norm
            end = obs.shape[1] - self.ghost_mask_tail
            design = obs[:, end - k:end].argmax(dim=1)
            return self.input_norm(obs[:, :end - k], design), obs[:, end - k:]

        def design_forward(self, d: int, obs_dict):
            """Design d's actor-critic on the whole batch (no routing, no ghost mask)."""
            if d == 0:
                return A2CBuilder.Network.forward(self, obs_dict)
            return self.design_nets[d - 1](obs_dict)

        def project_sigma(self) -> None:
            if self.logstd_min is None and self.logstd_max is None:
                return
            sigmas = [getattr(net, "sigma", None) for net in self.design_subnets()] + list(self.sigma_heads or [])
            for sigma in sigmas:
                if isinstance(sigma, nn.Parameter):
                    with torch.no_grad():
                        sigma.clamp_(min=self.logstd_min, max=self.logstd_max)

        def _routed_forward(self, obs_dict):
            obs = obs_dict["obs"]
            k = self.per_design_nets
            end = obs.shape[1] - self.ghost_mask_tail
            design = obs[:, end - k:end].argmax(dim=1)
            outs = [self.design_forward(d, obs_dict) for d in range(k)]
            if not self.is_continuous or len(outs[0]) != 4:
                raise NotImplementedError("per_design_nets: continuous actions only")

            def pick(j):
                x = torch.stack([o[j] for o in outs], dim=0)
                idx = design.view(1, -1, *([1] * (x.dim() - 2))).expand(1, *x.shape[1:])
                return x.gather(0, idx)[0]

            return pick(0), pick(1), pick(2), outs[0][3]

        def forward(self, obs_dict):
            self.project_sigma()
            if self.input_norm is not None:
                body, tail = self.design_normalised_obs(obs_dict["obs"])
                obs_dict = dict(obs_dict, obs=torch.cat([body, tail], dim=-1))
            if self.design_nets is not None:
                out = self._routed_forward(obs_dict)
            elif self.mu_heads is not None:
                out = self._heads_forward(obs_dict)
            else:
                out = super().forward(obs_dict)
            n = self.ghost_mask_tail
            if n <= 0 or not self.is_continuous or len(out) != 4:
                return out
            mu, sigma, value, states = out
            ghost = obs_dict["obs"][:, -n:] < 0
            zero = torch.zeros_like(mu)
            return torch.where(ghost, zero, mu), torch.where(ghost, zero, sigma), value, states

    def build(self, name, **kwargs):
        return BoundedSigmaA2CBuilder.Network(self.params, **kwargs)


model_builder.register_network(NETWORK_NAME, BoundedSigmaA2CBuilder)
