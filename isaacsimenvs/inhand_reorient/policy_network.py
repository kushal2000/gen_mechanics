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
            self.design_nets = None
            if self.per_design_nets > 1:
                if self.is_rnn():
                    raise NotImplementedError("per_design_nets: MLP networks only")
                self.design_nets = nn.ModuleList(
                    [_DesignNet(params, **kwargs) for _ in range(self.per_design_nets - 1)])

        def design_subnets(self) -> list:
            return [self] + list(self.design_nets or [])

        def own_parameters(self) -> list:
            extra = {id(p) for p in self.design_nets.parameters()} if self.design_nets is not None else set()
            return [p for p in self.parameters() if id(p) not in extra]

        def design_forward(self, d: int, obs_dict):
            """Design d's actor-critic on the whole batch (no routing, no ghost mask)."""
            if d == 0:
                return A2CBuilder.Network.forward(self, obs_dict)
            return self.design_nets[d - 1](obs_dict)

        def project_sigma(self) -> None:
            if self.logstd_min is None and self.logstd_max is None:
                return
            for net in self.design_subnets():
                sigma = getattr(net, "sigma", None)
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
            out = self._routed_forward(obs_dict) if self.design_nets is not None else super().forward(obs_dict)
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
