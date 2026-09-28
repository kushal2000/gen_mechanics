"""rl_games actor-critic with a bounded, state-independent policy log-std.

Registered as ``inhand_actor_critic``: rl_games' stock ``actor_critic``
(``A2CBuilder``), plus two optional keys under ``network.space.continuous``:

    logstd_max: 0.0     # project the log-std parameter to <= this before every forward
    logstd_min: -5.0    # ... and to >= this (omit either to leave that side open)

Why (I41): the population controller's log-std parameter (``a2c_network.
sigma``, ``fixed_sigma: coef_cond``) drifted from 0 to about 6.7 over seven
evolution-pilot generations (std ~250). Actions are clipped to [-1, 1]
before they reach the env, so any std well above 1 only makes the executed
actions bang-bang: it adds no exploration, it removes the reward's grip on
the std (the executed action distribution stops changing as sigma grows),
and a large std shrinks the policy KL per update, which pushes the adaptive
learning rate to its 1e-2 ceiling. Projecting the parameter (rather than
clamping only the forward output) keeps its gradient alive inside the range:
a hard clamp on the output alone would leave a parameter that overshot the
bound with zero gradient, stuck there.

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


class BoundedSigmaA2CBuilder(A2CBuilder):
    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            continuous = params.get("space", {}).get("continuous", {}) or {}
            lo = continuous.get("logstd_min")
            hi = continuous.get("logstd_max")
            self.logstd_min = None if lo is None else float(lo)
            self.logstd_max = None if hi is None else float(hi)

        def project_sigma(self) -> None:
            sigma = getattr(self, "sigma", None)
            if not isinstance(sigma, nn.Parameter):
                return
            if self.logstd_min is None and self.logstd_max is None:
                return
            with torch.no_grad():
                sigma.clamp_(min=self.logstd_min, max=self.logstd_max)

        def forward(self, obs_dict):
            self.project_sigma()
            return super().forward(obs_dict)

    def build(self, name, **kwargs):
        return BoundedSigmaA2CBuilder.Network(self.params, **kwargs)


model_builder.register_network(NETWORK_NAME, BoundedSigmaA2CBuilder)
