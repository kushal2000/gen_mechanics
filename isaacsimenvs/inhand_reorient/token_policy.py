"""The team's joint-token transformer (``coevolution/networks/joint_transformer``)
on the in-hand envelope, registered as ``inhand_joint_transformer``.

One token per articulation column of the 32-slot envelope, no arm, one
global token (``token_layout.py`` is the layout the env emits under
``hora.token_obs``). Full attention over valid joint tokens plus the global
token; one shared per-joint action head; the value head reads the masked
mean of the joint tokens, the global token and the global vector.

Differences from ``JointTransformerNet``, all for one controller over many
hands:

- The mask comes from the RAW ``enabled`` column. rl_games' input
  normaliser must be off (``config.normalize_input: False``,
  ``InHandHoraTokenPPO.yaml``): it would drive the constant enabled columns
  towards 0 and flip the mask (project-notes/investigation-2026-09-21).
- Token features are normalised inside the network per feature over VALID
  tokens only (``MaskedRunningNorm``); ghost tokens never enter the
  statistics, so a small hand's features are not compressed by padding.
- The global vector is normalised by a running norm, pooled or, with
  ``per_design_global_norm``, per design (``GroupRunningNorm``, the design
  read from the one-hot that ``hora.design_id_obs`` appends; the one-hot is
  not an input feature).
- Ghost tokens get no action: mean 0 and log-std 0, with no gradient.
- The arm head is not used (no arm).
- ``space.continuous.logstd_min`` / ``logstd_max`` project the log-std
  parameter into a range before every forward (unset: unbounded).
- ``mu_head_init_scale`` (default 1) scales the action head's last layer at
  init (bias 0), so the initial mean actions can start near 0.

Statistics update in training mode only (rl_games' PPO epochs), as rl_games'
own ``RunningMeanStd`` does; the player runs in eval mode with them frozen.
"""

from __future__ import annotations

import torch
from torch import nn
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder

from coevolution.networks.joint_transformer import JointTransformerNet

from . import token_layout as tl
from .policy_network import GroupRunningNorm

__all__ = ["NETWORK_NAME", "InhandJointTransformerBuilder", "InhandJointTransformerNet", "MaskedRunningNorm"]

NETWORK_NAME = "inhand_joint_transformer"


class MaskedRunningNorm(nn.Module):
    """Per-feature running mean and variance over the valid tokens of each
    training batch; output (x - mean) / sqrt(var + eps), clipped, with
    ``skip_cols`` passed raw and invalid tokens set to 0."""

    def __init__(self, dim: int, skip_cols=(), eps: float = 1e-5, clip: float = 5.0):
        super().__init__()
        self.eps, self.clip = float(eps), float(clip)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        skip = torch.zeros(dim, dtype=torch.bool)
        skip[list(skip_cols)] = True
        self.register_buffer("skip", skip, persistent=False)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        if n == 0:
            return
        bmean, bvar = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = bmean - self.mean
        self.mean.copy_(self.mean + delta * n / tot)
        self.var.copy_((self.var * self.count + bvar * n + delta ** 2 * self.count * n / tot) / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update(x[valid])
        y = ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
        y = torch.where(self.skip, x, y)
        return y * valid.unsqueeze(-1).to(y.dtype)


class InhandJointTransformerNet(JointTransformerNet):
    def __init__(self, params, **kwargs):
        object.__setattr__(self, "_actions_num", int(kwargs["actions_num"]))  # one token per action column
        super().__init__(params, **kwargs)
        self.token_norm = MaskedRunningNorm(tl.TOKEN_DIM, skip_cols=(tl.ENABLED_COL,))
        k = self.design_id_width
        self.global_norm = GroupRunningNorm(k if (self.per_design_global_norm and k > 0) else 1, tl.GLOBAL_DIM)
        if self.mu_head_init_scale != 1.0:  # small initial means: HORA integrates actions into the targets
            last = self.mu_head[-1] if isinstance(self.mu_head, nn.Sequential) else self.mu_head
            with torch.no_grad():
                last.weight.mul_(self.mu_head_init_scale)
                last.bias.zero_()

    def load(self, params):
        params = dict(params)
        params.setdefault("robot_spec", "inhand_envelope")
        params.setdefault("obs_list", ["hora_tokens", "hora_token_global"])
        super().load(params)
        self.design_id_width = int(params.get("design_id_obs") or 0)
        self.per_design_global_norm = bool(params.get("per_design_global_norm", False))
        self.mu_head_init_scale = float(params.get("mu_head_init_scale", 1.0))
        cont = params.get("space", {}).get("continuous", {}) or {}
        lo, hi = cont.get("logstd_min"), cont.get("logstd_max")
        self.logstd_min = None if lo is None else float(lo)
        self.logstd_max = None if hi is None else float(hi)
        if self.per_design_global_norm and self.design_id_width <= 0:
            raise ValueError("per_design_global_norm needs design_id_obs (the design one-hot width)")

    def _build_layout(self, env_obs_dim: int) -> dict:
        n_tokens = int(self._actions_num)
        lay = tl.layout(n_tokens, self.design_id_width)
        if lay["obs_dim"] != env_obs_dim:
            raise ValueError(f"inhand_joint_transformer: the token layout is {lay['obs_dim']}-d "
                             f"({n_tokens} tokens x {tl.TOKEN_DIM} + {tl.GLOBAL_DIM} + {self.design_id_width}), "
                             f"rl_games says {env_obs_dim}-d")
        self._layout = lay
        return {"n_hand": n_tokens, "n_arm": 0, "token_dim": tl.TOKEN_DIM, "enabled_col": tl.ENABLED_COL,
                "token_columns": lay["token_columns"], "global_slices": [list(lay["global_slice"])],
                "global_dim": tl.GLOBAL_DIM, "obs_dim": lay["obs_dim"]}

    def _design(self, obs: torch.Tensor) -> torch.Tensor:
        if self.design_id_width <= 0 or not self.per_design_global_norm:
            return torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        s, e = self._layout["design_slice"]
        return obs[:, s:e].argmax(dim=1)

    def _trunk(self, obs: torch.Tensor):
        raw = torch.index_select(obs, 1, self.token_gather).view(obs.shape[0], self.n_hand, self.token_dim)
        valid = raw[:, :, self.enabled_col] > 0.5  # RAW: no normaliser runs before this line
        tokens = self.token_proj(self.token_norm(raw, valid))
        glob = self.global_norm(torch.index_select(obs, 1, self.global_index), self._design(obs))
        x = torch.cat([tokens, self.global_proj(glob).unsqueeze(1)], dim=1)
        key_mask = torch.cat([valid, torch.ones_like(valid[:, :1])], dim=1)
        for layer in self.layers:
            x = layer(x, key_mask)
        if self.final_norm:
            x = self.ln_out(x)
        return x[:, : self.n_hand], x[:, self.n_hand], glob, valid

    def forward(self, obs_dict):
        if self.logstd_min is not None or self.logstd_max is not None:  # projected, as policy_network
            with torch.no_grad():
                self.sigma.clamp_(min=self.logstd_min, max=self.logstd_max)
        obs = obs_dict["obs"]
        joints, glob, glob_vec, valid = self._trunk(obs)
        w = valid.unsqueeze(-1).to(joints.dtype)
        pooled = (joints * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        value = self.value_head(torch.cat([pooled, glob, glob_vec], dim=-1))
        mu = self.mu_act(self.mu_head(joints).squeeze(-1))
        sigma = mu * 0 + self.sigma_act(self.sigma)
        zero = torch.zeros_like(mu)
        return torch.where(valid, mu, zero), torch.where(valid, sigma, zero), value, None


class InhandJointTransformerBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return InhandJointTransformerNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


model_builder.register_network(NETWORK_NAME, InhandJointTransformerBuilder)
