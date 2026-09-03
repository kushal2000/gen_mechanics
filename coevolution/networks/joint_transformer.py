"""A decentralized per-joint transformer policy, as an rl_games network.

The MLP and this network receive the same explicit observation vector.  This
network gathers each HAND joint's 32 values into a token, runs full attention
over them plus one global token, and reads every hand command from one shared
head.  There is deliberately no joint-name parser or learned joint identity:
ordered link-box keypoints describe the controlled mechanism geometrically.

The 7 arm joints are not tokenized. Their observations are routed into the
global token and their actions come from a separate MLP head, because arm and
hand actions mean different things downstream (``obs_utils/actions.py``
accumulates arm velocity deltas but rescales hand targets absolutely), and
sharing one head across that boundary would be sharing across a real seam.

Registered as ``joint_transformer``; select it with ``params.network.name`` in
the train YAML. Nothing under ``third_party/rl_games`` is modified -- this
plugs in through ``model_builder.register_network``.

SAPG compatibility. The vendored fork appends one column to every observation
carrying that env's exploration coefficient, and expects the network to (a)
replace it with a learned 32-d embedding (``type: extra_param``) and (b) select
a per-block sigma row from it (``fixed_sigma: coef_cond``). Both are replicated
here exactly as ``network_builder.py`` does them. The embedding goes on the
single global token, not on all 22 joint tokens: it is a per-env constant, and
repeating it per token would spend 22x the width to say the same thing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_games.algos_torch.network_builder import NetworkBuilder


def _mlp(in_dim: int, units, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for width in units:
        layers += [nn.Linear(dim, width), nn.ELU()]
        dim = width
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class _EncoderLayer(nn.Module):
    """Pre-LN transformer block with explicit attention.

    Written out rather than using ``nn.TransformerEncoderLayer`` so nothing in
    the block depends on a fused kernel's shape support.

    Attention is a plain matmul pair, NOT
    ``F.scaled_dot_product_attention``. SDPA launches a grid indexed by
    ``batch * n_heads`` and CUDA caps a grid dimension at 65535, so with 4
    heads every SDPA kernel dies with "invalid configuration argument" at
    minibatch >= 16384 -- which is precisely where raising the minibatch stops
    being an option. The fused kernels exist to avoid materializing the
    (B, heads, T, T) scores, and at T = 23 that matrix is 2 KiB per head per
    sample: there is nothing to avoid. Measured on an RTX 6000 Ada at
    minibatch 16384, the explicit form is also slightly faster than SDPA
    (28.2 vs 30.4 ms fwd+bwd per layer) and agrees with it to 1e-6.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.dropout = dropout
        self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        heads = self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, tokens, heads, dim // heads).transpose(1, 2)
            for t in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        weights = scores.softmax(dim=-1)
        if self.dropout and self.training:
            weights = F.dropout(weights, p=self.dropout)
        attn = weights @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, tokens, dim))
        return x + self.ff(self.ln_ff(x))


class JointTransformerNet(NetworkBuilder.BaseNetwork):
    """One token per hand joint, one global token, one shared action head."""

    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        input_shape = kwargs.pop("input_shape")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)
        self.net_type = kwargs.pop("type", "simple")

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)

        # --- SAPG's exploration-coefficient column -------------------------
        # Mirrors network_builder.py:205-209. `input_shape` is unreliable here:
        # a2c_continuous passes obs_dim+1 and the reference builder then
        # overwrites it again, so the true env obs width is `coef_id_idx`.
        if self.net_type == "extra_param":
            self.param_ids = kwargs["coef_ids"]  # plain attr, NOT a buffer
            param_size = kwargs.pop("param_size", 32)
            self.pid_idx = kwargs["coef_id_idx"]
            self.extra_params = nn.Parameter(
                torch.randn((len(self.param_ids), param_size), dtype=torch.float32)
            )
            env_obs_dim = self.pid_idx
            coef_dim = param_size
            if len(self.param_ids) != 6:
                print(
                    f"[joint_transformer] WARNING: {len(self.param_ids)} SAPG "
                    "exploration blocks. rl_games' PpoPlayerContinuous hardcodes "
                    "6 (players.py:48), so this checkpoint will NOT restore in "
                    "eval. Train with num_envs = 6 * expl_coef_block_size."
                )
        else:
            self.param_ids = None
            self.pid_idx = None
            env_obs_dim = input_shape[0]
            coef_dim = 0

        if self.final_norm and self.d_model % 4:
            # torch's layer_norm_kernel takes a vectorized path only when the
            # normalized dim is a multiple of 4 (pytorch#145145); otherwise it
            # falls back to RowwiseMoments + LayerNormForward. Measured on an
            # RTX 6000 Ada over (16384, 23, d): d=46 costs 2.11 ms, d=48 costs
            # 0.47 ms. A width chosen as 1024/22 = 46 lands exactly on it.
            print(f"[joint_transformer] WARNING: d_model={self.d_model} is not "
                  "a multiple of 4, so LayerNorm falls off torch's vectorized "
                  "kernel and costs ~4.5x more. Round to a multiple of 8.")

        layout = self._build_layout(env_obs_dim)
        n_hand = layout["n_hand"]
        n_arm = layout["n_arm"]
        self.n_hand = n_hand
        self.n_arm = n_arm

        self._register_indices(layout)

        d_model = self.d_model
        # The un-projected global vector, kept for the skip below.
        self.global_raw_dim = layout["global_dim"] + coef_dim
        self.token_proj = nn.Linear(layout["token_dim"], d_model)
        self.global_proj = nn.Linear(self.global_raw_dim, d_model)
        self.layers = nn.ModuleList(
            _EncoderLayer(d_model, self.n_heads, self.ff_mult, self.dropout)
            for _ in range(self.n_layers)
        )
        self.ln_out = nn.LayerNorm(d_model)

        # Mean-pooled joint tokens, the global token, and the raw global vector.
        # The value must see the whole hand, not one joint's view of it.
        self.value_head = _mlp(
            2 * d_model + self.global_raw_dim, self.value_head_units, self.value_size
        )

        if self.central_value:
            self._maybe_compile()
            return

        # Shared across all 22 tokens -- this is what makes it decentralized.
        #
        # A bare Linear(d_model, 1) is an N=1 GEMM: it spends 1.00 ms per
        # fwd+bwd at minibatch 16384 on 2,816 MACs/env, i.e. essentially all
        # launch and bandwidth floor. Linear(d_model, 8) measured 1.00 ms --
        # identical -- so per-joint width is free until it clears that floor.
        # ``mu_head_units`` buys real decentralized capacity for nothing;
        # empty (the default) keeps the single Linear, and with it the
        # `mu_head.weight` state_dict key that existing checkpoints carry.
        if self.mu_head_units:
            self.mu_head = _mlp(d_model, self.mu_head_units, 1)
        else:
            self.mu_head = nn.Linear(d_model, 1)
        # The arm carries the 6D pose task: once the object is grasped, driving it
        # through SE(3) goals is mostly arm motion, and the hand only has to hold
        # on. A first run that read the arm off the d_model-wide global token
        # alone reached and lifted as well as the MLP baseline but scored ~0 on
        # the post-lift keypoint term -- every palm/object/keypoint/goal number
        # was being squeezed through one token. Hence the raw skip.
        self.arm_head = _mlp(d_model + self.global_raw_dim, self.arm_head_units, n_arm)
        self.mu_act = self.activations_factory.create(
            self.space_config.get("mu_activation", "None")
        )
        self.sigma_act = self.activations_factory.create(
            self.space_config.get("sigma_activation", "None")
        )
        if self.fixed_sigma == "fixed":
            self.sigma = nn.Parameter(
                torch.zeros(actions_num, dtype=torch.float32), requires_grad=True
            )
        elif self.fixed_sigma == "coef_cond":
            self.sigma_ids = kwargs["coef_ids"]
            self.sigma_id_idx = kwargs["coef_id_idx"]
            self.sigma = nn.Parameter(
                torch.zeros(len(self.sigma_ids), actions_num, dtype=torch.float32),
                requires_grad=True,
            )
        else:
            raise ValueError(
                f"fixed_sigma={self.fixed_sigma!r} is not supported by "
                "joint_transformer; use 'fixed' (PPO) or 'coef_cond' (SAPG)"
            )
        sigma_init = self.init_factory.create(
            **self.space_config.get("sigma_init", {"name": "const_initializer", "val": 0})
        )
        sigma_init(self.sigma)
        self._maybe_compile()

    def _maybe_compile(self) -> None:
        """Compile the WHOLE forward, not just the encoder stack.

        Scope is the single biggest lever measured on this network. At d_model
        64, batch 16384, one L1 train step: eager 21.66 ms, compiling only
        ``self.layers`` 15.42 ms (1.41x), compiling the whole module 10.94 ms
        (1.98x), and with mode="max-autotune" 9.83 ms (2.20x). The extra factor
        comes from Inductor being able to fuse ACROSS the token gather,
        token_proj, the cat, ln_out and the heads -- the plumbing that the
        profile shows dominating, and which layer-scope compilation leaves
        outside the fusion region.

        ``Module.compile`` rather than ``torch.compile(module)``: the latter
        returns an OptimizedModule wrapper that prefixes every state_dict key
        with "_orig_mod.", silently breaking checkpoint compatibility with runs
        trained without the flag.

        dynamic=False deliberately. The batch takes only TWO shapes in a normal
        run -- num_envs for the rollout forward, minibatch_size for the
        gradient steps -- so static compilation pays for two compilations once
        and is then stable. (A minibatch that does not divide the batch adds a
        third, from PPODataset's oversized last minibatch; 16384 divides both
        458752 and 393216 exactly, so it does not arise here.) An earlier
        dynamic=True build REGRESSED the real training loop by 19%.

        NOTE mode="max-autotune" helped only in the small-d_model regime
        (d=64: 10.94 -> 9.83 ms; d=128: no improvement over the default mode)
        and costs minutes of startup, so it is opt-in via compile_mode.
        """
        if not self.compile_net:
            return
        kwargs = {"dynamic": False}
        if self.compile_mode:
            kwargs["mode"] = self.compile_mode
        self.compile(**kwargs)

    # ------------------------------------------------------------------ setup

    def load(self, params):
        self.params = params
        self.central_value = params.get("central_value", False)
        self.d_model = params.get("d_model", 128)
        self.n_layers = params.get("n_layers", 4)
        self.n_heads = params.get("n_heads", 4)
        self.ff_mult = params.get("ff_mult", 4)
        self.dropout = params.get("dropout", 0.0)
        # Off by default: torch.compile changes nothing numerically but it does
        # change startup cost and failure modes, so a run opts into it.
        # ``compile_layers`` is accepted as the old name for the same switch.
        self.compile_net = bool(
            params.get("compile_net", params.get("compile_layers", False))
        )
        self.compile_mode = params.get("compile_mode", "") or None
        self.arm_head_units = list(params.get("arm_head_units", [256, 128]))
        # Per-joint action head width. [] keeps the original single Linear.
        self.mu_head_units = list(params.get("mu_head_units", []))
        # The final LayerNorm over the residual stream. Meaningful once there
        # are blocks to stabilize; with n_layers 0 there is no residual stream
        # and it only normalizes a bare Linear whose scale the next Linear can
        # absorb -- and the MLP baseline this is compared against has no
        # normalization anywhere. Setting it False CHANGES the model, so it is
        # a deliberate flag rather than an automatic optimization.
        self.final_norm = bool(params.get("final_norm", True))
        self.value_head_units = list(params.get("value_head_units", [256, 128]))
        self.robot_spec_name = params["robot_spec"]
        key = "state_list" if self.central_value else "obs_list"
        if key not in params:
            raise KeyError(
                f"params.network.{key} is required by joint_transformer. Set it "
                f"in the train YAML by interpolation, e.g. {key}: "
                "${....env.obs." + key + "}"
            )
        self.field_list = list(params[key])
        self.space_config = params.get("space", {}).get("continuous", {})
        self.fixed_sigma = self.space_config.get("fixed_sigma", "fixed")

    def _build_layout(self, env_obs_dim: int) -> dict:
        # Imported here, not at module scope: this module is imported by
        # train.py before the env package is touched, and the layout builder
        # reaches into isaacsimenvs.
        from isaacsimenvs.pose_reaching_6d.obs_utils.layout import build_token_layout
        from isaacsimenvs.pose_reaching_6d.scene_utils.robots import get_robot_spec

        spec = get_robot_spec(self.robot_spec_name)
        layout = build_token_layout(spec, self.field_list)
        if layout["obs_dim"] != env_obs_dim:
            raise ValueError(
                f"joint_transformer layout mismatch: {self.robot_spec_name} with "
                f"{self.field_list} is {layout['obs_dim']}-d, but rl_games says "
                f"{env_obs_dim}-d. The train YAML's field list has drifted from "
                "the task YAML's."
            )
        return layout

    def _register_indices(self, layout: dict) -> None:
        """Gather indices that cut the flat observation into tokens.

        ``persistent=False`` throughout: these are derived from the config, not
        learned, and a checkpoint that carried them would have to agree with
        them at restore time for no benefit.
        """
        global_index = [
            i for start, end in layout["global_slices"] for i in range(start, end)
        ]

        # Flat, not (n_hand, token_dim): a 2-D advanced index dispatches to the
        # generic `index` kernel with an `index_put_(accumulate=True)` backward,
        # while a 1-D `index_select` has a dedicated kernel and an `index_add_`
        # backward. Same columns, same values, bit-identical -- and measured
        # 8.2x faster fwd+bwd at minibatch 16384 (2.23 ms -> 0.27 ms), which is
        # also faster than laying the observation out token-major and viewing
        # it (0.30 ms). The reshape back to (B, n_hand, token_dim) is free.
        token_columns = layout["token_columns"]
        self.token_dim = layout["token_dim"]
        self.register_buffer(
            "token_gather",
            torch.tensor(token_columns, dtype=torch.long).reshape(-1).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "global_index", torch.tensor(global_index, dtype=torch.long),
            persistent=False,
        )

    # ---------------------------------------------------------------- forward

    def _trunk(self, obs: torch.Tensor):
        """Flat observation -> (joint token outputs, global token output)."""
        if self.net_type == "extra_param":
            # Exact float compare on the raw coefficient column, on the tensor
            # as it arrives -- network_builder.py:319 does the same.
            idxs = (
                (obs[:, self.pid_idx].reshape(-1, 1) == self.param_ids)
                .float()
                .argmax(dim=1)
            )
            coef = self.extra_params[idxs]
            env_obs = obs[:, : self.pid_idx]
        else:
            coef = None
            env_obs = obs

        tokens = self.token_proj(
            torch.index_select(env_obs, 1, self.token_gather)
            .view(env_obs.shape[0], self.n_hand, self.token_dim)
        )

        glob = torch.index_select(env_obs, 1, self.global_index)
        if coef is not None:
            glob = torch.cat([glob, coef], dim=-1)

        global_token = self.global_proj(glob)
        if self.layers:
            # Attention needs all n_hand + 1 tokens as one sequence.
            x = torch.cat([tokens, global_token.unsqueeze(1)], dim=1)
            for layer in self.layers:
                x = layer(x)
            if self.final_norm:
                x = self.ln_out(x)
            return x[:, : self.n_hand], x[:, self.n_hand], glob

        # No blocks: the concatenation would be undone by the split on the very
        # next line with nothing in between, and LayerNorm reduces only over the
        # last dim, so normalizing the pieces separately is BIT-IDENTICAL to
        # normalizing the concatenation (verified: max|diff| = 0.0). Skipping
        # the cat saves a full copy of the (B, n_hand+1, d_model) tensor.
        if self.final_norm:
            tokens = self.ln_out(tokens)
            global_token = self.ln_out(global_token)
        return tokens, global_token, glob

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        joints, glob, glob_raw = self._trunk(obs)
        value = self.value_head(
            torch.cat([joints.mean(dim=1), glob, glob_raw], dim=-1)
        )
        if self.central_value:
            return value, None

        mu_hand = self.mu_head(joints).squeeze(-1)
        mu_arm = self.arm_head(torch.cat([glob, glob_raw], dim=-1))
        # Canonical policy order is arm joints first, then hand joints.
        mu = self.mu_act(torch.cat([mu_arm, mu_hand], dim=-1))

        if self.fixed_sigma == "fixed":
            sigma = self.sigma_act(self.sigma)
        else:
            idxs = (
                (obs[:, self.sigma_id_idx].reshape(-1, 1) == self.sigma_ids)
                .float()
                .argmax(dim=1)
            )
            sigma = self.sigma_act(self.sigma[idxs])
        return mu, mu * 0 + sigma, value, None

    # ------------------------------------------------------ rl_games protocol

    def is_separate_critic(self):
        return False

    def is_rnn(self):
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value_head[-1]


class JointTransformerBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return JointTransformerNet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


__all__ = ["JointTransformerBuilder", "JointTransformerNet"]
