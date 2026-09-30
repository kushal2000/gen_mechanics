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
import torch.utils.checkpoint

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

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float,
                 affine_norm: bool = True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.dropout = dropout
        self.ln_attn = nn.LayerNorm(d_model, elementwise_affine=affine_norm)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_ff = nn.LayerNorm(d_model, elementwise_affine=affine_norm)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, dim = x.shape
        heads = self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, tokens, heads, dim // heads).transpose(1, 2)
            for t in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        if key_mask is not None:
            # A ghost joint is a token the design does not have. Masking the KEY
            # only: its own row still attends (to the global token at minimum,
            # which is never masked), so no row is all -inf and no softmax NaNs.
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        if self.dropout and self.training:
            weights = F.dropout(weights, p=self.dropout)
        attn = weights @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, tokens, dim))
        return x + self.ff(self.ln_ff(x))


class _ParallelEncoderLayer(nn.Module):
    """PaLM-style parallel block: attention and FF read the SAME normalized x.

        x + Wout( concat[ attn(LN(x)), gelu(ff_up(LN(x))) ] )

    rather than the serial ``x = x + attn(LN(x)); x = x + ff(LN(x))``. The two
    branches no longer depend on each other, which collapses six ops into
    three: one LayerNorm instead of two, qkv AND ff-up as a single
    Linear(d, 3d + ff_mult*d), proj AND ff-down as a single
    Linear(d + ff_mult*d, d), and one residual add instead of two.

    That matters here because this block is memory-bound, not compute-bound --
    at d_model 32 the arithmetic intensity is ~16 FLOP/byte against a machine
    balance of 190, so time tracks the number of intermediates written and
    re-read, not the number of multiplies. Measured at minibatch 16384, 2
    layers, bf16 + compile: 4.65 -> 4.19 ms (1.11x), with identical parameter
    count and unchanged attention.

    PaLM reports ~15% faster training from this and no quality degradation at
    62B, though a small degradation at 8B -- so it is not free in principle and
    this network is far smaller than either. It is a deliberate flag.

    Permutation equivariance over tokens is untouched: every weight here is
    applied per token, exactly as in the serial block.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float,
                 affine_norm: bool = True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_model = d_model
        self.ff_dim = ff_mult * d_model
        self.ln = nn.LayerNorm(d_model, elementwise_affine=affine_norm)
        self.w_in = nn.Linear(d_model, 3 * d_model + self.ff_dim)
        self.w_out = nn.Linear(d_model + self.ff_dim, d_model)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, dim = x.shape
        heads, d = self.n_heads, self.d_model
        h = self.w_in(self.ln(x))
        q, k, v, u = h[..., :d], h[..., d:2 * d], h[..., 2 * d:3 * d], h[..., 3 * d:]
        q, k, v = (
            t.view(batch, tokens, heads, dim // heads).transpose(1, 2)
            for t in (q, k, v)
        )
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        weights = scores.softmax(dim=-1)
        if self.dropout and self.training:
            weights = F.dropout(weights, p=self.dropout)
        attn = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return x + self.w_out(torch.cat([attn, F.gelu(u)], dim=-1))


class _RunningNorm(nn.Module):
    """Running mean/std standardisation that the network owns, so it chooses what to pool.

    Replaces rl_games' input normaliser for this network (``normalize_input`` must be False).
    That one standardises every flat column on its own statistics, which breaks a SHARED token
    projection: joint j's link-box x and joint k's link-box x get different scales, and a
    column that is constant per joint -- its limits, where its link sits -- normalises to 0 and
    is erased. Here a token field is standardised on statistics POOLED over every real joint,
    so a weight means the same thing for every joint and between-joint differences survive.

    Same arithmetic as rl_games' RunningMeanStd -- prior mean 0 / var 1 / count 1, unbiased
    batch variance, Chan merge, eps 1e-5, clip +-5, summed across ranks under DDP (tested
    equal to it) -- with a weight per row so ghost joints do not enter the statistics, and the
    same schedule (install_rl_games_hook): updated in the first mini-epoch of each update only. Buffers are updated in place and outside any
    compiled graph.
    """

    def __init__(self, dim: int, eps: float = 1e-5, clip: float = 5.0):
        super().__init__()
        self.eps, self.clip = eps, clip
        self.register_buffer("running_mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))   # rl_games' prior

    @torch.compiler.disable
    @torch.no_grad()
    def update(self, x: torch.Tensor, weight: torch.Tensor | None = None) -> None:
        """Fold a batch into the statistics, if this normaliser is in training mode.

        Its OWN mode, not the network's: rl_games freezes input statistics after the first
        mini-epoch of every update by calling ``model.running_mean_std.eval()`` while the
        model stays in training mode (a2c_common.py train_epoch). ``install_rl_games_hook``
        routes that call here. ``x`` is (..., dim); ``weight`` is x.shape[:-1].
        """
        if not self.training:
            return
        x = x.detach().reshape(-1, x.shape[-1]).double()
        w = (torch.ones(x.shape[0], dtype=torch.float64, device=x.device) if weight is None
             else weight.detach().reshape(-1).double())
        stats = torch.cat([w.sum().view(1), (x * w[:, None]).sum(0), (x * x * w[:, None]).sum(0)])
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            torch.distributed.all_reduce(stats)
        n = stats[0]
        if n <= 0:
            return
        dim = x.shape[-1]
        batch_mean = stats[1:1 + dim] / n
        # Unbiased, as rl_games' input.var(0).
        batch_var = ((stats[1 + dim:] / n - batch_mean ** 2) * n / (n - 1).clamp(min=1.0)).clamp(min=0.0)
        delta = batch_mean - self.running_mean
        total = self.count + n
        m2 = (self.running_var * self.count + batch_var * n
              + delta ** 2 * self.count * n / total)
        self.running_mean.add_(delta * n / total)
        self.running_var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.running_mean.to(x.dtype)
        std = torch.sqrt(self.running_var.to(x.dtype) + self.eps)
        return ((x - mean) / std).clamp(-self.clip, self.clip)


class _InputNormModeProxy(nn.Module):
    """Stands in for rl_games' ``model.running_mean_std`` so rl_games drives our normalisers.

    rl_games updates input statistics during the FIRST mini-epoch of each update only: it calls
    ``model.running_mean_std.eval()`` after it, and ``model.train()`` / ``model.eval()`` around
    rollouts and updates. With ``normalize_input: False`` it has no normaliser to call, so this
    proxy takes the attribute and forwards train/eval to the network's own normalisers. It never
    transforms anything (``model.norm_obs`` still sees ``normalize_input`` False on the model)
    and holds no state, so rl_games' checkpoint entry for it is an empty dict.
    """

    def __init__(self, norms):
        super().__init__()
        object.__setattr__(self, "_norms", tuple(norms))    # not submodules: no state_dict copy

    def forward(self, x):
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        for n in self._norms:
            n.train(mode)
        return self


def install_rl_games_hook(algo) -> bool:
    """Give rl_games' input-normalisation schedule to a joint transformer's own normalisers.

    Called from an AlgoObserver's ``after_init`` (the model exists by then). No-op, returning
    False, for any other network. Sets ``algo.normalize_input`` so rl_games' train_epoch issues
    its per-update freeze; the MODEL's flag stays False, so rl_games still never normalises.
    """
    net = getattr(getattr(algo, "model", None), "a2c_network", None)
    if not isinstance(net, JointTransformerNet):
        return False
    if getattr(algo.model, "normalize_input", False):
        raise ValueError("joint_transformer normalises its own input; normalize_input must be False")
    algo.model.running_mean_std = _InputNormModeProxy([net.token_norm, net.global_norm])
    algo.normalize_input = True
    net._rl_games_hooked = True
    return True


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
        # This network normalises its own input (_RunningNorm). rl_games' per-column
        # normaliser on top would erase what the pooled one keeps, and would turn the raw
        # joint_enabled column the attention mask reads into noise.
        if kwargs.get("normalize_input", False):
            raise ValueError(
                "joint_transformer normalises its own input; set normalize_input: False "
                "in the train YAML (rl_games' per-column normaliser breaks the shared token "
                "projection and the attention mask)")
        self.token_norm = _RunningNorm(layout["token_dim"])
        self.global_norm = _RunningNorm(layout["global_dim"])
        self._rl_games_hooked = False     # set by install_rl_games_hook
        self._warned_unhooked = False

        d_model = self.d_model
        # The un-projected global vector, kept for the skip below.
        self.global_raw_dim = layout["global_dim"] + coef_dim
        self.token_proj = nn.Linear(layout["token_dim"], d_model)
        self.global_proj = nn.Linear(self.global_raw_dim, d_model)
        block = _ParallelEncoderLayer if self.parallel_block else _EncoderLayer
        self.layers = nn.ModuleList(
            block(d_model, self.n_heads, self.ff_mult, self.dropout,
                  affine_norm=self.affine_norm)
            for _ in range(self.n_layers)
        )
        self.ln_out = nn.LayerNorm(d_model, elementwise_affine=self.affine_norm)

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
        mu_in = d_model + (d_model + self.global_raw_dim if self.hand_global_skip else 0)
        if self.mu_head_units:
            self.mu_head = _mlp(mu_in, self.mu_head_units, 1)
        else:
            self.mu_head = nn.Linear(mu_in, 1)
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
        self.n_heads = params.get("n_heads", 1)
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
        # Give the SHARED per-joint action head the global token and the raw global
        # vector as well as the joint's own token -- the hand's version of the raw
        # skip the arm head already has. Off by default, so existing checkpoints and
        # runs are unchanged. Why: for a hand-only robot there is no arm head, so
        # every action is mu_head(joint_token), and the goal (keypoints_rel_goal) and
        # object pose/velocity are GLOBAL-only fields (layout.py). Each joint can only
        # learn where to push through one attention read of a d_model token. On the
        # in-hand task with the reward fixed (fall_penalty 200, gamma 0.998) the MLP
        # reaches ~46 goals/episode at 20 deg while this network stays at ~0.06.
        self.hand_global_skip = bool(params.get("hand_global_skip", False))
        # Activation checkpointing: each encoder layer recomputes its activations in the backward
        # pass instead of storing them. Mathematically identical (same outputs and gradients --
        # dropout is 0), costs one extra forward per layer, and cuts activation memory so a deeper or
        # wider model fits at the same minibatch. Off by default.
        self.grad_checkpoint = bool(params.get("grad_checkpoint", False))
        # Ghost action dimensions. A padded design's ghost slots still get an action, and rl_games puts
        # every action dimension into the PPO log-probability, the KL that drives the adaptive lr, and the
        # entropy SAPG's exploration bonus rewards -- although a ghost joint is locked and its action does
        # nothing. On gen-SHARPA (9 ghosts of 30) their learned noise grew to sigma ~1.6 vs ~0.8 on real
        # joints, 44% of the policy's entropy. And here every action comes from ONE shared head, so an
        # update for real joints also moves ghost means. With this on, a ghost dimension gets a constant
        # mean (0) and log-std (0), no gradient: it cancels out of the ratio and the KL and contributes a
        # constant entropy. Masked PER ENV from that env's raw joint_enabled, so a mixed population masks
        # each design's own ghosts; arm dimensions are never masked. Off by default.
        self.mask_ghost_actions = bool(params.get("mask_ghost_actions", False))
        # The final LayerNorm over the residual stream. Meaningful once there
        # are blocks to stabilize; with n_layers 0 there is no residual stream
        # and it only normalizes a bare Linear whose scale the next Linear can
        # absorb -- and the MLP baseline this is compared against has no
        # normalization anywhere. Setting it False CHANGES the model, so it is
        # a deliberate flag rather than an automatic optimization.
        self.final_norm = bool(params.get("final_norm", True))
        # PaLM-style parallel block: attention and FF read the same LayerNorm
        # and their input/output projections fuse into one GEMM each. 1.11x at
        # d_model 32 with 2 layers, same parameter count, attention unchanged.
        # See _ParallelEncoderLayer -- it is a different model, not just a
        # faster one, so it is off by default.
        self.parallel_block = bool(params.get("parallel_block", False))
        # Every LayerNorm in this network is immediately followed by a Linear,
        # which can absorb any per-channel scale and shift -- so gamma/beta are
        # mathematically redundant here. Dropping them deletes the
        # GammaBetaBackward reduction (376832 rows down to d_model elements,
        # pure memory traffic) for ~7% and a little activation memory. Left on
        # by default so existing checkpoints keep loading.
        self.affine_norm = bool(params.get("affine_norm", True))
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
            "enabled_index", torch.tensor(layout["enabled_columns"], dtype=torch.long),
            persistent=False,
        )
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

        raw_tokens = (
            torch.index_select(env_obs, 1, self.token_gather)
            .view(env_obs.shape[0], self.n_hand, self.token_dim)
        )
        # The attention mask, from joint_enabled RAW: rl_games' normaliser is off for this
        # network, so the column is exactly 0 or 1 -- a ghost slot of a padded design is 0.
        # It is not a token feature. (Until 2026-09-29 the mask read "> 0.5" off a column
        # rl_games had standardised, where a fixed hand's constant 1 is 0: every joint was
        # masked and attended to the global token alone -- no joint could see another.)
        valid = torch.index_select(env_obs, 1, self.enabled_index) > 0.5
        glob = torch.index_select(env_obs, 1, self.global_index)
        if self.token_norm.training and not self._rl_games_hooked and not self._warned_unhooked:
            self._warned_unhooked = True
            print("[joint_transformer] WARNING: training without install_rl_games_hook: input "
                  "statistics update on every mini-epoch instead of the first only", flush=True)
        self.token_norm.update(raw_tokens, valid)         # pooled over real joints only
        self.global_norm.update(glob)
        tokens = self.token_proj(self.token_norm(raw_tokens))
        glob = self.global_norm(glob)
        if coef is not None:
            glob = torch.cat([glob, coef], dim=-1)

        global_token = self.global_proj(glob)
        if self.layers:
            # Attention needs all n_hand + 1 tokens as one sequence.
            x = torch.cat([tokens, global_token.unsqueeze(1)], dim=1)
            # The global token is always present, so every query keeps one key.
            key_mask = torch.cat(
                [valid, torch.ones_like(valid[:, :1])], dim=1)
            ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for layer in self.layers:
                if ckpt:
                    x = torch.utils.checkpoint.checkpoint(layer, x, key_mask, use_reentrant=False)
                else:
                    x = layer(x, key_mask)
            if self.final_norm:
                x = self.ln_out(x)
            return x[:, : self.n_hand], x[:, self.n_hand], glob, valid

        # No blocks: the concatenation would be undone by the split on the very
        # next line with nothing in between, and LayerNorm reduces only over the
        # last dim, so normalizing the pieces separately is BIT-IDENTICAL to
        # normalizing the concatenation (verified: max|diff| = 0.0). Skipping
        # the cat saves a full copy of the (B, n_hand+1, d_model) tensor.
        if self.final_norm:
            tokens = self.ln_out(tokens)
            global_token = self.ln_out(global_token)
        return tokens, global_token, glob, valid

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        joints, glob, glob_raw, valid = self._trunk(obs)
        # Mean over the joints this design HAS. An unmasked mean would be mostly
        # ghosts for a small hand, and diluted by a design-dependent amount --
        # exactly the variable a co-design run is trying to measure.
        w = valid.unsqueeze(-1).to(joints.dtype)
        pooled = (joints * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        value = self.value_head(torch.cat([pooled, glob, glob_raw], dim=-1))
        if self.central_value:
            return value, None

        if self.hand_global_skip:
            ctx = torch.cat([glob, glob_raw], dim=-1).unsqueeze(1).expand(
                -1, joints.shape[1], -1)
            mu_hand = self.mu_head(torch.cat([joints, ctx], dim=-1)).squeeze(-1)
        else:
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
        sigma = mu * 0 + sigma
        if self.mask_ghost_actions:
            arm_valid = torch.ones(valid.shape[0], self.n_arm, dtype=torch.bool, device=valid.device)
            act_valid = torch.cat([arm_valid, valid], dim=1)       # policy order: arm, then hand
            mu = torch.where(act_valid, mu, torch.zeros_like(mu))
            sigma = torch.where(act_valid, sigma, torch.zeros_like(sigma))
        return mu, sigma, value, None

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


__all__ = ["JointTransformerBuilder", "JointTransformerNet", "install_rl_games_hook"]
