"""Does skipping the ghost finger tokens speed up the per-finger transformer? (README.md). Pure torch, no simulator.

    .venv_isaacsim/bin/python experiments/dummy_experiments/29oct_finger_token_batching/bench.py [--quick]

The real network (coevolution/networks/joint_transformer.py, token_mode finger) on the real 100-hand template layout
(multi:template: over the 100 random grammar hands: 7 finger slots + the global token = 8 tokens), with every env's
real fingers from its own hand (round-robin, as the scene deals them): 3.9 real fingers of 7 on average, so 38% of
the tokens the trunk runs on are ghosts. Forwards compared, all with the network's own weights:

  padded      the network's forward as written: (B, 8) tokens, ghost fingers masked as attention keys
  ceiling     padded, but every row with only 5 tokens (the mean real count + the global token, rounded up): the most
              that skipping ghosts could buy, with no bookkeeping
  bucketed    rows grouped by real-finger count n, each group compacted to its n real tokens + the global token (no
              mask needed), the trunk run once per group, outputs scattered back to the slots. Static shapes, so the
              whole forward compiles: a group's row count is a fixed CAPACITY (the rollout's exact count; a training
              minibatch's expected count + 4 sigma, rounded up to 64), rows found by one sort, unused capacity rows
              pointed at a zero dummy row. No host sync. Two versions: gathers and scatters as 2-D advanced indexing,
              and as flat 1-D index_select through inverse indices (index_select's backward is the fast one).
  packed      the LLM way (padding-free): every row's real tokens concatenated into one (R, d) matrix -- the per-token
              work (LayerNorms, QKV, projection, feed-forward) runs on real tokens only, in one big batch -- and
              attention through FlexAttention with a document mask (block-diagonal: each env's tokens see only each
              other). Static R (a capacity, as above), so it compiles; compiled only (FlexAttention needs it).

Timed: rollout inference at 24,576 envs (no grad, fp16 autocast) and one train step at 16,384 rows (forward,
backward, Adam; fp16 autocast), compiled (the trained configuration) and eager. Modelled epoch = the template
scene's simulation time + 16 inferences + 56 train steps (+ 2 ms per minibatch of loop overhead), with the
simulation time read off job 801730 (finger, 2 layers) -- the network terms are what change.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

HAND = "multi:template:assets/urdf/27oct_grammar_random_hands/manifest.json"
OBS_LIST = ["joint_pos", "joint_vel", "prev_joint_pos", "prev_joint_vel", "prev_action_targets",
            "joint_link_bbox", "joint_lower", "joint_upper", "joint_enabled",
            "keypoints_rel_ee", "keypoints_rel_goal", "object_vel", "palm_extents", "task_onehot"]
N_ENVS, HORIZON, AUG_ROWS, MB, MINI_EPOCHS = 24576, 16, 458752, 16384, 2
UPDATES = MINI_EPOCHS * -(-AUG_ROWS // MB)                 # 56
LOOP_MS = 2.0                                              # rl_games' per-minibatch overhead (29oct_transformer_speed)
SPACE = {"continuous": {"mu_activation": "None", "sigma_activation": "None", "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0}, "fixed_sigma": "fixed"}}


def make_net(d_model: int, n_layers: int):
    from coevolution.networks.joint_transformer import JointTransformerNet, _WithArmJoints
    from isaacsimenvs.common.obs_layout import field_offsets
    from isaacsimenvs.common.robots import get_robot_spec
    spec = _WithArmJoints(get_robot_spec(HAND), 6)
    obs_dim = max(e for _, e in field_offsets(OBS_LIST, spec, {"task_onehot": 8}).values())
    params = {"robot_spec": HAND, "obs_list": OBS_LIST, "extra_arm_joints": 6, "extra_field_sizes": {"task_onehot": 8},
              "token_mode": "finger", "d_model": d_model, "n_layers": n_layers, "n_heads": 4, "ff_mult": 4,
              "dropout": 0.0, "decoder_units": [256], "arm_head_units": [1024, 512], "value_head_units": [512, 256],
              "affine_norm": False, "mask_ghost_actions": True, "masked_arm_dims": [3, 4, 5], "space": SPACE}
    net = JointTransformerNet(params, actions_num=6 + spec.num_hand_joints, input_shape=(obs_dim,), value_size=1,
                              num_seqs=1, type="simple")
    return net, obs_dim


def env_observations(net, obs_dim: int, n: int, device, seed: int = 0) -> torch.Tensor:
    """Random observations whose joint_enabled columns are each env's own hand's (round-robin over the 100)."""
    from isaacsimenvs.common.robots.multi_hand import hand_set
    hs = hand_set(HAND)
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    valid = torch.as_tensor(hs.per_env(hs.hand_of_env(n))["joint_valid"], dtype=torch.float32)
    obs[:, net.enabled_index.cpu()] = valid
    return obs.to(device)


# --- the forward, in pieces ----------------------------------------------------------------------------------------

def _pre(net, obs):
    """Flat observation -> projected finger tokens (B, F, d), global token (B, d), the raw global vector, joint
    validity (B, J) and finger validity (B, F). As JointTransformerNet._trunk up to the encoder (no norm updates)."""
    B = obs.shape[0]
    raw = torch.index_select(obs, 1, net.token_gather).view(B, net.n_hand, net.token_dim)
    valid = torch.index_select(obs, 1, net.enabled_index) > 0.5
    glob = net.global_norm(torch.index_select(obs, 1, net.global_index))
    fingers, glob, fvalid = net._finger_tokens(net.token_norm(raw), valid, glob)
    return net.token_proj(fingers), net.global_proj(glob), glob, valid, fvalid


def _post(net, xf, xg, glob, valid, fvalid):
    """Encoder outputs -> (mu, sigma, value), as JointTransformerNet.forward."""
    if net.final_norm:
        xf, xg = net.ln_out(xf), net.ln_out(xg)
    w = fvalid.unsqueeze(-1).to(xf.dtype)
    pooled = (xf * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
    value = net.value_head(torch.cat([pooled, xg, glob], dim=-1))
    arm_out = net.arm_head(torch.cat([xg, glob], dim=-1))
    slot_mu = net.decoder(xf).reshape(xf.shape[0], -1)
    mu_hand = torch.index_select(torch.cat([slot_mu, arm_out[:, net.n_arm:]], dim=-1), 1, net.slot_of_joint)
    mu = net.mu_act(torch.cat([arm_out[:, :net.n_arm], mu_hand], dim=-1))
    sigma = mu * 0 + net.sigma_act(net.sigma)
    arm_valid = torch.ones(valid.shape[0], net.n_arm, dtype=torch.bool, device=valid.device)
    arm_valid[:, net.masked_arm_dims] = False
    act_valid = torch.cat([arm_valid, valid], dim=1)
    return (torch.where(act_valid, mu, torch.zeros_like(mu)), torch.where(act_valid, sigma, torch.zeros_like(sigma)),
            value)


def _layers(net, x, mask):
    for layer in net.layers:
        x = layer(x, mask)
    return x


def forward_padded(net, obs):
    xf, xg, glob, valid, fvalid = _pre(net, obs)
    x = torch.cat([xf, xg.unsqueeze(1)], dim=1)
    x = _layers(net, x, torch.cat([fvalid, torch.ones_like(fvalid[:, :1])], dim=1))
    return _post(net, x[:, :-1], x[:, -1], glob, valid, fvalid)


def forward_ceiling(net, obs, n_tok: int = 5):
    """Padded with only ``n_tok`` tokens per row (the first n_tok - 1 finger slots + the global): wrong outputs, the
    right cost."""
    xf, xg, glob, valid, fvalid = _pre(net, obs)
    k = n_tok - 1
    x = torch.cat([xf[:, :k], xg.unsqueeze(1)], dim=1)
    x = _layers(net, x, torch.cat([fvalid[:, :k], torch.ones_like(fvalid[:, :1])], dim=1))
    xf_out = torch.cat([x[:, :k], torch.zeros_like(xf[:, k:])], dim=1)
    return _post(net, xf_out, x[:, -1], glob, valid, fvalid)


def make_forward_bucketed(net, caps: dict[int, int]):
    """``caps``: real-finger count n -> row capacity of its group. Rows with a count not in ``caps`` are dropped
    (the capacities are sized so that does not happen; ``overflow`` checks it)."""
    counts_sorted = sorted(caps)

    def forward(obs):
        xf, xg, glob, valid, fvalid = _pre(net, obs)
        B, F, d = xf.shape
        counts = fvalid.sum(dim=1)
        order = torch.sort(counts, stable=True)[1]                       # rows by real-finger count
        size = torch.bincount(counts, minlength=F + 1)
        start = torch.cumsum(size, 0) - size
        # a zero dummy row B: unused capacity reads it and writes into it
        xf_e = torch.cat([xf, xf.new_zeros(1, F, d)])
        xg_e = torch.cat([xg, xg.new_zeros(1, d)])
        fv_e = torch.cat([fvalid, fvalid.new_zeros(1, F)])
        out_f = xf.new_zeros(B + 1, F, d)
        out_g = xg.new_zeros(B + 1, d)
        for n in counts_sorted:
            cap = caps[n]
            pos = torch.arange(cap, device=obs.device)
            rows = torch.where(pos < size[n], order[(start[n] + pos).clamp(max=B - 1)], torch.full_like(pos, B))
            # each row's n real finger slots, in slot order (a stable sort puts the real ones first)
            slots = torch.sort((~fv_e[rows]).to(torch.int8), dim=1, stable=True)[1][:, :n]
            xb = torch.cat([xf_e[rows.unsqueeze(1), slots], xg_e[rows].unsqueeze(1)], dim=1)   # (cap, n + 1, d)
            yb = _layers(net, xb, None)                                   # every token real: no mask
            out_f = out_f.index_put((rows.unsqueeze(1), slots), yb[:, :n])
            out_g = out_g.index_put((rows,), yb[:, n])
        return _post(net, out_f[:B], out_g[:B], glob, valid, fvalid)

    def overflow(obs) -> bool:
        with torch.no_grad():
            fvalid = _pre(net, obs)[4]
            size = torch.bincount(fvalid.sum(dim=1), minlength=fvalid.shape[1] + 1).tolist()
        return any(size[n] > caps.get(n, 0) for n in range(len(size)) if size[n])

    return forward, overflow


def make_forward_bucketed_flat(net, caps: dict[int, int]):
    """As make_forward_bucketed, but every gather is a 1-D ``index_select`` on the flattened (row x slot) tokens and
    the scatter back is a gather through an inverse index: index_select's backward is an index_add, where a 2-D
    advanced index / index_put backward is the generic accumulate kernel the network's own token gather avoids (8x,
    joint_transformer._register_indices)."""
    counts_sorted = sorted(caps)

    def forward(obs):
        xf, xg, glob, valid, fvalid = _pre(net, obs)
        B, F, d = xf.shape
        dev = obs.device
        counts = fvalid.sum(dim=1)
        order = torch.sort(counts, stable=True)[1]
        size = torch.bincount(counts, minlength=F + 1)
        start = torch.cumsum(size, 0) - size
        fv_e = torch.cat([fvalid, fvalid.new_zeros(1, F)])
        xf_flat = torch.cat([xf.reshape(B * F, d), xf.new_zeros(F, d)])        # the dummy row's F slots last
        xg_e = torch.cat([xg, xg.new_zeros(1, d)])
        outs_f, outs_g, idx_f, idx_g = [], [], [], []
        for n in counts_sorted:
            cap = caps[n]
            pos = torch.arange(cap, device=dev)
            rows = torch.where(pos < size[n], order[(start[n] + pos).clamp(max=B - 1)], torch.full_like(pos, B))
            slots = torch.sort((~fv_e[rows]).to(torch.int8), dim=1, stable=True)[1][:, :n]
            flat = (rows.unsqueeze(1) * F + slots).reshape(-1)                    # (cap * n,)
            xb = torch.cat([torch.index_select(xf_flat, 0, flat).view(cap, n, d),
                            torch.index_select(xg_e, 0, rows).unsqueeze(1)], dim=1)
            yb = _layers(net, xb, None)
            outs_f.append(yb[:, :n].reshape(cap * n, d))
            outs_g.append(yb[:, n])
            idx_f.append(flat)
            idx_g.append(rows)
        yf = torch.cat(outs_f + [xf.new_zeros(1, d)])                          # + a zero row for ghost slots
        yg = torch.cat(outs_g + [xg.new_zeros(1, d)])
        idx_f, idx_g = torch.cat(idx_f), torch.cat(idx_g)
        # inverse maps: every (row, slot) -> its output position, or the zero row (dummy entries land past B)
        inv_f = torch.full(((B + 1) * F,), yf.shape[0] - 1, dtype=torch.long, device=dev)
        inv_f = inv_f.scatter(0, idx_f, torch.arange(idx_f.numel(), device=dev))
        inv_g = torch.full((B + 1,), yg.shape[0] - 1, dtype=torch.long, device=dev)
        inv_g = inv_g.scatter(0, idx_g, torch.arange(idx_g.numel(), device=dev))
        out_f = torch.index_select(yf, 0, inv_f[:B * F]).view(B, F, d)
        out_g = torch.index_select(yg, 0, inv_g[:B])
        return _post(net, out_f, out_g, glob, valid, fvalid)

    return forward


BLOCK = 128                                                # FlexAttention's sparse block


def _packed_layer(layer, x, block_mask):
    """One _EncoderLayer on packed tokens (R, d): every per-token op on the real tokens only, attention through
    FlexAttention with a document mask (a token attends to its own env's tokens). Same weights, same arithmetic."""
    from torch.nn.attention.flex_attention import flex_attention
    R, d = x.shape
    H = layer.n_heads
    q, k, v = layer.qkv(layer.ln_attn(x)).chunk(3, dim=-1)
    q, k, v = (t.view(R, H, d // H).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    a = flex_attention(q, k, v, block_mask=block_mask).squeeze(0).transpose(0, 1).reshape(R, d)
    x = x + layer.proj(a)
    return x + layer.ff(layer.ln_ff(x))


def make_forward_packed(net, r_cap: int):
    """LLM-style padding-free: the real tokens of every row (its real fingers, then its global token) packed into one
    (r_cap, d) matrix, ``r_cap`` a static capacity (a multiple of BLOCK; padding tokens are each their own document).
    A row has at most F + 1 <= BLOCK tokens, so it spans at most two adjacent blocks: query block i only ever needs key
    blocks i-1, i, i+1, a block structure the same for every batch -- only the document ids change."""
    from torch.nn.attention.flex_attention import BlockMask
    assert r_cap % BLOCK == 0
    nb = r_cap // BLOCK
    dev = next(net.parameters()).device
    base = (torch.arange(nb, device=dev) - 1).clamp(0, max(nb - 3, 0))
    kv_idx = torch.zeros(1, 1, nb, nb, dtype=torch.int32, device=dev)
    kv_idx[0, 0, :, :min(3, nb)] = (base[:, None] + torch.arange(min(3, nb), device=dev)[None]).to(torch.int32)
    kv_num = torch.full((1, 1, nb), min(3, nb), dtype=torch.int32, device=dev)
    bm0 = BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=BLOCK, mask_mod=lambda b, h, q, k: q >= 0,
                                   seq_lengths=(r_cap, r_cap))

    def forward(obs):
        xf, xg, glob, valid, fvalid = _pre(net, obs)
        B, F, d = xf.shape
        T = F + 1
        tok = torch.cat([xf, xg.unsqueeze(1)], dim=1).reshape(B * T, d)
        tok = torch.cat([tok, tok.new_zeros(1, d)])                          # a zero row for padding tokens
        real = torch.cat([fvalid, torch.ones_like(fvalid[:, :1])], dim=1).reshape(-1)
        ar = torch.arange(B * T, device=obs.device)
        key = torch.sort(torch.where(real, ar, ar + B * T))[0][:r_cap]       # the real tokens first, in row order
        is_real = key < B * T
        idx = torch.where(is_real, key, torch.full_like(key, B * T))
        doc = torch.where(is_real, key // T, B + torch.arange(r_cap, device=obs.device))
        # Inductor (torch 2.7) cannot lower a mask_mod that reads a buffer computed in the same graph: break here so
        # the document ids reach the attention graph as an input.
        torch._dynamo.graph_break()

        def mask_mod(b, h, q_idx, kv_idx_):
            return doc[q_idx] == doc[kv_idx_]
        # the block structure from outside the graph (Inductor cannot lower in-graph index buffers); only the mask is new
        bm = BlockMask(bm0.seq_lengths, bm0.kv_num_blocks, bm0.kv_indices, bm0.full_kv_num_blocks, bm0.full_kv_indices,
                       bm0.q_num_blocks, bm0.q_indices, bm0.full_q_num_blocks, bm0.full_q_indices, bm0.BLOCK_SIZE,
                       mask_mod)
        x = torch.index_select(tok, 0, idx)
        for layer in net.layers:
            x = _packed_layer(layer, x, bm)
        inv = torch.full((B * T + 1,), r_cap, dtype=torch.long, device=obs.device)
        inv = inv.scatter(0, idx, torch.arange(r_cap, device=obs.device))
        out = torch.index_select(torch.cat([x, x.new_zeros(1, d)]), 0, inv[:B * T]).view(B, T, d)
        return _post(net, out[:, :F], out[:, F], glob, valid, fvalid)

    return forward


def packed_capacity(fvalid_counts: np.ndarray, rows: int, exact: bool, sigmas: float = 4.0) -> int:
    """Real tokens (fingers + the global) in ``rows`` rows: exact for the rollout, expected + ``sigmas`` sigma for a
    minibatch drawn from the population; rounded up to BLOCK."""
    t = fvalid_counts.astype(np.float64) + 1
    n = t.sum() if exact else rows * t.mean() + sigmas * math.sqrt(rows * t.var())
    return int(math.ceil(n / BLOCK) * BLOCK)


def capacities(fvalid_counts: np.ndarray, rows: int, exact: bool, sigmas: float = 4.0, grain: int = 64
               ) -> dict[int, int]:
    """Per real-finger count: the exact count (rollout: every env, fixed) or a minibatch of ``rows`` drawn from that
    population's expected count + ``sigmas`` sigma, rounded up to ``grain`` (training). At 4 sigma a group overflows
    in ~3e-5 of minibatches: a trainer would check the counts (one small host read) and fall back to the padded
    forward for that minibatch."""
    vals, freq = np.unique(fvalid_counts, return_counts=True)
    p = freq / freq.sum()
    if exact:
        return {int(v): int(c) for v, c in zip(vals, freq)}
    return {int(v): int(min(rows, math.ceil((rows * q + sigmas * math.sqrt(rows * q * (1 - q))) / grain) * grain))
            for v, q in zip(vals, p)}


# --- timing ----------------------------------------------------------------------------------------------------------

def ms(fn, iters, warm) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / iters


def time_forward(net, fwd_inf, fwd_tr, x_inf, x_tr, iters, warm) -> tuple[float, float]:
    def inf():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            fwd_inf(x_inf)
    t_inf = ms(inf, iters, warm)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = sum(t.float().square().mean() for t in fwd_tr(x_tr))
        loss.backward()
        opt.step()
    return t_inf, ms(step, iters, warm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    dev = torch.device("cuda")
    iters, warm = (5, 3) if args.quick else (20, 6)
    log = (REPO / "debug_outputs/train_logs/29oct_template_scene").glob("TS_g100_finger_L2-801730.log")
    print(f"{torch.cuda.get_device_name()}; torch {torch.__version__}", flush=True)
    rows = []
    for d_model, n_layers in ((128, 2), (128, 4)):
        torch.manual_seed(0)
        net, obs_dim = make_net(d_model, n_layers)
        net = net.to(dev).eval()
        x_inf = env_observations(net, obs_dim, N_ENVS, dev, seed=0)
        # a training minibatch: rows drawn uniformly from the rollout's envs (rl_games shuffles env x time rows)
        pick = torch.randint(0, N_ENVS, (MB,), generator=torch.Generator().manual_seed(1)).to(dev)
        x_tr = x_inf[pick] + 0.0
        counts_all = (_pre(net, x_inf)[4].sum(1)).cpu().numpy()
        frac_ghost = 1 - (counts_all.mean() + 1) / (_pre(net, x_inf)[4].shape[1] + 1)
        caps_inf = capacities(counts_all, N_ENVS, exact=True)
        caps_tr = capacities(counts_all, MB, exact=False)
        fwd_b_inf, _ = make_forward_bucketed(net, caps_inf)
        fwd_b_tr, overflow = make_forward_bucketed(net, caps_tr)
        fwd_f_inf, fwd_f_tr = make_forward_bucketed_flat(net, caps_inf), make_forward_bucketed_flat(net, caps_tr)
        r_inf, r_tr = packed_capacity(counts_all, N_ENVS, exact=True), packed_capacity(counts_all, MB, exact=False)
        r_tr_actual = int((_pre(net, x_tr)[4].sum(1) + 1).sum())
        assert r_tr_actual <= r_tr, (r_tr_actual, r_tr)
        fwd_k_inf, fwd_k_tr = make_forward_packed(net, r_inf), make_forward_packed(net, r_tr)
        assert not overflow(x_tr), "a training group outgrew its capacity"
        # 1. the pieces reproduce the network, and bucketing changes nothing (fp32, eager)
        with torch.no_grad():
            ref = net({"obs": x_inf[:4096]})
            out_p = forward_padded(net, x_inf[:4096])
            fb, _ = make_forward_bucketed(net, capacities(counts_all[:4096], 4096, exact=True))
            out_b = fb(x_inf[:4096])
            out_f = make_forward_bucketed_flat(net, capacities(counts_all[:4096], 4096, exact=True))(x_inf[:4096])
            out_k = torch.compile(make_forward_packed(net, packed_capacity(counts_all[:4096], 4096, exact=True)))(
                x_inf[:4096])
        err_p = max(float((a - b).abs().max()) for a, b in zip(ref[:3], out_p))
        err_b = max(max(float((a - b).abs().max()) for a, b in zip(out_p, o)) for o in (out_b, out_f))
        err_k = max(float((a - b).abs().max()) for a, b in zip(out_p, out_k))
        print(f"d{d_model} L{n_layers}: ghost tokens {100 * frac_ghost:.0f}% of the trunk's; padded pieces == network "
              f"to {err_p:.1e}; bucketed == padded to {err_b:.1e}; packed == padded to {err_k:.1e}; packed tokens "
              f"{r_inf} / {r_tr} (training: {r_tr_actual} real); capacities inf {caps_inf} train {caps_tr} "
              f"({sum(caps_tr.values())} rows for {MB})", flush=True)
        variants = [
            ("padded", lambda x: forward_padded(net, x), lambda x: forward_padded(net, x)),
            ("ceiling (5 tokens)", lambda x: forward_ceiling(net, x), lambda x: forward_ceiling(net, x)),
            ("bucketed, 2-D index", fwd_b_inf, fwd_b_tr),
            ("bucketed, flat index", fwd_f_inf, fwd_f_tr),
            ("packed, document mask", fwd_k_inf, fwd_k_tr),
        ]
        for mode in ("compiled", "eager"):
            for name, f_inf, f_tr in variants:
                if mode == "eager" and name.startswith("packed"):
                    continue                    # FlexAttention's eager path is a slow reference implementation
                if mode == "compiled":
                    f_inf, f_tr = torch.compile(f_inf, dynamic=False), torch.compile(f_tr, dynamic=False)
                net.train()
                t_inf, t_tr = time_forward(net, f_inf, f_tr, x_inf, x_tr, iters, warm)
                net.eval()
                r = {"config": f"d{d_model} L{n_layers}", "variant": name, "mode": mode, "inference_ms": t_inf,
                     "train_ms": t_tr, "rollout_s": HORIZON * t_inf / 1e3,
                     "update_s": UPDATES * (t_tr + LOOP_MS) / 1e3}
                rows.append(r)
                print(f"  {mode:8s} {name:20s} inference {t_inf:6.1f} ms @{N_ENVS}  train step {t_tr:6.1f} ms @{MB}"
                      f"  -> rollout {r['rollout_s']:.2f} s + update {r['update_s']:.2f} s per epoch", flush=True)
                torch._dynamo.reset()
        del net
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps({"rows": rows, "envs": N_ENVS, "minibatch": MB, "updates": UPDATES,
                                          "log": str(next(log, ""))}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
