"""Padded vs packed trunk for a multi-embodiment batch of finger tokens (README.md). Pure torch.

    .venv_isaacsim/bin/python experiments/29oct_transformer_speed/bench_packing.py

Every env has T_max = 8 token slots (6 fingers + a palm-cut slot + the global token) but only its real ones carry
anything. Token counts per env are drawn from the 100 random grammar hands (2-6 fingers, 55% with a cut): 5.5 real
tokens of 8 on average. Three trunks with identical arithmetic per real token, a pre-LN encoder stack
(LN -> QKV -> attention -> proj -> residual -> LN -> FF -> residual):

  padded   (B, 8, d) dense, ghost tokens masked as keys; the per-token work runs on every slot
  packed   the per-token work (LayerNorms, QKV, proj, FF) runs on the (sum of real tokens, d) rows only; for
           attention the packed q, k, v are scattered to the padded layout, SDPA runs with the mask, and the
           result is gathered back (two small copies of a d-wide tensor; the FF's 4d-wide tensors never pad)
  nested   torch nested tensors (jagged layout) end to end, if this torch supports every op on them

Timed: inference at 24,576 envs (fp16 autocast, no grad) and forward + backward at 16,384 (the 14 Oct minibatch).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_MAX, N_INF, N_TR = 8, 24576, 16384
FINGERS = {2: 20, 3: 23, 4: 19, 5: 20, 6: 18}            # the 100 random grammar hands
P_CUT = 0.55


def sample_lengths(n: int, rng: np.random.Generator) -> torch.Tensor:
    f = rng.choice(list(FINGERS), size=n, p=np.array(list(FINGERS.values())) / 100.0)
    cut = rng.random(n) < P_CUT
    return torch.as_tensor(f + cut + 1, dtype=torch.long)       # + the global token


class Layer(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def attend(self, q, k, v, mask):
        B, T, d = q.shape
        q, k, v = (t.view(B, T, self.h, d // self.h).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :] if mask is not None else None)
        return a.transpose(1, 2).reshape(B, T, d)

    def forward_padded(self, x, mask):
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        x = x + self.proj(self.attend(q, k, v, mask))
        return x + self.ff(self.ln2(x))

    def forward_packed(self, x, rows, real, mask):
        # x: (R, d) packed real tokens; rows: (B*T_MAX,) index into x (ghost slots -> a zero row R);
        # real: (R,) the flat padded positions of the real tokens. index_select both ways: its backward is an
        # index_add, where a boolean-mask gather's backward is a scatter with a host sync.
        B, T = mask.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        zero = x.new_zeros(1, x.shape[-1])
        qp, kp, vp = (torch.index_select(torch.cat([t, zero]), 0, rows).view(B, T, -1) for t in (q, k, v))
        a = torch.index_select(self.attend(qp, kp, vp, mask).reshape(B * T, -1), 0, real)
        x = x + self.proj(a)
        return x + self.ff(self.ln2(x))

    def forward_nested(self, x):
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        d = x.size(-1)
        q, k, v = (t.unflatten(-1, (self.h, d // self.h)).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(-2)
        x = x + self.proj(a)
        return x + self.ff(self.ln2(x))


class Trunk(nn.Module):
    def __init__(self, d: int, layers: int, heads: int, ff_mult: int, token_dim: int = 100):
        super().__init__()
        self.inp = nn.Linear(token_dim, d)
        self.layers = nn.ModuleList(Layer(d, heads, ff_mult) for _ in range(layers))
        self.out = nn.Linear(d, 5)                                           # the per-finger decoder

    def padded(self, tok, mask):
        x = self.inp(tok)
        for l in self.layers:
            x = l.forward_padded(x, mask)
        return self.out(x)

    def packed(self, tok_packed, rows, real, mask):
        x = self.inp(tok_packed)
        for l in self.layers:
            x = l.forward_packed(x, rows, real, mask)
        return self.out(x)

    def nested(self, tok_nt):
        x = self.inp(tok_nt)
        for l in self.layers:
            x = l.forward_nested(x)
        return self.out(x)


def ms(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / iters


def inputs(n, token_dim, rng, dev):
    L = sample_lengths(n, rng).to(dev)
    mask = torch.arange(T_MAX, device=dev)[None, :] < L[:, None]           # (B, T) real slots
    tok = torch.randn(n, T_MAX, token_dim, device=dev) * mask[..., None]
    R = int(mask.sum())
    rows = torch.full((n, T_MAX), R, device=dev, dtype=torch.long)         # ghosts -> the zero row
    rows[mask] = torch.arange(R, device=dev)
    tok_packed = tok[mask]                                                  # (R, token_dim)
    real = mask.flatten().nonzero().squeeze(1)                              # flat padded positions of real tokens
    nt = torch.nested.nested_tensor([tok[i, :int(L[i])] for i in range(n)], layout=torch.jagged, device=dev)
    return tok, mask, tok_packed, rows.flatten(), real, nt, float(mask.float().mean())


def main():
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    print(f"{torch.cuda.get_device_name()}; torch {torch.__version__}; T_max {T_MAX}")
    for d, layers in ((128, 4), (256, 4), (256, 2)):
        torch.manual_seed(0)
        net = Trunk(d, layers, 1, 4).to(dev)
        res = {}
        for phase, n in (("inference", N_INF), ("train", N_TR)):
            tok, mask, tokp, rows, real, nt, util = inputs(n, 100, rng, dev)
            if phase == "inference":
                def run(fn):
                    def f():
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            fn()
                    return f
            else:
                def run(fn):
                    def f():
                        net.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16):
                            out = fn()
                        (out.float().square().mean() if not out.is_nested else
                         out.values().float().square().mean()).backward()
                    return f
            res[(phase, "padded")] = ms(run(lambda: net.padded(tok, mask)))
            res[(phase, "packed")] = ms(run(lambda: net.packed(tokp, rows, real, mask)))
            try:
                res[(phase, "nested")] = ms(run(lambda: net.nested(nt)))
            except Exception as e:                                          # noqa: BLE001
                res[(phase, "nested")] = float("nan")
                err = f"{type(e).__name__}: {str(e)[:90]}"
        # equality of padded and packed outputs on the real slots
        with torch.no_grad():
            tok, mask, tokp, rows, real, nt, _ = inputs(512, 100, rng, dev)
            diff = float((net.padded(tok, mask)[mask] - net.packed(tokp, rows, real, mask)).abs().max())
        print(f"d{d} L{layers}: real tokens {100*util:.0f}% of slots; packed == padded on real slots to {diff:.1e}")
        for phase in ("inference", "train"):
            p, k, nn_ = res[(phase, "padded")], res[(phase, "packed")], res[(phase, "nested")]
            print(f"   {phase:9s} padded {p:6.1f} ms   packed {k:6.1f} ms ({p / k:4.2f}x)   nested "
                  + (f"{nn_:6.1f} ms ({p / nn_:4.2f}x)" if nn_ == nn_ else f"n/a ({err})"))


if __name__ == "__main__":
    main()
