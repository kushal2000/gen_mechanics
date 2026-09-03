"""Why the joint transformer is slow, and what to do about it.

Two separate problems, measured here rather than argued:

1. ``F.scaled_dot_product_attention`` launches a grid indexed by
   ``batch * n_heads``, and CUDA caps a grid dimension at 65535. With 4 heads
   the joint transformer therefore dies with "invalid configuration argument"
   at any minibatch >= 16384 * 4 -- i.e. exactly when we try to use minibatch
   size as a throughput lever. A 23-token sequence does not need a fused
   attention kernel at all; the explicit matmul form has no such limit.

2. At d_model 128 and 23 tokens the layer is bandwidth-bound on its
   LayerNorm / GELU / residual chain, not FLOP-bound, so operator fusion
   (torch.compile) is worth more than any algebraic change to attention.

    .venv_isaacsim/bin/python \
      experiments/decentralized_control/profile_transformer_attention.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SdpaLayer(nn.Module):
    """The current ``_EncoderLayer``, verbatim."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def attend(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x):
        batch, tokens, dim = x.shape
        heads = self.n_heads
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, tokens, heads, dim // heads).transpose(1, 2)
            for t in (q, k, v)
        )
        attn = self.attend(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, tokens, dim))
        return x + self.ff(self.ln_ff(x))


class MatmulLayer(SdpaLayer):
    """Same layer with attention written out.

    At 23 tokens the score matrix is (B, heads, 23, 23) -- 2 KiB per head per
    sample, and the thing SDPA's fused kernels exist to avoid materializing is
    not a problem at this size. Writing it out removes the grid-dimension
    ceiling on the batch.
    """

    def attend(self, q, k, v):
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return scores.softmax(dim=-1) @ v


def bench(layer, batch, tokens, d_model, device, iters, warmup, backward=True):
    x = torch.randn(batch, tokens, d_model, device=device, requires_grad=backward)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16)

    def iteration():
        with autocast:
            out = layer(x)
            loss = out.square().mean()
        if backward:
            layer.zero_grad(set_to_none=True)
            loss.backward()

    for _ in range(warmup):
        iteration()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(iters):
        iteration()
    torch.cuda.synchronize(device)
    ms = 1e3 * (time.perf_counter() - start) / iters
    return ms, torch.cuda.max_memory_allocated(device) / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="16384,32768,65536,98304,114688")
    parser.add_argument("--tokens", type=int, default=23)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    batches = [int(v) for v in args.batches.split(",") if v.strip()]

    print(f"one encoder layer, tokens={args.tokens} d_model={args.d_model} "
          f"heads={args.n_heads} fp16 autocast, fwd+bwd\n", flush=True)
    header = f"{'batch':>8}  {'SDPA':>18}  {'matmul':>18}  {'matmul+compile':>18}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for batch in batches:
        cells = []
        for name, cls, compile_it in (
            ("sdpa", SdpaLayer, False),
            ("matmul", MatmulLayer, False),
            ("matmul_compiled", MatmulLayer, True),
        ):
            layer = cls(args.d_model, args.n_heads, args.ff_mult).to(device).train()
            if compile_it:
                layer = torch.compile(layer)
            iters = max(4, round(args.iters * batches[0] / batch))
            try:
                ms, peak = bench(layer, batch, args.tokens, args.d_model,
                                 device, iters, args.warmup)
                cells.append(f"{ms:8.2f} ms {peak:6.0f}M")
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                first = str(exc).strip().splitlines()[0][:16]
                cells.append(f"{'FAIL: ' + first:>18}")
            del layer
            torch.cuda.empty_cache()
        print(f"{batch:>8,}  " + "  ".join(cells), flush=True)

    print("\nequivalence check (fp32, batch 4096):", flush=True)
    torch.manual_seed(0)
    ref = SdpaLayer(args.d_model, args.n_heads, args.ff_mult).to(device).eval()
    alt = MatmulLayer(args.d_model, args.n_heads, args.ff_mult).to(device).eval()
    alt.load_state_dict(ref.state_dict())
    x = torch.randn(4096, args.tokens, args.d_model, device=device)
    with torch.no_grad():
        a, b = ref(x), alt(x)
    print(f"  max abs diff {(a - b).abs().max().item():.3e} "
          f"(allclose atol=1e-5: {torch.allclose(a, b, atol=1e-5)})", flush=True)


if __name__ == "__main__":
    main()
