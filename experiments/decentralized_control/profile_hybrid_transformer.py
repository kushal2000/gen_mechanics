"""Capacity/throughput comparison; synthetic actor only, no PPO or simulator.

python experiments/decentralized_control/profile_hybrid_transformer.py --output /tmp/hybrid.json
All transformers use two layers and one head. Global context uses a mean
of nonlinear joint embeddings, never flattened ordered joint observations.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from profile_mlp_vs_transformer import (
    ACT_DIM, GLOBAL_DIM, N_ARM, N_HAND, OBS_DIM, TOKEN_DIM,
    DenseMLP, EncoderLayer, JointTransformer, _mlp, measure, report,
)


class HybridTransformer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.token_proj = nn.Sequential(nn.Linear(TOKEN_DIM, d_model), nn.ELU())
        self.context = _mlp(GLOBAL_DIM + d_model, [1024, 1024, 1024], 128)
        self.condition = nn.ModuleList(nn.Linear(128, d_model) for _ in range(3))
        self.layers = nn.ModuleList(EncoderLayer(d_model, 1, 2) for _ in range(2))
        self.ln_out = nn.LayerNorm(d_model)
        self.mu_head = _mlp(d_model, [64], 1)
        self.arm_head = _mlp(128 + d_model + GLOBAL_DIM, [256, 128], N_ARM)

    def forward(self, obs):
        joints = obs[:, :N_HAND * TOKEN_DIM].reshape(-1, N_HAND, TOKEN_DIM)
        glob = obs[:, N_HAND * TOKEN_DIM:]
        x = self.token_proj(joints)
        context = self.context(torch.cat([glob, x.mean(dim=1)], dim=-1))
        for condition, layer in zip(self.condition, self.layers):
            x = layer(x + condition(context).unsqueeze(1))
        x = self.ln_out(x) + self.condition[-1](context).unsqueeze(1)
        hand = self.mu_head(x).squeeze(-1)
        arm = self.arm_head(torch.cat([context, x.mean(dim=1), glob], dim=-1))
        return torch.cat([hand, arm], dim=-1)


def sdpa_forward(self, x):
    b, t, d = x.shape
    q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
    q, k, v = (z.reshape(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
               for z in (q, k, v))
    attn = F.scaled_dot_product_attention(q, k, v)
    x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
    return x + self.ff(self.ln_ff(x))


def check_and_count(net, equivariant):
    """Check actual forward shapes, permutation behavior and finite gradients.

    Count Linear and attention matrix FLOPs; excludes norms/activations, as
    the original benchmark does. Hooks are removed before compilation.
    """
    x = torch.randn(4, OBS_DIM)
    flops = 0

    def linear_hook(module, inputs, output):
        nonlocal flops
        flops += 2 * output.numel() * module.in_features // len(x)

    def attn_hook(module, inputs, output):
        nonlocal flops
        _, t, d = inputs[0].shape
        flops += 4 * t * t * d

    handles = []
    for module in net.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, EncoderLayer):
            handles.append(module.register_forward_hook(attn_hook))
    y = net(x)
    for handle in handles:
        handle.remove()
    assert y.shape == (4, ACT_DIM)
    if equivariant:
        perm = torch.randperm(N_HAND)
        shuffled = torch.cat([
            x[:, :N_HAND * TOKEN_DIM].reshape(4, N_HAND, TOKEN_DIM)[:, perm].flatten(1),
            x[:, N_HAND * TOKEN_DIM:]], dim=-1)
        expected = torch.cat([y[:, :N_HAND][:, perm], y[:, N_HAND:]], dim=-1)
        torch.testing.assert_close(net(shuffled), expected, atol=2e-6, rtol=2e-5)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    net.zero_grad(set_to_none=True)
    return flops


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', type=int, default=16384)
    p.add_argument('--compile', type=int, default=1)
    p.add_argument('--attention', choices=['explicit', 'sdpa'], default='explicit')
    p.add_argument('--output')
    args = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
    if args.batch % world:
        raise ValueError('Global batch must be divisible by WORLD_SIZE')
    local_batch = args.batch // world
    torch.set_num_threads(4)
    if args.attention == 'sdpa':
        EncoderLayer.forward = sdpa_forward
    torch.manual_seed(0)
    factories = [('MLP', lambda: DenseMLP([1024, 1024, 512, 512]))]
    for d in (64, 128, 256, 384):
        factories.append((f'JT d{d} h1 ff2 L2', lambda d=d: JointTransformer(d, 1, 2, 2, [64], [256, 128])))
    for d in (64, 128):
        factories.append((f'Hybrid d{d} h1 ff2 L2', lambda d=d: HybridTransformer(d)))
    rows = []
    metadata = dict(batch=args.batch, local_batch=local_batch, world_size=world,
                    precision='fp16', compile=bool(args.compile),
                    attention=args.attention, device=torch.cuda.get_device_name(),
                    torch_version=torch.__version__)
    if rank == 0:
        print(metadata, flush=True)
    for name, factory in factories:
        net = factory()
        flops = check_and_count(net, name != 'MLP') * args.batch
        params = sum(p.numel() for p in net.parameters())
        context_params = sum(p.numel() for p in net.context.parameters()) if isinstance(net, HybridTransformer) else 0
        compile_in_measure = bool(args.compile)
        if world > 1:
            net = net.cuda()
            if args.compile:
                net = torch.compile(net, dynamic=False)
            net = DistributedDataParallel(net, device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
            compile_in_measure = False
            dist.barrier()
        act, fwd, both = measure(net, local_batch, 'cuda', iters=50,
                                 compile_net=compile_in_measure, precision='fp16')
        if world > 1:
            stats = torch.tensor([act, fwd, both], device='cuda')
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            act, fwd, both = stats.tolist()
        row = dict(name=name, params=params, context_params=context_params, flops=flops,
                   act=act, t_fwd=fwd, t_both=both, tflops=flops / (fwd * 1e-3) / 1e12)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
        if args.output and rank == 0:
            with open(args.output, 'w') as f:
                json.dump(dict(metadata=metadata, rows=rows), f, indent=2)
        del net
        torch.cuda.empty_cache()
    if rank == 0:
        report(rows)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
