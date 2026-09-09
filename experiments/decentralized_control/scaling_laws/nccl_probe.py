import os
from datetime import timedelta
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
print(f'rank={rank} init', flush=True)
dist.init_process_group('nccl', timeout=timedelta(seconds=10))
x = torch.ones(32, device='cuda')
print(f'rank={rank} broadcast start', flush=True)
dist.broadcast(x, 0)
torch.cuda.synchronize()
print(f'rank={rank} broadcast complete', flush=True)
dist.destroy_process_group()
