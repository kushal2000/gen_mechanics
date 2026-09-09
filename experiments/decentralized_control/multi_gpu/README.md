# Hybrid architecture profiling

Submit from any working directory:

```bash
sbatch /share/portal/kk837/gen_mechanics/experiments/decentralized_control/multi_gpu/profile_2gpu.sub
sbatch /share/portal/kk837/gen_mechanics/experiments/decentralized_control/multi_gpu/profile_4gpu.sub
```

These are synthetic actor profiling jobs, not RL learning runs. Each compares
the MLP, two-layer one-head transformers at widths 64/128/256/384, and hybrid
transformers at widths 64/128. Both explicit attention and PyTorch SDPA are
measured with compiled FP16 autocast. The hybrid uses a large shared context
MLP over global features and mean-pooled nonlinear joint embeddings; its
context conditions both attention layers and the shared action head.

The default **global** batch is 16384, divided across DDP ranks. Override with
`GLOBAL_BATCH=32768 sbatch ...`. Backward timings include gradient all-reduce;
forward timings use the local batch. Reported times and extra peak allocated
memory are the maximum across ranks; memory is per GPU, not summed. FLOPs and
effective TFLOP/s are global. These omit PPO, critic, optimizer and GradScaler.
The loss is synthetic, so timings do not establish learning quality.

JSON results go to a unique directory under `debug_outputs/bench_logs/`.
Slurm stdout/stderr go to the submission directory. A matching single-GPU run:

```bash
python experiments/decentralized_control/profile_hybrid_transformer.py --attention explicit --output /tmp/hybrid_explicit.json
python experiments/decentralized_control/profile_hybrid_transformer.py --attention sdpa --output /tmp/hybrid_sdpa.json
```

The benchmark checks FP32 permutation equivariance, arm invariance, output
shapes, and finite gradients before timing. Multi-GPU jobs require one node
with 2 or 4 GPUs; scripts have not been submitted automatically.
