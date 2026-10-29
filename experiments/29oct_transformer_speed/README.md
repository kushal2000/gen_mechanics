# 29 Oct: what makes the joint transformer slower than the MLP, and how to get a fast one with many parameters

Started 29 Oct 2026 on branch `transformer_multi_embodiment`. Pure network profiling, no simulator: `bench.py` builds
the real networks on the real layout (hand `coarse_s0_000`: 20 hand joints + 6 wrist joints = 21 tokens, the
olympics observation list, 471 values, 26 actions) and times one rollout inference at 24,576 envs and one train step
at each learner's minibatch, both under fp16 autocast, on one RTX 6000 Ada. `results.json` holds every number;
`analysis/plot.py` draws `plots/`.

Measured in training (27-28 Oct, hand 000, same GPU type): the MLP epoch is ~3.1 s (127k samples/s), the 14 Oct
transformer's ~10.9 s (36k). The simulation alone is 1.85 s of either (212k sim steps/s x 16 x 24,576).

## Cost model

An epoch is 393,216 samples: 16 rollout steps of 24,576 envs, then `mini_epochs` passes over the SAPG-augmented
458,752 rows in minibatches.

    epoch = sim + 16 x inference(24,576) + mini_epochs x (458,752 / minibatch) x train_step(minibatch)

The MLP runs its learner (minibatch 114,688, 5 mini-epochs: 20 train steps), the transformer the 14 Oct one
(16,384, 2: 56 train steps). The model under-counts the real loop (rl_games' dataset building, SAPG's batch
augmentation, KL, the obs copies) by a constant few seconds, so read the differences, not the absolutes.

Why the transformer costs more per sample, before any measurement: the MLP is one matrix chain per sample
(~2.1M multiply-accumulates). The transformer runs its 4 layers over 21 tokens per sample: 12 d^2 per token per layer
(4 d^2 attention projections + 8 d^2 feed-forward at ff x4) = 197k at d = 128, x 21 x 4 = 16.5M, plus the heads.
Parameters, though, are counted once, not per token: the trunk's 0.8M parameters do 16.5M MACs per sample, the heads'
1.0M do 1.0M. That ratio -- 21 MACs per parameter per sample in the trunk against 1 in a per-sample MLP -- is the
whole story of "slow with few parameters".

## Results (hand 000; fp16 autocast; one RTX 6000 Ada, no sim running on it)

| Configuration | Params (trunk / heads) | MACs/sample | Inference 24,576 | Train step | Modelled epoch | Samples/s |
|---|---|---|---|---|---|---|
| **MLP [1024, 1024, 512, 512]** | 2.33M (2.32 / 0.01) | 2.3M | **1.1 ms** | 16.8 ms @114,688 | **2.2 s** | 178k |
| **transformer d128 L4 (today's)** | 1.84M (0.81 / 1.03) | 17.6M | 41.4 ms | 88.1 ms @16,384 | **7.4 s** | 53k |
| d128 L0 (no layers: the plumbing floor) | 1.04M (0.01 / 1.03) | 1.1M | 4.9 | 9.5 | 2.5 | 160k |
| d128 L1 / L2 / L8 | 1.24 / 1.44 / 2.63M | 5.2 / 9.3 / 34.1M | 14 / 23 / 77 | 32 / 50 / 163 | 3.8 / 5.0 / 12.2 | 102 / 78 / 32k |
| d32 / d64 / d256 L4 | 0.89 / 1.11 / 4.48M | 1.9 / 5.1 / 67.5M | 14 / 24 / 84 | 32 / 50 / 166 | 3.9 / 5.0 / 12.5 | 102 / 78 / 31k |
| d512 L2 | 8.17M (6.35 / 1.82) | 134M | 117 | 213 | 15.6 | 25k |
| d256 L2 | 2.90M (1.60 / 1.29) | 34.4M | 46 | 93 | 7.8 | 51k |
| d128 L4, 4 heads | 1.84M | 17.6M | 52 | 102 | 8.4 | 47k |
| d128 L4, ff x2 | 1.57M (0.54 / 1.03) | 12.1M | 36 | 78 | 6.8 | 58k |
| d128 L4, parallel block | 1.84M | 17.6M | 39 | 115 | 8.9 | 44k |
| d128 L4, no affine LayerNorm | 1.84M | 17.6M | 41 | 78 | 6.9 | 57k |
| d128 L4, small heads [256, 128] | 1.01M (0.81 / 0.20) | 16.8M | 41 | 88 | 7.4 | 53k |
| d128 L4, **big heads [2048, 1024]** | **6.09M** (0.81 / 5.29) | 21.9M | 44 | 92 | **7.7** | 51k |
| d128 L4, per-joint head [256] | 1.87M | 18.3M | 43 | 91 | 7.6 | 52k |
| d128 L4, per-joint head [1024, 512] | 2.49M (0.81 / 1.69) | 30.7M | 51 | 105 | 8.5 | 46k |
| d128 L4, minibatch 57,344 | 1.84M | 17.6M | 41 | 312 @57,344 | 7.5 (24 GB) | 52k |
| d128 L4 **compiled** | 1.84M | 17.6M | 26 | 56 | **5.4** | 73k |
| d128 L4 compiled + parallel + no affine | 1.83M | 17.6M | 24 | 49 | 5.0 | 79k |
| d128 L4 ff x2 compiled + no affine | 1.57M | 12.1M | 21 | 45 | 4.7 | 84k |
| d256 L4 compiled | 4.48M | 67.5M | 63 | 117 | 9.4 | 42k |
| **candidate A:** d128 L4, heads [2048, 1024], per-joint head [256], compiled, no affine | **6.12M** (0.80 / 5.32) | 22.5M | 29 | 60 | **5.7** | 69k |
| **candidate B:** d256 L2, heads [2048, 1024], compiled, no affine | **7.67M** (1.60 / 6.07) | 39.2M | 35 | 91 | 7.5 | 52k |

Modelled against measured: the MLP's real epoch is 3.1 s (model 2.2), today's transformer's 10.9 s (model 7.4): the
loop adds ~1 s to the MLP and ~3.5 s to the transformer (rl_games' dataset building over 56 minibatches, SAPG's
augmentation, the KL pass). Read the table's differences; add ~3 s for the transformer's wall clock.

### What the numbers say

1. **Per sample the transformer costs 37x the MLP, for 7.6x the arithmetic.** Inference: 41 ms vs 1.1 ms at 24,576
   envs. Training: 186k rows/s vs 6.8M. The arithmetic ratio is only 7.6x (17.6M vs 2.3M MACs), so most of the gap
   is GPU under-use: d = 128 matmuls over 21 tokens are small, launch- and bandwidth-bound, where the MLP's
   1024-wide chain is compute-bound. Evidence: d256 does 3.8x the MACs of d128 in 1.9x the time; d512 L2 7.6x in
   2.4x. A wider trunk is more efficient per parameter than a deeper one.
2. **The update dominates, not the rollout.** Today's transformer spends 4.9 s of its modelled 7.4 s in the 56 train
   steps per epoch and 0.7 s in the 16 rollout inferences; the MLP's 20 big steps take 0.3 s. A bigger minibatch does
   not help (57,344: the same 5.0 s of update, 24 GB), because the total rows per epoch are fixed; only fewer passes
   would, and 28 Oct showed that fewer, larger updates are what stalls learning.
3. **The attention layers are 85% of the network.** L0 (gathers, normaliser, projections, heads) is 2.5 s; each
   d128 layer adds ~1.2 s. Width and depth cost what the arithmetic predicts: d256 L4 and d128 L8 both ~12.3 s.
4. **Heads are almost free.** Five million more parameters in the wrist and value heads (per-sample MLPs) cost 0.3 s
   per epoch; a per-joint MLP head [256] costs 0.2 s. These run once per sample (or once per token, narrow), not 21 x
   layers times.
5. **The plumbing cuts are real:** compile 1.4x (7.4 -> 5.4 s), compile + no affine LayerNorm 5.0 s, with ff x2 4.7 s.
   4 heads and the parallel block are slower in eager mode. No affine norm and the parallel block change the model
   slightly (the docstrings say how), so they are for new runs, not for resuming.

## What "fast with many parameters" means here

The trunk's parameters each do 21 x (layers) multiply-accumulates per sample; a head's parameter does one. So
parameters are cheap in the per-sample heads and in a per-joint head, and expensive in the attention trunk, and the
trunk gets cheaper per parameter the wider it is.

- **Candidate A** (d128 L4, heads [2048, 1024], per-joint head [256], compiled, no affine): 6.1M parameters, 3.3x
  today's, at 5.7 s per epoch against today's 7.4 -- bigger and faster. The extra parameters sit in the wrist and value
  heads and a small per-joint head; the finger pathway's trunk is unchanged.
- **Candidate B** (d256 L2, heads [2048, 1024], compiled, no affine): 7.7M parameters, 4.2x, at 7.5 s -- today's
  speed with a 2x wider trunk, which doubles the global token the fingers read the task through (H1).
- The untouched lever is the token count: cost is tokens x layers x d^2, and 21 tokens is fixed by one-token-per-joint.
  Fewer, richer tokens (one per finger) would be a different policy, not a setting.
- The learner's 56 small steps per epoch are the other half of the time, and they are there because the 27 Oct
  debugging found 8 large steps per epoch stall learning. 1 mini-epoch (28 steps) is untested.

The whole study is pure-network timing on random observations. Whether candidate A or B learns as well as today's
transformer is a training run, not a benchmark; A is the safer bet (same trunk), B the one that addresses the
bottleneck the debugging pointed at.

## Round 2: the levers, measured

### In the real loop (hand 000, 24,576 envs, 12 epochs each, today's 21-joint-token transformer)

| Variant | Epoch | Steps/s |
|---|---|---|
| baseline | 10.1 s | 39k |
| fp16 rollouts (`rollout_autocast`, new, opt-in in rl_games) | 8.5 s | 46k |
| compiled | 7.8 s | 51k |
| compiled + fp16 rollouts | **6.1 s** | 65k |
| + no affine norm + fused attention | 6.0 s | 65k |
| + 1 mini-epoch instead of 2 (28 updates per epoch) | 4.4 s | 90k |
| MLP, same env (action delay off) | 2.6 s | 152k |
| MLP at minibatch 16,384 (140 steps per epoch) | 2.9 s | 137k |

rl_games ran every rollout forward in fp32 (no autocast in `get_action_values`): 102 vs 47 ms per step for this
network, 0.05 s per epoch for the MLP. The loop's per-minibatch overhead is ~2 ms (the MLP at 140 steps costs 0.3 s
more than at 20), so the transformer's loop-versus-model gap is the network itself, not the step count.

### Attention
- Fused attention (`attention: sdpa`, torch's kernel): nothing at 21 tokens (attention is <3% of the trunk) -- and
  under `torch.compile` it BREAKS the graph: 53 vs 26 ms per train step at width 256 x 2 layers, 2.3x the memory.
  Keep the explicit attention when compiling.
- Heads add no parameters and no arithmetic (the projections are the same size) and cost 8-13% per epoch (h smaller
  kernels; at 4 heads the fused kernel's grid limit bites). A learning question only.
- A kinematic-graph mask would save nothing: the scores are 21 x 21 per sample; the per-token work is what costs.
- Compile modes: max-autotune 5.0 s vs default 5.4 (modelled, width 128 x 4); CUDA graphs ("reduce-overhead") nothing.

### Padded vs packed finger tokens (`bench_packing.py`; 69% of the 8 slots real for the 100 random hands)

| Trunk | Inference: packed / nested vs padded | Train step: packed / nested vs padded |
|---|---|---|
| width 128, 4 layers | 1.12x / 0.76x | 0.99x / 0.34x |
| width 256, 4 layers | 1.06x / 1.41x | 1.05x / 1.47x |
| width 256, 2 layers | 1.05x / 1.41x | 1.05x / 1.16x |

Hand-rolled packing (per-token work on real rows, scatter/gather around attention) recovers almost nothing: the two
copies cost what the ghost rows cost. torch's jagged nested tensors recover the ghost share at width 256 (1.4x) but
lose at 128 and cannot go through `torch.compile` (worth 1.4x itself). Build with padding; packing is a later
optimisation worth ~10-15% of an epoch for the multi-embodiment run.

### Parity with the MLP on the per-finger decomposition (the mock trunk: 7 hand tokens of 100 features + the global
token, compiled, explicit attention, no affine norm; the MLP: 2.33M params, 2.2 s modelled)

| Candidate | Params (trunk / heads) | MACs/sample | Inference 24,576 | Train step | Modelled epoch | Samples/s |
|---|---|---|---|---|---|---|
| **width 128, 2 layers, ff x4, heads [2048, 1024] / [1024, 512]** | **3.79M** (0.42 / 3.38) | 6.6M | 6.7 ms | 14.6 ms | **2.8 s** | 142k |
| width 128, 3 layers, ff x4 | 1.65M (0.62 / 1.03) | 5.9M | 7.8 | 16.6 | 2.9 | 136k |
| width 192, 2 layers, ff x4 | 2.08M (0.92 / 1.16) | 8.4M | 8.7 | 19.4 | 3.1 | 128k |
| width 256, 2 layers, ff x2 | 2.39M (1.10 / 1.29) | 9.9M | 9.6 | 21.2 | 3.2 | 123k |
| **width 256, 2 layers, ff x4** | **2.92M** (1.62 / 1.29) | 14.1M | 12.6 | 25.8 | **3.5 s** | 112k |
| width 256, 3 layers, ff x2 | 2.92M (1.62 / 1.29) | 14.1M | 13.7 | 28.2 | 3.6 | 108k |
| width 256, 2 layers, ff x4, big heads | 5.52M (1.62 / 3.90) | 16.7M | 13.9 | 28.0 | 3.6 | 108k |
| width 384, 2 layers, ff x2 | 3.99M (2.43 / 1.56) | 20.7M | 16.8 | 32.6 | 3.9 | 100k |
| today's network (21 joint tokens, width 128 x 4, eager) | 1.84M | 17.6M | 41.4 | 88.1 | 7.4 | 53k |

The MLP's network share of the modelled epoch is 0.35 s (sim 1.85); the first row's is 0.95 s, the width-256 row's
1.6 s. The per-finger decomposition with two attention layers lands within 1.3-1.6x of the MLP's epoch at 1.3-1.6x
its parameters -- against 3.4x for today's transformer. Parameters are cheapest in the per-sample heads: the
width-128 row carries 3.4M of its 3.8M there for 0.3 s; the width-256 row spends its budget on a trunk at the GPU's
ridge point, which is the row that widens the global token the fingers read the task through.

### The floor with 56 updates per epoch (the learner fixed): 6 finger tokens + the global token

The palm-cut joint folds into the global token (it is part of the palm, like the wrist; its action comes from the
wrist head), so 7 tokens, not 8. Compiled, explicit attention, no affine norm:

| Configuration | Params | Train step | Modelled epoch | vs MLP |
|---|---|---|---|---|
| no attention layers, big heads (the plumbing floor) | 3.40M | 5.5 ms | 2.2 s | 1.0x |
| width 64 x 2, big heads | 3.22M | 8.1 ms | 2.4 s | 1.09x |
| width 128 x 2, default heads | 1.45M | 10.9 ms | 2.5 s | 1.14x |
| **width 128 x 2, big heads** | **3.79M** | 12.9 ms | **2.7 s** | **1.23x** |
| width 256 x 2 | 2.92M | 23.0 ms | 3.3 s | 1.5x |

The plumbing plus 3.4M parameters of heads costs exactly the MLP's epoch; everything above it is the attention
layers. Minibatch size does nothing for the per-row cost here either (256 x 2: 3.5 s at 16,384, 57,344 and 114,688;
the train step scales exactly with rows), so a larger minibatch would only buy fewer updates.

### The fastest two-layer trunks with default heads (7 tokens, compiled, no affine)

| Width x layers, ff, heads | Params | Train step | Modelled epoch | vs MLP | Samples/s |
|---|---|---|---|---|---|
| 64 x 2, ff x2, 1 head | 0.98M | 5.6 ms | 2.2 s | 1.00x | 179k |
| 64 x 2, ff x4, 1 head | 1.01M | 6.7 ms | 2.3 s | 1.05x | 174k |
| 96 x 2, ff x4, 1 head | 1.21M | 8.7 ms | 2.4 s | 1.09x | 164k |
| 128 x 2, ff x2, 1 head | 1.32M | 9.1 ms | 2.4 s | 1.09x | 162k |
| 128 x 2, ff x4, 1 head | 1.45M | 10.9 ms | 2.5 s | 1.14x | 155k |
| **128 x 2, ff x4, 4 heads** | **1.45M** | 11.9 ms | **2.6 s** | **1.18x** | 151k |
| 128 x 2, ff x4, 4 heads, eager | 1.45M | 18.5 ms | 3.0 s | 1.36x | 130k |

## Recommendation (29 Oct, final -- the user's calls: default heads, speed over parameters, 4 attention heads)

**The per-finger transformer, width 128, 2 layers, ff x4, 4 attention heads, default heads, 7 tokens**: 6 finger
slots (5 joint slots x 19 features + 5 validity bits each) + the global token with the wrist and the palm-cut joint
in it; a shared decoder MLP [256] -> 5 per finger token (the palm-cut action from the wrist head); wrist head
[1024, 512], value head [512, 256]; compiled with explicit attention (the fused kernel breaks the compiled graph and
cannot run 4 heads at 16,384 anyway), no affine norm, fp16 rollouts; the 14 Oct learner unchanged (minibatch
16,384, 2 mini-epochs, lr 1e-4, 56 updates per epoch); ghost fingers, ghost slots and the locked wrist masked.

| | MLP [1024, 1024, 512, 512] | per-finger 128 x 2, 4 heads | today's joint-token 128 x 4 |
|---|---|---|---|
| parameters | 2.33M | 1.45M (0.6x) | 1.84M |
| MACs / sample | 2.3M | 3.9M (1.7x) | 17.6M |
| inference at 24,576 envs | 1.1 ms | 5.9 ms | 41 ms |
| train step at 16,384 rows | 2.4 ms (16.8 per 114,688) | 11.9 ms | 88 ms |
| modelled epoch | 2.2 s | **2.6 s (1.18x)** | 7.4 s (3.4x) |
| real-loop epoch | 2.6 s measured | ~2.9 s estimated | 10.1 s measured (6.1 compiled + fp16) |
| weights shared across fingers and hands | no | yes | yes |

The 4 heads cost 4% over 1 head here (2.6 vs 2.5 s); the strict-parity fallback is 64 x 2 (2.2-2.3 s, 1.0M params)
at the cost of a 64-wide global token on the finger pathway that stalled on 27 Oct; 128 x 3 is the depth fallback.
All of it is random-input timing; the training run decides.
