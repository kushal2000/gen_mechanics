# 29 Oct: managing variable token counts in the forward pass (packing)

On the 100-hand template scene (`multi:template:`, 5 + 2 finger slots + the global token = 8 tokens per env) the
envs have 2 to 6 real fingers, 3.9 on average, so **38% of the tokens the per-finger trunk runs on are ghosts**
(masked as attention keys, but every per-token LayerNorm, projection and feed-forward still runs on them). Does
handling the variable counts better buy real speed? `bench.py`, pure torch on one RTX 6000 Ada, the repo's network
(`token_mode: finger`) with random inputs and every env's real finger count from its own hand.

## Variants (all numerically checked against the network: max |diff| 6e-7 in fp32)

- **padded**: the network as written.
- **ceiling**: padded with 5 tokens per row (the mean real count + the global, rounded up). Wrong outputs, right
  cost: the most skipping ghosts can buy without bookkeeping.
- **bucketed**: rows grouped by real-finger count, each group compacted to its real tokens (no mask) and run through
  the trunk separately; static group capacities (exact for the rollout, expected + 4 sigma for a minibatch) so it
  compiles. Two versions: 2-D advanced-index gathers/scatters, and flat `index_select` through inverse indices.
- **packed** (the language-model way: padding-free): every row's real tokens concatenated into one (R, d) matrix --
  80.5k real tokens for a 16,384-row minibatch against 131k padded -- so the per-token work runs on real tokens only,
  in one large batch; attention through FlexAttention with a document mask (each env's tokens attend only to each
  other). Every row has at most 8 tokens, so a 128-token query block only ever needs key blocks i-1, i, i+1: the
  block structure is built once, only the document ids change per batch. Static R (a capacity), compiled.

## Results (d128, 4 heads, ff x4, no affine norm; fp16 autocast; compiled unless noted)

| network | variant | inference @24,576 | train step @16,384 | network s / epoch (16 inf + 56 steps) |
|---|---|---|---|---|
| 2 layers | padded (now) | 6.5 ms | 11.1 ms | 0.83 |
| 2 layers | ceiling | 4.7 ms | 8.2 ms | 0.65 |
| 2 layers | bucketed, 2-D index | 4.5 ms | 9.6-12.2 ms | 0.70-0.80 |
| 2 layers | bucketed, flat index | 4.4 ms | 14.7 ms | 0.94 |
| 2 layers | **packed** | **4.3 ms** | **8.3 ms** | **0.64** |
| 4 layers | padded (now) | 11.4 ms | 19.4 ms | 1.38 |
| 4 layers | ceiling | 8.0 ms | 13.5 ms | 0.87 |
| 4 layers | bucketed, 2-D index | 6.8 ms | 16.2-24.7 ms | 1.02-1.50 |
| 4 layers | bucketed, flat index | 6.8 ms | 24.0 ms | 1.46 |
| 4 layers | **packed** | **7.0 ms** | **13.4 ms** | **0.97** |

(bucketed 2-D ranges are two runs of the same code: its compiled training step is not stable. Eager numbers are in
`results.json`: every bucketed variant is 1.5-2.5x slower than padded in eager training.)

## What it says

- **Packing reaches the ceiling, inference and training both**: the network 1.3x (2 layers) and 1.45x (4 layers)
  faster per epoch. Bucketing reaches it for inference only. A training minibatch split into five groups of ~3,300
  rows turns each layer into five small, launch-bound kernel sets, forward and backward. Packing keeps one large batch.
- **Per epoch it is 5-9%**, because the simulation dominates on the template scene: 801730 (finger, 2 layers) runs
  3.65 s per epoch of which the network is ~0.9 s, so packing saves ~0.19 s (5%); 801731 (4 layers, 4.4 s) saves
  ~0.41 s (9%). The share grows with depth and width, and with a faster simulator.
- **The bigger target is the joint-token network**: on the template grid it runs 36 joint slots + the global token
  for hands with 12-26 real joints (more than half the tokens are ghosts), and its epochs are 7.8 s (2 layers) and
  11.3 s (4 layers) against the per-finger 3.9 / 5.3 s. The same packing would cut its trunk work by ~half. Not
  measured here.

## The implementation (`packed_tokens: true`, the transformer YAML's default since 29 Oct)

`coevolution/networks/joint_transformer.py`, both token modes; `bench_network.py` times the network itself,
compiled as training runs it (so the padded rows here include the running-norm updates and graph breaks bench.py's
bare forwards leave out), same weights with the flag off and on, outputs checked first:

| network | packed == padded | padded, s/epoch | packed, s/epoch | network speedup |
|---|---|---|---|---|
| finger, 2 layers | 6.6e-7 | 1.04 | 0.86 | 1.22x |
| finger, 4 layers | 6.6e-7 | 1.58 | 1.18 | 1.34x |
| joint, 2 layers | 7.5e-7 | 4.17 | 2.18 | **1.91x** |
| joint, 4 layers | 8.3e-7 | 7.43 | 3.25 | **2.28x** |

(network s/epoch = 16 rollout inferences at 24,576 envs + 56 train steps at 16,384 rows + 2 ms loop overhead each.)
The joint-token network gains most: 36 joint slots + the global token for hands with 12.4 real joints on average
(64% ghosts), and attention is quadratic in the tokens. Measured in-scene epochs: below.

## How it works, and its rules

- The encoder's input (B, T, d) is packed by one sort over the B x T token keys (real tokens first, in row order) and
  an `index_select`; the layers run on (capacity, d) with FlexAttention and a document mask; an inverse index puts the
  outputs back in (B, T, d), ghost tokens zero. The heads are unchanged.
- Capacity: per batch size, set on its first batch (its real-token count + 6 standard deviations of a batch's count,
  a multiple of 128) and grown, with one recompile, if a batch ever has more. That needs one small count read to the
  host per forward (outside the compiled graph). The rollout and the minibatch are two sizes, as before.
- Block structure: a row has at most 128 tokens, so query block i needs key blocks i-1..i+1 only, built once per
  capacity outside the graph (Inductor cannot lower in-graph block indices); only the document ids change.
- A graph break before the attention graph: Inductor (torch 2.7) cannot lower a mask function that reads a buffer
  computed in the same graph.
- Refused (ValueError) without compile_net (FlexAttention's eager path is a reference implementation), with
  mask_ghost_actions off (ghost outputs are zero, not computed), parallel_block, grad_checkpoint, a head dim under 16
  (FlexAttention's minimum) or more than 128 tokens per row. Code default off, so saved configs replay as trained.
- Tests: `coevolution/networks/tests/test_finger_tokens.py` (packed == padded per env on a mixed template set, both
  modes, including a capacity regrow; the refusals).

    .venv_isaacsim/bin/python experiments/dummy_experiments/29oct_finger_token_batching/bench.py [--quick]
