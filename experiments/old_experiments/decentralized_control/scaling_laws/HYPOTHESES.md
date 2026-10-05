# Hypotheses and decision criteria

Current study: one shared actor-critic trunk over clean privileged observations. The
separate asymmetric critic has been removed. Width/depth now change both
policy and value representations, so reward gains cannot be attributed to the
policy alone. Test whether sharing reduces update time and memory enough to
offset possible policy/value gradient interference. Both heads retain the
privileged inputs; the policy now sees them too. The SPS estimates below describe the cancelled asymmetric
design; they are historical context, not predictions for the shared study.

The primary outcome is leader-policy reward/success versus elapsed training
time on two RTX 6000 Ada GPUs. Also examine reward versus global environment
transitions, critic loss, KL, and throughput. Use the same three learning seeds
and common time/transition checkpoints. Reward near zero, loss divergence, or
OOM is not a throughput win. A 60-epoch tuning trial cannot establish learning
quality. Confirm promising checkpoints with the same held-out evaluation setup
before calling a model the winner; this folder does not implement evaluation.

## Per-model hypotheses

| Model | Hypothesis | Evidence against it |
|---|---|---|
| d32 L1 | A minimal shared trunk suffices for policy and value learning. | Fast SPS but consistently low success/plateau or poor value fitting. |
| d32 L4 | Four rounds compensate for narrow representations. | d64 L1 beats it despite the same approximate block parameter budget. |
| d64 L1 | Width is more useful than depth at a small capacity budget. | d32 L4 beats it on reward/time and reward/transitions. |
| d64 L4 | More rounds are useful at a moderate capacity budget. | d128 L1 beats it despite the same approximate block parameter budget. |
| d128 L1 | Rich representations plus one exchange suffice. | d64 L4 learns better at a similar block parameter budget. |
| d128 L2 | Both representation width and two rounds matter. | Smaller variants match its learning curves. |
| d256 L1 | Width is the main capacity bottleneck. | Narrower, deeper models outperform at comparable compute. |
| d256 L4 | Maximum tested capacity improves eventual performance enough to justify cost. | No late improvement, persistent optimization problems, or impractical memory. |

The L1 ladder (d32/d64/d128/d256) isolates width. d32/L4 vs d64/L1
each have about 32,768 block matrix weights, and d64/L4 vs d128/L1
each have about 131,072: these test width versus communication depth at
approximately matched block capacity. Heads and embeddings mean total actor
counts are not exactly matched. d128/L1 vs L2 tests the value of a second
round directly. d256/L1 vs L4 tests whether depth is useful once width is ample.

## Throughput hypotheses for every model

1. Increasing per-GPU minibatch amortizes synchronization and launch overhead
   until memory traffic or activation storage dominates. Reject a monotonic
   assumption if SPS drops at the larger batch.
2. More environments improves simulator utilization until physics, memory,
   or rollout storage dominates. Larger/deeper policies may peak at fewer envs.
3. Small shared networks hit the simulator floor, so reducing network FLOPs
   further yields little end-to-end improvement.
4. The highest-SPS setting may have worse sample efficiency because PPO batch
   size and update counts change. If so, compare the runner-up throughput
   setting and tune LR for the top model sizes in a separate experiment group.

## Historical predictions for the cancelled asymmetric design

These are broad planning estimates, **not measurements or fitted scaling laws**.
They refer to full-loop environment transitions/sec after warmup, using each
model's eventual selected pair from the tuning grid and the fixed d128/L2
critic. They are not actor forward/backward samples/sec or PPO augmented rows.

| Model | Predicted SPS |
|---|---:|
| d32 L1 | 95k–135k |
| d32 L4 | 80k–120k |
| d64 L1 | 90k–130k |
| d64 L4 | 75k–110k |
| d128 L1 | 85k–120k |
| d128 L2 | 75k–110k |
| d256 L1 | 70k–105k |
| d256 L4 | 45k–75k |

Anchors: the late portion of the compiled d64/L1 training run
`debug_outputs/train_logs/jt_l1_d64_tuned_sharpa_iiwa14_seed0_2026-09-04_01-26-14/slurm.out`
shows about 59.5k–59.8k SPS on one GPU; the tuned MLP run dated
`2026-09-03_22-09-51` shows roughly 67k–68k SPS. The older uncompiled benchmark
`debug_outputs/bench_logs/tp_l1_l1_d64/slurm.out` reported 43.6k SPS, and d256/L1
reported 20.1k SPS, demonstrating how much execution settings change results.
These old transformer runs used four heads, FF x4 and differently sized
critics; they are calibration context, not matched baselines.

The two-GPU synthetic L2 actor measurements were 5.64/13.18/25.10 ms at
d64/128/256, global minibatch 16384. They support the expected cost ordering,
but omit physics, critic, optimizer, and full PPO. The ranges above start from
roughly twice the old single-GPU throughput and allow for communication,
the larger fixed critic, and increasing actor cost. Node contention,
simulation scaling, NCCL behavior, and the selected env/batch pair can move
actual SPS outside these ranges. Replace them with observations after Stage 1.
