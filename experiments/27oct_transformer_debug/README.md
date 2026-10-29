# 27 Oct: why the joint transformer doesn't learn pen repose

Started overnight 27-28 Oct 2026. On grammar hand `coarse_s0_000` (6 fingers, 20 joints, a palm cut) the MLP learns
pen repose v2 (0.25 goals per episode by epoch 400, about 1 by 2,500) and the joint-token transformer barely moves
(0.07 at epoch 1,400, 0.2 at 3,400; job 697372). The same holds at width 128 (job 696227) and with 5 mini-epochs
(job 701199, worse), and for the 100-hand multi-embodiment run (job 697447, about 0 at epoch 1,400). An MLP on the
same new tokens (job 705905) learns, a little behind the old-token MLP, so the tokens are not the main cause.

## What the transformer runs show

- Policy loss 3-10x the MLP's from epoch 100 on (its updates do not improve PPO's objective on their own batch); 5
  mini-epochs make it worse.
- Entropy climbs (52 at 2 mini-epochs, 65 at 5) where the MLP's stays at about 40.
- Per-update stats (`coevolution/utils/update_stats.py`, the H3 probes in
  `experiments/27oct_grammar_random_hands/h3_*.sub`): each finger action's KL per update is about half the MLP's,
  each wrist action's about 3x; the 6 wrist dimensions carry about 60% of the KL (the MLP: 20%). The gradient clip
  fires on 40-75% of steps (the MLP: 4-30%); the value head's gradient is the largest part of the norm; the adaptive
  lr climbs to 4.7e-3.
- The full 400-epoch probes (median per update, epochs 21-400) sharpen that. The value head is 76-80% of the
  gradient norm in BOTH networks, and the clip only fires in the transformer's first ~100 epochs, so neither
  explains the gap. What does: the transformer's finger MEANS barely move. Of each finger action's KL per update,
  the MLP's is mostly the mean shifting (about 2/3: 4-6e-4 of 6.7e-4), the transformer's almost all the noise
  level (sigma) changing: its mean-shift part is 6.5-13e-5, 5-7x smaller than the MLP's, while its sigma part grows
  (the entropy climb). The wrist head, a plain MLP, moves normally (mean-shift KL 0.6-1.5e-3 per dimension).
  Reading: the finger-mean pathway -- one shared Linear on each joint's token, its gradient the sum over 20 joints
  -- cancels or cannot express the per-joint changes the advantage asks for, so PPO widens the noise instead. Not
  H3's amplification.

## Round 1: one change each from the stalled base

Base: `experiments/27oct_grammar_random_hands/repose_coarse_s0_000_jt_d32.sub` (width 32, 4 layers, 1 head, ff x4,
arm head [1024, 512], value head [512, 256], minibatch 114,688, 2 mini-epochs, lr 1e-4 adaptive, critic_coef 4,
fp16 autocast, compiled, extent tokens). 1,500 epochs, seed 100, wandb `gen_mechanics_jt_debug`. The `.sub`s are
written by `make_subs.py r1`.

| Arm | Change | Tests |
|---|---|---|
| `critic1` | critic_coef 4 -> 1 | the value loss crowding the shared trunk and setting off the clip |
| `maxlr` | adaptive lr capped at 2e-4 | the lr wandering up |
| `lrconst` | constant lr 1e-4 | no adaptive lr at all |
| `fp32` | no fp16 autocast | precision in attention / LayerNorm |
| `d64` | width 64 (minibatch 57,344 to fit) | the 32-wide global token as the only route from goal to fingers (H1) |
| `heads4` | 4 attention heads | one head for 20 joints (H2) |
| `wristsmall` | arm head [256, 128] | the wrist head absorbing the update (H7) |
| `mu64` | a per-joint MLP head [64] | one Linear per joint is too little per-joint capacity |
| `net30sep` | the 30 Sep default network (width 64, ff x2, mu head [64], value [512, 256]) | the in-hand default |
| `oldtokens` | ordered boxes, no joint_origin | the 27 Oct tokens, for the transformer |

`analysis/compare.py --round r1` compares them at matched epochs with the base and the MLP and writes
`plots/r1_curves.png`.

## Round 2: controls

Round 1 at epochs 500-850 (28 Oct, ~01:30): every arm still at about 0 goals per episode (the MLP: 0.27 at 500, 0.60 at
750), the old-token arm included. And the olympics transformer's own record (wandb
`gen_mechanics_inhand_olympics_carried`, 13-14 Oct, all width 128, minibatch 16384, wrist rotation FREE at 1.5 rad/s):
on Wuji thumb + index, pen lift, it learned in 2 of 3 seeds (about 45 goals per episode by epoch 1,500-2,500, the
MLP's pace); on the 5-finger Wuji, pen lift, 0 in all 3 seeds (the MLP: 1 of 3). Since 15 Oct the wrist cannot rotate,
so the fingers must do all the reorienting -- through exactly the pathway the probes show barely moving. Round 2 (seed
102, 1,500 epochs) separates the code, the wrist lock, the task and the hand:

| Arm | Hand | Task | Network | Question |
|---|---|---|---|---|
| `ctrl_ti` | Wuji thumb + index | pen lift, wrist free | exactly 257159's (width 128, old tokens) | did anything regress since 14 Oct? |
| `ti_locked` | Wuji thumb + index | pen lift, wrist locked | the same | did the 14 Oct transformer need the wrist to turn? |
| `s000_lift` | coarse_s0_000 | pen lift, wrist locked | the base transformer | repose or the hand? |
| `s000_lift_mlp` | coarse_s0_000 | pen lift, wrist locked | the MLP | its reference |

## Round 3: the joint count

Round 2 so far (epoch ~1,000): `ctrl_ti` reproduces 257159 almost exactly (0.36 / 1.81 / 30.7 goals at epochs 500 /
750 / 1,000; 14 Oct: 0.35 / 1.79 / 31.5), so nothing regressed; `ti_locked` learns even faster with the wrist locked
(21 at 750, 34 at 1,000), so the transformer did not need the wrist to turn the pen. Hand 000 pen lift is 0 for the
MLP too, so it separates nothing. The pattern: the transformer learns on thumb + index (few joints) and fails on the
20-joint hands (5-finger Wuji, coarse_s0_000) -- what cancellation in a shared head predicts. Round 3 (1,500 epochs,
the 14 Oct transformer setup now verified: width 128, minibatch 16,384, old tokens; wrist locked; repose):

| Arm | Hand (hand joints) | Against |
|---|---|---|
| `ti_repose`, `ti_repose_mlp` | Wuji thumb + index | each other (seed 102): does the task alone break it? |
| `cd0` | claude_design_0 (12) | its MLP, job 695902 (seed 100, old tokens) |
| `s025` | coarse_s0_025 (13) | its MLP, job 695903 |
| `s000_jt14` | coarse_s0_000 (20) | its MLP, job 694301; completes the width-128 run cancelled at epoch 344 |

## Results

### Round 1 (seed 100, 1,500 epochs; goals per episode, mean of the 40 epochs up to each)

| Arm | ep 1,000 | ep 1,500 |
|---|---|---|
| MLP (705905, same tokens) | 0.77 | 0.86 |
| base (697372) | 0.029 | 0.085 |
| `net30sep` | 0.079 | **0.175** |
| `d64` | **0.091** | (1,440: running) |
| `mu64` | 0.021 | 0.061 |
| `oldtokens` | 0.018 | 0.061 |
| `wristsmall` | 0.012 | 0.047 |
| `critic1` | 0.011 | 0.038 |
| `fp32` (gradient accumulated over 2 chunks) | 0.011 | (running) |
| `heads4`, `lrconst`, `maxlr` | ~0 | ~0 |

No single change comes near the MLP. Width helps a little (64 wide: 2-3x the base at matched epochs; single seeds);
lowering the lr hurts (capped or constant: ~0); the rest are within noise of the base or below it.

### The cancellation diagnostic (`diag_cancel`, the base with compile off, 400 epochs)

| Epochs | head/cancel | token cosine | finger KL from the mean | finger KL total |
|---|---|---|---|---|
| 1-20 | 0.58 | 0.49 | 1.4e-4 | 1.5e-4 |
| 21-100 | 0.55 | 0.44 | 1.2e-4 | 2.4e-4 |
| 101-200 | 0.46 | 0.40 | 6.6e-5 | 5.6e-4 |
| 201-300 | 0.39 | 0.47 | 4.8e-5 | 8.7e-4 |
| 301-400 | 0.34 | 0.48 | 4.6e-5 | 1.1e-3 |

head/cancel = ||sum_j g_j|| / sum_j ||g_j|| over the 20 joints' contributions to the shared head's gradient (1:
aligned; 20 independent random directions give about 0.22). The joints' pushes on the shared head cancel more and
more as training goes on, the finger means' movement shrinks with it, and the noise takes over. The tokens are not
collapsed (cosine ~0.45): the head is shared by joints that need different things, not fed identical inputs.

### Round 2 (seed 102, 1,500 epochs)

| Arm | ep 500 | ep 750 | ep 1,000 | ep 1,500 | first 1 goal/ep |
|---|---|---|---|---|---|
| `ctrl_ti` (257159 rerun) | 0.36 | 1.81 | 31.7 | 43.4 | 647 |
| 257159 itself (14 Oct) | 0.35 | 1.79 | 31.5 | | |
| `ti_locked` | 1.11 | 21.3 | 33.9 | 36.8 | 462 |
| `s000_lift` / `s000_lift_mlp` | 0 | 0 | 0 | 0 | - |

No regression: today's code reproduces the 14 Oct run almost to the decimal. The transformer does not need the wrist
to rotate (it learns faster locked). Hand 000 cannot learn pen lift with either network, so that pair says nothing.

### Round 3 (in progress at ~05:00; the 14 Oct setup, repose, wrist locked)

| Hand (joints) | Transformer ep 500 / 750 / 1,000 | MLP on the same hand ep 500 / 750 / 1,000 |
|---|---|---|
| Wuji thumb + index | 0.99 / 1.39 / 2.23 | 0.92 / 1.18 / 1.49 (`ti_repose_mlp`) |
| coarse_s0_025 (13) | 0.68 / 0.95 / - | 0.84 / 0.96 / 1.17 (695903) |
| claude_design_0 (12) | 0.11 / 0.40 / - | 0.89 / 1.00 / 1.21 (695902) |
| coarse_s0_000 (20) | 0.00 / - / - | 0.42 / 0.68 / 0.79 (694301) |

The transformer learns repose about as fast as the MLP on thumb + index and on the 13-joint hand, somewhat slower on
the 12-joint cross, and not at all (so far) on the 20-joint hand. So neither the task nor the architecture is broken
as such: the transformer fails on hand 000 (20 joints, 6 fingers, a twist joint, a palm cut). Round 1's width-64 arms
on hand 000 are still rising at epoch 1,500 (`d64` 0.26, `net30sep` 0.17; the MLP 0.86).

### Round 4 (submitted ~05:00)

| Arm | Question |
|---|---|
| `s025_fast`, `cd0_fast` | the fast config (width 32, minibatch 114,688, new tokens; what the 100-hand run uses) on the hands where the 14 Oct setup learns: is the fast config itself a problem? |
| `d64_more`, `net30sep_more` | r1_d64 / r1_net30sep resumed from epoch 1,500 to 3,000: does width 64 catch the MLP on hand 000? |

### Round 3 (final) and round 4: the fast config was the main problem

| Goals per episode | ep 500 | ep 750 | ep 1,000 | ep 1,500 | first 1/ep |
|---|---|---|---|---|---|
| thumb + index: 14 Oct transformer setup (`r3_ti_repose`) | 0.99 | 1.39 | 2.23 | 5.52 | 496 |
| thumb + index: MLP (`r3_ti_repose_mlp`) | 0.92 | 1.18 | 1.49 | 5.60 | 531 |
| coarse_s0_025: 14 Oct setup (`r3_s025`) | 0.68 | 0.95 | 1.05 | 2.50 | 894 |
| coarse_s0_025: fast config (`r4_s025_fast`) | 0.01 | 0.10 | 0.20 | 0.46 | - |
| coarse_s0_025: MLP (695903) | 0.84 | 0.96 | 1.17 | 3.48 | 837 |
| claude_design_0: 14 Oct setup (`r3_cd0`) | 0.11 | 0.40 | 0.79 | 1.03 | 1,320 |
| claude_design_0: fast config (`r4_cd0_fast`) | 0.07 | 0.21 | 0.39 | 0.69 | - |
| claude_design_0: MLP (695902) | 0.89 | 1.00 | 1.21 | 2.40 | 728 |
| coarse_s0_000: 14 Oct setup (`r3_s000_jt14`) | 0.00 | 0.08 | **0.41** | **0.80** | |
| coarse_s0_000: fast config (base, 697372) | 0.00 | 0.003 | 0.029 | 0.085 | - |
| coarse_s0_000: MLP (694301) | 0.42 | 0.68 | 0.79 | 0.90 | 2,123 |

The 14 Oct transformer setup (width 128, minibatch 16,384: 48 updates per epoch) learns repose on every hand,
level with the MLP on thumb + index and close on the 13-joint hand, and it also learns hand 000 (0.41 at epoch 1,000,
the MLP 0.79). The fast config (width 32, minibatch 114,688: 8 updates per epoch; introduced 27 Oct for speed) is
4-14x worse on every hand. The width-128 run on hand 000 that was cancelled at epoch 344 (job 696227) was on the
working setup: this one too is at 0 until epoch ~500. Width 64 on hand 000 resumed to 3,000 (`r4_d64_more`,
`r4_net30sep_more`): 0.77 and 0.53 at 3,000 -- the MLP's 0.90-at-1,500 level in twice the epochs, if at all. The 100-hand run (697447) is on the
fast config: 0 at epoch 3,881.

### Round 5 (submitted ~07:00): which half of the fast config

| Arm | Width | Minibatch (updates/epoch) | Tokens |
|---|---|---|---|
| `r3_s000_jt14` (reference) | 128 | 16,384 (48) | old |
| `d128_bigbatch` | 128 | 114,688, accumulated in 7 chunks (8) | old |
| `d32_smallbatch` | 32 | 16,384 (48) | old |
| `newtokens` | 128 | 16,384 (48) | new (extent + joint_origin) |

### Round 5 at epoch ~1,000 (10:15 on 28 Oct)

| Arm (hand 000) | ep 750 | ep 1,000 | ep 1,500 |
|---|---|---|---|
| `r3_s000_jt14`: width 128, minibatch 16,384, old tokens | 0.08 | 0.41 | 0.80 |
| `d128_bigbatch`: width 128, minibatch 114,688 | 0.00 | 0.04 | |
| `d32_smallbatch`: width 32, minibatch 16,384 | 0.02 | 0.08 | 0.17 |
| `newtokens`: width 128, minibatch 16,384, NEW tokens | 0.00 | 0.00 | |

Both halves of the fast config hurt; the big minibatch (8 updates per epoch) most. And the 27 Oct tokens look worse
for the working transformer too (0 at epoch 1,000 vs 0.41), as they were for the MLP on this hand (705905 vs 694301:
0.12 vs 0.26 at epoch 400, 6.4 vs 10.3 goals at 5,000). Single seeds -- round 6 checks it.

### Round 6 (submitted ~10:20): old vs new tokens, more seeds, hand 000

MLP old / new tokens x seeds 101, 102 (2,500 epochs); the 14 Oct transformer setup old / new tokens, seed 101 (1,500
epochs).

## Conclusions (28 Oct)

1. The transformer works; the 27 Oct "fast config" broke it. The 14 Oct setup (width 128, minibatch 16,384, 48
   updates per epoch) learns pen repose on every hand tested -- level with the MLP on thumb + index (5.5 vs 5.6 goals
   per episode at epoch 1,500), close on the 13-joint hand (2.5 vs 3.5) and on hand 000 (0.8 / 0.6 over two seeds vs
   0.9). The fast config (width 32, minibatch 114,688, 8 updates per epoch) is 4-14x worse at matched epochs on every
   hand and never recovers. Nothing regressed in the code; the wrist lock is not the cause.
2. Both halves hurt: the big minibatch delays learning (width 128 catches up by epoch 1,500), width 32 keeps it low
   (0.17 at 1,500). The mechanism is cancellation in the shared finger head (head/cancel 0.58 -> 0.34): the finger
   means stop moving and the noise grows. Fewer joints learn faster. Not the cause: lr caps (they hurt), the critic
   weight, fp16, heads, wrist-head size, H3's amplification.
3. Per-epoch speed was a false economy: the 14 Oct setup reaches 0.8 goals/episode in ~4.5 h, the fast config 0.2 in
   5.4 h. The transformer is still ~5x slower than the MLP to the same score on hand 000. It was never compiled.
4. The 27 Oct tokens (extent boxes + joint_origin): level for the MLP over two seeds (0.88 / 0.95 old vs 0.99 / 0.94
   new at epoch 1,500); worse for the transformer over two seeds (0.80 / 0.58 old vs 0.02 / 0.11 new). Cause unknown;
   the pooled token normaliser over joint_origin is the first suspect. Transformer runs use the old tokens for now.
5. Decisions (28 Oct): transformer runs use the 14 Oct setup; `mask_ghost_actions` is on by default and the locked
   wrist's roll / pitch / yaw are masked too (`masked_arm_dims: [3, 4, 5]`), so dead action dimensions drop out of
   the ratio, the KL and the entropy; new runs centre finger actions on the home pose
   (`env.action.finger_target=home`). The 100-hand rerun: `repose_grammar100_jt14.sub`.
