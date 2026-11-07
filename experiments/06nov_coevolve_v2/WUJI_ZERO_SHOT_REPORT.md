# Wuji v2 zero-shot / transfer: overnight findings (6-7 Nov 2026)

Controllers: v1 `k5_none` gen 7 and `k1_none` gen 8 (pen repose v2, wrist translation only). Small local evals
(16-32 envs, sampled actions). Every hand-made design is rendered at q = 0 and collision-checked before it is scored
(`plot_morph_gallery.py` -> `debug_outputs/coevolve/studies/grammar_morph/gallery_zero_pose.png`).

## 1. Piping: no fundamental bug; one real bug fixed

* Audit (agent): template and finger scenes are identical for the same hand (obs to 4e-7, actions, slots, network);
  vendor Wuji is wired correctly (FK to 0.001 mm, action routing and signs, frames, home).
* **Fixed:** the table-safe reset in `multi:finger:` scenes read `arts[0]`'s PhysX view with slot body indices
  (wrong links, CUDA asserts). New `OlympicsMultiHand.slot_link_transforms()` (`isaacsimenvs/inhand_olympics/multi_hand.py`),
  used by `env._table_margin`. Template scenes unchanged. Check: reference hand, finger scene, table-safe on:
  **44.8 goals/ep vs template 44.6** (was 35.5 with the workaround off).
* Eval harness control: `wuji_wrapped_eval.py` on the reference grammar hand (finger scene): **47.1 (k5) / 18.7 (k1)**
  goals/ep, 100 % of envs. So the vendor zeros are real, not the harness.

## 2. Why zero-shot gives 0: layout (strong), abduction pattern (suspected), size (untested cleanly)

### Training labels (k5_none gen 7, 4096 hands, training-log goals/ep)
| hands | n | mean goals/ep |
|---|---|---|
| 5 fingers, largest angular gap < 180 deg | 1016 | ~17 |
| 5 fingers, gap 180-240 | 8 | 1.5 |
| 5 fingers, gap > 240 (Wuji: 290) | 0 | - |
| fingers < 140 mm (Wuji: 106) | 417 | 1.6 |
| 0 / 1 / 2 / 3+ fingers with an abduction joint | 2075 / 1802 / 212 / **7** | 10.2 / 17.5 / 12.9 / 11.0 |

Wuji has an abduction joint on all 5 fingers (4 x MCP abd as the 2nd joint + thumb CMC).

### Grammar morph ladder (best f5_m20-21 design -> Wuji-like, template scene, all C1-clear)
| design | k5 g7 | k1 g8 |
|---|---|---|
| ref | 42.2 | 18.4 |
| radial spread 240 | 35.6 | 39.1 |
| radial spread 180 | 20.9 | 25.2 |
| radial spread 120 | 12.3 | 9.3 |
| four fingers on one edge (pitch 40-60 mm, fan 60/30/0 deg) | 0.00-0.09 | 0.4-0.9 |
| + thumb turned toward fingers | 0.00 | 0.38 |

The cliff is the largest gap going ~180 -> ~240 deg, the same as in the training labels. (An earlier version of this
ladder had spread60 / parallel rungs whose fingers intersected at spawn, and size rungs whose fixed-size motors
collided once links shrank; those numbers were discarded.)

### Vendor Wuji copies (URDF edits, `morph_vendor_hand.py`, all collision-checked with `vendor_clearance.py`)
All 0.00 goals/ep with both controllers, adapter rungs A and C, episodes always time out:
x2 (effort 1.39 and 0.5), x2.2, x2 + training joint limits, x2 half-spread, x2 starfish (largest gap ~159 deg),
x2 starfish + training limits, x1 starfish, x2 + finger abductions locked, starfish + locked abductions (x1, x2,
x2 + training limits), recentred (x2, x2 starfish, x2 starfish + locked abd), each also with the reference hand's
palm_extents token (24 x 150 x 196 mm) -- no change.

### Abduction pattern (grammar, the user's hypothesis)
| design | k5 | k1 |
|---|---|---|
| ref | 48.0 | 25.5 |
| ref + Wuji's chain on the 4 fingers (flex, abd, flex, flex; 0/70/50/70 mm) | **0.00** | **0.00** |
| control: same lengths, all flexion | 4.9 | 1.4 |
The chain's lengths alone cost most of it; abduction on four fingers takes it to zero.

### Mount convention (a real difference, not the policy)
Grammar hands' URDF root frame -- where the wrist attaches -- is the palm centre (`urdf_export.py` derived mount);
the vendor Wuji's root is `l_wrist`'s origin, **38 mm (x1) / 75 mm (x2) from the palm centre**, so the pen spawned
beside the palm. `morph_vendor_hand.py --recentre` moves the palm frame to the centre (verified: palm centre at
xy (0, 0) at reset). Every vendor-hand experiment, including the Wuji RL runs, has this offset; the loader is not
changed.

### What the policy does on Wuji (behaviour diagnostics, 16 envs)
| hand | closest tip-pen | steps with a tip < 20 mm | joint pos (0 = lower, 1 = upper) |
|---|---|---|---|
| reference (control) | 15 mm | 341-457 | 0.50-0.58 |
| Wuji copies, no adapter | 45-99 mm | 0 | 0.17-0.29 (opens the hand: the vendor's -60 deg lower limits) |
| Wuji copies, rung C (training ranges) | 23-68 mm | 0-1 | 0.41-0.63 |
Sim joint angles: on Wuji the policy drives the fingers to about -50 deg (hyperextension) where the reference goes
to +50..+88; with the training-range adapter it flexes, but never closes on the pen.

### 7 Nov morning: the MCP token -- first zero-shot goals
* Rendered rollouts (`render_rollout.py`): the reference curls each finger from its BASE joint (j0/j1 +40..+90 deg)
  under the palm; on Wuji the policy left the MCP at -28..+39 and bent only the PIP. Not self-collision: a
  kinematic MCP sweep 0-90 deg is clear (`flex_sweep_clearance.py`) and every joint tracks its PD target.
* Cause: the MCP flexion token. In the locked copies (abd made `fixed`, stub link left in the chain) the MCP's
  `joint_link_vec` read **0 mm** while the real MCP -> PIP distance is 96-104 mm: a base joint that "moves nothing".
  **All `*_lockabd` results above were invalid.** In the unmodified vendor hand the MCP link is a real but tiny
  14 mm (x1) stub before the abduction axis -- the same pattern as the grammar control with a zero-length first
  link (48 -> 4.9).
* Fixed lock (links merged, MCP link 104 mm): x2 starfish + training limits + merged abd + recentred:
  **k5 0.56 goals/ep (44 % of envs), k1 0.75 (62 %)**, pen lifted 62-93 mm -- the first non-zero zero-shot. The
  rollout shows a real grasp (fingers curled under the palm, pen lifted) but late (step ~100-200 vs 15-25) and with
  the thumb unused (same 28 mm CMC stub + rolled axes).
* Mechanism in code (`hand_sampler/design_space.py:1288`, `joint_axes_and_link_vectors`): a joint's link vector
  is the origin of the next hand joint hung from its OWN child link (or the fingertip offset), else zeros.
* **Causal test, observation only** (`--first-link-scales`): every finger's first-joint `joint_link_vec` token
  scaled, physics unchanged.

  | token x | ref k5 | ref k1 | ref base-joint pos (k5) | Wuji merged k5 | Wuji merged k1 |
  |---|---|---|---|---|---|
  | 1 | **47.1** | **27.8** | 0.55 | 0.56 | 0.75 |
  | 0.5 | 2.4 | 1.3 | 0.47 | 0 | 0.19 |
  | 0.25 / 0.27 | 0.06 | 0 | 0.36 | 0 | 0 |
  | 0 | 0 | 0 | 0.30 | 0 | 0 |

  The policy reads link vectors as precise geometry (a kinematic model of the hand), not a coarse cue: halving
  the base links' token takes the working reference hand from 47 to 2.4. Adapters must report the TRUE geometry.
* Training population: 15732 / 17408 fingers start with a flexion joint and a 50-90 mm link; flexion with a 0 mm
  link: 30; abduction-first fingers have long links too (so `--knuckle 1`, abduction first at 0 mm, was itself
  out of distribution: 0 goals everywhere). Wuji's knuckle (14 mm flexion stub, then abduction) has no analogue.
* Next: `--knuckle 2` (hide: flexion's link to the PIP, which is true with the abduction held at 0; abduction slot
  shown as a ghost slot) and `--knuckle 3` (abduction as a zero-length stub at the PIP).
* (superseded) observation/action adapter `--knuckle 1` for the UNMODIFIED vendor joints: each flex+abd knuckle shown as
  abduction first (zero-length link, at the base) then flexion with its link to the next joint, the pattern of 1239
  training fingers; actions swapped back.

**Conclusion (before the MCP token fix -- superseded in part):** zero-shot from v1 is not reachable by fixing one factor -- layout (grammar: 42 -> 0), abduction on
five fingers (48 -> 0), wide vendor limits, the mount offset and size are each out of the training distribution.
The v1 policy is competent on hands like its population, not a general grasp strategy.

## 3. RL on Wuji (wandb `wuji_v2_transfer`)
* scratch learns: 0.02 (ep 250) -> 0.85 (ep 490) -> 1.03 -> **10.0 goals/ep at ep 929** (99.7 % with a goal).
* fine-tunes from v1 k5 / k1: 0.000 / 0.004 at ep 689 / 1079.
* **Why the fine-tunes are stuck (likely):** the v1 checkpoints' SAPG blocks 0-1 sit at the log-std cap 2.0
  (std 7.4: pure noise -- the same blocks that scored 0 goals in every v1 run), and blocks 2-5 have the converged
  std 0.58 (k5) / 0.31 (k1). The fine-tunes' sigmas have not moved in 500 epochs; scratch is at std ~0.86 everywhere.
  Test: `WUJI_rl_ft_k5_sig0` (job 60129), the k5 checkpoint with every block's log-std reset to 0 -> **no better**
  (2nd dump: approach reward 2.57 / keypoint 2.4 vs ft_k5 2.99 / 2.8 vs scratch 7.06 / 66.7). Noise is not it.
* **Next suspect, the critic:** the v1 value normaliser has count 1.9e10 (frozen at mean 9.5, var 260 -- v1's return
  scale), and the value head predicts v1 returns. Test: `WUJI_rl_ft_k5_fc` (job 67077): v1 actor + trunk, fresh
  value head, value normaliser reset (mean 0, var 1), log-std 0.
  -> **partly**: at the 2nd dump approach 6.47 / keypoint 47.0 (scratch 7.06 / 66.7; ft_k5 2.99 / 2.8), so the
  inherited critic is why the plain fine-tunes sat at exactly 0. But by epoch 1000 it is still far behind scratch:

  | goals/ep at dump (~100 epochs each) | 2 | 4 | 6 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|
  | scratch | 0.02 | 0.85 | 1.29 | 3.75 | 10.0 | **28.2** |
  | ft_k5 fresh critic | 0.00 | 0.41 | 0.96 | 1.26 | 1.50 | **1.84** |
  | ft_k5 log-std reset | 0.00 | 0.00 | 0.02 | 0.34 | 0.66 | 0.87 |
  | ft_k5 | 0 | 0 | 0 | 0 | 0 | 0 |

  The v1 actor itself is a worse start than random init on Wuji (scratch takes off at ~ep 800, the fine-tunes do
  not). Recipe if fine-tuning onto a new hand anyway: reset the value head, value_mean_std and SAPG log-stds
  (`debug_outputs/coevolve/studies/wuji_rl_ckpt/`).
* scratch: **28.2 goals/ep** at ~ep 1000, **43.3 at ep 1354** -- level with the best evolved grammar hands.

### Final comparison (goals/ep by design_stats dump, ~100 epochs per dump)
| run | d2 | d4 | d6 | d8 | d9 | d10 | d16 | d21 | final |
|---|---|---|---|---|---|---|---|---|---|
| scratch | 0.02 | 0.85 | 1.29 | 3.75 | 10.0 | 28.2 | 45.4 | 46.0 | **46.7** (ep 2604) |
| trunk only (v1 transformer + its input norms; fresh heads, critic, log-std) | 0.14 | 0.95 | 1.36 | 2.92 | 5.11 | 10.5 | | | running |
| fresh critic (v1 actor kept) | 0.00 | 0.41 | 0.96 | 1.26 | 1.50 | 1.84 | 12.0 | 41.1 | running |
| log-std reset | 0 | 0 | 0.02 | 0.34 | 0.66 | 0.87 | 1.53 | 2.83 | 8.65 (ep 2604) |
| ft_k1 | 0 | 0 | 0 | 0 | 0 | 0 | 0.12 | 0.89 | 1.42 (ep 2604) |
| ft_k5 | 0 | 0 | 0 | 0 | 0 | 0 | 0.62 | 1.18 | 1.66 (running) |

**Reading:** the inherited critic + value normaliser hold plain fine-tunes at 0 for 1000+ epochs; the v1 action heads
delay take-off by ~700 epochs; the v1 transformer trunk transfers neutrally (tracks scratch: a little ahead early,
a little behind at d8-10). No variant beats scratch on Wuji: the v1 controller carries no useful prior for this hand.

## Files
`experiments/06nov_coevolve_v2/`: `grammar_morph.py`, `morph_vendor_hand.py`, `vendor_clearance.py`,
`plot_morph_gallery.py`, `layout_vs_goals.py`, `wuji_wrapped_eval.py` (now with behaviour diagnostics and a hard
exit), `audit/`. Outputs: `debug_outputs/coevolve/studies/{grammar_morph,wuji_morph,audit,wuji_rl_ckpt}`.
