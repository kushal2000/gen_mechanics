# What exactly breaks IsaacLab's in-hand env when it is made to look like ours?

## The question

IsaacLab's Allegro in-hand env learns to hold the cube within ~50 epochs and then reorients it.
`ourdefault` (28sep folder, job 631568) is that env with **everything of ours applied except the
observation and the network**, and it reproduces our env's failure: it learns to hold the cube, then
abandons it (episode length 337 → 53, drop 0.50 → 1.00 between epochs 50 and 100).

Yet every change tried **on its own** left the reference learning: SAPG, γ 0.99, γ 0.98, cube
size, random start orientation, our PPO hyperparameters, 60 Hz / 10 s, our physical Allegro, our
spawn, our progress reward shape, and our full reward. So some combination breaks it. The one
pair that looked like it did (our reward + γ 0.98) was a single run stopped at epoch 235.

## Design: leave-one-out from `ourdefault`

Start from the broken configuration and revert **one group** of changes to the reference per run.
A group whose revert *prevents the collapse* is one the break requires.

| run | reverts to the reference |
|---|---|
| `control_seed43` | nothing — `ourdefault` again, second seed (seed 42 is 631568) |
| `lo_reward` | reward shape: dense `1/(θ+0.1)` + their penalties; bonus 250 on our success |
| `lo_success` | success test, goal distribution, per-goal timer (true angle, `rot_x·rot_y` goals, one clock) — at our 20° |
| `lo_tolerance` | tolerance only: our keypoint test at their 0.1 rad (5.73°) |
| `lo_timing` | 30 Hz, 20 s episodes, action EMA 0.95 |
| `lo_hand` | their Allegro USD, drives (0.5 N·m cap), mount (palm 5.40°) |
| `lo_object` | their 60 mm cube at their mass |
| `lo_spawnreset` | cube over the fingertips, identity start, their 20%-of-range hand reset |
| `lo_dr` | their friction / mass / gain randomization |
| `lo_misc` | their observation noise; drop measured from the root |
| `lo_gamma` | γ 0.998 |
| `lo_hparams` | their lr 5e-4, 5 mini_epochs, horizon 24, entropy 0.002, reward scale 0.1, no mixed precision |
| `lo_sapg` | plain PPO |

Configs: `isaacsimenvs/inhand_isaaclab/env_cfg.py`, classes `Loo*Cfg`, each `InHandIsaacLabOurDefaultCfg`
with one group reverted by copying it from a fresh reference cfg. The three learner groups are reverted
by the `.sub`'s flags instead. All nine env configs were smoke-tested at 768 envs.

Every run otherwise matches `ourdefault`: 12288 envs, SAPG 6 × 2048, γ 0.99, our PPO hyperparameters,
mixed precision, seed 42, 3000 epochs.

## Decision rule

**Primary metric: goals per episode** (`episode_final/successes`), at matched epochs. Episode length
and drop rate are read alongside, to explain *why* a run scores or doesn't.

Thresholds come from runs we already have, at 20°:

| reference point | goals/ep at e500 | at e1000 |
|---|---|---|
| a run that works (MLP recipe on our env, 623419) | 1.2 | 13.3 |
| the IsaacLab reference, their reward (5.73°) | 0.12 | 0.19 |
| **the control that breaks** (`ourdefault`, 631568) | — | ~0.4 (at e1286) |

- **Decision: goals/episode relative to the control at matched epochs**, read at epochs 1000–3000.
  A revert that gives clearly more goals/episode than the control identifies a group the break
  depends on. Dropping the cube is not held against a run. (An earlier absolute threshold, "broken =
  flat below 0.5", is dropped: the control itself reached 1.63 goals/episode by epoch 2073 while
  dropping 97% of episodes.)
- Episode length and drop rate stay in the table as diagnostics. Note that goals chain only within an
  episode (each success resets the clock) and a drop ends it, so possession bounds how high
  goals/episode can go: the fixed MLP runs reach ~48; the collapsed controls ~1.6.

A run that recovers identifies a **necessary** group. If none recovers, the break has redundant causes
and the next round is pairs among the most suspicious groups.

Three runs change what a goal is, so compare them on their own terms:
- `lo_tolerance` targets 5.73° instead of 20° — compare with the 5° MLP recipe (624497: 0.19 at e500,
  0.35 at e1000), not with the 20° runs.
- `lo_success` uses the true angle at 20°, stricter than our keypoint test at "20°" (which admits 20–35°
  by axis), so the same skill scores fewer goals.
- `lo_timing` is 30 Hz: an episode is twice as long in seconds per step.

Caveat inherent to goals/ep in our task: each success resets the per-goal clock, so a policy that
starts succeeding also gets longer episodes, and the metric compounds (saturates at the 50 cap).

One seed per run; `control_seed43` checks that the control itself reproduces.

## Launch

```bash
for f in experiments/old_experiments/29sep_what_breaks_reference/*.sub; do sbatch "$f"; done
```

Logs: `debug_outputs/train_logs/29sep_what_breaks_reference/`. wandb project [`gen_mechanics_what_breaks_reference`](https://wandb.ai/kk837/gen_mechanics_what_breaks_reference), group `29sep_what_breaks_reference`.
