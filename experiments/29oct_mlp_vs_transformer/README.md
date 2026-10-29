# 29 Oct: MLP vs joint transformer, same hands, same env

Started 29 Oct 2026 on branch `transformer_multi_embodiment`. A controlled comparison of the two policy architectures
on pen repose v2, after the 27-28 Oct debugging settled the transformer's setup
(`experiments/27oct_transformer_debug/README.md`).

## Design

Three hands, one seed, both architectures: 6 runs, 5,000 epochs each (the user's call: one seed, these hands).

| Hand | Hand joints | Why |
|---|---|---|
| `claude_design_0` | 12 | the design agent's grammar hand; MLP 23 goals/episode on 27 Oct (old tokens, "range" actions) |
| `coarse_s0_025` | 13 | the pick-from-100 agent's choice; MLP 27 goals/episode |
| `coarse_s0_000` | 20 | the first random hand, a twist joint and a palm cut; the transformer's hardest |

Everything but the architecture is shared, and explicit in every `.sub` (`make_subs.py` writes them):

- pen repose v2: palm rotation locked, uniform goals, first goal 3 cm then 1 cm; EMA 0.1
- **action delay off** (`env.action.use_action_delay=False`) for every run here, the user's call; the 7 Oct training
  default has it on, so these are not directly comparable with the 27 Oct MLP runs on the same hands
- `env.action.finger_target=home` (28 Oct): action 0 is the hand's default pose, +-1 its limits
- the pre-27-Oct tokens (ordered link boxes, no `joint_origin`): the 27 Oct tokens were worse for the transformer
- 24,576 envs, 5,000 epochs, seed 100, wandb `gen_mechanics_mlp_vs_jt`, logs in
  `debug_outputs/train_logs/29oct_mlp_vs_transformer/`

| | MLP | Transformer |
|---|---|---|
| Network | [1024, 1024, 512, 512] | width 128, 4 layers, 1 head, ff x4, shared per-joint action head |
| Learner | minibatch 114,688, 5 mini-epochs, lr 5e-4 adaptive | minibatch 16,384, 2 mini-epochs, lr 1e-4 adaptive |
| Dead action dims | counted | masked (ghost slots, locked wrist rotation) |
| Time per epoch (hand 000) | ~3 s | ~11 s |

The two differ in learner as well as network, on purpose: each runs its own established recipe. A learner-matched
pair is a follow-up if the gap needs explaining.

## What to read off

- goals per episode against epochs, and against wall-clock time (the transformer's epoch is 3-4x longer)
- the epoch each run first reaches 1 goal per episode (the user's headline metric); one seed, so a 2x gap is
  within the single-seed spread seen before
- how the gap changes with the joint count: the shared head's gradient cancellation grows with the number of joints

## Runs

| Hand | Arch | Seed | Job | Result |
|---|---|---|---|---|
| `claude_design_0` | MLP | 100 | 782694 | |
| `claude_design_0` | transformer | 100 | 782696 | |
| `coarse_s0_025` | MLP | 100 | 782703 | |
| `coarse_s0_025` | transformer | 100 | 782705 | |
| `coarse_s0_000` | MLP | 100 | 782707 | |
| `coarse_s0_000` | transformer | 100 | 782710 | |

Submitted 29 Oct, ~11:00.
