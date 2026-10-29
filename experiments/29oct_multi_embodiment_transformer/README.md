# 29 Oct: one transformer policy across the 100 random grammar hands

Started 29 Oct 2026 on branch `transformer_multi_embodiment`. The question: can the joint-token transformer learn pen
repose v2 across all 100 random grammar hands at once (`assets/urdf/27oct_grammar_random_hands/`, 2-6 fingers, 3-22
joints, 55 with a palm cut), and how does each hand fare inside the shared policy?

## Setup

`repose_grammar100_jt14.sub`: `HAND=multi:assets/urdf/27oct_grammar_random_hands/manifest.json` (every hand in one
scene, 245-246 envs each of 24,576, 22 padded joint slots), pen repose v2 (palm rotation locked, uniform goals, first
goal 3 cm), through the olympics trainer:

- **network / learner:** the 14 Oct transformer setup, `OlympicsJointTransformerSAPG.yaml`'s own: width 128, 4 layers,
  1 head, ff x4, minibatch 16,384, 2 mini-epochs, lr 1e-4 adaptive (48 updates per epoch); not compiled
- **dead dimensions masked:** `mask_ghost_actions` + `masked_arm_dims [3, 4, 5]` (the YAML's defaults since 28 Oct): a
  padded hand's ghost slots and the locked wrist rotation drop out of the ratio, the KL and the entropy
- **actions:** `env.action.finger_target=home`: action 0 is each hand's default (open) pose, +-1 its limits;
  **action delay off** (`use_action_delay=False`), the user's call for the 29 Oct runs (the 7 Oct default has it on)
- **tokens:** the pre-27-Oct tokens (ordered boxes, no `joint_origin`)
- 5,000 epochs, seed 100, about 20 s per epoch (the 100-articulation scene simulates at ~68k steps/s against ~210k
  for one hand), so roughly a day; wandb `gen_mechanics_pen_repose_grammar100`, logs in
  `debug_outputs/train_logs/29oct_multi_embodiment_transformer/`

Why these choices: `experiments/27oct_transformer_debug/README.md`. The first attempt (job 697447, 27 Oct, the "fast
config": width 32, minibatch 114,688) scored 0 goals per episode through 5,000 epochs.

## Per-hand results

The trainer logs `per_hand_<metric>/<hand>` for every hand in the scene (goals per episode, done reasons), so the
shared policy's score on each of the 100 hands comes straight from wandb. Analysis to follow once the run is in.

## Runs

| Job | What | Result |
|---|---|---|
| 782711 | `repose_grammar100_jt14.sub` (submitted 29 Oct, ~11:00) | |
