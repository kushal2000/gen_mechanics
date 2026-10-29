# 29 Oct: the per-finger transformer in training

Started 29 Oct 2026 on branch `per_finger_transformer`. The speed study (`experiments/29oct_transformer_speed/README.md`)
ended in a network that runs an epoch in ~2.6 s instead of the per-joint transformer's 10.1 s, next to the MLP's 2.6 s;
this folder trains it. The architecture is the transformer default in `OlympicsJointTransformerSAPG.yaml` from today.

## The network (`coevolution/networks/joint_transformer.py`, `token_mode: finger`)

One token per **finger** instead of per joint. A finger's token is its up-to-5 joint tokens, palm to tip, side by
side (the same 19 features a joint token had: 5 proprioceptive values, the 12-number link box, the limits) plus 5 raw
slot-validity bits, 100 numbers; a slot the finger lacks reads zero. 6 finger slots + the global token (wrist state,
task, object, and the palm-cut joint's token if the hand has one) = 7 tokens for every hand in the grammar (at most 6
fingers, 5 joints each, 1 palm cut). A shared decoder MLP [256] reads a finger's 5 actions off its token; a static
index puts them back in canonical joint order; the palm-cut joint's action comes from the wrist head. Width 128,
2 layers, 4 heads, ff x4, no affine LayerNorm, the whole network compiled, fp16 rollouts; ghost fingers, ghost slots
and the locked wrist masked; the 14 Oct learner unchanged (minibatch 16,384, 2 mini-epochs, lr 1e-4 adaptive).

Which joint belongs to which finger, and where along it, comes from the URDF tree
(`isaacsimenvs/common/robots/finger_layout.py`): each fingertip's path up to the palm lists its finger's joints; a
joint on two fingertips' paths, or driving a palm section, is the palm joint. Checked on all 40 registry hands, the 100
random grammar hands and the primitive hands (`isaacsimenvs/common/tests/test_finger_layout.py`). A multi-hand run
needs the finger-major template, `HAND=multi:finger:<specs>` (31 slots: finger f's k-th joint at 5f + k, the palm
joint at 30), so the network's slot index is one static map for every hand; plain `multi:` keeps the old dense
packing so earlier multi-hand checkpoints replay.

Tests: `coevolution/networks/tests/test_finger_tokens.py` (shapes, the palm-cut joint, the locked wrist, a finger's
actions from its own token only, the finger-major template's ghosts masked, dense refs refused in finger mode).

## Runs

Everything but the network as `experiments/29oct_mlp_vs_transformer`: pen repose v2 (palm rotation locked, uniform
goals, first goal 3 cm), `finger_target=home`, action delay off, the pre-27-Oct tokens, 24,576 envs, 5,000 epochs,
seed 100. Every setting is written out in the `.sub`.

| `.sub` | hand | job | controls (same hand, env, seed) | wandb |
|---|---|---|---|---|
| `s025_finger_s100.sub` | `coarse_s0_025` (13 hand joints, 4 fingers) | | MLP 782703, per-joint transformer 782705 (`29oct_mlp_vs_transformer`) | `gen_mechanics_mlp_vs_jt` |
| `grammar100_finger_s100.sub` | all 100 grammar hands, `multi:finger:` | not submitted | per-joint transformer 782711 (`29oct_multi_embodiment_transformer`) | `gen_mechanics_pen_repose_grammar100` |

Logs: `debug_outputs/train_logs/29oct_per_finger_transformer/`.

## Smoke (before submitting)

3 epochs at 6,144 envs on this node: the single hand and the 100-hand finger-major template both build, compile and
log epochs (observation width unchanged from the per-joint network on the single hand; 31 hand slots + 6 wrist on the
template). Filled in below from the smoke logs.

## Results

(to be filled in: epochs to 1 goal/episode and goals/episode at 5,000 epochs against the two controls, with the
wall-clock axis -- the point of the architecture is the epoch time)
