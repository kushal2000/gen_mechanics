# 29 Oct: one articulation for every hand (the template scene)

Started 29 Oct 2026 on branch `per_finger_transformer`, after the per-finger network left the 100-hand run
simulation-bound: 11 s per epoch of which 10 s was the physics step, at 46k sim fps.

## Why the multi-hand scene was slow

Isaac Lab and PhysX batch one articulation prim replicated across envs into a single GPU view. The 1 Oct multi-hand
scene (`isaacsimenvs/inhand_olympics/multi_hand.py`) gives every hand its own articulation over its own envs, so a
100-hand scene is 100 views of ~245 envs, and every step, state read and target write is 100 small calls. The
September generated-population path did the opposite: one padded template articulation in every env, each design's
primitives authored into its env, and ran 1,024 designs at 500k sim fps (12,288 envs, arm included) -- faster than a
single mesh hand. The docstring claim that "a padded topology made the sim much slower" came in with the multi-hand
scene commit and has no measurement behind it; the September logs say the opposite. House of Dextra (ICLR 2026) does
the same thing: every hand USD shares one 20-joint tree with absent servos as ghost joints, one articulation over
8,192 envs.

## What this is

`hand.robot_spec = "multi:template:<specs>"`: every env gets its own hand's primitives on ONE shared tree
(`isaacsimenvs/common/robots/template_robot.py`, authored by `inhand_olympics/author_template.py`):

- The tree is the hand set's `SlotGrid` (`finger_layout.py`): the 6-joint dummy wrist, the palm, the palm section
  behind the palm joint (a cut hand's moving section; a locked ghost otherwise), then `n_rigid` finger slots hanging
  off the palm and `n_moving` off the section, 5 joint slots each. `grid_for` sizes it from the set: the 100 random
  grammar hands need 5 + 2 (a cut moves at most 2 fingers, the rigid side holds up to 5), 36 hand joints + 6 wrist,
  43 bodies.
- A hand's URDF is parsed into boxes, capsules (cylinders, as the converter replaces them), convex meshes (the palm
  .obj files) and inertials, fixed joints merged, and each link placed in its slot at the URDF's own frame; the
  unused slots are ghosts: a 1e-6 kg link with no collider on a joint locked by limits (0, 1e-8), which
  `joint_enabled` masks as before. Verified Kit-free over all 100 hands (`common/tests/test_template_robot.py`).
- Each hand is authored once into its own USD layer and referenced into its envs (0.1 s for the hands, 36 s to
  reference 24,576 envs; authoring every env's prims was 33 ms per env).
- The env reads fingertips through per-env body indices (a finger's tip is its last slot's link, a different body
  per hand); the viewers get `real_joint_names` / `real_slots` from `TemplateHand`.
- New optional token field `joint_moving` (1 on a joint behind the hand's palm cut): a cut hand's moving fingers
  were not otherwise identifiable to the policy. Off unless an obs list names it.
- Plain `multi:` and `multi:finger:` are untouched, so earlier multi-hand checkpoints replay.

## Measurements (RTX 6000 Ada, portal-compute-03, per-finger transformer, 29 Oct)

| scene | hands | envs | sim-only fps | epoch | scene build |
|---|---|---|---|---|---|
| one hand, URDF to USD, replicated (job 796258) | 1 | 24,576 | 299k | 2.4-2.8 s | |
| articulation per hand (job 796276) | 100 | 24,576 | 46k | 10-11 s | |
| **template** (`multi:template:`, smoke) | 100 | 24,576 | **250k** | **3.3 s** | 43 s |
| template, smoke | 3 | 6,144 | 178k | 0.9 s | 8 s |

The update step is the same on both 100-hand scenes (0.9-1.2 s); the whole gain is the physics step. The template
articulation carries 42 joints per env against a real hand's 13-26, so a single hand through the template is slower
than through its own USD; the template is for sets.

## Physics equivalence

`equivalence.py`: one hand driven through both scenes (its own converted USD, and the template) with the same
sinusoidal joint commands for 90 steps, no action delay, palm rotation locked, from the default pose.

The first run found a real bug: every capsule fingertip in the template was 3.3 cm short. USD's Capsule `height` is
the cylindrical part only (tip to tip = height + 2r), and the URDF converter's `replace_cylinders_with_capsules`
puts the cylinder's full LENGTH there (a 0.069 m, r 0.0165 cylinder becomes a 0.102 m capsule); the template had
used length - 2r. On hand 000 finger 3 then sank 1.7 cm lower against the table, joint errors up to 0.4 rad. Fixed
in `template_robot._collider`. (So every grammar hand trained through the URDF path has fingertip capsules 2r longer
than the cylinder its URDF describes -- a modelling gap independent of this scene.)

After the fix:

| hand | joints, max / mean | fingertips, max / mean | palm position | limits |
|---|---|---|---|---|
| `coarse_s0_000` (cut, 6 fingers) | 0.068 / 0.003 rad | 14 / 0.9 mm | 4.6 mm | identical |
| `coarse_s0_026` (no cut, 6 fingers, its 6th on the ghost section; 5+2 grid) | 0.41 / 0.009 rad | 59 / 2.0 mm | | identical |

Every fingertip's lowest height agrees to the millimetre on both hands. The peaks are contact events, not offsets:
on hand 026 finger 4 tracks to 6 mm for 60 steps, meets finger 3 at step ~66, slips in one scene and holds in the
other (5.5 cm for a few steps), and is back within 3-5 mm by step 78. The 6th finger, hung off the 1e-6 kg locked
ghost section (8 of the 100 hands spill a finger there), tracks to 1 mm and 0.045 rad throughout.

## Runs (29 Oct): token mode x depth on the 100 hands

All on `multi:template:assets/urdf/27oct_grammar_random_hands/manifest.json`, wandb
`gen_mechanics_pen_repose_grammar100`, logs `debug_outputs/train_logs/29oct_template_scene/`; everything but the
network as `experiments/29oct_per_finger_transformer/grammar100_finger_s100.sub` (job 796276 on the per-hand scene).
Why: there the per-finger 2-layer network learned the keypoint shaping much slower than the per-joint 4-layer one
(782711): keypoint reward 0.033 vs 0.075 at epoch 680.

| `.sub` | job | network | first epochs |
|---|---|---|---|
| `grammar100_template_finger_L2_s100.sub` | 801730 | per-finger, 2 layers | 3.9 s |
| `grammar100_template_finger_L4_s100.sub` | 801731 | per-finger, 4 layers | 5.3 s |
| `grammar100_template_joint_L2_s100.sub` | 801732 | per-joint, 2 layers | 7.8 s |
| `grammar100_template_joint_L4_s100.sub` | 801733 | per-joint, 4 layers | 11.3 s |

Two per node (portal-compute-03 and -04): compare at matched epochs. All four: width 128, 4 heads, ff x4, compiled,
fp16 rollouts, no affine LayerNorm, the 14 Oct learner, masks on, `joint_moving` NOT observed.

## Limits

- The whole set's finger slots are fixed per set; a hand whose cut moves more fingers than the grid's moving slots,
  or that has more fingers than the grid, is refused at `hand_set` (the 100 hands fit 5 + 2).
- Joint gains are one actuator group over the slot names, and the home pose is all zero, so the scene refuses a set
  whose hands differ in either (`scene.py`); the grammar hands are uniform (stiffness 3.0, damping 0.0775, armature
  0.00058, home 0).
- Pose reaching refuses `multi:template:` (its wrapper puts fingertips in fingertip order; the template's tables are
  by finger slot).
- Branching finger chains and fixed-joint fingertips (commercial hands) are not placed; this path is for the
  primitive-built grammar and claude_design hands.
