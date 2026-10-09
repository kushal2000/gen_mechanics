# OOD robustness: does a design that trains well also generalise?

Each figure is a 2x2 of goals per episode: push force and friction on the top row, object mass and hand tilt angle on the
bottom. A dotted line marks the training value on each axis. Every policy was trained at a single fixed condition
(no friction, mass or push randomisation). Lines are the mean over training seeds, bands the min-max range. Policies are
the best-training-reward checkpoints, picked without looking at OOD data, unless a legend says otherwise. Positive tilt
angles slope the palm down toward the fingertips; training used 5 deg. The training-condition point is the same on
every panel: per seed, the mean of all distinct evaluations of it across the sweeps.

Protocol (`analysis/eval_ood.py`): 64 envs, the first episode of each env, greedy actions. Every setting is as trained
except the swept one. Redraw with `analysis/make_plots.sh` as results land.

## Headline: `full_vs_index_middle.png`

Full Wuji v2 vs the index + middle hand, MLP, 3 seeds each. Both score about the same in training
(37 vs 31 goals per episode). Under push and high friction the full hand keeps most of its score and index + middle
collapses: push 1 N gives 28 vs 2, friction 1.0 gives 24 vs 6. Mass and backward tilt separate the two moderately.
Forward tilt does not separate them.

## H1: fewer fingers, less robust (`h1_fewer_fingers_less_robust/`)

`full_vs_index_middle.png` is the headline pair on its own (the same figure as at the top level).

`finger_subsets.png` places the full hand, the hand without the ring finger, thumb + index + middle, and index + middle
side by side (MLP; full and index + middle 3 seeds, the other two 1 seed). All four score within 7 goals per episode
of each other at the training condition (31-38).

- Push: a thumb and a third finger are enough. No ring scores 23 and thumb + index + middle 17 at 1 N, against the
  full hand's 28 and index + middle's 2.
- High friction and heavy objects: only the full hand holds up. Without the ring finger the score drops to index +
  middle's level, 9 vs 24 at friction 1.0 and 8 vs 18 at 500 g.
- Tilt: no finger-count pattern; thumb + index + middle does best when the hand tilts back.

## H2: link geometry changes which perturbations a hand tolerates (`h2_link_geometry/`)

Only the finger links are scaled; the palm and the finger mounts stay the same.

- `link_length.png`: the full hand vs finger links 75% and 50% as long. Shorter links lose robustness to push, light
  objects and low friction, and gain it for heavy objects: at 500 g the 75%-length hand scores 28 against the full
  hand's 18. The 50%-length hand is worse than the full hand on push, light objects and friction.
- `link_width.png`: the full hand vs finger links 120% as wide, 3 seeds each. Wider links score higher at the training
  condition (43 vs 37), on friction above it (34 vs 24 at 1.0) and with the palm tilted back (22 vs 15 at -5 deg).
  Under push and at 500 g they are level with the full hand. That looks more like a better hand overall than one
  specially robust to friction.


## H3: architecture, transformer vs MLP (`h3_transformer_vs_mlp/`)

`full_hand.png` compares the full hand trained with the MLP (3 seeds) and with the joint transformer (1 seed). Both are
trained at 24k envs with EMA 0.1 and action delay, so architecture is the only difference.

The transformer scores higher on almost every axis: 44 vs 37 at the training condition, 31 vs 24 at friction 1.0, and
35 vs 28 at a 1 N push. On push and heavy objects it stays within the MLP's seed band. With one transformer seed this is
a lead, not a result.


## H4: multi-embodiment training imbues robustness (`h4_multi_embodiment_robustness/`)

Each figure puts a multi-hand transformer, evaluated on one hand, against a specialist for that hand trained on
identical settings. These are final checkpoints (epoch 5000), 1 seed each.

- `full_hand.png` pairs the 6 Oct 52-hand policy with the full-hand specialist: EMA 0.1, no action delay, 12k envs.
  Despite scoring 12 goals per episode less at the training condition, the 52-hand policy is better at
  high friction, with a bouncy object and with the palm tilted back. It is worse under push, with heavy objects and with
  the palm tilted forward. So multi-embodiment training changes which perturbations the policy tolerates; it does not
  make it robust across the board. This needs more seeds before it counts as a finding.
- `index_middle.png` pairs the policy for the 40 hands with 2 or 3 fingers (E4) with an index + middle specialist
  (make_runs.py T1), both with action delay: EMA 0.1 + action delay, 24k envs, 5 mini-epochs. The 40-hand policy has
  not learned this hand: 0.7 goals per episode at the training condition and at most 1.2 anywhere. The specialist
  scores 13 there. The figure therefore says nothing about robustness. E4 was still improving at epoch 5000
  (2.2 goals per episode over all 40 hands); it needs longer training before this comparison can be made. The 6 Oct
  52-hand policy is weak on this hand everywhere (9 goals per episode at the training condition), so it isn't a useful
  comparison either.

## H5: domain randomisation over the OOD axes (`h5_dr_on_ood_axes/`)

The same MLP recipe, trained with every OOD axis randomised over the middle of its sweep: push 0-1.5 N, friction
0.25-1.0, object mass 15-250 g and hand tilt -5 to 15 deg, each sampled per episode. Tilt is emulated with a constant
force on the object; it moves a cube on a still hand the same way a really tilted hand does. Baselines are the no-DR
runs above (3 seeds); the DR runs are 1 seed. The evaluation turns the DR off and changes one axis at a time.

`training_progress.png`: goals per episode during training. The DR runs are scored under their own randomisation, so
their curves are not on the same footing as the no-DR ones.

- `index_middle.png`: the DR policy is weaker everywhere inside the training range. It scores 13 goals per episode at
  the training condition against 31, and on push, light objects and heavy objects it stays below the no-DR policy.
  It is at or above the no-DR policy only where both are weak: friction 1.0 (7 vs 6) and 1.5 (5 vs 3), and tilt -10
  deg (3 vs 2). Within the randomised ranges, index + middle trained with DR is worse than without DR.
- `full_hand.png`: the full hand trained with DR scores at or above the no-DR policy at every evaluated value except
  friction 0.1, where both score 5. It scores 47 at the training condition against 37, 45 against 24 at friction
  1.0, 35 against 11 at friction 1.5, 37 against 28 at a 1 N push, and 21 against 18 at 500 g. Tilt is where it helps
  most: 37 against 4 with the hand tilted back 10 deg, outside the randomised range, and 18 against 7 at 20 deg
  forward. The final checkpoint (epoch 5000) gives the same picture.
- The two hands respond to the same DR in opposite ways. The full hand gets better everywhere. Index + middle gets
  worse, even at the training condition. With 1 seed each, this is a lead, not a result.
