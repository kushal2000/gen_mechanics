# 14 Oct: design search

Started 14 Oct 2026 on branch `2026-10-14-design_search`. The goal: generate primitive hands, then search over
designs for the best controllers.

## Primitive hands (`primitive_hands.py`)

- **Palm:** a disc, 5 cm radius, 2 cm thick. Its axis is the grasp normal.
- **Fingers:** 4 per hand, leaving the rim radially at mid-thickness. A design is just the bearings where the
  fingers sit around the rim.
- **Modules:** each finger is 4 identical capsule modules, 2.5 cm long with radius 1 cm (2 cm wide and
  thick), collinear at rest.
- **Joints:** one flexion joint at the base of each module. The axis is the rim tangent, and the range is
  0-90° toward the grasp face.

A design is a small JSON (`PrimitiveHand`), so a sampler, a mutation operator or an agent can propose one, and
`build` turns it into a URDF and spec JSON. The spec is in the format the envs already load (`spec_from_json`, as
for the uniform Wuji hands), with uniform-Wuji gains.

- **Naming:** a hand is named by its module size and canonical gaps, e.g. `prim4x4_r10l25_g090_090_090_090`.
  Designs that differ only by turning or flipping the disc get the same name.
- **Sampling:** `sample()` draws distinct designs on a 5° grid, with neighbouring fingers at least 25° apart
  (2 cm-wide fingers touch at 23°).

    .venv_isaacsim/bin/python experiments/14oct_design_search/primitive_hands.py --random 8 --seed 0
    .venv_isaacsim/bin/python experiments/14oct_design_search/primitive_hands.py --bearings 0,90,180,270
    # -> assets/urdf/generated/14oct_design_search/<name>.{urdf,spec.json,design.json} (gitignored: rebuilt from designs)

## Viewer (`viser_hand_play.py`): the hand in the olympics env

Any built hand loads into the in-hand olympics env by its spec path (`get_robot_spec` takes a `.spec.json`). It sits
on the 6-DOF wrist, palm down above the table, with the pen. The page offers:
- a hand dropdown of every built hand (loading one restarts the sim, about a minute)
- building a hand from bearings, or at random
- a slider per joint, grouped wrist / finger 0..3, steered there by the env's own action pipeline
- every joint's position limits, editable live in the sim

    .venv_isaacsim/bin/python experiments/14oct_design_search/viser_hand_play.py --port 8093

Self-collision is on, with the uniform hands' filter: each link against its parent, and the palm against each
finger's first two links.

## The hand-designed "optimal" placement: evenly spaced, 0/90/180/270

A dedicated design pass (14 Oct) recommends evenly spaced fingers for pen lift, with 0/70/140/250 as runner-up.
Its analysis used forward kinematics against the 10 x 3 x 3 cm pen and a quasi-static force check over every pen yaw.

- **The wrist decides it.** Its yaw range is only -20..+30 deg, and the pen's yaw is random. A layout with 90 deg
  symmetry can always be turned so all four fingers land on the pen's long sides, two per side, about +-2.5 cm from
  the centre. A finger more than about 65 deg off the line across the pen lies over it and can only hook its small
  end face.
- **Force needed to hold the pen** against gravity plus a typical push or twist, worst case over pen yaw:

  | Layout (gaps) | Worst-case finger force |
  |---|---|
  | Evenly spaced | 1.3-1.4 N |
  | 60/120/60/120 | 2.3 N |
  | 45/135/45/135 | 3.1 N |
  | Two-thumb, 30/150/30/150 | no secure grasp for 11% of pen yaws |

  A finger can push about 20 N, so reach and angle are the limit, not strength.
- **Spin and flip** should favour the same layout, so there's no conflict between tasks.
- **Caveat:** these are static numbers. Only training separates near-even layouts.

Renders: `plots/prim4x4_r10l25_g090_090_090_090_{face,curl45}.png`
(`analysis/render_primitive.py`).

## Runs (pen lift, MLP, box pen, stagger, seed 100, 5,000 epochs; wandb `gen_mechanics_design_search`)

| Hand (gaps) | Job |
|---|---|
| 30/55/55/220 (random) | 272196 |
| 35/65/195/65 (random) | 272198 |
| 50/65/55/190 (random) | 272200 |
| 90/90/90/90 (designed) | 272940 |
