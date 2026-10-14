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
