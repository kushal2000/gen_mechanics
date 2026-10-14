# Grammar viewer

A viser viewer for the hand grammar (`hand_sampler/grammar/`). The grammar itself, every decision and its reason, is explained in [`project-notes/grammar/GRAMMAR-OVERVIEW.md`](../../project-notes/grammar/GRAMMAR-OVERVIEW.md) and decided in `project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md`. Everything here runs on the CPU; Isaac is never imported.

## Setup (once)

```bash
cd "/home/singularity/Evolve to Generalize/gen_mechanics"
/home/singularity/.local/bin/uv venv .venv_viewer --python 3.11
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python viser "numpy==1.26.*" scipy trimesh yourdfpy pycollada pytest
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python -e . --no-deps
```

## Run

```bash
.venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1
```

Open http://127.0.0.1:8080 (from a laptop: `ssh -L 8080:127.0.0.1:8080 <this machine>`). `--hand allegro_right` starts with a commercial hand instead of a random one.

## The panel

One line per item; longer explanations are hover text.

- **Grammar**: **Random** draws hands under the current rules until the ticked viability checks pass, and says "found after N tries". Every draw is within the rules and passes C1 by construction (the hand is built finger by finger, each finger's base drawn among the placements whose servo fits and the finger redrawn until it clears the hand so far), so only C2 can reject.
- **Rules**: **Feetech Rules** (hands buildable from Feetech servos: what the simulator builds and evolution uses), **No Rules** (the whole grammar: coupled and sliding joints too) or **Custom Rules** (any field edited). The fields can only tighten the grammar: max fingers (2-6), max joints per finger (1-5), max palm joints (0-1), coupled and sliding joints, max finger length (Feetech: 250 mm), max link length (Feetech: 90 mm), the base grid (Feetech: 40 mm), min and max base distance from the palm centre (Feetech: 40-90 mm), fingers face outward, servo link lengths (links at least as long as two servos need, fingertips at least 30 mm). The next line says whether the hand on screen follows the rules; the last one gives the search space under these rules (hands with joints on the three kinds and with all 31 axis directions, before the clearance check; `hand_sampler/grammar/space.py`).
- **Viability**: the two checks, one line each with the measured value; untick one to stop Random requiring it.
  - **C1, clearance**: every two solids (links, base servos, the mount, the hinge servo, the palm) keep one printed wall (2 mm) apart at the zero pose and at the episode start pose, except parts that touch by construction; no base servo in a palm cut's seam.
  - **C2, fingertips meet above the palm**: at least one pair of fingers whose fingertips' reachable regions come within 20 mm above the plate, over the palm.
- **Commercial hand**: a commercial hand conformed onto the grammar's grid (10 mm / 30°) at the scale it can be built from Feetech servos (the smallest, in steps of 0.05 up to 5 x, with servo link lengths, no collisions and every joint keeping its kind), drawn over its own URDF meshes at that scale (ghosted), with one line: fingers and joints, the scale, the fit (largest joint, axis and fingertip error in millimetres at the real hand's size, target 10 mm / 20°) and whether it follows the rules. Curl 0 is the real hand's zero pose (the conformed hand at its zero-pose difference); the meshes follow the pose. SHARPA is the KaRMA model, hand only. MIDAS has no meshes on this branch (only its URDF was copied).
- **Mutation**: **Random mutation**, **Back**, and one button per operator, shown only when it has a child that keeps these rules and C1. One grid: every value moves 10 mm or 30°, a base one grid step. A mutation tries the operator's applications in random order and keeps the first such child. After a mutation, one line says what changed.
- **Pose**: **curl** and **palm bend** set every joint slider at once (bending joints at this fraction of their upper limit, 0.35 being the episode start pose, other joints at 0; the palm joint at this fraction of its ±30° range, shown only when the hand has a cut), **link bodies** (the opacity of each link's body, a box around its yoke and the servo it carries out to the fingertip, which the clearance check and the simulator use; see-through by default so the servos and fingertips show, 0 hides them), **show the object at its start** (the repose task's 72 mm cube where an episode starts, blue and half transparent; off by default) and **re-centre view**.
- **Joints**: one slider per joint of the hand on screen, in degrees (mm for a sliding joint): "finger 2 joint 1 (spread)" moves only that joint, so an MCP's spread and bend can be seen one at a time. A coupled joint's slider is disabled and follows the joint before it.

On screen: the palm plate (grey, see-through so the servos in it show) around the mount (dark grey, behind the palm's centre), the part beyond a cut (darker grey), each link's body (see-through, in its finger's colour) and every Feetech servo: its body in its finger's colour, its output horn (white: a disc with one arm, so its turn shows) and a dark rod along its axis; the hinge servo of a cut the same way. Each servo's body moves with the link before its joint, its horn with the link after it. Solids of a failing C1 pair are red. The palm faces up (+z).

## Words

| In the panel | In the code (`hand_sampler/grammar/hand.py`) |
|---|---|
| finger | `Finger`: base `y`, `z` (mm from the palm centre), `facing`, `tilt` (degrees), its joints |
| joint + link | `Joint`: `type` (hinge, coupled, sliding), `axis` (az, el) in degrees, `length` of the link after it (mm) |
| palm cut | `PalmJoint`: the line's direction `facing` and its distance `offset` from the palm centre; the part beyond it hinges along it |
| bend / spread / twist | the derived kinds flexion / abduction / roll (`derive.joint_kind`) |
| Feetech Rules / No Rules | `FEETECH_RULES` (also `EVOLUTION_RULES`) / `NO_RULES` (`Rules`) |
| servo | the Feetech HL-3915 (`motor.py`) |
| C1 / C2 | `viability.c1_self_overlap` / `viability.c2_workspace_overlap` |

## Parameters

Per hand: what evolution changes. Global: one value for every hand. Ranges under the Feetech rules (the grammar's own in brackets).

| Parameter | Per hand or global | Range | Step |
|---|---|---|---|
| fingers | per hand | 2-6 | add / remove a finger |
| joints per finger | per finger | 1-5 | split / merge a link |
| palm cut | per hand | none or one straight line, a finger on each side | cut / remove; move 10 mm; turn 30° |
| base position (y, z) | per finger | the 40 mm grid, 40-90 mm from the palm centre (10 mm grid, 10-160 mm) | one grid step |
| facing | per finger | outward (any) | 30° |
| tilt | per finger | -30° to +90° | 30° |
| joint axis (two angles) | per joint | 31 directions (a line; its sign is derived), in the link frame: x along the finger, y the closing direction (the palm normal turned by the tilt), z = x × y | 30° |
| joint type | per joint | hinge (hinge, coupled x1.1 the joint before it, sliding) | couple / uncouple |
| link length | per link | the servo floor for its two axes (30-50 mm) to 90 mm, 0 for two joints at one point between different axes square to the finger; fingertip 30-90 mm (0 or 10-90; fingertip 10-90); finger at most 250 mm | 10 mm |
| links | global | servo-sized: one box around each link's yoke and the servo it carries | - |
| palm plate | global | 24 mm thick; outline: hull of the mount and the servos' housings; a cut leaves an 8.6 mm seam | - |
| base depth | derived | the first servo's housing flush with the front face | - |
| clearance | random hands and every mutant | C1 at rest and at the start pose; a random finger is redrawn up to 50 times, a mutation keeps the first child that passes | - |
| joint ranges | global per kind | bend -30° to +90°, spread ±30°, twist ±90°, palm ±30°, sliding -20 to +25 mm | - |
| physics | global | 0.5 N m, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, 1750 kg/m³, friction 0.5 | - |

## Mutation operators

`hand_sampler/grammar/operators.py`. Each lists its applications, tries them in random order and returns the first child that keeps the rules and C1 (Vatsal's method), so nothing outside the rules is ever produced.

| Button | Operator | What it does |
|---|---|---|
| move a finger | `move_finger` | the base by one grid step along y or z |
| turn a finger | `turn_finger` | the facing by 30° |
| tilt a finger | `tilt_finger` | the tilt by 30° |
| lengthen or shorten a link | `length` | one link to its next allowed length (from its shortest down to 0 where allowed) |
| turn a joint axis | `axis_az` | the axis around the link by 30° (links move to their new shortest if needed) |
| tip a joint axis | `axis_el` | the axis toward or away from the link by 30° |
| move / turn the palm cut | `move_cut`, `turn_cut` | the line by 10 mm or 30° |
| add a finger | `add_finger` | a one-joint finger (bend, 40 mm) at a free spot, facing away from the centre |
| remove a finger | `remove_finger` | any finger in one step; a cut left with no finger on a side goes too |
| add a joint (split a link) | `split` | a new joint at any 10 mm point of a link, kind drawn with the commercial mix; links too short grow to their shortest |
| remove a joint (merge two links) | `merge` | one joint; the links on either side become one (at most 90 mm) |
| couple or uncouple a joint | `couple` | No Rules only: a joint follows the joint before it (x1.1), or moves on its own |
| cut the palm / remove the palm cut | `add_cut`, `remove_cut` | a straight cut with a finger on each side, or none |

Inverse pairs: each parameter step and its opposite (when no link was refit and away from the twist pole), add and remove a finger, cut and remove the cut, couple twice.

## Tests

```bash
cd experiments/grammar_viewer && ../../.venv_viewer/bin/python -m pytest -q tests
```

`tests/test_viewer.py` drives the viewer's callbacks headless against a local viser server: Random under the rules, the Rules dropdown and fields, the checks, the operator buttons, mutation and Back, a commercial hand over its meshes, the pose, one drawn body per solid, and a link's box mesh.
