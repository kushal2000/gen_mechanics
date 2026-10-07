# Grammar viewer

A viser viewer for the locked hand grammar (`hand_sampler/grammar/`). The grammar itself, every decision and its reason, is explained in [`project-notes/grammar/GRAMMAR-OVERVIEW.md`](../../project-notes/grammar/GRAMMAR-OVERVIEW.md) and decided in `project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md`. Everything here runs on the CPU; Isaac is never imported.

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

- **Grammar**: **Random** draws hands under the current rules until the ticked viability checks pass, and says "found after N tries" (every draw is within the rules by construction, so only the checks can reject).
- **Rules**: **Evolution Rules** (what the simulator builds and evolution uses: hinge and coupled joints), **No Rules** (the whole grammar: sliding joints too) or **Custom Rules** (any field edited). The fields can only tighten the grammar: max fingers (2-6), max joints per finger (1-5), max palm joints (up to 6), coupled and sliding joints, max finger length (250 mm), min finger spacing (19 mm), max base distance from the wrist centre (160 mm). Fixed by the grammar: link lengths 0 or 15-90 mm, fingertip at least 10 mm; every finger base on the rim of its plate (at most 8 mm inside the convex hull of the heel disc, the hinges and the bases on that plate), never in the middle of the palm. The last line says whether the hand on screen follows the rules.
- **Viability**: the two checks, one line each with the measured value; untick one to stop Random requiring it.
  - **C1, no overlap**: no two links (rounded boxes; the palm plate counts) overlap by more than 3 mm, at the zero pose and at the episode start pose.
  - **C2, fingertips meet above the palm**: at least one pair of fingers whose fingertips' reachable regions come within 20 mm above the plate, over the palm.
- **Commercial hand**: a commercial hand conformed onto the grammar's fine grid, drawn over its own URDF meshes (ghosted), with one line: fingers and joints, the fit (largest joint, axis and fingertip error) and whether it follows the rules. Curl 0 is the real hand's zero pose (the conformed hand at its zero-pose difference); the meshes follow the pose. Dex3, Wuji v2 and MIDAS have no meshes on this branch (only their URDFs were copied).
- **Mutation**: **Coarse (10 mm / 30°)** or **Fine (1 mm / 5°)**, **Random mutation**, **Back**, and one button per operator, shown only when it can act on this hand under these rules. After a mutation, one line says what changed.
- **Pose**: **curl** (bending joints at this fraction of their upper limit; 0.35 is the episode start pose; other joints stay at 0), **palm bend** (every palm joint at this fraction of its ±30° range, so the hinged sections fold; shown only when the hand has a palm joint) and **re-centre view**.

On screen: the palm plate (grey) from the heel disc at the wrist to the knuckles, hinged palm sections (darker grey), rounded-box links in one colour per finger, dark joint markers; links of a failing C1 pair in red. The palm faces up (+z) with the fingers along +x.

## Words

| In the panel | In the code (`hand_sampler/grammar/hand.py`) |
|---|---|
| finger | `Finger`: base `y`, `z` (mm from the wrist centre), `facing`, `tilt` (degrees), its joints, `palm_joint` |
| joint + link | `Joint`: `type` (hinge, coupled, sliding), `axis` (az, el) in degrees, `length` of the link after it (mm) |
| palm joint, palm section | `PalmJoint`: hinge point `y`, `z` and `axis`; the section is the plate around its fingers and the hinge |
| bend / spread / twist | the derived kinds flexion / abduction / roll (`derive.joint_kind`) |
| Evolution Rules / No Rules | `EVOLUTION_RULES` / `NO_RULES` (`Rules`) |
| Coarse / Fine | `stage="coarse"` / `"fine"` (`operators.STEPS`) |
| C1 / C2 | `viability.c1_self_overlap` / `viability.c2_workspace_overlap` |

## Parameters

Per hand: what evolution changes. Global: one value for every hand.

| Parameter | Per hand or global | Range | Coarse step | Fine step |
|---|---|---|---|---|
| fingers | per hand | 2-6 | add / remove a finger | - |
| joints per finger | per finger | 1-5 | split / merge a link | - |
| palm joints | per hand | 0-6, one per finger at most, never without a finger | give / move / remove | - |
| base position (y, z) | per finger | 10-160 mm from the wrist centre; neighbours at least 19 mm apart; on the rim of its plate (at most 8 mm inside) | 10 mm | 1 mm |
| facing | per finger | full circle | 30° | 5° |
| tilt | per finger | -30° to +90° | 30° | 5° |
| joint axis (two angles) | per joint | any direction (a line; its sign is derived) | 30° | 5° |
| joint type | per joint | hinge, coupled (x1.1 the joint before it), sliding | couple / uncouple | - |
| link length | per link | 0 or 15-90 mm, fingertip 10-90 mm; finger at most 250 mm | 10 mm (skips 1-14) | 1 mm |
| palm hinge position | per palm joint | 15-110 mm from the wrist centre | 10 mm | 1 mm |
| palm hinge axis | per palm joint | any direction | 30° | 5° |
| link cross-section | global | rounded box 19 x 18 mm, 6 mm corners | - | - |
| palm plate | global | 37 mm thick; outline: hull of the bases and a 20 mm heel disc at the wrist | - | - |
| joint ranges | global per kind | bend -30° to +90°, spread ±30°, twist ±90°, palm ±30°, sliding -20 to +25 mm | - | - |
| physics | global | 0.5 N m, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, 1750 kg/m³, friction 0.5 | - | - |

Every coarse value lies on the fine grid. Structural operators belong to the coarse stage only.

## Mutation operators

`hand_sampler/grammar/operators.py`. Each lists its applications and offers only those that keep the rules, so nothing is generated and then rejected.

| Button | Operator | What it does |
|---|---|---|
| move a finger | `move_finger` | the base by one step along y or z |
| turn a finger | `turn_finger` | the facing by one step |
| tilt a finger | `tilt_finger` | the tilt by one step |
| lengthen or shorten a link | `length` | one link by one step (shortening 15 mm gives 0, lengthening 0 gives 15) |
| turn a joint axis | `axis_az` | the axis around the link by one step |
| tip a joint axis | `axis_el` | the axis toward or away from the link by one step |
| move / turn / tip a palm hinge | `move_hinge`, `hinge_az`, `hinge_el` | a palm joint's hinge by one step |
| add a finger | `add_finger` | a one-joint finger (bend, 40 mm) at a free spot, facing outward |
| remove a finger | `remove_finger` | any finger in one step; a palm joint left without a finger goes too |
| add a joint (split a link) | `split` | a new joint inside a link (at 0, half or all of it), kind drawn with the commercial mix; the finger keeps its length |
| remove a joint (merge two links) | `merge` | one joint; the links on either side become one |
| couple or uncouple a joint | `couple` | a joint follows the joint before it (x1.1), or moves on its own |
| give a finger its own palm joint | `own_palm_joint` | a new hinged section under the finger (hinge halfway to the wrist, along the palm) |
| move a finger to another palm part | `move_to_section` | onto another palm joint's section or back onto the main palm |
| remove a palm joint | `remove_palm_joint` | its fingers return to the main palm |

Inverse pairs: each parameter step and its opposite (away from the 0/15 mm gap and the twist pole), split and merge, add and remove a finger, couple twice, give a finger its own palm joint and remove it (or move the finger back).

## Tests

```bash
cd experiments/grammar_viewer && ../../.venv_viewer/bin/python -m pytest -q tests
```

`tests/test_viewer.py` drives the viewer's callbacks headless against a local viser server: Random under the rules, the Rules dropdown and fields, the checks, the operator buttons, mutation and Back, a commercial hand over its meshes, the pose, and the rounded-box mesh.
