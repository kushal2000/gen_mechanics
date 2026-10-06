# Grammar viewer

Interactive viser viewers for the hand-kinematics grammar (`hand_sampler/grammar/`). Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

- `viewer.py` is the essential viewer: one panel to draw a design from a grammar variant, look at a commercial hand's projection, switch each viability check on or off, and apply the mutation operators.
- `viewer_full.py` is the full tool: seeds, gallery, mutation spread, per-joint sliders, analysis, views and export.

## Setup (once)

```bash
cd "/home/singularity/Evolve to Generalize/gen_mechanics"
/home/singularity/.local/bin/uv venv .venv_viewer --python 3.11
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python viser "numpy==1.26.*" scipy trimesh yourdfpy pycollada pytest
/home/singularity/.local/bin/uv pip install --python .venv_viewer/bin/python -e . --no-deps
```

`pycollada` is optional; without it the `.dae` meshes of SVH and Shadow are reported as unloadable. `.venv_viewer/` is git-ignored by the existing `.venv*/` rule.

## Run

```bash
.venv_viewer/bin/python experiments/grammar_viewer/viewer.py --port 8080 --host 127.0.0.1
```

Open http://127.0.0.1:8080. From a laptop: `ssh -L 8080:127.0.0.1:8080 <this machine>`, then open the same URL locally. `--variant G_V1` picks the starting variant (default `G_V3S`). The full viewer takes the same `--port`/`--host` plus `--seed`, `--no-until-viable` and `--out-dir`:

```bash
.venv_viewer/bin/python experiments/grammar_viewer/viewer_full.py --port 8081 --host 127.0.0.1
```

## The essential viewer

One panel, top to bottom:

- **Grammar**: a variant dropdown with a one-line note on the rule it adds, and **Random**. Random draws designs (consecutive seeds from a random start) until every enabled viability check passes, and reports "found after N tries". Changing the variant draws a new design.
- **Commercial hand**: a dropdown of the manifest hands. The hand's grammar projection is drawn as capsules over its real URDF meshes (one checkbox hides the meshes), with one line on fidelity: how closely the projection matches the URDF, and whether the simulator accepts it.
- **Viability checks**: one checkbox per check (all on by default) with the current hand's PASS/FAIL and value, and a verdict line. Switching a check off changes what Random accepts. Capsules overlapping by more than 3 mm are red when an overlap check fails. Under the reach check, "show cube and reach" toggles the spawn sphere (the cube's size), the 5 cm reach sphere and the fingertip colours (green reaches, orange does not).
- **Mutation**: one button per operator in the evolution driver's pool (`derive.EVOLUTION_OPERATORS`), **Random mutation** (draws operators until one applies) and **Back** (undo). After each mutation a line says what changed and whether the child passes the enabled checks; the parent stays on screen as a faint grey ghost. An operator with nothing to act on says so and leaves the hand unchanged.
- **Pose**: one curl slider, the fraction of every joint's range (0 lower limits, 1 upper limits). The default 0.35 is the reset pose every episode starts from.

## What the viability checks are

A design is viable when it passes all 13 checks; together they are exactly `viability_report`'s admitted flag plus "at least 2 fingertips reach" (`tests/test_checks.py` checks this on sampled designs). The first nine are simulator constraints: the simulator builds every hand from one fixed articulation of 32 revolute joint slots (5 finger chains of 6 slots, plus 2 palm-joint slots), and a design that does not fit that shape cannot be built at all. `gviewer/checks.py` evaluates each one on its own; they are the nine rejection reasons of `_admit_structural` (with `fits_envelope`).

| Check | What it requires | Why |
|---|---|---|
| revolute joints only | every movable joint is a hinge | the slots are revolute joints; sliding (prismatic) and continuous joints have no slot |
| no coupled joints | no joint follows another (mimic) | every slot has its own motor |
| no branching digits | no link carries two child joints | a digit is laid out as one serial chain of slots |
| at most 5 digits | 5 or fewer digits mounted on palm bodies | there are 5 finger chains |
| at most 6 joints per digit | 6 or fewer joints in each digit | each finger chain has 6 slots |
| at most 2 jointed palm bodies | 2 or fewer palm bodies with their own joint | there are 2 palm-joint slots |
| no stacked palm joints | no jointed palm body sits on another jointed one | both palm-joint slots hang off the root palm |
| at most 1 digit per jointed palm body | a jointed palm body carries 0 or 1 digits | each palm-joint slot carries one finger chain (chains 3 and 4) |
| root digits + jointed palm bodies at most 5 | digits on the rigid palm plus jointed palm bodies fit in 5 chains | each jointed palm body reserves a finger chain, even when it carries no digit |

The last four are physical:

| Check | What it requires | Why |
|---|---|---|
| no overlap > 3 mm at the zero pose | no two capsules interpenetrate by more than 3 mm with every joint at 0 | deeper starting overlaps made the physics engine push links apart at over 100 rad in one or two steps |
| no overlap > 3 mm at the reset pose | the same with every joint 35% of the way through its range | every episode starts from this pose |
| spawn point at least 20 mm above the palm | with the hand turned palm-up, the cube's spawn point is 20 mm or more above the palm | the hand must hold the cube up against gravity; below the palm the cube falls away |
| at least 2 fingertips reach the cube | in a 4000-sample random sweep of the joints, 2 or more fingertips come within 5 cm of the spawn point | a hand that cannot touch the cube with two fingers cannot turn it |

The physical checks need the simulator's 32-slot model of the hand, which only exists when every structural check passes. On a hand that fails a structural check they read n/a. With that structural check switched off, Random accepts such a hand without the physical checks, and the panel says so.

## Mutation operators

These are the 17 operators of `derive.EVOLUTION_OPERATORS`, the evolution driver's pool: five grow/shrink pairs (`toggle_palm_joint` is its own inverse) and eight small steps that move one value to its neighbour on the variant's menu. Operators that act on something a hand lacks (couplings on a revolute-only hand, bend offsets outside `G_BEND`) report that they cannot apply.

| Button | Operator | What it does |
|---|---|---|
| add a digit | `add_minimal_digit` | new one-segment digit (one hinge) on the root palm or a palm body |
| remove a short digit | `remove_digit_minimal` | removes a digit with 1-2 segments and no branches |
| add a segment | `insert_phalanx` | inserts a new segment (link and joint) somewhere in one digit |
| remove a segment | `delete_phalanx` | deletes one segment of a digit with at least 2; branches on it re-attach |
| add a palm body | `add_palm_body` | adds a palm piece on the root or another palm body, jointed with the variant's palm-joint probability (55% in `G_FULL`) |
| remove an empty palm body | `remove_palm_body_empty` | removes a palm body that carries nothing (the exact undo of "add a palm body") |
| joint/unjoint a palm body | `toggle_palm_joint` | gives a rigid palm body a joint with a new axis and limits, or makes a jointed one rigid |
| add a branch digit | `add_branch_digit` | adds a one-segment digit growing off an existing segment |
| remove a branch digit | `remove_branch_digit` | removes a one-segment branch digit |
| tilt one joint axis | `step_axis` | moves one joint's axis by one 15 deg step in elevation or azimuth |
| change one joint's limits | `step_limits` | moves one joint's limits to the neighbouring choice on the variant's menu |
| move one mount | `step_mount` | moves where one digit or palm body attaches: one step along its host, or one 15 deg step of its mount angle |
| change one coupling | `step_coupling` | steps a coupled joint's ratio or offset to the neighbouring choice |
| lengthen/shorten the palm | `step_root_length` | changes the root palm's length by one 5 mm step, within 20-80 mm |
| thicken/thin every link | `step_radius` | moves the hand's single capsule radius to the neighbouring choice (8, 10 or 12 mm) |
| change one rest bend angle | `step_bend_rpy` | steps one angle of one segment's rest bend (variants with a bend menu: `G_BEND`, V3, V3s) |
| change one rest bend offset | `step_bend_offset` | steps one component of one segment's rest bend offset (only `G_BEND` has an offset menu) |

No operator in this pool changes the length of an existing segment or extra palm body; only the root palm's length moves (`step_root_length`). `derive.py` has a length step (`perturb_parameter`, 5 mm, also reachable as `step_length`), but it is outside `EVOLUTION_OPERATORS`: `step_length` was dropped from the small-step pool as a duplicate of `perturb_parameter`, which belongs only to the older default pool `OPERATORS`. A new segment gets a fresh length when it is inserted.

## Full viewer panels

- **Source**
  - (a) Sample: variant dropdown (all of `NAMED_DISTRIBUTIONS`, G0 aliases in brackets), seed, prev/next/random. With "sample until viable" on, seeds are drawn upward (downward for prev) until `viability_report` admits a design with at least 2 fingertips reaching the spawn; the tries are shown.
  - (b) Commercial hand: every manifest hand outside the `excluded` split. The E13 projection is drawn as capsules over the original URDF meshes (toggle, opacity). The readout gives the fidelity (E13's measurement replayed: zero plus 64 random configurations, joint position and axis errors, fingertip error, pass at 5 mm / 10 deg), the 32-slot envelope fit with its rejection reasons, the capsule radius, `coverage` against the mutation distribution, and every merge, approximation and importer loss.
  - (c) File: a derivation JSON, a `population.json` (`designs[].derivation`), a driver `state.json` (archive elites) or a bare archive, browsable by index. A state file also sets the mutation distribution to its run's variant.
- **Pose**: a slider per independent joint (degrees, within its admissible range), curl-all (fraction of range), zero, env reset (`palm_up(...).default_q`), random, and an animated sweep (all joints together, or one joint at a time).
- **Mutate**: one button per `EVOLUTION_OPERATORS` entry, random mutation (the driver's pool), mutate until viable, a mutation distribution, back/forward and go-to-entry over the history, a lineage list labelled by operator, the parent drawn as a grey ghost, a parent-to-child diff (digits, phalanges, joints, changed parameters matched by step uid), and the mutation spread (K random children on a ring, coloured by viability, click one to adopt it).
- **Analysis**: `viability_report` (admitted, reasons, worst rest overlap, reaching fingertips per finger, spawn height), overlap pairs at q=0 and at the reset pose, live overlaps at the current pose, the MAP-Elites cell (bins from `evolution/archive.py`), the envelope slot map, counts, and the nearest commercial hands by `phenodist`.
- **View**: camera presets (top, side, palm-up), root frame or palm-up orientation (the env's `base_rot`), capsule convention, layer toggles, body and joint labels.
- **Gallery**: a rows by columns grid from one variant, or two variants one row each with the same seeds, coloured by viability. Click a hand to open it.
- **Export**: the derivation (plus a `.meta.json` with the report and lineage), a URDF via `adapters/urdf.to_urdf` with capsule collisions and palm-cell OBJ meshes, or the history. Files go to `outputs/grammar_viewer/` (git-ignored).

## What the colours mean

- Capsules are coloured per top-level digit; branch digits use a lighter shade of their host digit. Palm bodies are translucent grey nearest-spine cells (the cells of `geometry.build_geometry`).
- Joint arrows point along the joint axis (right-hand rule). At the rest pose, each digit joint is compared with its own link direction and the palm normal: blue is flexion, orange is abduction, blended by angle like the old sampler viewer. Purple is a twist joint (axis within 35 deg of its link), teal a palm joint.
- Fingertips are green when the oracle's sweep reaches the spawn point, orange when it does not. The blue sphere is the spawn point at the object's size (3 cm half size); the wire sphere is the 5 cm reach tolerance.
- Overlapping capsules are red beyond the oracle's 3 mm gate. In the essential viewer they mark the pairs of a failing overlap check (at that check's pose, whatever the curl slider shows). In the full viewer shallower overlaps are pink, and the highlight follows the current pose by default or the oracle's own poses (q=0 and reset).
- Capsule convention `simulator` is the PhysX capsule `rest_overlap_pairs` checks (total extent [0, L]); `grammar` is `geometry.Capsule` and the URDF export (core [0, L], extent [-r, L+r]).

## Tests

```bash
.venv_viewer/bin/python -m pytest experiments/grammar_viewer/tests -q
```

`pytest.ini` switches off the ROS pytest plugins that leak in through `PYTHONPATH`. The tests cover the render primitives (against `forward_kinematics` and `build_geometry`), the envelope loader (against `viability_report` and the archive bins), the per-check evaluation (against `_admit_structural` and `viability_report`, and its early-stop path), the commercial fidelity (equal to E13's `check_hand` without its Pinocchio part) for allegro, sharpa and svh, the mesh-overlay poses, the history round trip and diff, file parsing, and the callbacks of both viewers against a local server, including that switching each check on changes Random's tries.

## Known limitations

- The Pinocchio export cross-check of E13 is not run (it needs a separate interpreter); the readout says so.
- `arms_skel` has no meshes at all. The SVH and Shadow URDFs in `karma-hand-metric` point at mesh files that do not exist; the viewer finds same-named `.dae` files in the downloads (`SVH/`, `Shadow/`) and marks them "alignment unverified". About 10 SVH meshes have no same-named file.
- Mesh overlays are reduced by vertex clustering to at most 6000 faces per piece; they are a visual reference.
- Projected hands use one capsule radius (10 mm, the projection's constant), so their rest overlaps are artefacts; the simulator exempts them, the overlap checks and the full viewer's Analysis tab still count them.
- Nearest commercial: `phenotype_distance` aligns joints by name. Grammar and projection both name digit joints `d{k}p{i}_j`, but digit numbering is arbitrary and the two root frames follow different gauges, so treat it as a coarse similarity.
- Joint classes are computed at the rest pose. Allegro's first finger joints read as twist because at rest they rotate the straight finger about its own axis.
- The envelope's ghost and padding slots are never drawn: the scene is the grammar model, not the authored articulation.
