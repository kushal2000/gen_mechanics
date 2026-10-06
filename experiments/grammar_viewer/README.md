# Grammar viewer

Interactive viser viewers for the hand-kinematics grammar (`hand_sampler/grammar/`). Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

- `viewer.py` is the essential viewer: one compact panel to draw a design from a grammar variant under a set of generation limits, switch each viability check on or off, look at a commercial hand's projection, and apply the mutation operators.
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

## Three layers: capability, limits, viability

The hand grammar separates what it can express from what we let it generate and from what we can only measure afterwards.

1. **Capability**: the grammar itself (`hand_sampler/grammar/rules.py`, `derive.py`, `distributions.py`). It can express several jointed palm bodies, stacked palm joints, coupled joints, branching digits and several digits on one palm joint.
2. **Generation limits** (`hand_sampler/grammar/limits.py`, `GenerationLimits`): hard rules that sampling (`sample_derivation(..., limits=)`) and every mutation operator (`vary`, `apply_operator`, `vary_tracked`, all with `limits=`) obey by construction. They only choose among options that keep the hand within the limits (an allowed module kind, a digit count below the cap, a host with room, a palm joint only where one is allowed), so nothing is generated and then rejected. An operator with no admissible application is inapplicable, exactly like one with nothing to act on. `limits=None` (the default everywhere) reproduces the earlier behaviour byte for byte, and a limit that does not bind leaves the random draws unchanged. Where a limit binds, the sampler draws from the restricted options, so the distribution differs from rejection sampling; that is intended.
3. **Viability checks** (`gviewer/checks.py`): physical properties no generation rule can guarantee, measured with the simulator's own functions.

Two presets: `UNLIMITED` (no limit, the whole capability) and `SIMULATOR`, exactly the simulator's 32-slot articulation (5 finger chains of 6 revolute slots plus 2 palm-joint slots). A design is within `SIMULATOR` if and only if `grammar_envelope._admit_structural` admits it, and every design sampled or mutated under `SIMULATOR` is admitted (`hand_sampler/grammar_bench/tests/test_generation_limits.py`: 2000 samples per variant and 50 x 50 mutation chains, all admitted). The nine "structural checks" of the previous viewer are these limits.

| Field | Simulator | Meaning |
|---|---|---|
| `allowed_modules` | `("R",)` | joint module kinds that may be generated: R hinge, C continuous, P sliding, Coupled (a hinge driven by an earlier hinge) |
| `allow_branches` | `False` | digits may grow branch digits off a segment |
| `max_digits` | 5 | top-level digits (fingers) |
| `max_joints_per_digit` | 6 | joints in one finger, its branches included |
| `max_palm_bodies` | no limit | palm bodies besides the root palm (rigid ones fold into the root in the simulator) |
| `max_jointed_palm_bodies` | 2 | palm bodies with their own joint |
| `allow_stacked_palm_joints` | `False` | a jointed palm body below another jointed one |
| `max_digits_per_jointed_palm_body` | 1 | fingers carried by one jointed palm body (on it, or on rigid palm bodies below it) |
| `max_finger_chains` | 5 | fingers on the rigid palm plus jointed palm bodies; each jointed palm body reserves a chain even when it carries no finger |

How they are enforced: the digit and palm-body counts are drawn from ranges capped by the limits; a palm body gets a joint only where it is allowed (below the cap, not under another jointed body, and leaving room for every finger); each top-level finger mounts only on a host with room (the mount planners of V2 and V2s/V3s skip full hosts); phalanx counts respect a per-finger joint budget that also covers branches; module kinds come from the allowed set; branches only when allowed. Each operator applies the same rules to what it adds or changes (for example `add_minimal_digit` needs a host with room, `toggle_palm_joint` only toggles a body whose new state is allowed, `insert_phalanx` only grows a finger with a joint to spare). A hand already outside the limits (a commercial projection) can still be mutated, as long as no limit gets worse.

The evolution driver (`isaacsimenvs/inhand_reorient/evolution/driver.py`) is unchanged and still calls `sample_derivation(seed, dist)` and `vary(..., operators=EVOLUTION_OPERATORS)` without limits, then filters with `admit`. To generate within the envelope directly it would pass `limits=SIMULATOR` to both calls (`from hand_sampler.grammar.limits import SIMULATOR`); its `admit` call would then only ever reject on the physical checks.

## The essential viewer

One panel, one line per item; longer explanations are hover text.

- **Grammar**: the variant dropdown (hover: what the variant adds) and **Random**, which samples designs under the current limits until every enabled viability check passes. "Found after N tries" counts viability rejections only: every sample is within the limits by construction.
- **Limits**: a preset (Simulator, Unlimited, Custom) and one dropdown or checkbox per limit (fingers, joints per finger, palm bodies, palm joints, fingers per palm joint, finger chains, joint types, coupled joints, branches, stacked palm joints). Editing a field switches the preset to Custom. Sampling and mutation use these limits. The last line says whether the hand on screen is within them.
- **Viability**: the four physical checks, one line each ("overlap, zero pose: PASS 0.4 mm"), each with its own toggle. Unticking a check stops Random requiring it. Capsules overlapping by more than 3 mm are red when an overlap check fails. "show cube and reach" toggles the spawn sphere (the cube's size), the 5 cm reach sphere and the fingertip colours (green reaches, orange does not).
- **Commercial hand**: a dropdown of the manifest hands. The grammar projection is drawn over the real URDF meshes (one checkbox hides them), with one line: the projection's error against the URDF and whether it is within the current limits, and which ones it breaks if not (a commercial hand can break the Simulator limits, e.g. SVH carries two fingers on one palm joint).
- **Mutation**: **Random mutation** (draws operators until one applies), **Back** (undo), and one button per operator. A button is greyed out when the operator cannot act on this hand; its hover text says whether that is "not allowed by limits" or "nothing to act on". After each mutation a line says what changed; the parent stays on screen as a faint grey ghost.
- **Pose**: one curl slider, the fraction of every joint's range (0 lower limits, 1 upper limits). The default 0.35 is the reset pose every episode starts from.

## Viability checks

Only properties no generation rule can guarantee. They need the simulator's 32-slot model of the hand, which exists only for hands within the Simulator limits; on any other hand they read n/a and do not block Random. A design is viable when it passes all four; under the Simulator limits that is exactly `viability_report`'s admitted flag plus "at least 2 fingertips reach" (`tests/test_checks.py`).

| Check | What it requires | Why |
|---|---|---|
| overlap, zero pose | no two capsules interpenetrate by more than 3 mm with every joint at 0 | deeper starting overlaps made the physics engine push links apart at over 100 rad in one or two steps |
| overlap, reset pose | the same with every joint 35% of the way through its range | every episode starts from this pose |
| spawn above palm* | with the hand turned palm-up, the cube's spawn point is 20 mm or more above the palm | the hand must hold the cube up against gravity |
| 2 tips reach cube* | in a 4000-sample random sweep of the joints, 2 or more fingertips come within 5 cm of the spawn point | a hand that cannot touch the cube with two fingers cannot turn it |

\* Provisional, to be reworked: the spawn and reach checks (a sphere around a cube-sized spawn point) are placeholders for a better measure of whether a hand can hold and turn the object.

## Mutation operators

These are the 18 operators of `derive.EVOLUTION_OPERATORS`, the evolution driver's pool: five grow/shrink pairs (`toggle_palm_joint` is its own inverse) and nine small steps that move one value to its neighbour on the variant's menu. `step_segment_length` joined the pool on 2026-10-06; the earlier 17-operator pool is kept as `derive.EVOLUTION_OPERATORS_V1`, so runs made with it can be reproduced by passing `operators=EVOLUTION_OPERATORS_V1`. Under limits each one acts only in ways the limits allow; the last column says which limits can make it inapplicable (Simulator values).

| Button | Operator | What it does | Limited by |
|---|---|---|---|
| add a digit | `add_minimal_digit` | new one-segment digit (one hinge) on the root palm or a palm body | fingers, finger chains, fingers per palm joint (needs a host with room); needs R |
| remove a short digit | `remove_digit_minimal` | removes a digit with 1-2 segments and no branches | none |
| add a segment | `insert_phalanx` | inserts a new segment (link and joint) somewhere in one digit; its module from the allowed kinds | joints per finger |
| remove a segment | `delete_phalanx` | deletes one segment of a digit with at least 2; branches on it re-attach | none |
| add a palm body | `add_palm_body` | adds a palm piece on the root or another palm body, jointed with the variant's palm-joint probability where a joint is allowed, rigid otherwise | palm bodies; palm joints, stacking, finger chains (for the joint) |
| remove an empty palm body | `remove_palm_body_empty` | removes a palm body that carries nothing (the exact undo of "add a palm body") | none |
| joint/unjoint a palm body | `toggle_palm_joint` | gives a rigid palm body a joint with a new axis and limits, or makes a jointed one rigid | palm joints, stacking, fingers per palm joint, finger chains |
| add a branch digit | `add_branch_digit` | adds a one-segment digit growing off an existing segment | branches (always inapplicable under Simulator), joints per finger |
| remove a branch digit | `remove_branch_digit` | removes a one-segment branch digit | none |
| tilt one joint axis | `step_axis` | moves one joint's axis by one 15 deg step in elevation or azimuth | none |
| change one joint's limits | `step_limits` | moves one joint's limits to the neighbouring choice on the variant's menu | none |
| move one mount | `step_mount` | moves where one digit or palm body attaches: one step along its host, or one 15 deg step of its mount angle | none |
| change one coupling | `step_coupling` | steps a coupled joint's ratio or offset to the neighbouring choice | none (no coupled joints exist under Simulator) |
| lengthen/shorten the palm | `step_root_length` | changes the root palm's length by one 5 mm step, within 20-80 mm | none |
| thicken/thin every link | `step_radius` | moves the hand's single capsule radius to the neighbouring choice (8, 10 or 12 mm) | none |
| change one rest bend angle | `step_bend_rpy` | steps one angle of one segment's rest bend (variants with a bend menu: `G_BEND`, V3, V3s) | none |
| change one rest bend offset | `step_bend_offset` | steps one component of one segment's rest bend offset (only `G_BEND` has an offset menu) | none |
| lengthen/shorten one segment (5 mm) | `step_segment_length` | changes one existing segment's length (a finger link or an extra palm body) by exactly one 5 mm grid step, within the variant's range (15-80 mm for links, 20-80 mm for palm bodies); a move past the range is never offered, and `lengthen_segment`/`shorten_segment` are its two directions as an exact inverse pair | none (no limit concerns lengths) |

Before `step_segment_length`, no operator in the pool changed an existing segment's length; only the root palm's moved (`step_root_length`). The older `perturb_parameter` (also reachable as `step_length`, outside the pool) reflects off a range bound, so its step is not always exactly 5 mm; `step_segment_length` only offers moves that stay in range.

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

`pytest.ini` switches off the ROS pytest plugins that leak in through `PYTHONPATH`. The tests cover the render primitives (against `forward_kinematics` and `build_geometry`), the envelope loader (against `viability_report` and the archive bins), the viability checks (against `viability_report`, and their early-stop path), the commercial fidelity (equal to E13's `check_hand` without its Pinocchio part) for allegro, sharpa and svh, the mesh-overlay poses, the history round trip and diff, file parsing, and the callbacks of both viewers against a local server: limits presets and Custom, Random under the limits, each viability toggle changing Random's tries, operator buttons greyed out under the limits. The generation limits themselves are tested in `hand_sampler/grammar_bench/tests/test_generation_limits.py`.

## Known limitations

- The Pinocchio export cross-check of E13 is not run (it needs a separate interpreter); the readout says so.
- `arms_skel` has no meshes at all. The SVH and Shadow URDFs in `karma-hand-metric` point at mesh files that do not exist; the viewer finds same-named `.dae` files in the downloads (`SVH/`, `Shadow/`) and marks them "alignment unverified". About 10 SVH meshes have no same-named file.
- Mesh overlays are reduced by vertex clustering to at most 6000 faces per piece; they are a visual reference.
- Projected hands use one capsule radius (10 mm, the projection's constant), so their rest overlaps are artefacts; the simulator exempts them, the overlap checks and the full viewer's Analysis tab still count them.
- The viability checks need the simulator's 32-slot model, so under Unlimited or Custom limits a hand the simulator cannot build reads n/a on all four.
- Nearest commercial: `phenotype_distance` aligns joints by name. Grammar and projection both name digit joints `d{k}p{i}_j`, but digit numbering is arbitrary and the two root frames follow different gauges, so treat it as a coarse similarity.
- Joint classes are computed at the rest pose. Allegro's first finger joints read as twist because at rest they rotate the straight finger about its own axis.
- The envelope's ghost and padding slots are never drawn: the scene is the grammar model, not the authored articulation.
