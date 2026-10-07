# Grammar viewer

Interactive viser viewers for the hand-kinematics grammar (`hand_sampler/grammar/`). Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

- `viewer.py` is the essential viewer: one compact panel, in plain words, to draw a hand from a grammar variant under a set of generation limits, switch each viability check on or off, look at a commercial hand snapped onto the grammar, and apply the mutation operators.
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

## Words

The panel uses plain words; the code and the rest of this README use the grammar's terms.

| In the panel | In the code |
|---|---|
| finger | top-level digit (`Digit` with `top_level`) |
| branch finger | branch digit (a digit mounted on a phalanx) |
| joint + bone | phalanx (one module, i.e. one joint, plus its link) |
| palm | the root body |
| palm part | palm body (`PalmBody`) |
| palm joint | the joint of a jointed palm body |
| finger slot | finger chain of the simulator's 32-slot articulation |
| hinge / continuous / sliding / coupled joint | module `R` / `C` / `P` / `Coupled` |
| variant ("Basic", "+ curl and opposition", ...) | a named `Distribution` (`G_V1`, `G_V3S`, ...; the code is in the hover text) |

## Three layers: capability, limits, viability

The hand grammar separates what it can express from what we let it generate and from what we can only measure afterwards.

1. **Capability**: the grammar itself (`hand_sampler/grammar/rules.py`, `derive.py`, `distributions.py`). It can express several jointed palm bodies, stacked palm joints, coupled joints, branching digits and several digits on one palm joint.
2. **Generation limits** (`hand_sampler/grammar/limits.py`, `GenerationLimits`): hard rules that sampling (`sample_derivation(..., limits=)`) and every mutation operator (`vary`, `apply_operator`, `vary_tracked`, all with `limits=`) obey by construction. They only choose among options that keep the hand within the limits (an allowed module kind, a digit count below the cap, a host with room, a palm joint only where one is allowed, a digit on every palm leaf), so nothing is generated and then rejected. An operator with no admissible application is inapplicable, exactly like one with nothing to act on. `limits=None` (the default everywhere) reproduces the earlier behaviour byte for byte, and a limit that does not bind leaves the random draws unchanged. Where a limit binds, the sampler draws from the restricted options, so the distribution differs from rejection sampling; that is intended.
3. **Viability checks** (`gviewer/checks.py`): physical properties no generation rule can guarantee, measured with the simulator's own functions.

Three presets: `UNLIMITED` (no limit, the whole capability), `DEFAULT_LIMITS` (the whole capability, but no empty palm parts) and `SIMULATOR`, the simulator's 32-slot articulation (5 finger chains of 6 revolute slots plus 2 palm-joint slots) plus the no-empty-palm rule. `SIMULATOR_ENVELOPE` is `SIMULATOR` without that rule: a design is within it if and only if `grammar_envelope._admit_structural` admits it, and every design sampled or mutated under `SIMULATOR` is admitted (`hand_sampler/grammar_bench/tests/test_generation_limits.py`: 2000 samples per variant and 50 x 50 mutation chains, all admitted). The nine "structural checks" of the previous viewer are these limits.

| Field | Panel | Simulator | Meaning |
|---|---|---|---|
| `allowed_modules` | joint types, allow coupled joints | `("R",)` | joint module kinds that may be generated: R hinge, C continuous, P sliding, Coupled (a hinge driven by an earlier hinge) |
| `allow_branches` | allow branching fingers | `False` | digits may grow branch digits off a phalanx |
| `max_digits` | max fingers | 5 | top-level digits |
| `max_joints_per_digit` | max joints per finger | 6 | joints in one top-level digit, its branches included |
| `max_palm_bodies` | max palm parts | no limit | palm bodies besides the root (rigid ones fold into the root in the simulator) |
| `max_jointed_palm_bodies` | max palm joints | 2 | palm bodies with their own joint |
| `allow_stacked_palm_joints` | allow stacked palm joints | `False` | a jointed palm body below another jointed one |
| `max_digits_per_jointed_palm_body` | fingers per palm joint | 1 | digits carried by one jointed palm body (on it, or on rigid palm bodies below it) |
| `max_finger_chains` | max finger slots | 5 | digits on the rigid palm plus jointed palm bodies; each jointed palm body takes a slot even when it carries no digit |
| `require_digit_on_palm_body` | no empty palm parts | `True` | every palm body carries a digit (on it, or on a palm body below it); also on in `DEFAULT_LIMITS` |

How they are enforced: the digit and palm-body counts are drawn from ranges capped by the limits; a palm body is attached only where every palm leaf can still get its own digit, and gets a joint only where it is allowed (below the cap, not under another jointed body, leaving room for every digit); the first digits go one to each palm leaf, the rest only to hosts with room (the mount planners of V2 and V2s/V3s do the same); phalanx counts respect a per-digit joint budget that also covers branches; module kinds come from the allowed set; branches only when allowed. Each operator applies the same rules to what it adds or changes: `add_minimal_digit` needs a host with room, `toggle_palm_joint` only toggles a body whose new state is allowed, `insert_phalanx` only grows a digit with a joint to spare, `add_palm_body` brings a one-joint digit when palm bodies must not be empty, and removing a palm body's last digit removes the palm body. A hand already outside the limits (a commercial projection) can still be mutated, as long as no limit gets worse.

The evolution driver (`isaacsimenvs/inhand_reorient/evolution/driver.py`) is unchanged and still calls `sample_derivation(seed, dist)` and `vary(..., operators=EVOLUTION_OPERATORS)` without limits, then filters with `admit`. To generate within the envelope directly it would pass `limits=SIMULATOR` to both calls (`from hand_sampler.grammar.limits import SIMULATOR`); its `admit` call would then only ever reject on the physical checks. Without limits, the current pool never removes a palm body (see the operator table), so the driver should either pass limits or keep `operators=EVOLUTION_OPERATORS_V1`.

## The essential viewer

One panel, one line per item; longer explanations are hover text.

- **Grammar**: the variant dropdown in plain names (hover: the code and what the variant adds) and **Random**, which samples designs under the current limits until every enabled viability check passes. "Found after N tries" counts viability rejections only: every sample is within the limits by construction.
- **Limits**: a preset (Simulator, Default, Unlimited, Custom) and one dropdown or checkbox per limit. Editing a field switches the preset to Custom. Sampling and mutation use these limits. The last line says whether the hand on screen is within them.
- **Viability**: the four physical checks, one line each ("fingers don't overlap (open): PASS 0.4 mm"), each with its own toggle. Unticking a check stops Random requiring it. "show object and reach" (off by default) shows the object's start sphere, the 5 cm reach sphere and the fingertip dots (green reaches, orange does not).
- **Commercial hand**: a dropdown of the manifest hands, drawn over their real URDF meshes (a checkbox hides them). By default the hand is shown snapped onto the current variant's grids (`adapters/conform.py`), a genuine member of the grammar that every operator can mutate; "exact, off-grid version" shows the exact projection instead. One line gives the error against the URDF, whether the shown version is within the grammar's rules (and what snapping lost), and whether it is within the current limits (and which ones it breaks: SVH carries two fingers on one palm joint).
- **Mutation**: **Random mutation** (draws operators until one applies), **Back** (undo), and one button per operator, shown only when it can act on this hand under these limits. After each mutation a line says what changed; the parent stays on screen as a faint grey ghost.
- **Pose**: one curl slider (0 lower limits, 1 upper limits; the default 0.35 is the pose every episode starts from) and **re-centre view**.

What is on screen: one colour per finger (a branch finger a lighter shade of its finger's colour; colours follow the finger, so removing one finger never recolours the others), neutral grey palm parts, small dark joint markers, the grey ghost of the parent after a mutation, and red only on the links of a failing overlap check that is switched on. The palm stays fixed in the world, and the camera only moves for a new draw, a new commercial hand or **re-centre view**.

## Viability checks

Only properties no generation rule can guarantee. They need the simulator's 32-slot model of the hand, which exists only for hands within the Simulator limits; on any other hand they read n/a and do not block Random. A design is viable when it passes all four; under the Simulator limits that is exactly `viability_report`'s admitted flag plus "at least 2 fingertips reach" (`tests/test_checks.py`).

| Panel | Key | What it requires | Why |
|---|---|---|---|
| fingers don't overlap (open) | `overlap_zero` | no two capsules interpenetrate by more than 3 mm with every joint at 0 | deeper starting overlaps made the physics engine push links apart at over 100 rad in one or two steps |
| fingers don't overlap (start pose) | `overlap_reset` | the same with every joint 35% of the way through its range | every episode starts from this pose |
| object starts above palm* | `spawn_height` | with the hand turned palm-up, the object's start point is 20 mm or more above the palm | the hand must hold the object up against gravity |
| ≥2 fingertips reach object* | `reach` | in a 4000-sample random sweep of the joints, 2 or more fingertips come within 5 cm of the start point | a hand that cannot touch the object with two fingers cannot turn it |

\* Provisional, to be reworked: the spawn and reach checks (a sphere around a cube-sized start point) are placeholders for a better measure of whether a hand can hold and turn the object.

## Mutation operators

`derive.EVOLUTION_OPERATORS`, the evolution driver's pool, has 17 operators. It changed on 2026-10-06; the earlier pool is frozen as `derive.EVOLUTION_OPERATORS_V1` so runs made with it reproduce (pass `operators=EVOLUTION_OPERATORS_V1`). The changes: `step_segment_length` joined; "remove a finger" is now `remove_digit` (any finger, any length) instead of `remove_digit_minimal` (only 1-2 joints), so "add a short finger" and "remove a finger" are no longer an exact inverse pair; and `remove_palm_body_empty` left the pool, because with the no-empty-palm rule no palm part is ever empty (removing a palm part's last finger removes it). `derive.EVOLUTION_PAIRS` still lists V1's exact-inverse pairs, which E2/E12 report on.

| Button | Operator | What it does | Limited by (Simulator) |
|---|---|---|---|
| add a short finger (1 joint) | `add_minimal_digit` | new finger with one hinge joint and one bone, on the palm or a palm part | fingers, finger slots, fingers per palm joint (needs a host with room); needs hinges |
| remove a finger | `remove_digit` | removes one finger of any length, with its branch fingers; a palm part left without a finger goes too | at least one finger stays; no limit may get worse |
| add a joint to a finger | `insert_phalanx` | inserts a joint and bone at a random place in one finger; the bones beyond it move out | joints per finger |
| remove a joint from a finger | `delete_phalanx` | removes one joint and its bone from a finger with at least 2; the bones beyond it move in and stay attached; branch fingers on it re-attach to the neighbouring bone; a coupled joint that loses its driver becomes a new independent joint | none |
| add a palm part (with a short finger) | `add_palm_body` | adds a palm part on the palm or another palm part, jointed or rigid at random where a joint is allowed; with the no-empty-palm rule it comes with a one-joint finger, and the label says so | palm parts; palm joints, stacking, finger slots (for the joint); room for its finger |
| make a palm part rigid/jointed | `toggle_palm_joint` | gives a rigid palm part a joint with a new axis and range, or makes a jointed one rigid | palm joints, stacking, fingers per palm joint, finger slots |
| add a branch finger (1 joint) | `add_branch_digit` | adds a one-joint finger growing off a bone of a finger | branching (never under Simulator), joints per finger |
| remove a short branch finger | `remove_branch_digit` | removes a branch finger with one joint | none |
| tilt one joint axis | `step_axis` | tilts one joint's axis by one 15 deg step | none |
| change one joint's range | `step_limits` | moves one joint's range to the neighbouring option of the variant | none |
| move a mount (finger or palm part) | `step_mount` | slides one finger or palm part along what it is attached to, or turns it 15 deg at its base | none |
| change one coupled joint | `step_coupling` | steps a coupled joint's ratio or offset | none (no coupled joints under Simulator) |
| lengthen/shorten the palm | `step_root_length` | changes the palm's length by 5 mm, within 20-80 mm; the fingers keep their relative place along it, so they move with it | none |
| thicker/thinner (all links) | `step_radius` | every link, palm included, shares one thickness; steps it to 8, 10 or 12 mm | none |
| change one joint's rest bend | `step_bend_rpy` | turns the rest angle between two bones by 15 deg (variants with rest bends) | none |
| shift one joint sideways | `step_bend_offset` | shifts one joint sideways by 5 mm (only `G_BEND` has rest offsets) | none |
| lengthen/shorten one bone (5 mm) | `step_segment_length` | changes one bone's (or palm part's) length by exactly one 5 mm step within the variant's range (15-80 mm, palm parts 20-80 mm); parts beyond it move with it; `lengthen_segment`/`shorten_segment` are its two directions as an exact inverse pair | none (no limit concerns lengths) |

The older `perturb_parameter` (also reachable as `step_length`, outside the pool) reflects off a range bound, so its step is not always exactly 5 mm; `step_segment_length` only offers moves that stay in range.

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

The essential viewer's colours are described under "The essential viewer". The full viewer keeps its analysis colours:

- Capsules are coloured per top-level digit; branch digits use a lighter shade of their host digit. Palm bodies are translucent grey nearest-spine cells (the cells of `geometry.build_geometry`).
- Joint arrows point along the joint axis (right-hand rule). At the rest pose, each digit joint is compared with its own link direction and the palm normal: blue is flexion, orange is abduction, blended by angle like the old sampler viewer. Purple is a twist joint (axis within 35 deg of its link), teal a palm joint.
- Fingertips are green when the oracle's sweep reaches the spawn point, orange when it does not. The blue sphere is the spawn point at the object's size (3 cm half size); the wire sphere is the 5 cm reach tolerance.
- Overlapping capsules are red beyond the oracle's 3 mm gate; shallower overlaps are pink, and the highlight follows the current pose by default or the oracle's own poses (q=0 and reset).
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
