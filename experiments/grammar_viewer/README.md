# Grammar viewer

Interactive viser viewers for the hand-kinematics grammar (`hand_sampler/grammar/`). Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

- `viewer.py` is the essential viewer: one compact panel, in plain words, to draw a hand from the grammar (with its three rules) under a set of generation limits, switch each viability check on or off, look at a commercial hand snapped onto the grammar, and apply the mutation operators.
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

Open http://127.0.0.1:8080. From a laptop: `ssh -L 8080:127.0.0.1:8080 <this machine>`, then open the same URL locally. `--rules surface,curl_opposition` picks the rules switched on at start (default: all three). The full viewer takes the same `--port`/`--host` plus `--seed`, `--no-until-viable` and `--out-dir`:

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
| the grammar, its rules ("fingers sit on the palm surface", ...) | `variants.build_distribution(surface=, spacing=, curl_opposition=)` |

## One grammar, three rules

The viewer has one grammar, `variants.build_distribution(surface=True, spacing=True, curl_opposition=True)`: one base distribution, `GRAMMAR_BASE`, plus three generation rules that shape where random hands put their fingers.

- `GRAMMAR_BASE` is the default grammar (G_FULL's productions and sampling priors) with capability grids wide enough to contain the commercial hands: mount positions in 5% steps along a host, a 5 mm lateral mount grid up to 65 mm for fingers and palm parts, a rest bend at any joint on the 15 deg rotation grid, and joint ranges anywhere in +/-180 deg. The last three are support only (`Distribution.mount_lateral_sampled=False`, `bend_support_rpy_choices_rad`, `limits_support_continuous`): mutation reaches them and a conformed real hand may use them, but random sampling never draws them, so a random hand looks exactly as the rules say (`test_one_grammar.py` checks the draws are unchanged).
- The rules, each one checkbox in the panel: "fingers sit on the palm surface" (`surface`, was V1s: a finger's base on its host's capsule surface at a 15 deg angle), "fingers spaced apart" (`spacing`, was V2/V2s: finger bases planned at least 29 mm apart, in 3-D across all hosts when on the surface) and "fingers curl and oppose" (`curl_opposition`, was V3s: hinge axes roughly across the bone, every bone after the first curled 15-45 deg, the last finger turned to oppose the others). All three are on by default. That is V1s + spacing + V3s's curl and opposition on the wide grids; it is not byte-identical to `G_V3S` (different base counts and mount steps, plus spacing), but its random hands have the same rule properties (`test_one_grammar.py`).
- Restrictions such as hinge joints only, no branching, at most 5 fingers or 2 palm joints are generation limits (below), not grammars.

The named variants of `variants.NAMED_DISTRIBUTIONS` are historical, kept unchanged (byte for byte) for reproducing past experiments, and are not shown in the viewer:

| Variant | What it was | Now |
|---|---|---|
| `G_FULL` (= `DEFAULT_DISTRIBUTION`) | the default grammar | the base of `GRAMMAR_BASE` |
| `G_SERIAL`, `G_NOPALMJOINT`, `G_NOBRANCH`, `G_NOCOUPLE` | G_FULL without palm parts / palm joints / branches / coupled joints | limits: max palm parts 0, max palm joints 0, allow branching off, joint types without coupled |
| `G_V1` | G_FULL restricted to what the simulator builds | the grammar under the Simulator limits |
| `G_V1S`, `G_V2S`, `G_V3S` | V1 + surface mounts; + spacing; + curl and opposition | the three rules |
| `G_V2`, `G_V3` | first versions of spacing and curl (non-surface; V3 with the host-frame bug) | superseded by V2s/V3s |
| `G_FULL_INS`, `G_NOBRANCH_INS` | growth operators insert small pieces | experiment settings (E2/E3) |
| `G_BEND`, `G_CONT` | random rest bends; continuous joint ranges | support grids of `GRAMMAR_BASE` (not sampled) |
| `G_WIDE` | G_FULL with the grids widened and sampled | `GRAMMAR_BASE` has the same grids as support only |

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

- **Grammar**: **Random**, which draws hands from the grammar under the current limits until every enabled viability check passes, the three rule checkboxes (switching one draws a new hand), and "found after N tries", which counts viability rejections only: every draw is within the limits by construction.
- **Limits**: a preset (Simulator, Default, Unlimited, Custom) and one dropdown or checkbox per limit. Editing a field switches the preset to Custom. Sampling and mutation use these limits. The last line says whether the hand on screen is within them.
- **Viability**: the four physical checks, one line each ("fingers don't overlap (open): PASS 0.4 mm"), each with its own toggle. Unticking a check stops Random requiring it. "show object and reach" (off by default) shows the object's start sphere, the 5 cm reach sphere and the fingertip dots (green reaches, orange does not).
- **Commercial hand**: a dropdown of the manifest hands, drawn over their real URDF meshes (a checkbox hides them). By default the hand is shown snapped onto the grammar's grids (`adapters/conform.py`), a genuine member of the grammar that every operator can mutate; "exact, off-grid version" shows the exact projection instead. One line gives the error against the URDF, whether the shown version is within the grammar's rules (and what snapping lost), and whether it is within the current limits (and which ones it breaks: SVH carries two fingers on one palm joint).
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
| change one joint's range | `step_limits` | moves one joint's range to the neighbouring menu option, or (a range off the menu, e.g. a real hand's) one end by 15 deg within +/-180 deg | none |
| move a mount (finger or palm part) | `step_mount` | slides one finger or palm part along what it is attached to, or turns it 15 deg at its base | none |
| change one coupled joint | `step_coupling` | steps a coupled joint's ratio or offset | none (no coupled joints under Simulator) |
| lengthen/shorten the palm | `step_root_length` | changes the palm's length by 5 mm, within 20-80 mm; the fingers keep their relative place along it, so they move with it | none |
| thicker/thinner (all links) | `step_radius` | every link, palm included, shares one thickness; steps it to 8, 10 or 12 mm | none |
| change one joint's rest bend | `step_bend_rpy` | turns the rest angle between two bones by 15 deg about one axis | none |
| shift one joint sideways | `step_bend_offset` | shifts one joint sideways by 5 mm (only historical `G_BEND` has rest offsets; never shown for the grammar) | none |
| lengthen/shorten one bone (5 mm) | `step_segment_length` | changes one bone's (or palm part's) length by exactly one 5 mm step within the grammar's range (15-80 mm, palm parts 20-80 mm); parts beyond it move with it; `lengthen_segment`/`shorten_segment` are its two directions as an exact inverse pair | none (no limit concerns lengths) |

The older `perturb_parameter` (also reachable as `step_length`, outside the pool) reflects off a range bound, so its step is not always exactly 5 mm; `step_segment_length` only offers moves that stay in range.

## Commercial hands in the grammar

`adapters/projection.py` expresses a real hand exactly: continuous lengths, free axes and mount poses, the URDF's limits, a lateral offset for every finger mount, a rest bend at every joint and a 10 mm radius. `derive` accepts that, but the grammar never generates such values, so the exact projection is not a member of the search space (the `coverage` audit puts it out of support) and some grid steps cannot act on it (`step_axis` acts on 4 of the 15 hands). `hand_sampler/grammar/adapters/conform.py` snaps it onto a grammar's grids (`conform_to_grammar(derivation, dist, limits)`, closed-loop, so errors do not add up along a finger) and reports what that costs. The viewer conforms every commercial hand to the one grammar. `grammar_bench/tests/test_conform.py` checks that every conformed hand derives, lies on the grammar's grids and is in support by `coverage`, that a hand the grammar sampled conforms to itself exactly, and that the operators act on conformed hands (on all 15 hands every grid step applies: tilt an axis, change a range, move a mount, palm length, thickness, rest bend, bone length).

Fidelity after snapping, E13's metric (zero plus 64 random configurations): max joint position error mm / max joint axis error deg / max fingertip error mm; the target is 5 mm / 10 deg. The exact projection is 0 / 0 / 0 for every hand. "The grammar" is `build_distribution()` (all rules on; the rules shape sampling only, so they do not change these numbers, which equal the historical `G_WIDE`'s); "Basic" is the historical `G_V1` (the grids before this change, same numbers as `G_FULL`), for comparison.

| Hand | The grammar | Basic (`G_V1`, historical) | Within Simulator limits |
|---|---|---|---|
| allegro_right | 9 / 11 / 17 | 59 / 9 / 82 | yes |
| leap_right | 15 / 19 / 29 | 70 / 16 / 108 | yes |
| barrett_bh | 17 / 3 / - | 52 / 3 / - | yes |
| ability_right | 6 / 10 / 13 | 39 / 8 / 41 | yes |
| inspire_right | 6 / 7 / 8 | 90 / 7 / 117 | yes |
| dclaw | 6 / 3 / 10 | 73 / 5 / 80 | yes |
| wuji_right | 20 / 14 / 28 | 72 / 13 / 85 | yes |
| xhand_right | 8 / 7 / 9 | 47 / 5 / 57 | yes |
| tesollo_dg5f_right | 11 / 7 / 16 | 94 / 3 / 132 | yes |
| orca_right | 18 / 8 / 17 | 97 / 10 / 121 | yes |
| sharpa_left_on_iiwa14 | 24 / 13 / 19 | 147 / 11 / 171 | yes |
| shadow_right_local | 19 / 6 / 21 | 83 / 6 / 102 | yes |
| svh_right | 32 / 5 / 26 | 87 / 7 / 109 | no: 2 fingers on one palm joint |
| arms_skel | 30 / 14 / - | 45 / 20 / - | yes |
| coupled_finger (analytic) | 0 / 0 / - | 0 / 0 / - | yes |

Features the real hands need and the rules that forbade them (counts out of 15, against the historical Basic grids), with the extension and its status:

| Conflict | Hands | Rule before | Extension | Status |
|---|---|---|---|---|
| fingers side by side across the palm (18-62 mm off the palm's axis) | 14 | a finger mounts on its host's axis, or one radius off it (surface rule) | a 5 mm lateral mount grid for fingers and palm parts | in the grammar (support only; `Distribution.mount_lateral_grid_m`, off by default elsewhere) |
| palm part beside its parent | 4 | a palm part mounts on its parent's axis | the same lateral grid | in the grammar (same field) |
| rest bend between bones (up to 99 deg) | 13 | no rest bend (the curl rule only draws 15-45 deg) | a rest bend at any joint on the 15 deg grid | in the grammar (support only) |
| joint ranges off the menu | 15 | 6 range options | any range in +/-180 deg | in the grammar (support only) |
| mounts between the 5 fixed positions along a host | 14 | 0, 25, 50, 75 or 100% of the host | 5% steps | in the grammar |
| thumb below the palm's origin, or a palm longer than 80 mm | 12 | mounts lie on the root segment | conform slides the root frame along its own axis (an exact re-expression) | done in `conform_to_grammar` |
| two joints at one point (a 0 mm bone, e.g. knuckle abduction + flexion) | 7: barrett, orca, sharpa, shadow, svh, arms, coupled_finger | bones are 15-80 mm | allow a 0 mm bone for a second joint at the same point | reported only: a 0 mm link makes its neighbours' capsules touch over 2 radii, which the simulator's overlap check (`rest_overlap_pairs`, in isaacsimenvs) flags; the envelope's adjacency rule would have to change first |
| bone lengths outside 15-80 mm | 3: wuji, sharpa, arms | 15-80 mm | a wider length range | not done (would change sampling) |
| a mount beyond a palm part's ends | 1: svh | mounts lie on the host segment | mount positions beyond [0, 1] on palm parts | reported only |
| the 15 deg grid for mounts, bends and axes | all (above 10 deg error on 2) | `ANGLE_STEP_DEG = 15` is a module constant | a finer angle step (5 deg) | reported only: it runs through `sample_axis`, the step operators and `coverage`; with exact angles and lengths the grammar reaches 2-5 mm / 0 deg / 1-5 mm on every hand |
| coupled (mimic) joints | 5: ability, inspire, svh, arms, coupled_finger | the grammar has a coupled module; the projection keeps every mimic as its own motor (decision I22); the Simulator limits forbid couplings | none for the simulator; elsewhere the projection could emit `Coupled` modules | reported |
| two fingers on one palm joint (SVH's j5) | 1 | the grammar allows it; the Simulator limits allow 1 | a carrier slot with two finger chains in the envelope (isaacsimenvs), or a rigid palm joint in the simulator view | reported |

On hands conformed to the grammar the operators that cannot act are the structural ones a drawn hand of the same shape also lacks: no branch finger to remove, no coupled joint, no rest offset menu, no palm part to make rigid or jointed (11 hands). Under the Simulator limits "add a branch finger" is never allowed, and "add a short finger" and "add a palm part" need a free finger slot (5 of 15 hands have one). SVH is outside the Simulator limits and can still be mutated as long as no limit gets worse.

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
