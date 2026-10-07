# Grammar viewer

Interactive viser viewers for the hand-kinematics grammar (`hand_sampler/grammar/`). Everything runs on the CPU. Isaac, Kit and the GPU are never touched: the simulator's numpy-only envelope modules are loaded by file path (`gviewer/envload.py`), so `isaacsimenvs/__init__.py` and Isaac Lab are never imported.

- `viewer.py` is the essential viewer: one compact panel, in plain words, to draw a hand from the grammar (with its rules) under the simulator's limits, switch each viability check on or off, look at a commercial hand snapped onto the grammar, and apply the coarse or the fine mutation steps.
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
| steps: Coarse / Fine | `derive.EVOLUTION_OPERATORS_COARSE` / `EVOLUTION_OPERATORS_FINE` |

## Three layers: grammar, simulator limits, viability

The hand grammar separates what it can express from what the simulator can build and from what we can only measure afterwards.

1. **The grammar** (`hand_sampler/grammar/rules.py`, `derive.py`, `distributions.py`, with its rules in `variants.build_distribution`): every hand the productions can express, including several jointed palm bodies, stacked palm joints, coupled joints, branching digits and several digits on one palm joint. Its rules shape what it draws (below).
2. **The simulator limits** (`hand_sampler/grammar/limits.py`, `SIMULATOR`): hard rules that sampling (`sample_derivation(..., limits=)`) and every mutation operator (`vary`, `apply_operator`, `vary_tracked`, all with `limits=`) obey by construction. They only choose among options that keep the hand within the limits (an allowed module kind, a digit count below the cap, a host with room, a palm joint only where one is allowed, a bone short enough for the finger-length cap), so nothing is generated and then rejected. An operator with no admissible application is inapplicable, exactly like one with nothing to act on. A limit that does not bind leaves the random draws unchanged; where one binds, the sampler draws from the restricted options, so the distribution differs from rejection sampling, which is intended. `limits=None` means no limits and reproduces past runs byte for byte.
3. **Viability checks** (`gviewer/checks.py`): physical properties no generation rule can guarantee, measured with the simulator's own functions.

`SIMULATOR` is the simulator's 32-slot articulation (5 finger chains of 6 revolute slots plus 2 palm-joint slots) plus the finger-length cap. Without the cap, a design is within it if and only if `grammar_envelope._admit_structural` admits it, and every design sampled or mutated under it is admitted (`hand_sampler/grammar_bench/tests/test_generation_limits.py`: 2000 samples per variant and 50 x 50 mutation chains, all admitted). In the viewer the limits start at these values, can be edited, and **reset** puts them back.

| Field | Panel | Simulator | Meaning |
|---|---|---|---|
| `allowed_modules` | joint types, allow coupled joints | hinge only | joint module kinds that may be generated |
| `allow_branches` | allow branching fingers | no | a finger may grow a branch finger off a bone |
| `max_digits` | max fingers | 5 | top-level digits |
| `max_joints_per_digit` | max joints per finger | 6 | joints in one finger, its branches included |
| `max_palm_bodies` | max palm parts | any | palm parts besides the palm (rigid ones fold into the palm in the simulator) |
| `max_jointed_palm_bodies` | max palm joints | 2 | palm parts with their own joint |
| `allow_stacked_palm_joints` | allow stacked palm joints | no | a jointed palm part below another jointed one |
| `max_digits_per_jointed_palm_body` | fingers per palm joint | 1 | fingers carried by one palm joint (two: pending, see "Commercial hands") |
| `max_finger_chains` | max finger slots | 5 | fingers on the rigid palm plus palm joints; each palm joint takes a slot |
| `max_finger_length_mm` | max finger length (mm) | 250 | longest sum of bone lengths from a finger's base to one of its fingertips, branches included (a branch counts its host's bones up to the one it grows from, so moving a mount never changes it) |

How they are enforced: the digit and palm-body counts are drawn from ranges capped by the limits; a palm body gets a joint only where it is allowed; digits go only to hosts with room; phalanx counts respect a per-digit joint budget that also covers branches; module kinds come from the allowed set; branches only when allowed; each bone is drawn among the lengths that leave the rest of its finger room under the length cap. Each operator applies the same rules to what it adds or changes: `add_minimal_digit` needs a host with room, `toggle_palm_joint` only toggles a body whose new state is allowed, `insert_phalanx` only grows a digit with a joint and the length to spare, lengthening steps stop at the cap. A hand already outside the limits (SVH) can still be mutated, as long as no limit gets worse.

The evolution driver (`isaacsimenvs/inhand_reorient/evolution/driver.py`) is unchanged and still calls `sample_derivation(seed, dist)` and `vary(..., operators=EVOLUTION_OPERATORS)` without limits, then filters with `admit`. To generate within the simulator directly it would pass `limits=SIMULATOR` to both calls; its `admit` call would then only reject on the physical checks. For the fine stage it would pass `operators=EVOLUTION_OPERATORS_FINE`.

## One grammar, its rules

The viewer has one grammar, `variants.build_distribution(surface=True, spacing=True, curl_opposition=True)`: one base distribution, `GRAMMAR_BASE`, plus three generation rules that shape where random hands put their fingers, and one rule that always holds.

- `GRAMMAR_BASE` is the default grammar (G_FULL's productions and sampling priors) with capability ranges wide enough to contain the commercial hands: mount positions in 5% steps along a host, a 5 mm lateral mount grid up to 65 mm, a rest bend at any joint on the 15 deg rotation grid, joint ranges anywhere in +/-180 deg, and the length ranges below. All but the mount positions are support only: mutation reaches them and a conformed real hand may use them, but random sampling never draws them, so a random hand looks exactly as the rules say (`test_one_grammar.py` and `test_coarse_fine.py` check the draws are unchanged).
- The three rules, each one checkbox in the panel: "fingers sit on the palm surface" (`surface`, was V1s), "fingers spaced apart" (`spacing`, was V2/V2s: finger bases planned at least 29 mm apart) and "fingers curl and oppose" (`curl_opposition`, was V3s). All three are on by default.
- **Every palm part carries a finger** (`Distribution.palm_body_needs_digit`, on in `GRAMMAR_BASE`): a palm part without a finger is not a palm. Sampling and every operator keep it, with or without limits: the first fingers go one to each palm leaf, `add_palm_body` adds the part with a one-joint finger, and removing a palm part's last finger removes the part.

Length ranges (support is what mutation and conform reach; sampling keeps its priors). The numbers come from the 14 commercial projections of `grammar_bench/manifest.json`, measured before snapping:

| Length | Sampled | Support | Measured on the commercial hands |
|---|---|---|---|
| bone | 15-80 mm | 0-90 mm | 0-84 mm; 27 of 220 bones are 0 mm (two joints at one point, e.g. a knuckle's abduction and flexion joints), 37 are under 15 mm, 3 over 80 mm (DClaw, up to 84) |
| palm | 20-80 mm | 15-160 mm | 18-144 mm (Tesollo 144, Inspire 138, Orca 28, DClaw 18) |
| palm part | 20-80 mm | 10-80 mm | 10-66 mm; the top stays at 80 mm so the support contains what sampling draws |
| finger (sum of its bones) | - | at most 250 mm (`SIMULATOR`) | 57-221 mm: DClaw 221, then Allegro's thumb 160, 95th percentile 159; 250 = 221 x 1.1, rounded |

The named variants of `variants.NAMED_DISTRIBUTIONS` are historical, kept unchanged (byte for byte) for reproducing past experiments, and are not shown in the viewer:

| Variant | What it was | Now |
|---|---|---|
| `G_FULL` (= `DEFAULT_DISTRIBUTION`) | the default grammar | the base of `GRAMMAR_BASE` |
| `G_SERIAL`, `G_NOPALMJOINT`, `G_NOBRANCH`, `G_NOCOUPLE` | G_FULL without palm parts / palm joints / branches / coupled joints | limit fields: max palm parts 0, max palm joints 0, allow branching off, joint types without coupled |
| `G_V1` | G_FULL restricted to what the simulator builds | the grammar under the simulator limits |
| `G_V1S`, `G_V2S`, `G_V3S` | V1 + surface mounts; + spacing; + curl and opposition | the three rules |
| `G_V2`, `G_V3` | first versions of spacing and curl | superseded by V2s/V3s |
| `G_FULL_INS`, `G_NOBRANCH_INS` | growth operators insert small pieces | experiment settings (E2/E3) |
| `G_BEND`, `G_CONT` | random rest bends; continuous joint ranges | support grids of `GRAMMAR_BASE` (not sampled) |
| `G_WIDE` | G_FULL with the grids widened and sampled | `GRAMMAR_BASE` has the same grids as support only |

## Coarse and fine

Two stages of mutation, two operator pools, one grammar.

- **Coarse** (`EVOLUTION_OPERATORS_COARSE`, the same tuple as `EVOLUTION_OPERATORS`): structural operators plus the coarse steps (5 mm lengths, 15 deg axes, bends and mount turns, 5% mount positions, 5 mm lateral offsets, the radius menu, the joint-range menu or 15 deg range ends). For generating and evolving hands that work.
- **Fine** (`EVOLUTION_OPERATORS_FINE`): refinement without structural change. Every fine operator moves one parameter of one part by exactly one fine unit (`distributions.FINE_*`): 1 mm for bone, palm and palm-part lengths, lateral mount offsets and the capsule radius (within the radius menu's 8-12 mm span); 1% of the host's length for a mount's position along it; 5 deg for joint axes, mount and palm-part orientations, rest bends and joint-range ends. Mount positions are stored as fractions of the host, so the along-the-host step is 1% of the host (0.15-1.6 mm on 15-160 mm hosts): a fixed 1 mm step would stop being a grid as soon as the host's length changed, and the 5% coarse positions are not whole millimetres on most hosts. Fine steps stay in the support and keep the limits, finger-length cap included. `fine_lengthen_segment` / `fine_shorten_segment` are an exact inverse pair (`FINE_LENGTH_STEP_PAIR`).
- The fine grid contains the coarse one (5 mm lengths are whole millimetres, 15 deg angles are multiples of 5 deg, 5% positions are whole percents), so a coarse hand is a valid fine hand, and `coverage(model, dist, resolution="fine")` accepts every hand the coarse support accepts. Coarse steps also act on a refined hand: they move a fine-grid value by one coarse unit (15 deg, 5 mm, 5%, the next radius on the menu) and keep it on the fine grid.

`test_coarse_fine.py` checks the steps (one unit, one parameter, never the structure), the bounds, the limits, the inverse pair and that coarse is inside fine.

## The essential viewer

One panel, one line per item; longer explanations are hover text.

- **Grammar**: **Random**, which draws hands from the grammar under the current limits until every enabled viability check passes, the three rule checkboxes (switching one draws a new hand), and "found after N tries", which counts viability rejections only: every draw is within the limits by construction.
- **Limits**: one dropdown or checkbox per limit, starting at the simulator's values (max finger length included), and **reset**. Sampling and mutation use these limits. The last line says whether the hand on screen is within them.
- **Viability**: the four physical checks, one line each ("fingers don't overlap (open): PASS 0.4 mm"), each with its own toggle. Unticking a check stops Random requiring it. "show object and reach" (off by default) shows the object's start sphere, the 5 cm reach sphere and the fingertip dots (green reaches, orange does not).
- **Commercial hand**: a dropdown of the manifest hands, drawn over their real URDF meshes (a checkbox hides them). "shown as" picks the version: **fine grid** (default, the hand snapped onto the grammar's fine grid, the closest the grammar gets), **coarse grid**, or **exact (off-grid)**, the exact projection. One line gives the error against the URDF, whether the shown version is within the grammar's rules, and whether it is within the current limits (SVH carries two fingers on one palm joint).
- **Mutation**: **steps: Coarse / Fine**, **Random mutation** (draws operators from the chosen stage until one applies), **Back** (undo), and one button per operator of that stage, shown only when it can act on this hand under these limits. After each mutation a line says what changed; the parent stays on screen as a faint grey ghost.
- **Pose**: one curl slider (0 lower limits, 1 upper limits; the default 0.35 is the pose every episode starts from) and **re-centre view**.

What is on screen: one colour per finger (a branch finger a lighter shade of its finger's colour; colours follow the finger, so removing one finger never recolours the others), neutral grey palm parts, small dark joint markers, the grey ghost of the parent after a mutation, and red only on the links of a failing overlap check that is switched on. The palm stays fixed in the world, and the camera only moves for a new draw, a new commercial hand or **re-centre view**.

## Viability checks

Only properties no generation rule can guarantee. They need the simulator's 32-slot model of the hand, which exists only for hands within the simulator's structural limits; on any other hand they read n/a and do not block Random. A design is viable when it passes all four; under the simulator limits that is exactly `viability_report`'s admitted flag plus "at least 2 fingertips reach" (`tests/test_checks.py`).

| Panel | Key | What it requires | Why |
|---|---|---|---|
| fingers don't overlap (open) | `overlap_zero` | no two capsules interpenetrate by more than 3 mm with every joint at 0 | deeper starting overlaps made the physics engine push links apart at over 100 rad in one or two steps |
| fingers don't overlap (start pose) | `overlap_reset` | the same with every joint 35% of the way through its range | every episode starts from this pose |
| object starts above palm* | `spawn_height` | with the hand turned palm-up, the object's start point is 20 mm or more above the palm | the hand must hold the object up against gravity |
| ≥2 fingertips reach object* | `reach` | in a 4000-sample random sweep of the joints, 2 or more fingertips come within 5 cm of the start point | a hand that cannot touch the object with two fingers cannot turn it |

\* Provisional, to be reworked: the spawn and reach checks (a sphere around a cube-sized start point) are placeholders for a better measure of whether a hand can hold and turn the object.

**Joints at one point.** A 0 mm bone puts two joints at one point, and the capsules on either side of it meet there like a parent and its child. The overlap check (`grammar_envelope.rest_overlap_pairs`, which the viewer calls) treats two bodies joined through a chain of bones shorter than one capsule radius as adjacent (`adjacent_pairs`), and `canonicalize` records those pairs in `EnvelopeDesign.filtered_pairs` (`short_bone_pairs`), which `author_grammar.author_design` collision-filters (PhysX never collides a joint's two bodies, but it would collide these). One radius, not two: a longer middle bone keeps its neighbours apart at the rest and start poses, and where it does not (a finger folded back onto itself) the collision is real. No sampled bone is that short (sampled bones are at least 15 mm, radii at most 12 mm), so sampled designs keep exactly their parent and child adjacency. `author_grammar._link_mass_props` floors a link's mass and inertia at a solid sphere of its radius, so a 0 mm link is not near-massless between two real links (it was 1e-6 kg).

## Mutation operators

**Coarse**: `EVOLUTION_OPERATORS` (= `EVOLUTION_OPERATORS_COARSE`), the evolution driver's pool, 17 operators. The pool before 2026-10-06 is frozen as `EVOLUTION_OPERATORS_V1` (pass `operators=EVOLUTION_OPERATORS_V1` to reproduce runs made with it).

| Button | Operator | What it does | Limited by (simulator) |
|---|---|---|---|
| add a short finger (1 joint) | `add_minimal_digit` | new finger with one hinge joint and one bone, on the palm or a palm part | fingers, finger slots, fingers per palm joint; needs hinges |
| remove a finger | `remove_digit` | removes one finger of any length, with its branch fingers; a palm part left without a finger goes too | at least one finger stays; no limit may get worse |
| add a joint to a finger | `insert_phalanx` | inserts a joint and bone at a random place in one finger; the bones beyond it move out | joints per finger, finger length |
| remove a joint from a finger | `delete_phalanx` | removes one joint and its bone from a finger with at least 2; branch fingers on it re-attach to the neighbouring bone | none |
| add a palm part with a short finger | `add_palm_body` | adds a palm part on the palm or another palm part, jointed or rigid at random where a joint is allowed, with a one-joint finger (the grammar's palm rule) | palm parts; palm joints, stacking, finger slots; room for its finger |
| make a palm part rigid/jointed | `toggle_palm_joint` | gives a rigid palm part a joint with a new axis and range, or makes a jointed one rigid | palm joints, stacking, fingers per palm joint, finger slots |
| add a branch finger (1 joint) | `add_branch_digit` | adds a one-joint finger growing off a bone of a finger | branching (never under the simulator limits), joints per finger, finger length |
| remove a short branch finger | `remove_branch_digit` | removes a branch finger with one joint | none |
| tilt one joint axis | `step_axis` | tilts one joint's axis by one 15 deg step | none |
| change one joint's range | `step_limits` | moves one joint's range to the neighbouring menu option, or one end by 15 deg within +/-180 deg | none |
| move a mount (finger or palm part) | `step_mount` | slides one finger or palm part along what it is attached to (5%), shifts it sideways (5 mm), or turns it 15 deg at its base | none |
| change one coupled joint | `step_coupling` | steps a coupled joint's ratio or offset | none (no coupled joints under the simulator limits) |
| lengthen/shorten the palm | `step_root_length` | changes the palm's length by 5 mm, within 15-160 mm; the fingers keep their relative place along it | none |
| thicker/thinner (all links) | `step_radius` | every link, palm included, shares one thickness; steps it to 8, 10 or 12 mm | none |
| change one joint's rest bend | `step_bend_rpy` | turns the rest angle between two bones by 15 deg about one axis | none |
| shift one joint sideways | `step_bend_offset` | shifts one joint sideways by 5 mm (only historical `G_BEND` has rest offsets; never shown for the grammar) | none |
| lengthen/shorten one bone (5 mm) | `step_segment_length` | changes one bone's (or palm part's) length by 5 mm within 0-90 mm (palm parts 10-80 mm); parts beyond it move with it; `lengthen_segment`/`shorten_segment` are its exact inverse pair | finger length |

**Fine**: `EVOLUTION_OPERATORS_FINE`, 9 operators, each one parameter by one fine unit.

| Button | Operator | What it does |
|---|---|---|
| tilt a joint axis 5° | `fine_step_axis` | one joint's axis by 5 deg in elevation or azimuth (on the 5 deg spherical grid) |
| move one end of a joint's range 5° | `fine_step_limits` | the lower or upper end of one hinge's or palm joint's range by 5 deg, within the limit range |
| slide a mount 1% along its part | `fine_slide_mount` | one finger's or palm part's position along its host by 1% of the host |
| shift a mount 1 mm sideways | `fine_shift_mount` | one finger's or palm part's lateral offset by 1 mm in x or y, within 65 mm |
| turn a mount 5° | `fine_turn_mount` | one roll, pitch or yaw of a finger's mount or a palm part's direction by 5 deg |
| change a rest bend 5° | `fine_step_bend_rpy` | one component of one joint's rest bend by 5 deg, within the bend support's span |
| lengthen/shorten the palm 1 mm | `fine_step_root_length` | the palm's length by 1 mm, within 15-160 mm |
| thicker/thinner 1 mm (all links) | `fine_step_radius` | the shared thickness by 1 mm, within 8-12 mm |
| lengthen/shorten a bone 1 mm | `fine_step_segment_length` | one bone or palm part by 1 mm (a bone down to 0 mm); `fine_lengthen_segment`/`fine_shorten_segment` are its exact inverse pair; stops at the finger-length cap |

## Commercial hands in the grammar

`adapters/projection.py` expresses a real hand exactly: continuous lengths, free axes and mount poses, the URDF's limits, a lateral offset for every finger mount, a rest bend at every joint and a 10 mm radius. `adapters/conform.py` snaps it onto the grammar (`conform_to_grammar(derivation, dist, limits, resolution=)`, closed-loop, so errors do not add up along a finger), at the coarse or the fine resolution, and reports what that costs. At the fine resolution, with the one grammar's full bend support, each frame's turn about its own link and its joint axis are chosen together over the whole 5 deg rotation lattice (turning a frame about its link moves no joint, so many frames aim the link right and one usually holds a grid axis close to the hand's), and a beam search over each finger lets a link lean a little so that the next bend lands on the grid; the cost is the sum of the joints' and fingertip's distances from the hand's plus each axis error times its distance to the fingertip. The viewer shows the fine conform by default.

Fidelity, E13's metric (zero plus 64 random configurations): max joint position error mm / max joint axis error deg / max fingertip error mm; the target is 5 mm / 10 deg. The exact projection is 0 / 0 / 0 for every hand. "Coarse, before" is the coarse conform before the wider ranges (bones 15-80 mm, palm 20-80 mm); "+1 mm joint offsets" is an experiment, not in the grammar (below).

| Hand | Coarse, before | Coarse | Fine | Fine within target | Fine + 1 mm joint offsets | Within the simulator limits |
|---|---|---|---|---|---|---|
| allegro_right | 10 / 11 / 17 | 9 / 11 / 16 | 2.3 / 0.4 / 2.3 | yes | 1.3 / 0.6 / 1.6 | yes |
| leap_right | 15 / 20 / 29 | 16 / 20 / 31 | 2.5 / 2.6 / 4.5 | yes | 2.2 / 2.9 / 4.2 | yes |
| barrett_bh | 18 / 2 / - | 18 / 2 / - | 3.7 / 1.4 / - | yes | 4.1 / 1.6 / - | yes |
| ability_right | 6 / 10 / 13 | 6 / 10 / 13 | 1.6 / 3.4 / 2.5 | yes | 1.6 / 4.1 / 2.1 | yes |
| inspire_right | 6 / 7 / 8 | 6 / 7 / 8 | 1.6 / 1.4 / 2.0 | yes | 1.1 / 1.4 / 2.2 | yes |
| dclaw | 6 / 3 / 10 | 6 / 3 / 9 | 5.7 / 0.8 / 7.0 | no | 1.7 / 0.9 / 3.5 | yes |
| wuji_right | 20 / 14 / 28 | 17 / 14 / 28 | 1.9 / 0.8 / 2.9 | yes | 2.1 / 2.3 / 2.3 | yes |
| xhand_right | 8 / 7 / 9 | 6 / 7 / 9 | 1.9 / 1.5 / 4.8 | yes | 1.1 / 0.7 / 1.3 | yes |
| tesollo_dg5f_right | 11 / 7 / 16 | 11 / 7 / 15 | 1.6 / 0.5 / 1.9 | yes | 1.4 / 0.9 / 1.8 | yes |
| orca_right | 18 / 8 / 17 | 9 / 8 / 13 | 2.9 / 1.3 / 2.4 | yes | 1.0 / 1.5 / 1.4 | yes |
| sharpa_left_on_iiwa14 | 24 / 13 / 19 | 15 / 13 / 17 | 2.4 / 0.5 / 1.7 | yes | 1.3 / 1.6 / 1.9 | yes |
| shadow_right_local | 20 / 6 / 21 | 10 / 6 / 12 | 1.6 / 0.9 / 1.9 | yes | 1.6 / 0.9 / 1.9 | yes |
| svh_right | 32 / 5 / 26 | 9 / 5 / 13 | 1.7 / 1.4 / 2.5 | yes | 1.5 / 1.3 / 1.9 | no: 2 fingers on one palm joint |
| arms_skel | 30 / 14 / - | 16 / 14 / - | 2.7 / 6.0 / - | yes | 3.9 / 3.4 / - | yes |
| coupled_finger (analytic) | 0 / 0 / - | 0 / 0 / - | 0 / 0 / - | yes | 0 / 0 / - | yes |

At the fine resolution 14 of 15 hands are within 5 mm / 10 deg (most at 1.6-3 mm). DClaw misses (5.7 mm joint, 7.0 mm fingertip) because a small bend between two bones is not on the 5 deg rest-bend grid, whose angles from straight are 0, 5, 7.1, 10, ... deg whichever way the frames are turned, and near straight the only other freedom, a turn about the link, comes in 5 deg steps too, so the joint axis cannot be matched at the same time. DClaw's 2.8 deg bend between its middle bones becomes 0 deg with an accurate axis (the search prefers it: the axis has a 152 mm lever to the fingertip), which on a 68 mm bone puts the next joint 5 mm off at q = 0. A 1 mm lateral joint offset in the fine support (the `bend_offset` field `G_BEND` already has, within +/-5 mm) moves each joint onto its hand position and brings every hand inside the target (last column, measured without the beam search); a 2.5 deg bend grid would also do. Neither is in the grammar yet.

Every conformed hand (fine) derives, lies on the fine grid, is in `coverage(..., resolution="fine")` support, keeps the palm rule, is within `SIMULATOR` except SVH, and every fine operator acts on it (`grammar_bench/tests/test_conform.py`). On hands conformed to the grammar the coarse operators that cannot act are the structural ones a drawn hand of the same shape also lacks (no branch finger, no coupled joint, no palm part to make rigid or jointed); "add a short finger" and "add a palm part" need a free finger slot.

Features the real hands need, against the grammar before 2026-10-06, and their status:

| Conflict | Hands | Extension | Status |
|---|---|---|---|
| fingers side by side across the palm (18-62 mm off the palm's axis), palm parts beside their parent | 14, 4 | a lateral mount grid (5 mm coarse, 1 mm fine) | in the grammar (support only) |
| rest bends between bones (up to 99 deg); joint ranges off the menu | 13, 15 | a rest bend at any joint (15 deg coarse, 5 deg fine); any range in +/-180 deg | in the grammar (support only) |
| mounts between the 5 fixed positions along a host | 14 | 5% steps (1% fine) | in the grammar |
| two joints at one point (a 0 mm bone) | 7: barrett, orca, sharpa, shadow, svh, arms, coupled_finger | bones down to 0 mm; the overlap check and authoring treat the bodies on either side as adjacent | in the grammar and the envelope |
| bone lengths outside 15-80 mm; a palm longer than 80 mm | 3; 2 | bones 0-90 mm, palm 15-160 mm (support), finger length capped at 250 mm | in the grammar |
| thumb below the palm's origin | most | conform slides the root frame along its own axis (an exact re-expression) | done in `conform_to_grammar` |
| the 15 deg grid for mounts, bends and axes | all | the fine stage (5 deg) | done; DClaw still needs a lateral joint offset or a finer bend grid (above) |
| a mount beyond a palm part's ends | 1: svh | mount positions beyond [0, 1] on palm parts | reported only |
| coupled (mimic) joints | 5: ability, inspire, svh, arms, coupled_finger | none for the simulator; elsewhere the projection could emit `Coupled` modules | reported |
| two fingers on one palm joint (SVH's j5) | 1 | a carrier slot with two finger chains in the envelope | pending: no fixed 32-slot layout fits both SVH (2 fingers on one palm joint + 3 rigid) and arms_skel (2 palm joints with 1 finger each + 3 rigid); the options (keep 32 slots and move one chain, dropping "2 palm joints + 3 rigid fingers"; 6 chains x 5 joints + 2; 6 x 6 + 2 = 38) are Martin's call. `test_generation_limits.py` pins today's slot layout of every admitted design and makes SIMULATOR follow the envelope's `MAX_DIGITS_PER_CARRIER`, ready to switch on |

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

`pytest.ini` switches off the ROS pytest plugins that leak in through `PYTHONPATH`. The tests cover the render primitives (against `forward_kinematics` and `build_geometry`), the envelope loader (against `viability_report` and the archive bins), the viability checks (against `viability_report`, and their early-stop path), the commercial fidelity (equal to E13's `check_hand` without its Pinocchio part) for allegro, sharpa and svh, the mesh-overlay poses, the history round trip and diff, file parsing, and the callbacks of both viewers against a local server: the limit fields and reset, Random under the limits, each viability toggle changing Random's tries, operator buttons greyed out under the limits, the Coarse/Fine switch, and commercial hands on the fine and coarse grids. The limits, the length ranges and the coarse and fine operators are tested in `hand_sampler/grammar_bench/tests/` (`test_generation_limits.py`, `test_coarse_fine.py`, `test_conform.py`); the envelope's handling of 0 mm bones in `isaacsimenvs/inhand_reorient/tests/test_grammar_envelope.py` (run with `.venv_isaacsim/bin/python3`, no Kit needed).

## Known limitations

- Not yet checked in Isaac: that the authored joints match the grammar's forward kinematics for hands with 0 mm bones, and that such a hand (and, once the layout is decided, one with two fingers on a palm joint) loads without self-collision blow-ups. The CPU side (adjacency, filtered pairs, mass floor) is tested.
- The Pinocchio export cross-check of E13 is not run (it needs a separate interpreter); the readout says so.
- `arms_skel` has no meshes at all. The SVH and Shadow URDFs in `karma-hand-metric` point at mesh files that do not exist; the viewer finds same-named `.dae` files in the downloads (`SVH/`, `Shadow/`) and marks them "alignment unverified". About 10 SVH meshes have no same-named file.
- Mesh overlays are reduced by vertex clustering to at most 6000 faces per piece; they are a visual reference.
- Projected hands use one capsule radius (10 mm, the projection's constant), so their rest overlaps are artefacts; the simulator exempts them, the overlap checks and the full viewer's Analysis tab still count them.
- The viability checks need the simulator's 32-slot model, so a hand outside the simulator's structural limits reads n/a on all four.
- Nearest commercial: `phenotype_distance` aligns joints by name. Grammar and projection both name digit joints `d{k}p{i}_j`, but digit numbering is arbitrary and the two root frames follow different gauges, so treat it as a coarse similarity.
- Joint classes are computed at the rest pose. Allegro's first finger joints read as twist because at rest they rotate the straight finger about its own axis.
- The envelope's ghost and padding slots are never drawn: the scene is the grammar model, not the authored articulation.
