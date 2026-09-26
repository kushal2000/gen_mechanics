# Plan: prove the hand representation covers real hands (kinematics), before any integration

Status: for approval. Branch `martin/hand-grammar` (clean). Supersedes the overnight-program plan, which is complete.

## Context

Martin's priority before wiring anything into the evolution pipeline: the **representation** (the literal structure a hand is) must represent real hands relatively accurately. The **grammar** (how it expands, its grids, ranges, priors and operators) is tuned later for explorability. Fitting real hands is only an expressivity check; evolution starts simple and explores.

Decisions from Martin:
- kinematics only;
- tolerance about 5 mm / 10°;
- hands = the current 11 plus articulated-palm hands (SHARPA's pinky CMC, Shadow LFJ5, SVH j5, ARMS CMC4/5), local use, licences not a concern;
- geometry stays capsules for now, with a side study of whether a single rounded-rectangle cross-section "fits most";
- actuation (motors, couplings beyond what exists) is future work.

What the checks so far show:
- **"8 of 11" is not a representation problem.** The flat model (`KinematicModel`) already holds all 11 hands exactly. The 3 failures were grammar checks:
  - Ability and Inspire declare coupled-joint limits outside what their driving joint can reach.
  - ORCA's wrist joint and fixed offset joints were misread by a heuristic parser.
- **The 1e-16 figures are a code-correctness test** (our FK equals Pinocchio on the same file). They stay as tests; they are not the fidelity criterion.
- **What evolution manipulates is the parametric structure `derive()` compiles**: palm bodies, digits, and phalanges, each with mount, bend, axis, limits and length. Nobody has checked that this structure can hold real hands.
- **`derive()` is already a range-free compiler.** `validate_derivation` checks structure only. Every joint origin is a general pose (phalanx 0: `(bend_x, bend_y, frac·L)` + rpy; continuations: `(bend_x, bend_y, L_prev)` + rpy). One exception: a palm body must sit on its parent's segment line (`xyz=(0,0,frac·L)`).
- **Real-hand survey** (107 linked joint pairs):
  - 79% of axes are within 20° of perpendicular to their link.
  - 10 knuckle pairs are under 5 mm apart (SHARPA 5, Wuji 4, ORCA 1).
  - Out-of-plane rest bends: median 0°, 90th percentile 36°, maximum 99°.
  - Median joint spacing is 30–70 mm.

Expected outcome: every hand projects into the parametric structure with near-zero kinematic error. That makes the representation claim demonstrated rather than assumed, and it yields a real-hand parameter atlas that will drive the later grammar tuning.

## Work items (sequential; one Sonnet worker each; I review, test and commit; benchmark commits before implementation commits)

### 1. Close the one representation gap in `derive()`
- Add `mount_offset=(0.0, 0.0)` to PalmBody params. A palm body's origin becomes `(ox, oy, frac·L)`, as phalanx 0 already allows.
- The default is zero, so `REPLAY_HASHES_SEED_0_99` must stay byte-identical.
- Confirm that `derive()` accepts representation-level values the grammar never samples: length 0 (co-located knuckles), `mount_frac` outside [0, 1], more palm bodies than the grammar's range, off-grid values. Add tests for each.
- Document in `rules.py` that `derive()` is the representation compiler, while ranges, grids and priors belong to the grammar (`distributions.py`, sampling, operators).

### 2. Structured projection: real hand → `Derivation` with continuous values
New `hand_sampler/grammar/adapters/projection.py`, `project_to_derivation(model, annotations) -> (Derivation, ProjectionReport)`:
1. **Merge fixed joints** into neighbouring origins. Fingertips come from the manifest's `tip_frames` if given, else the leaf body (via fixed joints). If a leaf coincides with its last joint, report `fingertip_undefined`.
2. **Palm set.** Root, plus every body on the path to at least 2 fingertips, plus the child of every joint listed in the manifest's per-hand `palm_joints`. Assumption to confirm: SHARPA `left_5_pinky_CMC`, Shadow `rh_LFJ5`, SVH `right_hand_j5`, ARMS `CMC4`/`CMC5` are palm joints; thumb CMC/THJ5 joints stay thumb-digit joints.
3. **Canonical frames** (an exact gauge change):
   - Each body's origin sits at its joint, with +z pointing to the next joint or the fingertip. If they coincide, use the next non-coincident point.
   - Roll is fixed so the joint axis lies in the x–z plane with x ≥ 0.
   - Palm bodies point +z at the centroid of their children's mount points, with length = the largest projection onto +z.
   - Record the root re-orientation so comparisons are made in the original frame.
4. **Emit steps.**
   - A PalmBody for each palm body: parent, `mount_frac`, `mount_offset`, `direction_rpy`, `has_joint`/axis/limits.
   - A Digit for each chain leaving a palm body, and a Phalanx for each movable joint (`length`, module R/P, axis, limits, bend = relative rotation, offset = 0).
   - Mimic couplings with an earlier revolute source in the same digit become `Coupled`. Others become independent joints, reported as `coupling_not_in_structure` (deferred, since actuation is future work).
5. **Report** everything merged, dropped or assumed.

### 3. E13 representation check at 5 mm / 10°, plus the real-hand atlas
New `experiments/e13_representation.py`, over 14 hands (the 11 plus SVH, Shadow, ARMS):
- **Check:** `derive(project(hand))` against the original. Sample zero plus 64 configurations, all original movable joints independent within declared limits, with full q (so coupling bookkeeping cannot hide errors). Compare every joint position, joint axis and fingertip in the original root frame. PASS at max ≤ 5 mm and ≤ 10°. Also assert the joint count is conserved.
- **Atlas:** per hand and pooled:
  - palm bodies and palm-joint axes;
  - digits and phalanges per digit;
  - link lengths, including zero-length knuckles;
  - rest bends and axis-to-link angles;
  - mount layout (lateral spread, stagger along the palm, mount rpy);
  - limit ranges and couplings kept or dropped;
  - for each parameter, whether `DEFAULT_DISTRIBUTION`'s range covers the real values. This is input for the later grammar tuning; no grammar changes now.
- **Manifest (benchmark commit):** add SVH, Shadow and ARMS as local entries by path and sha256, with `hand_root` (wrist excluded), `palm_joints` and, where needed, `tip_frames`. Add `palm_joints` to SHARPA. Tests skip with `local-only:` on machines without the files.
- **Report:** `pilot-report` gains a "representation" column (projection error, pass/fail, notes). The old heuristic `topology_expressible` column is kept only for the record.

### 4. Cross-section study (secondary; no change to `geometry.py` or the simulator)
New `experiments/e14_cross_section.py` and `project-notes/grammar/cross-section-study.md`:
- **Data:**
  - Collision meshes on disk: SHARPA STL in the repo; the others under `~/karma/...`, STL/OBJ only, skipping unloadable formats.
  - Use trimesh from the `piper` conda env.
  - Slice each finger link at 25/50/75% of its length in the canonical frame from item 2.
- **Fit:**
  - Per section: a circle (capsule) and a rounded rectangle (w, h, corner r); record width, thickness, and boundary error in mm.
  - Then search for one normalised template (fixed h/w and r/w, one size per link) that fits most sections within about 2–3 mm, and compare it with the single-radius capsule.
- **Contact cost (desk study; GPU runs need your authorisation):**
  - The capsule is a native PhysX primitive.
  - A rounded rectangle is not native. Options: a box primitive with per-shape rest offset r (behaves like a Minkowski-rounded box, still a native box), a convex hull mesh (costlier contact generation on GPU), or two parallel capsules.
  - Note that the policy's joint tokens already use boxes.
  - Deliver a recommendation, plus a proposed small GPU benchmark for the RL owners.

## Reuse
- `adapters/urdf.load_urdf` (with `hand_root`), `fk.forward_kinematics`, `fk.rotation_error`.
- `derive.Derivation`, `DerivationStep`, `derive`, `validate_derivation`.
- `canonical.normalize_axis_sign`.
- Hand resolution as in `grammar_bench/evaluate.py` (`_resolve_hand`) and `experiments/e11_support_widening.py`.
- `experiments/runner.run_experiment` (provenance, `allow_dirty`).
- The mesh box extraction pattern in `hand_sampler/design_space.py` (`joint_link_boxes`) for the study.
- The fixed-joint and hand_root handling already in `load_urdf`.

## Out of scope now
Grammar tuning (grids, ranges, priors, operators, drift); cross-digit couplings and actuation; geometry changes in `geometry.py` or USD; pipeline integration (slot adapter, `build.py`, tokens, `evolve.py`).

## Guardrails
- At most 4 Sonnet runs and 1 Opus review (after item 3).
- No pushes, installs, GPU jobs or downloads.
- Suite green and replay hashes identical after each commit.

## Verification
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider hand_sampler/grammar_bench/tests -q
python3 -m hand_sampler.grammar.experiments.runner --list          # e13_representation, e14_cross_section registered
cat project-notes/grammar/experiments/E13_representation/summary.md # 14 hands, max mm / deg, PASS lines, atlas
python3 -m hand_sampler.grammar_bench.evaluate                      # pilot-report with the representation column
```
Expected results:
- All 14 hands PASS at 5 mm / 10°, most at numerical precision.
- Every dropped or assumed item is named in the report: cross-digit couplings, undefined fingertips, palm-joint annotations.
- The atlas lists each real-hand parameter the current grammar ranges exclude.
- The cross-section note gives fit errors for the capsule and the rounded rectangle, plus a cost-aware recommendation.
