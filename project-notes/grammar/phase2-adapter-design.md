# Phase 2 design: grammar to simulator adapter

Status: design (read-only pass, CPU exploration only, 2026-09-26). Not yet implemented. Paths are relative to the repo root.

## 1. How hands reach Isaac today
- **Old sampler (heterogeneous population in one scene):** `hand_sampler/build.py` authors capsule bodies and revolute joints straight into the layer (`author_hand`, build.py:387-488).
  - Every design is padded to one 5x6 envelope.
  - Ghost links are 1e-4 m, with no collider and limits (0, 1e-8).
  - The palm is a box OVER on the arm link.
  - Tables live in `HandPopulation` (robot_spec.py:265-390) and are gathered per env through `design_index`.
  - The scene authors each env's robot directly, with `replicate_physics=False`.
- **Fixed URDF (`hand_only.py`):** one hand per run. It uses `_convert_fixed_robot`, a regex spawn and `clone_environments` with `replicate_physics=True`.
- **Conclusion:** only the padded path supports a population. `replicate_physics=True` copies env 0's physics to every env.

## 2. Grammar side
- **`derive()`** returns a `KinematicModel`:
  - root and palm{i} palm bodies, with revolute or fixed joints;
  - phalanges d{id}p{k};
  - free SE(3) origins, with segments along +z;
  - `<body>_tip` frames at (0,0,L);
  - one radius per hand.
- **`build_geometry`** gives local capsules, and palm cells as nearest-spine convex hulls. Mass properties are about the body origin, not the COM.
- **`to_urdf(geometry=)` is not importer-ready:**
  - capsules become a cylinder plus 2 spheres;
  - palm cells are `package://` meshes;
  - frames become extra links;
  - effort and velocity are placeholders;
  - no inertials are written.
- **Projected hands:** all 14 project as revolute-only and pass `fits_envelope(allow_palm_joints=True)`.
  - SHARPA and Shadow have 1 palm joint carrying 1 digit.
  - arms_skel has 2 palm joints, each carrying 1 digit.
  - SVH has 1 palm joint carrying 2 digits.
  - Projection hard-codes a 0.01 m capsule radius.

## 3. Recommendation: padded fixed-topology envelope (route B)
- **Why not per-design URDF (route A):** it breaks articulation homogeneity across envs, so each run can hold only one design.
- **Why B:**
  - It reuses the proven path, which runs 24k envs at about 8 ms per hand.
  - Every design gets the same 32 tokens, action dimension and SAPG batch shape.
- **Layout:**
  - 5 digits x 6 joints, plus 2 palm-carrier joints (pc0_j, pc1_j).
  - Digit slots 0-2 hang off root; slot 3 hangs off pc0; slot 4 off pc1.
  - A ghost carrier (identity pose, locked) makes slots 3 and 4 root-mounted.
  - Fixed palm bodies merge into their nearest jointed ancestor.
- **Admission:** `fits_envelope(m,5,6,allow_palm_joints=True,allow_branches=False)` AND revolute-only AND no couplings AND at most 2 jointed palm bodies, each carrying at most 1 digit and with no jointed palm descendant.
- **Measured admission rates (1500 seeds each):**

  | Variant | Admitted |
  |---|---|
  | G_SERIAL, R-only | 83% |
  | DEFAULT, R-only, no branches | 58% |

  DEFAULT rejections: multi-digit palm joint 21%, 6 digits 17%, palm chains 8%.
- **For E-R1:** constrain the distributions rather than rejection-sample: `digit_count_range=(1,5)`, R-only, `branch_probability=0`, `palm_body_count_range=(0,2)`. Report rejection rates.
- **Commercial hands:** SHARPA, Shadow and arms_skel fit; SVH does not.

## 4. Specification
- **`scene/grammar_envelope.py`** (numpy only, CPU-testable):
  - `admit`, `canonicalize -> EnvelopeDesign`
  - `joint_local_frames (32,2,4,4)`
  - `authored_fk` (frame0 * Rz(q) * frame1^-1)
  - `token_boxes` (via `design_space._ordered_box`)
  - `mass_props` (shifted to the COM)
  - `rest_overlap_pairs`
  - `palm_up`
  - `GrammarPopulation`
- **`scene/population_file.py`:** `write_population` and `load_population` with sha256 over canonical JSON (sort_keys, compact separators, allow_nan=False); `sampled_entries`; `projected_entry`.
- **`scene/author_grammar.py`** (lazy pxr imports): `author_design`, `author_population`, `setup_grammar_robot`, mirroring `assembly._author_robots_into_envs`.
- **`make_grammar_population.py`:** the CLI.
- **Env interface needed:**
  - `assets.hand_population`; when it is set, require `replicate_physics=False` and `clone_in_fabric=False`, and skip robot cloning;
  - expose `env.hand_spec`, `env.palm_body_idx`, `env.scene_record` and `env.hand_tables`;
  - per-env spawn offset;
  - mask ghosts in the action penalties.
- **Tables (n designs):**

  | Table | Shape |
  |---|---|
  | `joint_link_boxes` | (n,32,4,3) |
  | `joint_valid` | (n,32) |
  | `joint_limits` | (n,32,2) |
  | `default_joint_pos` | (n,32) |
  | `hand_scale` | (n,) |
  | `fingertip_valid` | (n,5) |
  | `fingertip_offsets` | (n,5,3) |
  | `palm_center` | (n,3) |
  | `palm_keypoints` | (n,4,3) |
  | `palm_frame` | (n,7) |
  | `base_rot` | (n,4) |
  | `spawn_offset` | (n,3) |
  | `drive` | (n,32,5) |

- **Population JSON:** schema, grammar_version, envelope, git_sha, generator, and `designs[]` entries with source, derivation and sha256. `population_sha256` is the hash of the per-design hashes. The loader re-derives every design and re-admits it, and any failure is fatal.
- **FK check:**
  - CPU: `authored_fk` vs `fk.forward_kinematics` at q=0 and 16 random q, to 1e-9.
  - Kit, 64 envs: body poses after `sim.forward()` at q0 and at random q, to 1e-4 m / 1e-3 rad. Assert `is_homogeneous` and that joint names match the template.
- **Projected commercial hands:**
  - Follow the E13 pipeline; the same authoring.
  - Use a per-hand radius from the mesh cross-section.
  - Zero-shot eval on the original URDFs needs a name_map obs/action adapter.
- **Analytic palm calibration:**
  - n = normalize(tip centroid at mid q minus mount centroid); x = finger direction orthogonalised against n.
  - `base_rot` maps the palm frame onto world axes.
  - Spawn height = hull extent along n + object half-size + 5 mm.

## 5. Tests
- **CPU:**
  1. FK on 200 designs per variant plus the projected hands.
  2. Admission cases (SVH rejected with a reason).
  3. Table invariants.
  4. Population round-trip and tamper detection.
  5. Mass positive definite.
  6. `palm_up` sanity: DClaw +z, Allegro about +x.
- **Kit smoke, 64 envs, 16 designs (8 G_SERIAL, 4 carrier, projected allegro/dclaw/sharpa/wuji):**
  - homogeneity;
  - FK;
  - the reset table-vs-PhysX check;
  - 200 zero-action steps: no NaNs, ghosts still, cube rests.

## 6. Risks
1. Topology drift: assert homogeneity.
2. `replicate_physics=True` silently homogenises the population: hard assert.
3. Isaac checks default joint positions on env 0 only: put a design whose limits contain 0 first, and write per-env defaults after init.
4. Rest self-penetration: numpy overlap filter plus the zero-action smoke.
5. Convex hull vertex limit of 64: pre-simplify.
6. Actuation constants differ between the grammar and `HandOnlySpec`: use `HandOnlySpec` values, with URDF effort for projected hands.
7. The joint-transformer spec lookup goes through the pose_reaching registry: add an inhand resolver.
8. Choose n_designs dividing `expl_coef_block_size`.
9. Coverage loss: 26% of DEFAULT samples, plus SVH. Report it.

## Aside, relevant to I24
- In every projected hand the fingers run along root +z.
- The drop-test calibration chose local -z up, which points the fingers down. So the cube likely rests on the wrist face, out of reach.
- The analytic palm normal is about +x for Allegro and SHARPA and +z for DClaw.
- This was forwarded to the Phase 1c worker to verify in Kit.
