# Cross-section study: capsule vs. rounded rectangle (representation-check plan, item 4)

Secondary to E13 (item 3). No change to `geometry.py` or simulator code: geometry
stays capsules. This answers Martin's question: would a single rounded-rectangle
cross-section ("one cross-section fits most") represent real finger links better
than a capsule, and would it complicate RL or contact math? If there is no real
fit/cost difference, find the template that fits most.

**Bottom line: no.** A single fixed rounded-rectangle template, given one free
size per link (the fairest match to how a capsule already works -- one radius
per link), fits real finger links about as well as a per-link capsule radius
(15% of sections within 2 mm either way; 41% vs 39% within 3 mm) -- not better.
Individual cross-sections, fit with their own free aspect ratio, are genuinely
closer to a rounded rectangle than a circle (roughly half the boundary error),
but that advantage does not survive being forced into one shared shape, because
real links disagree on what that shape should be (see "Why the template search
converges near a circle" below). Capsules also stay cheaper and simpler in the
contact pipeline (established from documentation, not benchmarked -- see
"Contact cost"). Recommendation: keep capsules.

## Data

Script: `hand_sampler/grammar/experiments/e14_cross_section.py`
(+ `hand_sampler/grammar_bench/refgen/mesh_sections.py` for mesh I/O and the
shape-fitting math). Run via the `piper` conda env
(`/home/singularity/anaconda3/envs/piper/bin/python`, which has `trimesh`
4.7.3 and `yourdfpy` 0.0.60); the script imports fine on the system Python
used for the test suite, it just reports every hand `mesh_unavailable` there
(no `trimesh`).

Pipeline, per hand: `load_urdf` + `project_to_derivation` (exactly as E13,
reusing the same manifest `hand_root`/`palm_joints`/`tip_frames` fields) give
the digit/phalanx structure; this script reads `project_to_derivation`'s
public `name_map` to recover, per digit, the original body chain and tip
body (it does not re-derive the palm/digit-chain logic itself, since
`project_to_derivation` does not expose its internal per-body canonical
frames), then computes each phalanx body's own joint-to-joint (or
joint-to-fingertip) length and canonical +z direction directly from the
original model's zero-config forward kinematics -- the same arithmetic
`projection.py`'s internal `frame_digit` uses. Only links with length
&ge; 10 mm are kept (per the plan). The link's collision mesh (or, if it has
none loadable -- a primitive-only collision, or no collision element at all
-- its visual mesh) is loaded via `yourdfpy` (structure parse only,
`load_meshes=False`) + `trimesh.load` (via a `package://`/relative-path
resolver that walks up from the URDF's directory trying successively
shorter path suffixes -- needed for e.g. Allegro's
`package://drake/manipulation/models/allegro_hand_description/meshes/
base_link.obj`, which only resolves once the `drake/manipulation/models/
allegro_hand_description/` prefix is stripped). Some hands (ORCA) put no
geometry at all on the digit-chain body itself -- a zero-mass `*_jointbody`
pivot link -- and carry it on a link one or more **fixed** joints away;
`load_mesh_with_fixed_fallback` walks those fixed-joint descendants (the
same pattern as `design_space.py`'s `_nearest_geometry`, re-expressing the
found mesh in the original body's frame via the zero-config FK transform
between the two). The mesh is sliced at 25/50/75% of the link's length,
perpendicular to its own axis, in its own local (link) frame; the largest
closed loop (by area) is kept.

### Hands run (11/11 available; 146 finger links, 413 sections)

| hand | source | mesh format | links | sections |
|---|---|---|---|---|
| SHARPA (`sharpa_left_on_iiwa14`) | in-repo (`assets/urdf/kuka_sharpa_description`) | STL | 15 | 45 |
| Allegro | `~/karma/karma-data/.../Allegro` | OBJ (visual; collision is boxes) | 16 | 44 |
| LEAP | same tree, `Leap/` | OBJ (collision) | 16 | 43 |
| Inspire | same tree, `Inspire/` | OBJ (collision; visual is `.glb`) | 12 | 36 |
| Barrett | same tree, `Barrett/` | OBJ (collision) | 5 | 11 |
| D'Claw | same tree, `Dclaw/` | OBJ (collision) | 9 | 24 |
| Ability | same tree, `Ability/` | STL (collision) | 10 | 30 |
| xhand | same tree, `xhand/` | STL (collision, `package://`) | 12 | 36 |
| Wuji | same tree, `Wuji/` | STL (collision, relative `../meshes/...`) | 16 | 39 |
| Tesollo (dg5f) | same tree, `Tessolo 5f/dg5f_description` | STL (collision, `package://`) | 20 | 60 |
| ORCA | same tree, `Orca/` | STL (collision, `package://`; jointbody fixed-fallback) | 15 | 45 |

### Skipped: Shadow (DAE-only)

Shadow is skipped, and said so here per the plan's instruction. The
manifest's locally-resolvable kinematic URDF for Shadow
(`shadow_right_local`, at `~/karma/karma-hand-metric/robots/urdfs/
shadowhand.urdf`, used by E13 for the representation check) references only
`package://sr_description/meshes/components/*.dae` geometry. `trimesh` can
in fact load `.dae` in the `piper` env (`pycollada` is installed there), and
the *original* Shadow ROS package under `~/karma/karma-data/.../Shadow` does
ship STL alternatives for its components (71 `.stl` files under
`Shadow/meshes/components`) -- but that package's link names/layout are
generated by `xacro` macros (`hand_e.urdf.xacro`, `process_hand_type_
version_side_parameters.xacro`, etc.) and do not line up 1:1 with
`shadow_right_local`'s already-expanded link names without running that
macro expansion, which is out of scope for this time-boxed study. Skipped
and noted, not silently dropped.

Allegro is included via its **visual** mesh (its collision geometry is
14 box primitives, not a mesh at all -- there is nothing to slice) -- this
is itself a small data point: at least one shipped hand model already
treats its own collision shapes as boxes, for reasons its authors did not
record here.

## Fits

Every fit is centred on the link's own central (joint-to-joint) axis, not
the section contour's own centroid -- the physically meaningful question is
"how far is the surface from the capsule's/rounded-rect's actual axis",
matching how a capsule primitive is defined.

- **Circle (capsule).** `radius = sqrt(area / pi)` (equal-area) and the 95th
  percentile of boundary-vertex distance from the axis (min circle covering
  95% of the boundary). Boundary error = `| |p| - radius |` at every
  boundary vertex.
- **Rounded rectangle.** Oriented to the section's own principal axes (2nd
  moments of the contour about its own centroid, translated back to the
  axis-centred origin), free `(w, h, r)` fit by coordinate-descent
  minimising mean squared boundary error against the standard 2D
  rounded-box signed-distance function (Inigo Quilez's `sdRoundBox`).
  Verified on synthetic data (`test_e14_cross_section.py`): a circle section
  matches the circle model to numerical precision; a known rounded rectangle
  is recovered to 2-10% (the coordinate-descent optimiser's own convergence
  tolerance, not a closed-form solve).
- **Template search.** One normalised shape `(h/w, r/h)`, grid-searched over
  `h/w in [0.3, 1.0]` and `r/h in [0.05, 0.45]`; for each candidate shape,
  each LINK gets its own best size `w` (1D search minimising that link's own
  80th-percentile max boundary error across its up-to-3 sections); the
  shape's score is the 80th percentile of max boundary error pooled over
  every section in every hand, using each link's own best size. This mirrors
  how a capsule already works today (one radius per link) as closely as
  possible while adding only the shape as a hand-wide constant.

## Results

### Per-section independent fits (each shape's own best fit, no shared size or shape across links)

Pooled over all 413 sections (median / mean / max, mm):

| fit | mean boundary error | max boundary error |
|---|---|---|
| circle (equal-area) | 2.5 / 3.6 / 27.4 | 4.8 / 6.2 / 29.9 |
| rounded rectangle (free w, h, r per section) | 0.9 / 1.2 / 7.3 | 2.7 / 3.8 / 20.2 |

Median fitted rounded-rectangle shape: `h/w = 0.80`, `r/min(w,h) = 0.15` --
distinctly non-circular (a circle is `r/min(w,h) = 0.5`) and distinctly
non-square-cornered. Per hand, the rounded rectangle's median max error is
roughly half the circle's at every one of the 11 hands (e.g. LEAP: 13.2 mm
circle vs 8.8 mm rounded rect; Tesollo: 2.7 mm vs 1.1 mm; SHARPA: 2.0 mm vs
1.7 mm) -- full per-hand numbers in `E14_cross_section/result.json`.

**Reading this correctly:** this says real finger cross-sections, taken
individually, are shaped more like rounded rectangles than circles. It does
**not** say a shared rounded-rectangle template will do better than a shared
capsule radius -- that is a different question, answered next.

### Template search: one shape, one size per link (the fair comparison to "one radius per link")

Best single template: `h/w = 1.00`, `r/h = 0.45` (the edge of the searched
grid nearest a circle: `h/w = 1`, `r/h = 0.5` is exactly a circle) --
80th-percentile pooled max boundary error 6.4 mm.

| fit | fraction of sections within 2 mm | within 3 mm |
|---|---|---|
| (a) best single template (one size per link) | 0.15 | 0.41 |
| (b) per-link-radius capsule | 0.15 | 0.39 |
| (c) one radius per hand (single capsule radius per hand) | 0.10 | 0.23 |
| (d) one radius for all hands (single global capsule radius = 9.4 mm) | 0.05 | 0.17 |

(a) and (b) are within noise of each other. Both are well ahead of (c) and
(d) -- the thing that actually buys accuracy is **fitting per link**, not
the choice of circle vs. rounded rectangle.

### Why the template search converges near a circle

The per-section independent fits (previous subsection) show real links
disagree substantially on the "right" aspect ratio and corner roundness --
some sections are close to circular, others (like SHARPA's thumb CMC
bracket, which houses an actuator, not a slender phalanx: mean boundary
error 7.6 mm against its own best-fit circle at one slice) are chunky and
irregular no matter what smooth convex shape is tried. A single shared
`(h/w, r/h)` has to serve every link's own best aspect ratio at once; the
grid search's own gradient pushes it toward the one shape that is least bad
on average across a population with no consistent aspect ratio -- which is
the roundest one available (a rounded rectangle degenerates to a circle as
`r/h -> 0.5` and `h/w -> 1`; see `test_rounded_rect_sdf_degenerates_to_
circle_when_w_eq_h_eq_2r`). That the search runs straight to the edge of its
own grid, rather than settling at an interior optimum, is itself evidence
that there is no non-circular consensus shape to find.

## Contact cost (desk study; no GPU run)

The plan's framing: a capsule is native to PhysX. A rounded rectangle is
not. Three ways to approximate one, assessed against documentation actually
read this session (PhysX 5.4 SDK docs, `nvidia-omniverse.github.io/PhysX`):

**(i) Box primitive + per-shape `restOffset` equal to the corner radius.**
`restOffset`/`contactOffset` are confirmed **per-shape** (`PxShape::
setRestOffset()`/`setContactOffset()`), with `contactOffset` defaulting to
0.02 m at default `PxTolerancesScale` and the documented constraint
`contactOffset > restOffset` [PhysX 5.4, Advanced Collision Detection].
*What is established:* this is a real, per-shape, standard PhysX 5 API. A
positive `restOffset` makes two shapes settle with a **gap** equal to the
sum of their `restOffset`s ("at rest, the distance between two vertically
stacked objects is the sum of their rest offsets" -- same doc) -- useful for
avoiding jitter/tunneling. *What is NOT established by anything read this
session:* whether `restOffset` on a **box** actually reshapes the contact
manifold into a true Minkowski-rounded box (rounded corners/edges, not just
a uniform face-normal gap) for general poses -- e.g. a box corner striking a
flat surface at an angle. PhysX's own native rounding lives in its
support-mapped primitives (sphere, capsule); nothing read here confirms
`restOffset` gives a box the same corner-rounded *contact geometry*, only
that it changes the *distance* at which contacts are generated/rest. This
is exactly the kind of claim the plan says needs a GPU benchmark, not a docs
read, to settle (see below).

**(ii) Convex hull mesh.** *Established:* convex hulls need `PxCookingParams
::buildGPUData = true` at cook time to get GPU contact generation at all;
without it (or above a size limit) they fall back to CPU. The documented
size limit is precise: **"Convex hulls with more than 64 vertices or
polygons or with more than 32 vertices per-face will have their contacts
processed by the CPU rather than the GPU"** [PhysX 5.4, GPU Rigid Bodies].
A rounded box needs meaningfully more than 8 vertices to look rounded at all
(a coarse hull with a handful of rounded-corner facets per corner --
8 corners x roughly 3-6 extra vertices each -- lands in the 30-60 vertex
range quickly), i.e. comfortably inside the 64-vertex GPU ceiling for a
*single* rounded box, but with much less margin than a box (8 vertices) or
a capsule (a pure support-mapped primitive, no vertex/face budget at all).
At the scale of a full-hand policy (dozens of shapes per hand, thousands of
parallel envs), being close to a hard CPU-fallback cliff is a real risk that
a primitive shape does not carry. *Not established here:* the actual
wall-clock GPU contact-generation cost of N convex hulls vs. N capsules at
this vertex count, at RL-relevant batch sizes -- the docs state a
capability/fallback threshold, not a cost curve.

**(iii) Two parallel capsules** (spanning a rectangle's long axis with two
capsule "rails"). Native primitives, no cooking/vertex budget at all, but
doubles the shape count per link and the resulting swept volume is not the
same shape as a rounded box (two capsules leave gaps/overlaps a single
rounded box wouldn't have, worse as the aspect ratio moves from 1:1).

**Existing policy geometry, for context.** The policy's own per-joint token
already uses **boxes**, not capsules or rounded rectangles: `hand_sampler/
design_space.py`'s `joint_link_boxes`/`joint_boxes` produce an oriented box
per joint-link (`token_box`, an explicit 4-corner box) as the one
description a policy reads for any hand. So the RL side of this system
already consumes box-shaped tokens today; the question this study answers is
about the **simulated collision geometry** (currently capsules, per
`geometry.py`, unchanged by this study), a separate concern from the token
representation.

### What is established vs. what would need a GPU benchmark

**Established (documentation, read this session):**
- `restOffset`/`contactOffset` are per-shape PhysX 5 API, with a documented
  default and ordering constraint.
- Convex hulls need explicit GPU cooking and have a hard, documented
  vertex/face ceiling (64 vertices or polygons, 32 per face) above which
  PhysX silently falls back to CPU contact generation.
- Capsules and boxes are native primitives; a rounded box is not one of
  PhysX 5's primitive geometry types.

**Not established (would need a GPU benchmark, per the plan's own
guardrail -- no GPU jobs run in this study):**
- Whether box + `restOffset` actually produces rounded-corner contact
  behaviour in general poses, or only a uniform rest/contact gap.
- The actual GPU contact-generation wall-clock cost of capsule vs.
  low-vertex convex hull vs. box+restOffset, at RL-relevant parallel-env
  counts, on this project's actual hand shapes.
- Whether the convex-hull vertex ceiling is ever approached in practice once
  every link in a full hand model needs its own rounded-box hull.

**Proposed small benchmark for the RL owners (not run here):** in
Isaac Lab/PhysX, instantiate N (e.g. 4096) parallel single-joint pendulums,
each with one link's collider swapped between (a) a capsule, (b) a box with
`restOffset` set to a candidate corner radius, (c) a convex hull of a
rounded box at ~40 vertices, and (d) two parallel capsules; drop the link
onto a fixed rounded/flat obstacle at a range of impact angles (including
corner-first for the box/hull cases) and measure (1) steps/second at fixed
substep count, (2) whether the settled contact gap and corner behaviour
match the intended rounded shape (I.e., does the box+`restOffset` case
actually look rounded when it lands corner-first, or does it register a
sharp-corner contact with an offset gap), (3) whether case (c) ever exceeds
the 64-vertex/32-per-face GPU ceiling once corner tessellation is increased
for visual quality.

## Recommendation

Keep capsules. The fit-quality case for a single rounded-rectangle template
is a tie with per-link capsule radii (15%/41% vs 15%/39% of sections within
2/3 mm), not an improvement, once the template is constrained to one shape
usable across all links (the constraint any real "one cross-section fits
most" proposal would need). The genuine finding here -- that individual
cross-sections are on average about twice as close to a rounded rectangle
as to a circle -- is a fact about individual links, not a case for a shared
template, and the template search's own convergence to the near-circular
edge of its search grid is direct evidence there is no consensus non-circular
shape to standardise on. On the cost side, nothing read this session
suggests a rounded rectangle would be cheaper or simpler than a capsule in
PhysX's GPU contact pipeline: capsules are a native primitive with no
cooking step and no vertex budget; every rounded-rectangle approximation
considered here (box+restOffset, convex hull, twin capsules) is either of
uncertain contact-shape fidelity (box+restOffset) or carries a real
resource ceiling/complexity cost a capsule does not (convex hull's
documented 64-vertex GPU cliff; twin capsules' doubled shape count). If a
future need (e.g. matching link cross-sections more tightly for a specific
grasp-contact-accuracy study) reopens this question, the benchmark proposed
above is the next step, not another docs-only pass.

## Artifacts

- `hand_sampler/grammar/experiments/e14_cross_section.py` -- the experiment.
- `hand_sampler/grammar_bench/refgen/mesh_sections.py` -- mesh I/O (lazy
  `trimesh`/`yourdfpy`) + pure-numpy shape fitting.
- `hand_sampler/grammar_bench/tests/test_e14_cross_section.py` -- synthetic
  geometry tests (no mesh library needed) + a real-hand (SHARPA) smoke test,
  `local-only:`-skipped where `trimesh`/`yourdfpy` are not importable.
- `project-notes/grammar/experiments/E14_cross_section/result.json`,
  `summary.md` -- full per-hand, per-link, per-section data and the template
  search, from the real run (`piper` env, 11/11 target hands, 146 links,
  413 sections).
