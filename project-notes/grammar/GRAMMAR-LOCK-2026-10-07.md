# Locking in the hand grammar (2026-10-07)

The grammar combines our grammar (`martin/hand-grammar`), Vatsal's physical grammar (`2026-10-02_physical_grammar`, `hand_sampler/DESIGN.md`) and Kushal's uniform commercial hands (`2026-10-06-controlled_wuji_experiments`, `assets/urdf/unified_dynamics_commercial_hands/`). The order of work is: the grammar, then the simulator rules (Evolution Rules) with coarse and fine steps, then the viability checks. Martin decides each item; each entry records the decision and the reason for it.

## Decisions

### 1. Palm: the convex hull of the finger bases, plus hinged sections only where a palm joint is needed
- The main palm is rigid. Its outline is the convex hull of the finger bases plus a disc of 20 mm radius at the palm centre, so a hand with two fingers still has a palm for the object to rest on (Vatsal's construction).
- An extra palm section exists only when the hand has a palm joint. It is hinged to the main palm, and its outline is the convex hull of its own finger bases and the hinge. A hand has at most 2 palm joints.
- There are no rigid extra palm pieces and no palm length parameter.
- **Why:** an extra palm piece without a joint does nothing that the hull cannot do, since the hull already reaches wherever a finger sits. The palm then has no parameters of its own: it is set entirely by the finger bases and the palm joints. Hinged sections keep the palm joints needed by Shadow, SVH and arms_skel, and keep multiple palm DOFs possible.

### 2. Finger mount: a (y, z) position on the palm plate, a facing, and one tilt out of the plate
- Position (y, z) on the plate's plane; facing = the in-plane direction the finger leaves in (Vatsal's `Mount(y, z, facing)`); plus one tilt out of the plate's plane.
- **Why:** four numbers instead of our five (host, fraction along it, sideways offset, three mount angles), and no dependence on a host link now that the palm is a hull. Vatsal's facing alone keeps every finger in the plate's plane; the added tilt lets a thumb angle across the palm.

### 3. Joint axes: a kind in the coarse stage, plus a small tilt in the fine stage
- Coarse: every joint has a kind, fixed relative to its link: flexion (closes toward the palm), abduction (spreads), roll (spins about the link) (Vatsal's three kinds).
- Sign convention (Kushal's): + = flex toward the grasp point, + = spread toward +y, 0 = home.
- Fine: the axis may tilt away from its kind in 5 degree steps, up to +/-15 degrees in each of two directions. Every hand starts at zero tilt; the coarse stage never changes it.
- **Why:** with free axes (our grammar), nothing made "positive" mean "curl": the curl rule only kept each axis roughly perpendicular to its bone, and its direction around the bone was random, so random fingers curled only through the rest bends. Kinds make positive always mean curl. Kinds alone (Vatsal's grammar) cannot express slightly angled axes, such as fingers that converge as they curl; the fine tilt covers those without changing what positive means, and should bring the commercial fits back near 5 mm. If commercial hands need more than 15 degrees, report it with numbers before widening it.
- This also settles item 18 (closer commercial fits): the fine tilt, measured before it is final.

### 4. No angles between links (no rest bends, no leans)
- At the zero pose every link continues straight from the one before; a finger is a straight line at rest and bends only through its joints.
- **Why:** a built-in bend is an extra number per link that only shapes the finger at rest (a permanently bent bone). The thumb's angled base is covered by the mount's facing and tilt (item 2), and slightly off-square joints by the fine tilt (item 3). Vatsal's own design note says a right-angle corner is better expressed as an abduction joint next to a flexion joint. If a commercial hand misses the fit target because of genuinely bent links, report it before adding anything back.

### 5. Joint ranges: one global range per kind
- Flexion -30 to +90 degrees; abduction +/-30 degrees; roll +/-90 degrees; palm joints +/-30 degrees. No per-hand or per-joint ranges.
- **Why:** joint limits do not change the kinematic tree, so they need not vary per hand. A single +/-90 degree range (Vatsal; raised by Kushal on Oct 5) would let a side-to-side joint swing 90 degrees each way, which no real hand does; Martin's 9/23 note already observed that abduction needs a narrower range than flexion.

### 7. Joint types: the grammar has hinge, coupled and sliding joints; evolution uses hinge and coupled only
- The grammar can express hinge (revolute), coupled (a joint that follows an earlier joint) and sliding (prismatic) joints. Continuous (unlimited spin) joints are dropped.
- The Evolution Rules allow hinge and coupled joints only.
- **Why:** some research hands use sliding joints, so the grammar should be able to describe them. Continuous spin needs a slip ring and no hand needs it. Coupled joints are how several commercial hands drive a distal joint (Inspire, Ability, SVH, Shadow, Barrett), and the simulator's mimic joints (validated 2026-10-07, tie error at most 1.3e-3 rad) can carry a coupling ratio directly.

### 9. Fingers 2-6, joints per finger 1-5
- **Why:** the maxima match the simulator's 36-slot layout (6 finger slots of 5 joints) and Vatsal's grammar; the minimum of 2 follows Kushal's result that a single Wuji finger cannot learn the task (0.01 goals/episode, 2026-10-07).

### 11. Palm joints: at most 6, one per finger at most, and none without a finger
- **Why:** a palm joint without a finger is not a palm (Martin, 2026-10-06), so a hand can have at most one palm joint per finger. The simulator is ours to adapt; the 36-slot layout already gives every finger slot its own palm-joint carrier, so 6 palm joints need no simulator change.

### 12. Finger length at most 250 mm
- **Why:** the longest commercial finger (DClaw, 221 mm) x 1.1, rounded. Without it, random fingers grow longer than any real hand's (5-6 bones of 15-80 mm add up to 240-285 mm).

### 13. No arm-clearance rule
- **Why:** the task is a fixed hand with no arm. Add Vatsal's rule when an arm comes back.

### 14. Kushal's uniform physical properties for every hand
- 0.5 N m torque, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, link density 1750 kg/m^3, friction 0.5, generated and commercial hands alike.
- **Why:** consistency: hands then differ only in shape, and grammar hands share the settings Kushal's working policies were trained with (Wuji v2 at about 40 goals/episode, one policy over 52 Wuji hands).

### 7a. No branching fingers
- Every finger mounts on the palm; no finger grows off another finger's bone.
- **Why:** no hand we model has one, and it would need its own mount design in the hull-palm representation.

### 7b. Coupled joints: one global ratio, 1.1, with zero offset
- A coupled joint follows the joint just before it in the same finger at a ratio of 1.1 and an offset of 0. There is no per-joint ratio.
- **Why:** Martin asked for one common value taken from the commercial hands. Among the 18 within-finger couplings in the URDFs (Ability 4 x 1.059; Inspire 4 x 1.064 plus its thumb's 1.334 and 0.667; SVH's 1.015, 1.045, 1.045, 1.359, 1.359, 1.421, 1.423, 1.449), the most common value is about 1.06 (8 of 18), the median 1.064 and the mean 1.145; 1.1 is the rounded value between them. Offsets are 0 except Inspire's -0.045 rad. Not counted: SVH's cross-finger spread coupling (0.5) and thumb-to-palm coupling (1.0), ARMS's palm coupling (2.72), and hands that couple mechanically without saying so in their URDF (Shadow's distal joints, Barrett).

### 15. Adding and removing joints: split and merge
- Insert a joint by splitting a link in two; remove one by merging two links. Removing a finger stays a single step that removes any finger, whatever its length (decided 2026-10-06).
- **Why:** split and merge keep the finger's length and reach, so each step is a small change. Losing a finger must not take many steps, as it did when only single-joint fingers could be removed.

### 19. Commercial reference set: every hand any of us has
- Ours: Ability, Allegro, ARMS (skeleton), Barrett, DClaw, Dex1, Inspire, LEAP, Orca, Shadow, SHARPA, SVH, Tesollo DG-5F, Wuji (v1), XHand. Kushal's: Dex3, Wuji v2 (left and right), plus his uniform-dynamics versions of the shared hands. Vatsal's: MIDAS. Kushal's 52 Wuji v2 finger subsets are not included; the regular Wuji v2 is.
- **Why:** as many real hands as possible, to test how well the grammar covers the market and to give the policy the most held-out hands.

### 20. Fit metric: revisit later
- For now, ours (largest joint-position, axis and fingertip error against 5 mm / 10 degrees), counting a pure difference in the zero pose as a match (Vatsal). Martin expects a better similarity metric is needed.

### 17. Coarse steps twice as coarse; fine steps as they were
| Parameter | Coarse | Fine |
|---|---|---|
| link length | 10 mm | 1 mm |
| finger position on the palm | 10 mm | 1 mm |
| finger facing and tilt | 30 degrees | 5 degrees |
| palm joint hinge position | 10 mm | 1 mm |
| palm joint axis | 30 degrees | 5 degrees |
| joint axis tilt (item 3) | none | 5 degrees, up to +/-15 |

No coupling step (one global ratio, 7b). Every coarse value lies on the fine grid.
- **Why:** Martin wanted the coarse stage coarser; the 2026-09-18 meeting noted that mutations were not large enough between generations, and the coarse stage is the one meant to explore.

### 16 (partly). Coupling toggle, and split/merge for joints
- A mutation can switch a joint between independent and coupled to the joint before it (ratio 1.1). Adding and removing joints is split and merge (item 15).
- **Why:** with one global ratio, coupling is a yes/no property of a joint, so one toggle covers it.
- Still open: switching a joint's kind in place (see below).

### 6 (shape). Links are rounded boxes, one cross-section for every link of every hand
- Built in the simulator as an 8-vertex convex hull of the core box (outer size minus 2r) with PhysX `restOffset = r` and `contactOffset = r + 2 mm`, mass and inertia authored explicitly.
- The cross-section (width, height, corner radius) is the average of the commercial hands' finger links: study in progress, `project-notes/grammar/link-cross-section-study.md`.
- **Why (probe, Isaac Sim 5.1, RTX 4090, 4096 envs, ms per physics step):** capsule 5.1-5.3, sharp box 6.3-6.5, rounded box (hull + rest offset) 5.6. The hull version gave exact rounded-edge contact normals and no gap. The cheaper-looking alternative, a plain box collider with a rest offset, costs the same but is broken at edges against our box-shaped object: the object sank 4-5 mm into the rounded shell with no force, then was pushed along the box's face normals. One parameterisation covers capsule-like (Wuji), boxy (LEAP) and rounded (SHARPA) fingers; a rounded box is slimmer than a capsule around the same motor; the policy's per-link token features already describe links as boxes.
- **Risks:** the grammar's overlap and reach checks and the viewer must switch from capsule distance to rounded-box distance so they match the simulator; the behaviour relies on PhysX's GPU convex-convex path, so add a regression test on edge normals; keep the core at least about 1 mm thick or hull cooking falls back to CPU.

### 8. Link length: 0 mm, or 20 mm and up; never 0 mm for a fingertip link
- A link is either 0 mm (two joints at one point) or at least 20 mm. The last link of a finger is at least 15 mm (Vatsal) and never 0 mm. Lengths between 1 and 19 mm are not allowed.
- Length steps skip the gap: shortening a 20 mm link goes to 0 mm and lengthening a 0 mm link goes to 20 mm; above 20 mm the coarse and fine steps (10 mm and 1 mm) apply as usual.
- **Why:** mechanically, two co-located joints are convenient (one two-axis module), while two joints only a few millimetres apart are awkward to build; from 20 mm up it is easy again, since motors stack in series. Co-located joints are also common in commercial hands (27 of 220 projected bones are 0 mm, e.g. knuckles with abduction and flexion at one point). The grammar's 0 mm bones are already handled in the simulator (adjacent-pair collision filtering, a mass floor).

## Still to decide
10. Finger spacing: proposed at least the link width + 5 mm; Martin worries it will not fit commercial hands, so the cross-section study also measures every hand's finger spacing against candidate rules.
16. Switching a joint's kind in place: in the kind grammar this is the only coarse axis step (the three kinds are 90 degrees apart and the fine tilt stops at 15), so it is not redundant with axis steps; proposed weighting by the commercial hands' mix of kinds (to be measured; expected about 70% flexion, 25% abduction, 5% roll).

Checks come after the grammar and the Evolution Rules. Candidates: the overlap check, Vatsal's curl score, Martin's thumb-finger workspace overlap, and an opposition test (none exists yet).
