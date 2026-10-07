# Locking in the hand grammar (2026-10-07)

The grammar combines our grammar (`martin/hand-grammar`), Vatsal's physical grammar (`2026-10-02_physical_grammar`, `hand_sampler/DESIGN.md`) and Kushal's uniform commercial hands (`2026-10-06-controlled_wuji_experiments`, `assets/urdf/unified_dynamics_commercial_hands/`). The order of work is: the grammar, then the simulator rules (Evolution Rules) with coarse and fine steps, then the viability checks. Martin decides each item; each entry records the decision and the reason for it.

## Decisions

### Principle: limits come from the commercial hands, not from one motor
- Every limit is chosen so that the smallest and the largest commercial hands fit. No limit is derived from a particular actuator (Dynamixel XM335, Feetech). Restricting evolution to designs our hardware can build is a later, separate step (a selection or an extra rule set), not part of the grammar or the Evolution Rules.
- **Why (Martin, 2026-10-07):** the grammar must be able to express every commercial hand, and the shared policy has to work on all of them; hardware constraints can be imposed afterwards.


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

### 10. Finger spacing: neighbouring finger bases at least one link width (19 mm) apart
- **Why:** the commercial hands' closest non-thumb finger bases are 19.2-101.7 mm apart (Inspire 19.2, Ability 20.1, XHand 20.2, SHARPA 20.5, Wuji v2 20.9, Wuji v1 22.0, Shadow 22.4, SVH 23.3, Orca 23.4, Tesollo 24.8, MIDAS 31.0, ...). Spacing >= link width means bases cannot overlap and rejects no commercial hand (Inspire passes by 0.2 mm); width + 5 mm (24 mm) would reject 9 of 16, 29 mm 10 and Vatsal's 35 mm 11, because those hands' own fingers are 12-18 mm wide and the shared 19 mm width uses up their clearance. Only the ARMS skeleton model (15.8 mm) fails. Thumbs sit 42-113 mm from the nearest finger.

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
- **Cross-section: 19 mm wide (along the flexion axis) x 18 mm high (along the closing direction), corner radius 6 mm**, the median over 16 commercial hands (208 finger links, each hand weighted equally; IQR w 15.4-25.0, h 14.4-21.6, r 3.8-7.2 mm). Study: `project-notes/grammar/link-cross-section-study.md` (a29051c). Fingertip links are about 5 mm thinner than proximal ones on real hands; one global section ignores that, for simplicity. It cannot house an XM335 (19 x 22 mm body; a rounded box around it is at least 22.5 x 25.5 mm): by the principle above, buildability is a later step.
- **Why rounded boxes (dynamics):** what matters for the shared policy is how an object rests and rolls on the fingers, not only the outline. Flat faces give stable contact patches like real finger pads, and rounded edges let the object roll over them; a capsule is round everywhere (no flat pad) and a sharp box catches on its edges. By outline alone the shapes differ little (mean IoU with the real slices: rounded box 0.632, 19 mm circle 0.628, sharp box 0.618), so the case rests on contact behaviour and the probe.
- **Why (probe, Isaac Sim 5.1, RTX 4090, 4096 envs, ms per physics step):** capsule 5.1-5.3, sharp box 6.3-6.5, rounded box (hull + rest offset) 5.6. The hull version gave exact rounded-edge contact normals and no gap. The cheaper-looking alternative, a plain box collider with a rest offset, costs the same but is broken at edges against our box-shaped object: the object sank 4-5 mm into the rounded shell with no force, then was pushed along the box's face normals. One parameterisation covers capsule-like (Wuji), boxy (LEAP) and rounded (SHARPA) fingers; a rounded box is slimmer than a capsule around the same motor; the policy's per-link token features already describe links as boxes.
- **Risks:** the grammar's overlap and reach checks and the viewer must switch from capsule distance to rounded-box distance so they match the simulator; the behaviour relies on PhysX's GPU convex-convex path, so add a regression test on edge normals; keep the core at least about 1 mm thick or hull cooking falls back to CPU.

### 8. Link length: 0 mm or at least 15 mm; a fingertip link at least 10 mm, never 0 mm
- A link between two joints is either 0 mm (two joints at one point) or at least 15 mm. A finger's last link is at least 10 mm. Lengths between 1 and 14 mm are not allowed; steps skip the gap (shortening 15 mm gives 0 mm, lengthening 0 mm gives 15 mm). Longest link 90 mm.
- **Why:** mechanically, two co-located joints are convenient (one two-axis module) while joints a few millimetres apart are awkward, and longer links are easy again since motors stack in series (Martin). The floor is set by the commercial hands, not by a motor (principle above): Vatsal's motor-derived 20 mm floor would reject 13 real bones in 7 hands (Allegro 16.4, Inspire 16.8, XHand 17.8, LEAP 19.3, Wuji's thumb 16.1 mm, ...). With 15 mm every commercial bone fits except knuckle offsets of 4.6 mm (Wuji v1) and 5.0 mm (SHARPA), which snap to 0 mm within the 5 mm fit target, and the ARMS skeleton model's 9.3 mm. The shortest real fingertip link is SVH's 14.0 mm (the next is 25 mm), so the tip floor is 10 mm rather than Vatsal's 15 mm. The longest real link is DClaw's 84 mm. 27 of the 220 real bones are 0 mm.

## Still to decide
- Mount area and palm plate thickness: from the commercial extremes (measured during the build), no motor numbers.
16. Switching a joint's kind in place: in the kind grammar this is the only coarse axis step (the three kinds are 90 degrees apart and the fine tilt stops at 15), so it is not redundant with axis steps; proposed weighting by the commercial hands' mix of kinds (to be measured; expected about 70% flexion, 25% abduction, 5% roll).

Checks come after the grammar and the Evolution Rules. Candidates: the overlap check, Vatsal's curl score, Martin's thumb-finger workspace overlap, and an opposition test (none exists yet).
