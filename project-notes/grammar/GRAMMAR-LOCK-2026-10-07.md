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

### 3. Joint axes: a direction (two angles in the link's frame), stepped like every other angle; kinds are derived labels
- The axis is a direction in the link's frame, two angles on the same grid as every other angle: 30 degrees coarse, 5 degrees fine. There is no separate kind parameter, no kind-switch operator and no tilt cap.
- Kinds (Vatsal's three: flexion closes toward the palm, abduction spreads, roll spins about the link) are labels computed from the direction, and are used for three things:
  1. **Sign:** positive is set by the convention, not stored: if the joint can close the finger at all, positive closes it toward the grasp point; otherwise + spreads toward +y; 0 = home (Kushal's convention, generalised).
  2. **Range:** a joint takes the global range of its nearest kind (item 5).
  3. **Random draw:** a new joint starts exactly on a kind, chosen with the commercial hands' mix of kinds (measured in `palm-and-axis-study.md`).
- Oblique axes (between two kinds, e.g. a thumb base angled so the thumb sweeps across the palm as it closes) are allowed anywhere on the 5 degree grid.
- **Closing direction = toward the grasp point (Martin, 2026-10-07).** A finger's link frame at q = 0: x along the finger (from facing and tilt); y, the closing direction, the unit vector square to x pointing at the grasp point G, y = normalise((G - base) - ((G - base) . x) x); z = x cross y. G is the object's start point above the palm (`derive.object_start_m`: 53.5 mm above the plate's centre plane, i.e. half the 37 mm plate + a 30 mm object half-size + 5 mm, over the main palm outline's centroid). A finger pointing straight at G falls back to +x (the grasp side), then to the tilt direction. Axis angles, kinds, signs, the random draw on kinds, conform and the simulator tables all use this frame. Before, y was the tilt direction (toward the palm normal), and nothing made fingers close toward a common point.
  - Random hands, + on a bending joint at the zero pose: the fingertip moved nearer G for 79% of bending joints before (2,285 joints), 100% after (a test checks it).
  - G depends on every finger base and palm hinge (through the outline's centroid), so moving one base turns every finger's closing direction a little; the stored axis angles are relative to it.
  - Commercial fidelity: 17 of 19 within 5 mm / 10 degrees before and after. Misses before: ARMS 5.1 mm, Wuji v1 5.1 mm / 10.4 degrees; after: ARMS 5.2 mm, SVH 5.04 mm (Wuji v1 now 4.0 mm / 3.4 degrees). Conform now fits each finger twice, because a finger's frame depends on all the bases: first with its base free (G from the first estimate of the bases), then with every base fixed where it settled (and spaced), on the G those bases set.
  - Derived kinds of the 283 commercial hinge joints: bend 192 -> 203, spread 78 -> 67, twist 13 -> 13; DClaw now reads as one spread and two bends per finger, as built. The random draw keeps the measured mix (69/27/4).
  - Random hands alone (before item 25): C1 40% -> 29%, because fingers in a row now close toward one point and meet; C2 64% -> 80%; viable 17% -> 17%. A parallel row with plain bend axes (0, 0) converges as it closes, so the simulator's hand-made validation hands splay their rows 15 degrees per finger.
- **Why:** with free axes as before, nothing made "positive" mean "curl", and random fingers curled only through the rest bends; the computed sign and the kind-based random draw fix that. Vatsal's fixed kinds cannot express oblique axes, and kinds plus a capped tilt (an earlier draft of this item) needed three different step types (a 90 degree switch, a coarse tilt and a fine tilt) and still could not reach a 45 degree thumb axis. Treating the axis like every other angle gives one rule for all parameters: coarse 10 mm or 30 degrees, fine 1 mm or 5 degrees. Commercial hands fit on the 5 degree grid with no cap.
- This settles item 16 (no kind-switch operator) and supersedes item 18.

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
| joint axis direction (item 3) | 30 degrees | 5 degrees |

No coupling step (one global ratio, 7b). Every coarse value lies on the fine grid.
- **Why:** Martin wanted the coarse stage coarser; the 2026-09-18 meeting noted that mutations were not large enough between generations, and the coarse stage is the one meant to explore.

### 16. Changing a joint: step its axis direction, toggle its coupling; split/merge to add or remove it
- A mutation can switch a joint between independent and coupled to the joint before it (ratio 1.1). Adding and removing joints is split and merge (item 15).
- **Why:** with one global ratio, coupling is a yes/no property of a joint, so one toggle covers it.
- No kind-switch operator: the axis direction is stepped directly (item 3).

### 6 (shape). Links are rounded boxes, one cross-section for every link of every hand
- Built in the simulator as an 8-vertex convex hull of the core box (outer size minus 2r) with PhysX `restOffset = r` and `contactOffset = r + 2 mm`, mass and inertia authored explicitly.
- **Cross-section: 19 mm wide (along the flexion axis) x 18 mm high (along the closing direction), corner radius 6 mm**, the median over 16 commercial hands (208 finger links, each hand weighted equally; IQR w 15.4-25.0, h 14.4-21.6, r 3.8-7.2 mm). Study: `project-notes/grammar/link-cross-section-study.md` (a29051c). Fingertip links are about 5 mm thinner than proximal ones on real hands; one global section ignores that, for simplicity. It cannot house an XM335 (19 x 22 mm body; a rounded box around it is at least 22.5 x 25.5 mm): by the principle above, buildability is a later step.
- **Why rounded boxes (dynamics):** what matters for the shared policy is how an object rests and rolls on the fingers, not only the outline. Flat faces give stable contact patches like real finger pads, and rounded edges let the object roll over them; a capsule is round everywhere (no flat pad) and a sharp box catches on its edges. By outline alone the shapes differ little (mean IoU with the real slices: rounded box 0.632, 19 mm circle 0.628, sharp box 0.618), so the case rests on contact behaviour and the probe.
- **Why (probe, Isaac Sim 5.1, RTX 4090, 4096 envs, ms per physics step):** capsule 5.1-5.3, sharp box 6.3-6.5, rounded box (hull + rest offset) 5.6. The hull version gave exact rounded-edge contact normals and no gap. The cheaper-looking alternative, a plain box collider with a rest offset, costs the same but is broken at edges against our box-shaped object: the object sank 4-5 mm into the rounded shell with no force, then was pushed along the box's face normals. One parameterisation covers capsule-like (Wuji), boxy (LEAP) and rounded (SHARPA) fingers; a rounded box is slimmer than a capsule around the same motor; the policy's per-link token features already describe links as boxes.
- **Risks:** the grammar's overlap and reach checks and the viewer must switch from capsule distance to rounded-box distance so they match the simulator; the behaviour relies on PhysX's GPU convex-convex path, so add a regression test on edge normals; keep the core at least about 1 mm thick or hull cooking falls back to CPU.

### 8. Link length: 0 mm or at least 15 mm; a fingertip link at least 10 mm, never 0 mm
- A link between two joints is either 0 mm (two joints at one point) or at least 15 mm. A finger's last link is at least 10 mm. Lengths between 1 and 14 mm are not allowed; steps skip the gap (shortening 15 mm gives 0 mm, lengthening 0 mm gives 15 mm). Longest link 90 mm.
- **Why:** mechanically, two co-located joints are convenient (one two-axis module) while joints a few millimetres apart are awkward, and longer links are easy again since motors stack in series (Martin). The floor is set by the commercial hands, not by a motor (principle above): Vatsal's motor-derived 20 mm floor would reject 13 real bones in 7 hands (Allegro 16.4, Inspire 16.8, XHand 17.8, LEAP 19.3, Wuji's thumb 16.1 mm, ...). With 15 mm every commercial bone fits except knuckle offsets of 4.6 mm (Wuji v1) and 5.0 mm (SHARPA), which snap to 0 mm within the 5 mm fit target, and the ARMS skeleton model's 9.3 mm. The shortest real fingertip link is SVH's 14.0 mm (the next is 25 mm), so the tip floor is 10 mm rather than Vatsal's 15 mm. The longest real link is DClaw's 84 mm. 27 of the 220 real bones are 0 mm.

### 21. Palm plate thickness 37 mm; finger bases within 125 mm of the palm centre
- Thickness: 37 mm, the median of 15 commercial palms (IQR 25-42 mm; DClaw's 6 mm mounting plate and Barrett's 86 mm motor housing are the extremes). Base distance from the palm centre, in the plate: 0-125 mm (SVH's thumb at 110.8 mm x 1.1, rounded; every other thumb is 48-77 mm and every non-thumb finger at most 62 mm). Study: `project-notes/grammar/palm-and-axis-study.md` (4108105).
- **Why:** the principle above: limits from the commercial extremes with a 10% margin.
- Superseded by item 22 (anchor at the wrist centre): The study's centre, the centroid of the non-thumb bases, lies on the knuckle row, so a disc there makes a palm that is only a strip along the knuckles; proposed: anchor at the wrist (the hand's mount point), so the hull spans wrist to knuckles like a real palm. The distance limit is re-measured from whichever anchor is chosen.

### 3 (evidence). The commercial axes support the unified axis design
- Of 263 finger and thumb joints, after sliding each joint along its own axis onto the finger's centre line (exact for the motion): fingers lie within 15 degrees of a kind 96% of the time and within 30 degrees 99%; thumbs only 45% within 15 degrees, 85% within 30 and all within 45 (worst: thumb bases of Inspire, LEAP, SHARPA, ARMS, Allegro). Palm joints are often far from any kind (Shadow LFJ5 35, ARMS CMC4 45). A kind-plus-15-degree design would have missed most thumbs; directions stepped 30 / 5 degrees reach all of them. Kind mix for the random draw: flexion 69%, abduction 27%, roll 4%.

### 22. Palm anchor: the centre of the wrist
- The palm hull is the finger bases plus a heel disc at the wrist centre (the hand's mount point); finger positions (y, z) are measured from the wrist centre. The base-distance limit of item 21 is re-measured from the wrist.
- **Re-measured from the wrist (build, 2026-10-07), commercial extremes x 0.9 / x 1.1, rounded out to 5 mm:** finger bases 10-160 mm from the wrist centre (ARMS thumb 14.6 mm to Tesollo middle 141.2 mm); finger tilt out of the plate -30 to +90 degrees (Barrett thumb -25 to DClaw's fingers pointing straight out, +90); palm-joint hinges 15-110 mm from the wrist (ARMS CMC5 20.0 to Shadow LFJ5 98.1 mm). Facing: the full circle. These replace item 21's 0-125 mm, which was measured from the knuckle row.
- Not every commercial URDF's root is at a wrist: DClaw's is the centre of its base plate, Orca's is mid-palm, and Barrett's and Dex1's sit 45-56 mm below the plate in gripper housings.
- **Why:** real palms run from the wrist to the knuckles, and that area is where the object rests; a disc at the knuckle row would make palms a strip along the knuckles.

### 23. Viability checks: two
- **C1, self-overlap:** no two links (rounded-box distance, the palm plate included) overlap by more than 3 mm at the zero pose or at the episode start pose. **Why:** deeper initial overlaps made PhysX push links apart at over 100 rad/s within a step or two.
- **C2, fingertip workspace overlap (Martin's):** at least one pair of fingers whose fingertips' reachable regions intersect above the palm (above the plate, over its footprint). The tolerance is calibrated so every commercial hand passes. **Why:** a hand whose fingertips can never meet above the palm cannot hold or turn an object; this tests it from kinematics alone, cheaply.
- **Dropped:** the object-start-height check (the object starts at a fixed spot above the palm defined by the task, not by the fingertips); the old "2 fingertips within 5 cm of a spawn point" reach check (Martin disliked it); Vatsal's curl score as a check (positive rotation already closes toward the palm by convention; the score measures how much a finger can close, which C2 covers where it matters).
- **Deferred:** C5, a physics grasp test (finding a stable grasp in simulation, like the HORA grasp cache). Bring it up again once the grammar work is done.

### 24. Fingers sit on the palm's rim, never in the middle of the palm
- Every finger base lies on the rim of its plate: at most 8 mm inside the convex hull of that plate's points (the main plate: the heel disc at the wrist centre, the palm-joint hinges and its fingers' bases; a palm joint's section: its hinge and its fingers' bases). A grammar rule, kept by construction like the spacing rule: random hands place each base only where every base stays on the rim, and mutations only offer moves, added fingers and palm-joint changes that keep it; removing a finger always keeps it.
- **Tolerance 8 mm:** the smallest whole millimetre every commercial hand passes. Worst insets: Allegro's thumb 8.0 mm (inside the hull of its heel disc and index base), Inspire's thumb 5.6, ARMS's thumb 5.3, Wuji v2's thumbs 5.0 and 4.9, Tesollo 2.9, SVH 2.0, Wuji v1 1.3, every other hand 0 (middle knuckles of a straight row lie on the hull's edge). Dex3 breaks the rule: its thumb sits 22.5 mm inside (between the wrist and its two fingers on the palm's centre line), more than the 10 mm we allow a tolerance to grow for one hand, so it is reported, not accommodated; the simulator still admits it as a commercial hand.
- **Why (Martin, 2026-10-07):** "the fingers should be able to be on any of the edges but not sticking out of the middle of the palm". With uniform random hands (878a3d4) bases could land anywhere on the plate.
- Effect on random hands (Evolution Rules, coarse, 1000 hands): C1 37% -> 40%, C2 66% -> 64%, viable 16% -> 17% (6.1 -> 5.8 tries per viable hand).

### 25. Generation keeps C1: random hands are built clear, mutations return the first child that passes (Vatsal's method)
- **Random hands** are built finger by finger. Each finger (base, facing, tilt, joints and links) is drawn again, up to 50 times, until the hand so far is clear: no two bodies overlap at the zero pose (rounded-box distance at least 0) and none by more than 3 mm at the start pose (C1). Pairs that touch by construction are excluded (consecutive links, a link and its plate through 0 mm links, a palm section and the main plate). A finger that cannot be placed ends the hand there (fewer fingers); a hand left with fewer than the minimum is drawn again. The optional palm joint goes only on a finger for which the hand stays clear.
- **Mutations:** each operator lists its applications, shuffles them and returns the first child that keeps the rules and C1 (a parent that already fails C1, such as a commercial hand, needs a child no worse). `mutate` picks an operator uniformly among those with applications at the stage and returns nothing if that operator has no such child; `can_act`, which sets the viewer's operator buttons, stops at the first passing child.
- C1 and C2 stay viability checks (viewer panel, simulator admission). C2 is not built into generation, and no bias toward it was tried: the obvious ones (fingers aimed at the palm centre, shorter fingers) would skew the uniform draw.
- **Why (Martin, 2026-10-07):** Vatsal's sanity checks belong in generation, so that feasible designs are more likely; 60% of random draws failed C1 and were wasted.
- **Effect** (Evolution Rules, coarse; 1000 random hands; 500 mutations of viable parents; before = the commit before both changes, after = both changes):

  | | before | after |
  |---|---|---|
  | C1 / C2 / viable | 40% / 64% / 17% | 100% / 74% / 74% |
  | random draws per viable hand | 5.8 | 1.4 |
  | draw time per hand | 2.7 ms | 30 ms |
  | draw + checks per viable hand | 0.12 s | 0.07 s |
  | mutate() calls returning a child | 100% | 98.4% |
  | children passing C1 / viable | 95% / 90% | 100% / 96% |
  | time per mutation | 20 ms | 11 ms |
  | fingers per hand, joints per finger, finger length | 3.95, 3.01, 140 mm | 4.02, 2.79, 134 mm |
  | base spread (std y, z), facing and bearing spread, tilt std | 80, 80 mm; 0.99, 1.00; 42 deg | 79, 80 mm; 0.99, 0.98; 43 deg |
  | mean distance between two random hands | 155 | 152 |

  The clearance costs a little diversity in finger structure (fewer joints, shorter fingers: long many-jointed fingers collide more often); placement, facing and tilt stay uniform. Mutations got faster because an operator stops at its first passing child instead of listing every valid one.

## Revision of 2026-10-14: the grammar built around the Feetech servo

Martin's notes of 10/13 (Notion, "Hand grammar Notes"), the Feetech study (`experiments/E15_feetech_buildability/summary.md`) and an independent review (Codex, `experiments/E16_grammar_rebuild/`) were decided on 10/14. The items below replace the matching items above; the rest stand. `GRAMMAR-OVERVIEW.md` describes the result step by step.

- **Principle: limits from the commercial hands -> two rule sets.** The grammar's own limits (No Rules) still come from the commercial hands; evolution and the simulator now use the **Feetech rules**, the hands buildable from Feetech HL-3915 servos (hinges only, bases on a 40 mm grid 40-90 mm out, facing outward, links at least the servo floors). **Why (Martin, 10/12-10/13):** evolve buildable hands first, the general rules later; the motor sets the geometry.
- **Item 1 (palm: hull of the bases plus a 20 mm disc; at most 2 palm joints) -> the hull of the mount disc and the servos' housings; at most one palm joint, a straight cut.** The mount (a 40 mm disc, 40 mm deep, behind the palm's centre; size assumed) is the smallest palm. A cut is a straight line across the finished palm (30 degree directions, 10 mm offsets, at least 30 mm from the centre so the mount stays whole); every finger beyond it rides on the part that hinges, +/-30 degrees, about the line in the plate's mid-plane; an 8.6 mm seam lets a 24 mm plate swing; the hinge servo sits on the main side at the spot nearest the chord's middle that clears every base servo. **Why (Martin, 10/13):** the hand mounts at its back; palm joints are decided on the finished palm, several fingers can share one; a straight cut of a convex palm leaves two convex plates (a bent cut leaves a notch).
- **Item 3 (closing toward the grasp point; the 5 degree fine grid) -> the tilt-defined frame, one 30 degree grid.** The closing direction is the palm normal turned by the tilt; every finger's frame depends on that finger alone, and + on every bend moves its fingertip toward the palm's working face. Axes are 31 directions. **Why (Martin, 10/13):** + should move every finger the same way; the grasp-point frame tied each finger's frame to all the bases. The cost: + does not always move a fingertip toward the object; C2 keeps catching hands whose fingertips cannot meet.
- **Item 4 (fine tilt) -> tilts of -30, 0, 30, 60, 90 degrees only.**
- **Item 6 (one 19 x 18 mm rounded-box cross-section) -> servo-sized links.** Each link is one box, square to its frame, around its yoke (the bracket around its joint's servo) and the servo it carries, or the fingertip pad. **Why (Martin, 10/13):** simulated Feetech hands should be the size they will be built.
- **Item 7 (Evolution Rules: hinge and coupled) -> the Feetech rules: hinge only.** Coupled and sliding joints stay in the grammar (No Rules). **Why (Martin, note 2.11.1):** a coupling or a slide needs a mechanism a single servo per joint does not give.
- **Items 7b and 8 (couplings at 18/180; 0 mm links at 27/220; links 0 or 15-90 mm) -> uniform draws; lengths from the servo.** Links are 0 or 10-90 mm in the grammar. Under the Feetech rules a link is at least the floor for the two axes it joins (E15's plane rule rounded up to 10 mm: bend to bend 50, spread to bend 40, spread to twist 30, at most 50 over all 31 axes), a fingertip at least 30 mm (at most 40); two joints share a point only between two different axes square to the finger, never two in a row, the earlier servo slid along its own axis (26.5-34 mm). Only the 69/27/4 kind mix is kept from the commercial hands. **Why:** E15; Martin's note 2.11.2 (crossings anywhere, a 2-2-1 thumb); rounding up to the grid costs at most 10 mm per link (bend to bend 50 instead of 41), a later fine stage can win it back.
- **Item 10 (19 mm spacing) and item 24 (the rim rule) -> one overlap rule.** Bases sit on a 40 mm grid; a finger's first servo must clear the mount and every other servo by a wall (2 mm), and C1 checks every solid. **Why:** no spacing short of 67 mm is safe for every pair of servo directions (Codex); with 40 mm, neighbouring first servos collide in 28% of direction pairs against 57% with 35 mm (`E16/ours/compare_layouts.py`); a base inside the palm is fine if its servos clear everything. Martin (10/14): keep a spacing that avoids most overlaps, 40 mm.
- **Item 11 (up to 6 palm joints) -> at most one** (item 1 above; item 1's old "at most 2" and item 11's "6" disagreed).
- **Item 17 (coarse and fine stages) -> one grid: 10 mm, 30 degrees;** a base steps by the rules' grid (40 mm). **Why (Martin, note 1):** treat coarse as normal, refinement is future work.
- **Item 20 (fit target 5 mm / 10 degrees) -> 10 mm / 20 degrees,** half the grid's steps; a fit keeps every hand's finger count, joints per finger, joint types and kinds. The palm cut's own errors are reported, not scored. **Why (Martin, 10/14):** commercial hands need their structure, roughly; scale does not matter. Result: 18 of 19 within target (Wuji v1 by 1.2 degrees, the grid's worst snapping error).
- **Item 21 (37 mm palm) -> 24 mm,** a flat bend servo in its housing; each base joint sits just deep enough that its first servo's housing is flush with the front face (12 mm below it for a bend first, 16.5 spread, 11.5 twist); deeper servos stick out behind. **Why (Martin, notes 2.1 and 2.2, and 10/14):** the thickness comes from the components; the base joint sits half a servo plus a wall from the face.
- **Item 22 (the wrist anchor, the heel disc, bases 10-160 mm, hinges 15-110 mm) -> the mount at the palm's centre, bases 40-90 mm (Feetech) or 10-160 mm (grammar), no hinge position.** **Why (Martin, notes 2.3, 2.4 and 2.6):** the base is the palm's centre; 10-160 mm was too wide (random palms twice the commercial size). The inner radius of one grid step keeps rows of five fingers.
- **Item 25 (fewer fingers when one cannot be placed; 3 mm allowed at the start pose) -> the drawn finger count is kept; a wall at both poses.** A finger is drawn again up to 50 times, then the whole hand with the same count, up to 20 times, then generation reports that no such hand exists. A finger's base, facing, tilt and first kind are drawn together from the placements whose servo fits. C1: every two solids that do not touch by construction keep 2 mm apart at the zero pose and at the start pose; at the zero pose a plate is also checked against the far end of a link hinged to it, so a link cannot lie in the palm. **Why (Martin, notes 2.11.3 and 2.12):** generate buildable hands directly, and never silently change the finger count.
- **The object start** is over the outline's centroid, at the lowest height where the object clears every solid by 5 mm at the start pose. **Why:** at the old fixed height a link overlapped the start cube in 68% of random hands.
- Effect, 1000 random hands under the Feetech rules: C1 100%, C2 81% (1.23 draws per usable hand, 0.05 s each); palms 171 cm^2 at the median (commercial 97 cm^2 under the same outline rule); a cut on 49% of hands; something sticks out behind the palm on 99.6% (median 21 mm). The object starts raised on 64% of hands (60 mm cube), and 12.5% of fingers weigh more than 0.5 N m can hold flat: both open (`GRAMMAR-OVERVIEW.md`, "Not settled yet").

## Revision of 2026-10-14 (afternoon): commercial hands built at scale; the Feetech's torque

Martin, 10/14, on the viewer: "scale commercial hands up minimally so that they can be built with feetechs / our feetech grammar without collisions"; "use feetech torque numbers"; SHARPA from KaRMA, without the iiwa arm.

- **Item 20 (fit target; commercial hands overlap-exempt) -> each commercial hand scaled up by the smallest factor that Feetech servos can build.** Scales are tried at 1, 1.25, ..., 2.5, 3, ..., 5, then in 0.05 steps below the first that works. A scale works when the fit keeps the servo link lengths, passes C1, keeps every joint's kind (or lies within 20 degrees of the commercial axis, for axes about 45 degrees between two kinds) and stays within twice the fit target (20 mm, 40 degrees). A finger that fails C1 is refitted with its clash counted: by local search, then in each of the 60 grid frames; the difference goes into its zero-pose offset. Errors count in millimetres at the real hand's size, so fits at different scales compare. A hand no scale up to 5 x makes buildable keeps its real-size fit and its collision filter. **Why:** Martin (above, and earlier: "we can be scale invariant"); the twice-target floor because moving fingers clear can leave a fit that is no longer the hand (Dex3 at its real size: 25 mm and 38 degrees, against 4 mm and 2 degrees at 2 x).
- **The grammar's own limits (No Rules) -> the scaled commercial hands' extremes x 1.1, rounded out to 10 mm**, so every commercial hand stays in the grammar. The Feetech rules are unchanged: links up to 90 mm, fingers up to 250 mm, bases 40-90 mm from the centre.
- **Item 14 (physics) -> the joint torque limit is the Feetech HL-3915-C001's stall torque, 1.39 N m** (14.2 kg cm at 12 V; it was 0.5 N m). The other values stay Kushal's. Its rated load is 0.44 N m and its no-load speed 10.5 rad/s (the speed limit stays 5 rad/s).
- **Item 19 (reference set): SHARPA is KaRMA's model of the hand alone** (`sharpa_right`, local-only, not committed), replacing SHARPA on the iiwa arm.
- Effect: 17 of 19 commercial hands are buildable, at 1-3.1 x (SHARPA 3.1, SVH and Wuji v2 right 3.05, Shadow 2.6, Wuji v1 2.55; DClaw at its real size); 9 fit within 10 mm / 20 degrees and all within twice that. ARMS and Dex1 are not buildable at any scale up to 5 x and keep their real-size fits and collision filters. The grammar's own limits become links up to 200 mm, fingers up to 500 mm, bases up to 370 mm from the centre (10^63 four-finger hands; the Feetech rules stay at 10^31). With 1.39 N m no random Feetech finger is too heavy to hold flat; SHARPA at 3.1 x needs 2.9 N m.

## Revision of 2026-10-16: finger length, tilt, fingertips, adding a finger, the mount, turned servos

- **Item 12 (finger length at most 250 mm) -> at most 225 mm under the Feetech rules.** **Why:** Martin, 10/16 ("Change it to 225"). Link lengths are on the 10 mm grid, so the longest finger is 220 mm. The grammar's own limit stays 500 mm, so the commercial fits are unaffected.
- Effect: five joints on the three kinds need at most 230 mm at their shortest (240 over all 31 axes), so the cap leaves out the five-joint fingers whose four links all need 50 mm: three bends in the middle, a bend or twist first, a bend or spread last (those ending in a bend and a spread still fit when the two meet at one point). Finger chains per base orientation drop from 249,941 to 89,795 (five-joint chains by 70%), and the search space from 10^31.3 to 10^29.5 four-finger hands (about 10^27.9 pass C1 and C2). Of 1000 random hands, 14% of fingers have five joints (21% before); the heaviest finger needs 0.69 N m flat (0.94 before); C2 is unchanged at 80%. Since a short link turns the bend servo after it (below), five joints on the three kinds need at most 200 mm and every five-joint finger fits again.
- **Tilt (-30 to +90 degrees; item 2 and the 10/07 re-measure) -> 0 to +90.** **Why:** Martin, 10/16: no finger points under the palm. The -30 bound was for Barrett's thumb at -25; the five commercial fits that used it (Allegro, ARMS, Shadow, Wuji v1, XHand) got it from the refits that move fingers clear, and do without it. A bend's -30 degree hyperextension stays (Martin: fine for the base bend to swing there).
- **A finger whose first joint is a bend has tilt 0.** **Why:** Martin, 10/16: the tilt turns the finger about its bend axis, which lies in the plate whatever the tilt, so a tilted bend-first finger is the same chain as an untilted one with its first bend's zero moved by the tilt. Random draws keep the 69/27/4 mix of first-joint kinds (each base placement weighted by its kind's share over the tilts it may take); commercial fits fold such a tilt into that joint's zero-pose offset (exact: the same chain).
- **The fingertip (item 6's servo-sized links, 10/14) -> a capsule** along the link, radius 16.5 mm (half the yoke's 33 mm width), from the yoke's back to the tip. **Why:** Martin, 10/16: a rounded fingertip, then a little thicker than the yoke's 24 mm depth. The simulator's collider and the viewer's mesh are a 50-point hull of it (inside PhysX's 64-vertex, 64-face GPU limit); the clearance check uses the exact capsule, and intersecting solids count as touching, as for boxes.
- **Adding a finger (was one 40 mm bend) -> as many joints as the hand's fingers have on average, rounded up, its joints and links drawn as for a random hand**, at a free spot pointing away from the centre, tilt 0. **Why:** Martin, 10/16.
- **The mount at the palm's centre (new; `Rules.mount_at_centre`, on for the Feetech rules and the commercial fits):** the centroid of the hull of the base servos' housings lies within half a base step of the mount (20 mm on the 40 mm grid, 5 mm on the 10 mm grid). **Why:** Martin, 10/16: no two hands may differ only in where the mount sits; 13% of random hands had such a twin (every base moved one grid step: the same palm and fingers). The hull leaves the mount out, so moving every base by d moves its centroid by exactly d and at most one placement passes. Random draws and mutations move a hand on the grid to centre it; a mutation whose centred child breaks a rule is not offered (a finger can no longer always be removed in one step: 78% of fingers can). Martin chose this over keeping only the twin nearest the centre, knowing it also removes off-centre palms (41% of the random hands drawn before it). It is a rule-set setting because it follows the rule set's base step; the grammar's own limits leave the mount free and so still contain every rule set's hands. The exact search-space count leaves it out; 42% of uniform four-finger samples keep it.
- **The servo's orientation (E15: always along the link) -> a bend servo turns 90 degrees about its axis, its body on the back of the finger, when the link before it is too short for it to lie along the link.** **Why:** Martin, 10/16: turned, only the servo's 20 mm width lies along the link (10 mm behind its axis against 24.5), so that link can be 30 mm instead of 40-50; by default it lies along the link as before. (Martin first asked for every bend servo after the base to turn, then for this.) The ranges are unchanged: at the -30 degree hyperextension a turned servo's back corner swings 20.9 mm forward (13.8 lying along the link), so the link after it needs 40 mm before another bend, 50 before a spread, and a fingertip after it 40. Servos at the base and after a 0 mm link lie along the finger.
- Effect of the day's changes together: 10^29.7 four-finger hands under the Feetech rules (exact count, before the centre rule and the clearance check), about 10^27.2 usable (42% of uniform samples centred, 0.27% also pass C1 and C2). Random hands: C2 73%, 1.37 draws per usable hand, 0.1 s each; the object starts raised on 43% of hands by 17 mm at the median (was 66% and 105 mm). Commercial hands: 17 of 19 buildable, 7 within 10 mm / 20 degrees; ARMS now buildable at 3 x, SHARPA not (at 3.1 x two fingers touch the palm plate at rest and a base servo reaches into the cut's seam; cause not confirmed). Isaac smoke: PASS.

## Still to decide
- After the grammar work: raise C5 (physics grasp test) again.
- The object start and the effort limit for servo-sized hands (above).

