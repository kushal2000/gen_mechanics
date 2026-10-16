# The hand grammar (October 2026)

**This is the current, up-to-date grammar**: branch `hand-grammar`, labelled with the tag `hand-grammar-2026-10-16` (moved from `martin/hand-grammar` on 2026-10-16).

Evolve to generalize trains one joint-token transformer across many hands and lets evolution choose which hands it trains on. The grammar defines the space of hands evolution searches. It has to do three things: express the commercial hands we train or test on (their structure, roughly in place), narrow down to hands evolution can search and we can build (the Feetech rules), and produce hands that all load into one simulator layout for one shared policy.

Since 2026-10-14 the grammar is built around one motor, the Feetech HL-3915 servo. The rebuild followed Martin's notes of 10/13, the Feetech study (`experiments/E15_feetech_buildability/summary.md`) and an independent review (`experiments/E16_grammar_rebuild/summary.md`). Every decision, with its reason, is in `GRAMMAR-LOCK-2026-10-07.md` (revision of 2026-10-14). It combines three lines of work: ours on `martin/hand-grammar`, Vatsal's physical grammar (the hull palm, the finger mount, joint kinds, building each finger clear of the others) and Kushal's uniform commercial hands (shared dynamics, the joint sign convention).

This document describes the code in `hand_sampler/grammar/` and is updated with every change to the grammar.

## Three layers

1. **The grammar:** what a hand can be.
2. **The rules:** limits that random generation and every mutation obey by construction, so no hand is generated and then thrown away for breaking a rule. Evolution and the simulator use the **Feetech rules**: hands buildable from Feetech servos. In the viewer: Feetech Rules, No Rules (the whole grammar) or Custom Rules.
3. **Viability checks:** the two properties of a whole hand that no rule can guarantee: its parts keep clear of each other (C1), and its fingertips can meet above the palm (C2).

## How a hand is built, step by step

### What a hand stores

Only these numbers, all whole: millimetres in steps of 10, degrees in steps of 30.
- **Per finger:** its base position (y, z) on the palm, its facing, its tilt, and for each joint: its type (hinge, coupled or sliding), its axis (two angles) and the length of the link after it.
- **Per palm cut (at most one):** its direction and its distance from the palm centre.

That is 4 numbers per finger, 3 per joint and 2 for a cut, plus each joint's type. Random Feetech hands have 14-98 numbers (mean 51); the commercial fits range from 14 (Dex1) to 85 (SHARPA, Shadow). Everything in steps 8-13 is computed from these numbers and the motor.

**The search space** (`hand_sampler/grammar/space.py`, counted exactly; `experiments/E16_grammar_rebuild/ours/search_space.py`). Hands the Feetech rules allow, before the centre rule (step 10) and the clearance check; a palm cut multiplies each by at most 73 more:

| Hands | Joints on the three kinds (where random hands start) | All 31 axis directions (what mutation reaches) |
|---|---|---|
| 4 fingers | 10^29.7 | 10^44.2 |
| 2-6 fingers | 10^43.7 | 10^65.4 |
| 4 fingers with the mount at the palm's centre that pass C1 and C2 (estimated from 20,000 uniform samples; 42% keep the centre rule) | 10^27.2 (0.27%) | 10^40.4 (0.015%) |

Before the rebuild the same count was about 10^50 four-finger hands (10^70 with all axes); the grammar's own limits (No Rules) allow 10^54 (10^110). Most of the count is five-joint fingers with long links, which collide more often; the random draw (step 5) picks joint counts uniformly instead.

### Building a random hand

**1. The origin and the mount.**
- The origin (0, 0) is the centre of the palm, where the hand mounts. The mount is a disc 40 mm across and 40 mm deep behind the palm's back face (an assumed size until the real mount is drawn). It attaches the hand to the arm, or to the world in our palm-up task.
- Axes: **x** points out of the palm on the grasp side; **y** and **z** lie in the palm. Nothing distinguishes "toward the fingers": fingers can go all round the centre.
- The palm is a flat plate 24 mm thick (step 10), centred on x = 0: its faces are at x = -12 and +12 mm.
- The mount disc is always part of the palm, so it is the smallest palm. It replaces the old 20 mm wrist (heel) disc, which had no physical reason; this one is where the hand is held.
- **Why (Martin, 10/13):** the hand mounts at its back, not at a wrist on one side, so its base is the middle of the palm.
- **The mount sits at the palm's centre** (Martin, 10/16): the centroid of the hull of the base servos' housings lies within half a base step of it (20 mm on the 40 mm grid; 5 mm for commercial fits on the 10 mm grid). Without this rule, moving every base by one grid step gave the same palm and fingers with the mount elsewhere: 13% of random hands had such a twin. Moving every base by d moves that centroid by exactly d, so at most one placement passes and no two hands differ only in where the mount sits. Random draws and mutations move a hand on the grid to centre it (`operators.recentre`); commercial fits move the palm frame back with it, so their errors are unchanged. The exact count of the search space leaves the rule out; 42% of uniformly drawn four-finger hands keep it.

**2. How many fingers:** uniform 2-6, and the hand keeps that count. If a finger cannot be placed (step 6), the whole hand starts again with the same count. Two because a single Wuji finger cannot learn the task (Kushal); six because the simulator has six finger slots.

**3. Where each finger's base goes.** Bases sit on a 40 mm grid, 40-90 mm from the centre: 20 spots, in rows of up to five.

```
   z
   ^     .   o   o   o   .        o  a spot: (y, z) in steps of 40 mm, 40-90 mm from the centre
   |     o   o   o   o   o           (rings at 40, 56.6, 80 and 89.4 mm)
   |     o   o   M   o   o        M  the mount, at the centre (not a spot)
   |     o   o   o   o   o        .  113 mm out: too far
   |     .   o   o   o   .
   +-----------------------> y    (y and z: -80, -40, 0, 40, 80 mm)
```

- A finger's base is drawn together with its facing, tilt and first joint's kind (steps 4 and 5): from all the combinations whose first motor clears the mount and the motors already placed by one printed wall (2 mm). It is drawn from what fits, not drawn and then rejected. Each combination is weighted so that first joints keep the kind mix (69/27/4) although a bend first has one tilt and the others four.
- There is no spacing rule and no rim rule. One rule (motors may not overlap the mount, each other or the palm; step 14) replaces the old 10 mm grid, the 19 mm spacing and the rim rule.
- **Why 40 mm:** with 35 mm, neighbouring first motors collide in 57% of the pairs of directions they can take; with 40 mm, 28%. No spacing short of 67 mm is safe for every pair, so the overlap rule is needed anyway, and 40 mm leaves it far more choices.
- **Why 40-90 mm:** an inner radius of one grid step keeps rows of five fingers (a 55 mm inner radius, which a larger mount would need, allows rows of three). 90 mm keeps the palm compact; the old 10-160 mm ring made palms twice the commercial size.

**4. Which way each finger points.**
- **Facing:** the direction the finger leaves its base, in the palm's plane, in 30 degree steps (0 points along +z, 90 along +y). Under the Feetech rules a finger faces outward: it makes at most 90 degrees with the line from the centre to its base, so it never points back across the palm. Tangential is allowed.
- **Tilt:** out of the plate, toward the grasp side: 0, 30, 60 or 90 degrees. +90 points straight up from the palm (DClaw). No finger points under the palm (Martin, 10/16; the range began at -30 for Barrett's thumb at -25, and the commercial fits now do without it). **A finger whose first joint is a bend has tilt 0** (Martin, 10/16): the tilt turns the finger about its bend axis, which lies in the plate whatever the tilt, so a tilted bend-first finger is the same chain as an untilted one with its first bend's zero moved by the tilt (only its base servo would sit differently). Commercial fits fold such a tilt into that joint's zero-pose offset.
- **Why outward (Martin, 10/13):** a finger pointing back across the palm runs over the other fingers' bases and the object's place.

**5. Each finger's joints and links,** from the base out.
- **Joints:** uniform 1-5. Joint j sits at the start of link j; the last link ends at the fingertip. Under the Feetech rules every joint is a hinge driven by its own servo.
- **Axis:** two angles in the link's frame (step 11). Each joint starts exactly on a kind, drawn with the commercial mix: bend 69%, spread 27%, twist 4%. Angled axes come only from later mutations; the grid has 31 axis directions.
- **Link lengths:** each link is drawn uniformly among the lengths it may take, given the joints at its two ends:

| Earlier joint, then next | bend | spread | twist |
|---|---|---|---|
| bend (lying along the link) | 30-90 (30-40 turn the next servo) | 50-90 | 40-90 |
| bend (turned) | 40-90 (40 turns the next servo) | 50-90 | 40-90 |
| spread | 30-90 (30 turns the next servo) | 40-90 | 30-90 |
| twist | 30-90 (30-40 turn the next servo) | 50-90 | 40-90 |

  The shortest length keeps the next servo's back clear of this servo's swinging front, plus a wall (E15's rule, rounded up to the 10 mm grid). A servo lies along the finger, its body behind its axis. **A bend servo turns only when the link before it is too short for that** (Martin, 10/16): turned 90 degrees about its axis, its body on the back of the finger, only its 20 mm width lies along the link (10 mm behind its axis, against 24.5), so that link can be 30 mm where it would need 40-50. The turned body then stands 24.5 mm out behind the finger (its link is 36.5 mm deep, not 24), and at the bend's -30 degree hyperextension its back corner swings 20.9 mm forward (13.8 for a servo lying along the link), so the link after it needs 40 mm before another bend and 50 before a spread, and a fingertip after it 40. A servo after a 0 mm link, and every servo at the base, lies along the finger. The ranges are unchanged. Over all 31 axes the shortest is 30-60 mm (60 only after a turned servo whose axis leans 30 degrees toward both spread and twist). The fingertip link is at least 30 mm, 40 after a turned bend (up to 50 for an oblique turned one): the last servo's swing, a wall and a 10 mm pad. Under the Feetech rules a whole finger is at most 225 mm (Martin, 10/16); five joints on the three kinds need at most 200 mm at their shortest (five bends 190: 30 + 40 + 40 + 40 + a 40 mm tip), so every five-joint finger fits.
- **Two joints at one point (a 0 mm link)** are allowed between two different axes square to the finger (a bend and a spread, or axes between them), never two in a row, never with a twist. The earlier servo then slides along its own axis, away from the joint, by 26.5 mm (square axes) to 34 mm (30 degrees apart; 41 mm when a spread crosses a turned bend): far enough that a plane square to its axis keeps the two servos a wall apart over the whole range. A spread crossing a bend at the base puts the spread servo 31 mm behind the palm. A 2-2-1 thumb (spread-bend-spread-bend-bend) is possible.
- **No angles between links:** at the zero pose a finger is a straight line and bends only through its joints.
- **Why:** Martin's notes 2.11 (hinges only, crossings anywhere, buildable by construction) and E15.

**6. The clearance test.** After each finger is drawn, the whole hand so far must pass C1 (step 14) at the zero pose and the start pose. If it fails, that finger's base, direction and joints are drawn again, up to 50 times; after that the whole hand starts again with the same finger count, up to 20 times, then generation reports that no such hand exists. **Why:** Vatsal's method (each finger clear of the others as it is placed), and Martin's note 2.12 (never silently fewer fingers).

**7. An optional palm cut.** On half the hands (when the rules allow a palm joint), the palm is cut along a straight line and the part beyond the line hinges on the rest, +/-30 degrees. The cut is drawn uniformly among the lines (30 degree directions, 10 mm offsets) that leave at least one finger on each side, keep the mount whole on the fixed part (at least 30 mm from the centre) and keep the hand clear. Every finger beyond the line rides on the moving part. **Why:** Martin's note 2.13 (palm joints decided on the finished palm; several fingers on one part, like a pinkie and a ring finger); a straight cut of a convex palm leaves two convex plates. If no cut fits, the hand is drawn again, so half the hands have one (48% of 1000).

### What is computed from the stored numbers

**8. The motor and how deep each base sits.** Every joint is the same servo: a 34 x 20 x 29 mm box, its axis along the 29 mm edges, 9.5 mm from one end and centred in the 20 mm width (`motor.py`). It lies along the finger, its body behind its axis, except a bend servo whose link before it is too short for that, which is turned with its body on the back of the finger (step 5). Its body sits in the link before its joint (the palm, for a finger's first joint); the link after it holds the horn and the idler in a yoke. A finger's base joint sits below the palm's front face, just deep enough that the housing of its first servo (and of the second, when the two share the point) is flush with the face: 12 mm for a bend first, 16.5 for a spread, 11.5 for a twist (flat fingers). **Why (Martin, 10/14):** the base joint sits half a servo plus a wall from the face, so the fingers start as close to the working side of the palm as the servo allows.

**9. Solids.** Every solid the clearance check and the simulator see:
- each link but the last: one box, square to its frame, around its yoke and the servo it carries. A bending link is 33 mm wide and 24 mm deep (36.5 mm when the servo it carries is turned) and runs from 12 mm behind its joint to the next servo's front: links are servo-sized;
- the fingertip (each finger's last link): a capsule along the link, radius 16.5 mm (half the yoke's 33 mm width along the last servo's axis, so the fingertip is as wide as the bracket around that servo), from the yoke's back (12 mm behind its joint) to the fingertip. **Why (Martin, 10/16):** a rounded fingertip, a little thicker than the yoke's 24 mm depth. The simulator's collider and the viewer's mesh are its 50-point convex hull (within 1 mm of the capsule, inside PhysX's 64-vertex, 64-face GPU limit); the clearance check uses the exact capsule;
- each finger's base servo, fixed to its plate;
- the main plate, the part beyond a cut, the mount, and the cut's hinge servo.

Mass comes from a density of 1750 kg/m^3 over each box and capsule: a finger weighs 344 g at the median (up to 862 g).

**10. Palm plates.** The palm's outline is the convex hull of the mount disc, every base servo's housing seen from above, and the hinge servo's housing, at most 32 corners. Every servo sits inside the palm by construction; there is no other palm parameter. The plate is 24 mm thick: a flat bend servo in its housing. Servos that stand deeper stick out behind: a spread servo 9 mm, a twist servo 14 mm, a crossed spread 31 mm; in 73% of random hands something sticks out behind (median 14 mm, at most 43). Behind the palm is free in our palm-up task.

A cut splits the outline into two convex plates, each stopping 4.3 mm short of the line: an 8.6 mm seam, because a 24 mm plate cannot swing +/-30 degrees through a zero-width cut. The hinge axis runs along the line in the plate's mid-plane. Its servo sits on the main palm's side at the 10 mm spot along the line nearest the middle of the chord that clears every base servo. Random palms measure 177 cm^2 at the median (commercial hands 97 cm^2 at their real size, 196 cm^2 at the scale Feetech servos build them, under the same outline rule).

**11. Each finger's frame,** at the zero pose: **x** along the finger (from facing and tilt); **y**, the closing direction, the palm's normal turned by the tilt; **z** = x cross y, the bend axis, in the plate. The frame depends on the finger alone. Joint axes are written in it; links are straight, so every link of a finger shares it at rest. **Why (Martin, 10/13):** + on every bend then moves the fingertip toward the palm's working face, the same way on every finger. (From 10/07 to 10/13 the closing direction pointed at the object's start point; it made a finger's frame depend on the other fingers.)

**12. Each joint's kind, sign and range.** The kind is whichever of the frame's z, y or x the axis lies closest to; it is a label, not a stored choice.

| Kind | Axis near | + means | Range |
|---|---|---|---|
| bend | z | the fingertip moves toward the closing direction (the palm's working side) | -30 to +90 deg |
| spread | y | the fingertip moves toward +y of the palm (if it cannot, +z, then +x) | +/-30 deg |
| twist | x | a right-handed turn about the link | +/-90 deg |
| palm joint | the cut's line | the part beyond the cut rises toward the grasp side | +/-30 deg |
| sliding | the axis | along the axis | -20 to +25 mm |

- A coupled joint (No Rules only) keeps its own kind and sign, follows the joint before it at 1.1 times its angle, and takes 1.1 times that joint's range.
- **Why:** computed signs make + mean the same on every hand (Kushal's convention); one range per kind; 1.1 is the typical commercial coupling.

**13. Poses and the object start.** At the zero pose every joint is at 0 and every finger is straight. At the start pose, where episodes begin, every independent bend joint is at 0.35 of its upper limit (31.5 degrees), every other joint at 0. The object starts over the centroid of the palm's outline, at the lowest height (from 5 mm above the front face, in 1 mm steps) where it clears every solid by 5 mm at the start pose. It never starts inside the hand. On most random hands the curled fingers stand over the palm's centre, so the object starts high (step 18, "Not settled yet").

### Checking, changing and simulating

**14. Viability checks.**
- **C1, clearance:** every two solids that do not touch by construction keep one printed wall (2 mm) apart, at the zero pose and at the start pose (exact box distances). Touching by construction: the main plate with the mount, the hinge servo and the part beyond the cut; a plate with its base servos (they sit in pockets); within a finger, the two sides of each joint (and, across a 0 mm link, the links on either side). One exception keeps a link from lying in the palm: at the zero pose a plate is checked against the far end of a link hinged to it. A base servo's housing must also stay out of a cut's seam. Collisions while the fingers move are left to the physics and the policy (Martin, 10/14). Generation keeps C1 (step 6).
- **C2, fingertips can meet:** each fingertip is sampled 1,500 times over its own joints' ranges (the palm joint included, one hinge angle shared by every finger beyond the cut); a sample counts if it is above the plate and over the palm (the outline grown by 20 mm); a hand passes if some pair of fingers' samples come within 20 mm of each other. Every commercial hand passes.
- Of random Feetech hands, 73% pass both (C1 100%, C2 73%): 1.37 draws per usable hand, 0.1 s each (1000 hands, re-measured 2026-10-16 after the day's rule changes).
- A physics grasp test (C5) is deferred until the grammar work is done.

**15. Rule sets.** The grammar's own limits (No Rules) allow hinge, coupled and sliding joints, links of 0 or 10-200 mm, fingers up to 500 mm, bases anywhere on a 10 mm grid 10-370 mm from the centre facing any way. These limits are the commercial hands' extremes at the scale Feetech servos build them (step 18) x 1.1, rounded out: the longest link (SHARPA at 3.1 x, 180 mm), the longest finger (SHARPA, 450 mm) and the farthest base (SVH at 3.05 x, 329 mm), set on 10/14. Since the rebuild of 10/16 the scaled hands reach less far (140 mm, 370 mm, 161 mm, SHARPA no longer scaled), so these limits are wider than the commercial hands need; kept for now (open). The **Feetech rules** (`FEETECH_RULES`, also `EVOLUTION_RULES`) are what evolution and the simulator use: hinges only; bases on the 40 mm grid, 40-90 mm out, facing outward; links at least the servo floors (step 5). Custom Rules can only tighten the grammar: joint types, maximum fingers (6), joints per finger (5), palm joints (1), finger length (Feetech 225 mm), link length (Feetech 90 mm), base grid and distance, outward facing, servo link lengths. The search space: 10^29.7 four-finger hands under the Feetech rules (joints on the three kinds, before the clearance check), 10^62 under the grammar's own limits. Random hands and every mutation keep the rules by construction.

**16. Mutation.** One grid: every value moves 10 mm or 30 degrees, a base by one grid step (40 mm under the Feetech rules). The fine stage is future work.
- **Parameter operators:** move a finger's base (y or z); turn its facing; tilt it; change a link to its next allowed length (from its shortest down to 0 where two joints may share a point); turn or tip a joint's axis (links whose shortest length changes move to it); move the cut 10 mm; turn the cut 30 degrees.
- **Structural operators:** add a finger at a free spot, pointing away from the centre, tilt 0, with as many joints as the hand's fingers have on average (rounded up; Martin, 10/16), its joints and links drawn as for a random hand; remove a finger (a cut left with no finger on a side goes too); split a link to add a joint (at any 10 mm point on it; links too short for their new joints grow to their shortest); merge two links to remove a joint (at most 90 mm); couple or uncouple a joint (No Rules only); cut the palm; remove the cut.
- **The mount stays at the palm's centre:** a child whose palm centre moves more than half a base step from the mount is moved on the grid to centre it (`operators.recentre`); if that puts a base outside the ring, that option is not offered. So a hand can lose any of 78% of its fingers in one step (every hand can lose some finger), where before it could lose any.
- **One mutation:** an operator chosen at random among those that can act; its options shuffled (a split's new joint kind weighted by the commercial mix); the first child that keeps the rules and C1 is returned (Vatsal's method). From a usable hand, 94% of calls return a child and 96% of children are usable.

**17. In the simulator** (Isaac Lab, `isaacsimenvs/inhand_reorient/scene/`).
- The hand is fixed palm-up with no arm: the palm's +x is turned to world up.
- Physics is the same for every hand, generated or commercial: each joint's torque limit is the Feetech HL-3915's stall torque, 1.39 N m (14.2 kg cm at 12 V; Martin, 10/14; it was 0.5 N m); the rest are Kushal's values: 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, density 1750 kg/m^3, friction 0.5. The servo's rated load is 0.44 N m and its no-load speed 10.5 rad/s.
- Every hand fits one layout of 36 joint slots: 6 finger slots, each one palm-joint slot plus 5 finger joints, with a fingertip body. Unused slots are tiny, have no collider and are locked.
- The palm joint and couplings are PhysX mimic joints: a second finger beyond the cut follows with gearing -1, a coupled joint with -1.1. The policy does not control coupled joints.
- Colliders: each link is its servo-sized box; the root body carries the main plate, the mount, the hinge servo and the main palm's base servos; the part beyond the cut carries its plate and its base servos. Solids rest one wall apart (rest offset 1 mm each, contact offset 3 mm). Plates are convex hulls of up to 64 corners.
- Admission: grammar hands must pass C1. Commercial hands are fitted at the scale where they pass it (step 18); the two that no scale makes buildable (ARMS, Dex1) have their overlapping pairs filtered from collision instead.
- The object's size comes from the task (72 mm in the repose task); it sets the start height (step 13).
- Isaac check of the rebuilt layout (2026-10-14, RTX 4090, the validation population: five commercial hands, a hand with two fingers beyond a cut, crossed knuckles, oblique axes, three random hands; 64 envs, `population_diagnostics.py --task-profile hora --random-steps 300`): PASS. The authored bodies match the grammar's kinematics to 0.0005 mm at reset and 0.014 mm after 20 random steps; joint limits exact; 300 random steps without NaN (largest joint speed 5.09 rad/s, the 5 rad/s limit); mimic ties within 1.1e-3 rad; no collider-cooking warning; every cube still resting after 100 zero-action steps. Again after the commercial hands were scaled and the torque limit raised to 1.39 N m (the same day, the validation population and the default one with SHARPA at 3.1 x and Wuji v1 at 2.55 x): PASS both; 0.0005 mm at reset, at most 0.26 mm after 20 random steps; no NaN; joint speeds up to 6.5 rad/s against the 5 rad/s limit; no contact between collision-filtered pairs; every cube resting. The 2026-10-07 validation is `isaac-validation-2026-10-07.md`.

**18. Commercial hands.** Eighteen hands (19 fits, counting Wuji v2's left and right): Ability, Allegro, ARMS (a human skeleton model), Barrett, DClaw, Dex1, Dex3, Inspire, LEAP, MIDAS, Orca, Shadow, SHARPA (KaRMA's model of the hand alone), SVH, Tesollo DG-5F, Wuji v1, Wuji v2 and XHand. Each is scaled up by the smallest factor at which Feetech servos can build it, then fitted onto the grid with its own mount at its wrist (`hand_sampler/grammar_bench/conformed_hands.json`, rebuilt with `python3 -m hand_sampler.grammar.build_conformed`, about an hour on 19 processes). Martin, 10/14: "scale commercial hands up minimally so that they can be built with feetechs / our feetech grammar without collisions"; only the DOF structure has to match, and the hand may be bigger.
- **The scale:** tried at 1, 1.25, ..., 2.5, 3, ..., 5, then in 0.05 steps below the first that works. A scale works when the fit keeps the servo link lengths, passes C1, keeps every joint's kind (or lies within 20 degrees of the commercial axis, for an axis about 45 degrees between two kinds, as on SHARPA's thumb) and stays within twice the fit target (20 mm, 40 degrees). The last condition exists because moving fingers clear can leave a fit that is no longer the hand: Dex3 passes C1 at its real size, but at 25 mm and 38 degrees, against 7 mm at 1.6 x.
- **Moving fingers clear:** a finger that fails C1 is refitted with its clash counted, first by local search, then in each of the 48 grid frames (12 facings x 4 tilts), the best frames refined in turn; the difference goes into its zero-pose offset. A thumb lying along the palm comes out standing off it.
- **Errors** are millimetres at the real hand's size (20 mm on a hand at 2 x counts 10 mm), so fits at different scales compare. The target is 10 mm / 20 degrees.
- **Result** (`experiments/E16_grammar_rebuild/ours/commercial_scaled.py`; rebuilt 2026-10-16 with the tilt rules, capsule fingertips, the mount at the palm's centre and turned servos): 17 of 19 buildable; 7 within the target and every fit within twice it. Each keeps its finger count, joints per finger, joint types and kinds; SHARPA's, Shadow's and SVH's palm joints each become a cut with the same fingers beyond it, ARMS's two one cut carrying both.

| Hand | Scale | Joint (mm) | Axis (deg) | Tip (mm) | Heaviest finger flat (N m) |
|---|---|---|---|---|---|
| Ability | 1.8 | 4.2 | 13.1 | 3.6 | 0.21 |
| Allegro | 1.3 | 17.8 | 15.5 | 5.8 | 0.40 |
| ARMS | 3 | 12.5 | 25.2 | 8.6 | 0.33 |
| Barrett | 1.45 | 3.7 | 3.6 | 1.2 | 0.24 |
| DClaw | 1 | 3.9 | 2.4 | 2.0 | 0.00 |
| Dex1 | 1 (not buildable) | 9.6 | 15.7 | 8.4 | 0.01 |
| Dex3 | 1.3 | 9.3 | 1.8 | 2.9 | 0.11 |
| Inspire | 1.8 | 13.1 | 14.1 | 2.6 | 0.37 |
| LEAP | 1.5 | 16.2 | 32.6 | 12.4 | 0.65 |
| MIDAS | 1.55 | 9.8 | 20.3 | 8.6 | 0.56 |
| Orca | 2.15 | 7.6 | 24.6 | 6.0 | 0.30 |
| Shadow | 2.6 | 7.6 | 15.6 | 1.2 | 0.08 |
| SHARPA | 1 (not buildable) | 7.6 | 12.2 | 3.2 | 0.31 |
| SVH | 2.35 | 10.5 | 26.1 | 10.6 | 0.29 |
| Tesollo DG-5F | 1.65 | 13.0 | 22.5 | 3.5 | 0.28 |
| Wuji v1 | 2.25 | 19.9 | 30.9 | 1.7 | 0.70 |
| Wuji v2 left | 2.15 | 9.9 | 21.0 | 8.4 | 0.81 |
| Wuji v2 right | 2.55 | 8.3 | 28.0 | 7.8 | 1.04 |
| XHand | 2 | 16.2 | 15.2 | 2.1 | 0.75 |

- **Not buildable up to 5 x:** SHARPA and Dex1. SHARPA at 3.1 x (buildable on 10/14) fits closely (4.7 mm, 9.2 degrees) but fails C1: two of its fingers touch the palm plate at rest and a base servo reaches 6.3 mm into the cut's seam, and the refit cannot clear them; at 4 x its thumb is 550 mm, past the grammar's own 500 mm. The likely cause is the mount moving to the palm's centre (it sat beside the thumb, 270 mm from the fingers), which moves the cut up to 5 mm relative to the fingers; not yet confirmed. Dex1 is a gripper whose two sliding jaws face each other; pulling them clear spoils the fingertips at any scale. Both keep their real-size fit under the grammar's own limits and their overlaps collision-filtered in the simulator. ARMS, not buildable on 10/14, is buildable at 3 x.
- **Torque:** no commercial hand at its scale has a finger heavier than the 1.39 N m limit can hold flat, palm up (the heaviest: Wuji v2 right at 2.55 x, 1.04 N m).
- Scaled, commercial palms have a median area of 174 cm^2 under the outline rule; random Feetech palms 162 cm^2.

## The grammar in one table

"Per hand" means it can differ between hands (evolution can change it); "global" means one value for every hand. The Feetech rules apply on top of the grammar.

| Part | Parameter | Per hand or global | Grammar (No Rules) | Feetech rules | Step | Source |
|---|---|---|---|---|---|---|
| Hand | fingers | per hand | 2-6 | same | add / remove a finger | max: simulator layout; min: one finger cannot learn (Kushal) |
| Hand | joints per finger | per finger | 1-5 | same | split / merge a link | commercial max is 5 |
| Hand | palm joint | per hand | none or one straight cut, a finger on each side | same | cut / remove; move 10 mm; turn 30 deg | Martin 10/13 |
| Palm | origin and mount | global | the palm centre; a 40 mm disc, 40 mm deep, behind the plate | same | - | Martin 10/13; size assumed |
| Palm | outline | derived | convex hull of the mount and the base and hinge servos' housings | same | - | Vatsal's hull |
| Palm | thickness | global | 24 mm | same | - | a flat bend servo in its housing |
| Palm | cut seam, hinge | derived | 8.6 mm seam; hinge along the cut in the mid-plane, its servo on the main side | same | - | a 24 mm plate swinging +/-30 deg |
| Finger | base position (y, z) | per finger | 10 mm grid, 10-370 mm from the centre | 40 mm grid, 40-90 mm | one grid step | Martin 10/13; E16 layouts; scaled commercial extremes x 1.1 |
| Finger | facing | per finger | any (30 deg steps) | outward | 30 deg | Martin 10/13 |
| Finger | tilt | per finger | 0 to +90 deg; 0 when the first joint is a bend | same | 30 deg | never under the palm; a first bend already tilts the finger (Martin 10/16); DClaw's +90 |
| Finger | base depth | derived | the first servo's housing flush with the front face | same | - | Martin 10/14 |
| Finger | finger length | rule | <= 500 mm | <= 225 mm | - | longest scaled commercial finger x 1.1; Martin 10/16 |
| Joint | axis | per joint | 31 directions in the link's frame | same | 30 deg | |
| Joint | kind | derived from the axis | bend / spread / twist | same | - | Vatsal |
| Joint | sign | global convention | + closes toward the working side; + spreads toward +y; 0 = home | same | - | Kushal; Martin 10/13 |
| Joint | range | global per kind | bend -30 to +90; spread +/-30; twist +/-90 deg | same | - | Martin's 9/23 note |
| Joint | type | per joint | hinge, coupled, sliding | hinge only | couple / uncouple | E15: one servo per joint |
| Joint | random draw | global | starts on a kind: 69% bend, 27% spread, 4% twist | same | - | commercial mix |
| Link | shape | derived | one box around its yoke and the servo it carries | same | - | Martin 10/13 (servo-sized) |
| Link | length | per link | 0 or 10-200 mm; fingertip 10-200 mm | the servo floors (30-50 mm; a link under 40-50 mm turns the bend servo after it) to 90, fingertip >= 30 (40 after a turned bend), 0 only for crossings | 10 mm | E15; Martin 10/16 (turned servos); scaled commercial extremes x 1.1 |
| Physics | torque, speed, gains, density, friction | global | 1.39 N m (the Feetech's stall torque), 5 rad/s, 3, 0.078, 0.00058, 1750 kg/m^3, 0.5 | same | - | Feetech datasheet (torque); Kushal (the rest) |
| Check | C1 clearance | - | solids a wall (2 mm) apart at the zero and start poses | same | - | |
| Check | C2 fingertips meet | - | some fingertip pair's reachable regions meet above the palm | same | - | Martin; every commercial hand passes |

## Where each part came from

| | Ours | Vatsal | Kushal | E15 / E16 (Feetech, Codex) |
|---|---|---|---|---|
| Palm | the cut, the mount at the centre | the hull plate | | the 24 mm plate, the seam, the hinge servo |
| Fingers and mount | the 40 mm grid, outward facing | (y, z) plus facing | | the base depth, the overlap rule |
| Joints | free axes, coupling, the tilt-defined frame | bend / spread / twist kinds | the sign convention | hinges only, crossings |
| Links | 0 mm links | | | servo floors, servo-sized links |
| Generation | uniform draws, keeping the finger count | building each finger clear of the others | | |
| Physics | | | the uniform values | |
| Commercial hands | the fitting and our 15 hands | MIDAS | Dex3, Wuji v2 | |

## Summary tables

### 1. What varies from hand to hand (what evolution mutates), under the Feetech rules

| Parameter | Per | Allowed values | Random draw | Step |
|---|---|---|---|---|
| Number of fingers | hand | 2-6 | uniform 2-6, kept | add / remove a finger |
| Joints per finger | finger | 1-5 | uniform 1-5 | split / merge a link |
| Base position (y, z) | finger | the 20 spots of the 40 mm grid 40-90 mm from the centre | with facing, tilt and first kind, among the placements whose servo fits | 40 mm |
| Facing | finger | outward, 30 deg steps | uniform among outward facings | 30 deg |
| Tilt | finger | 0, 30, 60, 90 deg; 0 when the first joint is a bend | uniform | 30 deg |
| Joint axis | joint | 31 directions | exactly on a kind: 69% bend, 27% spread, 4% twist | 30 deg |
| Link length | joint | the servo floor (30-50 mm; a short link turns the bend servo after it) to 90 mm; 0 for a crossing; fingertip 30-90 mm (40 after a turned bend); finger at most 225 mm | uniform among the allowed lengths | 10 mm (the floor and 0 are neighbours) |
| Palm cut | hand | none, or a line 30-80 mm from the centre (30 deg directions) with a finger on each side | on half the hands, uniform among the cuts that fit | move 10 mm; turn 30 deg |

### 2. What is the same for every hand (fixed, not mutated)

| Constant | Value | Where it comes from |
|---|---|---|
| Motor | Feetech HL-3915: 34 x 20 x 29 mm, axis 9.5 mm from one end; one printed wall (2 mm) between moving parts | Martin; E15 |
| Mount | a 40 mm disc, 40 mm deep, behind the palm's centre | Martin 10/13; size assumed |
| Palm thickness | 24 mm | a flat bend servo in its housing |
| Palm outline | convex hull of the mount and the servos' housings | Vatsal's hull |
| Link shape | one box around each link's yoke and the servo it carries; straight at rest | Martin 10/13 |
| Joint ranges | bend -30 to +90; spread +/-30; twist +/-90; palm joint +/-30 deg | Martin's 9/23 note |
| Joint sign | + closes toward the working side, or spreads toward +y; 0 = home | Kushal; Martin 10/13 |
| Physics | 1.39 N m (Feetech stall torque), 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, density 1750 kg/m^3, friction 0.5 | Feetech; Kushal |
| Object start | over the outline's centroid, the lowest height clear of the hand by 5 mm at the start pose | the task |

### 3. Limits: the whole grammar against the Feetech rules

| Limit | Grammar (No Rules) | Feetech rules |
|---|---|---|
| Fingers | 2-6 | 2-6 |
| Joints per finger | 1-5 | 1-5 |
| Palm joints | at most one cut | same |
| Joint types | hinge, coupled, sliding | hinge |
| Base grid | 10 mm, 10-370 mm from the centre | 40 mm, 40-90 mm |
| Facing | any | outward |
| Link length | 0 or 10-200 mm; fingertip 10-200 mm | servo floors (30-50 mm; a short link turns the bend servo after it) to 90; 0 only for crossings; fingertip 30-90 (40 after a turned bend) |
| Finger length | at most 500 mm | at most 225 mm |

### 4. Removed from the grammar to keep it simple

| Removed | What replaced it |
|---|---|
| The fine stage (1 mm / 5 deg) | one grid (10 mm / 30 deg); refinement is future work |
| The wrist origin and the 20 mm heel disc | the mount at the palm's centre |
| The 10 mm base grid, the 19 mm spacing, the rim rule | the 40 mm grid and one overlap rule |
| The 10-160 mm base ring | 40-90 mm |
| Closing toward the grasp point | the tilt-defined frame |
| The 37 mm palm | 24 mm, from the servo |
| One 19 x 18 mm rounded-box cross-section | servo-sized links |
| Free palm hinges (position, axis), several palm joints | at most one straight cut |
| Commercial frequencies (0 mm links 27/220, coupling 18/180, palm joints 4/18) | uniform draws within the rules |
| Falling back to fewer fingers | the same finger count, drawn again |
| Earlier: per-hand radii, per-joint ranges, rest bends, palm length, per-joint coupling, continuous joints, branching fingers, named variants, a hand-shaped template | see the lock |

### 5. Structural mutations

| Mutation | Effect |
|---|---|
| Add a finger | a finger at a free spot, pointing away from the centre, tilt 0, with as many joints as the hand's fingers have on average (rounded up; Martin, 10/16), its joints and links drawn as for a random hand |
| Remove a finger | a finger in one step, unless centring the rest puts a base outside the ring (78% of fingers can go); a cut left with no finger on a side goes too |
| Split a link | adds a joint at any 10 mm point on it; links grow to their shortest if needed |
| Merge two links | removes a joint; the merged link is at most 90 mm |
| Couple / uncouple a joint | No Rules only |
| Cut the palm / remove the cut | a straight cut with a finger on each side, or none |

### 6. Viability checks

| Check | Passes when | Why |
|---|---|---|
| C1, clearance | every two solids a wall apart at the zero and start poses (except parts that touch by construction); no base servo in a cut's seam | the parts must fit; deep starting overlaps make the physics blow up |
| C2, fingertips can meet | at least one pair of fingertips can reach a common region above the palm | a hand whose fingertips never meet cannot hold or turn an object |

## Not settled yet

- **The object start.** At the start curl, tilted fingers on the inner ring stand over the palm's centre, so the object starts raised on 43% of random hands (60 mm cube, by 17 mm at the median; 60% for the repose task's 72 mm cube); the worst start 209 mm up, near the 0.24 m drop termination. Since 10/16 (tilts 0-90 degrees, tilt 0 for a bend first) far fewer fingers stand over the palm: it was 66% and 105 mm. Even with straight fingers, 49% of hands raise the 72 mm cube. A smaller object, a lower start curl or another spawn point are task decisions.
- **Torque.** The torque limit is now the HL-3915's stall torque, 1.39 N m: no random Feetech finger is too heavy to hold flat (the heaviest needs 0.71 N m; 8% need more than the old 0.5 N m), nor any commercial hand at its scale (at most 1.04 N m, Wuji v2 right). The servo's rated load is 0.44 N m, so stall is an upper bound.
- **The mount's real size** (assumed 40 mm across, 40 deep).
- **Palm cuts between neighbouring bend-first fingers** are impossible: their housings are 7 mm apart, the seam needs 8.6.
- **Coupled joints and the policy:** they are tied and not controlled; whether they appear as tokens is open.
- **A physics grasp check (C5)** comes up again once the grammar work is done.

Status: the current grammar is on the branch `hand-grammar` (until 2026-10-16 it was built on `martin/hand-grammar`); the grammar viewer is `experiments/grammar_viewer/`.
