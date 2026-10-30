# The hand grammar (October 2026)

**This is the current, up-to-date grammar**: branch `hand-grammar`, latest label the tag `hand-grammar-2026-10-20` (moved from `martin/hand-grammar` on 2026-10-16). Since 2026-10-23 every grammar hand can also be generated as printed, screw-assembled parts (step 19); that work and the grammar changes it needed are on branch `hand-build`, not yet merged.

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

**The search space** (`hand_sampler/grammar/space.py`, counted exactly; `experiments/E16_grammar_rebuild/ours/search_space.py`). Hands of 2-6 fingers each stage allows (step 16), before the centre rule (step 10) and the clearance check; a palm cut multiplies each by at most 73 more. Search-space numbers always count every finger count (Martin, 10/20):

| Stage | Hands (2-6 fingers) | Turns and mirror images counted once (step 16) | Usable: mount at the centre, pass C1 and C2 (3,000 uniform samples) |
|---|---|---|---|
| Coarse: the three kinds, links in 20 mm steps (where random hands start) | 10^38.0 | about 10^37.1 | about 10^35 (measured 10/20 at 10^38.3: 65% centred; 2 of 3,000 pass C1 and C2) |
| Fine: 31 axis directions, 10 mm steps | 10^64.8 | about 10^63.9 | below 10^62 (measured 10/20: 67% centred; none of 3,000 pass C1) |
| The grammar's own limits (No Rules) | 10^123 | | |

On 10/23 the counts fell from 10^38.3 and 10^65.4: a crossing's first servo now always lies along the finger, and the link before a crossing must fit the servo after it (step 5).

Six-finger hands are about 99% of each total. Earlier counts were of four-finger hands only: 10^29.7 on the three kinds and 10^44.2 with all axes (10/16), about 10^50 before the rebuild. Most of the count is five-joint fingers with long links, which collide more often; the random draw (step 5) picks joint counts uniformly instead. Ways to cut the space further are listed, none adopted, in `search-space-options.md`.

### Building a random hand

**1. The origin and the mount.**
- The origin (0, 0) is the centre of the palm, where the hand mounts. The mount is a column 40 mm across and 40 mm deep behind the palm's back face. It attaches the hand to the arm, or to the world in our palm-up task. Its far end is the **board bay** (Martin, 10/23: the Palm Hub board behind the palm): a box 76 x 60 mm and 22 mm deep, whose floor bolts straight to the arm's flange. The bay is turned in 15 degree steps and shifted up to 15 mm per hand so that it keeps a wall from every finger's base links over their whole ranges (`derive.hand_geometry`); a hand where no turn and shift does fails C1.
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
| bend (lying along the link) | 30-90 (30 turns the next servo, 40 flips or turns it) | 50-90 | 40-90 |
| bend (turned) | 40-90 (40 flips or turns the next servo) | 50-90 | 40-90 |
| spread | 30-90 (30 turns the next servo, 40 flips or turns it) | 41, 50-90 | 31, 40-90 |
| twist | 30-90 (30 turns the next servo, 40 flips or turns it) | 50-90 | 40-90 |

  The shortest length keeps the next servo's back clear of this servo's swinging front, plus a wall (E15's rule), rounded up to the 10 mm grid, or to 1 mm past it where the grid would leave less than 1 mm between the swinging front and the printed wall (Martin, 10/23: raise those by the least that works, not a whole step; after a spread, 39.7 becomes 41 rather than 40). Longer links stay on the grid (41, then 50, 60, ...). A servo lies along the finger, its body behind its axis. **A bend servo turns only when the link before it is too short for that** (Martin, 10/16): turned 90 degrees about its axis, its body on the back of the finger, only its 20 mm width lies along the link (10 mm behind its axis, against 24.5), so that link can be 30 mm where it would need 40-50. The turned body then stands 24.5 mm out behind the finger (its link is 36.5 mm deep, not 24), and at the bend's -30 degree hyperextension its back corner swings 20.9 mm forward (13.8 for a servo lying along the link), so the link after it needs 40 mm before another bend and 50 before a spread, and a fingertip after it 40. **Where the links on both sides have room, such a servo is flipped instead of turned** (Martin, 10/29: turned, it stands 24.5 mm out on the finger's back; "these motors could have been rotated 90 degrees to allow the finger to be thinner ... the output of the MCP DOF connects to the output of the PIP DOF"): it lies along the link after its joint, its body in that link with its back toward the fingertip, and the link before holds its horns in a yoke round its front, which swings back 13.8 mm as a lying servo's swings forward. That link then carries two yokes back to back and joins two servos' outputs: 31 mm after a bend or a spread, 40 after a twist or a turned bend. The link after holds the flipped body: 40 mm before a turned bend, 50 before another flipped one, 60 before a lying one, and a fingertip 40 (`motor.pose_floor_mm`). Of the ways a finger's bend servos can sit with room in every link, the one with the fewest turned, then the fewest flipped, then with the turned ones nearest the palm (`hand.servo_poses`). Turning still sets the floors (it fits wherever anything does), so the lengths a finger may take, the operators and the search space are unchanged; but a 30 mm link, the grid's shortest, is 1 mm short of two yokes, so a servo after one still turns. In 400 random hands under the fine rules, 38% of the servos that turned now flip and 29% of the fingers with a turned servo have none left; under the coarse rules (20 mm steps), 17% and none. A servo after a 0 mm link, and every servo at the base, lies along the finger. The ranges are unchanged. Over all 31 axes the shortest is 30-60 mm (60 only after a turned servo whose axis leans 30 degrees toward both spread and twist). The fingertip link is at least 30 mm, 40 after a turned bend (up to 50 for an oblique turned one): the last servo's swing, a wall and a 10 mm pad. Under the Feetech rules a whole finger is at most 225 mm (Martin, 10/16); five joints on the three kinds need at most 203 mm at their shortest (a bend then four spreads; five bends 190: 30 + 40 + 40 + 40 + a 40 mm tip), so every five-joint finger fits.
- **Two joints at one point (a 0 mm link)** are allowed between two different axes square to the finger (a bend and a spread, or axes between them), never two in a row, never with a twist. The earlier servo then slides along its own axis, away from the joint, by 28 mm (square axes) to 35.6 mm (30 degrees apart): far enough that a plane square to its axis keeps the two servos a wall apart over the whole range, plus 1.5 mm for the horn's centre screw head between them (Martin, 10/23). A spread crossing a bend at the base puts the spread servo 32.5 mm behind the palm. Both servos of a crossing lie along the finger (the first is never turned), and the link before a crossing must also fit the crossing's second servo, which sits in the next part: 45 mm after a bend, 44 after a spread, 52 after a turned bend (`motor.crossing_floor_mm`; found building the parts, 10/23). A 2-2-1 thumb (spread-bend-spread-bend-bend) is possible.
- **No angles between links:** at the zero pose a finger is a straight line and bends only through its joints.
- **Why:** Martin's notes 2.11 (hinges only, crossings anywhere, buildable by construction) and E15.

**6. The clearance test.** After each finger is drawn, the whole hand so far must pass C1 (step 14) at the zero pose (also the start pose since the fingers start open). If it fails, that finger's base, direction and joints are drawn again, up to 50 times; after that the whole hand starts again with the same finger count, up to 20 times, then generation reports that no such hand exists. **Why:** Vatsal's method (each finger clear of the others as it is placed), and Martin's note 2.12 (never silently fewer fingers).

**7. An optional palm cut.** On half the hands (when the rules allow a palm joint), the palm is cut along a straight line and the part beyond the line hinges on the rest, +/-30 degrees. The cut is drawn uniformly among the lines (30 degree directions, 10 mm offsets) that leave at least one finger on each side, keep the mount whole on the fixed part (at least 30 mm from the centre) and keep the hand clear. Every finger beyond the line rides on the moving part. **Why:** Martin's note 2.13 (palm joints decided on the finished palm; several fingers on one part, like a pinkie and a ring finger); a straight cut of a convex palm leaves two convex plates. If no cut fits, the hand is drawn again, so half the hands have one (48% of 1000).

### What is computed from the stored numbers

**8. The motor and how deep each base sits.** Every joint is the same servo: a 34 x 20 x 29 mm box, its axis along the 29 mm edges, 9.5 mm from one end and centred in the 20 mm width (`motor.py`). It lies along the finger, its body behind its axis, except a bend servo whose link before it is too short for that, which is flipped to lie along the link after it or, where that has no room, turned with its body on the back of the finger (step 5). Its body sits in the link before its joint (the palm, for a finger's first joint), and the link after it holds the horn and the idler in a yoke; a flipped servo's the other way round. A finger's base joint sits below the palm's front face, just deep enough that the housing of its first servo (and of the second, when the two share the point) is flush with the face: 12 mm for a bend first, 16.5 for a spread, 11.5 for a twist (flat fingers). **Why (Martin, 10/14):** the base joint sits half a servo plus a wall from the face, so the fingers start as close to the working side of the palm as the servo allows.

**9. Solids.** Every solid the clearance check and the simulator see:
- each link but the last: one box, square to its frame, around what it holds at its two ends: its yoke (a flipped servo's body), and the next servo's body (the yoke on a flipped one). A bending link is 33 mm wide and 24 mm deep (36.5 mm when a servo it holds is turned) and runs from 12 mm behind its joint to the next servo's front: links are servo-sized;
- the fingertip (each finger's last link): the convex hull of a capsule along the link, radius 16.5 mm (half the yoke's 33 mm width along the last servo's axis), from the yoke's back (12 mm behind its joint) to the fingertip, and of the yoke's box (after a flipped servo the fingertip holds its body: the box covers its front and horns, the capsule the rest) (Martin, 10/23: the printed bracket on the last servo sticks out of the capsule). **Why (Martin, 10/16):** a rounded fingertip, a little thicker than the yoke's 24 mm depth. The simulator's collider, the clearance check and the viewer's mesh all use the same 58 points (the capsule's 50 and the yoke's 8; at most 45 vertices and 61 faces, inside PhysX's 64-vertex, 64-face GPU limit);
- each finger's base servo, fixed to its plate;
- the main plate, the part beyond a cut, the mount, the board bay, and the cut's hinge servo.

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

**13. Poses and the object start.** At the zero pose every joint is at 0 and every finger is straight. Episodes start from the zero pose: the fingers start open (Martin, 10/20; until then every bend started at 0.35 of its upper limit, which stood curled fingers over the palm). The object starts over the centroid of the palm's outline, at the lowest height (from 5 mm above the front face, in 1 mm steps) where it clears every solid by 5 mm at the start pose. It never starts inside the hand.

### Checking, changing and simulating

**14. Viability checks.**
- **C1, clearance:** every two solids that do not touch by construction keep one printed wall (2 mm) apart, at the zero pose, which is now also the start pose (exact box distances); the board bay must keep a wall from every finger's base links over their ranges (step 1). Touching by construction: the main plate with the mount, the hinge servo and the part beyond the cut; a plate with its base servos (they sit in pockets); within a finger, the two sides of each joint (and, across a 0 mm link, the links on either side). One exception keeps a link from lying in the palm: at the zero pose a plate is checked against the far end of a link hinged to it. A base servo's housing must also stay out of a cut's seam. Collisions while the fingers move are left to the physics and the policy (Martin, 10/14). Generation keeps C1 (step 6).
- **C2, fingertips can meet:** each fingertip is sampled 1,500 times over its own joints' ranges (the palm joint included, one hinge angle shared by every finger beyond the cut); a sample counts if it is above the plate and over the palm (the outline grown by 20 mm); a hand passes if some pair of fingers' samples come within 20 mm of each other. Every commercial hand passes.
- Of random Feetech hands, 73% pass both (C1 100%, C2 73%): 1.37 draws per usable hand, 0.1 s each (1000 hands, re-measured 2026-10-16 after the day's rule changes).
- A physics grasp test (C5) is deferred until the grammar work is done.
- **The bay** (10/23): a hand fails C1 when no turn and shift of the board bay keeps a wall from every finger's base links over their ranges (most often a finger pointing along the palm's edge, its link swinging back across the palm at -30 degrees). About 3-7% of random hands did before it became part of C1; random draws now take 0.16-0.24 s.

**15. Rule sets.** The grammar's own limits (No Rules) allow hinge, coupled and sliding joints, links of 0 or 10-200 mm, fingers up to 500 mm, bases anywhere on a 10 mm grid 10-370 mm from the centre facing any way. These limits are the commercial hands' extremes at the scale Feetech servos build them (step 18) x 1.1, rounded out: the longest link (SHARPA at 3.1 x, 180 mm), the longest finger (SHARPA, 450 mm) and the farthest base (SVH at 3.05 x, 329 mm), set on 10/14. Since the rebuild of 10/16 the scaled hands reach less far (140 mm, 370 mm, 161 mm, SHARPA no longer scaled), so these limits are wider than the commercial hands need; kept for now (open). The **Feetech rules** (`FEETECH_RULES`, also `EVOLUTION_RULES`; the fine stage, with `FEETECH_COARSE_RULES` the coarse stage, step 16) are what evolution and the simulator use: hinges only; bases on the 40 mm grid, 40-90 mm out, facing outward; links at least the servo floors (step 5). Custom Rules can only tighten the grammar: joint types, maximum fingers (6), joints per finger (5), palm joints (1), finger length (Feetech 225 mm), link length (Feetech 90 mm), base grid and distance, outward facing, servo link lengths. The search space (2-6 fingers, before the clearance check): 10^38.0 hands in the coarse stage, 10^64.8 in the fine stage, 10^123 under the grammar's own limits. No Rules also allows a link at its servo floor where that floor is 1 mm past the grid, so every Feetech hand is inside the grammar's own limits. Random hands and every mutation keep the rules by construction.

**16. Mutation, in two stages** (Martin, 10/20). Random hands are drawn, and evolve first, in the **coarse stage** (`FEETECH_COARSE_RULES`): joints only on the three kinds (an axis turns 90 degrees, from one kind to another) and links at their shortest, then in 20 mm steps. The **fine stage** (`FEETECH_RULES`) then refines them: 31 axis directions (an axis turns or tips 30 degrees) and 10 mm steps. Every coarse hand is a fine hand, so the switch needs no conversion (`evolution/driver.py --rules coarse --fine-from-generation G`). In both, a base moves one grid step (40 mm), a facing or tilt 30 degrees.
- **Turns and mirror images are one hand** (Martin, 10/20): the palm-up task cannot tell apart a hand turned 90 degrees about the palm normal or mirrored, so random draws and mutations keep one copy of the eight (`operators.canonical`: the copy whose sorted fingers come first, among those that keep the rules and C1; a servo slid by two joints at one point keeps its side, so a mirror image is checked again). Its fingers are kept sorted.
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
- **The scale:** tried at 1, 1.25, ..., 2.5, 3, ..., 5, then in 0.05 steps below the first that works. A scale works when the fit keeps the servo link lengths, passes C1, keeps every joint's kind (or lies within 20 degrees of the commercial axis, for an axis about 45 degrees between two kinds, as on SHARPA's thumb) and every fingertip map stays within 15 mm RMS and 40 mm at worst (below). The last condition exists because moving fingers clear can leave a fit that is no longer the hand (on 10/16 Dex3 passed C1 at its real size but 25 mm and 38 degrees off; the fingertip refit now builds it at its real size within the limits).
- **Moving fingers clear:** a finger that fails C1 is refitted with its clash counted, first by local search, then in each of the 48 grid frames (12 facings x 4 tilts), the best frames refined in turn; the difference goes into its zero-pose offset. A thumb lying along the palm comes out standing off it.
- **Judged by the fingertip** (Martin, 10/20: "maybe doing it in fingertip manipulability space rather than joint space?"). Each finger is sampled at 256 joint configurations drawn uniformly over the commercial ranges (couplings followed). The fingertip map's error is the distance between the grammar fingertip and the commercial one over those samples, in real-size mm: RMS at most 15 mm, worst at most 40 mm (`conform.map_within`; both thresholds a first guess, pending Martin). At every scale, after the clearing refit, each finger is refitted on 64 of the samples (`conform.map_refit`), keeping every joint's kind; its base may move. Until 10/20 a fit was accepted when each joint, axis and fingertip was within twice the target (20 mm, 40 degrees). That passed LEAP's thumb with axes 22-33 degrees off and a fingertip map off by 39-72 mm RMS. Judging joint by joint also dragged bases together, leaving collisions the clearing refit could not remove.
- **Errors** are millimetres at the real hand's size (20 mm on a hand at 2 x counts 10 mm), so fits at different scales compare. The target is 10 mm / 20 degrees.
- **Result** (`experiments/E16_grammar_rebuild/ours/commercial_scaled.py`; rebuilt 2026-10-23 under the buildable-parts grammar: crossing slides +1.5 mm, floors 1 mm past the grid, the fingertip with its yoke, the board bay, the crossing rules): 18 of 19 buildable, every one with each finger's fingertip map within 15 mm RMS and 40 mm worst at the real size. LEAP (2 x) and Wuji v2 left (3.3 x) become buildable; SVH no longer is (at every scale one finger's map or a clash fails; it keeps its real-size fit). Shadow (2 to 2.6), Tesollo (2.05 to 2.6), Orca (1.75 to 2), SHARPA (1.8 to 2) and Inspire (1.65 to 1.7) grow; the rest keep their scale. Each hand keeps its finger count, joints per finger, joint types and kinds; SHARPA's, Shadow's and SVH's palm joints each become a cut with the same fingers beyond it, ARMS's two one cut. (On 10/20: 17 of 19, LEAP and Wuji v2 left not buildable.)

| Hand | Scale | Fingertip map, worst finger: RMS / worst (mm) | Joint (mm) | Axis (deg) | Heaviest finger flat (N m) |
|---|---|---|---|---|---|
| Ability | 1.3 | 14.3 / 20.9 | 21.3 | 24.3 | 0.13 |
| Allegro | 1.3 | 13.2 / 28.4 | 23.9 | 15.3 | 0.60 |
| ARMS | 3 | 10.0 / 21.6 | 19.1 | 24.4 | 0.38 |
| Barrett | 1.45 | 6.1 / 14.9 | 11.9 | 3.6 | 0.24 |
| DClaw | 1 | 6.3 / 11.8 | 5.2 | 2.6 | 0.00 |
| Dex1 | 1.75 | 14.9 / 28.0 | 21.5 | 25.2 | 0.00 |
| Dex3 | 1 | 12.1 / 20.8 | 20.5 | 1.8 | 0.07 |
| Inspire | 1.7 | 7.8 / 14.7 | 29.6 | 38.2 | 0.14 |
| LEAP | 2 | 14.7 / 32.9 | 8.0 | 10.2 | 0.66 |
| MIDAS | 2.1 | 11.8 / 28.6 | 20.0 | 13.4 | 0.51 |
| Orca | 2 | 14.2 / 28.2 | 17.1 | 24.6 | 0.39 |
| Shadow | 2.6 | 9.6 / 21.4 | 14.4 | 17.1 | 0.65 |
| SHARPA | 2 | 12.7 / 28.9 | 26.9 | 31.2 | 2.07 |
| SVH | 1 (not buildable) | 15.5 / 26.3 | 9.1 | 10.6 | 0.12 |
| Tesollo DG-5F | 2.6 | 13.6 / 27.8 | 13.9 | 9.8 | 0.64 |
| Wuji v1 | 2.15 | 12.3 / 27.6 | 18.7 | 38.5 | 0.74 |
| Wuji v2 left | 3.3 | 11.9 / 26.9 | 5.5 | 15.5 | 1.06 |
| Wuji v2 right | 2 | 13.6 / 36.5 | 14.2 | 13.9 | 0.43 |
| XHand | 1.5 | 12.4 / 28.6 | 15.4 | 41.0 | 0.40 |

- **Not buildable up to 5 x:** SVH (10/23; buildable at 2 x on 10/20). It keeps its real-size fit under the grammar's own limits and its overlaps collision-filtered in the simulator.
- **Torque:** at its 10/23 scale SHARPA (2 x) has a finger heavier than the 1.39 N m limit can hold flat, palm up (2.07 N m); the next heaviest are Wuji v2 left at 3.3 x (1.06) and Wuji v1 (0.74).
- Scaled, commercial palms have a median area of 174 cm^2 under the outline rule; random Feetech palms 162 cm^2.

**19. Building a hand** (Martin, 10/23: "hit Generate full design, and get a full 3D printable and screw-assemblable model"; `hand_sampler/cad/`, the viewer's Build folder, `python -m hand_sampler.cad --hand hand.json`). The grammar drives and OpenSCAD builds: Python writes the hand's frames (from `derive.fk`, `finger_geometry`, `hand_geometry`; no new kinematics) into a readable `hand.scad` over a fixed library (`hand_sampler/cad/scad/feetech_hand.scad`), whose build settings (fits, walls, arm thicknesses, flange angles) can be edited and re-rendered. The servo is measured from Feetech's STEP: a 34 x 20 x 23 mm case, 16 mm horns on both sides out to 14.5 mm from its mid-plane, 8 M2 holes on a 12 mm circle in each horn, 4 M2 holes on each horn face of the case, a 3-pin 1.25 mm socket on each 34 x 23 face.
- **A joint:** the servo's back end slides into a socket on the part before the joint (Martin, 10/24: an alignment feature, not the screws alone, and no more material than that and the screws): 0.1 mm a side, 1.2 mm walls on its 34 x 23 faces, 2 mm plates on its horn faces as far as the bosses of the 4 M2 x 3.5 screws through the case's far holes, and a back plate; the rest of the case is bare, and nothing passes round or under a horn (Martin, 10/26: there is no room there; the full-face plates of 10/24 are gone) (a crossing's second servo gets a closed housing). A servo whose idler horn is off (a twist: one arm on its output horn) is held by its idler face instead, in a flange cup: a 2 mm plate with that face's four holes (near and far) and the stub's, and a 1.2 mm rim 4 mm along the case for alignment; it slides in along its axis and is screwed through the holes in the previous U's bridge before that link goes on. The part after the joint carries a yoke: a U (a 3 mm arm on the driven horn with 4 x M2 x 5, a 2 mm arm on the idler horn with 4 box screws: every horn takes four (Martin, 10/26); a bridge as wide as the arms' discs just past the case's swing); one arm on the driven (output) horn for an axis leaning 30 degrees or more along the finger (a twist: a cantilever is fine, Martin, 10/26; the double-supported wrap of 10/24 was dropped), the idler horn (the one nearer the plugs) taken off; that arm and the link's connector are one plate from the horn's face, at least 6 mm thick; a plate on the driven horn for a crossing. Every part's outer edges are rounded 0.8 mm on its body as a whole (Martin, 10/28: rounding each piece on its own left fillets in the middle of faces wherever a gusset or a connector ran on from it): a part's pieces are built sharp, each reaching 1.2 mm into the pieces it joins, and their union is shrunk by a 0.8 mm ball and grown back (a morphological opening, with manifold3d), which rounds every outer edge, leaves inside corners sharp and drops slivers under 1.6 mm; the palm's corners keep their 3 mm round. OpenSCAD reads each rounded body from `bodies/` next to `hand.scad` (`ROUNDED = false` builds the bodies sharp there, after a setting they depend on changes). Gussets stiffen the L pieces' inside corners (Martin, 10/26): 45-degree triangles 16 mm a leg in the layers beside the servo, where the corner is open (the case fills it in between): in a socket's side-plate layers, between the connector's plate and the socket's front, where the connector lands on its side (there at most about 14.9 mm, so the 45-degree face clears the horn's keyhole); in a U's arm layers, between each arm's edge and the connector's back face. None round a flange cup (Martin, 10/28, of the twist's holder: "what is this"): there they lay along the servo's sides with nothing to hold, and only read as ears. Each triangle is its own convex piece (a sweep hulls each piece), reaches 1.2 mm into both pieces it joins (so it is one solid with them), is kept to where the connector is, and is cut back like the rest of the part where something turns or must stay clear. The yoke slides on with the horns' centre screws out, which then go back through the arms.
- **A link** (Martin, 10/23, after the first parts "doesnt look buildable": bracket frames) is its yoke, the next servo's socket and a solid connector between them (a slicer's infill keeps it light; the hollow one of 10/23 opened into thin troughs wherever a servo's keep-out or a sweep crossed it). Martin, 10/24: the connector runs from the U's whole base to the socket's whole face toward it (a loft); where that face is turned a quarter turn against the U (a spread and a bend at one knuckle, 226 of the 682 U-yoke links in the batch), or the next servo lies across the finger (a turned bend, a twist), a loft twists or slants, so it is a straight block in the U's frame covering both faces, all right angles; into a flange cup it is a short straight box from the U to the cup's plate. A link that renders in two pieces gets connectors to its socket's other faces and is rendered again. Around a flipped servo (step 5) a link joins whatever its two ends hold, face to face: the link before carries a yoke at each end (two U's back to back, an H; or a single arm and a U), the one after holds the flipped servo's socket and the next servo's socket or yoke, and a fingertip after one is its socket and a rounded end. A single arm's face always leads a quarter-turned connector's block, which then stands out from the arm along its servo's axis, clear of that servo's case (built from the other end, it reached round a flipped single-arm servo's front and, swept, took the bosses of its flange cup).
- **Every part** is its pieces less its servos' keep-outs (case, horns, plug, screws and a screwdriver's way in) less what its neighbours sweep over the joints' ranges (chained hulls every 5 degrees; across a crossing, both joints, every 10): the part after a joint keeps its shape and the part before it gives way. Across a crossing the first servo's case, swept back about both joints, can graze the far link's bridge in lobes; that link gives way to the convex hull of what the sweep takes from each of its pieces, a clean relief. Links and fingertips are clipped to their simulator bodies grown by 2.4 mm. OpenSCAD writes each part as OFF, which keeps its mesh indexed (an STL read back welds faces meeting along one edge wrongly, which had made solid parts look cut or split), and the STL for printing is written from it.
- **The palm** is a tray designed face first (Martin, 10/28: "the palm face should be the first thing in the design, with the motors mounted under it"; 10/29, after the assembled palm looked "super cobbled together, lots of inelegant gaps and openings, structural things that are thinly supported, open gaping holes"; 10/30: material rather than slots, and "our palm also lacks internal gusseting and structure"): a 3 mm plate whose face stands 3.5 mm over the base servos' tops (so the screwdriver for a lying servo's case screws passes under it along the servo's axis: no slots in the face), one 2.5 mm wall round the whole outline down to the palm's back face, which stays open (cables leave there), and under the face a grid of 2.5 mm ribs as deep as the wall, spread evenly about 30 mm apart along the palm's y and z (Martin chose the grid over ties from each housing or gussets alone). A rib stops at a pocket and at a case screw's lane, none runs alongside a wall or housing closer than 8 mm (within 10 degrees of it), and every rib ends on something (the wall, a housing, a pocket's wall or another rib): an end left in the open goes back to the last rib it crosses. Gaps narrower than 8 mm between what stands under the face (wall, housings, pocket walls, ribs) are filled solid and their inside corners rounded 4 mm (a morphological closing of their outlines, extruded through the tray). The face stops 10.5 mm behind a lying base servo's axis across its yoke, so no plate is left round or between the U outputs at the palm's edge (Martin, 10/28): where what lies ahead of that line is a strip along the edge (no other base servo's housing or window in it, at most a quarter of the palm) the face follows the line out to the outline; elsewhere the outline's corners within the yoke's reach drop out of the face's hull (10/30: that hull everywhere had cut a slant past a window, slicing its floor and wall). Every base servo screws straight into its housing under the face (Martin, 10/29: no intermediary pieces): a lying one into a housing from its pocket's back face to its back end, the case closed in up to where its yoke turns (10/30: no slot between a short socket and the pocket's wall); one standing on its back end into a socket hung from its pocket's floor; a crossing's into a cup under the tray, its pocket's walls reaching down to it (a tube round the slot, open at the bottom, where it goes in from the back). Every finger is built off the hand and slotted into the palm whole with its base servo (Martin, 10/30: "best to make it so we can always build the finger off the hand and then slot it into the palm, from the sides or bottom"; deep screw holes would cause problems for other hands): a lying servo slides in from the palm's edge, one under the face drops in through its window, and a crossing's servo standing on its back end goes in from the palm's open back, the finger up through its pocket (open through the back too) and its driven horn up a slot in the cup's side plate, so the cup holds it by its socket and the two case screws on its idler side. No holes are cut for the arms' screws. A servo lying in the palm (a base servo, a cut's hinge) has its case screws driven along its axis from one side (its two screws are enough): the side with less in the way, never through another servo's housing nor up through the face (a tilted servo's screw lines are followed in 3D). A 30 mm way runs out from that side plate, cut through whatever else is in it, which is mostly nothing, or the wall at the palm's edge (a hole straight out through it); no rib or fill enters it. Only a servo boxed in on both sides gets its way through another wall (a note in the checks). Each finger's base turns in a pocket: a box in its base servo's frame round what turns there (the arms and bridge, 19.6 mm along the axis and 20 mm round it; behind a single arm, as far as its connector turns; a crossing's part and the next link's yoke), run out of the palm the way the finger mostly leaves (through the wall for a servo lying at the edge, through the face for a finger rising out of it, up along the axis for a twist standing out of the face; a finger leaving along its servo's axis in the palm's plane gets the arm's layer beside the case only), walled (2.5 mm) on its other faces down to the servo's housing, and emptied of everything but the other fingers' housings. Where the printed base link (its connector, where its gussets reach, and the next servo's housing or yoke), turned 10 degrees either way, leaves the box inside the palm (a slanting finger), a convex pocket clears it there; where it stays in the box, the box is its pocket (10/30: the convex one, grown 3.4 mm past the box, only shaved the box's walls). A box round a slanting finger would take a whole corner of the palm, and with it a neighbour's housing; the simulator's link body, wider than the printed link, took a standing servo's own pocket walls and left its socket loose. Past its pocket a finger that folds onto the palm stops on it, as it does in the simulator, whose palm is a solid plate (Martin, 10/29: a clean window round the base joint only, not a cut for the whole fold); the check reports the free range. The arm's mounting and the board are left for later (Martin, 10/29): no bay, mount or column screws for now.
- **Checks** before anything is printed (`hand_sampler/cad/checks.py`): every part one closed piece; every joint over its whole range without its two bodies, or the servo case and plug, cutting into each other (a finger stopping on the palm past its pocket is a note, unless it cannot turn 10 degrees either way); every servo held (3 of the 4 screw bosses round its case's far holes, a warning under 4 and under 40% of its back wall; a crossing's servo slotted in from the palm's back, the 2 on its idler side); a 25 mm screwdriver reaches every screw in the assembly order: two of the case screws past the part holding the servo, then the arm screws at some joint angle ("servo first") or the link slid in on the servo ("link first"); a finger on the palm is built off the hand and slid in whole with its base servo, from the servo's front or, a crossing's, from the palm's back, passing the palm all the way (60 mm);  no two other parts touch at the zero pose; how far each part reaches past its simulator body (2.5 mm allowed; warnings past it).
- **The kit** (`outputs/designs/<id>/`, zipped): `hand.scad` and the library, every part's STL in its own frame and laid flat for printing, `BOM.md` (servos, screws, the board, cables, filament), `ASSEMBLY.md` (servo IDs 10(i+1)+j, the hinge 1; each finger on board port CH(i+1); the build order with a picture per joint; zero calibration) and the check report. A two-finger hand takes about 10 s, a six-finger hand about 60-150.
- **Result** (10/30, the tray with ribs, fingers built off the hand; `outputs/cad/batch/v37_*`, run on the code before the last screw-way changes): coarse 23 of 24 (v36: 23), fine 19 of 24 (22), mutated 20 of 24 (23), oblique: a servo boxed in. Every hand it failed on for a new reason (crossing cups broken off, servos boxed in for their case screws, one palm whose rounding ran past the time limit) now builds with every check passing or is fixed in the code; the full rerun on the final code is pending, and Martin has minor design revisions to make. Left from before: finger links' housings keeping 2 of their 4 bosses (open since v29), mutated 18 (a clearance the grammar checks at rest only, and now a face corner cut off by its neighbours' pockets), Ability and Inspire, Dex1. v36 (the tray with pockets): 86 of 90 random hands and 15 of the 18 buildable commercial fits; the cut test hand (three base servos in a row 40 mm apart) cannot screw its middle servo in (open).
- **Result** (10/28, whole-body rounding and the face-first palm, `outputs/cad/batch/v29_*`): 82 of 90 random hands and 14 of the 18 buildable commercial fits build with every check passing, median 268 g a hand; open: six hands where something hung under the raised plate comes loose from it, a lying servo inside the palm whose yoke cannot get onto its horns under the plate, a housing keeping 2 of its 4 bosses (LOG, 10/28), and the four below. Before (10/24, sockets and full-face connectors, `outputs/cad/batch/v13_*`): 88 of 90 random hands with 2-6 fingers (coarse, fine, mutated 14 times, tipped axes) build with every check passing, median 19 s; the two failures are a crossing followed by a 30 mm link and a turned servo (the crossing's slid servo sweeps through that servo's housing; open). Of the 18 buildable commercial fits, 16 pass every check; in Ability (1.3 x) and Inspire (1.7 x), one base servo's case screws face a neighbour's cradle and no two pads fit on their narrow palms.

## The grammar in one table

"Per hand" means it can differ between hands (evolution can change it); "global" means one value for every hand. The Feetech rules apply on top of the grammar.

| Part | Parameter | Per hand or global | Grammar (No Rules) | Feetech rules | Step | Source |
|---|---|---|---|---|---|---|
| Hand | fingers | per hand | 2-6 | same | add / remove a finger | max: simulator layout; min: one finger cannot learn (Kushal) |
| Hand | joints per finger | per finger | 1-5 | same | split / merge a link | commercial max is 5 |
| Hand | palm joint | per hand | none or one straight cut, a finger on each side | same | cut / remove; move 10 mm; turn 30 deg | Martin 10/13 |
| Palm | origin and mount | global | the palm centre; a 40 mm column, 40 mm deep, behind the plate; the board bay (76 x 60 x 22 mm) at its far end, turned per hand | same | - | Martin 10/13, 10/23 (the board behind the palm) |
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
| Link | shape | derived | one box around its yoke and the servo it carries; the fingertip the hull of a capsule and its yoke | same | - | Martin 10/13 (servo-sized), 10/23 (fingertip) |
| Link | length | per link | 0 or 10-200 mm; fingertip 10-200 mm | the servo floors (30-51 mm, 1 mm past the grid where the grid is too tight; a link under 40-50 mm flips the bend servo after it where the next link has room, else turns it) to 90, fingertip >= 30 (40 after a turned or flipped bend), 0 only for crossings (44-52 mm before one) | 10 mm | E15; Martin 10/16 (turned servos), 10/23 (floors, crossings), 10/29 (flipped servos) |
| Physics | torque, speed, gains, density, friction | global | 1.39 N m (the Feetech's stall torque), 5 rad/s, 3, 0.078, 0.00058, 1750 kg/m^3, 0.5 | same | - | Feetech datasheet (torque); Kushal (the rest) |
| Check | C1 clearance | - | solids a wall (2 mm) apart at the zero pose (the start pose: fingers open); the bay clear of the base links' sweeps | same | - | |
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
| Link length | joint | the servo floor (30-51 mm; a short link flips or turns the bend servo after it) to 90 mm; 0 for a crossing (44-52 mm before one); fingertip 30-90 mm (40 after a turned or flipped bend); finger at most 225 mm | uniform among the allowed lengths | 10 mm (the floor and 0 are neighbours) |
| Palm cut | hand | none, or a line 30-80 mm from the centre (30 deg directions) with a finger on each side | on half the hands, uniform among the cuts that fit | move 10 mm; turn 30 deg |

### 2. What is the same for every hand (fixed, not mutated)

| Constant | Value | Where it comes from |
|---|---|---|
| Motor | Feetech HL-3915: 34 x 20 x 29 mm, axis 9.5 mm from one end; one printed wall (2 mm) between moving parts | Martin; E15 |
| Mount | a 40 mm column, 40 mm deep, behind the palm's centre, the board bay at its far end | Martin 10/13, 10/23 |
| Palm thickness | 24 mm | a flat bend servo in its housing |
| Palm outline | convex hull of the mount and the servos' housings | Vatsal's hull |
| Link shape | one box around each link's yoke and the servo it carries (the fingertip: the hull of a capsule and its yoke); straight at rest | Martin 10/13, 10/23 |
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
| Link length | 0 or 10-200 mm (and the off-grid servo floors); fingertip 10-200 mm | servo floors (30-51 mm; a short link flips or turns the bend servo after it) to 90; 0 only for crossings; fingertip 30-90 (40 after a turned or flipped bend) |
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
| C1, clearance | every two solids a wall apart at the zero pose, where episodes start (except parts that touch by construction); no base servo in a cut's seam; the board bay clear of every base link's sweep | the parts must fit; deep starting overlaps make the physics blow up; the built hand must hold its board |
| C2, fingertips can meet | at least one pair of fingertips can reach a common region above the palm | a hand whose fingertips never meet cannot hold or turn an object |

## Not settled yet

- **The object start** is set aside (Martin, 10/20): the fingers start open, and the object starts just clear of them.
- **Torque.** The torque limit is now the HL-3915's stall torque, 1.39 N m: no random Feetech finger is too heavy to hold flat (the heaviest needs 0.71 N m; 8% need more than the old 0.5 N m), nor any commercial hand at its scale (at most 1.04 N m, Wuji v2 right). The servo's rated load is 0.44 N m, so stall is an upper bound.
- **The mount** (the simulator's 40 mm column and board bay) is not printed for now (Martin, 10/29: drop the arm and board mounting until the palm is right); the printed bay and mount of 10/23 come back at the end, their slot and pin angles to be checked against the robot's flange drawing.
- **The printed hand** (step 19) has not been printed yet: the fits (0.2 mm), the 2 mm walls and arms, and the horn screws' hold in PLA/PETG need a test print. Cables run outside the links on the finger's back; their routing past each joint is left to the builder.
- **Palm cuts between neighbouring bend-first fingers** are impossible: their housings are 7 mm apart, the seam needs 8.6.
- **Coupled joints and the policy:** they are tied and not controlled; whether they appear as tokens is open.
- **A physics grasp check (C5)** comes up again once the grammar work is done.

Status: the current grammar is on the branch `hand-grammar` (until 2026-10-16 it was built on `martin/hand-grammar`); the buildable-design work and its grammar changes (10/23) on `hand-build`; the grammar viewer is `experiments/grammar_viewer/`.
