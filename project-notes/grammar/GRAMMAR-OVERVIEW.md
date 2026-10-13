# The hand grammar (October 2026)

Evolve to generalize trains one joint-token transformer across many hands and lets evolution choose which hands it trains on. The grammar defines the space of hands evolution searches. We need it to do three things: express every commercial hand we train or test on (within 5 mm and 10 degrees), be searchable at two resolutions (coarse to explore, fine to refine), and produce hands that all load into one simulator layout for one shared policy.

It combines three lines of work: our grammar on `martin/hand-grammar` (coarse and fine steps, palm joints, commercial-hand fitting, the simulator layout), Vatsal's physical grammar on `2026-10-02_physical_grammar` (the hull palm, the finger mount, joint kinds) and Kushal's uniform commercial hands on `2026-10-06-controlled_wuji_experiments` (shared dynamics, the joint sign convention). Every decision below, with its reason, is in `GRAMMAR-LOCK-2026-10-07.md`; the measurements behind the numbers are in `link-cross-section-study.md` and `palm-and-axis-study.md`.

This document describes the code in `hand_sampler/grammar/` and is updated with every change to the grammar.

## Three layers

1. **The grammar:** what a hand can be.
2. **The rules:** limits that random generation and every mutation obey by construction, so no hand is generated and then rejected. Evolution and the simulator use the Evolution Rules. In the viewer: Evolution Rules, No Rules (the whole grammar) or Custom Rules.
3. **Viability checks:** the two properties of a whole hand that no rule can guarantee.

Every limit comes from the commercial hands (their extremes with a 10% margin), never from a particular motor. Restricting evolution to hands our hardware can build will be a separate rule set; the Feetech study (`project-notes/grammar/experiments/E15_feetech_buildability/summary.md`) proposes one.

## How a hand is built, step by step

### What a hand stores

Only these numbers, all whole: millimetres, and degrees in multiples of 5.
- **Per finger:** its base position (y, z) on the palm, its facing, its tilt, which palm part it sits on (the main palm or a palm joint's section), and for each joint: its type (hinge, coupled or sliding), its axis (two angles) and the length of the link after it.
- **Per palm joint:** its hinge position (y, z) and its axis (two angles).

That is 4 numbers per finger, 3 per joint and 4 per palm joint, plus each joint's type. Random hands have 14-97 numbers (mean 51); the commercial fits range from 14 (Dex1) to 88 (ARMS). Everything in steps 8-13 is computed from these numbers.

### Building a random hand

**1. The origin and the wrist disc.**
- The origin (0, 0) is the centre of the wrist, where the hand mounts. Axes: **x** points out of the palm on the grasp side, **y** runs across the palm, **z** runs from the wrist toward the fingers.
- The palm is a flat plate 37 mm thick (the median of 15 commercial palms), centred on x = 0: its faces are at x = -18.5 and +18.5 mm.
- A disc of 20 mm radius around the origin, the **wrist disc** (called the heel disc in the code and the lock), is always part of the palm. It is a minimum palm: whatever the fingers do, the plate reaches the wrist mount and is at least 40 mm across, so a 2-finger hand still has something for the object to rest on. It is neither a keep-out zone nor where fingers attach; fingers start 10-160 mm from its centre.
- Where the disc ends up depends on the fingers. With every finger on one side (a human-like hand) it is the heel of the palm; with fingers all round the wrist (like DClaw) it is the middle of the palm.
- **Why:** real palms run from the wrist to the knuckles. Anchoring at the knuckle row made palms that were thin strips along the knuckles (lock item 22).

**2. How many fingers:** uniform 2-6. Two because a single Wuji finger cannot learn the task (Kushal); six because the simulator has six finger slots.

**3. Where each finger's base goes.** Fingers are placed one at a time. Each base goes on a point of a 10 mm grid, chosen uniformly among the points that pass three rules (796 points on an empty palm):
- **Distance:** 10-160 mm from the wrist centre (ARMS's thumb at 14.6 mm x 0.9 to Tesollo's middle finger at 141.2 mm x 1.1, rounded out).
- **Spacing:** at least 19 mm (one link width) from every base already placed. Every commercial hand passes except ARMS, a skeleton model.
- **Rim rule:** every base stays on the edge of the palm.

The rim rule. Stretch a rubber band around the wrist disc and every finger base. Each base must touch the band. A base the band does not touch, because other bases or the wrist disc surround it, would be a finger growing out of the middle of the palm, and is not allowed. The palm's outline (step 8) is this band widened by half a finger's width around each base.

```
   z (toward the fingers)
   ^
   |      A ---- B ---- C          A, B, C, D, E touch the band: allowed
   |     /               \         (B lies on the straight stretch from A to C)
   |    D        X        E        X is inside the band: not allowed
   |     \               /
   |      `---( W )-----'          W: the wrist disc, 20 mm radius, around (0, 0)
   +-----------------------> y
```

- Real hands are not exactly on the band, so a base may sit up to 8 mm inside it: the smallest whole millimetre every commercial hand passes. Allegro's thumb is 8.0 mm inside; Inspire 5.6, Wuji v2 5.6 and 5.0, ARMS 5.3, Tesollo 2.9, Wuji v1 2.7 and SVH 2.2; every other hand is on the band.
- Dex3 breaks the rule: its thumb sits between the wrist and its two fingers, 22.6 mm inside. It is reported, not accommodated, and the simulator still loads it.
- A finger on a hinged palm section (step 7) is measured against that section's own band, around the section's fingers and its hinge.
- A base inside the wrist disc is more than 8 mm inside the band, so in practice a base is at least 12 mm from the wrist centre, not 10.
- **Why (Martin, 2026-10-07):** "the fingers should be able to be on any of the edges but not sticking out of the middle of the palm". A finger based inside the palm starts inside the 37 mm plate: lying flat, its first link is buried in the palm; tilted up, it grows out of the palm's face, where the object rests. C1 cannot catch this, because a finger's first link and its own plate touch by design and are never compared.
- **Effect** (400 random hands): without the rule, 35% have a base more than 8 mm inside the palm (median 21 mm, up to 69 mm); with it, none. Viability does not change (C2 73% without, 72% with).

The uniform draw spreads bases over the area of the 10-160 mm ring, and most of that area is far from the wrist: the median base sits 112 mm from the wrist, and random palms come out larger than commercial ones (median 169 cm^2 against 71; 77% are larger than the largest commercial palm, LEAP's 95 cm^2).

**4. Which way each finger points.**
- **Facing:** the direction the finger leaves its base, in the palm's plane. 0 degrees points away from the wrist (+z), 90 degrees toward +y. Uniform over the full circle, independent of where the base is.
- **Tilt:** out of the plate, uniform over -30 to +90 degrees. +90 points straight up from the palm (DClaw); -30 covers Barrett's thumb at -25.
- On the coarse grid that is 12 facings and 5 tilts.
- **Why:** Vatsal's mount, (y, z) and a facing, plus a tilt so a thumb can angle across the palm (lock item 2).

**5. Each finger's joints and links,** from the base out.
- **Joints:** uniform 1-5. Joint j sits at the start of link j; the last link ends at the fingertip.
- **Link lengths:** a link between two joints is 0 mm with probability 0.12 (27 of 220 commercial bones are 0 mm: two joints at one point, like a knuckle that both spreads and bends), otherwise uniform over 20-90 mm on the coarse grid (15-90 mm on the fine grid). The fingertip link is 10-90 mm, never 0. Lengths of 1-14 mm do not exist. Each length is drawn from what is left of the 250 mm finger budget, keeping 10 mm for the fingertip.
- **Axis:** two angles in the link's frame (step 10). Each joint starts exactly on a kind, drawn with the commercial mix: bend 69%, spread 27%, twist 4%. Angled axes come only from later mutations. The coarse grid has 31 axis directions, the fine grid 1,261.
- **Type:** a hinge; after the first joint, coupled to the joint before it with probability 0.1 (18 couplings per 180 commercial joints after a finger's first); sliding only under No Rules (probability 0.0075).
- **No angles between links:** at the zero pose a finger is a straight line and bends only through its joints (lock item 4).

**6. The clearance test.** After each finger is drawn, the whole hand so far must have no overlap at all at the zero pose and none over 3 mm at the start pose (step 13). If it fails, that finger's base, facing, tilt and joints are drawn again, up to 50 times. A finger that still cannot be placed ends the hand with fewer fingers; a hand left with fewer than 2 is drawn again. The whole hand is tested, not only the new finger, because each new base moves the grasp point (step 9) and so turns every earlier finger's closing direction a little. **Why:** Vatsal's method; before it, 60% of random draws failed C1 and were wasted (lock item 25).

**7. An optional palm joint.** With probability 4/18 (SHARPA, Shadow, SVH and ARMS have palm joints), one finger gets its own hinged palm section: the first finger, in random order, for which the hand stays within the rules and clear. The hinge goes about halfway from the wrist to that finger's base (on the 10 mm grid, 15-110 mm from the wrist), with its axis along +z, so the section folds across the palm and cups it; it moves +/-30 degrees. Random hands get at most one palm joint. Mutations can make up to 6, each carrying at least one finger, and several fingers can share one (SVH's ring and little fingers do).

### What is computed from the stored numbers

**8. Palm plates.** The main plate's outline is the convex hull of the wrist disc plus a 9.5 mm disc (half a link width) at every base on the main palm and at every palm hinge, simplified to at most 32 corners, 37 mm thick. A palm section's outline is the hull of its bases and its hinge, built the same way. The palm has no parameters of its own (lock item 1).

**9. The grasp point G.** Above the centroid of the main plate's outline, 53.5 mm above the plate's mid-plane (half the plate, 18.5 mm, plus the object's half-size, 30 mm, plus 5 mm). The simulator starts the object there. G depends only on the bases and the hinges.

**10. Each finger's frame,** at the zero pose: **x** along the finger (from facing and tilt); **y**, the closing direction, square to x and pointing at G; **z** = x cross y, the plain bend axis. A finger that points straight at G falls back to +x of the palm (the grasp side), then to its tilt direction. Joint axes are written in this frame; links are straight, so every link of a finger shares it at rest. **Why:** every finger then curls toward the object wherever it sits and whichever way it faces (Martin, 2026-10-07); before, 21% of random bending joints closed away from it.

**11. Each joint's kind, sign and range.** The kind is whichever of the frame's z, y or x the axis lies closest to; it is a label, not a stored choice.

| Kind | Axis near | + means | Range |
|---|---|---|---|
| bend | z | the fingertip moves toward G | -30 to +90 deg |
| spread | y | the fingertip moves toward +y of the palm (if it cannot, +z, then +x) | +/-30 deg |
| twist | x | a right-handed turn about the link | +/-90 deg |
| palm joint | the hinge axis | its fingertips move toward the grasp side (+x) | +/-30 deg |
| sliding | the axis | along the axis | -20 to +25 mm |

- Any joint that can move its fingertip toward G at all follows the bend convention, so + means close; 0 is home.
- A coupled joint keeps its own kind and sign, follows the joint before it at 1.1 times its angle, and takes 1.1 times that joint's range, so the coupling never drives it into its own limit.
- **Why:** computed signs make + mean the same on every hand (Kushal's convention, lock item 3); one range per kind (item 5); 1.1 is the typical commercial coupling, rounded (item 7b).

**12. Link shapes.** Every link is a rounded box 19 mm wide (along the bend axis) by 18 mm high (along the closing direction), with 6 mm corners: the median of 208 commercial finger links. A link between two joints reaches 6 mm past both joints, so neighbouring links meet; the fingertip link ends exactly at the fingertip; a 0 mm link is a 19 x 18 x 13 mm block at its joint. Mass comes from a density of 1750 kg/m^3 and the exact rounded-box volume. **Why rounded boxes:** flat faces give stable contact patches like finger pads and rounded edges let the object roll over them; a capsule has no flat pad and a sharp box catches on its edges (lock item 6).

**13. Poses.** At the zero pose every joint is at 0 and every finger is straight. At the start pose, where episodes begin, every independent bend joint is at 0.35 of its upper limit (31.5 degrees), every other joint at 0, and coupled joints follow.

### Checking, changing and simulating

**14. Viability checks.**
- **C1, no self-overlap:** no two bodies (links and palm plates) overlap by more than 3 mm at the zero pose or the start pose. Pairs that touch by design are skipped: a link and the next one, links joined through 0 mm links, a finger's first link and its plate, a palm section and the main plate. **Why:** deeper overlaps made the physics push links apart at over 100 rad/s within a step or two. Generation already keeps C1 (step 6).
- **C2, fingertips can meet:** each fingertip is sampled 1,500 times over its own joints' ranges (its palm joint included); a sample counts if it is above the plate and over the palm (the outline grown by 20 mm); a hand passes if some pair of fingers' samples come within 20 mm of each other. The tolerance and sample count are set so every commercial hand passes. **Why:** a hand whose fingertips never meet above the palm cannot hold or turn an object.
- Of random hands, 74% pass both (C1 100%, C2 74%): 1.4 draws per usable hand.
- The object's start spot is set by the task, not checked against the fingers. A physics grasp test (C5) is deferred until the grammar work is done.

**15. Rule sets.** The grammar's own limits (No Rules) allow hinge, coupled and sliding joints. The Evolution Rules are the same limits without sliding joints, and are what evolution and the simulator use; there is no arm rule, since the task has no arm. Custom Rules can only tighten: joint types, maximum fingers (6), joints per finger (5), palm joints (6), finger length (250 mm), spacing (19 mm) and base distance (10-160 mm). Random hands and every mutation keep the rules by construction.

**16. Mutation.** Two stages: coarse steps of 10 mm or 30 degrees, fine steps of 1 mm or 5 degrees; every coarse value lies on the fine grid.
- **Parameter operators (both stages):** move a finger's base (y or z); turn its facing; tilt it; lengthen or shorten a link (stepping over the 1-14 mm gap: shortening 15 mm gives 0, lengthening 0 gives 15); turn or tip a joint's axis; move a palm hinge; turn or tip a palm hinge's axis.
- **Structural operators (coarse stage only):** add a finger (one 40 mm bend joint at a free spot, pointing away from the wrist, tilt 0); remove any finger in one step; split a link to add a joint (the finger keeps its length); merge two links to remove a joint; couple or uncouple a joint; give a finger its own palm joint; move a finger to another palm part; remove a palm joint (its fingers return to the main palm).
- **One mutation:** an operator chosen at random among those that can act; its options shuffled (a split's new joint kind weighted by the commercial mix); the first child that keeps the rules and C1 is returned (Vatsal's method). 98% of calls return a child, and 96% of children pass both checks.

**17. In the simulator** (Isaac Lab, `isaacsimenvs/inhand_reorient/scene/`).
- The hand is fixed palm-up with no arm: the palm's +x is turned to world up.
- Physics is the same for every hand, generated or commercial (Kushal's values): 0.5 N m torque, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, link density 1750 kg/m^3, friction 0.5. Hands then differ only in shape.
- Every hand fits one layout of 36 joint slots: 6 finger slots, each one palm-joint slot plus 5 finger joints, with a fingertip body. Unused slots are tiny, have no collider and are locked.
- Palm joints and couplings are PhysX mimic joints: a second finger on the same palm section follows with gearing -1, a coupled joint with -1.1. The policy does not control coupled joints. The largest tie error in testing was 2.4e-3 rad over 3000 random-action steps.
- Colliders: each link is the 8-corner core box with a 6 mm rounded shell (rest offset 6 mm, contact offset 8 mm). That gives correct contacts at rounded edges for 5.6 ms per physics step at 4096 environments (capsules 5.1-5.3 ms, sharp boxes 6.3-6.5 ms). Palm plates are convex hulls of up to 64 corners.
- Admission: grammar hands must pass C1. Commercial hands are exempt and have their overlapping pairs filtered from collision instead.
- Validation (2026-10-07, `isaac-validation-2026-10-07.md`): the authored hands match the grammar's kinematics to 0.0005 mm at reset and 0.028 mm after random steps.

**18. Commercial hands.** Eighteen hands: Ability, Allegro, ARMS (a human skeleton model), Barrett, DClaw, Dex1, Dex3, Inspire, LEAP, MIDAS, Orca, Shadow, SHARPA, SVH, Tesollo DG-5F, Wuji v1, Wuji v2 and XHand. Each is fitted onto the grammar's fine grid and checked against the 5 mm / 10 degree target: 17 of 19 (counting Wuji v2's left and right hands) fit, most within 2 mm and 4 degrees. The misses are ARMS (bases 15.8 mm apart, under the 19 mm spacing; 5.2 mm) and SVH (5.04 mm, its couplings held at 1.1). Wuji v1 fits at 4.0 mm and 3.4 degrees, and Wuji v2 within 1.6 mm. Dex3 breaks the rim rule (step 3). The fit compares geometry only: joint positions, axes and fingertips.

## The grammar in one table

"Per hand" means it can differ between hands (evolution can change it); "global" means one value for every hand. Steps: coarse explores, fine refines. Evolution Rules apply on top of the grammar.

| Part | Parameter | Per hand or global | Value or range | Coarse step | Fine step | Evolution Rules | Source |
|---|---|---|---|---|---|---|---|
| Hand | fingers | per hand | 2-6 | add / remove a finger | - | same | max: simulator layout; min: one finger cannot learn (Kushal) |
| Hand | joints per finger | per finger | 1-5 | split / merge a link | - | same | commercial max is 5 |
| Hand | palm joints | per hand | 0-6, each carrying >= 1 finger, a finger on at most one | give / move / remove | - | same | covers Shadow, SVH, ARMS |
| Hand | branching fingers | - | none | - | - | - | no commercial hand has one |
| Palm | outline | derived | convex hull of the finger bases + the 20 mm wrist disc | - | - | - | Vatsal; anchor at the wrist |
| Palm | thickness | global | 37 mm | - | - | - | median of 15 commercial palms |
| Palm | palm-joint hinge position | per palm joint | 15-110 mm from the wrist centre | 10 mm | 1 mm | - | |
| Palm | palm-joint hinge axis | per palm joint | any direction | 30 deg | 5 deg | - | |
| Palm | palm-joint range | global | +/-30 deg | - | - | - | |
| Finger | base position (y, z) | per finger | 10-160 mm from the wrist centre | 10 mm | 1 mm | - | commercial extremes x 0.9 / x 1.1 |
| Finger | facing (in-plate direction) | per finger | any | 30 deg | 5 deg | - | Vatsal |
| Finger | tilt (out of the plate) | per finger | -30 to +90 deg | 30 deg | 5 deg | - | |
| Finger | spacing between bases | rule | >= 19 mm (one link width) | - | - | same | every commercial hand passes except ARMS |
| Finger | rim rule | rule | every base on the palm's edge, at most 8 mm inside | - | - | same | Martin; every commercial hand passes except Dex3 |
| Finger | finger length (sum of links) | rule | <= 250 mm | - | - | same | longest commercial finger x 1.1 |
| Joint | axis | per joint | any direction in the link's frame (2 angles) | 30 deg | 5 deg | - | thumbs need up to 45 deg from a kind |
| Joint | kind | derived from the axis | bend / spread / twist | - | - | - | Vatsal |
| Joint | sign | global convention | + closes toward the grasp point / + spreads toward +y; 0 = home | - | - | - | Kushal |
| Joint | range | global per kind | bend -30 to +90; spread +/-30; twist +/-90 deg | - | - | - | Martin's 9/23 note |
| Joint | type | per joint | hinge, coupled, sliding | couple / uncouple | - | hinge and coupled only | |
| Joint | coupling | global | follows the previous joint, ratio 1.1, offset 0 | - | - | - | typical commercial ratio |
| Joint | random draw | global | starts on a kind: 69% bend, 27% spread, 4% twist | - | - | - | commercial mix |
| Link | cross-section | global | rounded box 19 x 18 mm, 6 mm corners | - | - | - | median of 208 commercial links |
| Link | length | per link | 0 mm or 15-90 mm; fingertip >= 10 mm, never 0 | 10 mm (skips 1-14) | 1 mm | same | commercial range |
| Link | rest bend | global | none (straight at the zero pose) | - | - | - | |
| Physics | torque, speed | global | 0.5 N m, 5 rad/s | - | - | - | Kushal |
| Physics | stiffness, damping, armature | global | 3, 0.078, 0.00058 | - | - | - | Kushal |
| Physics | link density, friction | global | 1750 kg/m^3, 0.5 | - | - | - | Kushal |
| Check | C1 self-overlap | - | no two links overlap > 3 mm at the zero or start pose | - | - | - | physics stability |
| Check | C2 fingertips meet | - | some fingertip pair's reachable regions meet above the palm | - | - | - | Martin; every commercial hand passes |

## Where each part came from

| | Ours | Vatsal | Kushal |
|---|---|---|---|
| Palm | palm joints, several fingers per palm joint, the wrist anchor, the rim rule | the hull plate | |
| Fingers and mount | spacing from the commercial hands | (y, z) plus facing | |
| Joints | free axes, coupling, coarse and fine steps, closing toward the grasp point | bend / spread / twist kinds | the sign convention |
| Links | 0 mm links, length limits from the commercial hands | | |
| Generation | uniform draws, commercial priors | building each finger clear of the others | |
| Physics | | | the uniform values |
| Commercial hands | the fitting and our 15 hands | MIDAS | Dex3, Wuji v2 |

## Summary tables

Ranges are commercial extremes x 0.9 / x 1.1; random draws are uniform within the ranges given.

### 1. What varies from hand to hand (what evolution mutates)

| Parameter | Per | Allowed values | Random draw | Coarse step | Fine step |
|---|---|---|---|---|---|
| Number of fingers | hand | 2-6 | uniform 2-6 (fewer if a finger cannot be placed clear) | add / remove a finger | none |
| Joints per finger | finger | 1-5 | uniform 1-5 | split / merge a link | none |
| Base position (y, z) on the plate | finger | 10-160 mm from the wrist centre; at least 19 mm from every other base; on the palm's edge (rim rule) | uniform over the 10 mm grid points that keep those rules | 10 mm | 1 mm |
| Facing (in-plane direction) | finger | any | uniform over the full circle | 30 degrees | 5 degrees |
| Tilt (out of the plate) | finger | -30 to +90 degrees | uniform | 30 degrees | 5 degrees |
| Joint axis direction | joint | any direction (2 angles); kind (bend / spread / twist) read off it | exactly on a kind: 69% bend, 27% spread, 4% twist | 30 degrees | 5 degrees |
| Link length | joint | 0 mm or 15-90 mm; fingertip 10-90 mm, never 0; whole finger at most 250 mm | 0 mm with probability 0.12, otherwise uniform 20-90 mm; fingertip uniform 10-90 mm; within the 250 mm budget | 10 mm (0 and 15 mm are neighbours) | 1 mm |
| Coupled to the previous joint | joint | yes / no | yes with probability 0.1 (not the first joint) | toggle | none |
| Which fingers share a palm joint | hand | up to 6 palm joints, each carrying at least one finger | with probability 4/18, one finger gets its own palm joint | give a finger its own / move it onto another / remove a palm joint | none |
| Palm joint hinge position | palm joint | 15-110 mm from the wrist centre | halfway between the finger's base and the wrist | 10 mm | 1 mm |
| Palm joint hinge axis | palm joint | any direction | along the palm (+z; the section cups) | 30 degrees | 5 degrees |

A four-finger hand with three joints per finger and no palm joint has 52 numbers (4 per finger, 3 per joint) and 12 joint types; each palm joint adds 4 numbers. In the commercial set: Dex1 14, Dex3 33, Inspire 56, Allegro 64, Wuji v2 80, Shadow 87, ARMS 88; random hands 14-97, mean 51.

### 2. What is the same for every hand (fixed, not mutated)

| Constant | Value | Where it comes from |
|---|---|---|
| Link cross-section | rounded box 19 mm wide x 18 mm high, 6 mm corners | median of 208 finger links on 16 commercial hands |
| Link shape at rest | straight (no bends between links) | Martin: an unnecessary per-link number |
| Palm thickness | 37 mm | median of 15 commercial palms |
| Palm outline | convex hull of the finger bases plus the 20 mm wrist disc | Vatsal's hull, anchored at the wrist (Martin) |
| Joint ranges | bend -30 to +90; spread +/-30; twist +/-90; palm joints +/-30 degrees | Martin's 9/23 note: spread needs a narrower range than bend |
| Joint sign | + closes toward the grasp point, or spreads toward +y; 0 = home | Kushal's convention |
| Coupling | follows the previous joint at a ratio of 1.1, offset 0 | typical of the commercial couplings (median 1.064, mean 1.145) |
| Physics | 0.5 N m, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, link density 1750 kg/m^3, friction 0.5 | Kushal's uniform values |
| Object start | 53.5 mm above the plate's mid-plane, over the palm outline's centroid | set by the task |

### 3. Limits: the whole grammar against the Evolution Rules

| Limit | Grammar (No Rules) | Evolution Rules |
|---|---|---|
| Fingers | 2-6 | 2-6 |
| Joints per finger | 1-5 | 1-5 |
| Palm joints | up to 6, each with at least one finger | same |
| Joint types | hinge, coupled, sliding | hinge, coupled |
| Link length | 0 or 15-90 mm; fingertip 10-90 mm | same |
| Finger length | at most 250 mm | same |
| Finger spacing | at least 19 mm | same |
| Base distance from the wrist centre | 10-160 mm | same |
| Rim rule | every base at most 8 mm inside the palm's edge | same |

Sliding joints are the only part of the grammar that evolution does not use. Every limit is a commercial extreme with a 10% margin; none comes from a motor.

### 4. Removed from the grammar to keep it simple

| Removed | What replaced it |
|---|---|
| Per-hand capsule radius | one rounded-box cross-section |
| Per-joint joint ranges | one range per kind |
| Rest bends between links (and Vatsal's leans) | straight links; mount facing and tilt, and free joint axes |
| Palm length and rigid extra palm pieces | the hull palm |
| Per-joint coupling ratio and offset | one ratio (1.1), offset 0 |
| Continuous (unlimited spin) joints | dropped: they need a slip ring and no hand needs them |
| Branching fingers | dropped: no hand we model has one |
| Joint-kind switch, capped tilt | the axis is a direction, stepped like every other angle |
| Named grammar variants and rule toggles | one grammar |
| A hand-shaped random template | uniform draws within the limits |
| Limits derived from a motor (Dynamixel, Feetech) | limits from the commercial hands |

### 5. Structural mutations (coarse stage only)

| Mutation | Effect |
|---|---|
| Add a finger | one 40 mm bend joint at a free spot that keeps the spacing and the rim rule, pointing away from the wrist |
| Remove a finger | any finger, in one step; a palm joint left with no finger is removed |
| Split a link | adds a joint; the finger keeps its length |
| Merge two links | removes a joint |
| Couple / uncouple a joint | it follows, or stops following, the joint before it |
| Give a finger its own palm joint | a new hinged section under that finger |
| Move a finger onto another palm joint | the two fingers then move together |
| Remove a palm joint | its fingers return to the main palm |

### 6. Viability checks

| Check | Passes when | Why |
|---|---|---|
| C1, no self-overlap | no two links (palm included) overlap by more than 3 mm at the zero pose or the start pose | deeper starting overlaps make the physics blow up |
| C2, fingertips can meet | at least one pair of fingertips can reach a common region above the palm (tolerance set so every commercial hand passes) | a hand whose fingertips never meet cannot hold or turn an object |

## Not settled yet

- **A Feetech-buildable rule set.** The Notion plan (10/12) is to evolve with it first and try the general rules later. The E15 study proposes its values: hinge joints only, at least 41 mm between joints, fingertip links of at least 16 mm, servo-aware base spacing, links of about 35 x 26 mm.
- **Palm size.** Uniform base placement makes random palms about twice the commercial size (median 169 against 71 cm^2).
- **The rim rule and Dex3:** keep the rule (Dex3 stays outside it) or drop it.
- **The object's start spot is not checked against the fingers.** At the grammar's start pose, some link touches the 60 mm start cube in 68% of random hands and in 9 of the 19 commercial fits.
- **Coupled joints and the policy:** they are tied and not controlled; whether they appear as tokens is open.
- **A physics grasp check (C5)** comes up again once the grammar work is done.
- **A better similarity metric** for fitting commercial hands, one that includes contact and rolling behaviour.

Status: the grammar is being built on `martin/hand-grammar`; the grammar viewer is `experiments/grammar_viewer/`.
