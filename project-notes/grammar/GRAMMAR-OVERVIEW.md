# The hand grammar (October 2026)

Evolve to generalize trains one joint-token transformer across many hands and lets evolution choose which hands it trains on. The grammar defines the space of hands evolution searches. We need it to do three things: express every commercial hand we train or test on (within 5 mm and 10 degrees), be searchable at two resolutions (coarse to explore, fine to refine), and produce hands that all load into one simulator layout for one shared policy.

It combines three lines of work: our grammar on `martin/hand-grammar` (coarse and fine steps, palm joints, commercial-hand fitting, the simulator layout), Vatsal's physical grammar on `2026-10-02_physical_grammar` (the hull palm, the finger mount, joint kinds) and Kushal's uniform commercial hands on `2026-10-06-controlled_wuji_experiments` (shared dynamics, the joint sign convention). Every decision below, with its reason, is in `GRAMMAR-LOCK-2026-10-07.md`; the measurements behind the numbers are in `link-cross-section-study.md` and `palm-and-axis-study.md`.

## The grammar in one table

"Per hand" means it can differ between hands (evolution can change it); "global" means one value for every hand. Steps: coarse explores, fine refines. Evolution Rules apply on top of the grammar.

| Part | Parameter | Per hand or global | Value or range | Coarse step | Fine step | Evolution Rules | Source |
|---|---|---|---|---|---|---|---|
| Hand | fingers | per hand | 2-6 | add / remove a finger | - | same | max: simulator layout; min: one finger cannot learn (Kushal) |
| Hand | joints per finger | per finger | 1-5 | split / merge a link | - | same | commercial max is 5 |
| Hand | palm joints | per hand | 0-6, each carrying >= 1 finger, a finger on at most one | give / move / remove | - | same | covers Shadow, SVH, ARMS |
| Hand | branching fingers | - | none | - | - | - | no commercial hand has one |
| Palm | outline | derived | convex hull of finger bases + 20 mm heel disc at the wrist centre | - | - | - | Vatsal; anchor at the wrist |
| Palm | thickness | global | 37 mm | - | - | - | median of 15 commercial palms |
| Palm | palm-joint hinge position | per palm joint | on the plate | 10 mm | 1 mm | - | |
| Palm | palm-joint hinge axis | per palm joint | any direction | 30 deg | 5 deg | - | |
| Palm | palm-joint range | global | +/-30 deg | - | - | - | |
| Finger | base position (y, z) | per finger | within 125 mm of the wrist centre (re-measured from the wrist in the build) | 10 mm | 1 mm | - | commercial max x 1.1 |
| Finger | facing (in-plate direction) | per finger | any | 30 deg | 5 deg | - | Vatsal |
| Finger | tilt (out of the plate) | per finger | open: set from the commercial hands in the build | 30 deg | 5 deg | - | |
| Finger | spacing between neighbouring bases | rule | >= 19 mm (one link width) | - | - | same | every commercial hand passes |
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

## Three layers

1. **The grammar:** what a hand can be.
2. **The Evolution Rules:** limits that random generation and every mutation obey by construction, so no hand is generated and then rejected. They are also what the simulator can load. In the viewer: Evolution Rules, No Rules (the whole grammar) or Custom Rules.
3. **Viability checks:** the two properties of a whole hand that no rule can guarantee.

Every limit comes from the commercial hands (their extremes with a 10% margin), never from a particular motor. Restricting evolution to hands our hardware can build is a later, separate step.

## A hand

**Palm.** One rigid plate, 37 mm thick (the median of 15 commercial palms). Its outline is the convex hull of the finger bases plus a 20 mm heel disc at the centre of the wrist, so the palm runs from the wrist to the knuckles; the palm has no parameters of its own. A palm joint is a hinged section of the plate, shaped the same way around its own fingers and the hinge; its parameters are the hinge's position and axis, and it moves within +/-30 degrees. A hand has up to 6 palm joints, each carrying at least one finger, and several fingers may share one (as SVH's ring and little finger do).

**Fingers.** 2 to 6. Each finger has a base position (y, z) on the plate, measured from the wrist centre; a facing, the in-plane direction it leaves in; and a tilt out of the plate. Neighbouring bases are at least 19 mm (one link width) apart, which every commercial hand satisfies.

**Joints.** 1 to 5 per finger. A joint's axis is a direction in its link's frame (two angles). Its kind is read off that direction: bend (closes the finger toward the palm), spread (swings it sideways) or twist (spins it about its length). The kind sets two things:
- the sign, following Kushal's convention: positive closes toward the grasp point, or spreads toward +y; zero is home;
- the range: bend -30 to +90 degrees, spread +/-30, twist +/-90.

Axes between kinds are allowed; real thumbs use them (thumb axes lie up to 45 degrees from the nearest kind, finger axes within 15 degrees 96% of the time). The grammar has hinge, coupled and sliding joints. A coupled joint follows the previous joint in the same finger at a ratio of 1.1 (the typical value among the commercial couplings) with zero offset.

**Links.** Rounded boxes with one cross-section everywhere: 19 mm wide, 18 mm high, 6 mm corners (the median over 208 finger links of 16 commercial hands). Links are straight at the zero pose; the finger bends only through its joints. A link is 0 mm (two joints at one point, like a knuckle with spread and bend together) or 15 to 90 mm. A fingertip link is at least 10 mm, and a whole finger at most 250 mm.

**Physics.** Every hand, generated or commercial, uses Kushal's uniform values: 0.5 N m torque, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, link density 1750 kg/m^3, friction 0.5. Hands then differ only in shape.

A typical four-finger hand with three joints per finger has about 50 numbers: 4 per finger for the mount, and 3 per joint (two axis angles and the link length), plus a coupling flag per joint and 4 per palm joint.

## Evolution Rules

Hinge and coupled joints only, within the limits above. There is no arm rule, since the task has no arm.

## Mutation: coarse and fine

Every number steps by the same rule: coarse 10 mm or 30 degrees, fine 1 mm or 5 degrees. That covers link lengths (stepping over the 1-14 mm gap), base positions, facings, tilts, joint axes and palm-joint hinges.

The coarse stage also changes structure:
- add a finger (one bend joint), or remove any finger in one step;
- split a link to add a joint (the finger keeps its length), or merge two links to remove one;
- couple a joint to the one before it, or uncouple it;
- give a finger its own palm joint, move it onto another palm joint, or remove a palm joint (its fingers return to the main palm).

Coarse steps explore; fine steps refine a hand without changing its structure.

## Random hands

Counts are drawn within the limits. Each joint starts exactly on a kind, drawn with the commercial mix (69% bend, 27% spread, 4% twist), and one finger is placed as an opposing thumb with the same probability as in the commercial set.

## Viability checks

- **C1, no self-overlap:** no two links, the palm included, overlap by more than 3 mm at the zero pose or the start pose. Deeper overlaps made the physics push links apart at over 100 rad/s within a step or two.
- **C2, fingertips can meet:** at least one pair of fingers whose fingertips' reachable regions meet above the palm. The tolerance is set so every commercial hand passes.

The object starts at a fixed spot above the palm, set by the task.

## Commercial hands

Eighteen hands: Ability, Allegro, ARMS (a human skeleton model), Barrett, DClaw, Dex1, Dex3, Inspire, LEAP, MIDAS, Orca, Shadow, SHARPA, SVH, Tesollo DG-5F, Wuji v1, Wuji v2 and XHand. Each is fitted onto the grammar's fine grid and checked against the 5 mm / 10 degree target; per-hand results come with the build.

## In the simulator

Every hand fits one layout of 36 joint slots: 6 finger slots, each a palm-joint carrier plus 5 finger joints, with a fixed fingertip body. Fingers that share a palm joint, and coupled joints, are tied with PhysX mimic joints (tie error at most 1.3e-3 rad in testing). Links are built as an 8-vertex convex hull of the box's core, padded by a 6 mm rest offset. That gives correct contacts at rounded edges for 5.6 ms per physics step at 4096 environments, against 5.1-5.3 ms for capsules and 6.3-6.5 ms for sharp boxes. We chose rounded boxes because objects rest and roll on fingers the way they do on real ones: flat faces give stable contact patches like finger pads, and rounded edges let the object roll over them.

## Where each part came from

| | Ours | Vatsal | Kushal |
|---|---|---|---|
| Palm | palm joints, several fingers per palm joint | the hull plate | |
| Fingers and mount | spacing from the commercial hands | (y, z) plus facing | |
| Joints | free axes, coupling, coarse and fine steps | bend / spread / twist kinds | the sign convention |
| Links | 0 mm links, length limits from the commercial hands | | |
| Physics | | | the uniform values |
| Commercial hands | the fitting and our 15 hands | MIDAS | Dex3, Wuji v2 |

## Summary tables

"Set in the build" marks a value the build fills in from the commercial hands, in the same way as the other limits.

### 1. What varies from hand to hand (what evolution mutates)

| Parameter | Per | Allowed values | Random draw | Coarse step | Fine step |
|---|---|---|---|---|---|
| Number of fingers | hand | 2-6 | set in the build | add / remove a finger | none |
| Joints per finger | finger | 1-5 | set in the build | split / merge a link | none |
| Base position (y, z) on the plate | finger | within 125 mm of the wrist centre (being re-measured from the wrist); neighbouring bases at least 19 mm apart | set in the build | 10 mm | 1 mm |
| Facing (in-plane direction) | finger | any | set in the build; one finger may be an opposing thumb | 30 degrees | 5 degrees |
| Tilt (out of the plate) | finger | set in the build | 0 | 30 degrees | 5 degrees |
| Joint axis direction | joint | any direction (2 angles); kind (bend / spread / twist) read off it | exactly on a kind: 69% bend, 27% spread, 4% twist | 30 degrees | 5 degrees |
| Link length | joint | 0 mm or 15-90 mm; fingertip 10-90 mm, never 0; whole finger at most 250 mm | set in the build | 10 mm (0 and 15 mm are neighbours) | 1 mm |
| Coupled to the previous joint | joint | yes / no | no | toggle | none |
| Which fingers share a palm joint | hand | up to 6 palm joints, each carrying at least one finger | set in the build | give a finger its own / move it onto another / remove a palm joint | none |
| Palm joint hinge position | palm joint | set in the build | set in the build | 10 mm | 1 mm |
| Palm joint hinge axis | palm joint | any direction | set in the build | 30 degrees | 5 degrees |

A four-finger hand with three joints per finger and no palm joint has 52 numbers (4 per finger, 3 per joint) and 12 coupling flags; each palm joint adds 4 numbers.

### 2. What is the same for every hand (fixed, not mutated)

| Constant | Value | Where it comes from |
|---|---|---|
| Link cross-section | rounded box 19 mm wide x 18 mm high, 6 mm corners | median of 208 finger links on 16 commercial hands |
| Link shape at rest | straight (no bends between links) | Martin: an unnecessary per-link number |
| Palm thickness | 37 mm | median of 15 commercial palms |
| Palm outline | convex hull of the finger bases plus a 20 mm heel disc at the wrist centre | Vatsal's hull, anchored at the wrist (Martin) |
| Joint ranges | bend -30 to +90; spread +/-30; twist +/-90; palm joints +/-30 degrees | Martin's 9/23 note: spread needs a narrower range than bend |
| Joint sign | + closes toward the grasp point, or spreads toward +y; 0 = home | Kushal's convention |
| Coupling | follows the previous joint at a ratio of 1.1, offset 0 | typical of the commercial couplings (median 1.064, mean 1.145) |
| Physics | 0.5 N m, 5 rad/s, stiffness 3, damping 0.078, armature 0.00058, link density 1750 kg/m^3, friction 0.5 | Kushal's uniform values |
| Object start | a fixed spot above the palm | set by the task |

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
| Base distance from the wrist centre | at most 125 mm (being re-measured) | same |

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
| Limits derived from a motor (Dynamixel, Feetech) | limits from the commercial hands |

### 5. Structural mutations (coarse stage only)

| Mutation | Effect |
|---|---|
| Add a finger | one bend joint at a free spot that keeps the spacing |
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

- A physics grasp check, finding a stable grasp in simulation, comes up again once the grammar work is done.
- Selecting or restricting evolution to hands our hardware can build.
- A better similarity metric for how well the grammar fits a commercial hand.

Status: the grammar is being built on `martin/hand-grammar`; the grammar viewer is `experiments/grammar_viewer/`.
