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
- Ours: Ability, Allegro, ARMS (skeleton), Barrett, DClaw, Dex1, Inspire, LEAP, Orca, Shadow, SHARPA, SVH, Tesollo DG-5F, Wuji (v1), XHand. Kushal's: Dex3, Wuji v2 (left and right), plus his uniform-dynamics versions of the shared hands. Vatsal's: MIDAS.
- **Why:** as many real hands as possible, to test how well the grammar covers the market and to give the policy the most held-out hands.

### 20. Fit metric: revisit later
- For now, ours (largest joint-position, axis and fingertip error against 5 mm / 10 degrees), counting a pure difference in the zero pose as a match (Vatsal). Martin expects a better similarity metric is needed.

## Still to decide
6, 8 and 10 wait for the link-shape check (capsule, box or rounded box), since the cross-section sets the radius, the shortest link and the finger spacing.
6. Capsule radius (global): 15 mm from the XM335 motor (Vatsal) or 10 mm.
8. Evolution Rules link length: 20-80 mm, last link at least 15 mm (a motor must fit between joints); the grammar keeps 0-90 mm for commercial hands.
10. Finger spacing at least 2 x radius + 5 mm.
16. Changing a joint (under discussion): switch its kind in place, and toggle coupling to the joint before it.
17. Coarse / fine steps: Martin wants coarse steps coarser and fine as is (1 mm, 5 degrees); proposed coarse: 10 mm lengths and positions, 30 degrees for facing, tilt and palm-joint axes. No coupling step (one global ratio).

Checks come after the grammar and the Evolution Rules. Candidates: the overlap check, Vatsal's curl score, Martin's thumb-finger workspace overlap, and an opposition test (none exists yet).
