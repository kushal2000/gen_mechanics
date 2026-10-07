# hand_sampler — design

A grammar over hand designs and a set of mutations on it, for the evolution step
of the co-design loop.

Status: **design plus implementation of the offline half.** The genotype,
kinematics, validator, sampler, mutation operators and viewers are built and
tested. The integration layer (URDF build, collision gate, policy features) is
deliberately not — see §10.

---

> **Read Part A for the grammar as it stands.** Sections 1–13 after it are the
> original design document. They were written for the box-palm grammar and were
> not updated through the rollback of 2026-10-02 or the radial palm that
> followed, so their *rationale* still reads true while their *specifics* —
> representation, palm, mount, joint axes, operators, constants — describe
> something the code no longer does. §A9 lists exactly where they diverge.
>
> The docstrings in `design_space.py`, `validate_design.py` and
> `mutate_design.py` are the authority; Part A is a map of them.

---

# Part A — the grammar as it stands

## A1. The genotype

```
Hand    := Palm, [Finger]              2 to 6 fingers
Palm    := thickness                   25 mm, fixed; the OUTLINE is derived
Finger  := Mount, [Segment]            1 to 5 segments
Mount   := y, z, facing                where the base sits, and which way it leaves
Segment := Joint, length, lean         one joint per link
Joint   := kind                        roll | flexion | abduction
```

The genotype, the kinematic tree and a per-joint policy's message-passing graph
are still the same graph, which is §3's point and it survives unchanged. What
changed is the alphabet at each node.

A *generated* joint carries nothing but its kind. `Joint.offset`, `axis_override`,
`limits` and `drive` exist on the dataclass for **imported** hands only — a
vendor URDF has assembly angles and per-joint ranges, and `check_segment` makes
any of them a fault on a generated design.

## A2. Frames and conventions

The palm is a flat plate in the **y–z plane**. `GRASP_DIR` is **+x**, the palm
normal, the way fingers close. `+z` is away from the wrist; the arm comes in
along `−z`.

Angles in the plane — a `facing`, or the `bearing` of a point — are measured
**from +z**, counterclockwise, in `[0, 2π)`. So the wrist lies at bearing π.

Two constants place the hand on the arm:

| | | |
|---|---|---|
| `WRIST_STANDOFF` | 25 mm | the fingers are centred this far **forward** of the palm frame's origin. The origin is where the arm bolts on |
| `ARM_FACE_Z` | −50 mm | where the arm stops. Measuring the iiwa14's own meshes in this frame puts its frontmost vertex at exactly `−rpc.FLANGE_TO_PALM_Z_M` |

`PALM_CENTRE = (0, 0, WRIST_STANDOFF)` is the point every mount is an offset
from. It is *not* the frame origin, and the difference is what keeps a hand out
of the arm: centred on the origin, every vendor hand's thumb went ~17 mm inside
it.

## A3. The palm

**A palm stores one number, its thickness, and nothing else.** Its outline is
derived: the convex hull of

* the palm's own disc — `PALM_MIN_RADIUS` = 20 mm about `PALM_CENTRE` — and
* a disc of `PALM_RIM` = 5 mm about each place a finger starts.

So the plate always has a body whichever side the fingers crowd, it reaches out
to every base, and it **stops there**. Hulling circles rather than points is
what makes the outline smooth: there is no corner anywhere to fillet, and one
finger or six fall out of the same construction. `palm_outline` samples each
circle at 32 points, which inscribes the true shape by 0.1 mm.

It used to be grown by the full 20 mm, which buried a whole first link — the
shortest is 20 mm and a capsule only 15 mm wide.

`PALM_THICKNESS` = 25 mm = `2 × CAPSULE_RADIUS − 5 mm`, a quantum thinner than a
link is wide. `PALM_THICKNESS_RANGE` is degenerate, so thickness is not mutated
and there is **no `perturb_palm`** — a palm has nothing left to mutate.

Everything downstream is derived too: `palm_center` is the outline's centroid,
`palm_area` feeds the mass, `palm_hull` extrudes the outline into **one convex
mesh** that both the URDF and the USD author directly.

## A4. The mount

`Mount(y, z, facing)` — a (y, z) offset from `PALM_CENTRE`, plus the direction
the finger leaves in.

* **`y` and `z` are both on `PALM_QUANTUM` = 5 mm.** One grid in both
  directions, so a site 70 mm out is placed as precisely as one 20 mm out. This
  was polar until 2026-10-06 — a radius on the 5 mm grid and a bearing on the
  15° one — which sited the far ring 3.6× more coarsely than the near one, a
  bearing step being 5.2 mm of arc at the inner rim and 18.3 mm at the outer.
* **The ring is a bound, not a spelling.** `PALM_MIN_RADIUS` ≤ `hypot(y, z)` ≤
  `MAX_MOUNT_RADIUS` (20 to 70 mm). It stays a *circle* over a square grid
  because `MAX_MOUNT_RADIUS` is a reach, which has no corners. `Mount.radius`
  and `Mount.bearing` survive as derived properties, for reporting.
* **`facing` is on `ANGLE_QUANTUM` = 15°, and it is absolute** — measured from
  +z like any other bearing here, *not* from the radial direction. A finger at
  facing 0 points away from the wrist whichever side of the palm it sits on,
  which is what lets a thumb sit off one side and reach back across the palm.
* Where a finger **sits** and where it **points** are independent. `move_mount`
  moves one, `aim_mount` the other.

`Mount.polar(radius, bearing, facing)` converts and snaps, for the places where
polar is the natural way to *say* where a finger goes — seeds, a measured hand,
tests. It is a constructor, not a representation.

There is **no wedge**. A `WRIST_NOGO` sector used to forbid bearings pointing at
the wrist; it was a proxy for the arm and a poor one, since a bearing says where
a finger starts and what reaches the arm is where it points and how far. §A7's
arm-clearance rule measures the real thing.

## A5. Joints: three kinds and five leans

In the joint's own frame, where the link runs along **+x**:

| kind | axis | what it does |
|---|---|---|
| `ROLL` | (1, 0, 0) | spins the link about its own length |
| `FLEXION` | (0, 0, 1) | closes the finger — the only kind with curl authority |
| `ABDUCTION` | (0, 1, 0) | spreads the finger. Its axis **is** `GRASP_DIR`, so its curl authority is exactly 0 by construction |

`JOINT_LIMIT` is symmetric **±90°** for every joint, which is what makes the
absolute value in `curl_authority` sound.

A **lean** is how a link is bolted to its parent: one of five mounting
directions — straight on, or tipped `LEAN_QUANTUM` = 45° in four ways.
`lean_rot` applies it by the **shortest arc**, so a lean carries no spin of its
own; spin about the link is what a roll joint does, and two knobs for it is what
this grammar stopped having. `LEAN_NEIGHBOURS` makes two leans one step apart
when they are within 60°, so straight-on reaches all four tips and a tip reaches
straight-on and the two tips perpendicular to it.

45° is as far as a lean goes. A right angle between links is not two leans; it
is an abduction joint next to a flexion joint — the same corner, and an
**actuated** one.

This replaces the old `axis(theta, phi)` continuum and the per-joint `offset`
that aimed a finger. Three kinds and five leans are not a discretisation of that
continuum, they are a different and smaller alphabet: a kind says how a joint
sweeps, a lean says how the link is mounted, and the link body turns with it.

## A6. Links, the motor, and the envelope

Every constant below the grid is set by **one actuator**, the XM335-T323-T
(19.0 × 35.0 × 22.0 mm), not chosen:

| | | why |
|---|---|---|
| `CAPSULE_RADIUS` | 15 mm | running the 35 mm axis along the link leaves a 19 × 22 mm section, smallest enclosing circle 14.5 mm, up to the grid |
| `MIN_LINK_LENGTH` | 20 mm | a motor has to fit between two joints |
| `MIN_DISTAL_LINK_LENGTH` | 15 mm | the last link carries no child joint, so it needs no motor beyond its own |
| `MIN_MOUNT_SEPARATION` | 35 mm | `2r + 5`: two parallel capsules are tangent at 2r, plus shell clearance |
| `PALM_THICKNESS` | 25 mm | `2r − 5` |
| `MAX_LINK_LENGTH` | 80 mm | unchanged |
| `LINK_QUANTUM` | 5 mm | unchanged |

Envelope: `MIN_FINGERS` 2, **`MAX_FINGERS` 6**, **`MAX_JOINTS_PER_FINGER` 5**.

**The cap binds, not the geometry.** The ring holds 568 grid points, and greedy
packing at the 35 mm floor fits 15 mounts in it against a cap of 6. This inverts
§5's intent, where palm capacity was what stopped a hand growing.

## A7. Validity

`validate_design.check` returns every reason a hand is illegal, in this order:

| rule | what it enforces |
|---|---|
| `check_envelope` | finger and joint counts inside the envelope |
| `check_palm` | thickness in range and on the grid |
| `check_layout` | every mount inside the ring, `y` and `z` on the 5 mm grid, `facing` in `[0, 2π)` and on the 15° grid |
| `check_finger` | per segment, via `check_segment`: link lengths in range and on the grid, the distal floor, and **no offset on a generated joint** |
| `check_packing` | every pair of mounts at least `MIN_MOUNT_SEPARATION` apart, then `check_base_clearance`: no two links closer than `2 × CAPSULE_RADIUS` at rest, consecutive links in one finger excluded |
| `check_arm_clearance` | nothing the hand owns reaches behind `ARM_FACE_Z` — the palm plate or any link, at rest |

The last two share one walk of the links, because doing it twice cost the test
suite 60% of its runtime.

`check_packing` and `check_arm_clearance` are **necessary conditions at the rest
pose**, not sufficient ones: what a hand sweeps once its joints move is the
collision gate's business, as §7 already says.

## A8. Mutation and seeding

Nine operators, four of them structural:

| operator | step | scope |
|---|---|---|
| `split_link` | divide a link, inserting a joint | one finger |
| `merge_links` | join two links, removing a joint | one finger |
| `add_finger` | attach a new single-joint finger | one finger |
| `remove_finger` | delete a single-joint finger | one finger |
| `perturb_kind` | change one joint's kind | one joint |
| `perturb_length` | ±1 quantum | every link |
| `move_mount` | ±5 mm in one of four directions | one finger |
| `aim_mount` | ±1 angle quantum of `facing` | one finger |
| `perturb_lean` | to a neighbouring lean | one segment |

`perturb_axis`, `perturb_offset` and `perturb_palm` are gone with the things
they moved. `move_mount` takes four neighbours rather than eight, so every move
is the same 5 mm and the walk stays isotropic; a diagonal is two steps.

**Every operator shuffles its candidates and returns the first that validates**,
which is why adding a rule is nearly free — a site that would reach the arm
becomes a different site rather than a failed move. Adding `check_arm_clearance`
cost 0.2 points of mutation success over 60 parents.

`perturb_kind` is weighted, not uniform: `KIND_WEIGHTS` is roll 1, flexion 6,
abduction 2. Flexion is the only kind that closes a hand; roll is the kind LEAP
spends one joint of sixteen on.

Seeding draws a pair of fingers at bearings ±30/±45/±60 or 0/90, radius 25–35 mm,
1–2 joints, links 35–50 mm, kinds weighted flexion 3 to abduction 1. Seeds are
drawn by **rejection, not repair**.

## A9. The measures

**`complexity(hand)` → `(n_motors, n_joints)`.** One motor per joint; couplings
are deferred, so this is readable without a simulator.

**`curl_score(hand)` → the second-best finger's `curl_authority`.** Each joint
contributes `(axis × (tip − joint)) · GRASP_DIR` — the Jacobian column projected
on the grasp direction — taking the max over joints and dividing by the finger's
reach. Verified against a numerical derivative of the tip over 300 random
fingers covering every kind, lean and facing: they agree to 0.013 nm/radian.

Two things it does **not** say, both measured:

* The max lands on the **base joint 86%** of the time, so it is close to asking
  whether the first joint closes the finger.
* A high score does not mean two fingers can **reach each other**. Each finger
  is measured alone, so two fingers pointing opposite ways with their tips
  300 mm apart both score 1.00. Of the hands this calls "ok" (≥ 0.35), 24%
  cannot bring two fingertips within 60 mm at any uniform flex. A real
  opposition test is a separate measure and nothing implements one yet.

## A10. The commercial hands

`commercial.fit(name)` reads a vendor URDF and returns a Hand in this grammar
plus notes on every way it is not the vendor's.

| | fingers/joints | curl | base slip mean/worst | shape error worst | legal |
|---|---|---|---|---|---|
| `leap` | 4 / 16 | 1.00 | 1.8 / 2.0 mm | 8.0 mm | yes |
| `wuji2` | 5 / 20 | 1.00 | 10.4 / 21.0 mm | 17.5 mm | yes, by 0.0 µm |
| `midas` | 4 / 16 | 0.78 | 4.3 / 6.0 mm | 27.8 mm | yes |
| `allegro` | 4 / 16 | 0.87 | 2.2 / 3.0 mm | 27.4 mm | yes |

Three things worth knowing before trusting a fit:

* **`_straighten` slides joint origins along their own axes**, which is
  kinematically exact and is what turns an L-bracket into a straight link —
  LEAP's 79° brackets come down to 0.6°. It is **skipped** when a chain is
  already straight on the grammar's own grid (under half a `LEAN_QUANTUM`),
  because its cost term measures axis *alignment* and would otherwise read a
  straight rod's mounting ANGLE as a bend to remove. That is what cost wuji2 its
  splay.
* **wuji2's knuckles are 22 mm apart against a 35 mm floor**, so the fit spreads
  its row 13 mm wider than the vendor's. It then clears the link floor by
  *exactly nothing* — 0.0 µm. It is not a hand anyone should build; the
  separator simply stops the moment it clears. Generated hands do not do this:
  over 60 drifted designs the median clearance was 13.6 mm.
* **Allegro's base joints read `roll`, and that is correct.** Its axis is along
  the finger's own direction, so at full extension that joint moves the tip
  0.0 mm; driven ±20° it sweeps 41 mm at 15° of flex and 71 mm at 45°. The
  grammar names a joint by its axis at the **rest pose**. LEAP's equivalent
  joint is a true abduction, 94 mm with the finger straight out.

## A11. Where §§1–13 still hold, and where they do not

**Still true:** §1 (what this replaces), §2 (the ghosting constraint and why the
envelope is a hard cost), §3's *argument* that genotype and phenotype should be
one graph, §8 (policy interface) in structure, §9 (the four warnings — complexity
versus policy exposure, sparse fitness, bloat, keep the Pareto archive), §10
(the integration layer), §12, §13's method.

**Superseded in detail:**

| section | what it says | what is true now |
|---|---|---|
| §3 | `Palm := box(w,l,t)`, `Mount := face, (u,v)`, `Joint := axis(theta,phi), offset` | §A1 |
| §4 | joint axes as a `theta`/`phi` continuum on a 15° grid; `offset` aims a finger | §A5 — three kinds and five leans; no offset on a generated joint |
| §5 Palm | a box with a mutated width and length, 40–100 mm | §A3 — derived outline, thickness only |
| §5 Mount | three thin faces, normalised `(u, v)`, `MOUNT_EDGE_MARGIN` | §A4 — a (y, z) offset in a ring, no faces, no margin |
| §5 Links | floor 15 mm, radius 10 mm | §A6 — 20 mm and 15 mm, both set by the actuator |
| §5 Fingers | `MAX_FINGERS` 5, 6 joints each, packing decides the count | §A6 — 6 and 5, and the **cap** decides the count |
| §6 | `perturb_axis`, `perturb_offset`, `perturb_palm` | §A8 — `perturb_kind`, `perturb_lean`, `aim_mount` |
| §7 | two mount-separation floors, 15 mm across faces and 25 mm within one | §A7 — one floor of 35 mm, plus the arm plane |
| §11 | the "held back" list, written in terms of faces and `phi` | the entries still name real deferrals, but not in this grammar's words |

**Retired measurements.** Four results were taken in a space that no longer
exists, and each is marked RETIRED where it sits: §5's joint-count distribution
at a 15 mm floor, §5's `radius_scale` Spearman, §5's palm-packing claim, and
§6's `add_finger` balance table with `perturb_palm` as its remedy. They are kept
as the record of how the space was arrived at. They are **not** evidence about
this grammar, and re-running them is not planned — the constants they measured
are set by the actuator now, and the questions they answered are not live.

---

## 1. What this replaces

`minimal/` answers *"draw me a random hand from a fixed space."* Evolution needs
*"given a hand, which hands are one step away?"* The second is the deliverable:
the mutation operators define the search topology.

So enumerability stops being the goal. Almost every constraint in `minimal/`
existed to make the space finite and countable so uniform sampling meant
something. Under local search from a simple seed, three properties replace that:
**connectivity** (can mutation reach complex hands), **locality** (does one
mutation change performance a little), **reversibility** (can the search back
out).

The production genotype it also replaces, `params.HandParams`, gave every hand
5 slots × 6 joints and expressed smaller hands by *ghosting*. Because
`enabled_for` took the first *n* rungs of a fixed `ACTIVATION_ORDER`, joint count
and joint identity were the same variable — `MCP_FE` and `PIP` were never ghosted
in 41,928 fingers while `CMC_AA` was ghosted 57.1% of the time, so "12 joints is
best" decoded to "has `CMC_AA` on both fingers." A tree has no ladder, so the two
vary independently.

## 2. The ghosting constraint

Ghosting is still how a design reaches the simulator: one Isaac Lab
`Articulation` view must hold every design in a population, so every design must
present the same joint count. It is no longer how a design is *represented*.

The genotype is variable-topology; the builder ghosts on the way out into a
**fixed envelope** (`MAX_FINGERS` × `MAX_JOINTS_PER_FINGER` = 5 × 6 = 30). Every
design pays for the envelope whether it uses it or not: 30 is the token budget
the policy attends over, and attention is quadratic in it. That number is the one place the simulator reaches back into the genotype,
and it wants a per-ghosted-joint cost measurement it does not yet have.

## 3. Representation

**The genotype, the kinematic tree, and a per-joint policy's message-passing
graph are the same graph.** A hand is a tree rooted at the palm; nodes are
joints, edges are links.

```
Hand   := Palm, [Finger]
Palm   := box(w, l, t)
Finger := Mount, Chain
Mount  := face, (u, v)
Chain  := Joint, link(length), [Chain]
Joint  := axis(theta, phi), offset
```

Grammar-guided GP normally separates genotype from phenotype, and the standard
failure is **locality**: a small genotype edit produces a large phenotype change
and local search degenerates into random restart. Hands escape that because the
natural genotype *is* the structure. Mutations become graph edits, there is no
decode step to go wrong, "does the policy transfer to a mutant" becomes "does the
GNN handle one added node," and subtree crossover is finger-swapping.

Do not add a separate encoding. Any vector or string decoded into a hand buys
nothing and costs locality.

## 4. Joint axes

The link runs along the joint's local **+x**; the hinge axis is spherical about
it:

```
axis(theta, phi) = [cos phi, sin phi sin theta, sin phi cos theta]
```

* **theta** ∈ [0, π) rotates the hinge within the plane perpendicular to the
  link. 0 is flexion, π/2 abduction — these stop being categories and become
  ends of one continuum.
* **phi** is the polar angle from the link, and is **pinned at π/2** — every
  hinge perpendicular to its bone. Below π/2 the link sweeps a *cone* of
  half-angle `phi` rather than a flat fan: still one revolute axis, but it reads
  as a two-axis joint, so it is held back (§11.4) until the space needs the
  complexity. The field stays in the genotype, and the validator enforces the
  pin, so re-enabling is one line in `perturb_axis`.

**Joint limits are symmetric ±90° for every joint.** Anatomical asymmetric ranges
stop meaning anything once the axis is a continuum — there is no principled way
to interpolate an asymmetric flexion range into a symmetric abduction one.
Practical range of motion comes from **contact** instead: a finger that bends
backwards hits the palm and stops, in simulation, for free.

**`theta` does not move the link at rest.** Changing a joint's axis changes which
way that joint *sweeps*, not where its child sits at zero angle — verified:
identical joint positions and link directions, different swept paths. That is
correct revolute behaviour.

One thing for `build.py`: a URDF can carry the axis as `axis = axis_of(theta)`
with zero origin rpy, or as SHARPA's `axis = [0 0 1]` with the orientation in the
rpy. Identical kinematics; they differ only in whether the **child link's frame
rolls about its long axis**. With capsules that is unobservable — a capsule is
symmetric about its axis and its OBB is a square-section box invariant under the
roll — but it becomes observable for any non-symmetric link, so pick the
convention deliberately and have §8 name the same one.

`theta` spans **[0°, 180°) in 15° steps — 12 values**, and that is already
minimal. An axis and its negation are the same hinge, so `theta` and
`theta + 180°` describe one joint differing only in which sign of rotation swings
which way; with symmetric joint limits the reachable set is identical, and
`wrap_theta` collapses them (measured: the tip sweeps the same set of points, to
0.00 mm). It cannot be narrowed further — `theta` and `theta + 90°` are
*perpendicular* hinges, and their tip paths differ by 70.7 mm.

**`offset` is the joint's zero angle** — where its link sits when the actuator
is at neutral, i.e. the angle it is assembled at. Structural, costing no motor,
and it carries the joint's travel with it.

It is what aims a finger. A base joint's offset reproduces exactly what a mount
pointing direction used to do (verified: every reachable rest direction matched
to 2e-12), and an offset further out gives the finger a **resting curl**, which
no mount orientation could express at all. One primitive doing both, available at
every joint rather than only the first — so `theta` and `offset` are the pair
that between them decide which way a joint sweeps and where it starts.

All angles lie on a **15° grid**. The reason is exact inverses (§6), not
tidiness: continuous parameters cannot give them, so add/remove pairs would leak
a little on every step. It also makes a design hashable, so fitness can be
memoised.

## 5. The design space

### Palm

Discretised at 5 mm. **Only width and length are mutated**, in ±10 mm steps.

| dim | range | mutated |
|---|---|---|
| thickness (x) | 15–40 mm | no — set by the seed |
| width (y) | 40–100 mm | yes |
| length (z) | 40–100 mm | yes |

Thickness is the dimension geometry cares least about; width and length move
mount separation and reach directly. The step is twice the grid because
separation's measured optimum band is centimetres wide and a 5 mm step crawls
across it.

Frame: origin at the wrist-face centre, palm occupying z ∈ [0, length], `+x` the
palm surface fingers close toward, `-z` the wrist.

### Mount

`(face, u, v)` — **position only, no orientation**.

Fingers mount on the **three thin faces** — `+z` and `±y`. The large faces are
excluded: a finger growing out of the gripping surface is awkward to build and to
mount an arm behind. Opposition comes from `±y` fingers curling toward `+x` to
meet a `+z` finger, measured closing to 6 mm against a 40 mm object.

`(u, v)` are **normalised** face coordinates, so a palm resize carries every
mount with it — that is what makes the palm cheap to mutate. Mutation
nevertheless **steps in metres**, because faces differ 2–4× in span and the spans
shrink with the palm.

A mount must stay `MOUNT_EDGE_MARGIN` = one capsule radius from every boundary,
or half the base capsule hangs off the palm. The margin is tight on the thin
axis — a 25 mm palm carrying a 20 mm finger leaves 5 mm of play — and that is
what a 20 mm finger on a 25 mm palm looks like.

The mount used to carry a pointing direction `(alpha, beta)`, plus a roll that
was dropped as gauge. Both are gone: a finger leaves along its face normal and
aiming it is the base joint's `offset` (§4). That also removes work a crossing
had to do by hand — moving to a new face now rotates the world direction by the
angle between normals while leaving the tilt relative to the face untouched,
which is the behaviour that had to be coded explicitly before.

### Links

| quantity | value |
|---|---|
| quantum | 5 mm |
| floor | 15 mm |
| ceiling | 80 mm |

Total finger length is not fixed. **One joint per link** — every segment is at
least the floor, so no two joints share a point.

The floor sits **below** `2 × CAPSULE_RADIUS`, and both facts follow from one
decision: truly co-located axes need a gimbal, so they are excluded, and the
nearest this space gets to a compact knuckle is two ordinary revolutes a 15 mm
spacer apart. (An earlier version allowed zero-length segments for exactly the
MCP-style knuckle that costs.) A link shorter than its own diameter is
geometrically a **sphere** — the cylinder vanishes and both joints sit in one
ball, a fair model of a knuckle housing. `urdf.py` drops the collider for such a
segment, so **`build.py` must emit that sphere** or short-linked fingers go
transparent to contact. About 9% of links sit in the `[15, 20) mm` band.

**Depth is coupled to length**: *k* joints need ≥ *k* × 15 mm of reach, and a
link must reach 30 mm before it can split. That makes the floor a far bigger
lever than its size suggests — dropping it from 20 mm moved the unselected
distribution from shallow to deep:

> **RETIRED.** The floor is back at 20 mm and set by the actuator, so this
> table compares two values of a constant that is no longer free. It is kept as
> the record of why the floor was lowered, not as a description of the space.

| joints | share (15 mm) | mean reach | share (20 mm) |
|---|---|---|---|
| 1 | 11.5% | 49 mm | 32.5% |
| 2 | 9.9% | 77 mm | 39.5% |
| 3 | 9.9% | 117 mm | 22.9% |
| 4 | 12.9% | 159 mm | 4.2% |
| 5 | 19.2% | 198 mm | 0.7% |
| 6 | 36.5% | 224 mm | 0.1% |

A neutral walk now sits near the joint cap rather than the floor, and above the
measured 145–160 mm reach optimum rather than below it. Since one link cannot
exceed 80 mm, **selecting for that optimum selects for at least two joints per
finger**; the two effects cannot be cleanly separated here.

### Fingers

> **RETIRED.** A palm has no size to pack against. The ring holds 15 mounts at
> the separation floor against a cap of 6, so `MAX_FINGERS` binds and geometry
> never gets the chance — the inverse of what this paragraph intends. See §A6.

Between `MIN_FINGERS` (2) and `MAX_FINGERS` (5). The intent is that **mount
packing decides how many fit** — measured, a 50 mm palm packs 4 and a 60 mm palm
5, both short of the cap, while an 80 mm palm reaches it. The cap exists only so
a runaway search cannot hand the simulator an arbitrarily wide articulation; per
§2 it is a hard cost every design pays.

Joints per finger are capped at 6 by the same envelope.

### Fixed

> **RETIRED.** A link's radius is `CAPSULE_RADIUS` = 15 mm, set by the
> XM335-T323-T's cross-section. It was never really free, so "ruled out on
> evidence" is the right answer to the wrong question — the measurement stands,
> but nothing in the grammar could act on it either way.

**Radius 10 mm** — the one parameter ruled out on evidence. `radius_scale` scored
Spearman −0.005 across a 2× range and every volume measure −0.006 to −0.018
across 7–20× ranges; a 10× span in finger volume moved decile means 1.979 → 2.089
against noise of ±0.046.

**One motor per joint.** Couplings are deferred, so `n_motors == n_joints` and
complexity is a genotype-level integer, readable without simulating.

## 6. Mutation operators

Nine, and the count follows two rules. **An operator reachable by chaining others
earns nothing** — it adds surface for a bug and a second place the same rule can
drift. But **operators that differ only in where they attach are kept apart**,
even where a tree makes them formally one operation, because that is what makes
the mutation mix controllable.

### Structural — ±1 joint each

| operator | effect | inverse |
|---|---|---|
| `split_link` | divide a link, inserting a joint | `merge_links` |
| `merge_links` | join two links, removing the joint between | `split_link` |
| `add_finger` | attach a new single-joint finger to the palm | `remove_finger` |
| `remove_finger` | delete a single-joint finger | `add_finger` |

`merge_links` acts only on fingers with two or more joints; emptying a finger is
`remove_finger`'s job. Both removals are correctly impossible at the floor —
`MIN_FINGERS` single-joint fingers.

These were briefly one `add_node`/`remove_node` pair, since attaching to a joint
and to the palm are the same operation in a tree. Separating them again matters
for two measured reasons: pooled uniformly, a new finger competed against every
splittable link and so was rare; and once the palm filled, splits kept succeeding
under the same operator name, **masking that palm capacity had run out**.

*Unit steps* keep the performance-versus-motors front dense. *Exact inverses* are
load-bearing for §9.3 and hold for both pairs.

Balance falls with depth, and with four operators the cause is visible:

> **RETIRED.** `add_finger` collapsed here because the palm filled. It cannot
> fill now, so the numbers below are not this space's, and the remedy the text
> names — `perturb_palm` — does not exist. What replaced the diagnosis is
> `test_operators_are_unbiased`, which holds P(up) inside 40–60% at n = 4 and 6.

| n | split | merge | add_finger | remove_finger | P(up) |
|---|---|---|---|---|---|
| 4 | 97% | 89% | 100% | 62% | 56.1% |
| 6 | 96% | 92% | 73% | 83% | 48.3% |
| 10 | 100% | 100% | 52% | 80% | 45.5% |
| 14 | 93% | 100% | 16% | 70% | 38.7% |

`add_finger` collapses as the palm fills while `merge_links` stays near 100%, so
a deep hand drifts down. That is palm **capacity**, not operator bias, and
`perturb_palm` is what relieves it.

A removed link folds into the proximal neighbour (the exact inverse of a split),
else the distal one, else the proximal merge clamped to the ceiling. The clamp is
unreachable from `split_link`, so it costs nothing in exactness — without it, a
finger whose adjacent links summed past the ceiling could not shed that joint.

### Parametric

| operator | step | scope |
|---|---|---|
| `perturb_axis` | ±15° in theta | **every joint** |
| `perturb_offset` | ±15° in the zero angle | **every joint** |
| `perturb_length` | ±1 quantum | **every link** |
| `move_mount` | ±5 mm across the surface, crossing face edges | one finger |
| `perturb_palm` | ±10 mm in width or length | one dimension |

`perturb_offset` replaces a mount-orientation operator that could only aim a
whole finger from its base (§4). It is whole-hand for the same reason
`perturb_axis` is: offset and theta are the same kind of per-joint angle on the
same grid, so they should explore at the same rate.

`perturb_axis`, `perturb_offset` and `perturb_length` are **whole-hand** moves: each joint or link
steps independently up or down, so a 20-joint hand has all 20 changed at once.
That trades locality for exploration rate, and it matters most for
`perturb_length`, which is the only operator that changes total reach —
`split_link` divides and `merge_links` rejoins, both reach-preserving. One link
per mutation would grow a hand 5 mm at a time against a reach optimum band tens
of millimetres wide. Each value reflects into range on its own, so only the
whole-hand rules can reject a draw, and a few independent redraws are tried
before the operator reports failure.

`move_mount` absorbs what a separate `remount` would do, since a step that
overflows a face carries onto the face across that edge; on an axis-aligned box a
face's tangents are its neighbours' normals, so no cube net is needed.

When a step overflows toward a face that hosts no finger it **clamps** rather
than refusing, since the thin axis has a 5 mm band against a 5 mm step.

### Deferred

**Crossover** (`swap_finger`) is free from the tree structure, but crossover is
the specific mechanism behind bloat in the GP literature, so it should be added
only with the joint-count instrumentation of §9.3 already running.

Design *parameters* the operator set does not yet reach — branching chains,
off-perpendicular axes, coupled and passive joints — are listed in §11.

## 7. Validity, in two tiers

Loosening the grammar relocates constraints rather than deleting them. In
`minimal/` validity was free by construction; once faces, mounts and axes are all
free, hands where one finger lives inside another become expressible.

**Cheap** — every mutation: lengths, angles and palm dimensions in range and on
their grids; joint and finger counts within the envelope; mount separation; base
clearance. It names *which* parameter is wrong so a mutation can be reflected
rather than discarded.

Mount separation has two floors. Across faces, 15 mm — deliberately loose,
because fingers there leave along different normals and diverge. Within a face,
2.5 × radius = 25 mm, because they run parallel and capsules are **tangent at
2r = 20 mm**. A single 15 mm floor permitted 5 mm of interpenetration and 18% of
same-face pairs genuinely intersected.

Separation alone is not enough: it constrains where a finger *starts*, not where
it *points*, and two fingers rooted a legal 25 mm apart can lean together until
their base links cross. `check_base_clearance` closes that with one closed-form
segment-segment distance per pair — proximal links at rest only, since collisions
further out depend on flexion and belong to the gate.

**Expensive** — once, before evaluation: self-collision over sampled
configurations, which is configuration-dependent and so cannot be a per-mutation
check. Use `gates/capsule.py`; it already replaces a 6.02 s/hand mesh check.
Its skip set is computed once from template link and joint *names*, which were
constants only because the old space had fixed topology — that needs deriving
structurally.

### Joints that point into the palm

Offsets let part of a joint's commanded range put its link inside the palm. This
is left alone, on measurement:

* **A base link can never rest inside the palm** — exhaustive over every
  `(face, theta, offset)` with a maximum-length link, deepest rest penetration is
  −0.0 mm. A link leaves from a point *on* the face and `offset` caps at 90°,
  which only reaches the tangent plane. Nothing starts broken, so a rest-pose
  palm check would be dead code rather than a guard.
* **Commanding adds another ±90° and that is what reaches in**, up to 12.5 mm —
  costing **3.2%** of a base joint's range on average, median 0%, worst 36.6%.
* **Distal links never enter** (0%), and they are not collision-filtered against
  the palm, so contact would stop them anyway.

Collision-derived joint limits would be the obvious fix and are **ill-posed for a
chain**: joint *k*'s free range depends on joints 1…*k*−1, so there is no single
correct limit, only a pose-dependent approximation. Against 3.2% that is a bad
trade, and motion into the palm is not useful motion, so selection pays for it.

**A hazard for `build.py`.** The palm must be collision-filtered against each
finger's *first* solid link — its cap sits inside the palm shell by construction.
The hand-written SHARPA map filters palm against **link_0 and
link_1**, and copying that would let the second link through too, turning a
measured 0% into a real problem. Filter the first solid link and no more.

If it ever must go, the route is not per-joint collision limits but **absolute
limits on the base joint** — bound it to ±90° from the face normal regardless of
offset, so `offset` picks where zero sits inside a fixed range. More physical (a
hard stop does not move because a link was assembled at an angle), at the cost of
offset-dependent travel, and it applies only to the base joint.

## 8. Policy interface

The learning step tokenises per joint, so the output contract is a graph with
typed features, not a URDF. **Nothing implements this yet** — no GNN,
transformer or tokenisation code exists — so the schema below is a proposal made
so policy work is not blocked, and is cheap to change while that holds.

**Node, per joint:** hinge axis in the parent's frame (3), limits (2), parent link
length (1), depth (1), is_root/is_tip (2), plus runtime angle and velocity.
**Edge, parent→child:** relative transform. For a root joint the parent is the
palm, so the edge carries the **mount transform** — which is how layout becomes
visible to the policy at all, and layout is what the evidence says matters.
**Global:** palm dimensions, finger and joint counts.

Everything in the **parent's frame**, never world frame: that is what makes the
representation invariant to placement and lets a policy trained on one topology
read another. Ghosted joints need a masked token rather than a zeroed one, so
"does not exist" stays distinguishable from "is at zero."

The current system instead passes a 143-dim flat morphology descriptor, and the
commit adding its ablation is candid that the premise is untested. A fixed-length
vector constant within an episode is what a policy learns to ignore; in a GNN the
morphology *is* the message-passing graph. **Whether that ablation ever ran is
open**, and it matters: if the descriptor is unused, the geometry effects below
are purely mechanical and the design search has been optimising properties the
controller cannot perceive.

> Do not add a design parameter unless a mutation operator moves it **and** a
> policy feature sees it.

## 9. Warnings

### 9.1 Complexity versus policy exposure

From the 24k population eval:

| fingers | share | mean goals | never scored |
|---|---|---|---|
| 2 | 85.3% | 1.938 | 4.3% |
| 3 | 13.4% | 1.753 | 3.4% |
| 4 | 1.2% | 0.487 | 23.2% |
| 5 | 0.1% | 0.000 | 100% |

Performance tracks population share almost exactly. Under i.i.d. sampling that is
a fixed skew; **under evolution it compounds every generation**, because the
population is redrawn from survivors. Complexity gets penalised for being
unfamiliar rather than worse — which runs against the simple-to-complex arc that
is the project's premise.

The design loop is deliberately training-free and names the same risk: a gain can
be the population learning to suit this controller rather than becoming better
hardware, and *the compounding is invisible to seed averaging, because it is bias
not variance*. Its stated mitigation — read the arms against each other — does
not cover the topology direction. **Adding `add_finger` to a fixed-policy loop
makes complexity reachable but not selectable.**

Options, increasing in cost: protect young topology classes from culling for some
generations; oversample under-represented topologies when the policy next trains;
or move toward *score after k gradient steps on this design*, which is the real
fix and the expensive one. Minimum instrument: log complexity against policy
exposure per generation.

### 9.2 Sparse fitness at the simple end

Seeds are 2–4 motors. Measured closure against a 40 mm object: 74% of 2-motor
seeds can touch it, 84% at 3, 100% at 4.

That is a **gradient, not a trap**. The trap is *every* seed scoring zero — hit
for real once, when an earlier seed set paired fingers on opposite faces and 58%
of the population could not reach the object at any joint angles. Here the seeds
that close outscore those that do not and `split_link` is the one-step path
between them.

Forcing two joints per finger would remove the failures and cost more than it is
worth: with `MIN_FINGERS = 2` it puts the whole population at four motors, and
the cheap end of the performance-versus-motors curve is a result, not a defect.

### 9.3 Bloat, and the conditions against it

The position taken is that explicit parsimony pressure and an explicit
generalisation term are both unnecessary: added complexity must pay for itself or
drift removes it, and if fitness is averaged over freshly sampled conditions,
overfitting to one instance is itself selected against. Sound, and no penalty is
added. But it is conditional on two things this package controls:

1. **Mutation symmetry** — additions and removals equally available, else a
   silent ratchet. Instrumented by `Stats`; read per-move balance, not raw accept
   rates, since several operators are structurally gated near a boundary.
2. **Object resampling** — the object accounted for 17.9% of variance across
   designs, larger than every geometry effect combined, because each design held
   one object for its whole run. `mutate.py`'s index alignment already cancels
   this for paired deltas; it remains live for absolute ranking.

Cheap settlement either way: log mean joint count per generation.

### 9.4 Keep the Pareto archive

Independent of selection: record `(performance, n_motors)`. The headline claims
are slices through it, it costs one integer per design, and it commits the
selection scheme to nothing.

## 10. Deferred: the integration layer

Not built here, and belonging with whoever owns the simulator side. In dependency
order:

1. `build.py` — genotype → URDF, ghosting into the envelope (§2). Needs the
   envelope number; should follow `rotations.py` for rpy and `inertia.py` for
   inertials.
2. `gates/` adaptation — the skip set must derive from the tree, not from
   template names (§7). The closed-form capsule test itself is reusable verbatim.
3. `features.py` — node/edge emission per §8.

Deferred *design parameters*, as opposed to deferred plumbing, are §11.

## 11. Held back, for later complexity

Deliberately absent so the space stays small enough to reason about. Each is
reversible — the genotype already carries the field or the shape — and roughly
ordered by value against cost.

| | what it adds | cost to enable |
|---|---|---|
| 1 | **fingers on the two large palm faces** — `+x` gives an opposition post rising from the palm, the most thumb-like arrangement here and the closest-closing pair measured | one tuple; the crossing logic already handles any face |
| 2 | **joint types beyond independent revolute** — rigid/mimic coupling, fully passive spring-loaded, differential | large; see below |
| 3 | **branching chains** — a finger splitting beyond some joint | `Chain` already carries children; roughly doubles validator work |
| 4 | **off-perpendicular axes (`phi`)** — the link sweeps a cone rather than a flat fan | one line in `perturb_axis`, and relax the validator's equality to a range |
| 5 | **coincident joints** — two axes sharing a point, an MCP-style knuckle | re-allow zero-length segments, and the special cases they carry through builder and renderers |
| 6 | **palm thickness** | one entry in `MUTABLE_PALM_DIMS` |
| 7 | **per-design joint limits** — currently a global ±90° | no evidence yet that searching over it pays |
| 8 | ~~**link radius**~~ — RETIRED: the radius is the motor's cross-section now, not a parameter | — |
| 9 | **actuator properties** — gear ratio, reflected inertia, torque limits | the node schema (§8) already has room |
| 10 | **a larger envelope, or a non-box palm** | both real simulator cost (§2) |

Only #2 needs more than a line. `minimal/` has a worked version of coupling and
passive joints — adjacency rules, mid-range rest poses, per-joint stiffness — and
the hard part is not representing them but mutating them: removing a joint must
decide what happens to its partner. The cost of *not* having it is worth naming.
Real anthropomorphic hands are heavily underactuated, and that is precisely how
they buy dexterity per motor, so "matches a market hand at equal motor count" is
uphill while every joint here has its own motor.

### Not on this list, deliberately

Three things were *consolidated* rather than removed, and re-adding them would
restore redundancy rather than capability:

* **mount roll** — gauge. Rotating the mount by *r* about the finger axis while
  subtracting *r* from the first joint's `theta` gives an identical hand, so
  carrying it would give one hand two spellings.
* **mount pointing direction** — reproduced exactly by the base joint's `offset`,
  which also does strictly more (§4).
* **`remount`** — a step that overflows a face now carries onto the next, so a
  separate teleport reaches nothing new.
Note this list is about *redundancy*, not about size. `add_finger` /
`remove_finger` were briefly folded in on the same argument — in a tree,
attaching to a joint and to the palm are formally one operation — and have been
separated again (§6), because the argument was wrong in practice: pooling them
made a new finger rare and hid palm exhaustion behind a still-succeeding split.

## 12. Open

1. What articulation envelope is affordable? This asks for 7 × 6 = 42 against
   the previous 5 × 6 = 30, and wants a per-ghosted-joint cost to justify it.
2. Did the morphology-descriptor ablation ever run (§8)?
3. Does the design loop stay training-free (§9.1)?
4. Is the object resampled per evaluation (§9.3)?

## 13. Evidence

Numbers above come from `docs/analysis.md` — all 24,576 seed-3 designs at 3 cm
tolerance against the epoch-16,600 checkpoint (14% of training), mean 1.893 / 10
goals. Layout matters and bulk does not: mount separation is an inverted U
peaking at 4–5 cm against a 4 cm object, fingertip reach an inverted U peaking at
14.5–16 cm, both ~0.7 goals of swing and independent (r = −0.016); every
link-size measure is flat. Both optima are object-relative, which argues for
resampling objects and against hard-coding a length scale.

Two caveats carried forward. Everything there describes one undertrained policy,
and its own open list expects the 4f/5f collapse may not survive a later
checkpoint. And that file was deleted from the active branch along with the rest
of `docs/`, so the findings are reproduced here because the source is gone.
