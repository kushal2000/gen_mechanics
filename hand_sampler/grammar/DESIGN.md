# hand_sampler — design

A grammar over hand designs and a set of mutations on it, for the evolution step
of the co-design loop.

Status: **design plus implementation of the offline half.** The genotype,
kinematics, validator, sampler, mutation operators and viewers are built and
tested. The integration layer (URDF build, collision gate, policy features) is
deliberately not — see §10.

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
**fixed envelope** (`MAX_FINGERS` × `MAX_JOINTS_PER_FINGER` = 6 × 6 = 36, against
the old 5 × 6 = 30). Every design pays for the envelope whether it uses it or
not. That number is the one place the simulator reaches back into the genotype,
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

It leaves one thing for `build.py` to decide. A URDF can carry the axis two ways:
`axis = axis_of(theta)` with zero origin rpy, or SHARPA's convention — which
`params.py` follows — of `axis = [0 0 1]` with the orientation in the origin rpy.
The kinematics are identical; they differ only in whether the **child link's own
frame rolls about its long axis** by `theta`. With capsule links that is
unobservable, since a capsule is rotationally symmetric about its axis and its
oriented bounding box is a square-section box invariant under that roll. It
becomes observable the moment a link is not symmetric, so the convention should
be picked deliberately rather than inherited, and §8's "hinge axis in the
parent's frame" should name the same one.

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
| floor | 20 mm = 2 × radius, and a fabrication limit |
| ceiling | 80 mm |

Total finger length is not fixed. **One joint per link** — every segment is at
least the floor, so no two joints share a point.

An earlier version allowed zero-length segments, expressing a multi-DOF knuckle
as coincident joints. Dropped for the same reason the floor exists: coincident
axes need a gimbal, whereas two axes 20 mm apart are ordinary revolutes in
series. It cost a true MCP-style knuckle and bought no zero-length special case
anywhere downstream.

**Depth is coupled to length, by geometry**: *k* joints need ≥ *k* × 20 mm of
reach, and a link must reach 40 mm before it can split. Measured over an
unselected walk — 1 joint / 52 mm, 2 / 65, 3 / 79, 4 / 102, 5 / 123, 6 / 134.
Since the measured reach optimum is 145–160 mm and one link cannot exceed 80 mm,
**selecting for the reach optimum selects for at least two joints per finger**;
the two effects cannot be cleanly separated here.

### Fixed

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

| n | split | merge | add_finger | remove_finger | P(up) |
|---|---|---|---|---|---|
| 4 | 83% | 84% | 100% | 62% | 55.8% |
| 6 | 96% | 97% | 74% | 92% | 47.2% |
| 10 | 78% | 100% | 34% | 84% | 38.3% |
| 14 | 77% | 100% | 4% | 34% | 38.2% |

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

Everything below is deliberately absent so the space stays small enough to reason
about. Each entry is a way to buy design complexity when the search needs it, and
each is reversible — the genotype carries the field or the shape already, so
re-enabling is closer to a flag than a rewrite. Roughly ordered by value against
cost.

**1. Fingers on the two large palm faces.** Currently `+z` and `±y` only. `+x` —
the surface fingers close toward — gives an opposition post rising from the palm,
the most thumb-like arrangement this space can express, and it measured the
closest fingertip approach of any pair. Excluded because a finger growing out of
the gripping surface is awkward to build and to mount an arm behind. One tuple in
`FINGER_FACES`; the crossing logic already handles arbitrary faces.

**2. Joint types beyond independent revolute.** The largest missing capability,
and three separable steps:

* *rigid / mimic coupling* — two joints driven as one, so a finger has more
  joints than motors;
* *fully passive, spring-loaded* — no input at all, a return spring and contact
  decide the angle;
* *differential coupling* — one motor driving several joints through a
  differential, which is how most underactuated hands actually work.

`minimal/` has a worked version of the first two, including adjacency rules,
mid-range rest poses and per-joint stiffness. The cost is that mutation over
coupled topologies is much harder: a mutation removing a joint must decide what
happens to its partner. The cost of *not* having it is worth naming — real
anthropomorphic hands are heavily underactuated, and that is precisely how they
buy dexterity per motor. A claim of "matches a market hand at equal motor count"
against an underactuated baseline is uphill while every joint here has its own
motor.

**3. Branching chains.** A finger splitting into two beyond some joint. The
operator most likely to produce genuinely novel morphologies rather than
variations on hands we can already picture — the tree representation already
permits it, since `Chain` carries a list of children. Deferred because it roughly
doubles validator work.

**4. Off-perpendicular joint axes (`phi`).** Pinned at π/2. An oblique hinge is a
real mechanism — the link sweeps a cone of half-angle `phi` rather than a flat
fan — and it is a genuine single revolute, but it reads as a two-axis joint
unless you know what you are looking at. Held back until the space needs the
complexity. One line in `perturb_axis`, plus relaxing the validator's equality
back to a range.

**5. Coincident joints.** Two joints sharing a point, which is how an MCP knuckle
combining flexion and abduction is normally modelled, and what `params.py` did
with zero-length virtual links. Dropped for the same reason `MIN_LINK_LENGTH`
exists: coincident axes need a gimbal, where two axes 20 mm apart are ordinary
revolutes in series. Re-adding means allowing zero-length segments again and
restoring the special cases they carry through the builder and the renderers.

**6. Palm thickness.** Seeded and never mutated — the dimension geometry cares
least about, where width and length move separation and reach directly. One entry
in `MUTABLE_PALM_DIMS`.

**7. Per-design joint limits.** Currently a global ±90°. Range of motion is a
real design variable; there is simply no evidence yet that searching over it
pays.

**8. Link radius.** Fixed at 10 mm, and the *least* promising entry here despite
being trivial to enable: it is the one parameter measured to do nothing, at
Spearman −0.005 across a 2× range. Listed for completeness, not as a candidate.

**9. Actuator properties** — gear ratio, reflected inertia, torque limits. Every
joint currently gets the same motor. The node feature schema (§8) already has
room, and this is where reflected inertia would enter if the search should care
about it.

**10. A larger envelope, or a non-box palm.** `MAX_FINGERS` × `MAX_JOINTS_PER_FINGER`
is a hard simulator cost paid by every design (§2), and the palm is a box because
face frames are then trivial. Both are relaxable; both are expensive.

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

1. What articulation envelope is affordable, against the current 5 × 6 = 30?
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
