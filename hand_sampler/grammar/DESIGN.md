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
Mount  := face, (u, v), direction(alpha, beta)
Chain  := Joint, link(length), [Chain]
Joint  := axis(theta, phi)
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
* **phi** ∈ (0, π/2] is the polar angle from the link. π/2 is
  perpendicular-to-bone, the conventional assumption; φ→0 is a roll joint.

`phi` exists so *"joint axes are perpendicular to the links"* is a testable
choice rather than an unstated one. Seeds draw π/2; mutation may leave it.

**Joint limits are symmetric ±90° for every joint.** Anatomical asymmetric ranges
stop meaning anything once the axis is a continuum — there is no principled way
to interpolate an asymmetric flexion range into a symmetric abduction one.
Practical range of motion comes from **contact** instead: a finger that bends
backwards hits the palm and stops, in simulation, for free.

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

`(face, u, v, alpha, beta)`.

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

Orientation is a pointing direction `(alpha, beta)` — two angles, not three. The
third would be roll about the finger's own axis, and that is **not a free
parameter** for two independent reasons: it is gauge (rotating the mount by *r*
while subtracting *r* from the first joint's theta gives an identical hand), and
the previous design space derived it from the rest of the geometry.

**Mount orientation stays structural rather than folding into the first joint's
axis.** A base abduction joint reaches the same configurations, but it costs a
motor where mount tilt is free. On a performance-versus-motors axis those are not
the same design.

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

Seven, and the count is a design goal. **An operator reachable by chaining others
earns nothing** — it adds surface for a bug and a second place the same rule can
drift. Taking more iterations to get somewhere is an acceptable price.

### Structural — ±1 joint each

| operator | effect |
|---|---|
| `add_node` | split a link, or start a new finger from the palm |
| `remove_node` | detach a joint, merging its link; a finger's last joint takes the finger with it |

Two, not four: attaching to a joint and attaching to the palm are the same
operation in a tree. Keeping them apart also built a wall, and that wall is what
froze finger count in the previous design space — permanently, for any population
descended from it, because no operator touched it.

*Unit steps* keep the performance-versus-motors front dense; if the only
structural move added a whole 3-joint finger, complexity would jump in threes and
the headline plot would have holes. *Exact inverses* are load-bearing for §9.3 —
measured 150/150 recovery, and per-move balance 51% at 6 joints, 42% at 10.
Balance falls further in, which is the §5 geometry pushing back rather than
operator bias.

A removed link folds into the proximal neighbour (the exact inverse of a split),
else the distal one, else the proximal merge clamped to the ceiling. The clamp is
unreachable from `add_node`, so it costs nothing in exactness — without it, a
finger whose adjacent links summed past the ceiling could not shed that joint at
all.

### Parametric

| operator | step |
|---|---|
| `perturb_axis` | ±15° in theta, or 1-in-4 in phi |
| `perturb_length` | ±1 quantum |
| `move_mount` | ±5 mm across the surface, crossing face edges |
| `perturb_direction` | ±15°, jittered in the tangent plane |
| `perturb_palm` | ±10 mm in width or length |

`move_mount` absorbs what a separate `remount` would do, since a step that
overflows a face carries onto the face across that edge. On an axis-aligned box a
face's tangent directions are its neighbours' normals, so no cube net is needed.
All three faces are reachable from any seed.

Two details it needs. **`(alpha, beta)` are preserved across an edge**, so the
world direction rotates by the angle between the normals — a finger pointing
straight out of one face points straight out of the next. Preserving the world
direction instead lays it flat along the surface it is bolted to. And when a step
overflows toward a face that hosts no finger, it **clamps** rather than refusing:
the thin axis has a 5 mm band against a 5 mm step, so refusing froze that axis
entirely.

**`perturb_axis` and `perturb_direction` are not redundant**, though they look
it. A joint axis decides which way a joint *sweeps*; a mount direction decides
which way the finger *points at rest*, and no sequence of axis changes tilts a
rest pose.

### Deferred

`branch` — chains attached to non-leaf links, giving forked fingers. The operator
most likely to produce genuinely novel morphologies. Crossover (`swap_finger`) is
free from the tree, but crossover is the specific mechanism behind bloat in the
GP literature, so add it with the joint-count instrumentation running.

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
not cover the topology direction. **Adding `add_node` to a fixed-policy loop
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
that close outscore those that do not and `add_node` is the one-step path
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

## 10. Deferred

**The integration layer**, in dependency order, and belonging with whoever owns
the simulator side:

1. `build.py` — genotype → URDF, ghosting into the envelope (§2). Needs the
   envelope number; should follow `rotations.py` for rpy and `inertia.py` for
   inertials.
2. `gates/` adaptation — the skip set must derive from the tree, not from
   template names (§7). The closed-form capsule test itself is reusable verbatim.
3. `features.py` — node/edge emission per §8.

**Couplings** (passive joints, rigid pairs). `minimal/` has a worked-out version.
Deferred because mutation over coupled topologies is much harder — a mutation
removing a joint must decide what happens to its partner. The cost is worth
naming: real anthropomorphic hands are heavily underactuated, and that is how
they buy dexterity per motor. If the claim is "matches a market hand at equal
motor count" while the baseline is underactuated and this is not, that comparison
is uphill. Known debt, not a free simplification.

**`branch`, crossover, per-design joint limits, actuator properties.** The node
schema has room for the last of these; every joint currently gets the same motor.

## 11. Open

1. What articulation envelope is affordable, against the current 5 × 6 = 30?
2. Did the morphology-descriptor ablation ever run (§8)?
3. Does the design loop stay training-free (§9.1)?
4. Is the object resampled per evaluation (§9.3)?

## 12. Evidence

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
