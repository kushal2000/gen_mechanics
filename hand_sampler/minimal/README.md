# hand_sampler.minimal

A small, fully-enumerable design space of multi-fingered hands, plus a browser
viewer and a PNG renderer.

Rewritten from `gen_mechanics/hand_sampler/params.py` (branch
`2026-08-18-analyze_embodiments`), stripped down to a space where every
parameter is discrete and every rule is deliberate. Nothing here simulates — it
produces kinematics, joint roles and geometry; physics is somebody else's job.

Standalone: numpy, trimesh, viser, matplotlib. No Isaac Sim, no URDF build, no
yourdfpy.

Runs in the repo venv like the rest of `hand_sampler` — numpy, trimesh, viser
and matplotlib are all it needs, and it imports no simulator.

```bash
PY=.venv_isaacsim/bin/python
```

Not a drop-in replacement for `hand_sampler/params.py`: it emits no URDF and no
`RobotSpec`, so nothing on the Isaac path reads it yet. See `__init__.py` for
what a swap would additionally require.

Interactive viewer on http://localhost:8080 :

```bash
$PY hand_sampler/minimal/viewer.py
```

Still PNGs, no browser needed:

```bash
$PY hand_sampler/minimal/preview.py --seed 0 --count 6 --flex 55 --out preview.png
```

Design-space size and what would shrink it:

```bash
$PY hand_sampler/minimal/space_size.py
```

What the sampler favours, dimension by dimension:

```bash
$PY hand_sampler/minimal/audit_sampling.py --count 4000
```

---

# The design space

Each rule below carries the reason it exists, because most of them were chosen
to keep the space small and physically sensible rather than because the geometry
demanded it.

## Palm — fixed

A **25 × 60 × 60 mm** cuboid, thin in x like a human palm. Origin at the centre
of the **wrist face**, so an arm attaches at the origin and the palm occupies
z ∈ [0, 60] mm.

```
+x  palm surface   (the side fingers close toward)   <- large face
-x  back of hand                                     <- large face
±y  sides                                            <- thin, host fingers
+z  fingertip end                                    <- thin, hosts a finger
-z  WRIST                                            <- thin, arm attaches here
```

A cuboid has four thin faces. **Three host fingers; the fourth (−z) is where the
end-effector attaches**, the way a wrist does. Nothing about the palm is
sampled — it is fixed so that comparisons across designs are not confounded by
hand size.

## Fingers — 2 or 3, at fixed positions

Finger bases sit at the **centre** of their thin face. Position is not sampled;
only *which* faces are used.

```
+z  at (0,   0, 60) mm
+y  at (0,  30, 30) mm
-y  at (0, -30, 30) mm
```

Fingers are sampled **independently** — no constraint ties one finger's design to
another's. This was considered and explicitly rejected: constraining fingers to be
identical, or to a thumb-plus-identical-fingers pattern, would shrink the space by
6,000–150,000,000× (see `space_size.py`), but it forecloses layouts we want to
keep open.

## Mount angle — one, in-plane

`splay` ∈ **{−30°, 0°, +30°}**, a rotation about the palm normal (+x).

The requirement was that **every finger starts flat, parallel to the palm, at
zero flexion**. That holds because all three finger-face normals already lie in
the palm plane, and a rotation about the palm normal maps that plane to itself.
So splay changes which direction a finger sticks out *without ever tilting it out
of plane*.

**`splay` shares its axis with the base MCP AA joint** — both rotate about the
palm normal. Mounting splay is the fixed part of that motion; AA is the actuated
part. Verified: splay +30° and MCP AA +30° produce identical fingertip positions.
Flexion is the only thing that lifts a finger out of the palm plane, which is what
makes it flexion.

Roll (about the finger's own long axis) is **not sampled**. It is pinned by
requiring positive FE to curl the finger toward the palm surface.

## Links — fixed total length, quantised

Every finger is **exactly 100 mm** long, split into **1–3 links**, in **10 mm**
quanta, with a **30 mm minimum** per link.

```
1 link  : 1 split    (10)
2 links : 5 splits   (3+7, 4+6, 5+5, 6+4, 7+3)
3 links : 3 splits   (3+3+4, 3+4+3, 4+3+3)
```

Holding total length constant means finger *proportion* varies without reach
varying, so a design cannot win simply by being longer.

The 30 mm minimum exists for a geometric reason: capsule cylinder length is
`L − 2r`, and at `r` = 10 mm a 20 mm link would have **zero shaft** and render as
a bare ball. 30 mm leaves 10 mm of shaft.

Note the 3-link case is nearly determined — three links at a 30 mm floor consume
90 mm of the 100 mm budget, so all 3-link fingers are essentially evenly
segmented. If varied 3-link proportions matter, the lever is
`TOTAL_FINGER_LENGTH`, not `MIN_LINK_LENGTH`.

Capsule radius is fixed at **10 mm**.

## Joints — three locations, two DOFs each

Locations are **MCP, PIP, DIP** — one per link, proximal first. Each may carry
**FE** (flexion/extension), **AA** (abduction/adduction), or both, so a finger has
1–6 joints.

**The base joint (MCP) must always have FE.** A finger that cannot flex at the
palm cannot oppose anything.

Limits are **fixed**, not sampled: **FE [−10°, +120°]**, **AA [−20°, +20°]**.

Per finger, local frame: link along local **+x**, FE about local **+z**, AA about
local **+y** — perpendicular by construction. FE and AA at the same location are
coincident, i.e. a 2-DOF knuckle.

## Joint coupling — three roles

Every joint has exactly **one** role. A joint cannot be passive *and* coupled, and
cannot sit in two rigid pairs — those combinations are not physically meaningful.

| role | control | what it is |
|---|---|---|
| `independent` | its own actuator input | an ordinary driven joint |
| `passive` | none | a spring-loaded joint that returns to rest; only external force moves it |
| `rigid` | shares one input with a partner | gears or a linkage; no external force can separate them |

**Rigid pairs join adjacent locations only** — MCP↔PIP or PIP↔DIP. MCP↔DIP is not
allowed. They *may* cross DOF type: MCP FE coupled to PIP AA is legal. Same-location
pairs (MCP FE with MCP AA) are not, since those locations are not adjacent.

**"1:1" is in normalised range, not raw angle.** A joint at fraction *t* through
its travel drives its partner to fraction *t* of theirs. FE spans 130° and AA only
40°, so an identity map would either clip FE or barely move AA. Driving a coupled
MCP FE to +120° puts PIP AA at +20°, its own maximum.

> **Consequence, accepted deliberately.** Because the map is affine over the two
> ranges, a coupled input at 0 does **not** put both joints at 0. An AA driver at 0
> sits at fraction 0.5 of its symmetric range, so an FE partner goes to +55°. Coupled
> fingers therefore rest in a pose set by the range mismatch rather than one anyone
> chose. The alternative — anchoring the map at zero so 0→0, piecewise-linear — was
> considered and rejected in favour of keeping the mapping a single linear relation.

### Rules on roles

* **MCP FE may never be fully passive**, though it may be rigidly coupled. A finger
  whose base flexion is passive cannot actively flex at all, which would hollow out
  the rule that put that joint there. A side effect is that **every finger has at
  least one actuator**.
* **At most 3 passive joints** per finger.
* **A joint location may not have both its FE and AA passive.** A knuckle with both
  DOFs spring-loaded is a floating ball joint — two DOFs that nothing commands and
  nothing holds in any particular direction. One passive DOF at a location is a
  compliant axis; two is a hole in the finger. Locations carrying only one DOF are
  unaffected, so an AA-only PIP may be passive.
* **At most 3 actuators** per finger. A finger may carry up to 6 joints, but the
  extras must earn their place by being rigidly coupled or passive:

  ```
  n_actuators = n_joints - n_rigid_pairs - n_passive
  ```

  The cap bites only at 4+ joints and is always satisfiable — the tightest case is
  6 joints with no rigid pairs, needing exactly 3 passive against 3 available
  locations.

### Passive joints rest at mid-range

Not at zero. Passive AA sits at **0** (symmetric range) but passive FE sits at
**+55°**, the midpoint of [−10, 120]. A return spring on a real finger holds it
crooked, not flat.

> This means the "flat in the palm plane at rest" property holds only for fingers
> with no passive FE *and* no coupling driving an FE partner off zero.

Spring stiffness is derived per joint as
`PASSIVE_FULL_DEFLECTION_TORQUE / range` — 0.2 N·m to reach full travel, roughly
2 N at the tip of a 100 mm finger. A placeholder: nothing here simulates, so it is
metadata for whoever builds the physics.

---

# Size

```
PER-FINGER FACTORS
    kinematics (lengths x dofs)           86     6.4 bits
    x couplings (avg per kin)           15.7     4.0 bits
    x splay values                         3     1.6 bits
    = distinct finger designs          4,047    12.0 bits

TOTAL HANDS                       66,331,746,450
    2-finger                          49,134,627
    3-finger                      66,282,611,823   <- >99.9%
```

Fully discrete, so these are exact rather than "times a continuum".

**Coupling is not the dominant factor** — 4.0 bits per finger against kinematics'
6.4. The size comes from the *exponent*: three independently sampled fingers cube
the per-finger count, so shrinking any per-finger factor by k shrinks the total by
roughly k³.

`space_size.py` prints the remaining reduction levers. The largest are capping
links at 2 (~235×) and fixing splay (~125×); coupling restrictions are small
change (1.5–4×).

---

# Sampling

Legality and likelihood are separate. `couplings_of` enumerates the exhaustive
**legal** set; the sampler deliberately does not draw uniformly from it.

**Coupling is reweighted away from passivity.** Uniform over the legal set makes
`passive` an independent label on every unmatched joint, so the implied count is
Binomial(k, ½). Measured that way, 40% of joints came out passive and 17% of
fingers had no actuators at all. The sampler instead draws the *number* of passive
joints uniformly over the counts actually available, which encodes no prior on how
much passivity a finger should have. Resulting mix is roughly **43% independent,
38% rigid, 19% passive** — a sample statistic, stable to about a point across
seeds.

Because the one-passive-per-location rule makes the legal counts non-contiguous,
`sample_coupling_for` enumerates the legal passive subsets per matching and draws
a size from those available — rejection sampling would have reweighted toward
whichever subsets are easy to hit.

**Finger count is balanced, not proportional.** `n_fingers` is uniform over (2, 3),
giving ~50/50. Proportional would be absurd: 3-finger hands outnumber 2-finger ones
by ~1,300×, so uniform-over-designs would make the population >99.9% three-finger.

### Known bias, left in deliberately

`audit_sampling.py` shows the sampler leans **simple** on every axis at once:

```
mean joints per finger    sampled ~2.84  of space 4.44
1-joint fingers                     260x over-represented
6-joint fingers                      12x under-represented
fingers with a single actuator                    ~45%
```

This comes from two compounding choices: link count is uniform over 1–3 when the
space is 0.2 / 13.2 / 86.7%, and DOF patterns are uniform *within* a link count
where a 3-link finger has 18 patterns against 2 for a 1-link. Neither is
unreasonable alone; together they push the joint-count marginal far from both
uniform and proportional.

**TODO — revisit.** The narrow fix is to sample `n_joints` explicitly: uniform link
count, then uniform joint count within that level's achievable range (k to 2k),
then uniform over DOF patterns with that count.

---

# Conventions and gotchas

* Capsules span `p0 → p1` **tip to tip**: the cylinder is shortened by `2r` so the
  hemispherical caps land *on* the joints and adjacent links abut exactly at the
  shared joint centre. (`trimesh.creation.capsule` is centred on the origin, not
  starting at it — placing `p0` there shifts every link back by half its length.)
* `forward_kinematics` takes **per-joint** angles — the physical truth. Coupling
  semantics live in `expand_inputs`, which maps **actuator** inputs to joint angles.
  Anything driving the hand should go through `expand_inputs`, not straight to FK.
* For a rigid pair the **proximal** joint is the driver, so its input spans that
  joint's range.
* `GRASP_DIR` sets which way fingers close. Flip it to `(-1,0,0)` to close toward
  the back of the hand.
* `preview.py` draws links as fixed-width lines, so it understates radius.
  `viewer.py` uses real capsules.

---

# What changed from the original gen_mechanics sampler

| | gen_mechanics | here |
|---|---|---|
| palm | 3 sampled extents | fixed |
| finger bases | continuous (u,v) on 3 faces + wrist keep-out | face centres |
| fingers | 2–5, with 5 ghosted slots | 2–3, no ghosting |
| joint locations | CMC_FE/CMC_AA/MCP_FE/MCP_AA/PIP/DIP | MCP/PIP/DIP |
| AA availability | MCP and CMC only | every location |
| joint sets | prefix of a fixed 6-step ladder (5 options) | any non-empty subset, MCP needs FE |
| lengths | 4 independent continuous; total reach 111–260 mm | constant 100 mm, quantised |
| joint limits | MCP_AA half-range sampled 2–25° | fixed |
| radii | per-finger scale over a fixed tier table | fixed |
| mount angles | roll (sampled then overwritten) + tilt + azimuth | one in-plane `splay` |
| coupling | none — every joint independent | 3 roles |

---

# Checkpoints

`checkpoints/` holds frozen states with their recorded outputs.

* `v1_kinematic` — before joint coupling. 86 finger designs, 658,244 hands.
* `v2_coupling` — this state. 4,047 finger designs, 66,331,746,450 hands.

Restore with `cp checkpoints/<name>/*.py .`
