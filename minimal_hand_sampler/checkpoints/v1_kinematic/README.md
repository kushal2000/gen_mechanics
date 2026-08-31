# minimal_hand_sampler

A small, fully-enumerable hand design space. Rewritten from
`gen_mechanics/hand_sampler/params.py` (branch
`2026-08-18-analyze_embodiments`, the later of the two — Aug 18, +25 files,
`params.py` 29.5k → 32.3k).

Standalone: numpy, trimesh, viser, matplotlib. No Isaac Sim, no URDF build,
no yourdfpy.

```bash
PY=/opt/homebrew/Caskroom/miniforge/base/envs/handdim/bin/python
```

Interactive viewer on http://localhost:8080 :

```bash
$PY viewer.py
```

Still PNGs, no browser needed:

```bash
$PY preview.py --seed 0 --count 6 --flex 55 --out preview.png
```

## The space

**Palm — fixed.** 25 × 60 × 60 mm cuboid. Origin at the centre of the wrist
face, so an arm attaches at the origin and the palm occupies z ∈ [0, 60] mm.

```
+x  palm surface  (fingers close toward this)   <- large face
-x  back of hand                                <- large face
±y  sides                                       <- thin, host fingers
+z  fingertip end                               <- thin, hosts a finger
-z  WRIST                                       <- thin, arm attaches
```

**Fingers — 2 or 3**, mounted at the *centre* of the 3 non-wrist thin faces.
Base positions are fixed; only which faces are used varies.

**Mount angle — one, in-plane.** `splay` ∈ [−45°, +45°], a rotation about the
palm normal (+x). It turns the finger WITHIN the palm plane, so it picks which
direction the finger sticks out without ever tilting it out of plane. At zero
flexion every finger lies flat in the palm's mid-plane (x = 0), parallel to the
two large faces.

`splay` shares its axis with the base MCP AA joint — mounting splay is the fixed
part of that motion, AA the actuated part. Flexion is the only thing that lifts a
finger out of the palm plane.

**Joints.** Three locations — MCP, PIP, DIP — each with FE, AA, or both. MCP
must include FE. Max 6 joints per finger.

**Limits — fixed.** FE [−10°, +120°], AA [−20°, +20°].

**Links.** Total finger length constant at **100 mm**, split into 10 mm quanta,
minimum 30 mm per link, 1–3 links. Capsule radius fixed at 10 mm.

## Size

```
link partitions   1 link: 1     2 links: 5     3 links: 3
dof patterns      1 link: 2     2 links: 6     3 links: 18
finger designs                                        86
TOTAL discrete hands                             658,244
plus splay per finger (continuous)
```

**The 3-link case is nearly determined.** Three links at a 30 mm minimum use
90 mm of the 100 mm budget, so the only splits left are 3+3+4 cm and its two
permutations. If you want richer 3-link fingers, the lever is
`TOTAL_FINGER_LENGTH`, not `MIN_LINK_LENGTH` — at 120 mm the count goes back up.

## What changed from the original sampler

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
| mount angles | roll (sampled then overwritten) + tilt + tilt azimuth | one in-plane `splay` |

## Conventions worth knowing

Per finger, local frame: link along local **+x**, FE about local **+z**, AA
about local **+y** — perpendicular by construction. `forward_kinematics`
applies FE then AA at each location; both are coincident at one joint centre,
as in the original.

`GRASP_DIR` in `sampler.py` sets which way fingers close. Flip it to `(-1,0,0)`
to close toward the back of the hand.

## Two things to look at

**Fingers are now 1.67× the palm width** (100 mm against 60 mm), where a human
hand is nearer 0.8×. Opposed ±y bases sit 60 mm apart with 20 mm-thick fingers.
Deliberate if you want long fingers on a compact palm; worth a look if not.

**`preview.py` draws links as fixed-width lines, not true-radius capsules**, so
it understates thickness. `viewer.py` uses real trimesh capsules.
