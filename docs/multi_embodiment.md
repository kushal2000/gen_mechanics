# Multi-embodiment simulation: what it costs, and what it does not

Everything here is measured on this cluster (RTX 6000 Ada, Isaac Sim 5.1.0.0,
Isaac Lab 2.3.2) with the repo's production asset pipeline, not a simplified
probe. Reproduce with the tools named in each section.

The question this branch set out to answer: **can we simulate many different
robot embodiments at once, and what does it cost?**

Short answer: **diversity is free.** It costs nothing per step, and the setup
cost it carries on the URDF-conversion path disappears entirely when assets are
authored directly — including at one distinct embodiment per environment.

---

## 1. Different topologies coexist in one scene

`genmech/tools/multi_embodiment_demo.py`

An Isaac Lab `Articulation` reads `num_joints` from
`root_physx_view.shared_metatype.dof_count` — one scalar per view — so robots
with different joint counts cannot share a view. They can share a **scene**, over
disjoint environment subsets:

| view | envs | joints | every instance actuates |
|---|---|---|---|
| `sharpa_iiwa14` | 3 | 29 | 3/3 |
| `allegro_iiwa14` | 3 | 23 | 3/3 |
| `gen_sharpa_like` | 11 | 37 | 11/11 |

13 distinct embodiments, 3 views, 17 envs, one `sim.step()`. The generated
view's 11 envs showed a **14.58 cm fingertip-reach spread**, measured from body
poses after spawn — proof the envs received distinct geometry rather than one
template cloned.

**Ghosting is what makes this scale.** Every generated design is padded to the
same 37-joint template, so an arbitrary number of *sampled* hands costs exactly
one view. View count tracks hand FAMILIES, not population size — 17 sampled
hands ran in a single view.

Videos: `videos/multi_embodiment.mp4` (static home pose, orbiting camera),
`videos/multi_embodiment_fingers.mp4` (arm fixed, fingers closing).

---

## 2. Diversity costs nothing per step

`experiments/bench_k_scaling.sub`, 24,576 envs, generated hands:

| distinct designs | steps/s | env-steps/s |
|---|---|---|
| 8 | 43.0 | 1,057,374 |
| 64 | 42.7 | 1,049,616 |
| 256 | 43.1 | 1,058,817 |
| 512 | 43.4 | 1,065,483 |

**Flat across two decades of k** — 1.6% spread. Confirmed independently on the
authored path at the extreme, holding robot and env count fixed and varying only
the number of distinct designs:

| distinct | setup | steps/s | env-steps/s |
|---|---|---|---|
| 64 | 368.4 s | 14.3 | 350,860 |
| 24,576 | 366.5 s | 14.2 | 348,108 |

**Identical to 0.8%.** Even when every one of 24,576 envs holds a different
robot — nothing shared, nothing replicated — there is no per-step penalty.

### What DOES change step rate

Not diversity: **contact geometry and articulation depth**.

* SHARPA (29 joints, 34 convex-hull meshes) vs a generated hand (37 joints,
  analytic capsules) at 24,576 envs: **31.6 vs 37.1 steps/s** — the capsule hand
  is *faster* despite 8 more joints.
* One sampled hand (2.23 active fingers on average) vs `gen_sharpa_like`
  (5 fingers), same 37-joint template: **44.5 vs 37.1 steps/s**. Ghosted fingers
  carry no collision geometry, so contact-pair count drives this, not DoF.
* A 43-joint SERIAL chain runs at 350k env-steps/s where a 37-joint hand+arm
  (depth ~13) runs at 1.05M. Articulation solve tracks chain **depth**.

At 64 envs the same generated hand managed only 8,911 env-steps/s against 776k
at 24,576 — small-scale benchmarks measure per-step overhead, not physics, and
should not be extrapolated.

---

## 3. Setup is the real cost

`experiments/bench_scene_build.sub`

Scene construction dominates: **441 s for one design at 24,576 envs against ~8 s
of timed physics**, and `sim.reset()` is 77% of it. Fitted across four env counts
and k up to 512:

```
setup_seconds  ~=  1.1*k  +  0.017*n  +  0.00018*k*n
                   ^^^^^     ^^^^^^^      ^^^^^^^^^^
                   URDF->USD  per-env     USD composition,
                   conversion (PhysX)     one pass per design
```

Predicts within ~20% everywhere measured (k=512: predicted 2360 s, measured
1929 s).

### What does NOT help, all measured

| lever | effect |
|---|---|
| USD instancing (`make_instanceable`) | none (69.2 s vs 68.9 s) |
| PhysX GPU buffer sizing | none; *smaller* buffers are worse |
| disabling self-collision | 2% |
| disabling the arm's 8 collision meshes | none |
| stripping materials/shaders/visual prims | 6.5% (halves prim count, 400→190/env) |
| pre-materialising env prims before spawn | none (slightly worse) |
| `replicate_physics=True` | **7× on reset** — unusable with per-env distinct assets |

The last line is the crux, and it is a [known upstream
limitation](https://github.com/isaac-sim/IsaacLab/issues/4434): heterogeneous
assets force `replicate_physics=False`, and Isaac Lab's docs state the resulting
"slowdowns in setting up and parsing the scene" plainly. An open proposal for
`replicate_physics="grouped"` would replicate within groups of identical envs —
exactly the k designs × n/k envs shape a morphology sweep has. The underlying
`cloner.replicate_physics(source, targets)` API already takes a source and a
target list, so grouped replication looks implementable without upstream changes.

---

## 4. Direct USD authoring removes the setup cost

`genmech/tools/bench_authored_sps.py`, `genmech/tasks/pose_reach/utils/author_objects.py`

Kit's `UrdfConverter` takes ~876 ms per generated hand, **~90% of it importing
the arm's 16 STL meshes** — the same arm in every design. Nothing about a
capsule hand or a box-and-capsule object needs a mesh importer.

Authoring the prims directly with Sdf specs inside a `ChangeBlock`:

| | convert path | authored |
|---|---|---|
| 24,576 distinct robots, asset prep | ~8.3 hours | **207 s** (8.42 ms each) |
| PhysX reset | — | 155 s (6.31 ms each) |
| **total setup** | ~30 hours (extrapolated) | **362 s** |

Linear in k, verified at 4,096 and 24,576. All instances resolve into one
articulation view with joints that track their targets.

### Objects: verified physically equivalent

The task's 1,200 objects (6 types × 100 × box/capsule) are each ONE rigid body
with two analytic shapes and a closed-form mass/inertia. Authored vs converted:

* mass, diagonal inertia, centre of mass agree to **~1e-8 relative**
* dropped 30 cm and settled over 400 steps, resting poses agree to **~2e-7 m**

`genmech.tools.compare_object_assets`, `genmech.tools.compare_object_physics`.

**One trap worth recording.** The env converts objects with
`replace_cylinders_with_capsules=True`, which reads a URDF cylinder's `length` as
the CYLINDRICAL SECTION and adds a hemisphere at each end — total extent
`length + 2r`, up to +90% for the head. Authoring a plain `Cylinder` instead of a
`Capsule` produced objects that settled up to **0.53 mm** differently while still
passing a 1 mm tolerance. It was caught only because the box-only cases sat at
1e-8 and made 5e-04 look anomalous. Tolerances chosen without a known-exact
reference hide exactly this class of error.

---

## 5. Practical guidance

**Which asset path you use decides everything below.** The two have different
economics, and advice written for one is wrong for the other.

### On the authored path: one embodiment per env is free

Measured, not inferred — same robot, same env count, only the number of distinct
designs varying:

| distinct designs | setup | steps/s | env-steps/s |
|---|---|---|---|
| 64 | 368.4 s | 14.3 | 350,860 |
| 24,576 | 366.5 s | 14.2 | 348,108 |

There is no `k` term left. Authoring is ~8 ms per robot whether the robots differ
or not, PhysX reset does not care, and the step rate is identical. **k = n is a
perfectly good operating point** — arguably the natural one, since it gives the
search loop one evaluation per design with no bookkeeping about which envs share
an embodiment.

The remaining consideration is statistical, not computational: at k = n each
design gets a single rollout, so its score carries the full variance of one
object draw and one goal sequence. That is a choice about estimator noise —
answerable by longer episodes, more goals per episode, or repeating designs
across a second scene — not a reason the simulator makes k = n expensive. It
does not.

### On the convert path: keep k modest

Setup grows as `0.00018*k*n`, so distinct designs are genuinely expensive:
64 designs at 4,096 envs costs 122 s, 512 designs at 24,576 envs costs 2,286 s,
and one design per env extrapolates to tens of hours. Here — and only here — the
`k designs x n/k envs` shape is worth arranging deliberately.

### Independent of path

* **Build once, amortise.** Setup is minutes; make the rollout long enough that
  it does not dominate.
* **Prefer analytic colliders.** Capsule hands are faster to build *and* faster
  to step than mesh hands, and they author directly.
* **Small-scale benchmarks mislead.** At 64 envs, per-step overhead dominates and
  the ranking of two robots inverted relative to 24,576 envs.
* **Watch articulation depth, not joint count.** A 43-joint serial chain runs 3x
  slower than a 37-joint hand+arm, because the solve tracks chain depth.

## 6. Open

* Author the generated hand (not just objects) — needed for large-k sweeps, and
  it must reproduce masses, drive gains, self-collision filters and the arm
  reference before it is trusted.
* Wire object authoring into `scene_utils` behind `assets.author_object_usds`,
  then confirm the pretrained policy holds its goals/episode. **Faster assets
  that change the physics are a silent regression, not a win** — the acceptance
  test is the policy's score, not the clock.
* Prototype grouped `replicate_physics` per design block.
* The sampler rejects ~93% of draws on self-collision, stable across seeds. An
  analytic capsule-capsule gate inside `params.validate()` would avoid a URDF
  write, a load and a mesh penetration check per rejection.
