# Multi-embodiment simulation: what it costs, and what it does not

Everything here is measured on this cluster (RTX 6000 Ada, Isaac Sim 5.1.0.0,
Isaac Lab 2.3.2) with the repo's production asset pipeline, not a simplified
probe. Reproduce with the tools named in each section.

The question this branch set out to answer: **can we simulate many different
robot embodiments at once, and what does it cost?**

Short answer: **diversity is free.** It costs nothing per step, and the setup
cost it carries on the URDF-conversion path disappears entirely when assets are
authored directly — including at one distinct embodiment per environment.

The one thing that cost real time was not physics but bookkeeping: which
environment holds which asset. §4 has that story, and it is the part most likely
to bite again.

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

### Objects: equivalent assets, and the acceptance test that mattered

The task's 1,200 objects (6 types × 100 × box/capsule) are each ONE rigid body
with two analytic shapes and a closed-form mass/inertia. Authored vs converted:

* mass, diagonal inertia, centre of mass agree to **~1e-8 relative**
* dropped 30 cm and settled over 400 steps, resting poses agree to **~2e-7 m**
* composed collider geometry — position and half-extents in the object's own
  frame — matches **exactly**, for both shapes

`genmech.tools.compare_object_assets`, `genmech.tools.compare_object_physics`.

**None of that was sufficient, and it is worth being precise about why.** With
all of it green the pretrained policy still scored **3.00 goals/episode against
the converted path's 5.07**. The assets were identical; they were in the wrong
environments. `_build_object_scale_tensor` gives env *i* the scale of pool entry
`i % n_pool` by walking `find_matching_prim_paths`, which returns **numeric**
order; the authoring pass walked `sorted(env_prim_paths)`, which is
**lexicographic**. On 512 envs those disagree in **510**. Nearly every env held
an object its own `object_scales` observation did not describe, and since
`reset_utils` sizes the goal keypoints from that same tensor, the goal geometry
was wrong too.

The end-to-end number is what caught it:

| | goals/10 | lift | complete |
|---|---|---|---|
| converted (control) | 5.07 ± 0.18 | 77% | 33% |
| authored, ordering bug | 3.00 ± 0.17 | 71% | 17% |
| authored, fixed | **4.93 ± 0.18** | 76% | 31% |

The fix removes the second derivation rather than correcting it: authoring
records the env→pool map and `_build_object_scale_tensor` consumes it, so the
two cannot desynchronise again. Per-env mass, inertia, `object_scales` and asset
index are now bit-identical across all 512 envs.

**Three lessons, each one a habit rather than a fact:**

* **Asset equality is not pipeline equality.** Every check compared *an* authored
  object to *a* converted object. None asked whether env *i* held the right one.
  A faster asset path has two halves to verify, and the plumbing half is the one
  with no natural reference to diff against.
* **A green check can be vacuous.** With 600 pool entries and 512 envs every env
  holds a unique asset, so "asset_index → mass is single-valued" passes no matter
  how wrong the assignment is. Two checks inside the failing run read green for
  exactly this reason. `check_object_identity` now prints when the test cannot
  fail; the regression uses a 48-entry pool where it bites.
* **Do not ship two fixes in one run.** The ordering change was credited with a
  0.00 → 2.92 recovery that the kinematic goal-marker fix in the same run had
  actually earned. That conflation is what let a change in the wrong direction
  look like progress.

**One trap worth recording.** The env converts objects with
`replace_cylinders_with_capsules=True`, which reads a URDF cylinder's `length` as
the CYLINDRICAL SECTION and adds a hemisphere at each end — total extent
`length + 2r`, up to +90% for the head. Authoring a plain `Cylinder` instead of a
`Capsule` produced objects that settled up to **0.53 mm** differently while still
passing a 1 mm tolerance. It was caught only because the box-only cases sat at
1e-8 and made 5e-04 look anomalous. Tolerances chosen without a known-exact
reference hide exactly this class of error.

---

## 5. The multi-embodiment env

`genmech/tasks/pose_reach/env_multi.py`, `GenMech-PoseReachMulti-Direct-v0`

Sections 1–4 establish that a population *can* be simulated. This is the task env
that does it: one distinct hand per environment, one articulation view, one
policy, with the procedural objects and goals unchanged.

### Two env classes, not one flag

`PoseReachEnv` stays exactly what it was. The multi-embodiment env is a separate
class in a separate file that subclasses it and overrides two hooks
(`_resolve_spec`, `_post_init_hook`).

The split is not stylistic. The single-embodiment env is what the pretrained
policy, the SHARPA parity golden files and the running training jobs are defined
against, and every `if population is not None` added to it is a branch that can
regress that baseline. Everything genuinely shared — task, reward, reset, action
pipeline, object pipeline — already lives in `utils/` and is imported by both, so
the two cannot drift apart on the things that matter.

One consequence worth stating: `robot_spec` does **not** select the robot in the
multi env. The population supplies the specs, and a fixed hand and a generated
one do not share a joint set (29 vs 37), so honouring `robot_spec` would build
the observation and action spaces for a robot the scene does not contain.

### Fingertip padding: how designs with different finger counts share a scene

The cached population is uniformly five *slots* but not uniformly five *fingers*:
ghosting leaves 2 active tips on 51 of 64 designs, 3 on 11, and 4 on 2. The
observation width came from `spec.num_fingertips`, so those designs could not
share a layout — and therefore could not share a policy or a scene.

They now share a padded axis. `RobotSpec` carries `fingertip_slot_names`,
`_active` and `_offsets`; when they are empty everything falls back to the active
fingertips, so **a fixed hand is unchanged by construction** — SHARPA stays
5-of-5, Allegro 4-of-4, and the 140-dim observation the pretrained checkpoint
expects is untouched.

This works only because of a property of ghosting worth recording: **it removes a
finger's actuation and geometry, not its links.** Every generated design carries
all five distal links at the same body indices (verified across the 64-hand
population: 0 designs missing any slot), so the padded index set is a template
constant and only the mask varies per design.

The mask is applied at ONE point — where `_curr_fingertip_distances` is produced.
Zeroing there makes every downstream reduction inert at once: the reward sums
deltas (0), termination takes a max against 1.5 m (0 never trips), the running
minimum stays 0, and the observation reads a constant. Masking at each consumer
instead would leave whichever one gets forgotten reading the pose of a finger
that is not there.

### The morphology descriptor: why proprioception is not enough

A cross-embodied policy commands per-joint position targets on a hand it has
never seen. Joint angles and fingertip positions describe the mechanism's STATE
and nothing about its STRUCTURE. Two designs can present identical joint angles
and identical normalized limits while their fingers point in completely different
directions, because all of that lives in the mount transform — which never
entered the observation. Normalization makes it worse: a `joint_pos` of 0.5 is
1° of travel on a ±2° abduction joint and 12.5° on a ±25° one, and the sampler
varies exactly that range.

`utils/morphology.py` emits a fixed-width **143-dim** descriptor, constant per env
(computed once at scene build, indexed per env, never recomputed). Per finger
slot, ghosted slots included so the width cannot depend on finger count:

| block | dims | note |
|---|---|---|
| mount position | 3 | palm frame |
| mount orientation | 6 | 6D rotation representation |
| link lengths | 4 | mc, pp, mp, dp |
| link radii | 4 | mc, pp, mp, dp |
| fingertip pad offset | 3 | the distal cap, where contact happens |
| per-joint enabled mask | 6 | a locked joint is still in the action vector |
| AA half-range | 1 | the one limit the sampler varies |
| active flag | 1 | |
| | **28 × 5 = 140** | plus palm extents (3) → **143** |

Three choices in there are load-bearing:

* **The mount is a pose, not the sampled parameters.** `face`, `u_frac`,
  `v_frac`, `roll`, `tilt` and `tilt_azimuth` only mean something after
  `mount_on_face()` composes them into a transform. Emitting the composed
  palm-frame pose is the same information without spending network capacity
  relearning that arithmetic.
* **6D rotation, not Euler or quaternion.** Mount roll is sampled `U(0, 2π)`, so
  it visits every wrap point. Euler wraps and quaternions double-cover; both put
  a discontinuity in the middle of the sampled distribution.
* **Four radii, not one.** `radius_scale` is a single sampled knob, but it
  multiplies per-tier nominals spanning 2.6× (19.3 / 10.2 / 8.9 / 7.3 mm), and
  `dp` is the radius the fingertip actually grasps with. One degree of freedom,
  four distinct physical numbers.

Deliberately excluded: per-joint axes and origins (template constants here — the
FE/AA perpendicularity is carried entirely by fixed `ROLL_FE_TO_AA` segments, so
mount orientation determines them), and gains, armature and densities (fixed per
tier across the population, so they distinguish nothing). Per-joint axes are the
extension point if one policy ever has to span the fixed *and* generated
families, since they are what would describe SHARPA or Allegro.

The existing fields are untouched — `fingertip_pos_rel_palm` keeps its meaning
and position, and the descriptor is appended.

### What is checked, and why those checks

`tests/test_multi_embodiment_env.py` asserts, against the live sim rather than
the config:

* one articulation view holding every env — a second view means the joint
  template broke and designs stopped sharing `dof_count`
* joint limits **differ** across envs, so a collapsed pool cannot pass
* each env's fingertip mask matches the design it was assigned
* each env's descriptor matches its design: envs sharing a design share a
  descriptor, envs with different designs do not
* ghosted slots read exactly zero in both fingertip fields
* the descriptor does not move across a reset and 10 steps

Plus `_verify_robot_design_assignment`, which reads joint limits back out of
PhysX at startup and confirms envs sharing a design share limits. That check
exists because the identical assumption about the object pool, left unchecked,
gave 510 of 512 envs the wrong asset (§4). The rule is now explicit: **an
assignment made by someone else's iteration order gets verified against the
simulator, not trusted.**

---

## 6. Practical guidance

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

## 7. Open

* Author the generated hand (not just objects) — needed for large-k sweeps, and
  it must reproduce masses, drive gains, self-collision filters and the arm
  reference before it is trusted.
* ~~Decide whether `assets.author_object_usds` should default ON.~~ **Done — it
  now defaults ON.** The policy holds its score through it (5.01 ± 0.09 both
  paths at 2048 envs), a 120-step rollout is bit-identical, and per-env asset
  identity is bit-exact across 512 envs. That last check is the one that
  mattered: the policy eval alone passed while an env→pool assignment bug cost
  5.07 → 3.00 (§4). **Faster assets that change the physics are a silent
  regression, not a win** — the acceptance test is the policy's score, not the
  clock.
* Prototype grouped `replicate_physics` per design block.
* The sampler rejects ~93% of draws on self-collision, stable across seeds. An
  analytic capsule-capsule gate inside `params.validate()` would avoid a URDF
  write, a load and a mesh penetration check per rejection.
