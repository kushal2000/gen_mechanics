# Proposal: co-design of hand hardware and morphology-conditioned control

Status: draft proposal. Feasibility probed, not started.
Rendered version: <https://claude.ai/code/artifact/7a47100d-8b7b-44a3-be4f-13e0d781fda3>

Hand hardware and control policy are jointly determined but almost always
studied separately. Fixing a hand and learning a controller measures a joint
optimum you never got to choose.

This proposes an outer co-design loop around a morphology-conditioned policy:
propose a population of hands, train one shared policy across all of them,
evaluate, update the proposal distribution, repeat. Two things distinguish it
from existing co-design work:

1. the policy is **warm-started across outer iterations** rather than frozen or
   retrained per design, which changes what the search is measuring;
2. the fitness signal is **held-out generalization**, not training-distribution
   performance.

Two target results:

- **R1** — the co-design loop makes control training *faster*: start with
  embodiments simple enough to earn reward signal early, then add complexity.
- **R2** — co-designing for *generalization* selects different hardware than
  co-designing for nominal performance, including hardware that crosses the
  sim-to-real gap better.

Measurements below tagged **[verified]** were taken against the working
simulator and are reproducible from the probe scripts in this repo. Items tagged
**[open]** are untested and stated as questions.

## 1. The gap

Co-design research is overwhelmingly about **locomotion**. Manipulation
co-design exists but concentrates on two-finger grippers and grasping — shape
optimization for a static grasp, not dexterous in-hand behavior with a learned
policy.

There is a specific technical reason, and it is the obstacle to clear:

> **Locomotion has a morphology-agnostic reward. Manipulation does not.**
> "Track this commanded velocity" is well-defined for a biped, a quadruped or a
> wheeled robot — the same reward transfers across every morphology unchanged.
> Reward anything about *how* a hand accomplishes a task and you have encoded a
> preference over morphologies, which an optimizer will exploit.

Our task is unusually well-positioned against this. The pose-reaching reward
decomposes into seven terms, and the ones that define the *task* are stated in
terms of object state alone:

| Term | Morphology-dependent? | Role |
|---|---|---|
| keypoint distance to goal | no — object state only | task |
| lifting reward + bonus | no — object height | task |
| goal-reached bonus | no | task |
| arm action penalty | no — arm held fixed | regularizer |
| fingertip approach | **yes** — sums over fingertips | shaping |
| hand action penalty | **yes** — sums over hand DoFs | shaping |

The two morphology-dependent terms have measured magnitude (see
`methodology.md` §2): 5 fingertips → 4 costs ~20% of the approach reward, and
22 hand DoF → 16 gives ~27% less penalty.

For the fixed two-hand comparison this repo currently runs, that bias is
documentable and deliberately left in place. **For co-design it is not** — an
optimizer *will* discover that more fingers earn more approach reward.

> **Action item.** Both shaping terms must be dropped or count-normalized before
> the first co-design run. Dropping is cleaner: shaping exists to make a single
> morphology trainable, and a warm-started conditioned policy needs it less.
> This is the one place where co-design deliberately departs from
> `methodology.md` §2, and the departure must be recorded there when it happens.

## 2. Where this sits in the literature

| Work | Domain | Morphology search | Policy across designs |
|---|---|---|---|
| Self-assembling morphologies (Pathak et al. 2019) | locomotion | emerges from link/unlink actions | shared modular policy, dynamic graph net |
| RoboGrammar (Zhao et al. 2020) | locomotion | graph grammar + graph heuristic search | MPC, no learned policy |
| DERL (Gupta et al. 2021) | locomotion | evolutionary | retrained per morphology |
| LocoFormer | locomotion | none — 100k procedural robots, fixed pool | one Transformer-XL, unified joint space |
| **House of Dextra** | **dexterous hands** | grammar + graph heuristic search | cross-embodied policy, action masking |

(Venues for the last two are unconfirmed; cite before use.)

### The delta against House of Dextra

It is the closest work: grammar-based hand generation, Graph Heuristic Search
over designs, a morphology-conditioned policy with action masking, and design →
fabricate → deploy in under 24 hours. Three things are open.

- **The policy is not trained across search iterations.** Every design is scored
  under a policy that never adapts to it, so designs far from the policy's
  training distribution are penalized for the *policy's* ignorance rather than
  their own limitations.
- **Graph Heuristic Search carries a point estimate, not a posterior.** It
  learns a function from designs to best-achievable performance and expands
  promising branches. With no calibrated uncertainty a single low observation
  can write off a region — and combined with a frozen policy, that low
  observation is exactly the one most likely to be wrong.
- **Evaluation is nominal.** In-hand rotation, grasping and flipping on YCB
  objects, with no held-out physics, geometry, randomization or
  goal-distribution axes.

## 3. Approach

An outer loop around a shared policy. Prim paths are fixed when a simulation
instance is built, so the loop **restarts the simulator each iteration** — which
turns out to be a feature: the morphology pool can then change arbitrarily
between iterations, including its topology classes.

1. **Propose a population.** Thompson-sample a surrogate over embodiment
   parameters — draw N posterior samples, take each draw's argmax.
2. **Build the scene.** Continuous parameters vary per environment; topology
   classes get one articulation view each (§5).
3. **Train the conditioned policy**, warm-started from the previous iteration,
   so a new design inherits competence from related designs rather than starting
   cold.
4. **Evaluate on held-out axes** — the four in `genmech/eval/suites.py`, plus
   morphology perturbation (§4).
5. **Update the surrogate**, fit to *within-iteration rankings*, not absolute
   scores.
6. Restart the simulator, repeat.

### Why Thompson sampling

The usual argument is exploration, but the more immediate reason is that we need
a **batch**. We are not choosing one design, we are filling a population for the
next training run. EI and UCB are inherently sequential and need diversification
machinery to produce batches; Thompson sampling gives a batch for free, with
diversity falling out of posterior uncertainty.

It also has the right exploration property for this failure mode. Thompson
sampling exhibits *probability matching*: it samples each region at a rate equal
to the posterior probability that the region contains the optimum. A region with
one noisy low observation retains high variance and keeps getting probed. A
point-estimate heuristic writes it off.

**Honest limit.** Thompson sampling does *not* fix the attribution error, it
only slows over-commitment. If the policy is persistently bad in a region, TS
keeps collecting low scores and eventually concludes the region is bad —
correctly given the data, wrongly given the true objective. Warm-starting the
policy addresses the cause; TS buys time while it works.

### Non-stationarity, and the cheap fix

With a warm-started policy, performance is a function of (morphology, policy
state) and the policy improves. The same design scores differently at iteration
1 and iteration 10. Standard Bayesian optimization assumes a fixed objective, so
early observations become systematically pessimistic and drag their regions down
— a rich-get-richer dynamic that can collapse the search onto the first family
it finds.

The fix costs nothing: a population is always trained *together*, so
within-iteration comparisons are apples-to-apples. Fit the surrogate to
within-iteration **rankings** rather than absolute values. Rank is invariant to
global policy improvement, which is exactly the nuisance parameter to quotient
out.

## 4. The two target results

### R1 — co-design makes control learn faster

Start with embodiments simple enough to get reward signal early, then add
complexity once the simple body saturates.

There is a strong precedent: Bongard, *Morphological change in machines
accelerates the evolution of robust behavior* (PNAS 2011) — robots that changed
body form *while* learning to walk, "like tadpoles becoming frogs," acquired
more robust behavior faster than fixed-morphology controls. That was
evolutionary and fifteen years ago; the RL version, with complexity scheduled by
the co-design loop rather than hand-specified, is open.

Our reward already contains the ladder: **lift → transport → reorient**, with
monotonically increasing DoF requirements. A two-finger gripper collects lifting
and keypoint reward almost immediately; in-hand reorientation is what actually
needs five fingers. Mechanically, "add complexity" is unmasking action
dimensions — the same masking a cross-embodied policy already needs.

**Design against the known failure.** The classic curriculum failure is
*scaffolding becoming a crutch*: the policy finds a competent local optimum with
the simple body and never explores the new degrees of freedom — a five-finger
hand operated as a two-finger gripper. Instrument this directly by measuring
whether added joints carry action variance once unmasked, not merely whether
reward improved. Reward can rise for unrelated reasons.

A curriculum also makes the surrogate's non-stationarity *deliberate and
observable*: the complexity level is known at each iteration, so the surrogate
can condition on it rather than fighting an invisible drift.

### R2 — co-design for generalization, and for the sim-to-real gap

The sharp question: **does co-designing for generalization select a different
hand than co-designing for nominal performance?** If the argmax is the same
under both, the framing collapses and one should simply optimize nominal. If
they differ, that is a real and non-obvious result — and testing it requires a
held-out evaluation suite, which is the part almost nobody has and which this
repo already has (`genmech/eval/suites.py`, 21 conditions across four axes).

There is a second, underexploited payoff. A surrogate over morphology gives
*local sensitivity* for free — not just where the peak is, but how sharp it is.
Sharpness in morphology space is **manufacturing tolerance**: a design on a
narrow peak needs exact fabrication, one on a broad plateau tolerates
millimetres of print error and actuator variation. Preferring flat optima over
sharp ones is a sim-to-real robustness criterion expressed natively in the space
already being searched, with no hardware in the loop.

This suggests a fifth evaluation axis alongside the four we have: **perturb the
morphology itself** and measure retention, exactly as we do for object physics
and geometry.

**Caveat to state, not assume.** Robustness to *the randomization axes we chose*
is not robustness to the actual reality gap. Our axes cover latency, sensor
noise, disturbance wrenches, friction and mass — not unmodeled contact dynamics,
actuator nonlinearity, tendon stretch or backlash. There is also a perverse
direction: a design robust *because it is insensitive* — over-stiff,
over-constrained, low-dexterity — could win on retention while being worse at
the task. Report nominal and retention jointly; never optimize retention alone.

## 5. Feasibility, measured

The usual blocker is simulator support for heterogeneous embodiments. The
constraint is in the wrapper, not the physics: an Isaac Lab `Articulation` reads
`num_joints` from `root_physx_view.shared_metatype.dof_count` — one scalar for
an entire view — and every buffer is `(num_instances, num_joints)`. So one view
cannot span robots with different joint counts. Both consequences were probed.

**Geometry varies per environment, one view — 5.23 cm spread. [verified]**
`genmech/tools/probe_heterogeneous_envs.py`. Two hands differing only in finger
length, spawned into alternating environments. Palm-to-fingertip distance
0.3022 m and 0.3545 m — the variants did not collapse onto a shared template.
Link lengths, masses, joint axes and gains are therefore free to vary per
environment.

**Different topologies coexist in one scene — 29 and 23 joints. [verified]**
`genmech/tools/probe_multi_articulation.py`. A 29-joint and a 23-joint robot in
one scene over disjoint environment subsets, two articulation views. The
uniformity requirement is per-*view*, not per-scene — so variable finger count
is feasible, at one view per topology class.

**Per-robot asset conversion — 1.2 s, steady state. [verified]**
URDF → USD → physics bake. This is the real bottleneck at scale: 24,576 unique
embodiments is ~15.3 hours of conversion *per outer iteration*, which dominates
the RL cost.

**Scale variation via USD overrides — untested. [open, highest leverage]**
If morphology variation can be authored as transform/attribute overrides on one
shared template rather than N distinct USDs, 15 hours becomes seconds. The
question is whether PhysX picks up scaled collision geometry and recomputed
inertias through overrides. A ~30-minute probe, and it decides whether
unique-per-environment morphology is free or expensive.

The two mechanisms compose well for hands specifically. Hands have few natural
topology classes with a rich continuous space inside each, so most variation
lands in the cheap mechanism. That factorization is more favourable than for
legged robots, where the biped/quadruped/wheeled split is coarse and 100k robots
are needed to cover the space.

Both mechanisms require per-environment physics parsing rather than instanced
cloning (`replicate_physics=False`, `clone_in_fabric=False`) — the same path the
existing per-environment object pool already takes, so the cost is known and
already paid.

## 6. What already exists in this repo

The proposal is not starting from zero. The two-hand comparison has produced
most of the substrate.

- **A morphology parameterization** — `genmech/robots/spec.py`. A frozen,
  self-validating `RobotSpec` carrying joint names and order, gains, home pose,
  palm and fingertip geometry and self-collision adjacency, with observation and
  action dimensions derived from it rather than pinned.
- **Programmatic robot generation** — `genmech/tools/build_allegro_urdf.py`. The
  Allegro robot's URDF is generated, not hand-authored: spliced onto the same
  iiwa14 chain SHARPA uses, mirrored to left-handed with a full reflection of
  geometry and meshes, verified against the source to one micron.
- **A generalization evaluation harness** — `genmech/eval/`. 21 held-out
  conditions across four axes, a batched runner, and retention — each hand
  scored against its own nominal — as the headline metric.
- **A methodology for not fooling yourself** — `docs/methodology.md`. The arm is
  held identical by construction and asserted across the registry; training is
  epoch-bound rather than walltime-bound so the faster-simulating hand cannot
  buy extra gradient steps; the reward is byte-identical across hands.

## 7. Risks

**Measurement noise is the binding risk — SEM ±0.5 goals at 512 envs.
[verified]** If a morphology change moves performance by less than this, no
amount of compute rescues the search. Two claimed hand differences have already
been retracted in this project as sampling noise. The training runs currently in
flight give the first real read on whether *any* hand difference is resolvable
in this setup.

Ordered by how much they would reshape the work:

1. **Effect size below noise.** Everything downstream presumes morphology
   measurably affects performance. Mitigation: measure it first, on the two-hand
   comparison already running, before building the loop.
2. **Credit assignment between morphology and policy.** Ranking designs by
   performance at a fixed budget selects for *easy to optimize early*, not
   *ultimately better*. Warm-starting attacks the cause; evaluating after a short
   adaptation budget rather than zero-shot is the complementary defense.
3. **Reward hacking through morphology.** The two count-scaling shaping terms
   (§1) are an explicit invitation. Drop or normalize them before the first
   co-design run, not after.
4. **Asset generation cost.** 15.3 hours per iteration at 24,576 unique designs.
   Resolved or not by the USD-override probe (§5).
5. **Search collapse onto one family.** Non-stationarity plus a point estimate
   produces rich-get-richer. Rank-based surrogate fitting and Thompson sampling
   are the designed defenses; whether they suffice is empirical.

## 8. Sequence

Ordered so each step can invalidate the next cheaply, rather than by ambition.

| Step | Question | Kills the proposal if |
|---|---|---|
| **0** · Effect size | Do two real hands differ measurably at all? | differences sit inside ±0.5 goals |
| **1** · Override probe | Can morphology be authored without reconversion? | nothing — it sets the scale, not the viability |
| **2** · Conditioning | Does one policy across a morphology distribution match a specialist on a held-out design? | conditioned policy is far below specialists |
| **3** · Signal | Does performance vary systematically with morphology parameters? | variation is noise-dominated |
| **4** · Curriculum (R1) | Does scheduling complexity beat training the target morphology directly? | no speedup, or scaffolding becomes a crutch |
| **5** · Divergence (R2) | Does the generalization-optimal hand differ from the nominal-optimal hand? | they coincide — then optimize nominal and stop |

Steps 0 and 2 are the honest gates. If hands do not differ measurably,
optimizing over them is moot; if a conditioned policy cannot match specialists,
the amortization that makes the whole loop affordable does not exist.

## References

- Pathak, Lu, Darrell, Isola, Efros. *Learning to Control Self-Assembling
  Morphologies: A Study of Generalization via Modularity.* NeurIPS 2019.
- Kurin, Igl, Rocktäschel, Boehmer, Whiteson. *My Body is a Cage: the Role of
  Morphology in Graph-Based Incompatible Control.* ICLR 2021.
- Zhao, Xu, Luo, Wang, Matusik. *RoboGrammar: Graph Grammar for Terrain-Optimized
  Robot Design.* SIGGRAPH Asia 2020.
- Gupta, Savarese, Ganguli, Fei-Fei. *Embodied Intelligence via Learning and
  Evolution.* Nature Communications 2021.
- Bongard. *Morphological change in machines accelerates the evolution of robust
  behavior.* PNAS 2011.
- *LocoFormer.* One Transformer-XL across 100k procedural robots spanning
  bipeds, quadrupeds and wheeled variants (URL and venue to be confirmed).
- *House of Dextra.* <https://an-axolotl.github.io/HouseofDextra/>
