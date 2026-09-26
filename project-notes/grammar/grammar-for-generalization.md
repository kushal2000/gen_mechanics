# What "evolve to generalize" asks of the grammar

Written 2026-09-26 10:50-11:20 (coordinator). Re-read of the brief: README ("which hand's policy retains the most performance when the world shifts"; follow-on: co-design as a training curriculum, and co-design for generalization and sim-to-real), the investigation note §2 (bilevel: train a controller on one distribution, optimize hardware for performance under shifts in objects, physics, sensing, disturbances, tasks; outputs: a hand optimized for generalization and a co-design algorithm; three separate claims: curriculum, controllability, generalization), and the design note (useful, physically plausible hands without unnecessary complexity; performance(hand, current policy) confound).

## 1. Goals, restated as demands on the grammar

| Goal from the brief | What it demands of the grammar (as opposed to the process or the controller) |
|---|---|
| The evolving population trains a shared controller that should work on hands we care about (SHARPA, real hands, the eventual physical build) | The grammar's sampling support must contain those hands, or be dense near them; the population's structural variety is the controller's curriculum, so the grammar must span it deliberately, not incidentally |
| The evolved hand must retain performance under shifts it was not selected on | The grammar must be able to express robustness-conferring structure: redundant digits, opposition geometry, palm articulation, couplings that reduce control burden; it must not price these out by construction; whether they are selected is the process's job (evaluation under shifts) |
| Sim-to-real for evolved designs | Realizability constraints belong in the grammar's parameter table (module dimensions, travel, radius, mount spacing) and in derived geometry, so that everything the search proposes could be built; the design note's "physically constrained option" |
| Economical capability, not minimal fingers | The grammar reports structural cost; it does not encode a preference |
| Interpretable search under a 40-generation-scale budget | Locality, reversibility, size-neutral operators (already measured: E1, E2, E7) |

## 2. Where the current grammar stands against these demands

- Support: no real hand is inside the sampling support; the named reasons are gridded limits/lengths/axes and the straight-rod digit convention. E11 (running) quantifies how much widening brings them in and what it costs in bits per joint.
- Robustness features: redundant digits, opposition, palm articulation and couplings are all expressible and reachable (E3); nothing in the grammar penalises them; the cost term is a process choice.
- Realizability: gridded lengths (15-80 mm), radii, mount fractions and palm-joint travel exist; torque, actuator size and clearance do not. The old sampler's validator had mount spacing and capsule clearance rules; the new geometry overlay has cell containment and adjacency but no clearance check across non-adjacent bodies yet.
- Curriculum: the population is whatever selection keeps; today nothing guarantees structural variety persists (founder dominance), which is a process problem that directly damages the curriculum goal.

## 3. What the transfer literature says (lit-review-morphology-distribution-for-generalization.md)

Ranked by strength of evidence, as the review tags them:
1. Diversity beats similarity, verified twice in different domains: URMA (all quadrupeds except the target beats the two nearest) and DexFormer (25 -> 50 -> 100 hand morphologies gives a monotonic zero-shot gain, 79 -> 84%).
2. Distance from the training support predicts failure sharply: AnyBody's cross-category extrapolation is near 0%; GenLoco degrades to about 2x the training range then fails; LocoFormer names the real-robot features its 100k-robot family lacks (anhedral hip angles, offset axes) as a residual real-world gap. Our situation is the same shape: real hands sit outside our support on limits, lengths, axes and mid-chain bends.
3. Structural variety is a separate axis from parametric range: MetaMorph transfers to unseen dynamics but drops on unseen kinematics; GET-Zero needed a graph-aware architecture for structural changes. Widening continuous ranges does not create mid-chain bends; that needs a primitive.
4. Evolved populations help learning (DERL's Baldwin effect) but narrow under selection unless guarded (MetaMorph's anti rich-gets-richer balancing). No paper isolates evolved vs sampled populations at equal diversity.
5. Wide, crude randomisation substituted for accurate dynamics in sim-to-real (Peng; ADR). Coverage width matters more than geometric fidelity, at least for dynamics.
6. Anthropomorphic shape was the worst-transferring family where tested (House of Dextra: radial 3-finger 15/17 vs anthropomorphic 3/17 on real objects). One data point, but it argues against building human-likeness into the grammar.

## 4. Recommendation for a grammar that serves "evolve to generalize"

Grammar (representation and sampling):
1. **Support must contain the target hands.** Move joint limits, link lengths and axes from choice sets and grids to bounded continuous ranges (the 15-degree and 5-mm grids keep locality for mutation as a step size, not as the support), and add the rest-bend primitive (a per-phalanx continuation rotation and lateral offset) so real digits are expressible. E11 quantifies which relaxations unblock which hands and the bits-per-joint cost.
2. **Structural variety is a first-class dimension, not a by-product.** Keep digit count, palm bodies, palm joints, couplings, prismatic modules and branching all expressible, and make the sampling distribution for immigrants span them evenly (stratified over digit count x palm joints x couplings), because the controller's transfer along the structural axis is what fails first.
3. **Do not encode anthropomorphism.** The grammar already has no thumb or finger roles; keep it that way. Opposition and staggered mounts remain expressible through mount geometry.
4. **Geometry can stay simple.** Capsules and derived palm cells are consistent with the finding that width of support matters more than fidelity; realizability constraints (segment length floors, radius, mount spacing, palm-joint travel) belong in the parameter table.

Process (flagged for the RL owners; the grammar cannot do these):
5. Guard population diversity explicitly (immigrants from the stratified prior, crowding or an archive over structural bins), because the population is the curriculum and selection narrows it.
6. Evaluate and select under the shifts the brief names (held-out objects, physics, disturbances), separating training, selection-validation and untouched test distributions as the investigation note already prescribes.
7. Spend evaluation budget on more distinct designs rather than more samples per design (DexFormer, URMA).

Open: whether evolving the population beats sampling it at equal diversity is untested anywhere; it is the project's own claim 1 and needs the A/B the investigation note describes.


## 5. Support widening, measured (E11: `experiments/E11_support_widening/e11_report.md`)

Cumulative relaxation levels applied to the coverage check for every available real hand (SHARPA was reported unavailable by the script; issue I16):

| Level | Relaxation added | Hands in support (of 10) |
|---|---|---|
| L0 | none | 0 |
| L1 | rest bend (mid-chain rotation/offset allowed) | 0, but every hand's remaining items drop by one: rest bend is a prerequisite for all of them |
| L2 | + continuous joint limits within the grammar's global range (-45 to 110 deg) | 3: Allegro, Wuji, xhand |
| L3-L6 | + 1 mm lengths, 5 deg axes, continuous couplings, continuous lengths/axes | still 3 |
| L7 | + limit and length ranges widened 1.5x | 4: + LEAP |
| L8 | + fixed joints inside digits, unbounded children per body | 5: + ORCA |

Never in support: Barrett, D'Claw, Tesollo (joint limits such as (-180, 0) deg, outside even the widened range; some are sign-convention equivalents of ranges we do allow), Ability and Inspire (blocked only by our check that a coupled joint's limits equal the affine image of its source's; real URDFs declare tighter dependent limits). Design-space cost: continuous limits add no grid cost; 1 mm lengths 6.0 bits per joint (vs 3.8); 5 deg axes 11.4 bits (vs 8.3). Generated designs stay in support at every level.

Reading: the cheap, high-value changes are (1) the rest-bend primitive, (2) continuous limits inside a wider, sign-normalised range, (3) accepting dependent limits that lie within the source's image. Finer axis grids and continuous lengths buy nothing for these hands and cost the most bits, so the 15 deg / 5 mm grids can stay as mutation steps while limits and the bend become continuous parameters.

## 6. Recommendation, final

1. Grammar 0.5 = grammar 0.4 (exact-inverse operator pairs, insertion prior) plus: a rest-bend field on every phalanx (continuation rpy and lateral offset, continuous, small-step operator of one grid step, zero by default); joint limits as continuous bounded pairs with sign normalisation (axis flip with negated limits treated as equal, also in the canonical hash); dependent-joint limits accepted when contained in the source's image. Lengths and axes keep their grids.
2. Immigrant prior stratified over structural bins (digit count x palm joints x couplings x prismatic), so the population's structural variety is maintained by the process rather than hoped for.
3. No anthropomorphic assumptions; geometry stays capsules and derived palm cells; realizability limits live in the parameter table.
4. Process items (RL owners): diversity guard, selection under held-out shifts with a separated test set, breadth over depth in evaluation, and the evolved-vs-sampled population A/B that would test the project's claim 1.

Priority order for implementation: rest bend (unblocks every hand), then continuous sign-normalised limits, then the coupling containment rule, then the stratified immigrant prior.

## 7. Analysis record (this loop)
- E11: support widening levels vs real hands and bits-per-joint cost.
- Literature: which properties of a morphology training distribution drive controller generalization (coverage vs diversity vs realism vs scale).
- Then: a recommendation on the support (continuous vs grid, rest bend), on which structural variety the population should be forced to keep (curriculum), and on realizability constraints to add.
