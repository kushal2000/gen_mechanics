# Hand grammar and evolution: working notes

Consolidated September 24, 2026, from Martin's September 23–24 notes and the accompanying technical feedback. Updated September 25 with the completed [mutation and pruning audit](#9-completed-mutation-and-pruning-audit-september-25-2026). Proposed parameter values and experimental changes remain hypotheses, not implementation decisions.

The goal is to search for useful, physically plausible hands without introducing unnecessary complexity. A central difficulty is that we measure `performance(hand, current policy)`: a design's score depends on controller experience as well as hardware. The grammar, mutation process, and selection procedure need to be evaluated with that distinction in mind.

## 1. Refine the grammar and physical constraints

The grammar may allow excessive variation in link length and joint arrangement while excluding useful palm geometry or finger opposition. Broadening it everywhere would not necessarily improve the search.

### Palm structure and finger mounting

The current palm is a rectangular box, not necessarily square. A root-based, skeleton-like representation is worth exploring in stages:

1. Allow more general finger-root positions and orientations.
2. Add rigid metacarpal-like branches or other non-box support geometry.
3. Consider articulated palm branches only if the task warrants additional motion and actuators.

A different graph representation only adds physical capability when support geometry or relative motion changes. Skeletal rendering with the original box collider would not test this proposal. Human anatomy provides useful hypotheses about opposition and load paths, but need not be the target morphology.

Preserve the fixed joint-slot interface initially. New support geometry still requires consistent collisions, mass/inertia, mounting frames, and policy features, even without additional actuators.

### Dimensions and joint travel

| Proposal | Assessment and implementation consequence |
|---|---|
| Explicit maximum total finger length | Add a bound on the sum of segment lengths. Current limits permit 6 × 80 = 480 mm per finger. Choose the bound from intended hardware and object scale. |
| Increase minimum internal link length from 15 to 25 mm | A plausible packaging assumption, pending joint/housing dimensions. Splitting into two ordinary links would require at least 50 mm, making many seed links initially unsplittable. |
| Allow a shorter final link, perhaps 15 mm | Distinguish joint-to-joint spacing from last-joint-to-tip length. Structural edits must revalidate segments whose role changes. |
| Set same-face root spacing to 25 mm | Already the current value. Keep root spacing and link length as separate parameters because they constrain different hardware dimensions. |
| Fix palm thickness | Already fixed within each lineage, but seeds mix 20 and 25 mm. A single population-wide thickness would be an additional experimental control. |
| Continue mutating palm width/length | Retain this variation. Resizing currently moves finger roots in metres because mounts use normalized coordinates, so recheck all clearances. |
| Reduce joint travel from ±90° to ±60° | Test as a restriction, not an established realism improvement. Hardware-supported, joint-specific limits are preferable: flexion may need asymmetric/wider travel and abduction narrower travel. Oblique axes need an explicit classification rule. |
| Vary capsule radius from 7 to 15 mm | Start with fixed-radius comparisons before evolving radius. Update contact geometry, clearance, packing, mass/inertia, and policy features together. Absolute radii are clearer initially than an uncertain characteristic-length scaling law. |

Keep assembly/rest offset separate from available joint travel. The current validator bounds both using the global travel limits, so narrowing those limits changes both. Home-angle mutation cannot replace sufficient travel. Moving from 90° to 100° has no universal energy cost: mechanical travel constraints require their own justification.

Check interactions between dimensions. Parallel 15 mm-radius fingers require at least 30 mm axis separation, and a capsule of that radius cannot fit within a 15 mm total tip length. State whether “length” means joint spacing, capsule-axis length, or overall extent.

### Spacing and collision validity

Current checks include same-face root spacing (25 mm), cross-face root spacing (15 mm), and capsule clearance (20 mm axis separation at the nominal 10 mm radius), plus a palm-edge margin. These have different purposes. Keep explicit packaging rules and derive geometric clearance from the actual shapes.

For two capsules, allow contact but reject meaningful overlap: require axis-segment distance `d >= r_i + r_j - epsilon`, with a documented numerical tolerance. Rest-pose finger collision checks already exist, but need the reproduced distance-symmetry and radius-consistency fixes, plus palm checks.

During motion, contacts should be handled by the simulator. Rejecting every hand whose fingers can ever touch would exclude useful designs. Sampled-configuration checks can identify obvious interference but cannot certify the full motion range.

## 2. Choose initialization deliberately and make simplification reachable

The current experiment starts from simple hands after 500 neutral mutation rounds. It therefore does not test simple-to-complex evolution from the original seeds. Those rounds are CPU sampling steps, not 500 RL-trained generations.

Simplification operators exist: links can merge, and fingers can be removed. However, removal requires a one-joint finger. Simplifying a complex finger may require several intermediate changes that reduce performance and are eliminated before deletion becomes possible. Unused hardware also has little explicit cost in the current model.

Two starting strategies deserve comparison:

- **Simple seeds:** preserve the opportunity to discover useful intermediate designs, but need sufficient controller training to reveal their potential.
- **Pre-drifted hands:** provide more immediate variety, but introduce complexity and may make pruning difficult.

Two fingers can support pinching, although that does not establish adequate reorientation. Test grasp/lift progress before concluding that the seeds need more joints. Neutral pre-drift does not evaluate intermediate usefulness. A small 128-hand, 40-round neutral pilot increased mean joints from 2.96 to 5.52 without selection, demonstrating that the mutation process itself can increase complexity.

### Selection before designs are competent

Zero task successes do not imply zero information: shaped rewards can differ. Conversely, a numerical ranking may not reliably distinguish hardware quality. Current exact ties favor lower design indices.

Give simple hands enough training, use weak or neutral selection while rankings are unreliable, and provide balanced parent opportunities. Protect some existing designs while exploring offspring, but keep population size fixed. Keeping every parent and adding a child for each would double it.

A roughly 100-hand pilot is reasonable. Comparisons to 1,024 hands must account for different per-design training exposure. Once a cost objective is defined, compare simple versus pre-drifted initialization under both task-only and cost-aware selection, with matched training/evaluation budgets.

## 3. Balance structural cost with task efficiency

The goal is economical capability, not the fewest possible fingers. Five fingers may distribute load, resist disturbances, or require less effort than three. Extra structure should be justified by what it contributes, rather than automatically treated as waste.

Keep three measurements separate initially: structural cost, task operating cost, and task capability/reliability. A simple structural baseline at fixed palm dimensions and radius is:

```text
C(hand) = number of actuated joints + total link length / length_reference
```

This charges for added actuators and excessive finger length, but cannot determine whether they are worthwhile. The reference length expresses a tradeoff, not a biological constant. If radius/palm dimensions vary, use consistent material/size costs too. Finger order and simulator padding must not affect the measure.

### Holding effort is a useful starting point

Mechanical power `torque * angular_velocity` is zero during a static hold, but an actively holding motor can still consume power and heat up. A first motor model includes copper loss:

```text
P_copper = R * I^2
motor_torque ≈ K_t * I
```

For a fixed motor and an ideal fixed-ratio transmission, this gives a holding-loss proxy proportional to squared joint torque. Friction, brakes, self-locking mechanisms, electronics, and thermal limits require additional assumptions. Mechanical efficiency is not a useful ratio at standstill because useful shaft power is zero; static current/torque and thermal limits are the relevant quantities. The peak dynamic efficiency region does not describe holding.

This gives a concrete reason that additional fingers can improve efficiency. In a toy model with identical moment arms and actuators, sharing a total load `F` equally among `n` fingers gives:

```text
sum of squared finger forces = n * (F/n)^2 = F^2/n
```

Thus an additional load-bearing finger can reduce total modeled holding loss. Real hands have unequal moment arms, friction constraints, internal grasp forces, and hardware overhead, so this is a motivating example rather than a prediction that more fingers always win.

Start with a common actuator model across hands. Measuring operating efficiency does not require simultaneously evolving motor and gear choices. Full actuator co-design remains a later extension.

### A low-compute grasp-efficiency assay

For a small set of feasible contact configurations and external object loads, solve for contact forces that balance the load while respecting friction and actuator limits. Map those forces through the finger Jacobians to joint torques, and minimize a weighted sum of squared torques as a fixed-actuator holding-loss proxy. A grasp map expresses object force/torque balance; `joint_torque = J^T * contact_force` expresses joint loading.

Compare two-, three-, and five-finger examples at matched object loads. Allow fingers to contribute zero force, and include appropriate phalanx/palm contacts rather than requiring every fingertip to participate. Vary friction and disturbance loads, and report feasibility and remaining load capacity alongside estimated loss.

This uses small static optimization problems rather than policy training. Constructing valid grasps/contact geometry is real work, but the compute requirement is modest. It estimates effort for an assumed feasible grasp, not the ability to acquire that grasp, dynamic stability, or complete manipulation efficiency. Dynamic electrical energy needs trajectories later.

### Diagnostics are not selection rules

Do not require every fingertip to meet, every finger to contact an object, or every hand to satisfy a pinch-specific heuristic. Enveloping grasps, palm contacts, caging, and fingers used only during transitions can all be useful.

Use geometric and holding-effort assays first to understand designs. They should not reject candidates or enter evolutionary fitness until their relevance to the intended tasks is established. Hard constraints belong to physical feasibility and simulation validity; behavioral preferences need evidence.

For eventual selection, compare task performance, structural cost, and operating effort as separate objectives or a declared weighted tradeoff. Report failures and use matched tasks/time limits so inactivity does not appear efficient. The objective should allow additional fingers to survive when their benefits justify their costs. Selection cannot penalize a real-world cost that the simulator and objective never represent.

## 4. Test whether policy inheritance disadvantages new hands

Parents have more controller exposure than offspring. Selection may therefore reward familiarity, particularly when larger mutations change the required control strategy. This is plausible rather than established: nearby offspring inherit useful behavior, and the next training phase includes them too.

Current selection also aggregates returns during training. A child that struggles initially but finishes well can receive a poor generation-average score.

### Parent–child adaptation diagnostic

For sampled pairs:

1. Evaluate parent and child with the inherited policy at fixed difficulty and on balanced objects.
2. Clone the checkpoint separately for each hand, provide equal additional adaptation budgets, and reevaluate.
3. Repeat a smaller comparison from a common pretrained checkpoint not adapted to the current population.

Children that consistently catch up or overtake their parents would demonstrate that initial compatibility is an incomplete assessment. Equal extra training does not erase parental history, hence the common-prior comparison. These tests measure performance after a specified adaptation budget, not optimal control.

Potential remedies include end-of-training frozen-policy scoring, a minimum training exposure before offspring can be eliminated, continued exposure to archived morphologies, and resets to the same broad pretrained policy. Such resets differ from random reinitialization. The Notion-linked morphological-pretraining paper provides an example in a different robot setting, not proof for this task.

Choose whether selection should reward immediate controller compatibility, adaptation speed, or eventual performance. These are different objectives. Policy familiarity may contribute to ancestry collapse, but truncation selection, weak early signals, tie-breaking, mutation bias, and forgetting also need consideration.

## 5. Improve mutation coverage and preserve useful diversity

### Define what balanced mutation would mean

Equal probabilities for adding and removing are not enough: applicability, rejection, candidate counts, and intermediate states determine the actual transitions. Finger removal requires merging to one joint first, whereas adding a one-joint finger is a direct move. The question is whether this creates an unintended obstacle to pruning at the complexity levels we care about.

Separate three meanings of balance: equal operator-selection probabilities, equal accepted growth/shrinkage rates at a given state, and a desired long-run distribution over hand complexity. These are not equivalent. Even a symmetric bounded random walk grows on average when started at its minimum, because further shrinkage is unavailable. Initial growth alone does not demonstrate excessive bias.

The September 25 audit below measures changes in joint/finger count and total length, including failed proposals, across 256 independent parents. It also measures offspring under the actual retry procedure and follows pruning from hands with only multijoint fingers. Growth depends on topology and boundaries. The clearest coarse-pruning restriction is singleton-only finger removal, while merging succeeds readily when applicable. These results do not justify globally reweighting growth versus shrinkage.

The repository documents historical neutral experiments in `hand_sampler/experiments/README.md:70–118`, including growth without task selection. Those reported results use an older envelope (more than five fingers and a 42-joint ceiling), so they are not measurements of the current grammar. The new audit provides revision-matched evidence beyond the earlier 128-hand pilot, but does not establish the long-run distribution of hands. Both neutral studies differ from the Notion result about ancestry collapse under co-evolution.

Biological mutation is not required to balance complexity-increasing and complexity-decreasing changes. Genetic mutation rates vary by event and context, and developmental mappings make genetic changes unlike direct edits to fingers or joints. Selection acts on the variants that occur, so proposal bias can affect outcomes. Biology motivates studying mutation and selection together, not copying an assumed 50/50 balance. For this engineering search, choose and document an exploration preference rather than claiming an unbiased or biologically correct one.

### Resolution, locality, and scale

Keep quantization initially, while separating representation resolution from mutation size. A 5 mm grid can support 5, 10, or 20 mm steps. The current 5 mm / 15° grids are baselines, not demonstrated optima. Compare finer or continuous representations at matched physical perturbation scales.

Include unchanged links in length mutations. Prefer selecting one link or a small subset over modifying every link independently: even `{-5, 0, +5}` mm choices change more locations in larger hands. Record completely unchanged outcomes separately.

Control three quantities independently:

- Parameter step size.
- Number of edited links/joints.
- Number of composed structural operations.

Begin with mostly small edits and occasional larger edits throughout evolution. A 50 mm length change is comparable to a whole seed link. Fix the axis-wrap defect before interpreting nominal angle steps as local changes.

### Multiple chances per seed

Multiple offspring explore a parent's neighborhood. Multiple scales explore near and farther alternatives. Protected family representation prevents immediate elimination of an entire family. These are separate interventions.

An illustrative pilot is eight parents with eleven offspring each: 96 designs including parents. Allocate offspring across small, medium, larger, and structural edits, using a few scale multiples such as 1×, 2×, and 4× rather than a dense linear sweep initially. More trials per family require fewer families or additional compute.

Cloning alone does not prevent family extinction under global selection. Permanent ancestry quotas can waste resources on unproductive families. Track ancestry, topology, geometry, and behavior separately, and preserve useful differences rather than assuming different founders imply different capabilities.

### Mutation count and curricula

Forty generations do not imply 40 accepted changes along a lineage. Parents can persist unchanged, mutations can fail or reverse earlier changes, and the previously examined illustrated lineage had only 12 mutation events by generation 39. Extending one link by 40 mm with 5 mm steps requires eight accepted positive edits on that link, not eight generations.

Log attempted/accepted operators, rejection and copy rates, accepted lineage changes, geometric displacement, and distance from the founder. Large proposed mutations may mostly be rejected, leaving little actual exploration.

An early structural emphasis or coarse-to-fine schedule is worth testing, but retain local edits and reverse operations throughout. Geometric adjustment may be necessary before another finger fits, and newly added fingers may need substantial changes late in a run. Start with a persistent mixture of scales before introducing annealing, and keep the feasible grammar fixed for that comparison.

## 6. Nice to have, not current investigations

- Investigate whether evolution can be biased toward repeatable hand designs, such as identical fingers or shared finger modules. Description-length and assembly-regularity objectives are deferred, not part of the current complexity study.

### Actuator and transmission co-design

Joint-specific QDD versus high-ratio actuation is a useful hypothesis, but a substantial extension. QDD at MCP flexion and compact high-ratio actuation elsewhere could be beneficial under some tasks and packaging constraints. Remote tendon-driven motors could change that tradeoff.

Torque limits alone cannot test it. Model coupled torque–speed limits, mass/placement, reflected inertia, friction, efficiency, electrical losses, and packaging. In a simple transmission model, increasing ratio reduces output speed and increases available torque, while reflected rotor inertia grows approximately with ratio squared.

Start with a small catalog of physically supported actuator/transmission options. Optional drive fields already exist in the genotype but are not consistently used by generated simulation. This requires simulator integration and a decision about how the policy receives actuator information, beyond sampler-only mutations.

## 7. Evaluate convergence and possible local optima

Run independent restarts, separating initial-population, mutation/selection, and RL randomness where practical. Similar final hands can indicate a good solution, a restrictive grammar, or shared-policy bias. Different results can reflect training noise rather than distinct optima.

Compare finalists under a common evaluation and adaptation budget, not only their own co-evolved controllers. Add an escape test using small, larger, and structural perturbations with fair adaptation, plus a compute-matched random-search baseline.

Report the supported conclusion: no improvement found within the tested neighborhood and budget. Neither repeated convergence nor a failed escape test certifies global optimality.

## 8. Immediate research priorities under a small compute budget

The near-term goal is to understand the search and task-efficiency tradeoffs without another full co-evolution experiment.

| Priority | Question | Low-compute investigation | What it can establish |
|---|---|---|---|
| 1, completed September 25 | Does current mutation favor growth or obstruct pruning? | 256 parents, stratified operator probes, actual retried offspring, and short neutral walks. Results in §9. | Topology-dependent growth and restricted direct finger removal. No evidence of universal growth in complex hands. |
| 2 | Can additional fingers reduce holding effort or improve load tolerance? | Optimize force allocation for a small set of feasible grasps under common actuator assumptions, friction constraints, and external loads. | Static effort–capability tradeoffs, not learned grasp acquisition or dynamic stability. |
| 3 | What structural cost is justified alongside task efficiency? | Compare actuator count and normalized geometry/material costs on controlled design edits and the static grasp examples. | Transparent costs and counterexamples to assuming that fewer fingers are always better. |

The mutation audit answers the immediate transition-bias question without RL training. Any follow-up should compare a specific operator change to this baseline. Repair the known geometry defects before treating validator acceptance as evidence of physical feasibility.

For the efficiency assay, start with analytically understandable load-sharing examples before implementing a full grasp-force optimizer. Fix actuator type and transmission first. A common motor model is sufficient to study holding losses; evolving actuator types is not required. Report sensitivity to contact geometry and friction instead of claiming one grasp is representative of the entire task.

Keep diagnostic geometry and force tests out of evolutionary selection initially. They can inform the task/energy model, but unvalidated pinch or fingertip-contact requirements would impose the very morphology preference the study should examine.

Defer description-length/repeatability objectives, full initialization comparisons, policy-inheritance interventions, convergence restarts, and actuator co-design. When training resources are available, the parent–child adaptation diagnostic remains a priority.

Implementation should remain within `hand_sampler` unless broader changes are authorized. Simulator, observation, and policy changes require coordination. The mutation audit below is complete. The efficiency and structural-cost investigations remain proposals.

## 9. Completed mutation and pruning audit: September 25, 2026

### Question and conclusion

Does mutation favor growth beyond the expected bias at the minimum-size boundary, and can established multijoint fingers be pruned within a short mutation budget?

The sampled neutral-aged populations retain a modest average growth tendency, but direction depends strongly on finger count and topology. Five-finger hands shrink on average. Direct finger removal is restricted to singleton fingers, while multijoint fingers require intermediate merges before removal. This supports testing a targeted coarse-pruning move, rather than globally increasing all shrinkage probabilities.

### Protocol

- 256 independent parents: 64 each after 0, 40, 160, or 500 neutral mutation attempts from simple seeds. These are separate lineages across age groups, not repeated measurements of the same hands.
- 64 calls to each of nine operators per parent: 147,456 stratified calls. Failed proposals retain the parent. An operator call includes its internal candidate search, so it is not a single elementary geometry proposal.
- 128 calls to the actual `evolve.mutate_once` per parent: 32,768 offspring calls, with up to 64 retries each. These required 37,146 underlying attempts.
- An additional 40 ordinary neutral steps per parent, plus a focused pruning follow-up described below. No task selection, policy training, or simulation.

Reported 95% intervals use 5,000 bootstrap resamples of parent-level means. Repeated trials from one hand are not treated as independent hands. Results describe the sampled seed/neutral-age distributions, not a uniform sample of all legal hands.

### Growth is conditional on the starting hand

Expected joint-count change per uniformly selected operator call, including rejection:

| Prior neutral attempts | Parents | Mean starting joints | Mean Δjoints/call [95% CI] | Mean Δjoints/retried offspring |
|---|---:|---:|---:|---:|
| 0 | 64 | 3.02 | +0.139 [+0.128, +0.151] | +0.166 |
| 40 | 64 | 5.83 | +0.028 [+0.005, +0.050] | +0.042 |
| 160 | 64 | 7.50 | +0.030 [+0.009, +0.050] | +0.031 |
| 500 | 64 | 11.47 | +0.030 [+0.009, +0.050] | +0.026 |

Pooling these sampled cohorts by finger count gives a different view:

| Fingers | Parents | Mean Δjoints/call [95% CI] |
|---|---:|---:|
| 2 | 101 | +0.131 [+0.122, +0.141] |
| 3 | 52 | +0.062 [+0.045, +0.077] |
| 4 | 57 | +0.018 [0.000, +0.035] |
| 5 | 46 | −0.065 [−0.080, −0.048] |

The nine highest-joint examples, with 20–25 joints, all had five fingers and no singleton. Split and merge succeeded, while add and remove were unavailable, giving zero estimated one-call joint drift in these particular hands. This small group does not establish balance at high complexity generally.

For a fixed parent, retrying failed proposals rescales the one-call drift without reversing its sign. The empirical retry results are consistent with this prediction. They are offspring changes, not population changes per co-evolution generation, where parents can be retained and selection intervenes.

Local drift should not be extrapolated indefinitely: the age-500 cohort gained 0.69 joints on average over the additional 40 neutral steps, with a 95% interval of [−0.06, +1.42]. That longer-horizon increase is not separately resolved from zero by this sample.

### Merging is available; direct finger deletion is restricted

All 14,080 merge calls on the 220 parents containing a multijoint finger succeeded. The remaining 36 parents contained only singleton fingers. The implementation transfers length to a neighbor when possible and clamps to the 80 mm maximum when necessary, so length overflow does not itself block a merge. Merging can nevertheless shorten reach and change task behavior.

Across the panel, 48 of 155 hands above the two-finger minimum had no singleton and therefore could not use `remove_finger`. In the age-500 cohort, this affected 29 of 54 eligible hands (53.7%). All 614 counterfactual whole-finger deletions across those 155 parents passed the current validator. The deletion restriction therefore comes from the operator, rather than a requirement of the current validator. These are clustered geometry checks, not 614 independent designs or demonstrations of useful grasping.

A three-finger hand with depths `[2, 2, 2]` illustrates the asymmetry away from the minimum: split, merge, and add can succeed, while direct removal is unavailable. At age 500, add-minus-remove contributes +0.0260 joints/call to the mean drift, compared to +0.0035 from split-minus-merge.

### Existing mutations can prune original fingers, but removal counts overstate it

The follow-up started from all 48 parents above the two-finger minimum with no singleton. Each received 64 independent walks of 40 uniform-nine single-proposal steps, including failures: 3,072 walks. Finger identities were tracked so deleting a newly added finger did not count as pruning an original one.

| Starting subset | Parents | Chance of removing ≥1 original finger within 40 steps [95% CI] |
|---|---:|---:|
| All initially blocked parents | 48 | 42.4% [36.0%, 49.0%] |
| Age-500 blocked parents | 29 | 33.8% [26.0%, 41.8%] |
| Blocked parents below five-finger cap | 32 | 48.5% [41.7%, 54.9%] |

In the below-cap subset, 92.3% of walks removed some finger, but 43.8% of all walks had removals without removing any original finger. Mean finger count still increased by 0.715. Thus, original fingers often remain present while newly added singleton fingers turn over. Retained fingers can still change joint count, geometry, and mounting, so identity retention does not imply an unchanged hand.

Original-finger removal also differs from net simplification: only 13.9% of walks across all 48 parents ended with fewer fingers. Three parents whose fingers all started with at least four joints had no original-finger removals in 192 walks. This is a small conditional observation, not evidence that pruning is impossible. Age, depth, and finger count were not independently controlled.

These are 40 mutation attempts, not 40 accepted mutations or 40 trained evolutionary generations.

### Recommendation and limits

Keep the existing local split/merge moves and do not globally rebalance growth versus shrinkage based on this audit. If easier coarse simplification is desired, compare the baseline to an occasional whole-finger pruning proposal, preserving the two-finger minimum. A multijoint deletion is a larger edit than singleton addition, so report its size and disruption explicitly rather than calling the pair balanced. No such change was implemented, and its effect on task fitness remains untested.

The audit measures current software transitions. Known validator defects, including degenerate capsule-distance behavior, limit physical interpretation. It establishes neither a stationary morphology distribution nor the effect of reward-based selection, policy familiarity, or operating cost. The existing restricted `test_operators_are_unbiased` passed (1 test, 1.54 s), but does not establish global transition balance.

### Reproduction record

Source revision: `8d1395320f7027ccd088ba5593bde50a69eecf65`, branch `codex/martin-hand-sampler`. No repository files were changed. The pre-existing untracked `.DS_Store` was left untouched.

Temporary scripts and data: `/var/folders/ph/dytz2x1j59dbtssk8tc54btw0000gn/T/evolve-mutation-audit-20260925-y9hruz1f/`. `audit.py` creates the panel, probes, neutral walks, and summaries. `pruning.py` runs the original-finger follow-up. `metadata.json` records source hashes and environment details, while `panel.json`, `probes.jsonl`, `summary.json`, `pruning.json`, and `pruning_summary.json` retain the measurements. This is temporary storage, not a permanent archive.

The base seed is `250925`. For neutral age `a` and index `i`, the initial-design seed is `250925 + 10000*a + i`, with a separate neutral-walk RNG seeded by adding `2000000000`. The follow-up uses `25092500 + 1000*parent_index + trial_index`. Other probe and walk seeds are specified in `audit.py`.

Interpreter: `/tmp/evolve-hand-audit-venv/bin/python`, Python 3.14.0, NumPy 2.4.4, SciPy 1.18.1. This existing CPU environment differs from the project's declared Python 3.11 environment and is not an Isaac compatibility test. No dependencies were installed. Run `audit.py all`, then `pruning.py`, with the repository on `PYTHONPATH` and bytecode writing disabled. Panel generation took 18.0 s, main probes 133.4 s, and the pruning follow-up 83.9 s.

Evidence and code references: [September 23 project investigation](investigation-2026-09-23.md). This remains the single consolidated working note replacing the two earlier feedback documents.

### External references for these distinctions

- [Cano et al., Mutation bias and the predictability of evolution (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10067271/): genetic mutation bias and its interaction with selection; not a model of anatomical add/remove balance.
- [Maxon, On the heating of motors](https://support.maxongroup.com/hc/en-us/articles/360004427413-On-the-heating-of-motors-in-hand-held-tools): winding losses depend on squared current. The support page blocked direct retrieval; its indexed article excerpts support this point.

## 10. Kinematic grammar and evolvability experiments (September 25–26, 2026)

This section summarises work on `martin/hand-grammar`; the detailed record is in `project-notes/grammar/` (STATE.json, LOG.md, the experiment folders, two Opus reviews, and `grammar-for-evolution.md`).

### What was built

A representation of hand kinematics as a tree of bodies and 1-DoF joints at arbitrary SE(3) placements, with affine couplings, palm bodies as a flagged connected subtree, and geometry derived rather than searched (capsules on finger bodies; nearest-spine convex cells over a shared palm hull, with overlapping cell pairs recorded for collision filtering). Real hands import faithfully: forward kinematics agrees with Pinocchio to about 1e-16 m on Allegro, LEAP, Barrett, SHARPA, Ability and Inspire, and exported URDFs reproduce the source references bit-exactly. A grammar (`GRAMMAR_VERSION` 0.3) generates hands from typed productions with a separate parameter table; derivations replay exactly and carry lineage. The benchmark (2,906 tests) uses frozen tolerances and independent references (sympy closed forms, Pinocchio, urdf_parser_py) and never skips a fixture. No real hand lies inside the sampler's support; the named reason is that real fingers have a mid-chain frame rotation or lateral offset the straight-rod digit convention cannot express (open question I11).

### What the experiments established

With joint-aligned configurations, one mutation moves fingertips by 0–4 mm (small-step operators) or 10–16 mm (structural operators, except regrow at 44 mm). Neutral drift relaxes toward the prior the growth operators sample from: 33 joints at step 400 for the full distribution, 9 when new material is drawn from an insertion distribution limited to one to three phalanges without branches; the operator mixture makes no measurable difference. All four target skeletons, including an articulated-arch palm, are reached in 100% of 64 restarts under a union operator pool; the arch is unreachable without palm-body and palm-joint operators. Redundancy is negligible at this grid resolution. A simulator-matched evolution run (E5b: 5×6 envelope, noisy returns, random tie-breaks, paired starts, 24 restarts) shows the trade-off behind the insertion prior: it controls bloat but slows growth toward a five-digit target (median generation 10 to 15 under the default pool; 58–83% instead of 96–100% target reach under small-step-heavy pools). A cost term of 0.01–0.02 per unit structural cost halves motor count and lowers the raw pinch proxy by about 0.2. Palm operators are 100% rejected under the current envelope.

### Recommendation and limits

For a 40-generation run under the current envelope: default-heavy operator pool without palm operators, the insertion prior if bloat control matters more than growth speed (or the plain prior with an explicit cost term), branching off, structural cost reported and optionally penalised at 0.01–0.02. Palm operators and an extended envelope go together. Details, numbers and open questions are in `grammar-for-evolution.md`. These are CPU results on kinematics and geometric proxies; they say nothing about task reward. Before any co-evolution run, the fixed 5×6 simulator envelope needs an adapter that refuses over-envelope designs, since palm joints and long digits do not fit the current contract.
