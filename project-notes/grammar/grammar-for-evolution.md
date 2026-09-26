# Which grammar is best for evolution? Evidence and recommendation

Final version, 2026-09-26 ~06:30, written by the coordinator from experiments E1-E7 on `martin/hand-grammar`. Two independent Opus reviews (`opus-review-e1-e4.md`, `opus-review-final.md`) found the first versions of E1, E2, E4 and E5 confounded; every number below comes from the corrected re-runs (E1/E2 with joint-identity alignment and paired CIs, E5b matched to the simulator loop). All results are CPU kinematics and geometric proxies; nothing here predicts task reward.

## 1. Question and method

"Best for evolution" was decomposed into properties a fitness-driven search needs, each measured with seeded experiments recorded under `project-notes/grammar/experiments/` (result.json carries seeds, git SHA or diff hash, versions, wall time):

| Property | Experiment | Instrument |
|---|---|---|
| Locality of one mutation | E1: 1000 parents x 13 operators | fingertip displacement with configurations aligned by stable joint identity |
| Neutral drift | E2: 128 seeds x 400 steps x 3 mixtures x 4 distributions | joints/digits/motors vs step; paired difference CIs (shared starts) |
| Reachability | E3: 4 targets x 2 pools x 4 distributions x 64 restarts, 1500 proposals | Wilson CIs, censored medians |
| Redundancy | E4: 10k samples, 20k offspring | canonical phenotype hash |
| Mixture schedules | E7: 4 pools x 3 distributions x 2 fitnesses x 2 cost weights x 24 restarts, same mechanics as E5b | reach rate, first-reach generation, late-window tip displacement, paired differences |
| Selection, simulator-matched | E5b: 54 conditions x 24 restarts, (16+16), 40 generations, hard 5x6 envelope, noisy returns, no cost tie-break, starts paired across all factors | raw proxy, cost, motors, target-reach rate, per-operator rejection and improvement |

Grammar under test: version 0.3 (root palm segment; palm bodies as a tree with optional palm joints; digits with optional branching; R/C/P modules; intra-digit affine couplings; one capsule radius per hand), operators in `hand_sampler/grammar/derive.py`.

## 2. Findings

**Locality (E1, aligned).** Small-step operators move fingertips 0-4 mm at the median (perturb 0.8, step_axis 1.7, step_mount 4.1, step_root_length 2.5, step_limits 0.3, step_coupling 1.5, step_radius 0). Structural operators: insert_phalanx 8.6 mm, add_digit 10, remove_digit 11, delete_phalanx 13, resample 14, regrow 37 (p90 120 mm). Applicability >= 82% everywhere; genuine null mutations 0% (default) and about 1% (small steps; step_axis 4.9%).

**Neutral drift (E2).** Starts are 1.8 digits with 1.9 phalanges. Final joints at step 400: G_FULL 33 [29,36]; G_NOBRANCH 14-16; G_FULL_INS 8.9 [8.1,9.7]; G_NOBRANCH_INS 8.6-9.1. Paired differences: insertion prior minus plain = -22 to -24 joints for the G_FULL family (CI excludes zero by a wide margin); union-weighted minus default-uniform = within +-1.5 joints, CIs include zero for every distribution. At step 40 the mixture does matter for G_FULL (default +15 joints vs union-weighted +7.5), so the mixture affects speed, the insertion prior affects the plateau.

**Reachability (E3).** With the union pool (uniform), all four targets (staggered 5-digit anthropomorphic, 3-digit radial, prismatic gripper, articulated-arch palm with digits on jointed palm bodies) are reached in 100% [94,100] of restarts for every distribution. The default pool reaches the three non-palm targets 100% in 2-3x fewer proposals (anthropomorphic median 49 vs 143) and cannot reach the arch (0% [0,6]) because it has no palm-body or palm-joint operator. The insertion prior slows the 5-digit target about 1.5x (78 vs 49 proposals under default; 211 vs 143 under union).

**Redundancy (E4).** 10,000 random seeds give 10,000 distinct phenotypes; genuine null mutations are 0-1%; VariationImpossible 3-5%. Not a search problem at this grid.

**Selection, simulator-matched (E5b).** Envelope 5 digits x 6 joints, no palm joints, no branches; fitness = proxy + Gaussian noise (sigma = 10% of the initial range) - w x structural cost; ties random.
- *Growth toward a large target* (fitness = negative structural distance to the 5-digit, >= 3-phalanx target): under the default pool every condition reaches the target in 100% of 24 restarts, at median generation 10 without the insertion prior and 14.5-16 with it. Under the small-step-heavy union pools the plain distribution still reaches it 96-100% of the time (median generation 20-24), but the insertion-prior distributions only 58-83% within 40 generations. Paired difference in final distance, INS minus plain, is -0.29 to -0.46 with CIs excluding zero under union pools.
- *Cost term* (raw proxy reported): w = 0.01-0.02 cuts final motors from 5-6 to 2-3 but lowers the raw pinch proxy, e.g. 0.89 -> 0.68 -> 0.66 (G_FULL_INS, default pool). The cost is real, not a logging artifact.
- *Pinch proxy*: final best raw pinch 0.70-0.90 across conditions; INS minus plain differences have CIs that include zero in 5 of 6 pinch comparisons. Distributions are not separated by this proxy.
- *Palm operators under the envelope*: add_palm_body is envelope-rejected 100% of the time and toggle/remove_palm_body are inapplicable 100%, so the union-weighted pool wastes about 15% of proposals; the same pool without palm operators has no dead weight. Clone rate (all 8 retries rejected) is below 1.3e-4 everywhere.
- *Operator strict-improvement rates* (child raw proxy > parent, same seed; pinch, default pool): add_digit 18%, delete_phalanx 14%, remove_digit 10%, insert_phalanx 9%, perturb 9%, regrow 8%, resample 4%.


**Mixture schedules (E7).** Two schedules were compared with the static pools under E5b mechanics: `SCHEDULE_step` (default pool for generations 0-14, then the small-step-weighted pool without palm operators) and `SCHEDULE_linear` (structural share ramping 0.8 -> 0.2). On the large-hand target, `SCHEDULE_step` matches the default pool (reach rate and final distance differences have CIs including zero in 5 of 6 distribution x cost cells; first reach 0.6-5 generations later) while cutting late-window tip displacement per accepted mutation from 14-18 mm to 6-8 mm, the same as the static weighted pool. Against the static weighted pool it is better on every reach measure: reach rate +0.04 to +0.33 and first reach 6.5-15.4 generations earlier, all CIs excluding zero. `SCHEDULE_linear` recovers full reach only without the insertion prior; with it, it still lags the default pool by 0.04-0.17 in reach rate. On the pinch proxy `SCHEDULE_linear` scored highest (0.91-0.98 at cost 0; differences vs both static pools positive, e.g. +0.35 [0.10, 0.60] for G_NOBRANCH at cost 0.01), which is the one place the smooth ramp wins.

## 3. Recommendation

1. **Representation and productions**: keep grammar 0.3. Palm articulation and non-anthropomorphic layouts are expressible and reachable (E3); geometry stays derived (capsules; nearest-spine palm cells with recorded adjacency).
2. **Growth prior**: the insertion prior is a dial, not a free win, and the E7 step schedule removes most of its growth penalty. It is the only lever that controls neutral bloat (E2: 33 -> 9 joints) and it slows growth toward large hands (E3: 1.5x more proposals; E5b: median generation 10 -> 15 under the default pool, and 58-83% instead of 96-100% target reach under small-step-heavy pools). For a 40-generation budget starting from simple seeds, use the insertion prior with the default-heavy pool, or the plain prior with an explicit cost term; do not combine the insertion prior with a small-step-heavy pool if large hands must be reachable.
3. **Operator pool for the simulator loop**: a step schedule (E7): default structural pool for roughly the first 15 of 40 generations, then the small-step-weighted pool, both WITHOUT palm operators while the 5x6 envelope forbids palm joints (E5b: they are 100% dead). This keeps the default pool's growth (100% reach of the 5-digit target with the insertion prior, median generation 16-17) with half the late tip displacement. The linear ramp is the better choice only if the task rewards fine adjustment more than structure.
4. **Operator pool when palm articulation is allowed** (after an envelope change): add the palm-body and palm-joint operators; without them the arch is unreachable (E3).
5. **Branching**: off by default (envelope forbids it; it doubles the neutral prior and ratchets); G_NOBRANCH_INS vs G_FULL_INS is not separated by any experiment here.
6. **Cost**: report structural cost always. A term of 0.01-0.02 per unit cost halves motors and lowers the raw proxy by about 0.2 on pinch; whether that trade is right depends on the task and is a decision, not a finding. Under noisy returns there is no implicit parsimony (E5b broke ties at random), so bloat control must come from the prior or an explicit term.

## 4. What is not established
- Any relation between the proxies and RL reward. Opposition saturates and favours two digits; pinch has flat regions; the target-distance fitness is structural.
- The exact switch generation of the step schedule (15 of 40 tested only) and its interaction with a noisy task score.
- Whether the rest-bend production (issue I11) matters for evolution; real hands need it for fidelity, the targets here do not.
- Whether palm cells with 7% mean overlap among adjacent pairs behave acceptably in contact; the alternative is a per-body capsule palm.

## 5. Open questions for Martin
- Envelope: keep 5x6 without palm joints (then drop palm operators) or extend it (then keep them)?
- Insertion prior on or off for the first co-evolution run (bloat control vs growth speed)?
- Cost term in selection, or cost reporting only?
- Rest-bend production: needed?

## 6. Provenance
E1/E2/E5b results were generated from a working tree whose diff hash is recorded in each result.json (`git_diff_sha256` 4ee077ad...), then committed as e938fcf with that code; E3/E4 from the I14 code (commit history in LOG.md). Suite: 2,923 tests, all passing.

## 7. Grammar 0.5 status (2026-09-26 afternoon; commits through iteration C)

Implemented: rest-bend primitive on every phalanx (Pinocchio-verified), continuous sign-normalised limits (G_CONT), coupling containment rule, exact-inverse operator pool `EVOLUTION_OPERATORS` (five pairs: minimal digit, phalanx, empty palm body, palm-joint toggle, branch digit; plus small steps incl. bend and root/radius), undo targets, a stratified immigrant prior, and the balance test suite as experiment E12.

Balance test results (E12, `experiments/E12_balance/summary.md`):
- Reversibility: 100% for every pair (500 parents, two distributions).
- Locality per operator: pass (every structural p90 <= 42 mm; every small-step median <= 4.4 mm).
- Prior size histograms: pass (no bound holds more than 20% of mass). Redundancy: 5000/5000 distinct.
- Neutral drift, stationary regime (starts sampled from the prior, 200 steps): phalanx pair and palm-joint toggle balanced; digit pair +0.8 digits, empty-palm pair +0.5 palm bodies, branch pair +8 joints per 200 steps. The first two are the mismatch between the sampling prior and the walk's own stationary distribution (about +0.1-0.2 per 40 steps, negligible at the project's budgets); the branch pair drifts because branches with more than one phalanx cannot be removed while growth is almost always applicable. Branching is off under the current envelope; if it is turned on, `remove_branch_digit` should accept up to two phalanges.
- Reachability with the evolution pool alone (E3): arch palm 100% in 71-87 proposals, radial 75-92%, prismatic 53-73%, anthropomorphic 14-20% within 1500 proposals: minimal moves are slow to build five three-phalanx digits, which is why the step schedule (default pool early) or stratified immigrants remain part of the recommendation.

Coverage after 0.5 (`pilot-report.md`): 8 of 11 real hands are now topology-expressible (the rest bend removed the universal blocker); none is in the default sampling support, which is by design (grids and choice sets), and E11 gives the widening levels at which each enters.

Open: whether to sample limits continuously by default (G_CONT) for the co-evolution run; the branch-removal rule; suite runtime (about 3.5-5 min, worth trimming).
