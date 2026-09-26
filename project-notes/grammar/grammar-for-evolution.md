# Which grammar is best for evolution? Evidence and recommendation

Written 2026-09-26 by the coordinator from experiments E1-E5 (corrected instruments; E5 at commit after 40cb2ed). Everything here is CPU-only kinematics and geometry; no RL was run. Results describe the grammar at `GRAMMAR_VERSION` 0.3 with the operators in `hand_sampler/grammar/derive.py`; they do not predict task performance.

## 1. Question and method

"Best for evolution" was decomposed into properties a fitness-driven search over the grammar needs, each measured with seeded, reproducible experiments recorded under `project-notes/grammar/experiments/`:

| Property | Experiment | Instrument |
|---|---|---|
| Locality of one mutation | E1 (1000 parents x 13 operators) | fingertip displacement with joint-aligned configurations, joint/motor deltas |
| Neutral drift (bloat) | E2 (128 seeds x 400 steps x 3 mixtures x 4 distributions, paired CIs) | joints/digits/motors vs step |
| Reachability of target skeletons | E3 (4 targets x 2 pools x 4 distributions x 64 restarts, 1500 proposals) | Wilson CIs, censored medians |
| Redundancy | E4 (10k samples, 20k offspring) | canonical phenotype hash |
| Exploitability under selection | E5 (48 conditions x 6 restarts, (32+32), 40 generations) | geometric proxies: opposition, antipodal pinch, log reach; optional cost term |

An independent Opus review found the first versions of E1/E2/E4 confounded (misaligned configurations, wrong pool, null = impossible); the numbers below are from the corrected re-runs.

## 2. Findings

**Locality (E1).** Small-step operators move fingertips by 0-4 mm at the median and never change joint counts. Structural operators are moderate: add/remove digit 10-11 mm, insert/delete phalanx 14-16 mm, resample 15 mm; regrow_subtree is the only coarse one (44 mm median, 75% of joints changed at p90). Every operator is applicable >= 82% of the time; genuine null mutations are 0% for default operators and 1% for small steps.

**Neutral drift (E2).** From 3-joint starts, drift relaxes toward the prior the growth operators sample from, not toward the caps. Final joints at step 400: G_FULL 33 [29,36]; G_NOBRANCH 14-16; with an insertion distribution (new material limited to 1-3 phalanges, no branches) 9 [8,10] for both. The operator mixture barely matters (DEFAULT, UNION_uniform, UNION_weighted within CI); what the growth operators insert does. Branching doubles the prior joint count and ratchets (a phalanx hosting a branch cannot be deleted).

**Reachability (E3).** Under the union pool every target is reached in 100% [94%,100%] of restarts for every distribution: staggered 5-digit anthropomorphic, 3-digit radial, 2-digit prismatic gripper, and the articulated-arch palm, with medians of 26-211 proposals. Under the default pool the arch palm is unreachable (0% [0,6%]) because no default operator creates a palm body or palm joint; that is a property of the operator pool, not of the representation.

**Redundancy (E4).** 10,000 random seeds give 10,000 distinct phenotypes; sibling offspring coincide 2-3% of the time; derivation order differs from canonical order in 72-90% of models (the hash handles it). Redundancy is not a search problem at this grid resolution.


**Selection under proxies (E5).** Marginal means over conditions (per-cell 95% CIs over 6 restarts are in `experiments/E5_evolve/summary.md` and are wide):

| Factor | Level | Final best proxy | Final mean motors |
|---|---|---|---|
| distribution | G_FULL | 0.65 | 6.1 |
| | G_FULL_INS | 0.70 | 4.9 |
| | G_NOBRANCH | 0.55 | 2.9 |
| | G_NOBRANCH_INS | 0.73 | 3.5 |
| pool | DEFAULT_uniform | 0.66 | 4.7 |
| | UNION_weighted | 0.66 | 4.1 |
| cost term | none | 0.69 | 5.6 |
| | aware (0.02 x cost) | 0.63 | 3.1 |
| proxy | opposition | 0.85 (ceiling 1.0 reached in 4 cells) | 2.5 |
| | antipodal pinch | 0.76 | 5.5 |
| | log reach | 0.37 | 5.0 |

The insertion prior helps selection, not only neutral drift: both insertion variants beat their plain counterparts on every proxy while using fewer motors; plain G_NOBRANCH with the union pool did worst on pinch (0.27-0.33). The union pool matches the default pool on fitness with fewer motors; under it, small-step operators survive selection 42-49% of the time (perturb 48%, step_axis 44%, step_limits 42%, step_mount 41%, step_radius 48%, step_root_length 44%, step_coupling 49%) against 9-29% for structural moves (regrow 9%, add_digit 10%, add_minimal_digit 14%, add_palm_body 15%, insert_phalanx 16%, remove_digit 18-19%, delete_phalanx 29%, remove_palm_body 37%, toggle_palm_joint 31%). A cost term of 0.02 x structural cost halves motors at a 0.06 proxy cost. Constructs are neither eliminated nor demanded by these proxies: at generation 40, palm joints appear in 8-23% of individuals, couplings 15-35%, prismatic joints 17-44%, branches 18-39% where allowed. Caveats: the proxies are diagnostics with flat regions (opposition saturates at 1.0; 66-85% of random hands score 0), fitness is re-evaluated each generation with a new configuration seed, and 6 restarts give wide intervals; none of this predicts RL reward.

## 3. Recommendation

1. **Representation and productions**: keep grammar 0.3 (root palm segment, palm bodies as a tree with optional palm joints, digits with optional branching, R/C/P modules, intra-digit affine couplings, one capsule radius per hand). Palm articulation and non-anthropomorphic layouts are expressible and reachable.
2. **Growth prior separate from initialisation prior**: use the insertion distribution for add_digit, add_palm_body, regrow_subtree and add_minimal_digit. This is the single largest lever on bloat (3.5x).
3. **Operator pool**: the union pool (default structural + minimal structural incl. palm-body and palm-joint operators + small steps incl. root length and radius), weighted by evaluations roughly 55% small-step, 25% minimal structural, 10% resample, 10% coarse. Drop step_length (alias of perturb). Regrow_subtree stays as the rare large move.
4. **Branching**: keep it expressible; evolve with G_NOBRANCH_INS by default (best proxy results with the fewest motors) and use G_FULL_INS when branching is a design question; E5 shows selection tolerates branches (18-39% of individuals) but does not need them.
6. **Selection**: report structural cost always; a small cost term (0.02 x cost) halves motor count at a small proxy cost and is the cheapest guard against bloat under a noisy task score.
5. **Geometry**: derived, never a gene (capsules; nearest-spine palm cells with recorded adjacency).

## 4. Open questions for Martin
- Rest-bend production (I11): every real hand has a mid-chain frame rotation or offset the straight-rod digit cannot express. Not needed for the targets above; needed only if real-hand fidelity matters for the evolved space.
- Palm-joint travel and coupling semantics for underactuation (which joints share a motor) are parameter-table choices with no evidence yet.
- Whether structural cost should enter selection or only reporting (E5 cost-aware runs will show what it suppresses).
- Simulator contract: the fixed 5x6 envelope cannot hold palm joints or >6-joint digits; an envelope adapter that refuses over-envelope designs is required before any co-evolution run.

## 5. Provenance
Commits on `martin/hand-grammar` through the E5 results commit; each experiment folder has result.json with seeds, git SHA, versions and wall time. Reviews: opus-review-m1.md, opus-review-e1-e4.md.
