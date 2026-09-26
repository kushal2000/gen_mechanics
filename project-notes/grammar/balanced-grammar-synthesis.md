# A balanced grammar for evolving hands: synthesis and balance test suite

Written 2026-09-26 by the coordinator from two literature reviews (`lit-review-robot-morphology-evolution.md`, `lit-review-evolutionary-methods.md`; claims there are tagged verified/snippet) and our experiments E1-E7 (`grammar-for-evolution.md`). Framing from Martin: the balance that matters is expressive vs tractable, favouring simplicity and elegance while keeping diversity and expressivity; the grammar and the evolutionary process have separate responsibilities; fairness under the shared controller belongs to controller design; the number of evolution stages may change.

## 1. Who owns what

| Responsibility | Owner | Literature anchor |
|---|---|---|
| Expressivity (which hands exist), simplicity (few productions, few ways to say one design), operator locality and reversibility, size-neutral mutation by construction | GRAMMAR | Rothlauf (locality), Sendhoff (causality), Poli & McPhee (size evolution: no drift without selection when operators are symmetric), Green/RJMCMC (exact-inverse proposal pairs) |
| Diversity preservation, founder-share limits, innovation protection, parsimony pressure, noise-aware selection, operator-mixture schedule | PROCESS | ALPS (Hornby), MAP-Elites (Mouret & Clune), random immigrants and crowding, lexicographic parsimony (Luke & Panait), Tarpeian gate, elite re-evaluation (Jin & Branke), bandit adaptive operator selection (Fialho) |
| Evaluation fairness under one shared policy | CONTROLLER | Strgar & Kriegman 2025 (reset to pretrained checkpoint + short fine-tune each generation; the only tested fix), "Fails to select for morphological potential" 2025, DERL's morphological Baldwin effect |

The reviews found no hand-design paper that separates these; most fix the topology (tractable, not expressive), and none uses a diversity mechanism, a parsimony term, or a locality measurement. Our operator-locality numbers and the late-stage schedule result appear to be new.

## 2. What the literature says about our findings

- **Neutral drift toward the growth prior (E2) is an operator-asymmetry symptom, not a grammar law.** The size-evolution equation predicts zero expected size change under mutation alone when each structural move has an exact inverse applied with matched probability and magnitude. Ours do not: `add_digit` inserts material sampled from the initialisation prior (about 7 joints) while `remove_digit` is applicable only sometimes; `regrow_subtree` has no inverse; `delete_phalanx` was blocked by branches; palm removal was leaf-only. The insertion prior reduced the asymmetry (33 -> 9 joints) rather than removing it. The grammar-level fix is symmetric pairs, with any remaining bias declared as a target.
- **Locality bands (E1) are the right grammar metric.** Sendhoff's strong causality and Rothlauf's locality say small genotype steps should give small phenotype steps; we measured 0-4 mm for small steps and 9-14 mm for structural ones with no interquartile overlap, which is what the test suite below asks for.
- **Reachability (E3) is necessary, not sufficient.** Reaching targets under neutral hill-climbing says the operator set connects the space; it says nothing about what selection will do.
- **The step schedule (E7) is a process choice with thin precedent.** The literature's answer is bandit-based adaptive operator selection; a fixed schedule must be validated against it.
- **Founder dominance and end-stage refinement collapse** are what Martin wants to avoid. Only Strgar & Kriegman (controller reset) and LOKI (local search) address collapse directly; nobody treats late refinement collapse as a separate failure. The standard toolbox is process-level: immigrants, parent-vs-offspring replacement, an elite archive over structural bins, age-based protection.
- **Truncation with one offspring per survivor and no recombination loses diversity faster than the 50% suggests**, so a diversity mechanism is warranted independent of the grammar.

## 3. Grammar recommendation: version 0.4, simpler than 0.3

Productions (unchanged, already minimal): Hand -> Root PalmBody* Digit*; PalmBody(parent palm body, mount fraction, optional palm joint); Digit(mount body: a palm body or, when branching is allowed, a phalanx body); Phalanx -> Module in {R, C, P, Coupled(affine, revolute source in the same digit)} + segment. One capsule radius per hand. Geometry derived.

Operator set for evolution: five exact-inverse pairs plus small steps, nothing else in the default pool.

| Pair | Growth move | Shrink move | New material |
|---|---|---|---|
| digit | add_minimal_digit (1 phalanx, R) | remove_digit_minimal (<= 2 phalanges, no branch) | insertion prior |
| phalanx | insert_phalanx | delete_phalanx (re-attaches branches) | insertion prior |
| palm body | add_palm_body | remove_palm_body (re-attaches children) | insertion prior |
| palm joint | toggle_palm_joint (self-inverse; on/off keeps the stored axis) | – | – |
| branch | add_branch_digit (1 phalanx on a phalanx body) | remove_branch_digit | insertion prior |
| parametric | step_axis, step_limits, step_mount, step_length, step_root_length, step_radius, step_coupling (each +-1 grid step, self-inverse) | | |

Dropped from the default pool: `regrow_subtree` and full-size `add_digit` (no inverse, large jumps), `resample_parameter` (rewrites five fields). They remain available for the process to use as immigrants or rare large moves, which is where the literature puts large jumps.

Growth prior: the insertion prior is the only prior the operators sample from (1-3 phalanges, no branches); the initialisation prior is used only for immigrants. Simplicity: parameter grids stay at 15 degrees and 5 mm; branching stays expressible but is off unless the envelope allows it.

## 4. Process recommendation (needs coordination with the RL owners)

1. Founder-share tracking every generation using `Derivation.lineage`; ceiling well under 50% for the bulk of the run.
2. Random immigrants: 5-10% of each generation's evaluations from the initialisation prior, unconditionally.
3. Parent-vs-offspring replacement (deterministic crowding) instead of global ranking of the pooled population, at zero extra cost since each survivor already has one offspring.
4. Age-based protection (ALPS-style): a lineage carrying a new structural change is exempt from full rank competition for about 5-10% of the total budget in generations.
5. Parsimony: lexicographic (size breaks ties among individuals within the noise band) plus a Tarpeian pre-evaluation gate on structural cost; the explicit cost term stays optional and reported.
6. Noise: re-evaluate elites each generation; soften the cutoff within the noise width.
7. Mixture: keep the step schedule as the default but run a bandit adaptive-operator-selection ablation before fixing it.
8. Optional: a MAP-Elites archive over (digits x motors x palm joints) alongside the population, the most principled defence against one root hand dominating.

Controller (out of scope, flagged): Strgar & Kriegman's per-generation reset to a pretrained checkpoint plus a short fine-tune is the tested fix for familiarity bias.

## 5. Balance test suite (measurable, with what we already have)

| # | Metric | Computed with | Pass criterion | Owner |
|---|---|---|---|---|
| 1 | Neutral size drift per operator pair | E2 machinery, mutation only, per pair, paired CI over seeds | E[delta joints], E[delta digits], E[delta palm bodies] per pair within CI of 0; whole-pool drift within CI of 0 unless a target bias is declared | GRAMMAR |
| 2 | Locality bands | E1 (uid-aligned tip displacement) | small-step IQR entirely below structural IQR; structural median <= 20 mm; no operator with p90 > 60 mm in the default pool | GRAMMAR |
| 3 | Reversibility | new: apply pair (g then s) on 1000 parents | canonical hash of parent recovered in >= 95% of applicable cases | GRAMMAR |
| 4 | Reachability | E3 | every target skeleton reached in >= 90% of restarts within the proposal budget | GRAMMAR |
| 5 | Prior size histogram | support audit | initial population not saturated at any size bound; every construct present >= 1% | GRAMMAR |
| 6 | Redundancy | E4 | genuine null mutations <= 2%; distinct phenotypes >= 99% of samples | GRAMMAR |
| 7 | Founder share | lineage tracking in E5b/E7 (and the real loop) | max single-founder share < 50% through 75% of the run | PROCESS |
| 8 | Structural diversity | distinct (digits, motors, palm joints) bins per generation | bins do not collapse to <= 2 before the last quarter | PROCESS |
| 9 | Late-stage structural acceptance | E7 per-generation-third improvement/acceptance | structural moves still accepted in the last quarter (> 0) unless annealed by design | PROCESS |
| 10 | Elite reliability | E5b re-evaluation | recorded vs re-evaluated elite scores agree within the noise band | PROCESS |

Items 1-2 and 4-6 are already computable; 3, 7-8 need small additions (a reversibility check; founder-share and bin counters in the evolution loops).

## 6. Next steps proposed
1. E8: grammar 0.4 operator set (pairs, branch operators, immigrants API) and the reversibility test; re-run E1/E2 as balance tests 1-3.
2. E9: process ablation under E5b mechanics: truncation vs parent-vs-offspring vs immigrants vs archive, with founder share and structural diversity recorded.
3. E10: bandit adaptive operator selection vs the step schedule.
4. Hand the controller protocol question (reset + fine-tune) to the RL owners with the Strgar & Kriegman reference.

## 7. Read first
Strgar & Kriegman 2025 (arXiv 2502.10862); "Evolutionary brain-body co-optimization consistently fails to select for morphological potential" (arXiv 2508.17464); Hornby, ALPS (GECCO 2006); Poli & McPhee, size evolution equation (GP Field Guide, bloat chapter); Fialho et al., bandit-based adaptive operator selection (2010); House of Dextra (arXiv 2512.03743); the co-design taxonomy survey (arXiv 2512.04770).
