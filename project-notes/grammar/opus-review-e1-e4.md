# Opus 5.5 review of the overnight experiments E1-E4 (2026-09-26 ~03:40)

Read-only review with independent probes. The coordinator accepts the findings; E5 was stopped and the defects below are being fixed before E1/E2/E5 are re-run.

## Headline
1. E1 locality for joint-adding/removing operators was confounded: parent and child configurations were sampled independently, and adding a joint shifts the random stream for every later joint. Noise floor (a hand vs itself under another config seed) is 94 mm. With configurations aligned by joint name (new joints at 0), medians are: insert_phalanx 9.9 mm (reported 106), delete_phalanx 13.9 mm (86), add/remove_digit ~20 mm (~75), regrow_subtree 48 mm (98), resample_parameter 12.5 mm (30.5), perturb 0.7 mm (0.7). The LOG's "whole distal chain shifts" explanation is withdrawn.
2. E2 measured neutral drift under G_SERIAL only; walks relax toward the G_SERIAL generation prior (12.2 joints, 3.5 digits), not to the caps. Under the pool the LOG recommended (union) on G_FULL, a 3-joint hand grows to ~20 joints in 40 steps and ~40 by step 400 (48 seeds); DEFAULT pool on G_FULL: 23.6 at 40, 28.5 at 400 (G_FULL prior mean 29.2); union on G_NOBRANCH: 12.3 / 17.0. Causes: delete_phalanx refuses branch-hosting phalanges (ratchet); remove_palm_body only removes empty leaves while add_palm_body always applies; remove_digit_minimal only removes single-phalanx digits; growth operators sample new material from the initialisation prior.
3. E3: arch_palm at 0% under DEFAULT is true by construction (no DEFAULT operator creates a palm body from a palm-less start); the arch target ignores where digits mount (empty jointed palm bodies satisfy it), so MINIMAL's 100% is inflated; the budget counted accepted neutral moves, which penalises small-step pools artificially (use proposals); percentile-bootstrap CIs collapse for Bernoulli rates (use Wilson/Clopper-Pearson); conditional medians need censoring; the bend variant was never implemented.
4. E4: the "null mutation" rate is almost entirely VariationImpossible (5.2% predicted from E1 applicability vs 5.4% reported); the reorder fraction is not a redundancy measure; 10k distinct hashes is expected.

## Proxies
- opposition: counts tips on the same branched digit; satisfied by parallel neighbouring fingers touching; 66-85% of random hands score 0 (flat landscape).
- reach_coverage: convex hull of tip samples, not a union of reachable sets; "front" is world +x; divided by root length cubed while root length (0.02-0.08 m) is never mutated (64x lottery); rewards long links and branch tips.
- antipodal_pinch: BUG - pairs contacts from different configurations; sphere at the tip centroid; radial normals; 0 for 90-95% of hands then 1.0.
- structural_cost: as planned; counts palm-joint motors on empty palm bodies.

## Canonical form
Name- and order-invariant, geometry-sensitive. Not invariant to equivalent parameterisations (axis sign with negated limits; rpy aliases at gimbal lock); -0.0 after rounding can change the hash (fix: add +0.0).

## Operators
- step_length is an alias of perturb_parameter (double weight in any pool containing both).
- step_limits/step_coupling take neighbours in UNSORTED choice lists (one step can flip a multiplier's sign or jump limits from (0,110) to (-30,60)); sort the lists.
- Hand-step genes root_length and capsule_radius_m are never mutated.
- resample_parameter rewrites 5 PalmBody fields; delete_phalanx silently resamples a dependent coupled module; step_mount on a PalmBody rotates everything mounted on it.
- toggle_palm_joint is reversible only in distribution; remove_palm_body is leaf-and-empty-only.

## Statistics
E2 bootstrap over seeds is right but mixtures share starts (use paired CIs). E3 needs Wilson intervals and censored medians.

## Recommendation (reviewer)
Union pool with step_length removed, sorted lists, root-length and radius operators; weights by evaluations ~55% small-step (incl. perturb), 25% minimal structural (incl. insert/delete_phalanx and palm ops), 10% resample, 10% coarse; growth operators should draw new material from a small insertion distribution (no branches, 1-3 phalanges) separate from the initialisation prior. Provisional variant: G_NOBRANCH with palm bodies and joints (branches double the prior joint count, ratchet under deletion, inflate tip proxies). Most important missing experiment: neutral drift and proposal-budgeted reachability under the actual recommended pool on G_FULL and G_NOBRANCH; re-run E1 with aligned configurations; fix antipodal_pinch before any E5.

## Other defects
Provenance: all result.json have git_dirty true; E3/E4 SHA predates the operators' commit. Circular import e2_drift -> runner -> e3_reach -> e2_drift (import runner first). Plan deviations not stated in summaries (no bend variant; E3 compared pools, not grammars; E4 two variants).
