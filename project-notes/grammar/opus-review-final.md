# Opus 5.5 final review of the overnight program (2026-09-26 ~05:10)

Verdict: E2 and E3 sound; most E1-E4 fixes landed; the E5 write-up in grammar-for-evolution.md had false or confounded claims. Coordinator response: note flagged pending correction; fix iteration (I15) and E5 re-run planned.

## Residual defects
- phenodist alignment: joint names are positional (d2p4_j), so insert/delete_phalanx, remove_digit and remove_palm_body renumber later joints; 77/200 insert_phalanx children had >= 2 same-name joints with changed records. Coupled dependents copy the parent's q instead of applying the child's coupling (step_coupling reads 0 mm by construction; step_limits reads 0 because shared values are not clamped).
- reach_coverage normalised by total length^3 penalises every extra digit (reach_log populations converge to ~1.1 digits).
- Insertion distribution only differs in phalanx count and branching; it changes nothing for add_minimal_digit/add_palm_body.
- perturb_parameter sits in the 'coarse' 10% group instead of the small-step share.
- Provenance: every result.json git_dirty true and SHAs predate the code that produced them.
- Ratchets remain: delete_phalanx refuses branch-hosting phalanges; remove_digit_minimal only single-phalanx digits.
- E2 'paired' CIs are marginal CIs sharing resample indices; no difference CIs. E2 summary Reading labels deltas as absolute joints.
- E3 arch target does not require the digits to sit on the jointed palm bodies.

## E5 soundness
- Cost-aware final_best_fitness is the PENALISED value (raw proxy not logged); the reported 0.06 gap is about the penalty itself. 'Halves motors at a small proxy cost' unsupported.
- 'none' is not cost-free: ties broken by lower cost with parents first, a lexicographic parsimony pressure that vanishes under noisy returns. 7/12 G_NOBRANCH pinch runs collapsed to 1-digit hands.
- Starts differ across distributions (start seed includes dist index), so INS-vs-plain is unpaired at n=6 with bimodal outcomes.
- Proxy optima set the structure: opposition favours 2 digits (final 1.9 digits), reach_log favours a single chain; opposition saturates at 1.0 in 42/48 unpenalised runs.
- Operator survival ~ mu/(mu+lambda) for neutral children; it measures neutrality, not usefulness. Retry-exhausted clones are labelled with the last operator.
- Branch fractions under _INS come from the starts and the ratchet, not selection.

## Wording errors in grammar-for-evolution.md (to fix)
Starts are ~9 joints (G_FULL) / 4.6 (G_NOBRANCH), not 3; G_FULL_INS does not grow (+0.1); the mixture matters at step 40 (G_FULL +15.4 DEFAULT vs +7.5 UNION_weighted); E3 UNION is uniform, not weighted, and costs 2-3x more proposals on non-palm targets; insertion prior slows the 5-digit target ~1.5x; 'insertion wins on every proxy' false for reach_log; 'fewer motors' false for G_NOBRANCH; 'random hands 66-85% zero' stale; numbering 4/6/5; 'E5 will show' stale; envelope contradiction (recommended pool spends proposals on palm designs the 5x6 envelope refuses).

## Recommendation (reviewer)
Direction right; cite E2 (insertion prior) and E3 (union pool for palm reach) rather than E5. Report raw proxy and cost separately; no cost claim yet. Move perturb into the small-step group. Make palm operators conditional on an envelope adapter or drop them for the simulator loop. Consider insertion prior of 1-2 phalanges. State that G_NOBRANCH_INS vs G_FULL_INS is undecided.

## Next experiments (reviewer)
1. E5 rerun matching the simulator loop: hard 5x6 envelope (rejected/repaired proposals counted), 40 generations, small population, additive noise on returns, no cost tie-break, starts paired across distributions, >= 20 restarts, raw proxy/cost/motors logged, cost weight in {0, 0.01, 0.02}.
2. A fitness whose optimum needs 3-5 digits and >= 3 phalanges (noisy E3 target distance, or pinch + reach on several objects).
3. Operator-credit ablation under noise (strict improvement over parent under the same seed; lineage credit), comparing UNION_weighted, UNION with perturb in small-step, DEFAULT.
