# Opus 5.5 milestone review of M1 (2026-09-25, HEAD 49b886e)

Independent read-only review by an Opus 5.5 agent with its own probes. Verdict: **M1 partially met**. Recorded verbatim in substance; the coordinator accepts the findings.

## Verdict against the plan's acceptance criteria
- Analytic tests at 1e-12: met (errors ~1e-16). Caveats: 20 configurations per fixture, not 71; `analytic.py` is a generic sympy URDF evaluator, not a hand-derived closed form, so only Pinocchio independently checks the URDF convention.
- Dev hands vs Pinocchio at 1e-10: met (Allegro, LEAP, Barrett, SHARPA; Ability and Inspire locally).
- Export re-import exact: met for imports; for grammar output palm/radius/tip frames are lost by URDF (documented).
- Generated designs valid and Pinocchio-consistent: met (500 seeds valid; 200 seeds x 8 configurations vs Pinocchio, max 7.4e-16 m).
- Report lists coverage and missing constructs: partially met. `expressible` is near-vacuous (two of its checks can never fire after `validate`; it silently skips non-continuation joints, so LEAP 12/12 and Barrett 5/5 mid-digit joints with rpy/lateral offsets the grammar cannot produce are not flagged); ORCA's single wrist joint makes it one digit and its 5-child body is never checked against the grammar's max children per body; report SHA 644b21e came from a dirty tree; LOG/STATE lacked iteration 4.

## Defects, priority order (file:line per the reviewer)
1. Mount point off the host segment: `derive.py:235` uses `Rm @ (0,0,frac*L)`, so a digit base lies on a sphere around the host origin, not on the segment (62% of digits > 1 mm off, up to 8 cm). Breaks the settled rule that mounts lie on their palm body's cell. Fix: translate `(0,0,frac*L)` in the host frame, then rotate.
2. Palm segment inconsistent: `derive.py:215` places a palm body's origin L along its direction from the parent origin and `:227` puts its tip another L further; the stretch from parent origin to child origin belongs to no body.
3. No palm geometry when `palm_body_count` = 0 (`distributions.py:168`; root length 0 at `derive.py:205`): 56/300 seeds have every digit mounted at one point.
4. Two audit keys mislabelled: `nonperpendicular_axis` measures "not axis-aligned" (`test_acceptance_grammar.py:321`); `two_nonparallel_palm_joints` compares axes in local frames (`:335-341`). `coverage.inventory` has the correct axis definition.
5. `expressible` near-vacuous (`coverage.py:351-386`, `:521-527`); no children-per-body check.
6. Couplings ignore source type (`distributions.py:292`): prismatic or continuous sources drive revolute joints in ~45% of seeds.
7. No derivation validator (`derive.py:196-213`): junk steps and inconsistent counts are accepted silently.
8. Report provenance: no dirty flag (`evaluate.py:69`); stale LOG/STATE.
9. Test hygiene: local 1e-9 tolerances (`test_acceptance_grammar.py:72`); `pytest.approx` where `==` is required (`test_analytic_fk.py:67-82`); body-skip `continue`s never assert equal body sets; frozen analytic Pinocchio references unused by any test.
10. `vary` keeps the parent seed (`derive.py:687`); `rules.py` dataclasses are unused (derive passes dicts).

## Grammar-for-evolution observations
- `derive` is pure and replay exact; JSON round trip exact (0 failures over 150 seeds x 7 operators).
- Locality is coarse: only lengths have a one-grid-step operator; median fraction of joints changed: perturb 3%, resample 6%, insert/delete 10%, remove 18%, add 25%, regrow 29%.
- Built-in assumptions: flexion-biased revolute limit sets; digits are straight rods at q=0; couplings only within a digit.

## Recommendations for M2 (priority order)
1. Fix the geometry data model before the hull: one segment convention per body (origin to tip; child joints at `frac` along the host segment); mount = translate along segment then rotate; require at least one palm body or give the root a segment; per-body `segment(start,end)` accessor; tests that every joint/mount origin lies on its host segment and the palm has no gaps; bump GRAMMAR_VERSION.
2. Make coverage and the audit honest: rename to `topology_expressible` or add the missing checks (continuation pose, coupling scope/type/limit image, children per body <= 1 + max_branch_digits, fixed joints inside digits); compute audit keys from derived models in the world frame; wire the frozen analytic Pinocchio references into a test; add a dirty flag to the report SHA; name the ORCA limitation in the report.
3. Evolution-grade operators and a canonical form: small-step mutations per field; a derivation validator; canonical step/joint order; lineage field instead of a stale seed; restrict coupling sources to revolute or make couplings unit-aware; keep closed chains deferred.
