# G0 CPU grammar screen: rerun report (v2)

Branch `martin/hand-grammar`. Supersedes `report.md` (that file now carries a one-line pointer here; no other edit was made to it). Script: `experiments/grammar_screen/g0_screen.py` (rerun infra), `experiments/grammar_screen/g0_review_diagnostics.py` (before/after diagnostics, no oracle dependency). Run with `.venv_isaacsim/bin/python`, no Kit/Isaac/GPU. Oracle commit: `773385b` (`[oracle-v2]`, isaacsimenvs `grammar_envelope.py`/`population_file.py`, never edited here). Grammar commit at run time: `773385b` (clean tree; the oracle worker's own commit already included all of this worker's prior grammar commits since they share one branch).

## 1. Question

Martin's question, unchanged from `report.md`: how does the grammar shape exploration of hand designs, allowing diversity without making the problem intractable? `opus-review-g0.md` (I39) found the previous screen's oracle and grammar both had artifacts that dominated the absolute numbers (though the V1 < V3 ordering was robust). This rerun fixes the grammar-side issues assigned to this worker (review items 2, 5, 6) and reruns against the oracle fix (`[oracle-v2]`, items 1, 3, 4 -- a concurrent worker's commit, never edited here).

## 2. Fixes (rule changes, with before/after diagnostics)

### 2.1 Surface mounting + cross-host spacing (review item 2)

**Bug.** Digit mounts sat on the host capsule's CENTRE AXIS (`derive.py`'s `_process_digit`, `base_xyz = (0, 0, mount_frac * host_length)`). Two digits on the same host less than one `mount_frac` step apart landed less than `2 * capsule_radius` apart -- overlap "by construction". The old V2 spacing rule (`_plan_top_level_mounts`) only reasoned about same-host axial spacing, could not reach the target on short hosts, and never reasoned about cross-host coincidences.

**Fix.** Two new `Distribution` fields (`hand_sampler/grammar/distributions.py`), both off by default:
- `mount_on_host_surface`: a digit's mount origin is offset OFF the host's centre axis by the host's own capsule radius, at an azimuth angle -- drawn i.i.d. (V1s) or planned (V2s).
- `mount_min_separation_m` (existing field, reused): combined with `mount_on_host_surface`, dispatches to a NEW placement rule, `derive._plan_top_level_mounts_surface` -- a greedy furthest-point search over every (host, `mount_frac`, azimuth) grid point's ACTUAL root-frame 3-D position (a scratch forward-kinematics pass over the already-sampled root/palm bodies, `derive._host_transforms_from_steps`). A generative placement rule, not a rejection filter.

Every EXISTING named `Distribution` (`G_V1`/`G_V2`/`G_V3` included) is untouched: pinned by `test_g0_screen_v2.py`'s byte-identical `phenotype_hash` check against a snapshot of the pre-review code (commit `ccb5b66`).

**Before/after (n=1000 seeds, 4-5 digits, root length 20-80 mm, <=2 additional palm bodies -- `experiments/grammar_screen/g0_review_diagnostics.py`, `review_diagnostics.json`):**

| | V1 (before) | V1s (surface only) | V2s (surface + spacing) |
|---|---|---|---|
| exact mount coincidences (0 mm apart) | 1088 / 8100 pairs (13.4%) | 53 / 8100 (0.65%) | **0 / 8100 (0%)** |
| median pairwise separation (all pairs) | 22.5 mm | 26.9 mm | **40.0 mm** |
| median same-host separation | 16.25 mm | 22.8 mm | **31.8 mm** |
| fraction of pairs meeting the 2r+margin (29 mm) target | 40.0% | 46.3% | **73.1%** |

Confirmed against the real oracle in the 2000-seed rerun (section 5): the share of the 2000 seeds rejected for rest-overlap (any count) drops V1 39.1% -> V1s 34.8% -> **V2s 28.3%** -- the largest overlap reduction of any single fix, and specifically attributable to spacing (V3/V3s, which do NOT include the planned spacing rule, sit at 34.9%/32.0%).

### 2.2 Curl-axis + opposition host-frame fix (review item 5)

**Bug 1 (phalanx 0 bent).** `sample_bend` applied V3's forward-curl bend to a digit's FIRST phalanx too, silently tilting the digit's own MOUNT frame by 15-45 deg after the opposition prior had already committed to an orientation computed from the (pre-bend) `mount_rpy`.

**Bug 2 (wrong host frame).** The opposition prior averaged `mount_rpy`-derived forward vectors directly, as if every digit's `mount_rpy` were already expressed in a shared (root) frame -- wrong whenever digits mount on different (rotated) palm bodies.

**Fix.** `curl_skip_first_phalanx` (phalanx 0 never bends; `G_BEND` unaffected) and `opposition_use_host_frame` (mean-forward direction and the opposing target computed in the ROOT frame, each digit's `mount_rpy` rotated by its own host's accumulated rotation first).

**Before/after (n=1000 seeds, forced `palm_body_count_range=(1,2)`, `digit_count_range=(2,5)` for guaranteed mixed-host coverage; opposing angle = angle between the last digit's root-frame forward direction and the mean root-frame forward direction of the earlier digits; opus-review-g0.md's "within 90 deg" is the FAILURE share):**

| | V3 (before) | V3s (after) |
|---|---|---|
| median opposition angle, overall | 106.4 deg | **175.1 deg** |
| median opposition angle, mixed-host only | 103.5 deg | **174.8 deg** |
| "opposing" digits within 90 deg of the others (failure), overall | 33.6% | **0.0%** |
| "opposing" digits within 90 deg of the others (failure), mixed-host | 36.5% | **0.0%** |

The fix eliminates the failure mode entirely in this sample (median within a few degrees of the analytically-intended 180 deg, the residual gap from the 15-degree grid snap in `_best_opposing_rpy`).

**"Shared hinge plane" claim, dropped.** Measured directly (successive within-digit axis pairs within 20 deg): V3 7.9%, V3s 8.1% -- unchanged by either fix (the elevation band restricts elevation, never azimuth). Per the review ("either deliver ... or drop it from the docs"), the claim is removed from `variants.py`'s docstring and reported here, honestly, as still open.

**Aggregate effect (2000-seed rerun, section 5).** V3 -> V3s raises the viable fraction from 8.35% to 9.35%, but this specific pairwise comparison is the ONE pair in the whole McNemar table that is NOT significant (two-sided p = 0.26, section 6) -- the fix is real and large on the ISOLATED opposition-angle diagnostic above, but at n=2000 seeds its effect on the aggregate viable fraction is not distinguishable from noise. V3s is still the recommended variant over V3 (section 9): it is the one whose opposition mechanism actually does what its name claims.

### 2.3 Explorability metric (review item 6)

**Bug.** The CPU MAP-Elites archive admitted a child whenever it passed the STRUCTURAL gate, not when it was VIABLE (`admitted` AND `fingertips_reachable >= 2`) -- a materially easier, less interesting bar. Founder counts also differed between variants (9/20/40 in the superseded run) with no common target.

**Fix.** `_run_one_explorability_repeat`'s `try_insert` now requires viability before inserting. Every variant asks for the SAME founders target (20 for this rerun); a variant whose scan cannot fill it is reported via `founders_target_met: false`, never padded. In this rerun every variant reached 20/20 founders (section 8).

## 3. Variants

| | Definition |
|---|---|
| V1 | (kept, reference) `DEFAULT` constrained to the envelope: revolute-only, no branching, `palm_body_count_range=(0,2)`, `digit_count_range=(1,5)`. |
| V1s | V1 + surface mounting (2.1), i.i.d. azimuth. |
| V2s | V1s + planned cross-host spacing (2.1). Sibling of V3s (both built on V1s, not on each other). |
| V3 | (kept, reference) V2 (old spacing rule) + the UNFIXED curl-axis and opposition priors (I30) -- includes both 2.2 bugs. |
| V3s | V1s + the FIXED curl-axis and opposition priors (2.2). |

`G0_SCREEN_VARIANTS` has 7 entries (`V0`..`V3s`); default rerun roster is `V1, V1s, V2s, V3, V3s` (V0/V2 stay selectable for reference only). Every new field defaults off; the full grammar suite passed throughout (section 10).

## 4. Method notes (rerun-specific)

- **2000 fixed seeds per variant** (0..1999), identical across all five variants (required for the McNemar pairing).
- **Wilson 95% CI** on the viable fraction.
- **Exact McNemar** (`scipy.stats.binomtest`, two-sided and one-sided each direction) on the paired `viable` boolean, every pair of the 5 variants, on the shared 2000 seeds.
- **Per-digit-count viable fraction** (1-5): bucketed by the derivation's own SAMPLED `digit_count` (defined even for structurally-rejected designs), Wilson CIs per bucket.
- **Explorability**: 3000 evals x 3 repeats, founders target 20 for every variant.
- **Multiprocessing**: the viability sweep, explorability repeats, and diversity/evolvability/founder pools' extra-seed scans all batch through one `multiprocessing.Pool`, 28 workers (`cpu_count() - 4`, leaving 4 free).
- **"Viable" design**, unchanged: `admitted` AND `fingertips_reachable >= 2`.
- **Wall time**: 5 variants x 2000 seeds, full diversity (300)/evolvability (200)/explorability (3000x3, founders 20) sections, 28 workers: **~23 minutes** total (V1 384.8s, V1s 258.1s, V2s 276.7s, V3 248.6s, V3s 193.8s -- monotonically falling as the viable rate rises, since less extra-seed scanning is needed).

## 5. Results table (2000 seeds/variant)

| Metric | V1 | V1s | V2s | V3 | V3s |
|---|---|---|---|---|---|
| frac admitted | 27.35% | 32.80% | 45.95% | 34.75% | 35.45% |
| frac >=2 fingertips reachable | 14.05% | 14.10% | 14.55% | 38.10% | 34.35% |
| frac both ("viable") | 1.95% | 3.05% | 5.00% | 8.35% | **9.35%** |
| 95% Wilson CI | [1.43, 2.65]% | [2.38, 3.90]% | [4.13, 6.04]% | [7.22, 9.64]% | [8.15, 10.71]% |
| descriptor cells occupied (/30) | 11 | 12 | 14 | 14 | **15** |
| Shannon entropy (bits) | 3.02 | 3.07 | **3.50** | 3.11 | 3.19 |
| mean joint count (viable) | 10.28 | 10.44 | 11.82 | 10.60 | 11.22 |
| mean wall time / design | 99.1 ms | 117.5 ms | 128.1 ms | 121.1 ms | 116.6 ms |
| mean wall time / viable design | 5.08 s | 3.85 s | 2.56 s | 1.45 s | **1.25 s** |
| overlap-rejected (any count) | 39.1% | 34.8% | **28.3%** | 34.9% | 32.0% |
| palm-carrier-multiplicity-rejected | 25.6% | 25.8% | 23.7% | 21.8% | 24.4% |

Bold = best of the five on that row. Every metric with a large enough sample moves the same direction as the superseded screen (V1 < V1s < V2s < V3 <~ V3s), now against the fixed oracle: viable rate improves **4.8x** from V1 to V3s, and cost per viable design falls **4.1x** (5.08s -> 1.25s).

## 6. Exact McNemar (pairwise, shared 2000 seeds)

| Pair | n discordant | two-sided p | one-sided p (row > col) | Higher |
|---|---|---|---|---|
| V1 vs V1s | 90 | 0.0263 | 0.0132 | V1s |
| V1 vs V2s | 137 | 1.9e-07 | 9.4e-08 | V2s |
| V1 vs V3 | 188 | 3.4e-22 | 1.7e-22 | V3 |
| V1 vs V3s | 194 | 3.8e-29 | 1.9e-29 | V3s |
| V1s vs V2s | 153 | 0.00202 | 0.00101 | V2s |
| V1s vs V3 | 200 | 2.7e-14 | 1.4e-14 | V3 |
| V1s vs V3s | 222 | 5.7e-18 | 2.9e-18 | V3s |
| V2s vs V3 | 245 | 2.2e-05 | 1.1e-05 | V3 |
| V2s vs V3s | 253 | 4.8e-08 | 2.4e-08 | V3s |
| **V3 vs V3s** | 286 | **0.261** | 0.131 | V3s (n.s.) |

Every pairwise comparison is significant at p < 0.03 **except V3 vs V3s** (the one pair where the fix's effect on the aggregate viable fraction cannot be distinguished from noise at this sample size -- see 2.2's closing paragraph). V2s is clearly distinct from V1s (p = 0.002): the planned cross-host spacing rule earns its own arm.

## 7. Per-digit-count viable fraction (1-5 digits)

| Digits | V1 | V1s | V2s | V3 | V3s |
|---|---|---|---|---|---|
| 1 (n=367) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 (n=426) | 4.69% | 7.51% | 7.98% | 15.96% | **17.14%** |
| 3 (n=377) | 2.92% | 3.45% | 7.16% | 16.98% | **17.51%** |
| 4 (n=417) | 1.44% | 3.12% | 6.24% | 6.71% | **8.63%** |
| 5 (n=413) | 0.48% | 0.73% | **3.15%** | 1.69% | 2.91% |

Digit count 1 is trivially 0% for every variant (a single digit cannot supply the required >= 2 fingertips reaching, independent of any grammar rule -- not a bug). Every variant's viable fraction peaks at 2-3 digits and falls off toward 5 digits (more digits means more overlap surface and more reach difficulty), but V3/V3s hold up far better at 2-3 digits (~16-18%) than V1/V1s/V2s (~3-8%) -- the curl/opposition family's reach gain dominates specifically in the low-to-mid digit-count range the descriptor grid's own "coverage" axis cares about. V2s is the best of the non-curl variants at every digit count and the single best at 5 digits (3.15%), consistent with spacing mattering most exactly when there are the most digits to fit.

## 8. Diversity, evolvability, explorability

| | V1 | V1s | V2s | V3 | V3s |
|---|---|---|---|---|---|
| diversity pool (of 300 target) | 300 | 300 | 300 | 300 | 300 |
| mean pairwise tip displacement (mm) | **163.7** | 147.7 | 142.3 | 115.0 | 114.6 |
| evolvability pool (of 200 target) | 200 | 200 | 200 | 200 | 200 |
| child viability rate | 64.4% | 67.4% | 64.2% | **72.6%** | 70.7% |
| frac children change descriptor cell | 19.3% | **22.4%** | 19.4% | 17.0% | 19.3% |
| frac children identical-or-neutral | 74.8% | 70.0% | 71.2% | 75.7% | 72.2% |
| founders (of 20 target) | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| MAP-Elites coverage (mean +/- std, /30) | **54.4% +/- 1.6%** | 46.7% +/- 2.7% | 50.0% +/- 4.7% | 50.0% +/- 0.0% | 52.2% +/- 1.6% |
| distinct founders in final archive (/20) | 5.67 | 6.00 | **8.00** | 6.00 | 6.33 |

Every variant reaches its 300/200/20 pool targets in the fixed 2000-seed sweep plus a modest extra-seed scan (section 4) -- no honesty caveat needed here, unlike the superseded run.

**Continuous diversity trades off against viability, again.** Mean pairwise tip displacement falls monotonically as the viable rate rises (163.7 mm at V1 down to 114.6 mm at V3s) -- the same trade-off the superseded report flagged: the curl prior pulls every viable design's rest pose toward a similar "fingers curled forward" shape, so designs that differ in digit/joint COUNT still end up geometrically closer at rest. Structural diversity does not suffer the same way (cells occupied and evolvability's child-viability-rate both trend up with viable rate), but continuous pose diversity is a real, measured cost of the curl prior specifically -- worth flagging to Martin if the pilot's fitness or descriptor space cares about continuous pose variety, not just digit x joint count.

**Explorability coverage is NOT monotonic in viable rate.** V1 has the HIGHEST mean MAP-Elites coverage (54.4%) despite the LOWEST viable rate (1.95%) -- with founders now fixed at 20 for every variant and children admitted only when VIABLE, coverage measures how well 20 SPECIFIC founders' descendants fill the grid under `EVOLUTION_OPERATORS`, not how easy the grammar is to sample from; V1's particular 20 founders happen to seed a wide spread. V2s has the most distinct founders represented in its final archive (8.00/20, vs 5.67-6.33 for the others) -- the strongest evidence against single-founder dominance -- and the widest run-to-run spread (std 4.7%). V3's coverage is perfectly repeatable across all 3 repeats (std 0.0%) -- its curl/opposition-shaped founders converge to the same 15-cell archive regardless of the mutation RNG seed, a notable (not necessarily good or bad) property to watch in the pilot.

## 9. Recommendation

**Baseline: V1. Best: V3s.** This mirrors the superseded report's own logic, now on a fixed oracle and grammar: V1 is the fairest "no extra generative rule" baseline (same envelope, same digit/joint ranges as every other variant, differing from V1s/V2s/V3s by exactly the rules under test) and it functions well as one -- 39 viable designs from the fixed 2000-seed sweep, comfortably enough to seed a MAP-Elites founder pool without padding.

**Why V3s over V3, despite the McNemar tie (section 6).** V3 and V3s are statistically indistinguishable on the aggregate viable fraction at n=2000 (p=0.26) -- but V3's opposition mechanism is CONFIRMED BROKEN (median opposition angle 106 deg, 34-37% of "opposing" digits point the wrong way, section 2.2), while V3s's is confirmed working (175 deg median, 0% failure) on the same isolated diagnostic. Recommending V3 over V3s would mean shipping a mislabeled rule into the pilot for no measured benefit; V3s delivers the same aggregate numbers with a mechanism that actually does what it claims, at a marginally higher viable rate (9.35% vs 8.35%) and cost per viable design (1.25s vs 1.45s).

**Why V2s is a defensible third arm, not just an ablation.** V2s is clearly distinct from V1s (p=0.002) and has the best MAP-Elites founder diversity (8.00/20 distinct founders) and highest viability-pool Shannon entropy (3.50 bits) of all five variants -- if the pilot can afford a 3-arm comparison, V2s vs V3s isolates how much of the grammar's gain is "just spacing" (a purely geometric fix) vs. "curl toward opposition" (a posture prior with the continuous-diversity cost noted in section 8). On a 2-arm budget, V3s alone already captures most of V2s's overlap-reduction benefit (its own overlap-rejected share, 32.0%, sits between V1s's 34.8% and V2s's 28.3%) plus the much larger reach gain V2s does not have.

**Caveats for Martin to weigh:**
1. Every variant's viable rate is still under 10%; even V3s fails ~90.6% of samples for one reason or another. This screen shows a clear, now-trustworthy ranking, not a solved sampling problem.
2. The curl prior's continuous-diversity cost (section 8) is real and should be watched if the pilot's descriptor space or fitness cares about pose variety within a digit/joint-count cell, not just structural (digit x joint) coverage.
3. V3's perfectly-repeatable (std 0.0%) MAP-Elites coverage and V1's unexpectedly-high coverage (section 8) are both properties of this SPECIFIC 20-founder budget and `EVOLUTION_OPERATORS` mutation pool, not necessarily properties that will hold at the pilot's own (larger) population scale.
4. This remains a kinematics-only, RL-free proxy for viability and diversity. It answers exactly the question this screen was scoped to answer (does the grammar leave evolution something to climb), not whether V3s-sampled hands control well.

## 10. Test counts

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 timeout 1500 python3 -m pytest -p no:cacheprovider hand_sampler/grammar_bench/tests -q`: **3157 passed** (commit `773385b`, the oracle-v2 HEAD this rerun used; run three times across this work, always green). The previously-reported "3135 passed, 1 skipped" baseline's one skip is an environment/seed-dependent skip elsewhere in the suite (`test_grammar05_b.py`'s "add_branch_digit not applicable" skip, RNG-seed dependent), not affected by any change in this work; 21 new tests were added (`test_g0_screen_v2.py`).

## 11. Reproducing this run

```
.venv_isaacsim/bin/python experiments/grammar_screen/g0_screen.py \
  --variants V1 V1s V2s V3 V3s \
  --out-dir project-notes/grammar/experiments/g0 --tag v2 \
  --viability-seeds 2000 \
  --diversity-n 300 --diversity-pairs 3000 --diversity-n-configs 8 \
  --evolvability-parents 200 --evolvability-children 5 \
  --explore-founders 20 --explore-evals 3000 --explore-repeats 3 \
  --max-extra-seeds 20000 --n-workers 28

.venv_isaacsim/bin/python experiments/grammar_screen/g0_review_diagnostics.py \
  --n-seeds 1000 --out project-notes/grammar/experiments/g0/review_diagnostics.json
```

Outputs (this directory): `g0_results_v2.json`, `g0_summary_v2.csv`, `g0_per_digit_count_v2.csv`, `g0_viability_v2.png`, `g0_descriptor_coverage_v2.png`, `g0_explorability_v2.png`, `g0_per_digit_count_v2.png`, `review_diagnostics.json`.
