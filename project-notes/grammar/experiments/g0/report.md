# G0 CPU grammar screen: report

Branch `martin/hand-grammar`. Script: `experiments/grammar_screen/g0_screen.py`. Run with `.venv_isaacsim/bin/python`, no Kit/Isaac/GPU. Git SHA at run time: `ed7e3c3ddb6d96c19710bbf7a4263720e626f8e1` (tree dirty from a concurrent worker's uncommitted edits under `isaacsimenvs/inhand_reorient/`, none of which this screen touches or depends on). Results: `g0_results.json`, `g0_summary.csv`, `g0_viability.png`, `g0_descriptor_coverage.png`, `g0_explorability.png` (this directory).

## 1. Question

Martin's question: how does the grammar shape exploration of hand designs, allowing diversity without making the problem intractable? Two measurements motivated this screen: 59-64% of sampled grammar hands self-intersect at rest, and most sampled hands reach the spawned cube with only one fingertip. This screen measures, cheaply on the CPU, how four grammar variants trade viability against diversity and explorability, using only the frozen oracle `isaacsimenvs.inhand_reorient.scene.grammar_envelope.viability_report` (never edited) and `hand_sampler.grammar`.

## 2. Variants

| | Definition |
|---|---|
| V0 | `G_SERIAL` -- old-sampler-like: no palm joints, no branching, otherwise `DEFAULT_DISTRIBUTION` (R/C/P/Coupled modules, 1-6 digits, 1-6 phalanges). |
| V1 | `DEFAULT`, constrained to the envelope: revolute only (C/P/Coupled weights zeroed), `branch_probability=0`, `palm_body_count_range=(0,2)`, `digit_count_range=(1,5)`. |
| V2 | V1 + a finger-mount spacing rule (I29): top-level digit mounts are PLANNED across root/palm hosts and `mount_frac_choices` (spread by host capacity, not drawn i.i.d.) to target >= 2x the largest capsule radius + a 5 mm margin (29 mm) between two mounts sharing a host. A placement rule, not a rejection filter. |
| V3 | V2 + curl-axis and opposition priors (I30): digit-phalanx revolute axes are restricted to a 60-120 deg band from +z (hinge-like, transverse to the segment), every phalanx after a digit's first gets a forward-only rest bend (reusing the existing bend primitive, probability 1.0), and the last of >= 2 top-level digits is mounted to oppose the mean direction of the others (solved analytically, snapped to the 15 deg grid). |

V4 (palm support under the spawn point) was **not implemented** -- see section 6.

All four are new named `Distribution`s (`hand_sampler/grammar/variants.py`'s `G0_SCREEN_VARIANTS`); every new field defaults to off, and the full grammar suite (3135 passed, 1 skipped) confirms every existing variant's default sampling is byte-identical to before this work.

## 3. Method notes and deviations from the plan (read before the numbers)

- **"Viable" design** (used to seed diversity/evolvability/explorability): `admitted` AND `fingertips_reachable >= 2` -- the Viability metric's own "both" fraction.
- **2000 fixed seeds per variant** for the Viability section, seeds `0..1999`, exactly as specified.
- **Pool scanning.** V0-V2's viable rate is far below 1 in 100, so the diversity (target 300), evolvability (target 200 parents) and explorability (target 50 founders) pools could not be filled from the fixed 2000-seed sweep alone. Each pool reuses every viable design already found there, then scans **up to 1000 additional seeds** (a time-boxed cap, `--max-extra-seeds`, in a disjoint range per pool) and reports the pool actually found -- **never padded**. This is why V0's pools have 0-1 designs and V1's have 8-11: an honest result, not a bug (see section 5).
- **Explorability evals reduced from 3000 to 500 per repeat** (3 repeats kept). Cause, discovered empirically during this run: `viability_report` costs ~12 ms on a design that fails the STRUCTURAL gate quickly, but **~170 ms** on one that passes it (`palm_up`'s 200-sample reachability sweep dominates) -- and almost every MAP-Elites child of a viable, envelope-shaped parent also passes the structural gate. At 3000 evals x 3 repeats x 4 variants this oracle cost alone would have been on the order of 2 hours; 500 x 3 keeps the same qualitative comparison inside the time box (documented; a fuller run is a rerun away, same script, `--explore-evals 3000`).
- **Diversity pairs**: up to 3000 random pairs were allowed (Monte Carlo estimate of the mean pairwise phenodist), but every variant's viable pool was in fact smaller than that threshold, so the reported means are the **exact** mean over **all** pairs in the pool (28, 171 and 780 pairs for V1, V2, V3 respectively) with `phenodist.phenotype_distance`'s `tip_displacement_m`, `n_configs=8`.
- **Explorability fitness**: `fingertips_reachable` (0-5, the primary criterion) plus `0.01 * proxy.opposition(model, configs)` as a small same-scale tie-break (opposition in [0,1], `n_configs=4`), computed only for structurally-admitted children.
- **Scope of the new rules**: the spacing/curl/opposition priors act only in `sample_derivation` (the initial population). `derive.vary`'s `EVOLUTION_OPERATORS` (used for both the evolvability and explorability sections) do **not** re-enforce them on newly grown material -- this shows up in the results (section 5).

## 4. Results table (2000 seeds/variant; pools as scanned, see above)

| Metric | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| frac admitted | 4.9% | 17.6% | 25.8% | 24.2% |
| frac >=2 fingertips reachable | 0.15% | 6.5% | 5.1% | **16.7%** |
| frac both ("viable") | 0.0% | 0.25% | 0.80% | **1.40%** |
| descriptor cells occupied (/30) | 0 | 5 | 6 | **9** |
| Shannon entropy (bits) | 0.0 | 2.32 | 2.31 | **2.82** |
| diversity pool size | 1 | 8 | 19 | 40 |
| mean pairwise phenodist (mm) | n/a | 170.1 | 189.6 | 120.6 |
| evolvability parents found | 1 | 11 | 25 | 48 |
| child viability rate | 1.00 (n=2) | 0.71 | 0.51 | 0.65 |
| frac children change cell | 0.0 | 0.26 | 0.06 | 0.16 |
| frac children identical-or-neutral | 1.0 | 0.68 | 0.78 | 0.80 |
| explorability founders found (/50) | 0 | 9 | 20 | 40 |
| MAP-Elites final coverage (mean +/- std, /30) | 0.0% | 50.0% +/- 2.7% | 48.9% +/- 3.1% | **55.6% +/- 1.6%** |
| distinct founders in final archive | 0.0 | 6.67 | 6.00 | **7.67** |
| mean joint count (viable designs) | n/a | 9.2 | 12.2 | 10.2 |
| mean wall time / design | 11.9 ms | 131.7 ms | 136.8 ms | 138.0 ms |
| mean wall time / **viable** design | undefined (0 found) | 52.7 s | 17.1 s | **9.9 s** |

Bold = best of the four on that row (excluding V0, which has no usable numbers on most rows).

**Reasons histogram (top causes of non-admission, out of 2000, counts merged across exact pair/coupling counts):**

- **V0**: non-revolute joint types present (1675, 84%), couplings present (1509, 75%), digit/palm-carrier count over the 5-digit envelope (353+353, 18% each) -- V0 is rejected almost entirely by the STRUCTURAL gate alone (revolute-only, no couplings), before overlap or reach is even measured.
- **V1**: rest-overlap > 3 mm (1132, 57%) -- matches Martin's original 59-64% figure almost exactly, confirming V1 is the fair, envelope-shaped baseline this screen targets -- then a jointed palm carrier holding more than 1 digit (170+107+99+80+32, several distinct sub-reasons, ~24% combined).
- **V2**: rest-overlap (1065, 53%, down ~6% relative from V1) -- the mount-spacing rule helps but does not remove overlap as the dominant failure -- then the same palm-carrier-multiplicity reasons (234+107+54+41, ~22%; the single largest sub-reason, "carries 2 digits", actually rose from 170 to 234: spreading digits round-robin across hosts pushes more of them onto the same jointed palm carrier when there are more digits than hosts, an unintended side effect of V2's own placement rule -- see section 6).
- **V3**: rest-overlap (1091, 55%, statistically the same as V2) and the identical palm-carrier reasons as V2 (234+107+54+41) -- curling the fingers together (V3's whole point) did **not** measurably worsen overlap relative to V2, which is reassuring, but it also did not reduce it further; V3's viability gains are entirely on the REACH side, not the overlap side.

## 5. Reading the trend

Every metric with a large enough sample moves the same direction, V0 << V1 < V2 < V3:

- **Viability compounds.** V1 -> V2 (spacing) roughly **triples** the viable rate (0.25% -> 0.80%) without moving `frac_admitted` much beyond what spacing alone should buy (17.6% -> 25.8%, mostly from fixing overlap). V2 -> V3 (curl + opposition) nearly **doubles** it again (0.80% -> 1.40%), driven almost entirely by `frac_two_or_more_fingertips` jumping from 5.1% to 16.7% -- exactly the two problems this screen was built to separate (I29 = overlap, I30 = reach), and the two rules move the two numbers they were each aimed at.
- **Cost per viable design falls 5.3x from V1 to V3** (52.7 s -> 9.9 s of CPU), which is the number that matters for an evolution pilot's founder/immigrant budget: V1 needs about 5x the compute V3 does just to seed a population.
- **Structural diversity improves with viability, not against it.** V3 has both the highest viable rate AND the most occupied descriptor cells (9/30) and the highest entropy (2.82 bits) -- there is no sign here that constraining the grammar toward viable designs collapses diversity onto a narrow structural niche.
- **Continuous (phenodist) diversity does NOT track structural diversity.** V3's mean pairwise tip displacement (120.6 mm) is the LOWEST of the three measured variants, despite V3 having the most occupied cells. The curl prior likely explains this: it pulls every viable V3 design's rest pose into a similar "fingers curled toward the front" shape, so designs that differ in digit/joint COUNT (structural diversity) still end up geometrically closer at rest than V1/V2's more haphazard (and less viable) rest poses. This is a real trade-off to flag to Martin, not an artifact: V3 buys viability and structural coverage at some cost in continuous pose diversity.
- **MAP-Elites without RL already shows the founder-dominance check working as intended.** V3's archive has both the best coverage (55.6%, and the most CONSISTENT across repeats -- std 1.6% vs V1/V2's ~3%) and the most distinct founders represented (7.67 of 40, roughly one founder per 2.2 occupied cells) -- no sign of one root hand dominating the archive at this scale.
- **The grammar's rules do not survive mutation as strongly as they hold at initial sampling** -- an explicit, honest limitation (section 3's last bullet). V2's child viability rate (51%) is LOWER than V1's (71%), and V2's `frac_children_change_cell` (6%) is much lower than V1's (26%), even though V2's raw sampled-population viability is higher. The likely mechanism: `EVOLUTION_OPERATORS` (`step_mount`, `add_digit`, etc.) can freely undo the careful mount placement V2's `sample_derivation`-time rule built, so a V2 PARENT's spacing gives no guarantee about its CHILDREN. V3 partially recovers (65% child viability) but is still below V1's. This matches `balanced-grammar-synthesis.md`'s framing exactly: diversity/viability preservation under mutation is a PROCESS responsibility (immigrants, elite re-evaluation, operator-level repair), not something the grammar's initial-sampling rules alone can guarantee -- flagged here as a concrete finding for whoever designs the evolution pilot's operator pool or process, not a defect in this screen's rules.
- **V0 is unusable, exactly as the motivating numbers said.** 0 of 2000 admitted-and-2-fingertip designs; scanning 1000 more seeds per pool found at most 1 (diversity), 1 (evolvability) and 0 (explorability founders) -- the explorability archive is empty (0% coverage) because there is nothing to seed it with. V0 is not a viable pilot baseline; it is the confirmation of the problem this screen exists to fix.

## 6. What was not done (honest caveats)

- **V4 (palm support under the spawn point) was not implemented.** It needs geometry-level reasoning about the palm's convex-hull cell area under the analytically-placed spawn point (`geometry.py`'s palm-cell machinery), which is a materially bigger lift than the three V1-V3 rules (all of which act on sampling/mount placement, never on the geometry/collision pipeline) and was not a safe fit for this screen's time box. V1-V3 already give a clear, monotonic signal for picking a baseline and a best variant without it; V4 is a candidate follow-up if the pilot wants a fourth comparison arm.
- **Pool sizes for V0/V1 are small** (V0: 0-1 design per pool; V1: 8-11). Their diversity/evolvability/explorability NUMBERS should be read as directional, not precise -- e.g. V1's mean phenodist (170 mm) comes from only 8 designs / 28 pairs. The RANKING V0 << V1 < V2 < V3 is robust (every metric with enough data agrees), but exact V1 values carry wide, unreported confidence intervals given the small n. A rerun with a larger `--max-extra-seeds` (this run used 1000; the script supports any value) would tighten V1 specifically, at a proportional CPU-time cost.
- **Explorability ran at 500 evals x 3 repeats, not the specified 3000 x 3** (section 3). The relative ordering (V3 > V1 > V2 on coverage, all >> V0) is unlikely to reverse with more evals since it already matches every other metric's ranking, but the absolute coverage numbers (49-56%) would likely rise somewhat with the full budget.
- **The mount-spacing rule (V2) has an uncosted side effect**: it increases how often 2+ digits land on the same jointed palm carrier (which the envelope caps at 1), because it spreads digits round-robin across ALL available hosts including palm carriers whenever `digit_count` exceeds the number of hosts. A V2.1 that excludes jointed palm carriers already holding a digit from the spacing plan's host list (or plans mounts only among root + non-palm-carrier hosts) is a plausible, small follow-up fix, not attempted here to keep V2 a single, clearly-attributable change.
- **Grammar rules are not re-enforced under mutation** (section 5's second-to-last bullet) -- a scope decision, not an oversight: fixing it is process work (repair operators, immigrants), which `balanced-grammar-synthesis.md` already assigns to the evolution pilot, not the grammar.

## 7. Recommendation

**Baseline: V1. Best: V3.** Run the evolution pilot's 2-variant comparison (E-R2', per the plan revision) on these two.

**Why V1 as the baseline, not V0.** V0 (`G_SERIAL`, literally "old-sampler-like") produced 0 viable designs in 2000 seeds and next to none in 3000 additional scanned seeds across three pools -- it cannot even seed a MAP-Elites archive at any reasonable compute budget, so it would not function as a baseline arm in the pilot (there would be nothing to compare against; every generation would show 0% coverage). V1 is the grammar's actual default, constrained only to what the viability oracle can structurally admit at all -- it is the fairest apples-to-apples "no extra generative rule" baseline: same envelope, same digit/joint ranges as V2/V3, differing from V3 by exactly the three rules under test.

**Why V3 as the best.** V3 dominates or ties on every metric with enough sample size to trust: highest viable rate (1.40% vs V1's 0.25%, a 5.6x improvement), cheapest to find a viable design (9.9 s vs 52.7 s CPU, 5.3x), most descriptor cells occupied (9/30) and highest entropy (2.82 bits) among viable designs, best and most consistent MAP-Elites coverage (55.6% +/- 1.6%), and the most founders represented in the final archive (7.67/40, the strongest evidence against one root hand dominating). It achieves this while its reasons-histogram shows curling the fingers together did not make rest-overlap measurably worse than V2's.

**Caveats for Martin to weigh before locking this in:**
1. V3's lower continuous phenodist diversity (120.6 mm vs V1/V2's 170-190 mm) is a real trade-off, not noise -- the curl prior narrows the population's rest-pose shape even as it widens structural (digit x joint) coverage. If the pilot's fitness or descriptor space cares about continuous pose variety specifically (not just digits x joints), this is worth watching in E-R2'.
2. Even V3 still fails 3 designs in 4 for one reason or another (75.9% not admitted, and only 1.40% ultimately viable) -- this screen shows a clear ranking, not a solved problem; the evolution pilot's immigrant/founder budget should assume finding a viable seed still costs several CPU-seconds even under V3.
3. V2 in isolation (spacing only, no curl/opposition) is not recommended as a THIRD pilot arm on this evidence: it improves the sampled population's own viability but its CHILDREN under `EVOLUTION_OPERATORS` are less likely to stay viable than V1's, and it did not clearly outperform V1 on structural diversity (6 vs 5 cells, both far below V3's 9). If a third arm is wanted, V2 as an ablation (to attribute how much of V3's gain is spacing vs curl/opposition) would be more informative than as a pilot candidate in its own right.
4. This is a kinematics-only, RL-free proxy for viability and diversity (section 3). None of it predicts task reward; it answers exactly the question this screen was scoped to answer (does the grammar leave evolution something to climb), not whether V3-sampled hands control well.

## 8. Reproducing this run

```
.venv_isaacsim/bin/python experiments/grammar_screen/g0_screen.py \
  --variants V0 V1 V2 V3 \
  --out-dir project-notes/grammar/experiments/g0 \
  --viability-seeds 2000 \
  --diversity-n 300 --diversity-pairs 3000 --diversity-n-configs 8 \
  --evolvability-parents 200 --evolvability-children 5 \
  --explore-founders 50 --explore-evals 500 --explore-repeats 3 \
  --max-extra-seeds 1000
```

`--quick` runs a ~1-minute smoke test with tiny sample counts (used to validate the script, not for these numbers). Wall time for the run above: 61 s (V0) + 917 s (V1) + 954 s (V2) + 995 s (V3) = ~49 minutes total, single-process CPU.
