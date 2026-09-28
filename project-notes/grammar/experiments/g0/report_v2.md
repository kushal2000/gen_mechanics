# G0 CPU grammar screen: rerun report (v2)

Branch `martin/hand-grammar`. Supersedes `report.md` (that file now carries a one-line pointer here; no other edit was made to it). Script: `experiments/grammar_screen/g0_screen.py` (rerun infra), `experiments/grammar_screen/g0_review_diagnostics.py` (before/after diagnostics, no oracle dependency). Run with `.venv_isaacsim/bin/python`, no Kit/Isaac/GPU.

**STATUS: DRAFT -- placeholder, real-run sections not yet filled in. Do not treat any number below the "Fixes" section as final.**

## 1. Question

Martin's question, unchanged from `report.md`: how does the grammar shape exploration of hand designs, allowing diversity without making the problem intractable? `opus-review-g0.md` (I39) found the previous screen's oracle and grammar both had artifacts that dominated the absolute numbers (though the V1 < V3 ordering was robust). This rerun fixes the grammar-side issues the review assigned to this worker (items 2, 5, 6) and reruns against the oracle fix (`[oracle-v2]`, a concurrent worker's commit, never edited here).

## 2. Fixes (rule changes, with before/after diagnostics)

### 2.1 Surface mounting + cross-host spacing (review item 2)

**Bug.** Digit mounts sat on the host capsule's CENTRE AXIS (`derive.py`'s `_process_digit`, `base_xyz = (0, 0, mount_frac * host_length)`). Two digits on the same host less than `mount_frac` apart landed less than `2 * capsule_radius` apart -- an overlap "by construction", independent of anything the oracle's overlap check does. The old V2 spacing rule (`_plan_top_level_mounts`) only reasoned about same-host axial spacing and could not reach the target on short hosts, and never reasoned about cross-host coincidences at all.

**Fix.** Two new `Distribution` fields (`hand_sampler/grammar/distributions.py`), both off by default:
- `mount_on_host_surface`: a digit's mount origin is offset OFF the host's centre axis by the host's own capsule radius, at an azimuth angle -- either drawn i.i.d. (V1s) or planned (V2s).
- `mount_min_separation_m` (existing field, reused): when combined with `mount_on_host_surface`, dispatches to a NEW placement rule, `derive._plan_top_level_mounts_surface` -- a greedy furthest-point search over every (host, `mount_frac`, azimuth) grid point's ACTUAL root-frame 3-D position (via a scratch forward-kinematics pass over the already-sampled root/palm bodies, `derive._host_transforms_from_steps`). A generative placement rule, not a rejection filter: it always returns exactly as many mount triples as there are digits.

This is implemented so that every EXISTING named `Distribution` (including `G_V1`/`G_V2`/`G_V3`) is untouched: the new fields default to `False`/`None`, and each is read only at a call site gated on it being non-default -- pinned by `test_g0_screen_v2.py`'s `test_v1_v2_v3_unchanged_by_review` (byte-identical `phenotype_hash` for 4 seeds each, recorded from an isolated snapshot of the pre-review code at commit `ccb5b66`).

**Before/after (n=1000 seeds, 4-5 digits, root length 20-80 mm, <=2 additional palm bodies -- `experiments/grammar_screen/g0_review_diagnostics.py`, `project-notes/grammar/experiments/g0/review_diagnostics.json`):**

| | V1 (before) | V1s (surface only) | V2s (surface + spacing) |
|---|---|---|---|
| exact mount coincidences (0 mm apart) | 1088 / 8100 pairs (13.4%) | 53 / 8100 (0.65%) | **0 / 8100 (0%)** |
| median pairwise separation (all pairs) | 22.5 mm | 26.9 mm | **40.0 mm** |
| median same-host separation | 16.25 mm | 22.8 mm | **31.8 mm** |
| fraction of pairs meeting the 2r+margin (29 mm) target | 40.0% | 46.3% | **73.1%** |

Surface mounting alone materially reduces exact coincidences (13.4% -> 0.65%) just by giving same-frac mounts different azimuths; adding the planned cross-host search (V2s) removes exact coincidences entirely and nearly doubles the target-meeting fraction (40.0% -> 73.1%) even at the SHORT end of the root-length range, by using azimuth instead of relying on axial fraction alone.

### 2.2 Curl-axis + opposition host-frame fix (review item 5)

**Bug 1 (phalanx 0 bent).** `sample_bend` (called from `_sample_phalanx` for EVERY phalanx index, including 0) applied V3's forward-curl bend to a digit's FIRST phalanx too. Phalanx 0's own joint origin composes with the digit's `mount_rpy` (`derive._compose_bend_rpy`), so this silently tilted every digit's mount frame by 15-45 deg AFTER the opposition prior had already committed to an orientation computed from the (pre-bend) `mount_rpy` -- the prior's premise (that `mount_rpy` is the digit's true rest-pose forward direction) was false.

**Bug 2 (wrong host frame).** The opposition prior averaged `mount_rpy`-derived forward vectors directly, as if every digit's `mount_rpy` were already expressed in a shared (root) frame. A digit mounted on a rotated palm body has `mount_rpy` in a DIFFERENT local frame than one mounted on root; averaging them without first rotating into a common frame is only correct for root-only hands.

**Fix.** Two new fields:
- `curl_skip_first_phalanx`: phalanx 0 never receives a rest-bend, regardless of `bend_probability` -- every later phalanx still bends. `G_BEND` (which deliberately bends phalanx 0 too, per I16/I11's real-hand mid-chain-rotation motivation) is unaffected (field defaults `False`).
- `opposition_use_host_frame`: the mean-forward direction of the earlier digits, and the last digit's own opposing target, are computed in the ROOT frame -- each digit's `mount_rpy` is rotated by its OWN host's accumulated rotation (`derive._host_transforms_from_steps`) before averaging/opposing.

**Before/after (n=1000 seeds, forced `palm_body_count_range=(1,2)`, `digit_count_range=(2,5)` to guarantee mixed-host coverage; "opposing" angle = angle between the last digit's root-frame forward direction and the mean root-frame forward direction of the earlier digits -- true opposition is near 180 deg; opus-review-g0.md's own "within 90 deg" is the FAILURE share):**

| | V3 (before) | V3s (after) |
|---|---|---|
| median opposition angle, overall | 106.4 deg | **175.1 deg** |
| median opposition angle, mixed-host only | 103.5 deg | **174.8 deg** |
| "opposing" digits within 90 deg of the others (failure), overall | 33.6% | **0.0%** |
| "opposing" digits within 90 deg of the others (failure), mixed-host | 36.5% | **0.0%** |

The fix eliminates the failure mode entirely in this sample (median lands within a few degrees of the analytically-intended 180 deg, the residual gap explained by the 15-degree grid snap in `_best_opposing_rpy`).

**"Shared hinge plane" claim.** `variants.py`'s old comment implied `digit_axis_elevation_band_deg` keeps successive phalanges' revolute axes in a shared plane. Measured directly (share of successive within-digit axis pairs within 20 deg of each other): V3 7.9%, V3s 8.1% -- unchanged by either fix, since the elevation band restricts each axis's elevation only, never its azimuth, and neither review item asked for a new axis-alignment rule. Per the review ("either deliver ... or drop it from the docs"), the claim is dropped from `variants.py`'s docstring; this is reported honestly here as a still-open gap, not fixed.

### 2.3 Explorability metric (review item 6)

**Bug.** The CPU MAP-Elites archive (`g0_screen.py`'s `run_explorability`) admitted a child whenever it passed the STRUCTURAL gate (`cell is not None`), not when it was VIABLE (`admitted` AND `fingertips_reachable >= 2`) -- almost every structurally-admitted child of a viable parent also passes the structural gate, so the archive was measuring "how much of the descriptor grid is structurally reachable", not "how much of it is reachable with a VIABLE design", a materially easier and less interesting question. Founder counts also differed between variants (9/20/40 in the superseded run) with no common target to compare against.

**Fix.** `_run_one_explorability_repeat`'s `try_insert` now requires `_is_viable_report(r)` before inserting. Every variant is now asked for the SAME founders target (`--explore-founders`, default 20 for this rerun); a variant whose scan cannot fill it is reported honestly via `founders_target_met: false`, never padded.

## 3. Variants

| | Definition |
|---|---|
| V1 | (kept, reference) `DEFAULT` constrained to the envelope: revolute-only, no branching, `palm_body_count_range=(0,2)`, `digit_count_range=(1,5)`. |
| V1s | V1 + surface mounting (2.1), i.i.d. azimuth. |
| V2s | V1s + planned cross-host spacing (2.1). Sibling of V3s (both built on V1s, not on each other), so the two fixes' effects can be attributed independently. |
| V3 | (kept, reference) V2 (old spacing rule) + the UNFIXED curl-axis and opposition priors (I30) -- includes the phalanx-0-bend and wrong-host-frame bugs (2.2). |
| V3s | V1s + the FIXED curl-axis and opposition priors (2.2). |

`hand_sampler/grammar/variants.py`'s `G0_SCREEN_VARIANTS` now has 7 entries (`V0`..`V3s`); the default rerun roster is `V1, V1s, V2s, V3, V3s` (V0/V2, both superseded, stay selectable via `--variants` for reference only). Every new `Distribution` field defaults to off; `hand_sampler/grammar_bench/tests/test_g0_screen_v2.py` pins `phenotype_hash` byte-identity for `G_V1`/`G_V2`/`G_V3` against a snapshot of the pre-review code (commit `ccb5b66`), and the full grammar suite (`hand_sampler/grammar_bench/tests`, see section 10) passed throughout.

## 4. Method notes (rerun-specific)

- **2000 fixed seeds per variant** (seeds 0..1999), all five variants sharing the identical seed range -- required for the McNemar pairing in section 6.
- **Wilson 95% CI** on the viable fraction (`wilson_ci`), reported alongside every `frac_both`.
- **Exact McNemar** (`scipy.stats.binomtest`, two-sided AND one-sided in each direction, matching `opus-review-g0.md`'s own convention) on the paired `viable` boolean, every pair of the 5 run variants, on the shared 2000 seeds.
- **Per-digit-count viable fraction** (digits 1-5): bucketed by the derivation's own SAMPLED `digit_count` (defined even for structurally-rejected designs, unlike the oracle's own `digit_count` field, which is `None` on structural failure), with Wilson CIs per bucket.
- **Explorability**: 3000 evals x 3 repeats, founders target 20 for every variant (section 2.3).
- **Multiprocessing**: the 2000-seed viability sweep, the explorability repeats, and the diversity/evolvability/founder pools' extra-seed scans (needed whenever a variant's viable rate is too low to fill its pool from the fixed sweep alone) all batch through one `multiprocessing.Pool`, `--n-workers` cores (default `cpu_count() - 4`, leaving >= 4 free). Timing check on the pre-oracle-v2 code (V1, the lowest-viable-rate variant, 16 workers, all three pools filled to target -- 300/300 diversity, 200/200 evolvability, 20/20 founders): 241.6s.
- **"Viable" design**, unchanged from the superseded report: `admitted` AND `fingertips_reachable >= 2`.

## 5. Results table (2000 seeds/variant)

<!-- TODO: fill in after the real run against [oracle-v2]. -->

| Metric | V1 | V1s | V2s | V3 | V3s |
|---|---|---|---|---|---|
| frac admitted | | | | | |
| frac >=2 fingertips reachable | | | | | |
| frac both ("viable") [95% Wilson CI] | | | | | |
| descriptor cells occupied (/30) | | | | | |
| Shannon entropy (bits) | | | | | |
| mean wall time / design | | | | | |
| mean wall time / viable design | | | | | |

## 6. Exact McNemar (pairwise, shared 2000 seeds)

<!-- TODO: fill in after the real run. -->

## 7. Per-digit-count viable fraction (1-5 digits)

<!-- TODO: fill in after the real run. -->

## 8. Diversity, evolvability, explorability

<!-- TODO: fill in after the real run. -->

## 9. Recommendation

<!-- TODO: fill in after the real run. -->

## 10. Test counts

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 timeout 1500 python3 -m pytest -p no:cacheprovider hand_sampler/grammar_bench/tests -q`: 3157 passed (as of commit c77aff7, before the real rerun).

## 11. Reproducing this run

```
.venv_isaacsim/bin/python experiments/grammar_screen/g0_screen.py \
  --variants V1 V1s V2s V3 V3s \
  --out-dir project-notes/grammar/experiments/g0 --tag v2 \
  --viability-seeds 2000 \
  --diversity-n 300 --diversity-pairs 3000 --diversity-n-configs 8 \
  --evolvability-parents 200 --evolvability-children 5 \
  --explore-founders 20 --explore-evals 3000 --explore-repeats 3 \
  --max-extra-seeds 20000

.venv_isaacsim/bin/python experiments/grammar_screen/g0_review_diagnostics.py \
  --n-seeds 1000 --out project-notes/grammar/experiments/g0/review_diagnostics.json
```
