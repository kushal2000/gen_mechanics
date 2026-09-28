# Hand-kinematics grammar: experiment log (append-only)

Plan: `~/.claude/plans/floating-spinning-waterfall.md` (approved 2026-09-25). State: `STATE.json`.
Conventions: one entry per iteration; record commands, versions, seeds, artifact paths, results, and a keep/revise/reject decision. Benchmark commits precede implementation commits.

## Iteration 0 (2026-09-25): benchmark freeze

- Branch `martin/hand-grammar` created from `martin/random-experiments` (4b615f9).
- Wrote `hand_sampler/grammar_bench/tolerances.py` (frozen), `manifest.json` (14 hands: 5 dev, 6 held-out, 3 excluded; 9 committed with license evidence; Ability/Inspire dev but local-only), copied URDF text + license files for the committed hands from `~/karma/karma-data/all_urdfs/full_models_as_downloaded/` (sha256 in manifest).
- Design decisions after the Opus review are listed under `settled` in `STATE.json`.
- Environment: system `/usr/bin/python3` 3.10.12, numpy 1.26.4, scipy 1.8.0, sympy 1.14.0, pytest 6.2.5 (run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`); oracle env `/home/singularity/anaconda3/envs/piper/bin/python` (Pinocchio 3.4.0, MuJoCo 3.2.3, yourdfpy 0.0.60). No venv created yet (not needed for iteration 1).
- Decision: keep.

## Iteration 1 (2026-09-25): kinematic core, URDF import, analytic + Pinocchio tests

- Hypothesis: a 1-DoF-primitive kinematic graph with verbatim URDF poses and affine couplings reproduces Pinocchio FK on Allegro/LEAP/Barrett at 1e-10 and sympy closed forms at 1e-12.
- Worker: one Sonnet 5 run (~10 min, 47 tool uses). Coordinator reviewed the diff (scope, frozen files untouched, independence lint, conventions in fk.py/coords.py/urdf.py, fixture contents) and re-ran the suite.
- Commands: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python3 -m pytest -p no:cacheprovider hand_sampler/grammar_bench/tests -q -s` -> 13 passed in 3.69 s. References: `/home/singularity/anaconda3/envs/piper/bin/python hand_sampler/grammar_bench/refgen/make_configs.py` then `refgen/oracle_fk.py` (Pinocchio 3.4.0, numpy 1.26.4), seed 20260925.
- Versions (test env): python/numpy/sympy 3.10.12 1.26.4 1.14.0; git e87e887.
- Evidence: sympy vs ours: offaxis_tree 180 poses max 2.6e-17 m / 3.6e-16 rad; coupled_finger 80 poses 1.4e-17 / 2.2e-16. Pinocchio vs ours: allegro 1540 poses 5.6e-17 / 1.9e-16; barrett 630 poses 5.9e-17 / 5.3e-16; leap 1562 poses 2.0e-16 / 4.1e-16. Pinocchio vs sympy cross-check ~1e-16. DoF = movable - mimic for all three (16, 16, 8). LimitConflict on coupled_finger joint3: excess 0.7 (hand-computed).
- Worker-decided conventions (accepted): LimitConflict image = composed affine image of the ultimate source's declared limits, excess = sum of both ends; continuous joints sample in [-2pi, 2pi]; load_urdf validates before returning; oracle hand_root handled by re-expressing full-model poses relative to the hand_root body.
- Artifacts: commits 5bf270d (benchmark), e87e887 (implementation); references in hand_sampler/grammar_bench/references/.
- Open: I2 (coupling dropped by hand_root cut not reported), I3 (70 vs 71 configurations when 0 is outside the box). Not done this iteration: URDF export, JSON, coupled real hands (Ability/Inspire local-only), SHARPA hand_root import, closures.
- Decision: keep.

## Iteration 2 (2026-09-25): JSON, URDF export, coupling-cut fix, SHARPA import

- Hypothesis: exact JSON round trip and a URDF exporter that re-emits verbatim poses and mimic tags reproduce the frozen Pinocchio references on the exported files; SHARPA imports with hand_root=left_hand_C_MC.
- Worker: one Sonnet 5 run (~7 min, 53 tool uses). Coordinator verified frozen files unchanged, local references ignored, scope, and re-ran the suite.
- Command: same pytest command as iteration 1 -> 45 passed in 4.39 s, 0 skips (oracle subprocess ran). References: `refgen/make_configs.py --hand-root left_hand_C_MC` + `refgen/oracle_fk.py` (Pinocchio 3.4.0), seed 20260925; local-only Ability/Inspire references generated into references/local (gitignored).
- Evidence: Pinocchio(export) vs Pinocchio(source): allegro 1540 poses 0.0; leap 1562 0.0; barrett 630 0.0; sharpa 2310 4.5e-16 m / 3.0e-16 rad. Ours vs Pinocchio: sharpa 2310 poses 4.7e-16 / 7.5e-16; ability 1120 poses 5.7e-17 / 4.2e-16 (4 mimic); inspire 1330 poses 1.2e-16 / 4.6e-16 (6 mimic with offsets). SHARPA: 22 movable joints, 10 links dropped above the cut.
- Conventions (accepted): placeholder effort/velocity 1.0 flagged in LossReport; frames export as zero-geometry links on fixed joints named <frame>__frame_fixed; export-oracle check limited to hands with frozen references.
- Design decisions recorded this iteration (from discussion with Martin): representation is the evolution substrate, not an importer; palm-ness is a per-body flag; palm geometry = hull of palm capsules at rest cut per palm joint by a plane containing the axis; closed-chain solver deferred.
- Artifacts: commits dfdb04b (benchmark), 82e46e0 (implementation). Loop job bb7e458d every 10 min, stop after 20:42 local.
- Decision: keep.

## Iteration 3 (2026-09-25): grammar productions, derivations, vary

- Hypothesis: typed grammar with separate parameter table and replayable derivations generates only valid models whose exports Pinocchio agrees with; support covers the listed constructs.
- Worker: one Sonnet 5 run (~15 min). Command: same pytest command -> 754 passed in 35.7 s. Oracle: 200 seeds, 66,476 body poses, max 3.9e-16 m / 1.1e-15 rad. Support audit over 1000 seeds printed in the test (see commit 403e3cb). vary: 7 operators x 100 seeds valid.
- Coordinator review: frozen files untouched; scope ok. Probe over 300 seeds: 0 non-palm bodies with >=2 child joints, 0 palm bodies with >=2 palm children. The worker mounted 'branch' digits on palm bodies (its convention 2) and chained palm bodies (convention 5), so the audit's 'branch' key does not measure branching. This is a coverage mislabel and is being fixed before the coverage report.
- Accepted conventions: segment length stored as a `<body>_tip` frame; Coupled.source as a relative phalanx index; coupled-joint limits = exact affine image of the source range; URDF re-import equality checked kinematically because palm/radius flags and tip frames have no URDF form.
- Artifacts: commits 403e3cb (benchmark), acf57e5 (implementation).
- Decision: revise (I6 opened).

## Iteration 3b (2026-09-25): real branching and palm trees (fix for I6)

- Worker: one Sonnet 5 run (~11 min). Command: same pytest -> 755 passed in 38.4 s. Oracle: 200 seeds, 84,613 poses, max 7.4e-16 m / 1.4e-15 rad. Audit over 1000 seeds: in_digit_branch 708, palm_tree 335, palm_joint 877, two_nonparallel_palm_joints 297, coupling_nonzero_offset 3392, coupling_negative_multiplier 1474, continuous 2683, prismatic 2702, nonperpendicular_axis 22690, nonidentity_mount_rotation 7766, one_digit 146, five_plus_digits 354, six_plus_phalanges 1280.
- Coordinator probe (independent of the test): 300 seeds -> 220 models with a non-palm body having >= 2 child joints, 109 with a palm body having >= 2 palm children.
- Conventions accepted: hierarchical digit ids; branch on a last phalanx spawns >= 2 sub-digits; branch-hosting phalanx cannot be deleted; branch ids are stable identifiers even after host renumbering.
- Artifacts: commits ab37531 (benchmark), 901a0ce (implementation).
- Decision: keep. M1 status: representation, import/export, JSON, grammar done and verified; remaining: coverage + held-out report (iteration 4).

## Iteration 4 + 4b (2026-09-25): coverage inventory, pilot report, structural digit count

- Worker runs 5 and 6 (Sonnet 5). Command: same pytest -> 810 then 813 passed (~40 s). 11/11 available manifest hands pass Pinocchio fidelity (max 4.7e-16 m); 3 excluded by license. All 11 'expressible', none 'in_support' (real limits, link lengths, axes off the sampler's grids/sets). Fixed the vacuous digit_count_out_of_range item by counting movable chains off the root for imports (ORCA counts as 1 because of its wrist joint; limitation). Report at project-notes/grammar/pilot-report.{md,json} (the repo ignores any results/ dir).
- Commits: 1ce530e, 68b34da, 644b21e, 6b7387b, 8b86062, 49b886e.
- Decision: keep, with I7 closed.

## Opus milestone review (2026-09-25): M1 partially met

- Second and last Opus review (see opus-review-m1.md). Kinematic fidelity, import/export, and generation criteria met; coverage report overstates 'expressible'; geometry data model defects (mounts off segment, palm segment gap, palm_body_count=0) must be fixed before the hull step; several honesty/hygiene items.
- Budgets: worker runs 6/6, Opus reviews 2/2. Loop bb7e458d stopped. Decision: pause for Martin; proposed iteration 5 recorded in STATE.next_action.

## Iteration 5 (2026-09-25): segment convention, root palm, mounts on segments (I8)

- Worker run 7 (Sonnet 5, ~10 min). Suite: 2314 passed in 45 s (coordinator run). Oracle: 200 seeds, 88,484 poses, max 5.1e-16 m / 1.3e-15 rad. GRAMMAR_VERSION 0.2.
- Coordinator probe over 300 generated hands: root is a palm with L>0 in all; every joint origin is (0,0,t) with 0<=t<=L_host; worst world-frame offset from the host segment 3.9e-17 m.
- Audit keys nonperpendicular_axis and two_nonparallel_palm_joints now computed from derived models (joint frame / root frame).
- Commits: 565854c (benchmark), dca4d7a (implementation). Decision: keep.

## Iteration 6 (2026-09-26): honest coverage, validator, lineage, test hygiene (I9, I10)

- Worker run 8 (Sonnet 5). Suite: 2825 passed in 67 s (coordinator run). Commits 02739fe (benchmark), 3acff20 (implementation), report regenerated on a clean tree.
- Coverage now reports topology_expressible = no for every available real hand: 9/10 on continuation_pose (mid-chain frame rotation or lateral offset the straight-rod convention cannot express), Ability/Inspire also coupling_limits_not_image, ORCA on excess_children and fixed_in_digit. in_support = no for all (limits/lengths/axes off the sampler's grids). This is the honest picture; the earlier 'expressible: yes' was vacuous.
- Coupling sources restricted to revolute (3961 couplings over 1000 seeds, 0 non-revolute). Derivation validator rejects junk steps. vary records lineage; founder seed unchanged.
- Opened I11 (rest-bend production) for Martin's decision. Decision: keep. M1 closed.

- Note (2026-09-26): the evaluator must run as a module (`python3 -m hand_sampler.grammar_bench.evaluate`); as a script it cannot import the package under system Python. Dirty flag is now read before any work (commit b194651); report regenerated clean (5948cb2).

## Iteration 7 (2026-09-26): geometry overlay v0

- Worker run 10 (Sonnet 5). Suite 2835 passed in ~80 s. geometry.py: numpy convex hull, half-space clipping, cutting planes, analytic capsule and exact polytope mass properties, OBJ + URDF collision export. Over 200 seeds: mean 2.69 palm bodies/cells; 30% of hands hit the degenerate-normal fallback; 18 zero-volume cells; 104 hands with a digit mount outside its nominal cell.
- Finding: the literal rule (normal from the parent's origin to the joint) cuts the parent's own segment at every branch point. Refinement chosen: bisector plane between parent and child spines (still through the joint, still containing the axis); siblings by nearest spine; shared cut faces => adjacency. Committed as v0 with the gap stated; decision: revise (I12).

## Overnight program started (2026-09-26T01:24-04:00): which grammar is best for evolution

- Plan: plan-overnight-2026-09-26.md (approved). Loop job 18e9e258 every 30 min; stop after 09:24 local.
- Iteration 7b finding (worker): the bisector rule n = perp_a(d_c - d_p) does not keep the parent's spine on the parent side when the palm-joint axis is not perpendicular to the parent spine (seed 0: a digit mount 5 mm before the branch point lands 4.98 mm outside its cell). Decision: cutting plane contains BOTH the joint axis and the parent spine (n = component of d_c orthogonal to span(a, d_p)); child side is the side of the child's spine; points on the plane count as inside for containment. Fallback when a is parallel to d_p: n = perp_a(d_c).

## Iteration 7b (2026-09-26): cutting-plane partition abandoned

- Worker proved on seed 10 that when three or more palm bodies entangle, no single supporting plane both conserves hull volume and keeps both bodies' spines inside their cells (each body's own plane covers 74-95% of the hull). Across 200 seeds: 59 unresolved, 18 with real volume loss (up to 73%), 14.5% degenerate normals. Gram-Schmidt refinement (plane contains axis and parent spine) and the ancestor-safe walk were implemented and are correct for parent-child pairs but insufficient for siblings.
- Decision: nearest-spine cells (iteration 7c) with an honest contract (coverage + containment + recorded adjacency; overlaps allowed among adjacent pairs; no exact partition). Geometry is off the critical path for E0-E5.

## Iteration 7c (2026-09-26): nearest-spine palm cells (I12 closed)

- Worker run 11 (resumed twice). Full suite 2834 passed in 377 s; geometry test alone ~5 min at 200 seeds, so the routine suite now runs 40 seeds (env GRAMMAR_GEOMETRY_SEEDS=200 reproduces the full run). 200-seed statistics: 2.69 palm bodies/cells per hand; min cell volume 4.96e-6 m3; mean pairwise overlap 7.4% of hull volume; 2.49 overlapping pairs per hand (of 2.88 possible), all recorded as adjacent; union deficit 4.8% (reported); mounts outside 0; spines outside 0; hull samples uncovered 0. Acceptance tolerance 1e-6 m for convexity/containment (float64 hull planes at thousands of points; documented with worst cases).
- Open question for Martin: with nearest-spine cells most palm pairs overlap and are filtered; if that is too permissive for contact, the alternative is a per-body capsule palm (no shared hull). Decision: keep.

## E0 (2026-09-26 ~02:55): experiment infrastructure

- Worker run 12. Suite 2852 passed in 124 s. proxy.py (opposition, reach_coverage, antipodal_pinch, structural_cost; diagnostics only), canonical.py (order- and name-invariant hash; 500 seeds -> 500 distinct), phenodist.py, variants.py (G_FULL, G_NOPALMJOINT, G_NOBRANCH, G_NOCOUPLE, G_SERIAL, G_FULL_SMALL), derive.SMALL_STEP_OPERATORS (opt-in), experiments/runner.py (multiprocessing, provenance). Decision: keep.

## E1 locality + E2 neutral drift (2026-09-26 ~03:20)

- Worker run 13. Suite 2858 passed in 126 s. E1: 1000 parents x 12 operators, 57 s wall. E2: 256 seeds x 3 mixtures x 500 steps from reduced G_SERIAL starts, 3.4 s wall. Results in project-notes/grammar/experiments/E1_locality and E2_drift (result.json + summary.md, seeds and SHA recorded).
- E1 reading: small-step operators: tip displacement median 0.8-4.1 mm, p90 3-21 mm, joint count unchanged, null rate <= 0.8%. Default operators: perturb_parameter 0.8 mm median (local); resample_parameter 24 mm median / 136 mm p90; insert/delete_phalanx 1 joint but ~100 mm tip displacement (whole distal chain shifts); add/remove_digit 6.8/7.6 joints mean, ~77 mm; regrow_subtree 5.7 joints mean, 27% of joints changed median, 75% p90.
- E2 reading: joints change from start: uniform +7.1 [6.3,7.9] at step 40, +9.3 [8.5,10.1] at 500; balanced +8.7 / +10.0; small_heavy +4.4 / +8.5. Digits +1.1..+1.8. No two seeds converge to the same phenotype. Growth is boundary-driven (shrink operators inapplicable at the small end) and plateaus near the distribution caps; equal proposal weights do not balance it (consistent with the earlier sampler audit).
- Decisions: keep both. Opened I13 (minimal structural operators, palm-body operators). E3 will add them before measuring reachability.

## E3 reachability + E4 redundancy (2026-09-26 ~03:45)

- Worker run 14. Suite 2873 passed in 126 s. E3 wall 2.9 s; E4 wall 8.9 s. Results in experiments/E3_reach and E4_redundancy.
- E3 reading (64 restarts, budget 200 accepted): DEFAULT operators reach anthropomorphic_staggered 100% (median 39.5 accepted), radial_3 97% (48.5), prismatic_gripper 100% (29), arch_palm 0%. DEFAULT+SMALL: 91% / 81% / 98% / 0%, slower (small steps dilute structural moves). MINIMAL (minimal digit/palm ops + small steps): 14% / 52% / 61% / 100% (arch in 28 accepted). Conclusion: palm articulation is unreachable without palm-body/palm-joint operators; full-size digit insertion is what makes the anthropomorphic target fast. Recommended pool for evolution: union of default structural, minimal structural, and small-step operators, with mixture weights to be tuned in E5.
- E4 reading: 10,000 random seeds -> 10,000 distinct phenotypes for G_FULL and G_SERIAL; canonical order differs from derivation order in 90% (G_FULL) / 72% (G_SERIAL) of models (derivation encoding is redundant in order, hash handles it); null-mutation rate 5.4% default, 3.7% small-step (G_FULL); identical sibling offspring 1.7-2.7%.
- Decisions: keep both; I13 fixed.

## Opus review of E1-E4 (2026-09-26 ~03:45): readings revised

- See opus-review-e1-e4.md. E1's structural-operator displacements were measurement noise (misaligned configurations); the "distal chain shifts" reading in the E1 entry above is withdrawn. E2 characterised G_SERIAL only; under the union pool on G_FULL neutral drift reaches ~20 joints in 40 steps (reviewer probe). antipodal_pinch is buggy. Root length and radius were never mutated.
- Decisions: E5 stopped before completion (its results would be uninterpretable); I14 opened with the fix list; E1/E2/E3 to be re-run under the recommended pool before E5. Opus reviews used: 3 of 4.

## I14 fixes + E1-E4 re-run (2026-09-26 ~04:25)

- Worker run 16. Suite 2902 passed in 132 s. Wall: E1 62 s, E2 6.7 s, E3 10.5 s, E4 8.8 s. Stale e5_evolve.py from the stopped worker deleted.
- E1 (aligned, 1000 parents): perturb 0.8 mm; step_radius/coupling/limits 0 mm tip motion; step_root_length 2.5 mm; step_mount 4.1 mm; add_digit 10.0 mm; remove_digit 11.4; insert_phalanx 14.0 (legacy unaligned 99.2, the artifact); resample 15.2; delete_phalanx 16.1; regrow 44.1.
- E2 (128 seeds, 400 steps, final joints, paired 95% CI): G_FULL 32.7 [29.3,36.4] (DEFAULT) / 33.2 / 32.5 (UNION_weighted); G_NOBRANCH 14.4 / 14.6 / 15.7; G_FULL_INS 8.9 [8.1,9.7] / 8.8 / 9.1; G_NOBRANCH_INS 8.6 / 9.0 / 9.1. The insertion distribution, not the operator mixture, controls neutral bloat (3.5x lower than G_FULL).
- E3 (64 restarts, 1500 proposals): anthropomorphic, radial and prismatic targets 100% [0.943,1] under every pool x dist (median 26-211 proposals); arch_palm 0% [0,0.057] under DEFAULT (no palm operator) and 100% under UNION for all dists (median 46-55 proposals).
- E4: genuine null mutations 0% (default) / 1% (small-step); VariationImpossible 5.4% / 2.6-4.2%; 10k distinct phenotypes.
- Decision: keep. Provisional recommendation before E5: union pool with weighted mixture, insertion distribution for growth, branching optional (G_NOBRANCH_INS and G_FULL_INS behave alike under drift).

## E5 proxy-fitness evolution (2026-09-26 ~04:50)

- Worker run 17. Suite 2906 passed in 136 s. 288 tasks (48 conditions x 6 restarts), wall 690 s, no reductions. Results in experiments/E5_evolve.
- Reading: insertion variants best (G_NOBRANCH_INS 0.73 / 3.5 motors; G_FULL_INS 0.70 / 4.9) vs plain (G_NOBRANCH 0.55; G_FULL 0.65 / 6.1); pools equal on fitness, UNION fewer motors; small-step operators survive selection 42-49% vs 9-29% structural; cost term halves motors (3.1 vs 5.6) at -0.06 proxy. Constructs persist at 8-44% without being required.
- Decision: keep; recommendation written in grammar-for-evolution.md (G_NOBRANCH_INS or G_FULL_INS + union weighted pool + insertion prior + cost reporting/term).

## Final Opus review (2026-09-26 ~05:10): E5 write-up confounded

- See opus-review-final.md. Note flagged; consolidated note section 10 corrected to rest on E2/E3 only. I15 opened; E5b planned to match the simulator loop (envelope, noise, paired starts, no tie-break). Opus reviews used 4/4.

## I15 + E5b (2026-09-26 ~06:00): final evidence and write-up

- Worker run 18. Suite 2923 passed. E1 62 s, E2 6.6 s, E5b 185 s (54 conditions x 24 restarts). Commits through e938fcf.
- E5b reading: insertion prior slows growth to the 5-digit target (median gen 10 -> 14.5-16 under the default pool; 58-83% vs 96-100% reach under union pools); cost term lowers raw pinch 0.89 -> 0.68 while halving motors; palm operators 100% dead under the 5x6 envelope; clone rate < 1.3e-4.
- grammar-for-evolution.md rewritten on E1-E5b; design-grammar-and-evolution.md section 10 corrected. Program finished; loop stopped. Decisions for Martin listed in STATE.next_action.

## E7 mixture schedules (2026-09-26 ~06:25)

- Worker run 19. Suite 2935 passed. Wall 208 s (48 conditions x 24 restarts; diff hash d96ab7e6..., committed next). SCHEDULE_step matches the default pool on reach (CIs include 0) with late tip displacement halved (6-8 mm vs 14-18 mm) and beats the static weighted pool on reach rate (+0.04..+0.33) and first reach (6.5-15.4 generations earlier), all CIs excluding 0. SCHEDULE_linear lags under the insertion prior but scores highest on pinch. Recommendation item 3 updated. Decision: keep. Program closed.

## Literature reviews and synthesis (2026-09-26 morning)

- Two Sonnet reviews (robot morphology evolution; evolutionary methods in general) committed as lit-review-*.md. Synthesis with grammar/process/controller split and a 10-item balance test suite in balanced-grammar-synthesis.md. Key reframing: E2's drift is operator asymmetry (fixable in the grammar with exact-inverse pairs), founder dominance and late refinement collapse are process problems (immigrants, crowding, archive, age protection), controller fairness is Strgar & Kriegman's reset protocol.

## Generalization investigation (2026-09-26 10:48-11:20)

- Brief re-read (README, investigation note section 2, design note); lit review on morphology distributions and controller generalization (lit-review-morphology-distribution-for-generalization.md); E11 support-widening audit (rest bend is a prerequisite for every real hand; continuous limits bring 3 hands in; widened ranges 4; ORCA at L8; Barrett/D'Claw/Tesollo limit ranges outside; Ability/Inspire blocked by the coupling image-equality check). Suite 3018 passed. Recommendation in grammar-for-generalization.md section 6; issues I16, I17 opened. Loop stopped.

## Grammar 0.5 iteration A (2026-09-26 ~11:55)

- Worker run 20. 3037 passed. Rest bend (Pinocchio-checked on 20 G_BEND seeds), continuous limits (G_CONT), sign normalisation (no current hand affected; Barrett's ranges are out of range on either encoding), coupling containment (Inspire thumb couplings now accepted; 4 Inspire finger and 4 Ability couplings genuinely outside the image), I16 fixed (SHARPA at L7). Decision: keep.

## Grammar 0.5 iteration B (2026-09-26 ~12:45)

- Worker run 21. 3084 passed in 204 s. E12 balance suite: reversibility 100% (10/10 pair x dist); per-pair drift from minimum-size starts positive for digit/phalanx/branch pairs (applicability asymmetry at the boundary, plus the delete_phalanx aliasing defect); remove_palm_body p90 195 mm; histograms and redundancy pass. E2: EVOLUTION mixtures drift to 12-13 joints on G_FULL (vs 31 default) and 12-13 on G_FULL_INS (vs 8.7 default). Decision: keep code, revise criteria and fix I18 in iteration C.

## Grammar 0.5 iteration C (2026-09-26 ~13:35)

- Worker run 22. 3088 passed (210-290 s). I18 fixed. E12: reversibility 100%, locality per operator pass, histograms/redundancy pass, stationary drift residual small for digit and palm pairs, larger for the branch pair (applicability asymmetry). E3 with the evolution pool alone: arch 100%, anthropomorphic 14-20% (slow minimal moves). Coverage report regenerated: 8/11 real hands topology-expressible after the rest bend. Decision: keep; program paused for Martin. I19 opened (suite runtime).

## Representation check, items 1+2 (2026-09-26 ~15:10)

- A stray polling loop from the iteration-B worker (pgrep matched its own command, 6 h old) was found and killed; the first item-1+2 worker was stopped at Martin's request before editing and relaunched.
- Suite 3109 passed. project_to_derivation holds 12/12 available hands at ~1e-16 m / 2e-8 rad with joint counts conserved (allegro, leap, barrett, dclaw, wuji, xhand, tesollo, orca, sharpa, ability, inspire, coupled_finger). Notes: Barrett fingertips undefined; Ability/Inspire dependent limits replaced by the coupling image; SHARPA's pinky CMC stays a digit joint under the structural palm rule (annotation in item 3 makes it a palm joint). Decision: keep.

## Representation check, item 3 (2026-09-26 ~16:00)

- Manifest: SVH, Shadow (local, wrist cut at rh_palm), ARMS added as articulated_palm; palm_joints annotations (SHARPA pinky CMC, Shadow LFJ5, SVH j5, ARMS CMC4/CMC5; SVH's is redundant with the structural rule). Suite 3117 passed.
- E13 on a clean tree: 15/15 hands PASS at numerical precision (max 2e-13 mm, 1.7e-6 deg), joint counts conserved. Reported: Barrett and ARMS fingertips undefined; SVH index/ring spread and palm arch j5, and ARMS CMC5 couplings not in the structure (palm-joint couplings were silently dropped until the coordinator's fix in projection.py).
- Atlas grammar-gap table recorded as I21 for later grammar tuning. Decision: keep.

## Opus review of the representation check (2026-09-26 ~16:30)

- See opus-review-representation.md. E13 passes by construction (free SE(3) per joint origin); it confirms topology/joint types/code, not approximation quality. Atlas values for rest bend, axis angle, lateral offset, root length and coupling multipliers were gauge/pooling artefacts; I21 marked invalid. I22 opened with the fix list. Couplings scope stays deferred (actuation).

## Representation check, item 4: cross-section study (2026-09-26 ~17:00)

- E14 (piper env, allow_dirty with diff hash): 11 hands, 146 finger links, 413 sections (Shadow skipped: DAE only; Allegro via visual OBJ). Best single template h/w 1.00, r/h 0.45 (converges to a circle): 15%/41% of sections within 2/3 mm vs per-link capsule 15%/39%, per-hand radius 10%/23%, one radius 5%/17% (global radius 9.4 mm). Free rounded rectangles per section: median max error 2.7 mm vs circle 4.8 mm. PhysX docs: restOffset/contactOffset are per-shape but not documented as corner rounding; GPU convex hulls fall back to CPU above 64 vertices. Decision: keep capsules. I23 (runner --list quirk) opened.

## Representation check closed (2026-09-26 ~18:10)

- I22 fixes merged from the worker's worktree (merge 432dbe2): proper bend composition R(mount)R(bend) (byte-identical at zero), parallel-transport gauge, explicit palm set with a live branch check, one motor per joint (mimics independent), no wrists (ORCA cut at right_palm), fingertip n/a, Pinocchio check of exported derived URDFs, sha256 verification. Suite 3124 passed. E13: 15/15 PASS (14 real + fixture), max 1.1e-13 mm / 1.7e-6 deg; Pinocchio export check max 3.6e-13 mm. Worktree and branch removed after merge.
- Corrected atlas recorded in I21 (mount offsets median 22.5 mm on 94% of mounts; 18% of phalanx lengths outside 15-80 mm; palm length median 93 mm; limits 15-19% outside; physical bends median 0.9 deg, max 99). Representation milestone settled.

## RL phase 0: local stack (2026-09-26 ~21:40)

- uv 0.12.19 installed user-level (~/.local/bin, no profile changes). Pinned README recipe into .venv_isaacsim (Python 3.11, gitignored, 17 GB): torch 2.7.0+cu126 (CUDA on the RTX 4090), numpy 1.26.0, isaaclab 2.3.2.post1, isaacsim 5.1.0.0, rl_games resolving to third_party (README verify check passes), pytest in the venv. Install script and log: scratchpad install_isaac.{sh,log}.
- Cluster: cluster tool + slurm skill installed by Martin with local paths; Kerberos ticket valid; ssh control connection not yet open (Martin runs ssh -fN cluster).

## RL phase 0 (cluster) and phase 1 (2026-09-26 ~22:40)

- Cluster: cluster setup (env.sh pushed, uv present), repo synced to /data/pulkitag/users/mpeticco/code/gen_mechanics, venv built by cpu-tier job 2359859 (tig-cpu, 4 CPUs, 16 GB, 29 min, COMPLETED; log slurm/logs/gm-venv-2359859.out). GPU hours used: 0.
- Phase 1 (worker): isaacsimenvs/inhand_reorient (commits 9b7d501..ce3745e): URDF cutter, HandOnlySpec, InHandReorientEnv(PoseReachEnv), LSTM-MLP SAPG config. SHARPA 512-env smoke trains 5 epochs; kill-and-resume restores epoch and curriculum; dclaw boots. 31 package tests pass (venv, plugin autoload disabled). Gaps: no learning curve yet, counters not logged, palm-up orientation uncalibrated, allegro not booted -> Phase 1b running.

## Phase 1b result (2026-09-26 ~23:00)

- Worker (commits b7c8c0b, a9c6fd7, 73118b7): TensorBoard metrics (rot error, drops, timeouts, successes, tolerance, reward terms); palm-up calibration by drop test: sharpa, dclaw, allegro_right all had the palm facing DOWN under identity; the hand-only cut's local -z is the palm normal; base_rot = 180 deg about local x; the cube rests on the palm at zero action (score 1.0, 1-3 cm from the palm origin). Fixed allegro default joint positions outside limits. 47 package tests pass.
- SHARPA learning run (4096 envs, 2200 epochs = 144M steps, 22 min, 0.6 s/epoch, peak 5.9 GB GPU memory): no reorientation learning (rot error 2.21 -> 2.22 rad, i.e. chance; drops 100% -> 79%; tolerance stuck at 0.4). Allegro smoke (128 envs, 2 epochs) boots and exits 0. Decision: phase 1 acceptance NOT met; I24 opened; no cluster GPU jobs yet.

## Phase 1c: reorientation learns (2026-09-27 ~10:30)

- Root cause of I24: the palm-up calibration chose fingers-down; the cube rested on the wrist face out of reach. A read-only design pass flagged it (fingers run along root +z in every projected hand); the worker confirmed it in Kit (0/300 random joint sweeps brought a SHARPA fingertip within 3 cm of the resting cube).
- Other fixes (12fd6ab): action mapping centred on default pose; stale EMA targets after partial resets; first goal ignored the curriculum; IsaacGymEnvs AllegroHand reward and 0.24 m fall distance; object velocity and fingertip observations; articulation-order vs URDF-order joint names; importer-merged fingertip stubs.
- Learning, 15 min each, 4096 envs: allegro_right successes/episode 0.0002 -> 1.41, rot error 2.12 -> 1.27 rad; sharpa 0 -> 0.71, 2.16 -> 1.33 rad, drops 1.00 -> 0.75. Still improving at cutoff. Curricula did not advance (threshold 3 successes/episode).
- Kill-and-resume through rl_games restores epoch, tolerance and goal stage (350cacd adds a generic hook to the shared curriculum module, on this branch only).
- Phase 1 acceptance met. Open: dclaw calibration (I25), curriculum advancement (I26, 60-min run started). Kit can hang after SIGTERM: always `timeout -k` (I27).
- Coordination lessons: I approved a worker's edit to a collaborator-authored module without asking; Martin clarified the rule is branch-level (never touch other branches; shared files on this branch are fine). The worker also retried a commit after the auto-mode classifier denied it twice; reported to Martin.
- Phase 2 design saved: project-notes/grammar/phase2-adapter-design.md (padded 5x6 + 2 palm-carrier envelope, per-design tables, analytic palm-up).

## 60-min SHARPA run (I26), 2026-09-27 11:25

- 347M steps in 60 min (~100k fps, 4096 envs). Axis-only goals learned by 10 min (successes/episode 0.22, rot error 1.60 rad). At ~18 min both curricula fired in the same step (goal axis -> full, tolerance 0.40 -> 0.36); successes fell to 0.03 and rot error stayed near chance (~2.0 rad) for the remaining 40 min.
- Conclusion: full random-orientation reorientation is not learnable in ~1 GPU-hour with this setup; rotation about the palm normal is. Proposal: palm-normal-axis goals for E-R0/E-R1; decouple curricula; log the goal stage.
- Kit again ignored SIGTERM at the cap; `timeout -k 30` killed it (exit 137). I27 mitigation works.
- Phase 2 worker: its own forks collided with it (I28); forks stopped; 8 commits so far (envelope, population file, authoring, env wiring, CLI); Kit smoke in progress.

## Phase 2: grammar -> simulator adapter (2026-09-27 ~14:30)

- 11 commits (51be731..f81c3ca), all inside isaacsimenvs/inhand_reorient/. Padded 5x6 + 2 palm-carrier envelope; sha256-provenanced population files; per-env USD authoring; joint-order permutation between slot and PhysX order; CLI make_grammar_population.
- FK: CPU authored vs grammar FK ~1e-15; Kit readback vs grammar FK 4e-7 m / 1.2e-6 rad (q0 and random q); limits table vs PhysX exact.
- Kit bugs found and fixed: fixed-joint world anchor ignored env origin; per-body initial pose; rest self-penetration exploded ghost joints (153 rad) -> admission filter on rest overlap > 3 mm; stiffer carrier actuators.
- Training smoke with a 16-design population: 2 epochs, exit 0. Boot ~5.5 s for 64 envs.
- Admission: 13 commercial hands admitted; svh rejected (one palm joint carries 2 digits). G_SERIAL 100% and DEFAULT 76.5% structurally admitted, but ~75% of those fail the rest-overlap filter (I29); reach of random designs is poor (I30). Residual gaps: I31.
- Tests: 78 package + 7 curriculum pass; grammar suite 3124 passed, 1 skipped.

## E-R0 prep, part 1 (2026-09-27 ~15:00)

- b6dab74: rotation about the palm normal is the default task; curricula decoupled (I26). 7978f49: drop detection along world z (review item 2; all six calibrated single hands use local +z, so past single-hand runs were unaffected). 6715211: dclaw calibrates with per-joint curl (I25).
- xhand/wuji/tesollo scored 0 in calibration: the loader used kinematics-only fixture URDFs, so no collision geometry (I33). The worker concluded the meshes did not exist; they do, in the manifest source_root. Part 2 worker fixes the resolver, calibrates, and runs 15-min gates for all six hands plus a resume check.
- Cluster needs the external model tree synced (183 MB) via `cluster sync <dir> hand_models`.

## First cluster job (2026-09-27 ~16:50)

- Synced a clean `git archive` of 07a43be (secret scan: only a doc placeholder) to code/gen_mechanics, and the hand-model tree (183 MB) to code/hand_models.
- Smoke 2364675 (main tier, RTX A6000, 30-min limit, 20.5 min elapsed, 0.34 GPU-h): dclaw trained 42M steps in 12.9 min at ~66k fps (4090: ~250k fps); learning per step matches local. Boot 7.5 min: Kit startup 300 s, URDF->USD conversion 120 s (cold caches). Kit ignored SIGTERM again; `timeout -k` killed it (137), as designed.
- Cache test running on RTX 6000 Ada nodes: 2364778 (cold, saves a Kit cache seed to NFS) then 2364779 (warm, afterok).
- Found a silent-failure bug in the cluster tool's ada/ngpu self-cap check (I35); used the main tier with the ada6000-shared partition instead.
- Survey of the old co-evolution loop saved (coevolution-reuse-survey.md).

## Cluster throughput investigation (2026-09-27 ~17:30), total 0.98 GPU-h so far

- RTX 6000 Ada jobs (main QOS, ada6000-shared partition): boot <1 min (the 7.5-min boot was node-specific to improbablex005); training ~0.88 s/epoch, ~74k fps at 4096 envs (4090: ~0.26 s, ~250k).
- 16384 envs: ~90k fps (1.25x); GPU util ~24-35% throughout. 16 CPUs vs 8: no change. Capping Kit and PhysX thread pools at 8: no change. Kit cache seed: no change.
- Conclusion: the env step is CPU single-thread bound on the cluster's 3.5 GHz EPYC cores. 1 local 4090 ~ 3.4 cluster GPUs. Use the cluster for parallel runs; profile the env step for per-step overhead.

## Spawn-frame worker (2026-09-27 ~17:50)

- The frame math was consistent; the symptom came from two real bugs: xhand's "fingertips" included the palm root and cosmetic stubs (3fefee7, now 5 real tips), and calibration never required the spawn point above the palm (a17d14a, MIN_SPAWN_HEIGHT_ABOVE_PALM_M = 0.02, the same quantity the drop check uses). Recalibrated (dc539d7): all six hands spawn 0.14-0.21 m above the palm.
- 15-min gates v2: tesollo starts to learn (drop 1.00 -> 0.93, 0.045 goals/episode); xhand and wuji still 100% drop. Remaining cause: merged fixed-joint fingertip bodies sit 2.6-5 cm from the tip surface (I36). Not blocking the pilot.
- Single-hand baselines that learn from scratch in 15 min locally: dclaw (4.1 goals/episode), allegro (1.4), sharpa (0.71).

## Pilot prep: population fixes, graded scoring (2026-09-27 ~20:00)

- 8 commits (b390acb..03d4e06). Review bugs fixed except palm-cell colliders (I30, deferred). Faithful admission gate: G_SERIAL 21.5%, DEFAULT 17.7% admitted. `viability_report(model)` for the CPU screen.
- Graded per-design scoring banked in the env: graded_fitness = goals + 0.5 clip(rotation_progress/pi) + 0.25 min(time_held/episode_max, 1), per write window with curriculum snapshot.
- Kit: 16-design smoke FK 0.0005 mm, homogeneous, limits exact, ghost |q| max 0.0126 rad, 39% of cubes resting at 100 zero-action steps (0-75% per design). 10-min 32-design run at 4096 envs: ~113k fps; graded_fitness spread 0.02-0.14 (6.7x); aggregate successes flat at this horizon.
- Frozen-policy eval blocked by the vendored rl_games player (I38).
- Next: evolution driver (MAP-Elites, in-job generation loop with resume, probes = projected commercial hands); G0 screen running.

## G0 CPU grammar screen (2026-09-27 ~21:00), under Opus verification

- New off-by-default generative rules (95c577a): mount spacing (V2), curl-axis band and opposition (V3); existing variants byte-identical (phenotype-hash test). Grammar suite 3135 passed, 1 skipped.
- 2000 seeds per variant, viability = admitted AND >= 2 fingertips reach the spawn point: V0 (unconstrained G_SERIAL) 0%, V1 0.25%, V2 0.80%, V3 1.40%. Rest overlap rejects 53-57% of V1-V3; V3's gain is on reach (>= 2 fingertips 16.7% vs 5-6.5%). Cells occupied 5/6/9 of 30; CPU MAP-Elites coverage 50/49/56% with 6.7/6.0/7.7 founders (500 evals x 3). V3 has the lowest continuous phenotype diversity.
- Recommendation: pilot compares V1 (baseline) with V3. Opus is checking the oracle for artifacts (root-capsule overlap, spawn placement) and the statistics before the pilot relies on it.

## Evolution driver and local timing run (2026-09-27 ~22:30)

- Driver committed (3fa968a..77a19c7): MAP-Elites archive (digits x joint bins, 30 cells), in-process generation loop, resume from state.json (tested by killing mid-generation), probes = projected allegro_right, dclaw, sharpa, leap_right; eval player fixed (f4087f6: batch-dimension handling plus the SAPG block-id column). 237 package tests pass.
- Timing run on the local 4090 (G_V1, 64 designs, 4096 envs, 900 epochs/gen): boot ~155 s, train ~510 s, select <2 s, ~102k fps, i.e. ~11 min/generation. Coverage 8/8/9 of 30; QD-score 0.85/1.03/1.31; mean elite fitness 0.107/0.129/0.146; best ~0.18 flat; founders 8/8/7 (max share 0.22). Probe fitness flat (dclaw 0.13-0.14, others ~0.06). Oracle used here is the pre-fix one (I39).
- Cluster estimate: ~35 min/generation (training 3.4x slower, boot slower), so 8 generations ~ 4.7 GPU-h per run.

## Oracle v2 (2026-09-27 ~23:00)

- e9a9c26, 773385b [oracle-v2]: rest overlap uses the built capsule core [r, L-r] (matches author_grammar exactly), checked at q=0 and at the reset pose default_q; spawn = fingertip centroid at default_q + (object half-size + 5 mm) along the palm normal; dense batched-FK reach sweep (4000 samples, ~40 ms/design, faster than before); population schema 0.3 with palm_up in derived_sha256.
- Commercial projections: every 5-finger hand now reaches the spawn with >= 4 fingertips (LEAP 1 -> 4, xhand 2 -> 4, allegro 2 -> 3).
- 500 seeds: G_V1 1.2% viable (was 0.25%), G_V3 7.2% (was 1.4%).
- Kit: FK 0.0005 mm, homogeneous, 0 spawn-triggered drops, 42% of cubes resting at 100 zero-action steps.
- Eval-script hang fixed (f279df5), residual oddity I40. Package tests 250 pass.
