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
