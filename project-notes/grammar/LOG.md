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
