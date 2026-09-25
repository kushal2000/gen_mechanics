# Hand-kinematics grammar: experiment log (append-only)

Plan: `~/.claude/plans/floating-spinning-waterfall.md` (approved 2026-09-25). State: `STATE.json`.
Conventions: one entry per iteration; record commands, versions, seeds, artifact paths, results, and a keep/revise/reject decision. Benchmark commits precede implementation commits.

## Iteration 0 (2026-09-25): benchmark freeze

- Branch `martin/hand-grammar` created from `martin/random-experiments` (4b615f9).
- Wrote `hand_sampler/grammar_bench/tolerances.py` (frozen), `manifest.json` (14 hands: 5 dev, 6 held-out, 3 excluded; 9 committed with license evidence; Ability/Inspire dev but local-only), copied URDF text + license files for the committed hands from `~/karma/karma-data/all_urdfs/full_models_as_downloaded/` (sha256 in manifest).
- Design decisions after the Opus review are listed under `settled` in `STATE.json`.
- Environment: system `/usr/bin/python3` 3.10.12, numpy 1.26.4, scipy 1.8.0, sympy 1.14.0, pytest 6.2.5 (run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`); oracle env `/home/singularity/anaconda3/envs/piper/bin/python` (Pinocchio 3.4.0, MuJoCo 3.2.3, yourdfpy 0.0.60). No venv created yet (not needed for iteration 1).
- Decision: keep.
