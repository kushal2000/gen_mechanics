# hand_sampler

Hand designs, in pure Python (no Isaac): design sampling, mutation and the grammar's checks run on CPU.

**Current**
- `grammar/` — Martin's hand grammar: what a hand can be, the Feetech rules (hands buildable from Feetech servos), the
  viability checks, commercial hands fitted onto it. The design space evolution searches. Overview:
  `GRAMMAR-OVERVIEW.md` on the `hand-grammar` branch; viewer: `experiments/grammar_viewer/`.
- `grammar_bench/` — the grammar's tests and benchmarks (analytic and oracle FK, commercial-hand fits).
- `robot_spec.py` — `RobotSpec`: what a task needs to know about an arm + hand (joints, links, palm, fingertips).
- `robot_param_constants.py` — every fixed number (arm, hand mount, …).
- `design_space.py`, `build.py` — the generated-hand design used before the grammar; their math helpers
  (mass / inertia, frames, palm keypoints) are used by the environments.
- `population_io.py`, `gen_init_pop.py`, `validate_design.py` — populations of those designs (robot specs like
  `gen_s<seed>_n<count>` resolve through them).

**Legacy** (the pre-grammar design space, kept because assets and tests still use it)
- `commercial.py`, `sharpa_capsule.py`, `overlay_fits.py`, `viewer.py` — commercial hands in the old design space
  (builds `assets/urdf/unified_dynamics_commercial_hands_replicas/`).
- `mutate_design.py` — mutations of the old design.
- `DESIGN.md` — the old design space's design document. `tests/test_grammar.py` tests the old design space, not
  `grammar/`.
