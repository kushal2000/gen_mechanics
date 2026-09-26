# Plan: overnight research program on the grammar best suited to evolution

Status: DRAFT for approval (2026-09-26, ~01:00). Supersedes the M1 pilot plan, which is complete and closed (see `project-notes/grammar/STATE.json`, `LOG.md`, `opus-review-m1.md`). Branch `martin/hand-grammar`; nothing pushed.

## 1. Context

We now have a verified kinematic representation, a generator with productions and a separate parameter table, replayable derivations with variation operators, a URDF/JSON adapter layer, an honest coverage report, and a geometry overlay (capsules plus per-body palm cells; the bisector cutting-plane refinement is in progress as iteration 7b). What we do not know is whether this grammar is a *good substrate for evolution*: whether its mutations are local, whether interesting designs are reachable from simple seeds in ~40 generations, whether neutral drift bloats, how much of the space is redundant, and which productions and parameter choices help or hurt. The earlier notes (`design-grammar-and-evolution.md` §1, §5, §9) already showed the old sampler drifts toward complexity and blocks pruning; the Opus review measured our operators as coarse (regrow changes ~29% of joints).

Goal for tonight: answer "which grammar is best for evolution" with CPU experiments that measure evolvability properties directly, then converge on a recommended production set, operator set, and parameter table, with evidence and open questions written up for Martin.

Constraints carried over: work in `hand_sampler/grammar/` and `project-notes/grammar/`; no GPU, no simulator, no installs, no pushes; benchmark tolerances frozen; honest reporting; one Sonnet editing worker at a time; commits per kept step; the fixed 5×6 simulator envelope is an external contract and is not changed.

## 2. What "best for evolution" means here (measurable, no RL)

The co-evolution loop cannot be run tonight (needs Isaac and a GPU). Instead we measure the properties that determine whether any fitness-driven search over the grammar can work, using only kinematics and geometry:

| Property | Metric | Why it matters |
|---|---|---|
| Locality | Phenotype distance per operator application: fingertip-set displacement (root frame, 32 seeded configurations, optimal tip correspondence), joint-count and digit-count deltas, palm-cell volume change | Local search needs small edits to cause small changes (Opus review: regrow ≈29% of joints changed) |
| Reachability | Accepted-mutation count and success rate of a structural hill-climb from simple seeds to four target skeletons (staggered 5-digit anthropomorphic; 3-digit radial D'Claw-like; 2-digit prismatic gripper; arch-articulated palm) | 40-generation budgets need targets within reach |
| Neutral drift | Joint/digit/palm-body counts over 40- and 500-step neutral walks from simple seeds, per operator mixture | The old sampler bloats; we must know our bias before selection runs |
| Redundancy | Fraction of distinct derivations mapping to identical canonical phenotypes (canonical body/joint ordering and hash) among random samples and offspring | Redundant genotypes waste evaluations and confuse lineage statistics |
| Expressiveness vs. size | Constructs in support (existing audit) and design-space entropy estimate per grammar variant | Minimally expressive but sufficient |
| Proxy-fitness evolvability | Under cheap geometric proxies (below), (μ+λ) evolution per grammar variant: best proxy vs. generations, structural diversity, motor count, use of palm joints/couplings | Tests whether selection can exploit the grammar at all; declared as a diagnostic, not a task score |

Proxy fitness functions (pure numpy, defined in one module before any experiment, never claimed to predict RL reward):
- **Opposition**: fraction of digit pairs whose tips can come within 2r over sampled configurations.
- **Reach coverage**: volume of the union of fingertip reachable sets (sampled) in front of the palm, normalized by palm size.
- **Antipodal pinch**: existence of two tip contacts on a reference sphere in front of the palm with opposing surface normals within a friction-cone half-angle of 20°.
- **Structural cost**: motors + total link length / 0.1 m (from the notes §3), used only in cost-aware variants.

## 3. Grammar variants to compare

| Variant | Description |
|---|---|
| G-full | Current grammar 0.3: palm tree with optional palm joints, digits with branching, R/C/P modules, intra-digit affine couplings, gridded parameters |
| G-nopalmjoint | Palm bodies rigid (no palm joints) |
| G-nobranch | No in-digit branching |
| G-nocouple | No couplings (every joint a motor) |
| G-serial | Rigid single palm body, serial digits only (≈ the old sampler's topology, but with our mounting) |
| G-full+bend | G-full plus a "rest bend" production (continuation rpy and lateral offset on a phalanx), the construct every real hand needs (issue I11) |
| G-full+small | G-full plus small-step operators for every field (axis one grid step, limit choice neighbour, mount fraction ±1 step, coupling parameter neighbour) |

Variants are expressed as `Distribution` instances and operator mixtures; no forked code paths.

## 4. Experiments (each: hypothesis, seeds, command, artifact, keep/revise/reject)

| # | Experiment | Output |
|---|---|---|
| E0 | Instrumentation: `proxy.py` (metrics), `canonical.py` (canonical form + hash), `phenodist.py` (phenotype distance), experiment runner with seeds/versions recorded, writing `project-notes/grammar/experiments/<name>/{result.json,summary.md}` | infrastructure commit |
| E1 | Locality per operator on 1000 parents × 7 operators (+ small-step operators once E0 exists); distributions and medians | table + decision on operator set |
| E2 | Neutral drift: 256 seeds × {40, 500} steps × 3 mixtures (uniform; growth/shrink balanced; small-step-heavy) | drift curves; bias statement |
| E3 | Reachability hill-climb to the four targets from 2-digit seeds, 64 restarts each, budget 200 accepted mutations, for G-full, G-serial, G-full+bend | success rates; whether the bend production is needed |
| E4 | Redundancy: 10k random samples and 10k offspring, canonical-hash duplicate rate per variant | numbers; canonical-order decision |
| E5 | Proxy evolution: (μ+λ)=(32+32), 40 generations, 8 restarts, per variant × {opposition, pinch, reach} × {no cost, cost-aware}; report best proxy, diversity, motor count, palm-joint and coupling usage | plots as tables; which constructs selection actually uses |
| E6 | Synthesis: recommended grammar (productions, operator mixture, parameter table), with evidence and explicit open questions for Martin; update `design-grammar-and-evolution.md` with one new section | note + Opus review |

Runtime budget: every experiment ≤ 20 min CPU on 32 cores (multiprocessing by seed); E5 is the largest and is capped at 4 hours total across variants.

## 5. Loop procedure (one bounded step per invocation)

1. Read `STATE.json`; pick the next experiment or the next fix its result demands.
2. Implementation via a Sonnet worker (bounded, allowed files listed per task); analysis and decisions by me; Opus review after E3 and after E6 (2 reviews).
3. Run tests (`grammar_bench/tests`, ~2 min) after any code change; run the experiment with fixed seeds; record command, versions, git SHA, seeds, runtime, artifact path in `LOG.md`; keep/revise/reject in `STATE.json`.
4. Commit per kept step (code and experiment outputs separately); never push.
5. Pause conditions: three unproductive attempts on one step; any change outside `hand_sampler/grammar/`, `grammar_bench/tests/`, `project-notes/grammar/`; a result that would require changing frozen tolerances or the simulator contract; anything needing a GPU, download, or install.

Guardrails: ≤ 25 Sonnet worker runs, ≤ 2 Opus reviews, stop at 09:00 local or when E6 is written, whichever first. Cron loop cadence 10 min (fires only when idle).

## 6. First actions after approval

1. Review and commit the iteration-7b geometry result (bisector cutting planes); if its assertions fail, fix once, else record as open and proceed (geometry is not on the E1–E5 critical path).
2. E0 infrastructure.
3. E1.

## 7. Deliverables by morning

- `project-notes/grammar/experiments/E1..E5/` with results and summaries; `LOG.md` entries; `STATE.json` decisions.
- `project-notes/grammar/grammar-for-evolution.md`: the recommendation, the evidence table, and the open questions (rest-bend production, palm-joint travel, coupling semantics for underactuation, canonical form), plus a short section appended to `design-grammar-and-evolution.md`.
- Code: `hand_sampler/grammar/{proxy,canonical,phenodist,experiments/*}.py`, small-step operators, optional bend production behind a distribution flag; tests for each in `grammar_bench/tests/`.

## 8. Verification
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider hand_sampler/grammar_bench/tests -q
python3 -m hand_sampler.grammar.experiments.run --list        # experiments and their recorded seeds
cat project-notes/grammar/STATE.json; tail -80 project-notes/grammar/LOG.md; git log --oneline -30
```
Expected: suite green; each experiment folder has `result.json` with seeds, versions and SHA, and a `summary.md` with the decision; the recommendation note exists and states what was and was not established.
