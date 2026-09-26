# What "evolve to generalize" asks of the grammar

Draft started 2026-09-26 10:50 (coordinator). Re-read of the brief: README ("which hand's policy retains the most performance when the world shifts"; follow-on: co-design as a training curriculum, and co-design for generalization and sim-to-real), the investigation note §2 (bilevel: train a controller on one distribution, optimize hardware for performance under shifts in objects, physics, sensing, disturbances, tasks; outputs: a hand optimized for generalization and a co-design algorithm; three separate claims: curriculum, controllability, generalization), and the design note (useful, physically plausible hands without unnecessary complexity; performance(hand, current policy) confound).

## 1. Goals, restated as demands on the grammar

| Goal from the brief | What it demands of the grammar (as opposed to the process or the controller) |
|---|---|
| The evolving population trains a shared controller that should work on hands we care about (SHARPA, real hands, the eventual physical build) | The grammar's sampling support must contain those hands, or be dense near them; the population's structural variety is the controller's curriculum, so the grammar must span it deliberately, not incidentally |
| The evolved hand must retain performance under shifts it was not selected on | The grammar must be able to express robustness-conferring structure: redundant digits, opposition geometry, palm articulation, couplings that reduce control burden; it must not price these out by construction; whether they are selected is the process's job (evaluation under shifts) |
| Sim-to-real for evolved designs | Realizability constraints belong in the grammar's parameter table (module dimensions, travel, radius, mount spacing) and in derived geometry, so that everything the search proposes could be built; the design note's "physically constrained option" |
| Economical capability, not minimal fingers | The grammar reports structural cost; it does not encode a preference |
| Interpretable search under a 40-generation-scale budget | Locality, reversibility, size-neutral operators (already measured: E1, E2, E7) |

## 2. Where the current grammar stands against these demands

- Support: no real hand is inside the sampling support; the named reasons are gridded limits/lengths/axes and the straight-rod digit convention. E11 (running) quantifies how much widening brings them in and what it costs in bits per joint.
- Robustness features: redundant digits, opposition, palm articulation and couplings are all expressible and reachable (E3); nothing in the grammar penalises them; the cost term is a process choice.
- Realizability: gridded lengths (15-80 mm), radii, mount fractions and palm-joint travel exist; torque, actuator size and clearance do not. The old sampler's validator had mount spacing and capsule clearance rules; the new geometry overlay has cell containment and adjacency but no clearance check across non-adjacent bodies yet.
- Curriculum: the population is whatever selection keeps; today nothing guarantees structural variety persists (founder dominance), which is a process problem that directly damages the curriculum goal.

## 3. Open analysis (this loop)
- E11: support widening levels vs real hands and bits-per-joint cost.
- Literature: which properties of a morphology training distribution drive controller generalization (coverage vs diversity vs realism vs scale).
- Then: a recommendation on the support (continuous vs grid, rest bend), on which structural variety the population should be forced to keep (curriculum), and on realizability constraints to add.
