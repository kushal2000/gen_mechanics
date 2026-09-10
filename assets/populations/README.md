# Populations

One JSON per population, written by `hand_sampler.population_io` (sampled) or
`hand_sampler.drift` (drifted). A run can name one of these as its
`ROBOT_SPEC`, and `run.sh` then records its sha256 in the run directory, so
"which hands did this run train" is answerable from the run itself.

Gitignored -- 18 MB for 24576 designs -- so **durability is this directory, not
git**. The two kinds are not equally replaceable:

| kind | rebuild | keep? |
|---|---|---|
| sampled, `gen_s<seed>_n<count>.json` | `python -m hand_sampler.population_io <name>`, ~10 s | disposable |
| drifted, `*_drift.json` | hundreds of rounds of a SEEDED walk through `mutate_design` | **the only copy** |

A drifted population is reproducible only while the mutation code holds still --
the same fragility that made a `gen_s<seed>_n<count>` name a poor way to pin a
population in the first place. Every file carries a `provenance` block (source,
seed, rounds, git commit, whether it settled) and a drifted one has a
`.trace.json` beside it with the per-round statistics, so the approach to
equilibrium can be inspected rather than taken on trust.

    python -m hand_sampler.population_io gen_s0_n24576
    python -m hand_sampler.drift gen_s0_n24576 --max-rounds 4000
    python -m hand_sampler.viewer png --population <name-or-path> --designs 0-8
    python -m hand_sampler.population_io <name> --verify     # sampled only
