# Evolution-loop experiments

Runs a population of hands forward under the grammar in `hand_sampler/` and
records what the population becomes. The grammar answers *which hands are one
step away*; this answers *where a population of them ends up*.

## Running

```bash
python -m hand_sampler.experiments.run --out runs/null_s7 --gens 1000 --seed 7
python -m hand_sampler.experiments.run --out runs/null_s7 --gens 9000   # continues
```

The second call **continues** the first. Population and RNG state are both
checkpointed, so `1000 + 9000` is the same chain as `10000` in one go — locked
down by `tests/test_experiments.py`, because the failure mode is silent: a run
that restarts its RNG produces a perfectly plausible chain that is a different
one. Resuming with a different `--parents`/`--children`/`--seed`/`--mode` is
refused rather than spliced.

A run directory holds `config.json`, `stats.jsonl` (one row per generation,
appended and flushed, so a killed run keeps everything it finished) and
`checkpoint.pkl.gz` (~1.1 KB per hand, so ~22 MB at 20k).

| flag | default | |
|---|---|---|
| `--parents` | 2000 | selected per generation |
| `--children` | 10 | children per parent; population is the product |
| `--mode` | `random` | `random`, `min_joints`, `max_joints` |
| `--gens` | — | generations to run **now**, added to any already done |

One generation: select `--parents` from the population, mutate each
`--children` times with **one** operator drawn uniformly from the nine. A
mutation that cannot act returns the parent unchanged and is counted in
`null_rate`.

### Selection modes

`random` is no selection at all — the grammar's own prior, the null any fitness
result must be read against. `min_joints`/`max_joints` select on joint count at
the same truncation as the real loop; they are not fitness functions but a
**yardstick** for how fast selection can move a statistic, so a real run's rate
can be quoted as a fraction of it.

To plug in a real evaluator, replace `run.select`. Nothing else in the module
knows what a fitness function is.

## Figures

```bash
python -m hand_sampler.experiments.plots single     --runs runs/null_s7 --out one.png
python -m hand_sampler.experiments.plots envelope   --runs runs/null_s* --out env.png
python -m hand_sampler.experiments.plots starvation --runs runs/null_s* --out cheap.png
python -m hand_sampler.experiments.plots arms --runs runs/null_s* \
    --max runs/max_s* --min runs/min_s* --out arms.png
```

| figure | question |
|---|---|
| `single` | one run, six panels, including every structural operator's success rate |
| `envelope` | median + min–max across replicates — *is it equilibrated, or wandering?* |
| `starvation` | does the cheap end of the (performance, n_motors) front keep getting sampled? |
| `arms` | selection strength against drift |

Replicates of one experiment are drawn as one series with uncertainty, not N
coloured lines. Reference lines read their values from `genotype` — `MAX_FINGERS`
has already moved once, and a hardcoded one mislabels silently. A figure whose
field is missing from an older run says so on the panel instead of dying.

## Findings so far

From a 1/10-scale pilot: 200 parents × 10 children, 6 seeds × 800 generations
for the null, 3 seeds × 400 for each arm.

**Run replicates.** At 200 lineages the population *mean* random-walks: single
runs finished anywhere from 11.3 to 27.7 mean joints. Two flat readings in one
run are not a plateau. Judge equilibration by whether the spread **across**
replicates stops growing.

**The design space has two speeds.** Link length (~34 mm) and face share (~0.30,
i.e. roughly uniform over the three faces) settle by generation 100–300. Joint
count had *not* settled by 800 — its across-replicate spread was still growing
×2.57 between halves. Set run length by a stationarity criterion, not a number.

**Under no selection, complexity climbs.** 2.1 → ~5.3 fingers, 3.1 → 11–28
joints. The early part is largely arithmetic, not a ratchet: at the seed corner
`remove_finger` cannot act at `MIN_FINGERS` and `merge_links` cannot act on
one-joint fingers, so growth moves are available where shrink moves are gated.
The gap decays from +0.115 to ~+0.02 as hands leave the corner.

**Selection beats drift by ~13×.** At 10% truncation `max_joints` goes from 4.1
to the 42-joint ceiling in 50 generations (+75.6 joints/100 gen) against the
null's +5.9; `min_joints` holds 2.14 for 400 generations. So the drift bias will
not decide a fitness run's outcome — but it means the *objective* decides it
completely.

**The cheap end empties out.** Under the null, hands with the minimum two
fingers fall from 89% of the population to under 1% by generation 151–630, and
`joints_p10` climbs from 2.5 to 9.9 (to 22 in the worst run). `sample.py` seeds
that region deliberately. Since selection is ~13× stronger than drift, an
objective where performance rises with motor count vacates it *faster*, and no
choice of objective fixes that — only an explicit mechanism that reserves
samples per motor count (archive niching, or per-`n_motors` quotas in the
truncation step).

### Known limits

- Everything above is at 1/10 scale. Ratios and equilibration timescales should
  transfer; the *tail* will not — a 10× population preserves rare low-motor
  hands far better, so the starvation magnitude here is likely pessimistic.
- `min_joints` started at the floor, so it showed selection **holding** the
  cheap end, not **recovering** it from an already-drifted population.
- Under `random` the loop is neutral, so the population converges to the
  mutation kernel's stationary distribution — a property of the grammar, not of
  the population. Independent lineages (no selection, no resampling) sample the
  same distribution with no drift noise at ~1/10 the cost. Prefer them for the
  equilibrium question; keep the full loop for genuinely population-level
  effects such as diversity collapse.
