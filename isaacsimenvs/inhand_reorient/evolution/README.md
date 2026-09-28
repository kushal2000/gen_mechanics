# Evolution pilot: MAP-Elites over grammar hands

Implements the evolution pilot (E-R2') from `project-notes/grammar/plan-rl-grammar-tuning.md`'s
"Revision, 2026-09-27": a MAP-Elites archive over grammar-derived hands, driven by one shared
RL controller whose weights carry across generations. Lessons applied from
`project-notes/grammar/coevolution-reuse-survey.md` (the old co-evolution loop): fitness comes
from graded per-design scoring accumulated during training (`design_scoring.py`), not pooled
episode returns; the archive is real MAP-Elites (cells + elites), not truncation selection; and
the whole run is **one process, one SLURM job** -- no self-resubmitting chains.

## CLI

```
python -m isaacsimenvs.inhand_reorient.evolution.driver \
    --variant G_V1 --seed 0 --generations 20 --designs 64 \
    --probes allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right \
    --num-envs 4096 --epochs-per-gen 200 --fitness train_tail \
    --run-dir outputs/evolution_pilot/run0
```

Run under the isaacsim venv (`.venv_isaacsim/bin/python3 -m isaacsimenvs.inhand_reorient.evolution.driver ...`)
with `PYTHONPATH` clear of anything else (e.g. `env -u PYTHONPATH ...` if a ROS/other Python
environment is sourced in your shell) -- the driver itself never boots Kit (no `isaaclab` import
anywhere in this package), it only shells out to `coevolution/train.py`, which does.

Key flags (see `driver.py`'s `parse_args` for the full list and defaults):

- `--variant NAME`: resolved through `hand_sampler.grammar.variants.NAMED_DISTRIBUTIONS` and
  `G0_SCREEN_VARIANTS` (in that order) -- **never hard-code a variant list here**; a new G0
  variant a concurrent worker adds to either registry is picked up automatically by name.
- `--designs N` (default 64): total population size per generation, including probes.
- `--probes`: comma-separated manifest hand ids, fixed identity every generation, logged but
  never entered into the archive. Default: `allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right`
  (the plan's "allegro_right, dclaw, sharpa ... and one more admissible commercial hand" --
  `leap_right` is the fourth: verified admitted via `population_file.projected_entry` alongside
  the other three, no Kit needed).
- `--fitness {train_tail,eval}`: see "Fitness" below. `eval` currently falls back to
  `train_tail` with a printed warning (see "eval fitness" below for why and what would be
  needed to wire it in for real).
- `--gen-timeout-s`: the `<cap>` in `timeout -k 30 <cap>` wrapping every generation's
  `coevolution/train.py` subprocess.
- `--target-windows` (default 15): `GENMECH_DESIGN_SCORE_WRITE_EVERY` is derived from this so
  at least 10 (the plan's minimum) design-scoring write windows fall inside one generation,
  given `--epochs-per-gen` and `--horizon-length` (16, the yaml's own `horizon_length`).

## File layout

```
isaacsimenvs/inhand_reorient/evolution/
    __init__.py
    archive.py     # MAP-Elites archive: descriptor, Candidate/Elite, insertion, reporting,
                   # JSON (de)serialization. Pure python + numpy (no isaaclab/torch import) --
                   # CPU-testable under plain pytest, no Kit boot.
    driver.py      # The generation loop: population assembly (hand_sampler + this package's
                   # own scene.grammar_envelope/scene.population_file, also CPU-only), shells
                   # out to coevolution/train.py per generation, polls design_scoring's
                   # per-design snapshot file while training runs, computes fitness, updates
                   # the archive, writes generations.jsonl/.csv and state.json.
    README.md      # this file
isaacsimenvs/inhand_reorient/tests/
    test_evolution_archive.py   # 40 tests: descriptors, insertion/replacement, founder
                                 # tracking, sample_parent, state round-trip/resume.
    test_evolution_driver.py    # 29 tests: variant resolution, population assembly (gen-0,
                                 # offspring, immigrant), train_tail fitness weighting,
                                 # checkpoint discovery, build_train_cmd argument
                                 # construction, state.json round-trip.
experiments/evolution_pilot/
    submit_template.sh          # the exact `cluster submit` command for a cluster run,
                                 # not yet submitted (see its own header for recommended
                                 # --designs/--epochs-per-gen/--generations/--time).
```

Per run (`--run-dir DIR`):

```
DIR/
    state.json             # archive + driver RNG state + generation index + last checkpoint +
                            # config hash -- atomic write, read on startup to resume.
    generations.jsonl       # one JSON object per completed generation (see "Per-generation log").
    generations.csv          # the same, flattened to scalar columns only, for quick plotting.
    gen_<k>/
        population.json     # this generation's population file (population_file schema 0.2).
        train.log            # coevolution/train.py's stdout+stderr for this generation.
        train/               # hydra.run.dir for this generation's train.py -- rl_games nests
                              # its OWN experiment_dir one level below this
                              # (train/<config_name>/{nn,last,best}/...; see find_last_checkpoint's
                              # own docstring), and per_design_scores_rank0.json lives directly
                              # under train/ (design_scoring.py's own convention).
```

## Archive design

`archive.py`'s descriptor is `(digit_bin, joint_bin)`: digit_count clamped to 1-5 (5 bins),
joint_count (an admitted design's total valid envelope joint slots, 0-32) bucketed into
1-5/6-10/11-15/16-20/21-25/26-32 (6 bins) -- 30 cells total, matching the plan exactly.

Each generation, `driver.build_generation_population` assembles the population as: every
current archive elite (re-derived from its stored derivation, to be RE-EVALUATED this
generation under the shared controller's new weights) + offspring/immigrants filling the rest
of `--designs` minus the probe count + the fixed probes. Offspring: a parent is drawn uniformly
from the archive's elites (`Archive.sample_parent`), mutated via
`hand_sampler.grammar.derive.vary(..., operators=EVOLUTION_OPERATORS)`, and accepted only if
`grammar_envelope.admit` admits the resulting model; up to `--max-offspring-retries` (default
16, per the plan) attempts, each `vary` call itself retrying internally up to 32 times. If none
of those land on an admitted design, an immigrant is drawn instead (a fresh
`sample_derivation` + `admit` retry loop, exactly generation 0's own sampling, with a new
founder identity). Generation 0 has no elites yet, so every non-probe slot is a fresh founder.

After training, `Archive.update_generation` folds this generation's candidates (every
re-evaluated elite plus every offspring/immigrant) into the archive with a **per-generation
batch argmax per cell**: whichever candidate submitted THIS generation for a given cell has the
highest fitness becomes (or remains) that cell's elite. This is deliberately not a running max
against a stale previous-generation value -- see `archive.py`'s own module docstring for the
corollary this implies (an elite that is not resubmitted in a generation's batch offers no
protection at all; the real driver never hits this, since every elite is always part of every
generation's population).

## Fitness

**`train_tail`** (default): `driver.compute_train_tail_fitness` reads every design-scoring
write-window snapshot the driver captured while training ran this generation (polling
`per_design_scores_rank0.json`'s mtime -- design_scoring.py itself is out of this branch's edit
scope beyond the new `evolution/` subpackage, so it can only be read from outside, not made to
retain history on disk itself), takes the last `--tail-frac` (default 30%) of those windows, and
computes each design's mean `graded_fitness` weighted by that window's own episode count. A
design with fewer than `--min-episodes-per-design` (default 10) total tail-window episodes is
flagged `low_confidence` (still reported, never dropped -- a low-confidence 0 is informative:
either the design never got a chance to run, or it is genuinely uncontrollable).

Boot/train timings, fps and window count are all derived from the SAME poll loop
(`run_training_subprocess`): `boot_s` is wall time until the first window appears (an
approximation -- design-scoring has no other externally observable "Kit is up" signal),
`train_s` is the remainder until the subprocess exits, and `fps` is
`total_envs * cumulative_steps / cumulative_elapsed_s` read off the LAST window (design_scoring's
own `_score_total_steps`/`elapsed_s` are cumulative since env init, not per-window).

### `eval` fitness (I38)

`evaluate_population.py`'s frozen-policy evaluation was blocked (I38, commit 03d4e06) by two
bugs in how it drove rl_games' vendored player against a `coef_cond`/`mixed_expl_learn_param`
SAPG network (see that file's own updated docstring and the fix at the top of its `run`
function): `BasePlayer.has_batch_dimension` was never set (the script's own manual step loop
never calls `BasePlayer.run()`'s `get_batch_size`), and the network's declared input width is
genuinely one column wider than the env's own observation (a per-env SAPG exploration-block id
the env never produces). Both are now fixed, confined entirely to `evaluate_population.py` --
verified against a real 15-epoch SAPG checkpoint (see the worker report for the exact repro).

`--fitness eval` is accepted by the CLI but **falls back to `train_tail`** with a printed
warning: wiring a full per-generation frozen-policy eval into the archive-selection loop (an
extra Kit-booting subprocess per generation, on top of the training one) was out of this pass's
time budget beyond the fix itself and the one-generation correlation check below. The
`compute_train_tail_fitness`-shaped hook is `run_generation`'s own `fitness_by_source` local; a
follow-up `--fitness eval` implementation would run `evaluate_population.py` here instead (or
in addition), on the SAME `population.json` and the checkpoint just produced.

**train_tail vs eval correlation**: see the worker report for the measured Spearman correlation
on one generation's designs (computed from a real checkpoint + population, both fitness methods
against the identical set of designs).

## State and resume

`state.json` is written atomically (temp file + rename) after every COMPLETED generation:
the archive (`Archive.to_dict()`), the driver's own `numpy.random.Generator` bit-generator
state (so parent selection / mutation / immigrant sampling resume the EXACT same random
stream), the last completed generation index, the last checkpoint path, the last success
tolerance, the id-minter's counter (so ids minted after a resume never collide with ids minted
before it), and a sha256 of the resolved CLI config (mismatches are warned about, not fatal --
see `main`'s own comment). Restarting the driver with the same `--run-dir` loads this file and
continues from `generation_completed + 1`; a generation that was interrupted mid-flight is
simply redone from scratch (no partial-generation state is ever persisted), which the tiny
end-to-end verification run's own kill/resume test exercises directly (see the worker report).

The driver never self-resubmits to SLURM (per this branch's own rule): a cluster job runs it
once, with an honest `--time` limit; the run either finishes within that limit or is killed and
picked up later by rerunning the identical command with the same `--run-dir`.
