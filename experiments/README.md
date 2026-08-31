# experiments

SLURM submit scripts. Copy
[`train_single_embodiment_sharpa.sub`](train_single_embodiment_sharpa.sub) —
it is the reference run, and every knob is an environment variable with a
default, so a variant is a prefix rather than an edited file.

```bash
sbatch experiments/train_single_embodiment_sharpa.sub
SEED=1 MAX_EPOCHS=20000 sbatch experiments/train_single_embodiment_sharpa.sub
```

## Ask for the minimum you need

Queue time is dominated by the size of the request, not by how busy the
cluster is. A 24,576-env training run measured **32.7 GB MaxRSS against a
200 GB request** — so the memory ask was ~6x the actual use, and the CPU count
was what kept the job pending.

`--cpus-per-task=8 --mem=96000` is the current default. Do not go lower on
CPUs: 1 CPU regressed Kit boot from 733 s to 916 s.

## Pick the right GPU

```bash
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
```

`portal` has four nodes. `portal-compute-01` is the odd one out — 8x RTX A6000,
where `portal-compute-02/03/04` are RTX 6000 Ada. Exclude it; the Ada cards are
meaningfully faster and the run's epoch budget was characterised on them.

A6000s are plentiful *outside* `portal` (many nodes on `default_partition`
carry 8-10 of them). If `portal` is congested, that is where the short queue
is — at the cost of the slower card.

## Make the run end on epochs, not on the clock

`MAX_EPOCHS` ends the run; SBATCH `-t` is only a safety ceiling. This is the
one mistake here that fails *silently*: a hand that simulates faster collects
more gradient steps inside the same wall clock, which is indistinguishable in
the results from it being better hardware. Everything else in a bad submit
script fails loudly.

---

`old_experiments/` is an archive of pre-split runs. Several of its scripts
launch modules that no longer exist; it is a record, not working code.
