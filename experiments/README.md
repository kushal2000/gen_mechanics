# experiments

SLURM submit scripts. Copy
[`train_single_embodiment_sharpa.sub`](train_single_embodiment_sharpa.sub) —
it is the reference run, and every knob is an environment variable with a
default, so a variant is a prefix rather than an edited file.

```bash
sbatch experiments/train_single_embodiment_sharpa.sub
SEED=1 NUM_ENVS=8192 sbatch experiments/train_single_embodiment_sharpa.sub
```

## Name the run for what it is

`RUN_NAME` sets the folder under `debug_outputs/train_logs/` and
`train_dir/`, and the wandb run name — one label for all three.

```bash
RUN_NAME=debug_single_embodiment_sharpa_aug31 sbatch experiments/train_single_embodiment_sharpa.sub
```

`LOG_GROUP` picks the directory under `debug_outputs/` — `train_logs` by
default, `smoke_tests` for a short verification run.

These folders accumulate fast, and the point of the name is a directory
listing you can read. The default `sharpa_iiwa14_seed0_2026-08-31_12-40-59`
is unique but every run looks alike; purpose plus date both groups and sorts:

```
debug_single_embodiment_sharpa_aug31
sweep_friction_aug31
finetune_population24k_sep02
```

Keep the date in it — reusing a name writes two runs into the same folder.

## Ask for the minimum you need

Queue time is dominated by the size of the request, not by how busy the
cluster is, so size it from what runs actually use:

| | peak RSS |
|---|---|
| 24,576-env population run | ~34.5 GB |
| single-hand run | ~17 GB |

`--cpus-per-task=8 --mem=48000` is the current default — about 40% headroom
over the worst case observed. Do not go lower on CPUs: 1 CPU regressed Kit boot
from 733 s to 916 s.

Check a finished run with `sacct -j <jobid> --format=MaxRSS`, or watch it live
in the run's `usage.ram_usage.csv`. Both sample on an interval, so a spike
between samples is invisible — which is why the headroom is not trimmed
further.

## Pick the right GPU

```bash
#SBATCH --partition=portal
#SBATCH --exclude=portal-compute-01
```

`portal` has four nodes, and `portal-compute-01` is the odd one out:

| node | GPU | generation |
|---|---|---|
| `portal-compute-01` | 8x RTX **A6000** | Ampere, 2020 |
| `portal-compute-02/03/04` | 8x RTX **6000 Ada** | Ada Lovelace, 2022 |

Exclude 01. The naming misleads — NVIDIA dropped the `A` prefix for the Ada
generation, so "A6000" reads as the newer card when it is a generation older
and roughly half the FP32 throughput. The epoch budget in the reference script
was characterised on the Ada cards.

A6000s are plentiful *outside* `portal` (many nodes on `default_partition`
carry 8-10 of them). If `portal` is congested, that is where the short queue
is — at the cost of the slower card.

## Runs are time-bound

`SBATCH -t` is the budget — the reference script sets no `max_epochs`, and the
config's default of 1,000,000 never fires. The run trains until SLURM stops it.

`save_frequency: 3000` means a kill costs at most one checkpoint interval
(~2.3 h at the measured ~2.8 s/epoch), so pick `-t` for how long you want to
train, not for when you expect convergence.

One case still needs an epoch cap: comparing *two different hands* head to
head. The hand that simulates faster collects more gradient steps in the same
wall clock, which is indistinguishable in the results from it being better
hardware. That does not apply to a single morphology-conditioned policy, where
every design shares one run.

---

`old_experiments/` is an archive of pre-split runs. Several of its scripts
launch modules that no longer exist; it is a record, not working code.
