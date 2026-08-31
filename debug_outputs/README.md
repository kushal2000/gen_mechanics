# debug_outputs

Scratch space for things you look at once while diagnosing, then throw away.
Everything here is ignored by git except this file and the directory
placeholders — nothing in it should ever be an input to anything.

| | |
|---|---|
| `train_logs/` | `bootstrap_<jobid>.{out,err}` — see below |
| `videos/` | rollout captures |
| `smoke_tests/` | output from one-off checks: a rendered URDF, a stage dump, a printed observation |

Everything a training run produces goes to its run directory instead, because
it is per-run provenance rather than scratch:

```
train_dir/<project>/<group>/<run>/
  slurm.out  slurm.err        stdout/stderr
  usage.ram_usage.csv         RSS + CPU% over the process group, every 30 s
  usage.gpu_usage.csv         GPU util, memory, SM clock, power
  wandb/                      config.yaml + the diff.patch for reproducing HEAD
  nn/  summaries/             checkpoints and tensorboard
```

`train_logs/` holds only `bootstrap_<jobid>.{out,err}`, and those are worth
keeping despite being empty on a healthy run. `#SBATCH -o` is evaluated before
the job starts, so it cannot name a run directory built from a timestamp — and
`slurmstepd` keeps writing to it after the script exits. **An OOM kill or a
time-limit cancellation is reported only there.** Of 35 runs logged under the
previous scheme, which pointed `-o` at `/dev/null`, not one recorded why it
ended.
