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

`train_logs/` holds only the bootstrap logs. SLURM evaluates `#SBATCH -o`
before the job runs, so it cannot expand a run directory built from a
timestamp; those first lines have to land at a fixed path. The script
re-points stdout/stderr into the run directory as soon as it exists, so a
bootstrap file with more than a couple of lines in it means the job died
early — a rejected `NUM_ENVS`, a failed venv activate — and that file is where
the error is.
