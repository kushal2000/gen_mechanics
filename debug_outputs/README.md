# debug_outputs

Scratch space for things you look at once while diagnosing, then throw away.
Everything here is ignored by git except this file and the directory
placeholders — nothing in it should ever be an input to anything.

| | |
|---|---|
| `train_logs/` | logs pulled off a run for reading |
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

How a job ended comes from SLURM, not from a log file:

```bash
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,MaxRSS,Elapsed
```

That gives `TIMEOUT` / `OUT_OF_MEMORY` / `FAILED` plus peak RSS — which is also
the quickest check that a `--mem` request was sized right.
