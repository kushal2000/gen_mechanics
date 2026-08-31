# debug_outputs

Where runs write. Everything here is ignored by git except this file and the
directory placeholders.

| | |
|---|---|
| `train_logs/<run>/` | everything one run produces — see below |
| `videos/` | rollout captures |
| `smoke_tests/` | output from one-off checks: a rendered URDF, a stage dump, a printed observation |

One directory per run, holding all of it:

```
debug_outputs/train_logs/<run>/
  slurm.out  slurm.err        stdout/stderr
  usage.ram_usage.csv         RSS + CPU% over the process group, every 30 s
  usage.gpu_usage.csv         GPU name, util, memory, SM clock, power
  nn/  summaries/             checkpoints and tensorboard
  wandb/                      config.yaml + the diff.patch for reproducing HEAD
  .hydra/                     the resolved config
```

`deprecated/` holds finished runs in the same shape.

How a job ended comes from SLURM, not from a log file:

```bash
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,MaxRSS,Elapsed
```

That gives `TIMEOUT` / `OUT_OF_MEMORY` / `FAILED` plus peak RSS — which is also
the quickest check that a `--mem` request was sized right.
