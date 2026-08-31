# debug_outputs

Scratch space for things you look at once while diagnosing, then throw away.
Everything here is ignored by git except this file and the directory
placeholders — nothing in it should ever be an input to anything.

| | |
|---|---|
| `train_logs/<run>/` | `slurm.out`, `slurm.err`, `usage.ram_usage.csv`, `usage.gpu_usage.csv` |
| `videos/` | rollout captures |
| `smoke_tests/` | output from one-off checks: a rendered URDF, a stage dump, a printed observation |

A run's durable output stays out of here, under the same name:

```
train_dir/<project>/<group>/<run>/
  wandb/                      config.yaml + the diff.patch for reproducing HEAD
  nn/  summaries/             checkpoints and tensorboard
  .hydra/                     the resolved config
```

How a job ended comes from SLURM, not from a log file:

```bash
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,MaxRSS,Elapsed
```

That gives `TIMEOUT` / `OUT_OF_MEMORY` / `FAILED` plus peak RSS — which is also
the quickest check that a `--mem` request was sized right.
