# debug_outputs

Scratch space for things you look at once while diagnosing a run, then throw
away. Everything here is ignored by git except this file and the directory
placeholders — nothing in it should ever be an input to anything.

| | |
|---|---|
| `train_logs/` | per-run `<jobid>.out/.err` (symlinked under the run name), the wandb local dir, and `*.ram_usage.csv` / `*.gpu_usage.csv` |
| `videos/` | rollout captures |
| `smoke_tests/` | output from one-off checks: a rendered URDF, a stage dump, a printed observation |

The usage CSVs are sampled every 30 s by `experiments/monitor_usage.sh`: RSS
and CPU% summed over the job's process group, and per-GPU utilisation, memory,
SM clock and power. They are what tells you whether a `--mem` or
`--cpus-per-task` request was right.

Durable artifacts do **not** belong here: checkpoints and per-run configs stay
under `train_dir/<project>/<group>/<run>/`, populations and their manifests
under `assets/urdf/generated/population/`.
