# debug_outputs

Scratch space for things you look at once while diagnosing a run, then throw
away. Everything here is ignored by git except this file and the directory
placeholders — nothing in it should ever be an input to anything.

| | |
|---|---|
| `train_logs/` | slurm stdout/stderr and hydra logs pulled off a run for reading |
| `videos/` | rollout captures |
| `smoke_tests/` | output from one-off checks: a rendered URDF, a stage dump, a printed observation |

Durable artifacts do **not** belong here: checkpoints and per-run configs stay
under `train_dir/<project>/<group>/<run>/`, populations and their manifests
under `assets/urdf/generated/population/`.
