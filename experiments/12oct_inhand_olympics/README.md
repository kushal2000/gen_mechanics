# 12 Oct: in-hand olympics

The flagship environment (`isaacsimenvs/inhand_olympics/`): a hand on a 6-DOF dummy wrist, palm down over a table, and a
pen. Tasks (`env.task.names`, a one-hot tells the policy which): `pen_lift`, `pen_spin` (on the table), `pen_spin_air`,
`pen_flip_air`, `pen_repose` (6D goals), `pen_twist` (twist a cap off; needs `env.pen.twist=True`). Since 15 Oct the
palm's rotation is locked by default (the wrist only translates).

## Training: `olympics_sapg_mlp.sub`

Every olympics run (this folder's and 14oct–15oct's) goes through this script. Every knob is an environment variable:

| Variable | Default | What |
|---|---|---|
| `TASKS` | `pen_spin` | comma-separated tasks, one drawn per env at every reset |
| `HAND` | `wuji2_left_uniform_handonly` | robot spec: a registry name, a primitive hand's `.spec.json`, or `multi:a+b+...` (several hands, envs dealt round-robin) |
| `ARCH` | `mlp` | `mlp`, `lstm` or `transformer` (joint-token transformer) |
| `NUM_ENVS`, `MINIBATCH`, `EPOCHS`, `SEED` | 24576, 114688, 5000, 100 | |
| `PEN_SHAPE` | `capsule` | `capsule`, `box`, or for the twist pen `round` |
| `CHECKPOINT`, `CHECKPOINT_MODE` | – | start from a `.pth` (`weights`: model only; `resume`: everything) |
| `EXTRA_HYDRA` | – | extra Hydra overrides, e.g. `"env.pen_repose.goal_sampling=uniform"` |
| `RUN_TAG`, `HAND_TAG` | – | labels in the run name |
| `LOG_GROUP` | `12oct_inhand_olympics` | folder under `debug_outputs/train_logs/` (match it with `sbatch --output`) |
| `WANDB_PROJECT`, `WANDB` | `gen_mechanics_inhand_olympics`, 1 | |

```bash
TASKS=pen_repose PEN_SHAPE=box HAND=<spec> LOG_GROUP=<experiment> \
  sbatch --output=debug_outputs/train_logs/<experiment>/%x-%j.log experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub
# a quick local check
NUM_ENVS=3072 MINIBATCH=12288 EPOCHS=20 WANDB=0 bash experiments/12oct_inhand_olympics/olympics_sapg_mlp.sub
```

## Evaluation and viewing

- `analysis/eval_tasks.py`: goals per episode per task for a run's checkpoints (stochastic actions).
- `perturb_ladder.py`: fine-tunes a pen_lift policy under pushes and twists 10x stronger at each stage, until it
  stops coping (the force ladder).
- `viser_play.py`: this folder's policy viewer (also plays task_curriculum's runs through `play_worker_task_curriculum.py`,
  which needs that repo). For fixed-palm policies use `experiments/16oct_viser_evals/viser_play_all.py`.

## Results

- [`embodiment_curriculum.md`](embodiment_curriculum.md): thumb + index first, then a 2 + 5 finger mix, unlocks the
  5-finger hand on pen lift. **Caveat:** the mix ran before the multi-hand scene fix (0722d18, 15 Oct), when every env
  got env 0's hand, so the "thumb + index" envs physically had all five fingers; the 5-finger numbers stand, the mix's
  description does not. A rerun on the fixed code is the clean version.
- [`seed_test.md`](seed_test.md): is thumb + index flip's missing first goal chance or the task? (a seed test)
- `plots/`: figures (`plots/README.md`).
