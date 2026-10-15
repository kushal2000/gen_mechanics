# 15 Oct: pen repose

Started 15 Oct 2026 on branch `2026-10-15-pen_repose`. The `pen_repose` task is in the in-hand olympics env
(`isaacsimenvs/inhand_olympics/tasks/pen_repose.py`):

- **Start:** the pen lies flat on the table at a random yaw. The first goal is the pickup: flat, 8 cm up.
- **Then a random walk of goals.** Each goal turns the pen 15-30° about a random axis, so its long axis can point
  anywhere, and moves its centre up to 5 cm within ±4 cm sideways.
- **Goal height:** clipped between an orientation-dependent floor (the pen's lowest point 2 cm above the table) and
  12 cm above resting.
- **Success:** within 1 cm, counted the step it's reached. Runs here use a 3 cm tolerance for the first goal only
  (`env.goal.first_goal_tolerance=0.03`).
- **Pushes and dropping:** pushes start after the pickup, and touching the table after it ends the episode.

## Runs

The first runs were started on the design-search branch and log under `debug_outputs/train_logs/14oct_design_search/`.
They are the three random primitive hands, with MLP, box pen, stagger on, seed 100 and 5,000 epochs, in wandb
project `gen_mechanics_design_search`:

| Hand (gaps) | Job |
|---|---|
| 30/55/55/220 | 291747 |
| 35/65/195/65 | 291748 |
| 50/65/55/190 | 291749 |

New runs from this branch log under `debug_outputs/train_logs/15oct_pen_repose/`
(`LOG_GROUP=15oct_pen_repose`, with sbatch `--output` to match).
