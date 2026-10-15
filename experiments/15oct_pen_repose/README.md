# 15 Oct: pen repose

Started 15 Oct 2026 on branch `2026-10-15-pen_repose`. The `pen_repose` task is in the in-hand olympics env
(`isaacsimenvs/inhand_olympics/tasks/pen_repose.py`):

- **Start:** the pen lies flat on the table at a random yaw. The first goal is a random pose (uniform orientation, random
  centre in the bounds below; `env.pen_repose.random_first_goal`, on by default since 15 Oct).
- **Then a random walk of goals.** Each goal turns the pen 15-30° about a random axis, so its long axis can point
  anywhere, and moves its centre up to 5 cm within ±4 cm sideways.
- **Goal height:** clipped between an orientation-dependent floor (the pen's lowest point 2 cm above the table) and
  12 cm above resting.
- **Success:** within 1 cm, counted the step it's reached. Runs here use a 3 cm tolerance for the first goal only
  (`env.goal.first_goal_tolerance=0.03`).
- **Pushes and dropping:** pushes start after the pickup, and touching the table after it ends the episode.

## Runs

All repose runs go to wandb project **`gen_mechanics_pen_repose`** and log under
`debug_outputs/train_logs/15oct_pen_repose/` (`LOG_GROUP=15oct_pen_repose`, with sbatch `--output` to match).

### Three random primitive hands

MLP, box pen, random first goal, first goal 3 cm then 1 cm, stagger on, seed 100, 5,000 epochs:

| Hand (gaps) | Job |
|---|---|
| 30/55/55/220 | 292019 |
| 35/65/195/65 | 292020 |
| 50/65/55/190 | 292021 |

### Designed hand: the even cross, 0/90/180/270

A subagent's static analysis picked it for repose. It found that every layout holds the pen at any in-hand axis, and
that the wrist carries a fixed grip through a median of 16 goals, so about 90% of the turning is the wrist's. The
cross wins on the rest: it can pinch the pen at every pickup yaw (the flip pick manages 82%), and it can pivot the pen
about the most pinch-line directions (110° of 180°, against 55° for the flip pick). This needs a pair at least 127°
apart plus a third finger to push. The runner-up is 0/95/180/275, which scores the same. The same recipe as above:

| Hand (gaps) | Job |
|---|---|
| 90/90/90/90 | 292201 |
| 85/95/85/95 (runner-up 0/95/180/275; built under its canonical rotation 0/85/180/265) | 292295 |

Earlier submissions with a flat pickup as the first goal were cancelled within minutes (291747-9, then 291830-2).
