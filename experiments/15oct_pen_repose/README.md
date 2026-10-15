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

## Fine-tuning the v2 policies on spin and flip (15 Oct)

Played zero-shot on spin and flip goals, the repose policies snap to each 45° goal by the shortest route and stop.
They don't turn continuously. So each v2 hand's best-reward checkpoint (weights only) is fine-tuned on
`pen_spin_air` and on `pen_flip_air`, with each task's own one-hot. The reward is unchanged; only success and
termination change (`isaacsimenvs/inhand_olympics/tasks/rotation_course.py`):
- **Forward-only success** (`forward_only=True`): a goal counts only once the pen has also turned that far, in the
  episode's direction, about the task's axis since the pickup.
- **Off course** (a new termination, no penalty): the episode ends once the pen's long axis is more than 30° out of
  the task's plane (`max_off_axis_deg=30`), or once it has turned back 30° from its furthest (`max_backward_deg=30`).

Checked with scripted motion: turning 450° forward scores 11 goals (the pickup plus 10 × 45°) on both tasks; turning
back or out of the plane ends the episode at 30-33°.

wandb **`gen_mechanics_pen_spin_ft`** and **`gen_mechanics_pen_flip_ft`**; logs in
`debug_outputs/train_logs/15oct_spin_flip_ft/`; 5,000 epochs; first goal (the pickup) at 3 cm.

| Hand (gaps) | From | Spin job | Flip job |
|---|---|---|---|
| 30/55/55/220 | 292505 | 301039 | 301040 |
| 35/65/195/65 | 292506 | 301041 | 301042 |
| 50/65/55/190 | 292507 | 301043 | 301044 |
| 90/90/90/90 (the cross) | 292508 | 301045 | 301046 |
| 85/95/85/95 | 292509 | 301047 | 301048 |

## Runs

### v2: wrist orientation fixed, every goal uniform (15 Oct, current)

The first version was too easy. The wrist did about 90% of the turning, and consecutive goals were only 15-30° apart.
Two changes:

- **The wrist can't rotate.** `env.action.wrist_max_speed_ang=0` freezes the roll, pitch and yaw commands at the
  default (palm down). The wrist still translates. All reorientation must happen in the hand.
- **Every goal is a fresh random pose from the whole goal space.** `env.pen_repose.goal_sampling=uniform`: a uniform
  orientation and a centre uniform in that orientation's bounds, as for the first goal. Consecutive goals now differ by
  127° on average (checked with the task's own sampler).

Everything else is the same as before: MLP, box pen, first goal 3 cm then 1 cm, stagger on, seed 100, 5,000 epochs.
wandb project **`gen_mechanics_pen_repose_v2`**, `RUN_TAG=uni_fw`:

| Hand (gaps) | Job |
|---|---|
| 30/55/55/220 | 292505 |
| 35/65/195/65 | 292506 |
| 50/65/55/190 | 292507 |
| 90/90/90/90 (designed: the cross) | 292508 |
| 85/95/85/95 (runner-up) | 292509 |

### v1: random walk, free wrist (cancelled 15 Oct at about epoch 350-400, for v2)

All repose runs go to wandb project **`gen_mechanics_pen_repose`** and log under
`debug_outputs/train_logs/15oct_pen_repose/` (`LOG_GROUP=15oct_pen_repose`, with sbatch `--output` to match).

#### Three random primitive hands

MLP, box pen, random first goal, first goal 3 cm then 1 cm, stagger on, seed 100, 5,000 epochs:

| Hand (gaps) | Job |
|---|---|
| 30/55/55/220 | 292019 |
| 35/65/195/65 | 292020 |
| 50/65/55/190 | 292021 |

#### Designed hand: the even cross, 0/90/180/270

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
