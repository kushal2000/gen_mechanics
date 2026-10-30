# The Claude designer's pick (30 Oct)

A baseline for the evolution runs, asked for by the user: which of the 30 seed hands would a Claude agent choose, trained
alone? A fresh agent (Claude Opus 5.5, no conversation context) got the task as configured (pen repose v2, box pen
10 x 3 x 3 cm, wrist rotation locked, uniform goals), the policy and budget, the grammar and builder source, the env
source, the seed hands' designs and URDFs, and `seed_hands.png` (`seed_sheet.py`). It was told not to open any results
(debug_outputs, logs, wandb, other experiments, git history).

**Pick: `seed_f6_nocut_j2`** (6 fingers, 22 joints, no palm cut). Runners-up in order: `seed_f3_cut_j2`,
`seed_f5_nocut_j2`, `seed_f3_nocut_j2`.

**Its method** (`analysis/`, the agent's scripts): sample fingertip workspaces with the grammar's forward kinematics and
ask, for 3,000 random pen poses under the palm (uniform orientation, centre 3-9 cm below the palm, within +-5 cm
sideways), whether two different fingertips can reach opposite faces of the 3 cm box (within ~1.2 cm).

| hand | fingers / joints | pose coverage | shallow poses (< 6 cm) | poses with >= 3 fingers able to touch |
|---|---|---|---|---|
| seed_f6_nocut_j2 | 6 / 22 | 0.88 | 0.78 | 0.64 |
| seed_f3_cut_j2 | 3 / 13 | 0.77 | 0.68 | 0.43 |
| seed_f5_cut_j2 | 5 / 22 | 0.46 | 0.27 | 0.22 |
| seed_f5_nocut_j2 | 5 / 19 | 0.43 | 0.36 | 0.26 |
| seed_f3_nocut_j2 | 3 / 12 | 0.34 | 0.27 | 0.34 |

Most other hands scored near 0: short fingers, 1-2 joints, or curl paths that never meet under the palm.

**Its reasoning.** With the palm face-down and only translation at the wrist, every change of the pen's orientation
comes from finger contacts under the palm, and goals point the long axis anywhere. Every finger lies flat in the palm
plane at zero and curls down along its own direction, so two fingers only oppose if their curl paths cross under the
palm. The pick has six fingers spread round the palm, five long (170-220 mm) with 3-5 joints, five abduction/twist
joints for out-of-plane fingertip motion, about 3 fingers able to touch a typical pose (hold with two while one
regrasps; resist pushes), and a cage under the palm that should make the early pickup easier to stumble on.
**Its stated risks:** hardest to learn (22 joints, 6 finger tokens), a 90 mm one-joint stub (f2) that hangs down and
can hit the pen or table, links ~33 mm wide against a 30 mm pen, and single-seed variance.

Trained alone for the evolution runs' 8B steps as one continuous job: `../long_claude_pick.sub` (`manifest.json`); the runner-up
`seed_f3_cut_j2` too, at the user's request: `../long_claude_runner_up.sub` (`runner_up_manifest.json`).
