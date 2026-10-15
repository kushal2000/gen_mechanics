# 15 Oct: pen twist (twist the cap off)

The twist pen is the pen split into a barrel and a cap (3 cm), joined by a joint about the long axis. The cap resists
turning like a friction fit: it holds below `pen.twist_friction_torque` (0.01 N·m) and turns above it
(`isaacsimenvs/inhand_olympics/twist_pen.py`; checked: it slips at 0.0102 N·m).

**Two objects, two 6D goals.**
- The policy sees `pen_pose`, `cap_pose`, `pen_goal_pose` and `cap_goal_pose`: position plus quaternion. It sees no
  keypoints and no twist angle.
- Keypoints are for the reward and success only. Each object has rings of 4 cm that fix its full pose. The goal
  distance is the worse of pen-vs-goal and cap-vs-goal, and success means both are within 1 cm (about ±14° of twist).

**Levels.**
1. **Level 1:** the pen stands upright on the table, cap up, with the barrel fixed (`env.pen.twist_fixed_barrel=True`).
   The pen's goal is where it stands. The cap's goal is a quarter turn further each time, in the unscrewing direction
   (counter-clockwise seen from above). Goals per episode = quarter turns. The task also tracks the cap's total turn,
   which the policy doesn't see, and counts a goal only on the right turn: screwing back 270° looks like unscrewing 90°.
2. **Level 2:** the barrel is free (planned).

**Tools.**
- `viser_twist_play.py`: drive a hand by sliders against the pen (`--level 1`, the default).
- The wandb pose viewer draws the barrel, the cap (with a stripe) and both goals.

## Tightness curriculum

`env.pen_twist.curriculum=True`: the cap's friction torque starts at 0.01 N·m. After each window of finished
episodes (as many as there are envs), if they averaged at least 1 full turn, the torque is multiplied by 1.5, up to
0.2 N·m. The next window must then get back to 1 turn before it rises again. On wandb as `twist/friction_torque`.

## Runs

wandb project **`gen_mechanics_pen_twist`**; logs in `debug_outputs/train_logs/15oct_pen_twist/`.

| Level | Hand | Recipe | Job |
|---|---|---|---|
| 1 | Wuji full | MLP, SAPG defaults, seed 100, 5,000 epochs, `RUN_TAG=lvl1` | 293321 |
| 1 | Wuji full | the same, plus the tightness curriculum (`env.pen_twist.curriculum=True`), `RUN_TAG=lvl1_cur` | 295202 |
| 1 | Wuji full | the curriculum from 0.1 to 1.0 N·m (`env.pen.twist_friction_torque=0.1 env.pen_twist.curriculum_max_torque=1.0`), `RUN_TAG=lvl1_cur01to1` | 296479 |

## Early result

`viser_twist_policy.py --check 293321` on the epoch-400 checkpoint, 64 envs for 30 s:

| Cap friction | Turns per episode (of 12.5) | Episodes reaching ≥ 1 turn |
|---|---|---|
| 0.01 N·m | 11.65 | 100% |
| 0.05 N·m | 12.33 | 99% |

Level 1 at 0.01 N·m is solved early, which is why the curriculum run 296479 starts at 0.1 N·m and goes up to 1 N·m.

## Background

`literature_bimanual_sim2real.md`: a subagent's report on bimanual sim-to-real work, with every paper checked online.
