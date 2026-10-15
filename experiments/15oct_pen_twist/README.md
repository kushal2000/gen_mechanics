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

## Runs

wandb project **`gen_mechanics_pen_twist`**; logs in `debug_outputs/train_logs/15oct_pen_twist/`.

| Level | Hand | Recipe | Job |
|---|---|---|---|
| 1 | Wuji full | MLP, SAPG defaults, seed 100, 5,000 epochs, `RUN_TAG=lvl1` | 293321 |

## Background

`literature_bimanual_sim2real.md`: a subagent's report on bimanual sim-to-real work, with every paper checked online.
