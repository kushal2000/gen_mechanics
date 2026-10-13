# Seed test: is thumb + index flip's missing first goal chance or the task? (13 Oct, written before results)

Seed 100, thumb + index (wuji2_left_uniform_handonly_only_thumb_index), box pen, carried-task rules (dropped
after the first goal), gamma 0.99, MLP SAPG: pen_lift (252545) found its first goal at epoch 885; pen_flip_air
(252547) has none by epoch 2537. Before the first goal the two tasks are the same MDP (checked step by step:
every reward term, termination and the first goal's reward identical; the only differences are the task one-hot
and the random stream), so either chance or something unseen in the task explains it.

Runs (project gen_mechanics_inhand_olympics_carried, identical settings but the seed):

| seed | pen_lift | pen_flip_air |
|------|----------|--------------|
| 100  | 252545   | 252547       |
| 101  | 253117   | 253120       |
| 102  | 253118   | 253121       |
| 103  | 253119   | 253122       |

Measure: first epoch with goals/episode > 0.05 (or none by 5000).

Decision rule, fixed before the results:
- **Chance** if at least one flip seed finds its first goal within lift's range (about epochs 900-1500), or at
  least one lift seed has none past epoch 2500: the two tasks' times overlap.
- **The task** if every flip seed has no first goal by 2500 and every lift seed has one by 1500.
- Otherwise inconclusive: add seeds.

Expected fingerprint of a stuck run, whatever the task: keypoint reward per step stays ~0.01, the finger-motion
penalty shrinks (the hand moves less), and the split from a learning run shows within the first few hundred epochs.
