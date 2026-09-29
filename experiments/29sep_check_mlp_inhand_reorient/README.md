# Does an MLP policy do 5° in-hand reorientation on all three hands?

The joint transformer has not learned in-hand reorientation under anything tried (28sep folder,
FINDINGS.md), while an MLP on the same env does, on Allegro. Neither SHARPA hand has an MLP run at
5°: gen-SHARPA's MLP has only run at 20° (597298 → 637911, 1.46 goals/episode at ~e1170, no fall
penalty), and the real SHARPA hand has never been run with an MLP. This folder asks whether that holds at the real target,
**5°**, on each hand we care about. Once all three work, the transformer counterpart is the next
step.

| run | hand | `ROBOT_SPEC` |
|---|---|---|
| `mlp_5deg_sharpa` | the real SHARPA hand | `sharpa_handonly` |
| `mlp_5deg_allegro` | the real Wonik Allegro hand | `allegro_handonly` |
| `mlp_5deg_gen_sharpa` | SHARPA rebuilt in our design grammar (capsules) | `handonly:assets/populations/sharpa_capsule.json` |

Everything else is identical and is the recipe that already solved Allegro at 5° (624497,
48 goals/episode): MLP, fall penalty 200, γ 0.998, lr 5e-4 adaptive, 5 mini-epochs, 12288 envs,
SAPG 6 × 2048, keypoint success test at 5°, 15000 epochs chained over up to 10 links. (The stock
learner, 634907, was at 1.3 goals/episode after ~4750 epochs at 5°.) `mlp_5deg_allegro` is
therefore a reproduction of 624497 in this project, the baseline the two SHARPA hands are read
against.

**Working** means goals/episode (`episode_final/successes`) reaches 1+ and keeps rising. Dropping the
cube is not held against a run.

## Launch

```bash
for f in experiments/29sep_check_mlp_inhand_reorient/*.sub; do sbatch "$f"; done
```

Logs and run dirs: `debug_outputs/train_logs/29sep_check_mlp_inhand_reorient/`. wandb project
[`gen_mechanics_check_mlp_inhand_reorient`](https://wandb.ai/kk837/gen_mechanics_check_mlp_inhand_reorient),
one group per run.
