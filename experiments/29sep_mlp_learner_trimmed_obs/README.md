# MLP learner + trimmed observation, transformer and MLP on three hands

With the full state list as policy input, the joint transformer learns nothing at 5° on any hand
(≤0.005 goals/episode), while the MLP on the same input reaches 46 (real SHARPA), 32 (Allegro) and
20 (gen-SHARPA). See `29sep_check_mlp_inhand_reorient/` and `29sep_check_transformer_inhand_reorient/`.
About 43 of the global token's ~99 inputs were constant in-hand (mount pose and velocity, palm
keypoints, cube size, lift latch), and several were critic-type bookkeeping (reward, successes,
progress, best-so-far distances). This folder removes all of that, and the per-joint object view.

**Policy input** (`OBS_LIST`, identical for both networks):

| where | fields | per |
|---|---|---|
| joint token | `joint_pos`, `joint_vel`, `prev_joint_pos`, `prev_joint_vel`, `prev_action_targets`, `joint_link_bbox` (12), `joint_lower`, `joint_upper` | 19 per joint |
| attention mask only | `joint_enabled`, read raw, never a token feature (the MLP sees it as an input) | — |
| global token | `keypoints_rel_ee` (object keypoints, 12), `keypoints_rel_goal` (goal residual, 12), `object_vel` (6) | 30 |
| global token | SAPG exploration coefficient (appended by the algorithm, learned embedding) | — |

`object_keypoints_rel_joint` is gone, so a joint token holds only the joint's own state and geometry. The
object and the goal reach a joint only through attention to the global token. There is no separate
critic (`central_value_config=null`), so the value head reads the same trimmed input.

Runs, one MLP/transformer pair per hand. Both use the MLP recipe (fall penalty 200, γ 0.998, lr 5e-4
adaptive, 5 mini-epochs), so **a pair differs only in the network**:

| hand | MLP | transformer |
|---|---|---|
| real SHARPA | `mlp_5deg_sharpa_trimobs` | `tf_5deg_sharpa_trimobs` |
| Allegro | `mlp_5deg_allegro_trimobs` | `tf_5deg_allegro_trimobs` |
| gen-SHARPA | `mlp_5deg_gen_sharpa_trimobs` | `tf_5deg_gen_sharpa_trimobs` |

Read against the full-observation runs at matched epochs. The MLP rows check the trim costs the MLP
nothing; the transformer rows are the question. Full-observation baselines: MLP `mlp_5deg_<hand>`,
transformer `tf_5deg_<hand>_mlphp`.

Plumbing: `OBS_LIST` in `run_rank.sh` (unset keeps the full state list), and
`object_keypoints_rel_joint` is now an optional token field (`layout.OPTIONAL_HAND_TOKEN_FIELDS`).

## Code this runs on (29 Sep)

Launched after the joint-transformer fix (`fbf8311`): joints now attend to each other. Before it,
every joint was masked out of attention (see `29sep_check_transformer_inhand_reorient/README.md`).
The network normalises its own input, with statistics pooled over joints, and rl_games'
normaliser is off for it. Allegro also carries the palm × thumb_link_2 self-collision filter
(`a211a54`). The MLP keeps rl_games' normaliser.

Replaces the 29sep_check_transformer runs 651207–651212 (full observation, fixed network),
which were cancelled at epochs 800–2000 (best 0.08 goals/episode, real SHARPA, MLP learner).

## Launch

```bash
for f in experiments/29sep_mlp_learner_trimmed_obs/*.sub; do sbatch "$f"; done
```

Logs: `debug_outputs/train_logs/29sep_mlp_learner_trimmed_obs/`. wandb project
[`gen_mechanics_check_mlp_inhand_reorient`](https://wandb.ai/kk837/gen_mechanics_check_mlp_inhand_reorient).
