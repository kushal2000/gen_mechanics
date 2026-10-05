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

## keypoints_rel_ee was broken, and the relaunch (29 Sep)

The first launch (658186–8 MLP, 658929–31 transformer) ran with `keypoints_rel_ee` broken: it
subtracted the palm's WORLD position from env-local keypoints, so every env's value carried its
grid offset and was clipped at ±10 (median |value| 8.1 m). In this trimmed list it was the ONLY
source of the object's pose, so those policies were close to blind to the cube. All six stayed flat
(MLP ≤ 0.05 goals/episode at epochs 2,000–2,800, where the full-observation MLP was at 4–46).
Fixed in the observation builder. Relaunched under study names ending `_kpfix`.

## No fall penalty (29 Sep)

`mlp_5deg_<hand>_trimobs_nopen.sub`: the three MLP runs again with `env.reward.fall_penalty` at its
default 0, everything else identical. Is the penalty still needed once the rest of the recipe
(γ 0.998, lr 5e-4, 5 mini-epochs) is in place? Earlier no-penalty MLP runs at 20° (637911 gen-SHARPA,
637916 Allegro) were learning but slower; they were cancelled before a clear answer.

Early result (single seed): without the penalty the MLP learned FASTER -- at matched epochs 3x (real
SHARPA), 11x (Allegro) and 5x (gen-SHARPA) the penalty runs' goals/episode, with long episodes (no
drop collapse). `tf_5deg_<hand>_trimobs_nopen.sub` are the transformer counterparts.

## Capped learning rate (29 Sep)

`tf_5deg_sharpa_trimobs_nopen_lrcap.sub`: the no-penalty real-SHARPA transformer with the adaptive
schedule capped at 1e-3 (`++agent.params.config.max_lr=0.001`, applied by `SchedulerBoundsObserver`;
rl_games hard-codes 1e-2). Transformer runs pin lr at the 1e-2 cap while the MLP settles at
1.5e-4..5e-4 -- plausibly LayerNorm scale-invariance plus a shared, narrow action head making each
step move the actions little, with the shared value head taking 1e-2 steps unchecked.

`tf_5deg_sharpa_trimobs_nopen_lrconst.sub`: the same run with a CONSTANT lr of 5e-4 (`LR_SCHEDULE=constant`).

`tf_5deg_gen_sharpa_trimobs_nopen_lrconst1e4.sub`: the no-penalty gen-SHARPA transformer with a constant lr of 1e-4.
Its adaptive twin's lr pinned at 1e-2 and never came down, unlike Allegro / real SHARPA.

MLP runs cancelled once they had answered their question: the three with the penalty and Allegro
without it (saturated at ~44 goals/episode). Kept: real-SHARPA and gen-SHARPA without the penalty.

## Launch

```bash
for f in experiments/old_experiments/29sep_mlp_learner_trimmed_obs/*.sub; do sbatch "$f"; done
```

Logs: `debug_outputs/train_logs/29sep_mlp_learner_trimmed_obs/`. wandb project
[`gen_mechanics_mlp_learner_trimmed_obs`](https://wandb.ai/kk837/gen_mechanics_mlp_learner_trimmed_obs).
