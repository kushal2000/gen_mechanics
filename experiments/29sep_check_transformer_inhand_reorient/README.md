# Joint transformer at 5° on all three hands — the counterpart of the MLP check

`experiments/29sep_check_mlp_inhand_reorient/` shows an MLP learning 5° in-hand reorientation.
At about epoch 1000–1800 (29 Sep): real SHARPA 7.4 goals/episode, Allegro 0.70 (on 624497's curve,
which reached 48), gen-SHARPA 0.16 and rising. This folder runs the **joint transformer on exactly
the same configs**. Each `.sub` is its MLP counterpart with only `ARCH` changed, so any
difference in outcome is the network.

| run | hand | MLP counterpart |
|---|---|---|
| `tf_5deg_sharpa` | real SHARPA (`sharpa_handonly`) | `mlp_5deg_sharpa` |
| `tf_5deg_allegro` | real Allegro (`allegro_handonly`) | `mlp_5deg_allegro` |
| `tf_5deg_gen_sharpa` | capsule SHARPA (`handonly:…/sharpa_capsule.json`) | `mlp_5deg_gen_sharpa` |

Recipe (shared with the MLP runs): fall penalty 200, γ 0.998, lr 5e-4 adaptive, 5 mini-epochs,
12288 envs, SAPG 6 × 2048, 5° keypoint success test, 15000 epochs chained. Transformer:
`common.sh` defaults, d_model 64, 4 layers, 1 head, ff_mult 2, no hand-global skip.

Known risk, not corrected here to keep the comparison clean: on Allegro at 20° the transformer
never learned (≤0.07 goals/episode over 2000+ epochs) under adaptive or constant lr, with or
without the skip and at larger width (28sep FINDINGS.md). Under the adaptive schedule its lr ran
to the 1e-2 cap. These runs test whether that failure holds at the settings where all three MLPs
learn. Read goals/episode against the MLP counterpart at matched epochs.

## Launch

```bash
for f in experiments/29sep_check_transformer_inhand_reorient/*.sub; do sbatch "$f"; done
```

Logs: `debug_outputs/train_logs/29sep_check_transformer_inhand_reorient/`. wandb project
[`gen_mechanics_check_mlp_inhand_reorient`](https://wandb.ai/kk837/gen_mechanics_check_mlp_inhand_reorient)
(shared with the MLP runs), one group per run (`tf_5deg_<hand>`).
