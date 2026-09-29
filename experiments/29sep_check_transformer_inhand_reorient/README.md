# Joint transformer at 5° on all three hands — the counterpart of the MLP check

`experiments/29sep_check_mlp_inhand_reorient/` shows an MLP learning 5° in-hand reorientation.
At about epoch 1000–1800 (29 Sep): real SHARPA 7.4 goals/episode, Allegro 0.70 (on 624497's curve,
which reached 48), gen-SHARPA 0.16 and rising. This folder runs the **joint transformer** on the same hands, task, 5° tolerance and
fall penalty 200, with the **original SAPG learner** (`PoseReachJointTransformerSAPG.yaml`: γ 0.99,
lr 1e-4 adaptive, 2 mini-epochs, horizon 16, entropy 0). The MLP runs use γ 0.998, lr 5e-4 and
5 mini-epochs, so **each pair differs in network and learner**. Read the transformer runs against
the MLP counterparts with that in mind. The MLP stock-learner reference at 5° is 634907 (Allegro,
1.3 goals/episode at ~e4750).

| run | hand | MLP counterpart |
|---|---|---|
| `tf_5deg_sharpa` | real SHARPA (`sharpa_handonly`) | `mlp_5deg_sharpa` |
| `tf_5deg_allegro` | real Allegro (`allegro_handonly`) | `mlp_5deg_allegro` |
| `tf_5deg_gen_sharpa` | capsule SHARPA (`handonly:…/sharpa_capsule.json`) | `mlp_5deg_gen_sharpa` |

**Exact counterparts**, `tf_5deg_<hand>_mlphp.sub`: the same three hands with the MLP runs'
learner (γ 0.998, lr 5e-4 adaptive, 5 mini-epochs). Their exports differ from `mlp_5deg_<hand>.sub`
only in `ARCH`, the study name and the log folder, so these pairs **differ only in the network**.

| run | learner | differs from MLP counterpart in |
|---|---|---|
| `tf_5deg_<hand>` | original SAPG (γ 0.99, 1e-4, 2) | network and learner |
| `tf_5deg_<hand>_mlphp` | MLP's (γ 0.998, 5e-4, 5) | network only |

Transformer: `common.sh` defaults, d_model 64, 4 layers, 1 head, ff_mult 2. **No hand-global skip,
ever.** Each joint's action comes from its own token only.

History: on Allegro at 20° the transformer never learned (≤0.07 goals/episode over 2000+ epochs)
under adaptive or constant lr and at larger width (28sep FINDINGS.md). Two earlier launches of this
folder with the MLP's learner (644447–9, 644472–4) were cancelled within minutes.

## Launch

```bash
for f in experiments/29sep_check_transformer_inhand_reorient/*.sub; do sbatch "$f"; done
```

Logs: `debug_outputs/train_logs/29sep_check_transformer_inhand_reorient/`. wandb project
[`gen_mechanics_check_mlp_inhand_reorient`](https://wandb.ai/kk837/gen_mechanics_check_mlp_inhand_reorient)
(shared with the MLP runs), one group per run (`tf_5deg_<hand>`).
