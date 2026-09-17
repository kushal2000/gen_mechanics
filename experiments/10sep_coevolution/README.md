# Co-evolution of hand designs and a shared policy

Two arms on the same 1024-hand population and the same d64/L4 joint-transformer
policy, 24,576 envs on 2 GPUs:

- **co-evolution** — every 2000 epochs, rank the designs by mean episode return,
  keep the top 512, mutate each survivor once to refill, carry the policy and the
  success-tolerance curriculum into the next generation (`coevo_gen.sub`);
- **baseline** — the same loop with the selection step removed: same population
  every link, policy and curriculum carried across 24 h links (`baseline.sub`).

The two jobs source `common.sh`, so they differ only in the selection.

## Reproduce

```bash
E=experiments/10sep_coevolution
$E/make_population.sh 0 1024 500         # -> assets/populations/gen_s0_n1024_drift500_s0/round_0500.json
P=assets/populations/gen_s0_n1024_drift500_s0/round_0500.json
$E/launch.sh coevo    $P coevolution_v2            # 40 generations, self-chaining, ~6 days
$E/launch.sh baseline $P coevolution_v2_baseline   # 10 x 24 h links, self-chaining
```

Knobs go on the end as `KEY=VALUE`: `EPOCHS_PER_GEN=2000 KEEP=512 MAX_GEN=40`
for co-evolution, `EPOCHS=15000 MAX_CONT=10` for the baseline, `SEED=100` for
both. To start co-evolution from a trained policy rather than scratch pass
`CHECKPOINT=<.pth> RESUME_TOL=<tolerance>` (`python -m coevolution.loop.final_tolerance <run_dir>`
prints the latter for any finished run).

## What each piece does

| file | role |
|---|---|
| `common.sh` | model, env count, minibatch, seed, wandb, run root; `newest_checkpoint`, `own_run_dir` |
| `coevo_gen.sub` | one generation: train → `hand_sampler.evolve` → record checkpoint + tolerance → resubmit `GEN+1` |
| `baseline.sub` | one 24 h link: train from `SOURCE_RUN`'s checkpoint + tolerance → resubmit; chains from the rolling autosave on `USR1` 15 min before TIMEOUT |
| `launch.sh` | copies the population into `assets/populations/<label>/gen_0/` and submits generation 0, or submits baseline link 1 |
| `make_population.sh` | sample + neutral drift |
| `coevolution/design_rewards.py` | per-design returns, banked from the env step; what selection ranks on |
| `hand_sampler/evolve.py` | truncation selection + one mutation per survivor; writes `population.json` and `selection.json` |
| `coevolution/loop/final_tolerance.py` | reads the curriculum a run ended at |

## Where results land

```
assets/populations/<label>/gen_<k>/
    population.json         the 1024 designs trained in generation k
    selection.json          ranking, survivors, culled, child -> parent and operator   (k >= 1)
    design_rewards_rank*.json   per-design episodes / return / goals / success this generation
    checkpoint.txt  success_tolerance.txt  run_dir.txt  job_id.txt
debug_outputs/train_logs/coevolution/
    0_scale_train_<label>_g<k>_.../rank_0/.../nn/*.pth, summaries/events*   (or _c<k>_ for baseline links)
    coevo-<job>.log  baseline-<job>.log
wandb: project gen_mechanics, group <label>
```

Analysis and figures: `debug_outputs/coevo_analysis/` (`README.md` there).

## coevolution_v1 (Sep 2026)

Ran from `experiments/decentralized_control/scaling_laws/depth_d64/{coevo_gen,continue_run,pop1k_r500_l4}.sub`,
which these files replace; record in `assets/populations/coevo_r500_v1/`, runs in
`debug_outputs/train_logs/{coevo,depth_d64}/`. Co-evolution reached the 0.01
curriculum floor at generation 31 (~61k epochs) and held it to 40; the baseline
was at 0.029 after 66k epochs. Joints per hand 11.2 → 18.1. Caveat found
afterwards: by generation 13 every hand descended from one gen-0 design (#521),
selected during the first four generations when the policy was still on its
~300-return plateau and rank correlation with eventual quality was ~0.2. Things
a V2 should change are in `debug_outputs/coevo_analysis/README.md`.
