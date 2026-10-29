# experiments

One folder per experiment, named `<date>_<what>`: its SLURM scripts, analysis, plots and a README with the runs and
results. Training logs and checkpoints go to `debug_outputs/train_logs/<experiment>/` (not tracked).

## Current

| Folder | What |
|---|---|
| [`12oct_inhand_olympics/`](12oct_inhand_olympics/) | **The olympics trainer** (`olympics_sapg_mlp.sub`, used by every olympics run below), per-task evals, the embodiment-curriculum result |
| [`14oct_design_search/`](14oct_design_search/) | Primitive hands from finger bearings (`primitive_hands.py`), the joint-slider sim viewer, design-search runs on lift and flip |
| [`15oct_pen_repose/`](15oct_pen_repose/) | 6D pen repose (v2: palm rotation locked, uniform goals) across hand designs; spin / flip fine-tunes |
| [`15oct_pen_twist/`](15oct_pen_twist/) | Twisting a cap off the pen: the twist pen, the tightness curriculum, square vs round cap |
| [`16oct_viser_evals/`](16oct_viser_evals/) | One viewer for every fixed-palm policy (switch its task and goals live); the design gallery |
| [`27oct_grammar_random_hands/`](27oct_grammar_random_hands/) | 100 random grammar hands (`make_random.py`), the two design agents' picks, their repose runs; viewers for hands, joint tokens and link boxes |
| [`27oct_transformer_debug/`](27oct_transformer_debug/) | Why the joint transformer stalled on repose (the 27 Oct "fast config"), one-change arms, the conclusions |
| [`29oct_multi_embodiment_transformer/`](29oct_multi_embodiment_transformer/) | One transformer policy across all 100 grammar hands (the 14 Oct setup, masks, home actions) |
| [`29oct_mlp_vs_transformer/`](29oct_mlp_vs_transformer/) | MLP vs joint transformer on three hands, same env (home actions, action delay off) |
| [`29oct_transformer_speed/`](29oct_transformer_speed/) | Network profiling: why the transformer is slower than the MLP per sample, and the route to a fast one with many parameters |
| [`29oct_per_finger_transformer/`](29oct_per_finger_transformer/) | The per-finger transformer (one token per finger; the transformer default since 29 Oct) against the per-joint one and the MLP |
| [`grammar_viewer/`](grammar_viewer/) | Browse Martin's hand grammar (CPU only) |
| [`_shared/`](_shared/) | Helpers several experiments call (launchers, run templates, figure style, renderers) |

## Reference results (1–8 Oct, in-hand cube reorientation on the palm-up hand)

Kept for their results and figures; they run through `_shared/common.sh` and the `inhand_reorient` task.

| Folder | What |
|---|---|
| `01oct_uniform_dynamics/` | The commercial hands rerun with uniform dynamics |
| `01oct_unified_rl/` | One policy across 8 commercial hands (`multi:uniform`) |
| `01oct_embodiment_niches/` | Do hands rank differently on other axes (OOD physics, objects)? `eval_niches.py` is the shared OOD eval |
| `02oct_ood_evals/` | OOD push / mass / friction sweeps, unified vs single-hand policies |
| `05oct_embodiment_generalization/` | Missing-finger and action-noise sweeps |
| `06oct_minimal_embodiment/` | The smallest Wuji reduction that still does the task |
| `06oct_wuji_5rads/` | Wuji v2 at the 5 rad/s servo cap |
| `06oct_wuji2_multi_embodiment/` | The first 52-hand Wuji policy |
| `07oct_commercial_vs_generated/` | Commercial hands vs their grammar replicas |
| `08oct_repro_wuji_full/` | Reproduction of the Wuji runs; OOD robustness and multi-embodiment headline plots |
| `08oct_multi_embodiment_evolution/` | One policy per Wuji finger-drop family |

`train_single_embodiment_sharpa.sub` is the original SHARPA + iiwa14 pose-reaching run (the `pose_reaching_6d` task).

## Archive

`old_experiments/` (Aug–Sep): pose reaching, co-evolution V1/V2, the first in-hand reorientation runs, the scaling-law
study. A record, not working code: some scripts launch modules that were deleted (all reachable at git tag
`pre-cleanup-2026-10-27`).

## Running on the cluster

**Ask for the minimum you need.** Queue time is dominated by the size of the request. Peak RSS: ~34.5 GB for a
24,576-env population run, ~17 GB for a single-hand run; `--cpus-per-task=4-8 --mem=50000-64000` leaves headroom. Don't
go below 4 CPUs (Kit boot slows sharply). Check a finished run with `sacct -j <jobid> --format=MaxRSS`.

**Pick the right GPU.** `--partition=portal --exclude=portal-compute-01`: nodes 02–04 have RTX 6000 **Ada** cards;
01 has the older, roughly half-speed RTX **A6000** (the names mislead — NVIDIA dropped the "A" for Ada). Benchmark on a
pinned node: nodes differ by up to ~40%.

**Bound runs by epochs, not wall clock,** when comparing hands: a hand that simulates faster gets more gradient steps
in the same time. The olympics trainer takes `EPOCHS` (5,000 by default).

**Name runs for what they are.** The olympics trainer builds the name from the task, network, envs, seed, hand and
`RUN_TAG`; `LOG_GROUP` picks the folder under `debug_outputs/train_logs/` (pass `sbatch --output` to match).
