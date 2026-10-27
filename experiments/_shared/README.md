# Shared experiment helpers

Code that several experiments call, so no live script reaches into `old_experiments/`. Moved here on 27 Oct 2026 from
where it grew (the old path in brackets).

| File | What it is | Used by |
|---|---|---|
| `common.sh` | Launcher for the Oct 1–8 in-hand reorientation runs: sets the model, env count, wandb, run root; `run_and_chain` (`old_experiments/28sep_inhand_reorientation/`) | every `.sub` in 01oct–08oct and the old 28–30 Sep folders |
| `scaling_laws/run.sh`, `run_rank.sh`, `study.py` | What `common.sh` runs: one training job (single- or multi-GPU via torchrun), its rank wrapper and the study bookkeeping (`old_experiments/decentralized_control/scaling_laws/`) | `common.sh`, old scaling-law `.sub`s |
| `monitor_usage.sh` | Logs CPU / RAM / GPU usage while a job runs (`experiments/`) | `scaling_laws/run.sh`, `train_single_embodiment_sharpa.sub` |
| `ten_hands.py` | The 30 Sep commercial-hand runs' `.sub` template and observation list, and which checkpoint is each hand's policy (`old_experiments/30sep_ten_hands/{make_runs,viser_zero_shot}.py`) | the 01oct–08oct `make_runs.py`, `01oct_embodiment_niches/` |
| `paper_style.py` | Matplotlib figure style for papers (a copy of `08oct_repro_wuji_full/analysis/paper_style.py`) | `12oct_inhand_olympics/analysis/`, `16oct_viser_evals/analysis/` |
| `render_hands.py` | Matplotlib-only renderer of a URDF's visual meshes; draws the Wuji pictures into `12oct_inhand_olympics/plots/hands/` (`12oct_inhand_olympics/analysis/`) | `14oct_design_search/analysis/render_primitive.py`, `16oct_viser_evals/analysis/make_design_gallery.py` |

Paths are absolute in the `.sub` files (`/share/portal/kk837/gen_mechanics/experiments/_shared/common.sh`), so this
folder must not move without rewriting them.
