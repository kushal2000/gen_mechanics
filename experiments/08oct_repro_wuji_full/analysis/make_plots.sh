#!/bin/bash
# Draws plots/ from results/ood with plot_ood.py: one figure per comparison, goals per episode, a 2x2 of push force
# and friction (top row), object mass and palm pitch (bottom row), one folder per hypothesis -- plots/README.md says
# what each tests and what the figures show. PNG only. Rerun as OOD results land; figures are replaced.
#
#   plots/full_vs_index_middle                              headline: MLP, best checkpoints, 3 seeds
#   plots/h1_fewer_fingers_less_robust/finger_subsets       full, no ring, thumb + index + middle, index + middle
#   plots/h2_link_geometry/link_scaling                     full vs links 50% / 75% length, 120% width
#   plots/h3_transformer_vs_mlp/full_hand                   MLP vs transformer, both trained with action delay
#   plots/h4_multi_embodiment_robustness/52_hand_vs_specialist   6 Oct 52-hand transformer vs full-hand specialist
#
# plot_ood.py itself still draws the other views (--groups mid,final; goals per drop; 3-axis figures) on request.
#
#   experiments/08oct_repro_wuji_full/analysis/make_plots.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY="$HERE/../../.venv_isaacsim/bin/python $HERE/analysis/plot_ood.py"
P=$HERE/plots
cd $HERE/analysis
fig() { $PY --metric goals_per_episode --all-axes "$@" > /dev/null; }
fig --only mlp_full,mlp_index_middle --groups best --out $P --name full_vs_index_middle
H1=$P/h1_fewer_fingers_less_robust
fig --only mlp_full,mlp_no_ring,mlp_thumb_index_middle,mlp_index_middle --groups best --out $H1 --name finger_subsets
fig --only mlp_full,mlp_len75,mlp_len50,mlp_wid120 --groups best --out $P/h2_link_geometry --name link_scaling
fig --only mlp_full,tf_full_dact --groups best --out $P/h3_transformer_vs_mlp --name full_hand
fig --only tf6oct_full,tf6oct_multi52_on_full,tf6oct_multi52_on_index_middle --groups final \
    --out $P/h4_multi_embodiment_robustness --name 52_hand_vs_specialist
find $P -name "*.png" | sed "s#$P/##" | sort
