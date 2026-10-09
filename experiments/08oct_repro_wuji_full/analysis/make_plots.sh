#!/bin/bash
# Draws plots/ from results/ood with plot_ood.py: one figure per comparison, goals per episode, a 2x2 of push force
# and friction (top row), object mass and hand tilt angle (bottom row), one folder per hypothesis -- plots/README.md says
# what each tests and what the figures show. PNG only. Rerun as OOD results land; figures are replaced.
#
#   plots/full_vs_index_middle                              headline: MLP, best checkpoints, 3 seeds
#   plots/h1_fewer_fingers_less_robust/finger_subsets       full, no ring, thumb + index + middle, index + middle
#   plots/h2_link_geometry/link_length                      full vs finger links 75% and 50% as long
#   plots/h2_link_geometry/link_width                       full vs finger links 120% as wide
#   plots/h3_transformer_vs_mlp/full_hand                   MLP vs transformer, both trained with action delay
#   plots/h4_multi_embodiment_robustness/full_hand       6 Oct 52-hand transformer vs the full-hand specialist (no delay)
#   plots/h5_dr_on_ood_axes/full_hand, index_middle     MLP trained with DR over the middle of each OOD sweep vs without
#   plots/h4_multi_embodiment_robustness/index_middle    E4 (2-3 finger hands) vs the index + middle specialist T1 (delay)
#
# plot_ood.py itself still draws the other views (--groups mid,final; goals per drop; 3-axis figures) on request.
#
#   experiments/08oct_repro_wuji_full/analysis/make_plots.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY="$HERE/../../.venv_isaacsim/bin/python $HERE/analysis/plot_ood.py"
P=$HERE/plots
cd $HERE/analysis
# A comparison with no results yet is skipped (plot_ood.py exits non-zero), not fatal.
fig() { $PY --metric goals_per_episode --all-axes "$@" > /dev/null 2>&1 || echo "skipped (no results yet): $*"; }
fig --only mlp_full,mlp_index_middle --groups best --out $P --name full_vs_index_middle
H1=$P/h1_fewer_fingers_less_robust
fig --only mlp_full,mlp_no_ring,mlp_thumb_index_middle,mlp_index_middle --groups best --out $H1 --name finger_subsets
fig --only mlp_full,mlp_len75,mlp_len50 --groups best --out $P/h2_link_geometry --name link_length
fig --only mlp_full,mlp_wid120 --groups best --out $P/h2_link_geometry --name link_width
fig --only mlp_full,tf_full_dact --groups best --out $P/h3_transformer_vs_mlp --name full_hand
H4=$P/h4_multi_embodiment_robustness
fig --only tf6oct_full,tf6oct_multi52_on_full --groups final --out $H4 --name full_hand
fig --only tf_index_middle_dact,tf_multi2to3_on_index_middle --groups final --out $H4 --name index_middle
H5=$P/h5_dr_on_ood_axes
fig --only mlp_full,mlp_full_ood_dr --groups best --out $H5 --name full_hand
fig --only mlp_index_middle,mlp_index_middle_ood_dr --groups best --out $H5 --name index_middle
find $P -name "*.png" | sed "s#$P/##" | sort
