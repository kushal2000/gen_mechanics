#!/bin/bash
# For every run below, once it has finished (nn/last_*_ep_5000_*.pth exists): take rl_games' best-training-reward
# checkpoint (nn/<run>.pth, overwritten whenever the training reward reached a new high -- final once the run ends),
# read its epoch, copy it to nn/snap_best_ep_<E>_.pth and submit run_ood.sub with LABEL=<emb>[_s<seed>]_best_ep<E>.
# Loops until every run is done; prints one line per submission.
REPO=/share/portal/kk837/gen_mechanics; cd $REPO
D=debug_outputs/train_logs/08oct_repro_wuji_full
RUNS="118744:mlp_full 118745:mlp_index_middle 119957:mlp_wid120 120527:mlp_len75 119956:mlp_len50
123833:mlp_full_s101 123834:mlp_full_s102 123836:mlp_index_middle_s101 123837:mlp_index_middle_s102
123838:mlp_len50_s101 123839:mlp_len75_s101 123840:mlp_len75_s102 123841:mlp_wid120_s101 123842:mlp_wid120_s102
119038:tf_full 119276:tf_full_dact"
declare -A done; total=$(echo $RUNS | wc -w)
while [ ${#done[@]} -lt $total ]; do
  for r in $RUNS; do
    j=${r%%:*}; lab=${r#*:}; [ -n "${done[$j]}" ] && continue
    nn=$(ls -d $D/0_scale_train_*_${j}_0_*/rank_0/*/nn 2>/dev/null | head -1)
    if [ -z "$nn" ] || ! ls $nn/last_*_ep_5000_*.pth >/dev/null 2>&1; then
      if ! squeue -h -j $j 2>/dev/null | grep -q . && [ -n "$nn" ] && ! ls $nn/last_*_ep_5000_*.pth >/dev/null 2>&1; then
        echo "$j $lab: ended without finishing"; done[$j]=gone; fi
      continue; fi
    best=$(ls $nn/0_scale_train_*.pth 2>/dev/null | head -1)
    [ -z "$best" ] && { echo "$j $lab: no best checkpoint"; done[$j]=none; continue; }
    ep=$(.venv_isaacsim/bin/python -c "import torch,sys; ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); ck=ck[0] if 0 in ck else ck; print(ck.get('epoch'))" $best 2>/dev/null)
    snap=$nn/snap_best_ep_${ep}_.pth; cp $best $snap
    id=$(sbatch --job-name=ood_${lab}_best --export=ALL,CKPT=$snap,LABEL=${lab}_best_ep${ep} experiments/08oct_repro_wuji_full/run_ood.sub | grep -oE '[0-9]+$')
    echo "$j $lab: best checkpoint epoch $ep -> OOD job $id"; done[$j]=1
  done
  [ ${#done[@]} -lt $total ] && sleep 120
done
echo "best-checkpoint OOD watcher finished"
