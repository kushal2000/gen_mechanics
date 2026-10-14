# In-hand olympics plots

## Headline: `embodiment_curriculum.png`

Pen lift, 5-finger Wuji hand: learned via a thumb + index policy (blue), against the same network trained on the full
hand from scratch (orange, 3 seeds). The x axis is total training epochs, so the curriculum's 900 thumb + index epochs
(shaded) count against it. Goals per episode are training-time numbers (leader SAPG block, 25-epoch moving average).
The curriculum's 5-finger hand learns within about 100 fine-tune epochs; 2 of the 3 from-scratch seeds never reach
1 goal per episode, and the third gets there at epoch 2,695. Write-up and caveats: `../embodiment_curriculum.md`.

Redraw with `analysis/plot_embodiment_curriculum.py` (style: `08oct_repro_wuji_full/analysis/paper_style.py`).
