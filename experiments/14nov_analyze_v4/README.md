# v4 results analysis (14 Nov 2026)

Analysis of the v4 co-evolution campaign (`experiments/08nov_v4`, runs under `debug_outputs/coevolve/v4/`):
House of Dextra (`hod`, K 0), MAP-Elites (`k5_none`, K 5, 16 per cell) and greedy (`k5_global`, K 5, one pool),
49 generations of 1B steps each, all branching from hod's trained gen 0. Plots go to
`plots/` in this folder.

| Script | Output |
|---|---|
| `plot_fitness.py` | `plots/peak_fitness.png`, `plots/mean_fitness.png` (+ `plots/fitness.csv`): the best and the mean design goals / episode in each generation's training sample |
