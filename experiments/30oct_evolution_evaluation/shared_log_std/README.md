# One noise level for all actions (31 Oct)

The user's idea. The joint transformer's action noise was one learned log-std per padded action slot (per SAPG block).
In a population a slot is a template position, a different physical joint on each design, so a per-slot noise level is
not shared meaningfully across hands, and a slot that only a few designs have is trained by those few alone. In the
RL-only and MAP-Elites runs such slots ran away to log-std 9.6 (see ../log_std_cap/README.md). With
`agent.params.network.shared_log_std=true` (coevolution/networks/joint_transformer.py) every action dimension shares one
learned log-std per SAPG block. Masked ghost and locked wrist dimensions stay masked.

Solo check before using it in population runs: `shared_g0018_s10{0,1,2}.sub`, MAP-Elites' best hand g0018_c00 alone,
3,180 epochs, identical to `../lr_schedule_test/lr_adapt_g0018_s10*.sub` (per-slot noise, adaptive lr capped at 1e-3)
plus the flag. Compare the training goals/episode, epochs to 1 goal/episode, and the final evaluation (cap 200).
For reference, the solo per-slot runs learned log-stds from about -1.4 to 1.1 (std 0.25 to 3), so one shared level
gives up that per-joint spread.
