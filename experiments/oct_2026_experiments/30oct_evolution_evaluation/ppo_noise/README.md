# Noise design under plain PPO (31 Oct)

The user leans towards plain PPO for simplicity (`../ppo_vs_sapg/`: same final level as SAPG, but less reliable early).
This checks that the two new noise designs work under PPO, where there is one noise table instead of one per SAPG block:

- `ppo_shared_g0018_s10{0,1,2}.sub`: one learned log-std for all actions (`shared_log_std=true`).
- `ppo_predstd_g0018_s10{0,1,2}.sub`: each joint's log-std predicted from its token plus one learned offset
  (`predict_log_std=true`).

Each is exactly `../ppo_vs_sapg/ppo_g0018_s10*.sub` (MAP-Elites' best hand g0018_c00 alone, 3,180 epochs, adaptive lr
capped at 1e-3 from 1e-4) plus the flag, so it compares with that PPO run (per-slot noise) at a matched lr. It does
not use the new fixed-3e-4 default. The SAPG counterparts are `../shared_log_std/` and `../predicted_log_std/`.
