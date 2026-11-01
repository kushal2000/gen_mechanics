# Predicted per-joint noise with its safeguards (31 Oct)

The fair test of per-state noise (the user's preferred design). The earlier predicted-noise runs
(`../predicted_log_std/`, `../ppo_noise/ppo_predstd_*`) had no clamp and no penalty: under SAPG two of three seeds
started slowly, with the leader's noise inflated (arm log-std 1.1-1.85, spikes to 7-9; probe in
`predicted_log_std/probe_log_std.py`), and under PPO the head's spikes overflowed fp16 a few times.

Here, per algorithm, 3 seeds identical to those runs plus:
- the log-std clamp [-5, 2] (`max_log_std`, `min_log_std`, the yaml defaults since 31 Oct), and
- the bounds loss on the RAW predicted log-std past that band (squared hinge, the same bounds_loss_coef 0.0001 as
  rl_games' bounds loss on the mean), which gives the head a gradient back inside where the clamp gives none.

- `sapg_predb_g0018_s10{0,1,2}.sub` (SAPG; compare with `../predicted_log_std/`, `../shared_log_std/`, per-slot
  `../lr_schedule_test/lr_adapt_*`)
- `ppo_predb_g0018_s10{0,1,2}.sub` (PPO; compare with `../ppo_noise/`, per-slot `../ppo_vs_sapg/ppo_g0018_*`)

All keep the earlier runs' adaptive lr (capped at 1e-3 from 1e-4) so that only the noise safeguards differ. The
SAPG followers' entropy bonus still reaches the shared head (not decoupled here).
