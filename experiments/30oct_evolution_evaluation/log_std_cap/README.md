# Log-std cap (31 Oct)

Why: in the runs where many hands share one policy (the RL-only control, MAP-Elites), the learned action log-std on a
few rarely used joint slots grows without limit: 9.6 (std ~14,000) at RL-only epoch 30,800 (slots 24, 35-38), 9.5 in
MAP-Elites by iteration 40 (slots 10, 35). Greedy stays at or below 2.7, and solo runs at or below 1.1. Hypothesis: actions
are clipped to [-1, 1], so once a dimension's noise saturates it, more noise changes nothing, PPO stops pushing back, and
SAPG's entropy bonus inflates it. Above log-std ~5.5 the variance overflows fp16, so the loss goes inf/NaN. The GradScaler
then halves its scale on every such step, which caused the 31 Oct NaN crashes (`../nan_debug/`), and with the
`min_loss_scale` floor those steps are skipped (MAP-Elites iteration 45: 30-70% of steps in bad epochs).

Fix under test: `agent.params.network.max_log_std` (coevolution/networks/joint_transformer.py; None = off, the default)
clamps the log-std. Cap 2 = std 7.4, still saturating, as in SAC's implementations (log-std clamped to [-20, 2]).

- `rl_only_cap.sub`: the failure case. RL-only resumed from epoch 30,800 with the cap, 1,000 epochs, against the
  uncapped `../../29oct_evolution_from_scratch/long_rl_only_ext4.sub` (running). Expect: skipped steps ~0, no floor hits,
  KL mean ~ median, goals not worse.
- `solo_cap_g0018_s100.sub`: the no-harm check. Solo g0018, seed 100, against `../lr_schedule_test/lr_adapt_g0018_s100.sub`.
  The cap never engages there, so the curves should match up to seed noise.
