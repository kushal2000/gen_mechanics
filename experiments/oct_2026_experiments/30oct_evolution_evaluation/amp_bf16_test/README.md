# bf16 vs fp16 autocast (31 Oct)

Why: the NaN crashes of 31 Oct (`../nan_debug/`) came from the fp16 GradScaler's loss scale collapsing (16384 -> ~1e-37)
and never regrowing. The vendored rl_games now floors the scale (`min_loss_scale`, default 1.0) and can run in bfloat16
(`amp_dtype=bfloat16`, fp32's exponent range, no scaler). This tests whether bf16 learns as well and as fast as fp16.

- `bf16_g0018_s10{0,1,2}.sub`: MAP-Elites' best hand `g0018_c00` alone, 3,180 epochs, identical to
  `../lr_schedule_test/lr_adapt_g0018_s10*.sub` (adaptive lr capped at 1e-3) plus `++agent.params.config.amp_dtype=bfloat16`.
- The fp16 condition is `lr_adapt_g0018_*` (those started before the floor existed; their loss scale is not logged).

Compare: training goals/episode, epochs to 1 goal/episode, wall-clock per epoch (smoke: bf16 0.670 s vs fp16 0.631 s), and
the final evaluation (`../eval_hands_cap.py`, cap 200).
