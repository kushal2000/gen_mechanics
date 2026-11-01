# fp32 vs fp16 (31 Oct)

After bf16 collapsed (`../amp_bf16_test/`): what does full fp32 cost in speed, and does it change learning? With
`agent.params.config.mixed_precision=false` the GradScaler, the update's autocast and the fp16 rollouts
(`rollout_autocast` requires mixed_precision) are all off.

- `bench_fp16_vs_fp32.sub`: speed. One job, one GPU, the solo g0018 setup for 200 epochs in fp16, then 200 in fp32. Its
  log prints the mean time per epoch (and the play/update split) over epochs 51-200 of each. Same node: nodes differ by
  up to ~40%, so epoch times from different jobs are not comparable.
- `fp32_g0018_s10{0,1,2}.sub`: learning. Exactly `../lr_schedule_test/lr_adapt_g0018_s10*.sub` (the fp16 reference,
  adaptive lr capped at 1e-3), 3,180 epochs, plus mixed_precision=false.
