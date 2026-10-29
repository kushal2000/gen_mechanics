# Dummy experiments

Synthetic probes that answer one engineering question each, without the simulator or a training run: kernel and
layout benchmarks, numerical checks, shape and memory studies. Each subfolder is dated and self-contained (a script,
its results, a README with the answer); they run on one GPU in minutes. Use the real code paths (the repo's network,
layouts and hand sets) with random inputs, and check the variant against the original numerically before timing it.

| Folder | Question | Answer |
|---|---|---|
| [`29oct_finger_token_batching/`](29oct_finger_token_batching/) | Does skipping the ghost finger tokens speed up the per-finger transformer on the 100 hands? | Packed (padding-free, FlexAttention document mask): network 1.2-1.3x (finger), 1.9-2.3x (joint tokens); bucketing helps inference only. Now `packed_tokens: true` by default |
