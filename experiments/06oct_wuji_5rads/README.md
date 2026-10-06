# 06 Oct: Wuji v2 at the XM335's speed cap (5 rad/s)

The full Wuji v2 and the two two-finger reductions from `06oct_minimal_embodiment`, retrained from scratch with
every hand joint capped at **5 rad/s** -- the ceiling `2026-10-02_physical_grammar` derives from the XM335-T323-T
(53 rpm = 5.55 rad/s at no load; `GEN_JOINT_VELOCITY_RAD_S = 5.0`).

One change from the 10 rad/s runs, and nothing else: `HAND_VELOCITY_LIMIT=5.0`, i.e.
`env.physics.hand_velocity_limit=5.0`, which sets `velocity_limit_sim` on every hand joint and overrides the
10 rad/s the uniform URDFs carry. The URDFs and specs are untouched. Effort is the spec's 0.5 N.m, already the
XM335-derived ceiling; stiffness 3.0, damping 0.0775 and armature 0.00058 stay the uniform gains (the
physical_grammar drive's stiffness 0.5 / armature 0.003 were deliberately NOT adopted).

| run | spec | job | 10 rad/s counterpart |
|---|---|---|---|
| `wuji2_full_v5` | `wuji2_left_uniform_handonly` | 11983 | `01oct_uniform_dynamics/left_wuji2_uniform_canon` (epoch 5000 of the Wuji-only run) |
| `wuji2_only_thumb_index_v5` | `wuji2_left_uniform_handonly_only_thumb_index` | 11984 | `06oct_minimal_embodiment` 11734 |
| `wuji2_only_middle_ring_v5` | `wuji2_left_uniform_handonly_only_middle_ring` | 11985 | `06oct_minimal_embodiment` 11735 |
| `wuji2_full_v5_ema0.1` | `wuji2_left_uniform_handonly` | 12491 | `wuji2_full_v5` (11983): same run with `env.action.hand_moving_average=0.1` |

5000 epochs each, wandb project `gen_mechanics_minimal_embodiment`, logs in
`debug_outputs/train_logs/06oct_wuji_5rads/`. Check that the cap took: the log's reset line reads
`hand joint velocity limits (rad/s): min 5 max 5 (physics.hand_velocity_limit=5.0)`.

    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py full only_thumb_index only_middle_ring
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --ema 0.1 full

`wuji2_full_v5_ema0.1` asks what the hand-target moving average costs and buys. run_rank.sh pins it to 1.0 (off);
the .sub's later Hydra override sets 0.1. At 60 Hz that is a 158 ms time constant (~1 Hz cutoff), the same as
the OpenAI-style 0.3 at 20 Hz. Compare against 11983 on epochs to 1 goal/episode and on target jerk.

Since 6 Oct the uniform assets themselves carry 5 rad/s (see that folder's README), so HAND_VELOCITY_LIMIT=5.0
here is now redundant but kept: it pins these runs' cap regardless of what the assets say.
