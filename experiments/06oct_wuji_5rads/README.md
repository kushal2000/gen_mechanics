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
| `wuji2_full_v5_ema0.1_dobs` | `wuji2_left_uniform_handonly` | 21666 | `wuji2_full_v5_ema0.1` (12491) + observation delay only (0-2 steps, redrawn every step): which delay breaks learning (all three together, 13203, learned nothing) |
| `wuji2_full_v5_ema0.1_dact` | `wuji2_left_uniform_handonly` | 21667 | `wuji2_full_v5_ema0.1` (12491) + action delay only (0-2 steps, redrawn every step): which delay breaks learning (all three together, 13203, learned nothing) |
| `wuji2_full_v5_ema0.1_dobj` | `wuji2_left_uniform_handonly` | 21668 | `wuji2_full_v5_ema0.1` (12491) + object-state delay only (0-9 steps, redrawn every step, no pose noise): which delay breaks learning (all three together, 13203, learned nothing) |
| `wuji2_full_v5_ema0.1_kp200_pen0.003` | `wuji2_left_uniform_handonly` | 21331 (14665 failed at startup: int for a float field) | `wuji2_full_v5_ema0.1` (12491) with keypoint_rew_scale 2000 -> 200 and hand_actions_penalty_scale 0.0003 -> 0.003 |
| `wuji2_only_middle_ring_v5_ema0.1` | `wuji2_left_uniform_handonly_only_middle_ring` | 14606 | `wuji2_only_middle_ring_v5` (11985) with `env.action.hand_moving_average=0.1` |
| `wuji2_only_middle_v5_ema0.1` | `wuji2_left_uniform_handonly_only_middle` | 30762 | ONE finger (middle, 4 joints; make_missing_fingers.py --only wuji2 middle), 5 rad/s + moving average 0.1: is one finger enough? vs `wuji2_only_middle_ring_v5_ema0.1` (14606) |
| `wuji2_full_v5_ema0.1_dr` | `wuji2_left_uniform_handonly` | 12525 | `wuji2_full_v5_ema0.1` (12491) plus obs delay, action delay and object-state delay + noise (`--dr`) |
| `wuji2_full_v5_ema0.1_dr_nopn` | `wuji2_left_uniform_handonly` | 13203 | `wuji2_full_v5_ema0.1_dr` (12525) with the object pose noise zeroed: delay only (`--no-pose-noise`) |
| `wuji2_only_middle_ring_v5_ema0.1_dr_nopn` | `wuji2_left_uniform_handonly_only_middle_ring` | 13385 | the same settings on middle+ring; vs `wuji2_only_middle_ring_v5` (11985) |

5000 epochs each, wandb project `gen_mechanics_minimal_embodiment`, logs in
`debug_outputs/train_logs/06oct_wuji_5rads/`. Check that the cap took: the log's reset line reads
`hand joint velocity limits (rad/s): min 5 max 5 (physics.hand_velocity_limit=5.0)`.

The runs above were generated with the pre-7-Oct defaults (no moving average, no delay), e.g.
`make_runs.py --ema 0.1 --delay obs full`. Since 7 Oct the generator DEFAULTS to moving average 0.1 + action
delay (see Results), so the equivalents are now:

    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py full                        # _ema0.1_dact
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --no-action-delay full      # = 12491
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --ema 1 --no-action-delay full   # = 11983
    .venv_isaacsim/bin/python experiments/06oct_wuji_5rads/make_runs.py --dr --no-pose-noise full   # = 13203

`wuji2_full_v5_ema0.1` asks what the hand-target moving average costs and buys. run_rank.sh pins it to 1.0 (off);
the .sub's later Hydra override sets 0.1. At 60 Hz that is a 158 ms time constant (~1 Hz cutoff), the same as
the OpenAI-style 0.3 at 20 Hz. Compare against 11983 on epochs to 1 goal/episode and on target jerk.

Since 6 Oct the uniform assets themselves carry 5 rad/s (see that folder's README), so HAND_VELOCITY_LIMIT=5.0
here is now redundant but kept: it pins these runs' cap regardless of what the assets say.

`--dr` switches back on the three knobs run_rank.sh turns off, at the task config's magnitudes: observation delay
and action delay of 0-2 policy steps each (redrawn every step), and the observed object pose delayed 0-9 steps with
1 cm Gaussian / uniform +-5 deg noise (feeds the policy's object keypoints and velocity). Joint-velocity obs noise and pushes stay off.

## Results (7 Oct)

Goals/episode (training curve, `episode_final/successes`) and drop fraction near epoch 5000:

| run | job | epochs to 1 goal/episode | goals/episode @ ~5000 | drop | |
|---|---|---|---|---|---|
| `wuji2_full_v5_ema0.1` | 12491 | 1213 | 40.6 | 0.32 | baseline |
| `..._dact` (action delay only) | 21667 | 1632 | 35.6 | 0.39 | slower start, learns |
| `..._dobs` (obs delay only) | 21666 | 2978 | 1.5 | 0.09 | 92% timeouts: holds the cube, reaches no goals |
| `..._dobj` (object-state delay only) | 21668 | never | 0.07 | 0.44 | breaks |
| `..._dr_nopn` (all three delays) | 13203 | never | 0.01 | 0.89 | breaks |
| `..._kp200_pen0.003` | 21331 | never | 0.00 | 1.00 | 10x less keypoint shaping, 10x more hand penalty: nothing learned |
| `wuji2_only_middle_ring_v5_ema0.1` | 14606 | 653 | 43.5 | 0.25 | two fingers, learns fastest |
| `wuji2_only_middle_v5_ema0.1` | 30762 | never | 0.01 | 0.99 | one finger cannot |

Action delay alone costs little; obs delay and object-state delay each break learning on their own (a looser
success threshold under those delays is the next thing to try). Hence the 7 Oct defaults: moving average 0.1 +
action delay on, obs and object-state delay off.
