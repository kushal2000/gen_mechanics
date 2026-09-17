# Co-evolution V2 (17 Sep 2026)

Same two arms as `../10sep_coevolution` -- co-evolution (`coevo_gen.sub`) and
the fixed-population baseline (`baseline.sub`), sharing `common.sh` -- with the
fixes that came out of reading V1's record. Launch exactly as before:

```bash
E=experiments/17sep_coevolution
P=assets/populations/coevo_r500_v1/gen_0/population.json     # V1's starting population, unchanged
$E/launch.sh coevo    $P coevolution_v2
$E/launch.sh baseline $P coevolution_v2_baseline
```

## What changed from 10sep

| | 10sep (coevolution_v1) | 17sep | where |
|---|---|---|---|
| **Palm in the observation** | absent: palm width/length vary 40-100 mm across designs and nothing said so | `palm_keypoints`: the slab as four points, encoded like a link box | `obs_utils/observations.py`, `hand_sampler/build.palm_keypoints` |
| **Geometry frame** | link boxes and object keypoints measured from the design's palm centre, ÷ `hand_scale` -- palm length leaked into every token | measured from the end effector (`iiwa14_link_7` origin) in metres, one point every design shares; `ee_pos/ee_rot/ee_vel`, `keypoints_rel_ee` in the ee frame | `obs.geometry_origin: ee` in `PoseReach.yaml` |
| **Objects** | random pool of 1200; env *i* held entry *i* mod 1200, so each design met its own fixed dozen (the same dozen on both ranks) and designs were ranked on different objects | hand-picked pool of 24 (`curated_pools.DIVERSE_24`), dealt so every design meets every object exactly once per generation | `OBJECT_POOL=diverse24 OBJECT_ASSIGNMENT=design_cycle` in `common.sh`; `robot_spec.object_index` |
| **Per-design table** | last window before exit never written (`os._exit` skips atexit) | flushed explicitly | `coevolution/train.py` |
| **Epochs per generation** | 2000 (~3 h): the first four generations selected while the policy was still on its ~300-return plateau | 5000 (~8 h, `--time=10:00:00`) | `coevo_gen.sub` |
| Run root / wandb group | `debug_outputs/train_logs/coevo`, `coevolution_v1` | `debug_outputs/train_logs/17sep_coevolution`, `coevolution_v2` | |

Unchanged on purpose: model (d64/L4), 24,576 envs, minibatch, seed, keep 512,
40 generations (now ~14 days at 8 h each; `MAX_GEN` to shorten), the 1024-hand round-500
starting population, and the ranking key (`return_mean`).

The 10sep folder is pinned to the V1 setup (`env_modulo` objects, palm-centre
geometry via `EXTRA_HYDRA`) so it still reproduces what ran; the code's
defaults are now these.

## Not changed yet -- decisions still open

V1's lineage record showed selection on noise in generations 0-3 and one
family taking the whole population by generation 13 (`debug_outputs/10sep_coevo_analysis/README.md`).
Nothing here addresses that; these are the knobs and the candidates:

- **Start from a competent policy**: `launch.sh coevo $P coevolution_v2 CHECKPOINT=<.pth> RESUME_TOL=<tol>`
  works today (the baseline's last link is a natural source).
- **Selection pressure**: `KEEP=768` (keep 75 %) works today.
- **Lineage cap / niching / tournament**: not implemented; would go in
  `hand_sampler/evolve.py`.
- **Per-env, per-step banking** in `design_rewards.py` (goals per step, return per
  step, per-object breakdown): logging only; `--key` stays `return_mean`.

Smoked: 1024 designs with the new observation (382370), SHARPA (382118), 64
designs x 24 curated objects under `design_cycle` at 24 envs per design
(382472), all through scene build, three epochs and the reward tables.
