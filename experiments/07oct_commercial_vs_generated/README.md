# 07 Oct: commercial hands vs their grammar replicas

Each vendor hand next to its replica from `assets/urdf/unified_dynamics_commercial_hands_replicas` (built by
`hand_sampler.commercial.fit`), same uniform actuator: what does rebuilding a hand in the grammar change? First
question, before any training: what the policy SEES at spawn in the in-hand reorientation task.

    # one Isaac Sim process per hand -> results/spawn/<name>.json (gitignored)
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/capture_spawn.py \
        --spec wuji2_left_uniform_handonly --name wuji2_commercial
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/capture_spawn.py \
        --spec handonly:assets/urdf/unified_dynamics_commercial_hands_replicas/wuji2/wuji2_left_replica.json --name wuji2_replica
    # (likewise leap_*, allegro_*)
    .venv_isaacsim/bin/python experiments/07oct_commercial_vs_generated/viser_spawn.py --port 8084

`capture_spawn.py` builds `GenMech-InHandReorient-Direct-v0` with 12491's saved config (full Wuji v2, 5 rad/s,
moving average 0.1, canonical palm frame + palm_extents), swaps in the hand, resets, and saves env 0: the policy
observation split by term, every body's pose, joints, cube, goal, palm frame. `viser_spawn.py` draws each pair as
spawned (the viewing URDF placed at the sim root with the sim's joint angles -- checked: every body within 0.00 mm
of the sim's) and tabulates the observation term by term. Per-joint terms pair through the replica's
`*.joint_map.json` (vendor joint -> grammar slot); the replica's ten unused template slots are listed as B-only.

## Spawn, seed 0 (mean |difference| over paired entries; per-joint terms are normalised)

| | Wuji v2 | LEAP | Allegro |
|---|---|---|---|
| obs dim, vendor / replica | 433 / 633 | 353 / 633 | 353 / 633 |
| `joint_lower` | 0.57 | 0.94 | 1.34 |
| `joint_upper` | 0.34 | 0.50 | 0.32 |
| `joint_pos` | 0.19 | 0.47 | 0.60 |
| `joint_link_bbox` | 0.014 | 0.017 | 0.043 |
| `keypoints_rel_ee` | 0.008 | 0.008 | 0.024 |
| `palm_extents`, vendor -> replica | 35x76x88 -> 25x100x75 mm | 49x121x102 -> 25x100x85 | 42x112x95 -> 25x100x85 |
| cube to palm centre, vendor / replica | 8.7 / 10.0 cm | 10.8 / 9.3 | 11.4 / 8.9 |

* **Joint limits are the biggest difference**: the replicas do not carry the vendors' joint ranges (the grammar
  sets its own), and since `joint_pos` is normalised by the limits it differs too even where angles agree.
* The replica's observation is the grammar template's (5 fingers x 6 slots = 30), padded with ghost slots; the
  vendor's has only its own joints. A policy reads them through different layouts.
* Every replica palm is 25 mm thick (the grammar's palm), against 35-49 mm for the vendors.
* Ghost slots show nonzero `joint_vel` at spawn (up to 0.42 on Wuji): unexplained, worth checking before training.
* The cube and goal ORIENTATIONS are the same draw for every hand at seed 0 (checked: identical quaternions), so
  `keypoints_rel_goal` matches exactly and `keypoints_rel_ee` differs only in the cube's centre. That centre is set
  by the design, not by chance: `reset.object_in_hand` with `in_hand_placement: fingertips` puts it halfway from the
  palm centre to the fingertip-pad centroid in the palm plane, and above the highest pad (or the palm face) by the
  cube's half-reach + 13 mm along the grasp normal.

## Where the cube spawns, and why (analysis/decompose_spawn.py, after the Allegro fix below)

The cube's centre relative to the palm differs by design, not by chance, and the decomposition is exact (predicted =
measured to 0.1 mm). Canonical palm frame, mm, replica - vendor:

| | height (grasp normal) | across (width) | along (length) | height comes from | length gap from where fingers start rel. palm centre |
|---|---|---|---|---|---|
| Wuji v2 | +18.9 | +0.1 | -5.9 | highest pad 20.9 -> 39.8 | -8.2 |
| LEAP | -12.1 | +4.2 | -7.6 | highest pad 39.4 -> 27.3 | -10.2 |
| Allegro | -26.6 | +1.1 | -9.5 | highest pad 42.8 -> 16.2 | -14.8 |

* **Height is entirely the highest fingertip pad** at the reset pose; palm thickness never binds (every pad sits
  above the palm face). Matching palm thickness would change `palm_extents` but not where the cube spawns.
* **The in-plane gap is mostly along the palm's length**, from the palm CENTRE: the vendor palm box (the palm link's
  mesh bounds, onboard.py) reaches back toward the wrist, the replica's palm stops 5 mm past its finger bases.

The spawn table above was taken before the Allegro fix; its Allegro column is superseded.

## Zero-shot transfer: the goal for the replica

The replica is good enough when the 52-hand Wuji v2 multi-embodiment policy (06oct_wuji2_multi_embodiment, 31826,
epoch 5000), which never saw it, works on it about as well as on the vendor hand it trained on.
`run_zero_shot.sub` (SLURM array, one target per task; eval_niches.py --target, 256 envs x 240 s, 50-goal cap):

    sbatch --array=0-1 experiments/07oct_commercial_vs_generated/run_zero_shot.sub

| target (job 48715, replica as of 8b3d79d) | goals in first episode | goals/min | drops/min | timeouts/min |
|---|---|---|---|---|
| Wuji v2 vendor (in training) | **23.4** | 36.3 | 0.50 | 0.68 |
| Wuji v2 replica (zero-shot) | **0.004** | 0.003 | 0.85 | 5.74 |

Baseline: transfer fails. The policy holds the cube on the replica (episodes end by the 10 s no-goal timeout, cube
8 cm from the palm) but never turns it to a goal.

## Fixed on the way

`hand_sampler/commercial.py` `_palm_axes` took the palm normal from the ORDER of the vendor's finger bases, so for
Allegro and MIDAS it faced the back of the hand: their replicas were built turned 180 degrees about the fingers
(Allegro's thumb opposed from the wrong side, a 42 mm in-plane cube gap). The normal now points where the fingers
flex (the larger side of each joint's travel). All four frames now agree with onboarding's (+1.00 per axis, MIDAS has
no onboarded spec), overlay base errors are unchanged, 60/60 commercial tests pass. Replicas regenerated.

`hand_sampler/build.py` `_author_palm` referenced a `pose` that the radial-palm commit (27b97c4) had removed:
every generated hand failed to build in sim (NameError) on `2026-10-02_physical_grammar` and the merge. The palm's
mass is placed at its centre again (`palm_box`), as before that commit.
