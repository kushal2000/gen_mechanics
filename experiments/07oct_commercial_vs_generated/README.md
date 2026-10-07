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
* The cube and goal are drawn with the same seed but are NOT the same draw across hands (the joint count changes
  how the generator is consumed); the spawn distance above is one sample, not a distribution.

## Fixed on the way

`hand_sampler/build.py` `_author_palm` referenced a `pose` that the radial-palm commit (27b97c4) had removed:
every generated hand failed to build in sim (NameError) on `2026-10-02_physical_grammar` and the merge. The palm's
mass is placed at its centre again (`palm_box`), as before that commit.
