# In-hand reorientation (24 Sep)

A fixed palm-up hand with **no arm**, one 45 mm cube already in the palm, goal a
uniform random SO(3) orientation resampled on every success.

## Why

Co-evolution ranks 1024 designs by `return_mean` on a task where a 7-DOF iiwa14
carries the object to a 6D goal, so the arm can rescue a mediocre hand and the
fitness only weakly isolates the morphology being selected on. Parent-child
score correlation is ~0.15 and both V2 arms fixed to one lineage by generation
12. Removing the arm makes the hand nearly the only lever.

## What differs from `17sep_coevolution`

| | 17sep | here |
|---|---|---|
| task | `GenMech-PoseReach-Direct-v0` | `GenMech-InHandReorient-Direct-v0` |
| robot | iiwa14 + hand, 37 joints | **hand only, 30 joints** (`handonly:` prefix) |
| object | 24 curated tools, `design_cycle` | one 45 mm / 46 g cube |
| goal | 6D pose, position + orientation | **orientation only** |
| start | object on a table, must be lifted | **already in the palm** |
| dropped | object below z=0.1 | **> 0.3 m from the palm centre** |
| tolerance | 0.075 -> 0.01 curriculum | **fixed 0.0039 (0.1 rad), no curriculum** |
| env spacing | 1.2 | 0.6 |

`decimation`, the sim/PhysX block and every domain-randomization *range* are
unchanged from the base task on purpose -- they are tuned for this hardware, and
moving them alongside the task would make the comparison unreadable. The wrench
scales ARE zeroed, because `force_only_when_lifted` is always satisfied once the
object starts latched as lifted.

## Running

`ROBOT_SPEC` must carry the `handonly:` prefix; `common.sh` refuses without it.
That prefix is what drops the arm joints, and it has to live on the string
because the agent YAML interpolates the network's spec from the same value -- a
separate flag would let the policy build itself for 37 joints against a
30-joint articulation.

Co-evolution starts from the **initial seeded population**, `gen_s0_n1024` --
not the 500-round mutated `gen_s0_n1024_drift500_s0` that `17sep_coevolution`
began from. Starting at the seed means generation 0 is the grammar's own
distribution, so any drift in mass, joint count or reach is something THIS run
produced rather than something it inherited.

```bash
POP=handonly:gen_s0_n1024          # the seeded initial population, hand-only

# co-evolution, selecting every generation
bash launch.sh coevo $POP inhand_coevo

# the same population, never re-selected -- the control
sbatch --export=ALL,ROBOT_SPEC=$POP,STUDY_ID=inhand_baseline baseline.sub

# the imitation arm: gen-SHARPA alone, no population at all
sbatch imitation.sub
```

Three arms, and they answer different questions: **coevo vs baseline** is
whether selection helps at all; **coevo vs imitation** is whether evolution
finds anything better than the hand we already believe in.

## Known before you read results

* At 0.0039 the threshold is 5.7 deg, so **goals/episode may sit at 0 for a
  long time**. That is deliberate: the keypoint reward is progress-based and
  stays dense, so designs are still ranked by how far they rotate the cube, and
  the success rate stays an honest metric instead of one a curriculum loosened
  into existence.
* `keypoint_rew_scale: 2000` is a calculated starting point, not a tuned one --
  it restores roughly the pose task's shaping-to-bonus ratio now that the metric
  spans 0..117 mm rather than metres. Read `episode_cumulative/keypoint_rew`
  against `bonus_rew` in the first run.
* Fingertip reach is the live risk: a seeded population is mostly two-finger
  hands, and a design that cannot get a fingertip above the palm cannot touch a
  45 mm cube's equator at all. Those score identically and give selection
  nothing to rank. Worth measuring on the population before a full launch.
