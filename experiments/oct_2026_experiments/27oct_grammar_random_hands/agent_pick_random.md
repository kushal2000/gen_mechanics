# Pick among the 100 random grammar hands for pen repose v2 (27 Oct)

This is a static analysis only: CPU, the grammar's own forward kinematics and geometry, no sim and no training.
Scripts are in `agent_pick_random/`, figures in `plots/agent_pick_random/`.

## Pick: `coarse_s0_025`

Spec: `assets/urdf/27oct_grammar_random_hands/coarse_s0_025/coarse_s0_025.spec.json`

**3 fingers, 13 joints** (12 finger joints plus a palm cut), all fingers 210 mm, passes C2.

| Finger | Base (y, z) mm | Facing | Joints (B bend, S spread) | Links mm |
|---|---|---|---|---|
| f0 (on the hinged palm section, cut facing 0 at 30 mm) | (-80, -40) | 240 | B B B S | 50-70-0-90 |
| f1 | (0, 40) | 330 | B B S B | 70-50-0-90 |
| f2 | (80, 40) | 90 | B B B B | 90-50-30-40 |

The three bases sit roughly in a T. f0 and f2 face each other across the palm, and f1 comes in from the side. Two
fingers are spatial: each has a spread joint between bends, and f0 also gets the palm hinge. Every finger is long
enough to curl back under the palm centre, and all three reach every contact zone under it.

## Runner-up: `coarse_s0_039`

Spec: `assets/urdf/27oct_grammar_random_hands/coarse_s0_039/coarse_s0_039.spec.json`

**5 fingers, 16 joints**, palm cut, passes C2:
- f0: spread-spread, tilted 60°. It and f1 sit on the hinged section.
- f1, f4: 2-bend fingers, 160 mm.
- f2: 4-bend finger, 220 mm.
- f3: bend-bend-spread-bend-bend, 170 mm.

On holding the pen it nearly ties the pick (GC3 0.88 for both). It scores a little lower on rotation (GROT 0.85
against 0.89) and costs 3 more joints. It also hedges the pick's main risk: about 4 fingers can touch the pen at its
working point against exactly 3 for 025, so it can regrasp while 3 fingers still hold.

## Ranked top 10

The table columns:
- **pick**: the fraction of pen yaws (18) at which the pen can be picked off the table, with no wrist yaw.
- **prot**: how far below the palm face the fingers protrude when pulled up as far as they go.
- **CH3 / ROTC**: collision-free 3-finger hold and rotation coverage at the best single pen centre (stage 3).
- **fine**: the same on a 4x finer set of axis directions, with fresh samples.
- **GC3 / GROT**: coverage of the actual goal distribution, with the wrist's height limits (stage 4).
- **touch**: the mean number of fingers that can touch the pen there.
- **score**: GC3 x GROT x pick x min(1, (16/J)^0.5).

Every hand in the top 10 picks up the pen at all 18 yaws.

| # | Hand | Fingers / joints | Cut | prot mm | CH3 | ROTC | fine CH3 / ROTC | GC3 | GROT | touch | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **coarse_s0_025** | 3 / 13 | yes | 12 | 1.00 | 1.00 | 0.99 / 0.99 | **0.88** | **0.89** | 3.0 | **0.784** |
| 2 | **coarse_s0_039** | 5 / 16 | yes | 41 | 0.93 | 0.82 | 0.95 / 0.93 | **0.88** | 0.85 | 4.0 | 0.748 |
| 3 | coarse_s0_062 | 3 / 12 | no | 12 | 0.90 | 0.81 | 0.92 / 0.86 | 0.81 | 0.79 | 3.0 | 0.645 |
| 4 | coarse_s0_058 | 6 / 22 | yes | 102 | 1.00 | 0.95 | 0.99 / 0.94 | 0.84 | 0.84 | 5.0 | 0.599 |
| 5 | coarse_s0_081 | 6 / 18 | yes | 81 | 1.00 | 1.00 | 1.00 / 1.00 | 0.78 | 0.72 | 4.9 | 0.531 |
| 6 | coarse_s0_088 | 3 / 11 | yes | 41 | 0.63 | 0.38 | – | 0.78 | 0.68 | 3.0 | 0.528 |
| 7 | coarse_s0_073 | 5 / 20 | yes | 12 | 0.57 | 0.49 | – | 0.85 | 0.69 | 3.7 | 0.524 |
| 8 | coarse_s0_006 | 5 / 18 | yes | 12 | 0.93 | 0.59 | 0.93 / 0.78 | 0.81 | 0.67 | 4.1 | 0.514 |
| 9 | coarse_s0_021 | 5 / 15 | no | 14 | 0.87 | 0.68 | 0.83 / 0.76 | 0.72 | 0.71 | 4.0 | 0.514 |
| 10 | coarse_s0_075 | 6 / 22 | no | 81 | 1.00 | 0.99 | 1.00 / 0.99 | 0.78 | 0.76 | 5.7 | 0.506 |

The full table for all 100 hands is in `agent_pick_random/ranking.json`; `rank.py` prints it.

**How robust the order is:**
- **Without the joint factor:** 025 (0.78) and 039 (0.75) stay first and second; 058 (0.71) moves to third.
- **Every metric on its own:** 025 is in the top 3 on every metric.
- **Where 081 / 075 lead:** they lead or tie on the static holding and rotation metrics (CH3, ROTC, fine). They
  fall back once the wrist's height limit counts (GC3, GROT; see below).

## Why this pick: what the task demands and how each was measured

All geometry is in the palm frame (x = palm normal toward the table). The pen is a 100 x 30 x 30 mm box. Fingertip
contact uses the distal link's real capsule (radius 16.5 mm). Other links are the grammar's servo-sized boxes (up to
57 mm thick), approximated as capsules.

1. **Holding the pen in any orientation under a fixed palm (stage 1, `screen.py`).**
   - **Setup:** sample every finger's joints uniformly (30k configurations each; the palm hinge at 5 levels) and
     voxelise where its fingertip capsule can be, together with which surface normals it can push on (the link may not
     come from inside the pen).
   - **The test:** put the pen on a grid of 4,851 centres below the palm (10 mm steps), in 30 axis directions (a hemisphere) x 3 rolls
     about its axis. The roll is free because success only uses keypoints along the axis.
   - **Holds:** the pen counts as held with 3 fingers when two distinct fingers can touch opposite faces (or the two
     ends) and a third finger can also touch it.
   - **Result:** 40 hands hold every direction at some centre (C3 = 1). No 2-finger hand scores at all.
2. **Rotating it (stage 2, `rotate.py`).** For each rotation axis that moves the long axis (6 per direction), some
   antipodal pair must be able to produce its contact velocities ω × r inside its fingers' positional Jacobian spans.
   If the pair only pivots, a third finger has to push.
3. **Collisions (stage 3, `clean.py`).** Stages 1-2 saturate, so here every grasp must also keep every link out of
   the pen and the palm plate, and keep the grasping fingers clear of each other.
   - **What changes:** this is the step that separates hands. 053 falls from C3 = 1.0 to CH3 = 0.03 and 086 from 1.0 to 0.20.
     025, 081 and 075 stay at about 1.0.
   - **Robustness check:** I repeated it with a new seed and 8 centres (`clean_seed17_8c.json`), and on a 4x finer set
     of directions (`fine_check.json`). The results held.
4. **Goals the wrist can actually reach (stage 4, `goal_cover.py`).** The goal centre goes up to 135 mm above the
   table, but the palm face can't go above 188 mm (wrist z ≤ 0.20 m). So a hand that only holds the pen 100 mm or more
   below its palm can't reach high goals. A finger that can't retract (a tilt-90 "peg") also forces the palm up for low
   goals.
   - **The test:** I ran stage 3 at depths 35-135 mm and integrated over the real goal distribution: uniform axis
     direction, and a centre height uniform between that orientation's floor and 135 mm.
   - **Result:** 025 and 039 hold most directions from about 65 mm down (`plots/agent_pick_random/hold_by_depth.png`),
     and their fingers retract to the palm. 081 and 075 hold only deeper and protrude 81 mm.
5. **Pickup with the wrist yaw locked (`pickup.py`).** The pen lies at any yaw, so the hand needs an opposed pair for
   every yaw. The palm may sit anywhere from touching the pen to 15 cm above it, as long as no finger is forced into the
   table.
   - **025:** it picks at all 18 yaws, and with the palm right on the pen (gap 0).
   - **Earlier bug:** the first version of this test only allowed a 0-30 mm gap and wrongly failed the hands with
     downward pegs (e.g. 034).
6. **Joint count.** Each joint costs RL sample efficiency. 025 gets full static coverage with 13 joints, the fewest of
   any hand that reaches it. The 4 x 4 primitive hands (16 joints) reached about 30 goals on this task; Wuji (20
   joints) reached 26.

025 wins because it is the cheapest hand that saturates every static test and keeps a shallow working depth:
- every contact zone under the palm is reachable by all 3 fingers;
- two fingers are spatial, so the opposed pair can roll the pen about any axis;
- the fingers fold flat, so the palm can sit on the pen to pick it up and the pen can be held high.

## Caveats

- **Static only.** Contacts are kinematic, not physical. Nothing here models:
  - the 0.5 N·m effort or the stiffness-3 position control on 210 mm fingers (025's fingertips are soft: very roughly
    1-2 cm of give per newton at full extension);
  - friction, slip or pushes;
  - action delay and EMA;
  - the RL learning problem itself.
- **Exactly 3 fingers (025).** While one finger moves to a new grip, the other two pinch the pen alone. That suits
  gaiting but is fragile against pushes. 039 (5 fingers) is the hedge.
- **Weights are judgement calls.** The 16-joint knee and the product form of the score are my choices. The top 2 hold
  without the joint factor.
- **2-finger hands** (20 of 100) score zero here by design, since every metric needs a third finger to regrasp or
  push. That is a judgement about 127° average turns, not a measurement.
- **Single seeds vary hugely** on this project (e.g. SHARPA 40 vs 8). One training run per hand won't confirm or refute
  this ranking; 2-3 seeds of the pick, the runner-up and a mid-ranked control would.

## Files

**Scripts** (`agent_pick_random/`):
- `screen.py`: stage 1.
- `rotate.py`: stage 2 (`--res 0.3` writes `rotate.json`; `--res 0.5` writes `rotate_res50.json`).
- `pickup.py`: pickup. It supersedes `screen.py`'s own pickup numbers in `metrics.json`.
- `clean.py`: stage 3.
- `fine_check.py`: the finer re-check of stage 3.
- `goal_cover.py`: stage 4.
- `rank.py`: the table and figures.
- `render.py`, `grasp_viz.py`: the renders.

**Outputs** sit next to each script as `.json` and `.log`.

**Figures** (`plots/agent_pick_random/`):
- `ranking.png`, `hold_by_depth.png`;
- `<hand>_curl50.png` for the top 10;
- `<hand>_grasps.png`: actual collision-free 3-finger grasps found for the pen horizontal, tilted and vertical, for
  025, 039, 062, 058, 081 and 075.
