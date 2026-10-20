# E16: rebuilding the grammar around the Feetech servo (2026-10-13 to 10-14)

**Question.** Martin's Notion notes of 10/13, the Feetech study (E15) and the grammar chat each changed part of the grammar. Which design satisfies the three principles (an expressive grammar; narrowed by rules, by construction, to what evolution can search and what can be built; as simple as possible)?

**How.**
- Codex (GPT-6-Astra, maximum reasoning effort) reviewed everything independently from `codex/BRIEF.md` and wrote `codex/REPORT.md`. `codex/review_calculations.py` reproduces its numbers into `codex/results.json` and `codex/calculations.txt`. `codex/feetech_geometry.py` is E15's motor model, unchanged.
- `ours/compare_layouts.py` compares base layouts. `ours/measure_feetech.py` measures 1000 random hands of the rebuilt grammar and writes `ours/feetech_1000.json`.
- Martin decided each open point on 10/14; the decisions and their reasons are in `GRAMMAR-LOCK-2026-10-07.md` (revision of 2026-10-14), and the rebuilt grammar is described step by step in `GRAMMAR-OVERVIEW.md`.

## What was decided, and why

| Decision | Reason |
|---|---|
| One grid, 10 mm and 30 degrees; the fine stage is future work | Martin's note 1 |
| The origin is the mount at the palm's centre, behind the back face (a 40 mm disc, 40 mm deep, assumed) | Martin's note 2.4: the hand mounts at its back, not at a wrist; the mount replaces the heel disc and is the smallest palm |
| Bases on a 40 mm grid, 40-90 mm from the centre (20 spots) | 35 mm lets neighbouring first motors collide in 57% of direction pairs, 40 mm in 28% (`ours/compare_layouts.py`); an inner radius of one grid step keeps rows of 5 (Codex's 55 mm allows rows of 3 only) |
| One overlap rule replaces the 10 mm grid, the 19 mm spacing and the rim rule | no spacing short of 67 mm is safe for every pair of motor directions (Codex), and a base inside the palm is fine if its motors clear everything |
| Fingers face outward (never back across the palm) | Martin's note 2.9.1 |
| The finger frame is tilt-defined again | Martin's note 2.9: + on every bend moves the fingertip toward the palm's front face; the frame depends on the finger alone |
| The base joint sits half a motor plus a wall below the front face (12 mm for a bend first, 16.5 spread, 11.5 twist) | Martin, 10/14: the first motor's housing is flush with the palm face |
| Palm 24 mm thick | a flat bend motor in its housing (20 + 2 x 2 mm); motors that stand deeper stick out behind |
| Hinge joints only; link floors from the two axes they join (30-50 mm), fingertips at least 30 mm | E15's plane rule, rounded up to the 10 mm grid |
| Two joints at one point anywhere, between two different axes square to the finger, never two in a row; the earlier motor slides along its own axis | Martin's note 2.11.2 (a 2-2-1 thumb); Codex's slide rule keeps the two motors a wall apart over the whole range |
| At most one palm joint, a straight cut across the finished palm; drawn on half the hands | Martin's note 2.13; a straight cut of a convex palm leaves two convex plates |
| Generation keeps the drawn finger count | Martin's note 2.12 |
| Commercial frequencies dropped (0 mm links, coupling, palm joints); only the 69/27/4 kind mix kept | uniform within the rules (Martin, 10/07: no human-shaped template) |
| Commercial hands only need their structure fitted | Martin, 10/14: approximate is fine, scale does not matter |
| Commercial hands scaled up by the smallest factor Feetech servos can build | Martin, 10/14 afternoon: "scale commercial hands up minimally so that they can be built with feetechs ... without collisions" |
| Joint torque limit 1.39 N m, the HL-3915's stall torque (was 0.5) | Martin, 10/14 afternoon: "use feetech torque numbers" |
| Fingers at most 225 mm under the Feetech rules (was 250) | Martin, 10/16 |
| Tilt 0 to 90 degrees: no finger points under the palm (was -30 to 90) | Martin, 10/16 |
| Tilt 0 when a finger's first joint is a bend | Martin, 10/16: the tilt turns the finger about the bend's own axis, so it duplicates that joint |
| The fingertip a capsule, radius 16.5 mm (half the yoke's width) | Martin, 10/16: a rounded fingertip, a little thicker than the yoke's depth |
| Add a finger: the hand's mean joint count rounded up, joints drawn as for a random hand | Martin, 10/16 (was one 40 mm bend) |
| A bend servo turns (its body on the back of the finger) only when the link before it is too short for it to lie along the link | Martin, 10/16: turned, only its 20 mm width lies along the link, so that link can be 30 mm instead of 40-50; ranges unchanged |
| The mount at the palm's centre: the base servos' hull centred within half a base step of it; draws and mutations move a hand on the grid to centre it | Martin, 10/16: no two hands may differ only in where the mount sits (13% of random hands had such a twin) |

## Measured (1000 random hands under the Feetech rules, seed 20261014; re-measured on 10/16 after the day's rule changes, `ours/feetech_1000.json`)

| Quantity | Value |
|---|---|
| Time to draw a hand | 0.096 s (0.049 before the centre rule and turned servos) |
| Pass C1 (clearance) | 100% (by construction) |
| Pass C2 (fingertips can meet) | 73%: 1.37 draws per usable hand |
| Numbers per hand | 14-98, mean 51 |
| Finger counts 2 / 3 / 4 / 5 / 6 | 211 / 199 / 212 / 183 / 195 |
| Joints per finger 1 / 2 / 3 / 4 / 5 | 586 / 1057 / 973 / 899 / 437 (every five-joint finger fits the 225 mm cap now, but long fingers fail the clearance check more often and are drawn again) |
| Palm area (median) | 162 cm^2; commercial hands under the same outline rule 174 cm^2 at the scale Feetech servos build them |
| Hands with a palm cut | 46% (fingers beyond it: 1 in 356 hands, 2 in 102, 3 in 5) |
| 0 mm links | 6% of links |
| Something sticking out behind the palm | 73% of hands; median 14 mm, at most 43 mm |
| Finger length | median 170 mm, at most 220 mm |
| Finger mass | median 344 g, at most 862 g |
| Gravity torque on a finger's first joint at rest, palm up | median 0.13 N m, at most 0.71 N m: none above the 1.39 N m limit (8% above the old 0.5 N m) |
| Object start raised (60 mm cube / 72 mm cube) | 43% / 60% of hands; median raise 17 mm, at most 209 mm |

One mutation of a usable hand returns a child 94% of the time, and 96% of those children are usable.

Search space (`ours/search_space.py`, `ours/search_space.json`; counted exactly, before the clearance check; re-counted on 10/16 with fingers at most 225 mm, tilts 0-90 degrees, tilt 0 for a bend first, capsule fingertips, the mount at the palm's centre, and short links turning the bend servo after them): 10^29.7 four-finger hands with joints on the three kinds (10^31.3 on 10/14; Codex's estimate for its proposal: 10^30.8), 10^43.7 for 2-6 fingers; 10^44.2 and 10^65.4 with all 31 axis directions; a palm cut multiplies a hand by at most 73. Of 20,000 uniform samples of four-finger hands, 42% have the mount at the palm's centre and 0.27% also pass C1 and C2 (0.015% with all axes): about 10^27.2 (10^40.4) usable four-finger hands. Before the rebuild: about 10^50 (10^70).

Commercial hands (`hand_sampler/grammar_bench/conformed_hands.json`; `ours/commercial_scaled.py` and `.json`): in the morning, fitted at their real size, 18 of 19 fit within 10 mm / 20 degrees and none was buildable from Feetech servos (their bases sit closer than the servos allow). In the afternoon each was scaled up by the smallest factor (0.05 steps, at most 5 x) at which its fit keeps the servo link lengths, passes C1, keeps every joint's kind and stays within twice the target, fingers that clash refitted clear: 17 of 19 are buildable, at 1-3.1 x (SHARPA, now KaRMA's model of the hand alone, at 3.1 x and 9.5 mm / 8.2 degrees); 9 fit within the target and all within twice it, errors counted at the real size. ARMS and Dex1 are not buildable up to 5 x and keep their real-size fits and collision filters. Every hand keeps its finger count, joints per finger, joint types and kinds; SHARPA's, Shadow's and SVH's palm joints each become a cut with the same fingers beyond it; ARMS's two become one cut. The grammar's own limits follow the scaled hands x 1.1: links 200 mm, fingers 500 mm, bases 370 mm (10^63 four-finger hands); the Feetech rules were unchanged (10^31; 10^29.5 since fingers are capped at 225 mm on 10/16). Rebuilt on 10/16 with the day's rules: 17 of 19 buildable, 7 within the target, all within twice it; ARMS is now buildable (3 x) and SHARPA is not (at 3.1 x two fingers touch the palm plate at rest and a base servo reaches into the cut's seam; at 4 x its thumb passes 500 mm), so SHARPA keeps its real-size fit and collision filter.

On 10/20 (Martin): two stages, turns and mirror images counted once, every finger count. Search space (`ours/search_space.py`, `ours/search_space.json`, 2-6 fingers, 3,000 uniform samples per stage): the coarse stage (the three kinds, 20 mm steps) allows 10^38.3 hands, about 10^37.4 with symmetry counted once; 65% of samples are centred and 2 of 3,000 also pass C1 and C2, about 10^35.2 usable. The fine stage (31 directions, 10 mm steps) allows 10^65.4 (about 10^64.5); none of 3,000 samples pass C1, so fewer than 10^61.9 are usable. Six-finger hands are about 99% of each. Options for cutting further, none adopted: `project-notes/grammar/search-space-options.md` (`ours/coarse_options.py`).

Commercial hands judged by the fingertip (10/20; `outputs/scratch/conformed_map.json`, not yet installed): each finger's fingertip within 15 mm RMS and 40 mm at worst over 256 joint samples, refitted on 64 of them at every scale. 17 of 19 buildable: smaller for Ability (1.3 x, was 1.8), Dex3 (1 x), XHand (1.5 x), Inspire (1.65 x), ORCA (1.75 x), Shadow, SVH and Wuji v2 right (2 x), Wuji v1 (2.15 x); larger for Tesollo (2.05 x) and MIDAS (2.1 x); Dex1 (1.75 x) and SHARPA (1.8 x) now buildable; LEAP and Wuji v2 left no longer (no scale keeps their maps within the limits).

Isaac (RTX 4090, the validation population, 64 envs, 300 random steps): PASS: the simulated bodies match the grammar's kinematics to 0.0005 mm at reset, no NaN, mimic ties within 1.1e-3 rad, no collider-cooking warning. Again in the afternoon, with the scaled commercial hands and 1.39 N m (the validation population and the default one with SHARPA at 3.1 x): PASS both; at most 0.26 mm after 20 random steps; speeds up to 6.5 rad/s against the 5 rad/s limit; no contact between collision-filtered pairs. On 10/16, after the day's rule changes (capsule fingertips as 50-point hulls, turned servos): PASS for 16 random hands, the validation population and the default one (SHARPA at real size); 0.0005 mm at reset, no NaN, no contact between collision-filtered pairs, no cooking warning.

## Open

- **The object start.** The object starts at the lowest height where it clears the hand by 5 mm at the start pose. At the 31.5-degree start curl, tilted fingers on the inner ring fill the space over the palm's centre, so many cubes started 10 cm or more up and fell onto the fingers. Since 10/16 (tilts 0-90 degrees, tilt 0 for a bend first) the object starts raised on 43% of random hands, by 17 mm at the median (was 66% and 105 mm); the worst (209 mm) still come close to the 0.24 m drop termination. Even with straight fingers 49% of hands raise the 72 mm cube. A smaller object, a lower start curl or another spawn point are task decisions. Martin, 10/20: set aside; the fingers now start open.
- **Torque.** The limit is now the HL-3915's stall torque, 1.39 N m (was Kushal's 0.5): no random finger exceeds it (at most 0.71 N m), nor any commercial hand at its scale (at most 1.04 N m). The rated load is 0.44 N m, so stall is an upper bound.
- **The mount's size** (40 mm disc, 40 mm deep) is assumed until the real mount is drawn.
- **SHARPA is not buildable since 10/16** (see above); why the refit cannot clear it is not yet confirmed. The 10/20 fingertip refit builds it at 1.8 x (not yet installed).
- **The fingertip-map thresholds** (15 mm RMS, 40 mm worst) are a first guess; under them LEAP and Wuji v2 left lose their buildable fits. Martin decides before the 10/20 fits are installed.
- **Two neighbouring bend-first fingers cannot be separated by a palm cut**: their housings are 7 mm apart and the seam needs 8.6 mm.
