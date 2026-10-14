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

## Measured (1000 random hands under the Feetech rules, seed 20261014; re-measured the afternoon of 10/14, `ours/feetech_1000.json`)

| Quantity | Value |
|---|---|
| Time to draw a hand | 0.043 s |
| Pass C1 (clearance) | 100% (by construction) |
| Pass C2 (fingertips can meet) | 80%: 1.25 draws per usable hand |
| Numbers per hand | 14-102, mean 55 |
| Finger counts 2 / 3 / 4 / 5 / 6 | 198 / 200 / 207 / 197 / 198 |
| Joints per finger 1 / 2 / 3 / 4 / 5 | 382 / 965 / 926 / 893 / 831 (one-joint fingers are rejected more: a short finger's pad lies over the palm more often) |
| Palm area (median) | 177 cm^2; commercial hands under the same outline rule 97 cm^2 at their real size, 196 cm^2 at the scale Feetech servos build them |
| Hands with a palm cut | 48% (fingers beyond it: 1 in 342 hands, 2 in 130, 3 in 5) |
| 0 mm links | 7% of links |
| Something sticking out behind the palm | 99.1% of hands; median 24 mm, at most 43 mm |
| Finger mass | median 397 g, at most 931 g |
| Gravity torque on a finger's first joint at rest, palm up | median 0.04 N m, at most 0.94 N m: none above the 1.39 N m limit (12% above the old 0.5 N m) |
| Object start raised (60 mm cube / 72 mm cube) | 65% / 77% of hands; median raise 103 mm, at most 234 mm |

One mutation of a usable hand returns a child 94% of the time, and 96% of those children are usable.

Search space (`ours/search_space.py`, `ours/search_space.json`; counted exactly, before the clearance check): 10^31.3 four-finger hands with joints on the three kinds (Codex's estimate for its proposal: 10^30.8), 10^46.0 for 2-6 fingers; 10^46.6 and 10^68.9 with all 31 axis directions; a palm cut multiplies a hand by at most 73. Of 3000 uniform samples of four-finger hands, 3% pass C1 and C2 (0.3% with all axes): about 10^29.8 (10^44.0) usable four-finger hands. Before the rebuild: about 10^50 (10^70).

Commercial hands (`hand_sampler/grammar_bench/conformed_hands.json`; `ours/commercial_scaled.py` and `.json`): in the morning, fitted at their real size, 18 of 19 fit within 10 mm / 20 degrees and none was buildable from Feetech servos (their bases sit closer than the servos allow). In the afternoon each was scaled up by the smallest factor (0.05 steps, at most 5 x) at which its fit keeps the servo link lengths, passes C1, keeps every joint's kind and stays within twice the target, fingers that clash refitted clear: 17 of 19 are buildable, at 1-3.1 x (SHARPA, now KaRMA's model of the hand alone, at 3.1 x and 9.5 mm / 8.2 degrees); 9 fit within the target and all within twice it, errors counted at the real size. ARMS and Dex1 are not buildable up to 5 x and keep their real-size fits and collision filters. Every hand keeps its finger count, joints per finger, joint types and kinds; SHARPA's, Shadow's and SVH's palm joints each become a cut with the same fingers beyond it; ARMS's two become one cut. The grammar's own limits follow the scaled hands x 1.1: links 200 mm, fingers 500 mm, bases 370 mm (10^63 four-finger hands); the Feetech rules are unchanged (10^31).

Isaac (RTX 4090, the validation population, 64 envs, 300 random steps): PASS: the simulated bodies match the grammar's kinematics to 0.0005 mm at reset, no NaN, mimic ties within 1.1e-3 rad, no collider-cooking warning. Again in the afternoon, with the scaled commercial hands and 1.39 N m (the validation population and the default one with SHARPA at 3.1 x): PASS both; at most 0.26 mm after 20 random steps; speeds up to 6.5 rad/s against the 5 rad/s limit; no contact between collision-filtered pairs.

## Open

- **The object start.** The object starts at the lowest height where it clears the hand by 5 mm at the start pose. At the 31.5-degree start curl, tilted fingers on the inner ring fill the space over the palm's centre, so most cubes start 10 cm or more up and fall onto the fingers; the worst (235 mm) come close to the 0.24 m drop termination. Even with straight fingers 49% of hands raise the 72 mm cube. A smaller object, a lower start curl or another spawn point are task decisions.
- **Torque.** The limit is now the HL-3915's stall torque, 1.39 N m (was Kushal's 0.5): no random finger exceeds it; SHARPA at 3.1 x needs 2.9 N m. The rated load is 0.44 N m, so stall is an upper bound.
- **The mount's size** (40 mm disc, 40 mm deep) is assumed until the real mount is drawn.
- **Two neighbouring bend-first fingers cannot be separated by a palm cut**: their housings are 7 mm apart and the seam needs 8.6 mm.
