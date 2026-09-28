# Opus review: G0 grammar screen and viability oracle

Date: 2026-09-27. Read-only. The reviewer reproduced the report's numbers exactly (5/16/28 viable; 351/516/483 admitted out of 2000), and a reimplementation of the oracle matched it on 96 designs. Scripts are in the session scratchpad under `g0review/`.

## Confirmed issues
1. **Capsule length.** The oracle models each capsule as the segment [0, L] with radius r (grammar_envelope.py:793-798). The sim builds a capsule of total length L (author_grammar.py:119,125), so the oracle's capsules are 2r longer than PhysX's. Every same-digit, gap-2 rejection is false: 40/200 of V1's and 57/200 of V3's overlap rejections.
2. **Digits mount on the host's centre axis** (derive.py:614). Mount-fraction steps of 5-20 mm are smaller than 2r (16-24 mm), so mounts closer than 2r overlap by construction. This is the dominant overlap cause.
   - The V2 spacing planner cannot reach 29 mm on hosts shorter than 58 mm, and ignores cross-host coincident mounts.
   - Only 3/256 (V1) and 5/254 (V3) five-digit hands pass, so the digit axis of the descriptor grid is mostly closed.
3. **Wrong pose.** Overlap is checked at q=0, but the env resets every episode to default_q, a 35% curl. 17/351 (V1) and 37/483 (V3) admitted designs overlap by more than 3 mm at that pose.
4. **Spawn point beyond the fingertips.** The spawn sits 35 mm past the farthest mid-curl fingertip (grammar_envelope.py:920-925), and the 200-sample reach sweep undercounts.
   - 45% (V1) and 34% (V3) of fingers cannot reach it even fully extended. LEAP fails the >= 2 criterion.
   - Tip centroid + 3 cm raises V3's >= 2-reach from 24% to 46% (dense sweep).
5. **V3 is not as labelled.** `sample_bend` also bends phalanx 0 (derive.py:252), tilting every mount by 15-45°. This breaks opposition: 142° median, 105° in mixed-host hands, and 37% of "opposing" digits point within 90° of the others. The "shared hinge plane" claim is not delivered either (8% of successive axes within 20°).
6. **Explorability metric.** The CPU MAP-Elites archive admits structurally admitted but non-viable children, and the founder counts differ between variants (9/20/40).

## Correct
- Parent-child exclusion and ghost-carrier handling.
- The root-capsule model, apart from the r overhang at each end.
- Byte-identity of existing variants (400 seeds each).
- No hidden count changes, except more digits landing on one jointed carrier under V2.

## Statistics
- **Viability 95% confidence intervals:** V1 0.25% [0.08, 0.58], V2 0.80% [0.46, 1.30], V3 1.40% [0.93, 2.02].
- **Ordering (exact McNemar, one-sided):**
  - V1 < V3: p < 1e-4 at every threshold.
  - V1 < V2: p = 0.010, but it flips without the overlap gate.
  - V2 < V3: p = 0.044, marginal.
- **Cells occupied (5/6/9) are a sample-size effect.**

## Sensitivity (seeds 0-499, % viable)

| | 3 mm, >=1 | 3 mm, >=2 | 5 mm, >=2 | 10 mm, >=2 | built capsules, >=2 | built capsules + dense reach, >=2 | no overlap gate, >=2 |
|---|---|---|---|---|---|---|---|
| V1 | 16.8 | 0.2 | 0.4 | 0.8 | 1.0 | 1.4 | 7.4 |
| V3 | 21.6 | 1.4 | 1.8 | 5.0 | 4.0 | 5.6 | 16.6 |

## Verdict
Do not run the pilot with this oracle as is. V3 > V1 is real under every threshold, but the absolute rates, the digit-count reach and the reach metric are dominated by artifacts.

**Fix order:**
1. Use the built capsule core [r, L-r] in `rest_overlap_pairs`, and also check at default_q.
2. Spawn at the tip centroid plus clearance. Use a dense sweep or an IK minimum distance. Validate that LEAP, Allegro and xhand reach with >= 3.
3. Mount digits on the host surface, or add cross-host spacing.
4. Fix the phalanx-0 bend and the opposition host frame, or relabel V3.

Then rerun G0 (about 6 minutes on 28 cores).
