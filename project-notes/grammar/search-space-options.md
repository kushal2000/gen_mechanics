# Ways to shrink the search space further (2026-10-20)

Status: options only, none adopted. Martin, 2026-10-20: "save it as file covering options for future search space decreasing. lets not do it yet."

All counts are for every finger count together (2-6 fingers), exact unless marked approximate, and before the centre rule and the clearance check (C1). A palm cut multiplies each hand by at most 73 more and is left out.

## Where the search space stands

| Grammar | Hands (2-6 fingers) | Turns and mirror images counted once |
|---|---|---|
| Coarse stage (`FEETECH_COARSE_RULES`: joints on the three kinds, links in 20 mm steps), where random hands start and evolution runs first | 10^38.3 | about 10^37.4 |
| Fine stage (`FEETECH_RULES`: 31 axis directions, 10 mm steps), the refinement stage | 10^65.4 | about 10^64.5 |
| The grammar's own limits (No Rules) | 10^123 | |

Earlier versions: about 10^50 four-finger hands on 10/07 (coarse grid only; the fine 1 mm / 5 degree stage made it far larger), 10^31.3 four-finger hands after the Feetech rebuild of 10/14.

Six-finger hands are about 99% of each total. Of hands drawn uniformly from the coarse count, 65% have the mount at the centre and about 1 in 1,500 also passes C1 and C2, about 10^35 usable hands (`experiments/E16_grammar_rebuild/ours/search_space.py`).

## Why it is still large

One finger, on one base spot with one facing, has about 66,000 (10^4.8) coarse choices: joint count, joint kinds, link lengths, tilt. A hand of six independent fingers multiplies that six times, times where the six sit (20 spots, about 6 facings each). So any cut to one finger's options counts six times in the total, and the big cuts come from tying fingers together or from fewer independent per-link choices, not from trimming grid values further.

## Options for the coarse stage

From `experiments/E16_grammar_rebuild/ours/coarse_options.py` (starting from the coarse stage, 10^38.3):

| Option | Hands | Change | What is lost |
|---|---|---|---|
| Every finger the same design (placed freely; approximate) | about 10^14 | -24 | hands whose fingers differ |
| At most two finger designs per hand, each used by any number of fingers (approximate) | about 10^21 | -18 | hands with three or more different fingers (the fine stage can still make them differ) |
| At most 4 fingers | 10^26 | -12 | five- and six-finger hands |
| One link length per finger (all its links equal) | 10^27 | -11 | link proportions within a finger (the fine stage can still tune each link) |
| At most 3 joints per finger | 10^31 | -7 | four- and five-joint fingers, 2-2-1 thumbs |
| At most 5 fingers | 10^32 | -6 | six-finger hands |
| Fingers point straight out from the palm centre (no facing choice) | 10^34 | -5 | fingers angled across the palm |
| No twist joints | 10^35 | -4 | twist (4% of random joints) |
| At most 4 joints per finger | 10^35 | -3 | five-joint fingers |
| Links in 30 mm steps | 10^36 | -3 | link resolution |
| Tilt always 0 | 10^36 | -3 | fingers raised out of the plate |
| Only the outer ring of base spots (12 of 20) | 10^37 | -2 | bases close to the mount |

The cuts multiply only roughly when combined (they share the same per-finger choices).

## Recommendation, if we come back to this

1. **At most two finger designs per hand in the coarse stage.** Each finger keeps its own base spot, facing and tilt but takes one of two joint chains. Most commercial hands are four identical fingers and a thumb, and it is not a human-shaped template: placement and both designs stay free. The fine stage lets every finger diverge.
2. **One link length per finger in the coarse stage.** The coarse stage picks a finger's structure and overall size; the fine stage tunes each link in 10 mm steps.

Together about 10^10 to 10^12 hands. The other options remove something real (fingers, joints, twist, raised fingers) for 2-5 orders each.

No realistic cut makes the space small enough to enumerate: evolution sees a few thousand hands in a run (64 designs per generation). The case for these options is a coarse stage whose mutations make larger, more meaningful moves, not a countable space.
