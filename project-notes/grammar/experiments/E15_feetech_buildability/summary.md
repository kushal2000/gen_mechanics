# E15: how close can two Feetech servo joints be? (2026-10-13)

**Question (Martin, 10/13).** The "Feetech buildable" rule set needs the minimum distance between consecutive joints. For two servos end to end with parallel axes Martin estimated 34 - 9.5 + 9.5/sin 45 + 2 mm of structure = about 40 mm, and noted that "once we angle them, you can make the axes pass through each other". How does the minimum depend on the two axes, and what rule should generation and mutation enforce?

**Two independent analyses, same motor model.**
- `ours/`: joints as cylinders (the servo's front end, horn, idler and bracket), servo bodies as boxes, distances from dense surface samples, swept over the joint ranges.
- `codex/`: Codex CLI (GPT-6-Astra, maximum reasoning effort), from the brief in `codex/BRIEF.md`. Exact box-to-box distances, analytic bounds, continuous-motion error bounds. `python3 verify.py` reproduces its tables and asserts every quoted number (rerun 2026-10-13: all checks passed, tables identical). Full report: `codex/REPORT.md`.

The two agree on every main number; where they differ (by 1-6 mm) it is because ours reserves room for the bracket all the way round each joint, while Codex's leaves one 2 mm gap between moving bodies, which is Martin's model.

## Motor model (Martin)

Feetech HL-3915-C001 as a box 34 x 20 x 29 mm. The output axis runs along the 29 mm edges (29 mm includes the horn and the idler), 9.5 mm from one end of the 34 mm side; we assume it is centred in the 20 mm width. Each servo's body sits in the link before its joint; the next link holds the horn and idler through a printed bracket around the servo's front end. Parts that move relative to each other keep 2 mm (one printed wall).

## Results

1. **Servos end to end along the finger: 41 mm, whatever the angle between the axes.**
   - For any two axes square to the finger the exact minimum is 40.29 mm: the next servo's back (24.5 mm behind its own axis) must clear the previous servo's front corners, which swing on a circle of sqrt(9.5^2 + 10^2) = 13.79 mm, plus 2 mm. The worst angle is 46.5 degrees of bend, inside the range; at 40 mm the gap is 1.7 mm. Martin's estimate holds; on the 1 mm grid it is 41.
   - Bend then bend, bend then spread and spread then bend all need 40.3 (39.7 when the first joint is a spread, whose +/-30 degree range never swings the corner fully forward).
   - What matters is each axis's angle to the finger (x). A plane square to the link separates the two servos over the whole range, giving (Codex):

     `L_min(a, b) = 2 + 14.5 (|a.x| + |b.x|) + 13.79 sqrt(1 - (a.x)^2) + 24.5 sqrt(1 - (b.x)^2)`, rounded up

     a = the earlier joint's axis, b = the next. Twist then bend 41; bend then twist 31; two twists 31; axes tilted toward the finger up to 51 (the formula's maximum over all axes).
   - No rule based only on the angle between the two axes can work: bend-spread, bend-twist and twist-bend are all 90 degrees apart and need 40.3, 30.3 and 41.
   - Ours gives the same formula with 1-2 mm more when the next joint is a twist (its bracket plates lie along the finger).

2. **Crossing axes (two joints at one point, 0 mm) need one servo moved along its own axis.**
   - With both servos on the finger's line, 0 mm is impossible: both boxes contain the joint point.
   - Moving the earlier servo along its axis by `S = 16.5 + 14.5 |cos t| + 10 sin t` (t = angle between the axes) clears the next one over the whole range: 26.5 mm at 90 degrees, about 34 at 30-45 degrees, 31 for parallel axes. Its far face ends up 41-49 mm from the finger's line. Ours, with the bracket all round, needs 32-40 mm. Partial moves save little (perpendicular axes: a 29 mm move shortens the link by only 7 mm).
   - Where the moved servo goes: along a spread axis, behind the hand (or into the grasp side, which blocks the object); along a bend axis, sideways into the next finger's space; along a twist axis, back along the finger.
   - At the base, a spread-then-bend knuckle puts the spread servo 12-41 mm behind the joint, i.e. 22.5 mm (24.5 mm with a wall) behind the back face of the 37 mm palm.
   - LEAP does this. Its URDF collision boxes put the side motor about 26 mm along its axis under the finger, which is how it gets a 15 mm link from the side joint to the PIP. Codex built a Feetech assembly that reproduces LEAP's 38 / 15 / 36 mm links (spread servo moved 26.5 mm toward the back of the hand; the last servo's body pointing toward the fingertip) and checked every pair of servos over the joint ranges.
   - With unlimited offsets every pair works at 0 mm, so an offset has to be a named module with its own geometry, not a general freedom.

3. **Other compact layouts exist but can't be used on every link.** A servo body hanging off the back of the finger instead of lying along it: about 32 mm (ours). Two servos in the same link with their fronts facing: 19 mm. Each fixes how the neighbouring joints are mounted, so they can't repeat down a finger.

4. **Fingertip:** the tip must sit beyond the last servo's swinging corners: `2 + 14.5 |a.x| + 13.79 sqrt(1 - (a.x)^2)`, i.e. 16 mm after a bend or spread servo, 17 after a twist, at most 23. The grammar's 10 mm is too short.

5. **Finger cross-section:** a 2 mm shell around a bend servo is 33 mm wide (along its axis) x 24 mm high; 35.4 x 26.4 mm with the 6 mm corner radius. A spread servo is 24 x 33. The simulated 19 x 18 mm links are about half that.

6. **Finger bases:** neighbouring first servos in a row need 33 mm when the first joints are bends (servos side by side along their axes) and 24 mm when they are spreads. The 19 mm spacing is too small; no single value covers every orientation short of 60-67 mm.

7. **Palm:** a flat first servo fits the 37 mm plate (24 mm for a bend, 33 for a spread). Some axis tilts need up to 40.8 mm, so tilted fingers' first servos stick out of the plate, as does a base knuckle (item 2).

8. **Palm joints, coupled and sliding joints** need extra mechanisms: a split palm with a motor pocket and bracket; an 11:10 transmission for a motorless coupled joint; a screw drive for sliding.

9. **Whole-finger collisions are outside any pair rule.** Five bends 41 mm apart, each turned to +90 degrees, close a square and put the fifth servo on the first. A pair rule guarantees neighbouring servos never touch; servos further apart need a whole-finger check (as C1 does for links) or limits in the controller.

## What this means for the rule set (proposed, for Martin's decision)

| Rule | Value | Reason |
|---|---|---|
| Joint types | hinge only | one servo per joint; coupled and sliding joints need extra mechanisms (8) |
| Link between consecutive joints | at least `L_min(a, b)`, rounded up: 41 mm for axes square to the finger, at most 51 | servos end to end, each in the link before its joint, no offsets: Martin's 40 mm generalised to any axes (1) |
| Two joints at one point (0 mm) | only as a named base knuckle: the first servo moved 26.5-34 mm along its axis, behind the palm | the MCP case; LEAP's solution; costs 22.5-25 mm behind the palm (2) |
| Fingertip link | at least 16 mm (bend or spread), 17 (twist), formula up to 23 | the tip must clear the last servo (4) |
| Finger bases | first servos may not overlap (33 mm for a row of bend-first fingers, 24 for spread-first) | servos side by side in the palm (6) |
| Palm joints | none at first | needs a split-palm module (8) |
| Generation clearance (C1) | includes the servo bodies | pair rules miss servos further apart (9) |
| Simulated link shape | about 35 x 26 mm instead of 19 x 18 | the built finger is a shell around the servo (5) |

Fingers 2-6, joints 1-5 and finger length at most 250 mm stay: five joints need at most 4 x 51 + 23 = 227 mm.

## Open questions

- Allow the base knuckle (and so servos sticking out behind the palm)?
- Simulate Feetech hands with servo-sized links (or the servo boxes as extra colliders)?
- "2 mm of structure" is taken as one wall between moving parts. Two separate 2 mm shells would mean 4 mm and longer minimums.
- The shaft is assumed centred in the 20 mm width; the horn and idler bolt pattern, cable exits and real travel should be checked against a drawing or a servo in hand.
