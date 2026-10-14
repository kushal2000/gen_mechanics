# Brief: review and redesign the hand grammar (Codex, independent reviewer)

You are an independent design reviewer for a generative robot-hand grammar. Claude (another AI agent) is working on it with Martin, a robotics researcher. Martin wrote new notes (below), a separate Claude session studied Feetech servo buildability with you earlier today (study E15), and Claude has candidate ideas of its own. Your job: digest all of it, think hard, and recommend the simplest design that satisfies Martin's principles. Disagree wherever you think Martin's notes, E15 or Claude are wrong, and say why.

Work only inside your current working directory (write scripts and your report there). You may read anything under the repository (path below) but must not modify it. Do not read credential files.

## The project in two lines

"Evolve to generalize" trains one joint-token transformer policy (one token per joint) across many robot hands, and evolution chooses which hands it trains on. The grammar defines the space of hands evolution searches; the current task is in-hand reorientation of an object with a fixed, palm-up hand (no arm) in Isaac Lab.

## Martin's core principles (non-negotiable)

1. An expressive grammar (it should be able to describe essentially any hand, including commercial ones).
2. Narrowed and tractable for evolution by applying "rules" (rule sets that limit what generation and mutation produce, by construction, not by generating and rejecting).
3. Keep things as simple as possible. Martin strongly dislikes extra modes, variants, special cases and jargon. Every decision needs a stated reason.

## Read these first (repository: /home/singularity/Evolve to Generalize/gen_mechanics)

- `project-notes/grammar/GRAMMAR-OVERVIEW.md`: the current grammar as a step-by-step walkthrough (start here).
- `project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md`: every current decision with its reason.
- `project-notes/grammar/experiments/E15_feetech_buildability/summary.md` and `codex/REPORT.md`, `codex/feetech_geometry.py`: the Feetech servo study (your own earlier work; reuse the geometry code).
- Code: `hand_sampler/grammar/hand.py` (genotype, constants, rules, rim rule), `operators.py` (random draw, mutation), `derive.py` (frames, signs, palm outline, grasp point), `viability.py` (C1, C2).

## The motor

Feetech HL-3915-C001 (12 V, 1.39 N m stall, 100 rpm). Martin's model: a box 34 x 20 x 29 mm; the output axis runs parallel to the 29 mm edges (29 mm includes horn and idler), 9.5 mm from one end of the 34 mm side, assumed centred in the 20 mm width; 2 mm of structure between parts that move relative to each other. Each servo's body sits in the link before its joint; the next link holds the horn and idler through a printed bracket.

## Martin's newest notes (Notion, 2026-10-13), verbatim

1. For now, treat the coarse setting as the normal setting, save "refinement" / fine setting as a future work. For now just worry about coarse.
2. Grammar refinements from the current:
   1. Set palm thickness based on spacing we need for components, not the 37mm value we chose based on average commercial hand thickness.
   2. Fingers should be placed such that the "pads" / front of the finger are coplanar with the palm, or rather the "base" of the finger is placed so the joint is one half-feetech thickness away from the business end of the palm. this is so that the fingers are close as possible to the business end of things
   3. 20mm heel disc - minimal size? should we keep this?
   4. Currently assumes a spot to mount onto a "wrist" but change it so it is free 360deg and can be mounted at the back. This may change a lot of things, now our hands "base" is at the very center of the palm, instead of far away
   5. Reasoning for 6 fingers is weird - we just set it bc 5 already feels like a lot but we want to explore a bit above a human. Nothing deep here
   6. Grid in general is good versus our polar idea (Radius plus angle), but allows pretty large extremes of finger placements. We should make the distance between the minimum palm no-go zone circle and the outer edge (either circle or square) not that large. right now it is 10mm-160mm which is quite large? our reformulation of starting from the center rather than a wrist should help with this. We should also make the grid spacing wide enough to where we dont need a seperate general spacing (currently 10mm) and minimum finger distance (currently 19mm). Let's make the minimum distance based off the feetech feasibility thing, which i think would be ~35mm since a feetech is 29mm horn-to-horn and we want some space for the structure, but the agents that worked on the feetech freasbility should have the final say.
   7. Drop the rim rule / clarification w claude in progress
   8. the draw is also confusing
   9. facing and tilt are okay but i think we do a bad job of making sure the 90 degree side of the joint range points in the same direction for all fingers, i.e. moving positive in angle for all the fingers should move them in the same direction, in the direction facing normal out of the palm.
      1. However, does facing's current rules mean that a finger can spawn going inside of the palm? in other words, if we are at the north face of a square palm, and the finger "Faces" south, the finger is just inside the palm.
   10. 5 joints is good, decided based on how a fingertip poking something has one uncontrollable axis. So only need up to 5 joints
   11. The "Each finger's joints and limits" section will be heavily dependent on how we define the "buildable" / feetech feasibility criteria
       1. No coupled, prismatic, etc joints in this "buildable" ruleset. Only revolute/hinge.
       2. Having intersecting/0mm distance axes is not bad and not infeasible and should not be restricted to the MCP. Blocks interesting designs and blocks making a 5 dof 2-2-1 thumb, for example.
       3. The actual logic and stuff behind making it feasible to build with feetechs claude/codex should be able to figure out how to make it generate in a way that is buildable right away and not random gens + checks afterwards that rejects majority of designs.
       4. So claude/codex will decide the "min link length" (it will probably be more complicated than just one global. min link length since min link length depends on motor angles to each other, etc). Almost needing a "feetech grammar" where it knows for each joint, what the feetech's orientation is and it "picks" from feasible connections, and that orientation picked affect's what is "feasible" for the next joint.
       5. This will also affect how we spawn axes, but the 31 possible directions / choosing between x,y,z and palm at the start is fine.
       6. Straight at rest is good too.
   12. Clearance test is good but we should just give up if the 50 redraws dont work instead of falling back to fewer fingers so we dont bias toward fewer fingers
   13. Palm:
       1. Keep as a clearly optional toggle, on for now
       2. Palm geometry should be created AFTER the convex hull is drawn for the palm. In other words, we spawn the N fingers, make the palm, and the random palm axis we make should "slice" the palm in a plane defined by the joint axis and the vector normal to the palm, so at rest the palm is one cohesive shape
       3. Should be able to have multiple fingers on one palm joint from the beginning, based on if a finger lies to the "left or right" of the palm's axis. For example, on a human hand, if we choose to draw a palm dof between the middle and ring finger, the pinkie finger should end up on a palm body with the ring finger.
       4. Having the hinge ONLY be possible in +z is not good. We started formulating like: for the finger designated as being on a moving palm body, draw a line between its MCP and the MCP of the closest finger from the hand's origin (the palm finger should be the farthest from the origin),
          1. But then got stuck because theres situations where we could accidentally "cut" into another finger. A solution could be that its not like slicing a cake with an infinitely long knnife, we can cut a "slice" with a bend to the origin to permit a large enough palm area to manipulate the object but not to where it impedes other joints.

## Martin's plan from 2026-10-12 (Notion), verbatim summary

"Overall layout: Very expressive grammar which we tame using rules. Focus on 'feetech buildable' ruleset for now and try out 'general' ruleset later to see if generalization improves." His critique of House of Dextra (arXiv 2512.03743): one shared palm radius loses knuckle-plus-thumb palms; fingers only normal to the palm edge (no parallel fingers); convex-hull palm is good but avoid triangle palms (minimum wall angles, or a forced mounting point); a mounting point at the back of the palm (like D'Claw) with a minimum area; the same base joints for every finger is not expressive enough (joint ordering should vary); also sample x- and y-axis mounting tilts.

## Facts measured in Claude's session today (current grammar, Evolution Rules, coarse)

- Random hands: 74% viable (C1 100% by construction, C2 74%), 1.4 draws per viable hand; 4.0 fingers, 2.8 joints per finger, 134 mm per finger on average; 14-97 numbers per hand (mean 51).
- Rim rule (a base may be at most 8 mm inside the convex hull of the heel disc and the bases): without it 35% of random hands have a base more than 8 mm inside the palm (median 21 mm, up to 69 mm); with it none; viability is unchanged (C2 73% vs 72%). Dex3's thumb (between the wrist and its two fingers) is 22.6 mm inside, so Dex3 breaks it. C1 never compares a finger's first link with its own palm plate (they touch by design), so a finger lying inside the plate is not caught by C1.
- Facing is uniform over the full circle and independent of where the base is, so a finger on the north edge can face south across the palm (Martin's note 9.1 is correct).
- Palm size: bases are uniform over the 10-160 mm ring around the wrist, so the median base is 112 mm from the wrist and random palms have median area 169 cm^2 against 71 cm^2 for the 19 commercial fits; 77% of random palms are larger than the largest commercial palm (LEAP, 95 cm^2).
- Closing direction history: originally each finger's closing direction was the palm normal turned by the finger's tilt (the plain "curl toward the palm face" convention). With facing free, 21% of random bending joints then moved the fingertip away from the object (fingers pointing over the palm curl backward), and Martin saw fingers curling to the back. It was changed to "close toward the grasp point G" (above the palm outline's centroid, where the object starts); 100% then close toward G, but G depends on every base, so moving one base turns every finger's closing direction slightly.
- Object start: the object (60 mm cube) starts at G. At the start pose (every bend joint at 0.35 of its upper limit, 31.5 deg), some link already overlaps that cube in 68% of random hands and in 9 of the 19 commercial fits; nothing checks this.
- When a finger cannot be placed clear after 50 redraws, the current draw ends the hand with fewer fingers (Martin's note 12 objects).
- Commercial coverage of the current grammar: 17 of 19 commercial hands fit within 5 mm / 10 deg on the 1 mm / 5 deg grid.

## Open decisions left from E15 (asked to Martin, not yet answered except where his notes above answer them)

1. Base knuckle only for crossing axes: Martin's note 11.2 says no, allow 0 mm (crossing) axes anywhere, e.g. a 2-2-1 thumb.
2. Should simulated Feetech-hand links be servo-sized (about 33 x 24 mm, 35 x 26 with 6 mm corners) instead of the 19 x 18 mm links? Unanswered.
3. "2 mm structure" = one wall between moving parts; shaft centred in the 20 mm width. Assumed.

## Claude's candidate ideas (critique, improve or reject each)

a. **Origin at the palm centre, mounted at the back.** The origin (0, 0) is the mount point on the back face; a central disc is both the minimum palm and a keep-out zone for finger bases (its radius set by the mount footprint plus the base servos' bodies, which lie in the palm pointing back toward the centre).
b. **Bases on a square grid whose pitch is the rule set's finger spacing** (Feetech about 35 mm), inside an annulus or square between an inner keep-out radius and an outer bound; no separate spacing rule; the random draw picks N distinct grid points uniformly.
c. **Facing restricted outward:** within +/-90 deg of the radial direction from the origin, so no finger points across the palm.
d. **Closing direction back to "palm normal turned by the tilt"** (so + moves every finger toward the palm's business face, Martin's note 9); with facing outward it agrees with "toward the object" (fingers curl up and back over the palm), and it no longer depends on every base. A finger pointing straight up (tilt 90) closes toward the centre.
e. **Replace the rim rule with "no part of a finger may be inside the palm"** (C1 without the first-link exemption, apart from a short attachment length at the base), applied during generation like the rest of C1.
f. **Feetech joint chain, generated buildable by construction:** draw the axes first (on a kind at the start, from the coarse 30 deg set in mutations), then each link length from [L_min(a, b), 90 mm] using E15's plane rule, or 0 mm as a crossed pair (E15's offset module: the earlier servo moved along its own axis by S(t), allowed at any joint, with a fixed rule for which side it goes); fingertip at least T_min(a); finger at most 250 mm.
g. **Palm thickness from the servo:** 24 mm fits a bend-first servo lying flat; with the base joint on the mid-plane, a 24 mm-high finger's pads are flush with the palm face (Martin's note 2.2). Taller first servos (spread-first needs 33 mm) protrude from the back.
h. **Palm joints as straight chords:** after the hull palm is built, a palm joint is a straight cut across it (a chord); its hinge lies along the chord in the palm plane; every finger on the far side from the mount moves with that section (so several fingers can share one); the chord must keep clear of every base servo; chords may not cross. Martin's alternative: a line from the designated finger's base to the nearest finger's base, with a bend toward the origin to avoid cutting other fingers.
i. **Clearance:** keep the drawn finger count N and restart the hand if a finger cannot be placed, instead of ending with fewer fingers.
j. **Rule sets carry the physical constants:** the grammar allows any position, any axis, any length; each rule set fixes its grid pitch, palm thickness, link cross-section, minimum link lengths, joint types and whether palm joints are on. "Feetech buildable" now; "General" (commercial-derived, current Evolution Rules) later.
k. **Coarse only** for now (10 mm / 30 deg steps; the fine stage is future work).

## Questions

1. Evaluate a-k and Martin's notes together. Propose the simplest overall design, written as the order in which a random hand is generated (origin, bases, finger directions, joints and links, palm, palm joints, derived values), with concrete numbers for the Feetech rule set. Where Martin's notes are ambiguous, state your interpretation.
2. Base layout numbers, using the E15 servo model: with outward-facing fingers whose first servo lies in the palm with its body pointing back toward the centre, what inner keep-out radius keeps base servos clear of a central mount (say how it scales with the mount size; assume a 40 mm diameter mount if you need a number), and is a 35 mm square pitch enough for neighbouring base servos (on-kind first axes, facings within +/-90 deg of radial)? If not, what pitch or what single constraint? What outer bound do you recommend, and circle or square?
3. Palm thickness and base depth for "pads coplanar with the palm face": what fits bend-first, spread-first and twist-first base servos? Protrude at the back, or a thicker palm?
4. Crossing axes at any joint: is E15's offset module usable mid-finger (a 2-2-1 thumb)? Simplest rule for which side the moved servo goes; does it then collide with the finger's own other servos or with neighbouring fingers often? Should 0 mm be allowed for every axis pair, or only some?
5. Closing direction: is (d) right, and does it achieve Martin's note 9? Edge cases?
6. Is (e) a sound and simple replacement for the rim rule? Pitfalls?
7. Palm joints: compare (h) with Martin's line-plus-bend idea. Recommend the simplest construction that keeps the palm one shape at rest, lets several fingers share a section, and never cuts a finger's servo. What does a Feetech palm hinge need physically?
8. The object-start overlap (68% of random hands): the simplest fix?
9. Tractability: roughly how large is the coarse search space under your proposed Feetech rule set, compared with the current Evolution Rules? Anything else worth cutting?
10. What else would you drop or simplify?

## Deliverable

Write `REPORT.md` in your working directory (at most about 3,500 words, plain language, no hype):
1. A table of recommendations: item, recommendation, one-line reason.
2. The proposed grammar as a numbered generation walkthrough (origin first), Feetech rule set numbers included.
3. Answers to questions 1-10, with numbers; for every computed number say how it was computed (put scripts next to the report).
4. Where you disagree with Martin's notes, E15 or Claude, and why.
5. Questions that only Martin can answer.
