# Task: when are two consecutive servo joints buildable? (Feetech HL-3915 hand grammar)

You are helping a robotics researcher (Martin) finish a generative grammar for dexterous robot hands. The grammar can describe almost any hand; evolution runs inside a "rule set" that limits it. The next rule set is **"Feetech buildable"**: every hand it generates must be buildable from one servo per joint (Feetech HL-3915-C001) plus 3D-printed links. Your job is the geometric core of that rule set: **the minimum distance between consecutive joints, as a function of how their axes are oriented**, and a rule simple enough that a random generator and mutation operators can enforce it by construction (not by rejection where avoidable).

Work only inside the current directory (write scripts, results and the report here). Do not edit anything else, do not use the network, do not install packages. Python 3.10 with numpy 1.26 and scipy 1.8 is available (`python3`). You may read the repository (this repository) for context; the relevant files are listed at the end, but everything you need is in this brief.

## The motor (from Martin)

Feetech HL-3915-C001 (12 V, 14.2 kg·cm). Treat it as a rigid box:

- 34 mm long, 20 mm wide, and **29 mm along the output axis** (the 23 mm body plus the output horn on one side and the idler on the other; the 29 mm already includes both).
- The output axis is parallel to the 29 mm edges. It sits **9.5 mm from one end of the 34 mm dimension** (call that end the front). We assume it is centred in the 20 mm width (10 mm from each side); say so if your answer depends on it.
- Local frame used below: u along the output axis, v along the 34 mm length pointing from the axis toward the back end, w = u x v. The box is |u| <= 14.5, -9.5 <= v <= 24.5, |w| <= 10, and the output axis is the line v = w = 0.

Martin's rule for two servos end to end with **parallel** axes, as in a finger: 34 - 9.5 + 9.5/sin 45 deg + 2 mm of structure = about **40 mm** between the axes. Reasoning: the next servo's back end (24.5 mm behind its own axis) must clear the corners of the previous servo's front end, which swing on a circle of radius about 9.5/sin 45 = 13.4 mm (13.8 mm with the 10 mm half-width) as the joint turns, plus 2 mm for the printed bracket. His words: "that's just for lengthwise, like adding the motors end-to-end like for a finger. For things like the MCP, things get complicated ... the axis distance thing maybe isn't the best way to look at things: for the Feetech, yes our 40 mm distance is true for parallel axes, but once we angle them, you can make the axes pass through each other. It's a bit tough."

## The grammar's finger (what the rule must be stated in)

- A finger is a straight chain at the zero pose. Its link frame (shared by all its links at q = 0): **x along the finger**, **y = the closing direction** (toward the grasp point), **z = x cross y**.
- Joint j sits on the finger's centre line at p_j = base + X_j x, where X_j is the sum of the earlier link lengths. Link j runs from joint j to joint j+1; the last link runs to the fingertip.
- Joint j's axis is the **line through p_j** with direction `direction(az, el) = (sin el, cos el sin az, cos el cos az)` in (x, y, z): (0, 0) = z (a **bend**, i.e. flexion), (90, 0) = y (a **spread**, i.e. abduction), el = 90 = x (a **twist**, i.e. roll about the finger). Angles are on a 5 degree grid (coarse mutation steps are 30 degrees), so oblique axes between these kinds are allowed.
- Each joint takes the range of its nearest kind: bend -30 to +90 deg, spread -30 to +30, twist -90 to +90. Positive bend closes the finger toward the grasp point.
- Current link lengths (general rule set): 0 mm (two joints at one point) or 15-90 mm; the fingertip link 10-90 mm; a finger at most 250 mm; 1-5 joints per finger; 2-6 fingers per hand.
- The palm is a flat plate 37 mm thick; each finger's base (joint 0) sits on the plate's rim at its mid-plane; the finger leaves the base at an in-plane "facing" and a tilt out of the plate of -30 to +90 deg. A finger's first servo would sit in or on the palm.
- The grammar also has palm joints (a hinge in the plate carrying one or more fingers) and coupled joints (a joint that follows the previous one at ratio 1.1, no motor of its own); say whether the Feetech rule set can keep them.
- In simulation, links are currently rounded boxes 19 mm x 18 mm (corner radius 6 mm) along the centre line; the motors are not modelled.

## Real data to calibrate against (our fits of commercial URDFs, within 1 mm)

- **LEAP Hand** (Dynamixel XC330 servos, 20 x 34 x 26 mm, about the same size as the Feetech): each finger is bend -> [38 mm] -> spread -> [15 mm] -> bend -> [36 mm] -> bend -> [48 mm to the tip]. So: bend then spread at 38 mm, spread then bend at 15 mm, two parallel bends at 36 mm (less than Martin's 40).
- **DClaw** (Dynamixel XM430, 28.5 x 46.5 x 34 mm): spread -> [68] -> bend -> [69] -> bend -> [84 to the tip].

Explain these with your model if you can (e.g. which assembly conventions and offsets make LEAP's 15 mm spread-then-bend possible, and why its parallel pair is 36 mm, not 40).

## What to work out

1. **What determines feasibility.** Set up the problem precisely. Design freedoms to consider for each servo: which of its two links holds the body (the other link holds the horn and idler through a U-bracket around the front end); sliding the servo along its own axis (the axis is a line, so this does not change the kinematics, but it makes the hand bulkier and moves the motor off the finger's centre line); rotating it about its own axis (which way the 34 mm body points). Use a 2 mm minimum gap between bodies that move relative to each other (the printed bracket); bodies on the same link may touch. Check every joint position over the joint's range, not just the zero pose. Consider whether non-adjacent servos (j and j+2) matter when a link is 0 mm.
2. **The minimum link length L_min between consecutive joints** for:
   - the on-kind pairs: bend-bend, spread-spread, bend-spread, spread-bend, and the pairs with a twist;
   - oblique pairs: L_min as a function of the angle between the axes (and whatever else it depends on, e.g. the angle each axis makes with the finger);
   - with and without sliding along the axis (and if sliding is allowed, how far the motor then sticks out from the finger's centre line, and in which direction: sideways, toward the back of the hand, or toward the grasp side where it would hit the object).
3. **When can two axes pass through each other (L = 0)?** Which placement does it, how bulky is it, and is it only sensible at the finger's base, where the motor can sit inside or under the 37 mm palm plate?
4. **A recommended rule** for the rule set: as simple as possible (Martin strongly prefers simple, explainable rules), cheap to evaluate, enforceable by construction (e.g. a closed form or a small lookup by angle, so the generator can draw lengths from the allowed set). State exactly which assembly conventions it assumes, and what it gives up compared with the exact geometric answer.
5. **The other limits the same motor sets**, briefly: the shortest fingertip link after the last servo; the closest two finger bases can be when both first servos sit in the palm; whether the servos fit the 37 mm palm plate; palm joints; coupled joints; the link cross-section a printed shell around the servo needs (Martin's 2 mm structure).
6. **Verification:** a self-contained numpy script that checks your numbers (oriented-box separation, e.g. the separating-axis test, swept over each joint's range at a fine step; or an equivalent exact method), and prints the tables in your report.

## Output

- `REPORT.md` in this directory: the set-up, the results tables, the recommended rule with its assumptions, the other limits, and open questions for Martin. Plain language; give numbers; state limitations.
- Your scripts, runnable with `python3 <script>` from this directory.
- Your final message: a short summary of the recommended rule and the key numbers.

## Repository files (optional reading)

- `hand_sampler/grammar/hand.py` (the genotype, constants and rule sets)
- `hand_sampler/grammar/derive.py` (frames, axis directions, forward kinematics)
- `project-notes/grammar/GRAMMAR-LOCK-2026-10-07.md` (every grammar decision with its reason)
- `hand_sampler/grammar_bench/conformed_hands.json` (the commercial hands in grammar form)
