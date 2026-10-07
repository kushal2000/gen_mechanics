# Palm and joint-axis study (2026-10-07)

The locked grammar has a flat palm plate whose outline is the convex hull of the finger bases plus a disc at the palm centre, finger mounts given as (y, z) on the plate plus a facing and a tilt, and joints of three kinds relative to their link (flexion, abduction, roll) with a fine tilt. This note takes three limits from the commercial hands, with a margin, and none from a motor:

1. the palm plate's thickness;
2. how far finger bases sit from the palm centre, and how far they sit out of the plate;
3. how far real joint axes sit from the nearest kind.

Hands: the same set as `link-cross-section-study.md` (Allegro, LEAP, Barrett, Ability, Inspire, DClaw, Wuji v1, XHand, Tesollo DG-5F, Orca, SHARPA, Shadow, SVH, Dex3, Wuji v2), plus MIDAS and ARMS where kinematics suffice. Dex1 (two opposed prismatic jaws) has no palm plane and no revolute joints, so it drops out of all three.

## Proposed numbers

| quantity | proposal | from |
|---|---|---|
| palm plate thickness | **37 mm** | median of 15 hands; IQR 25-42, range 6-86 |
| finger base distance from the palm centre, in the plate | **0-125 mm** | 0 x 0.9 and 110.8 x 1.1 (SVH's thumb), rounded out to 5 mm |
| the same, non-thumb fingers only | 0-70 mm | max 61.9 (DClaw) x 1.1 |
| the same, thumbs only | 40-125 mm | 48.0 (Tesollo) x 0.9 and 110.8 x 1.1 |
| out-of-plane base offset (not in the grammar) | error median 0.3 mm, 90th percentile 1.7 mm, max 7.9 mm | Tesollo is the only hand above 2.6 mm |
| joint axis tilt | fine +/-15 deg for fingers; thumbs and palm joints also need a coarse 30 deg tilt step up to +/-45 deg | see item 3 |

Without SVH, whose thumb base sits deep at the wrist (111 mm from the centre against 48-77 mm for every other hand), the largest base distance is SHARPA's thumb at 77.4 mm, and the limit would be 90 mm (77.4 x 1.1 = 85.1, rounded out).

## 1. Palm plate thickness

The palm is every body that is not part of a finger (root, fixed attachments, palm-joint sections), at the zero pose, visual meshes. Rays along the palm normal n (item 2) on a 1 mm grid, inside the convex hull of the finger bases shrunk by 5 mm so that knuckle edges do not dominate. Thickness is the distance between the first hit from each side. Millimetres:

| hand | median | p10-p90 | min-max | angle between n and the mesh's thin axis (deg) | median along the thin axis |
|---|---|---|---|---|---|
| Allegro | 44 | 3-44 | 3-44 | 22 | 41 |
| LEAP | 38 | 32-46 | 1-46 | 3 | 38 |
| Barrett | 86 | 52-95 | 24-96 | (near-cubic housing) | |
| Ability | 30 | 25-35 | 10-37 | 9 | 30 |
| Inspire | 37 | 20-38 | 3-40 | 18 | 35 |
| DClaw | 6 | 6-6 | 6-35 | 0 | 6 |
| Wuji v1 | 17 | 8-21 | 0-23 | 13 | 16 |
| XHand | 43 | 34-45 | 7-45 | 3 | 42 |
| Tesollo DG-5F | 61 | 27-64 | 6-67 | 17 | 59 |
| Orca | 27 | 25-29 | 10-30 | 16 | 25 |
| SHARPA | 40 | 7-43 | 5-44 | 7 | 39 |
| Shadow | 22 | 15-22 | 1-22 | 11 | 22 |
| Dex3 | 40 | 37-41 | 2-41 | 3 | 40 |
| Wuji v2 | 22 | 18-24 | 1-27 | 13 | 21 |
| SVH | 37 | 7-45 | 0-73 | 24 | 37 |

Across the 15 hands: median 37 mm, IQR 25-42 mm, range 6 mm (DClaw, a mounting plate) to 86 mm (Barrett, a motor housing). Measured along the palm mesh's own thin axis (the short side of its minimum-volume box) instead of n, no hand changes by more than 3 mm, and the median becomes 35-37 mm depending on how Barrett is counted. The proposal is 37 mm. These are envelope thicknesses: they include whatever the palm houses (motors, electronics).

## 2. Finger bases around the palm centre

Palm frame per hand: plane = least-squares plane through the first joint of every finger, thumb included; n = its normal on the side the fingers close toward; centre = centroid of the non-thumb bases, projected into the plane. The closing side is the side the non-thumb fingertips move toward when each joint turns toward its larger limit.

Each base: in-plane distance from the centre in mm, with its out-of-plane offset in brackets.

| hand | fingers | thumb | max abs offset | extent |
|---|---|---|---|---|
| Allegro | 1 (-0.5), 44 (+0.3), 44 (+0.3) | 52 (0.0) | 0.5 | 87 |
| LEAP | 0 (0.0), 45 (0.0), 45 (0.0) | 66 (0.0) | 0.0 | 91 |
| Barrett | 25 (0.0), 25 (0.0) | 60 (0.0) | 0.0 | 65 |
| Ability | 10 (-0.4), 10 (-1.5), 30 (+0.8), 30 (+1.3) | 70 (-0.2) | 1.5 | 81 |
| Inspire | 10 (+0.1), 10 (+0.1), 29 (-0.1), 29 (-0.1) | 72 (0.0) | 0.1 | 83 |
| DClaw | 59 (0.0), 60 (0.0), 62 (0.0) | none | 0.0 | 106 |
| Wuji v1 | 11 (-1.1), 11 (-2.6), 33 (+2.3), 33 (+1.7) | 63 (-0.3) | 2.6 | 72 |
| XHand | 11 (+0.1), 12 (+0.3), 31 (-0.2), 34 (-0.3) | 76 (+0.1) | 0.3 | 93 |
| Tesollo DG-5F | 17 (-0.2), 19 (-7.5), 32 (+6.6), 41 (+7.9) | 48 (-6.8) | 7.9 | 69 |
| Orca | 12 (+0.1), 14 (+0.7), 34 (-0.7), 35 (-0.3) | 54 (+0.2) | 0.7 | 75 |
| SHARPA | 10 (-0.1), 12 (-0.5), 31 (+0.4), 32 (+0.3) | 77 (0.0) | 0.5 | 87 |
| Shadow | 11 (+0.3), 12 (+0.4), 33 (-0.4), 34 (-0.3) | 74 (+0.1) | 0.4 | 89 |
| SVH | 12 (+0.8), 12 (+0.4), 35 (-0.6), 36 (-0.7) | 111 (+0.1) | 0.8 | 115 |
| Dex3 | 28 (0.0), 28 (0.0) | 52 (0.0) | 0.0 | 59 |
| Wuji v2 | 11 (-1.0), 13 (-1.7), 31 (+1.4), 31 (+1.8) | 57 (-0.5) | 1.8 | 68 |
| MIDAS | 3 (+0.2), 31 (-0.1), 31 (-0.1) | 65 (0.0) | 0.2 | 80 |
| ARMS (human) | 8 (+0.4), 9 (0.0), 24 (-0.2), 25 (-0.3) | 64 (0.0) | 0.4 | 69 |

"Extent" is the largest base-to-base distance in the plate: 59-115 mm, median 81 mm.

- **In-plane distance**, 76 bases on 17 hands: min 0.0 mm (LEAP's middle finger, by symmetry), median 30.7, max 110.8 (SVH's thumb). Non-thumb bases: 0-61.9 mm (median 28.5; DClaw's three fingers sit 60 mm out). Thumbs: 48.0-110.8 mm (median 64.5).
- **Out-of-plane offsets**, the error of a grammar without them: median 0.3 mm, 90th percentile 1.7 mm, max 7.9 mm. Tesollo is the outlier: its five bases are staggered by +/-7-8 mm. Next are Wuji v1 (2.6), Wuji v2 (1.8) and Ability (1.5), and every other hand is under 1 mm. Hands with three bases (Barrett, DClaw, Dex3) are exact by construction, and so is LEAP, whose three fingers sit in a line, so they say nothing about this error.
- **Centre.** The non-thumb centroid lies on the knuckle row of every hand with fingers in a row, so a disc centred there sits at the top edge of the real palm. This does not change the limits above (they bound positions, not the disc), but the disc's placement in the outline may want a separate look.

## 3. Joint axes against the three kinds

Frame per joint, at the zero pose: x = from the joint to the next joint at least 5 mm away (or to the fingertip), y = the closing direction n made perpendicular to x, and z = x cross y. A joint's angle is the angle between its axis and the nearest of +/-x (roll), +/-y (abduction) and +/-z (flexion). With three orthogonal kinds the angle can never exceed 54.7 deg, so "within 60 deg" is always 100%.

Two variants:

- **A, as imported.** Raw URDF joint origins and the base-plane normal. This is what the projection gives today.
- **B, as an importer could express the same kinematics.** A joint's origin can slide along its own axis without changing the hand's motion, and the URDFs place origins arbitrarily: LEAP's flexion origins sit 14 mm beside the finger, so its links zigzag. In B, each origin is slid along its axis onto one straight centre line per finger, and the base-plane normal is made perpendicular to the fingers' distal axes (Tesollo's staggered bases tilt its plane 16 deg about the finger direction).

Share of joints within each angle of the nearest kind:

| joints | n | <= 5 | <= 15 | <= 30 | <= 45 | median | max |
|---|---|---|---|---|---|---|---|
| A: fingers (no thumb) | 203 | 67% | 86% | 96% | 100% | 2.9 | 38.6 |
| A: thumbs | 60 | 27% | 42% | 78% | 98% | 19.8 | 53.6 |
| A: all | 263 | 58% | 76% | 92% | 100% | 3.3 | 53.6 |
| B: fingers (no thumb) | 203 | 76% | 96% | 99% | 100% | 2.0 | 36.3 |
| B: thumbs | 60 | 30% | 45% | 85% | 100% | 18.6 | 40.3 |
| B: all | 263 | 65% | 84% | 95% | 100% | 2.7 | 40.3 |
| palm joints (A and B) | 5 | 0% | 40% | 40% | 80% | 35.0 | 45.2 |

Angles in degrees. Weighting hands equally gives the same picture (B, all joints: 66 / 85 / 96 / 100%).

Per hand, with the kind mix by count (variant A, and B gives the same overall mix):

| hand | joints | flexion | abduction | roll | median A | worst A | median B | worst B |
|---|---|---|---|---|---|---|---|---|
| Allegro | 16 | 10 | 2 | 4 | 2.0 | joint_12 (thumb #1, 36) | 2.0 | joint_12 (thumb #1, 27) |
| LEAP | 16 | 10 | 5 | 1 | 21.6 | 12 (thumb #1, 42) | 2.9 | 12 (thumb #1, 21) |
| Barrett | 8 | 6 | 2 | 0 | 0.7 | finger_1_prox_joint (finger #1, 34) | 0.3 | finger_2_prox_joint (finger #1, 36) |
| Ability | 10 | 9 | 1 | 0 | 4.0 | thumb_q1 (thumb #1, 31) | 5.8 | thumb_q2 (thumb #2, 29) |
| Inspire | 12 | 8 | 4 | 0 | 6.6 | thumb_proximal_yaw_joint (thumb #1, 54) | 5.1 | thumb_proximal_yaw_joint (thumb #1, 29) |
| DClaw | 9 | 6 | 3 | 0 | 2.6 | joint_f1_0 (finger #1, 4) | 2.7 | joint_f1_1 (finger #2, 4) |
| Wuji v1 | 20 | 15 | 5 | 0 | 7.2 | right_finger1_joint1 (thumb #1, 28) | 7.3 | right_finger1_joint4 (thumb #4, 27) |
| XHand | 12 | 9 | 3 | 0 | 0.4 | right_hand_thumb_rota_joint2 (thumb #3, 18) | 0.0 | right_hand_thumb_rota_joint2 (thumb #3, 18) |
| Tesollo DG-5F | 20 | 12 | 6 | 2 | 16.5 | rj_dg_2_1 (finger #1, 39) | 1.0 | rj_dg_5_1 (finger #1, 36) |
| Orca | 16 | 11 | 5 | 0 | 4.3 | right_thumb_abd (thumb #2, 20) | 6.2 | right_thumb_mcp (thumb #1, 23) |
| SHARPA | 21 | 14 | 7 | 0 | 1.0 | left_1_thumb_CMC_FE (thumb #1, 40) | 0.4 | left_1_thumb_CMC_FE (thumb #1, 40) |
| Shadow | 21 | 14 | 6 | 1 | 1.0 | rh_THJ1 (thumb #5, 6) | 0.0 | rh_THJ4 (thumb #2, 6) |
| SVH | 19 | 12 | 6 | 1 | 3.3 | right_hand_j4 (thumb #4, 26) | 0.0 | right_hand_j4 (thumb #4, 24) |
| Dex3 | 7 | 6 | 0 | 1 | 0.0 | left_hand_thumb_0_joint (thumb #1, 7) | 0.0 | left_hand_thumb_0_joint (thumb #1, 7) |
| Wuji v2 | 20 | 13 | 7 | 0 | 2.9 | l_thumb_ip (thumb #4, 37) | 3.4 | l_thumb_mcp (thumb #3, 37) |
| MIDAS | 16 | 12 | 4 | 0 | 7.7 | thumb_cmc_roll_joint (thumb #1, 30) | 3.0 | thumb_cmc_roll_joint (thumb #1, 32) |
| ARMS (human) | 20 | 14 | 6 | 0 | 7.9 | CMC1b (thumb #2, 44) | 8.2 | CMC1b (thumb #2, 39) |
| **all** | **263** | **181 (69%)** | **72 (27%)** | **10 (4%)** | | | | |

Hand-weighted mix: flexion 70%, abduction 26%, roll 4%, close to the 70 / 25 / 5 the lock assumed for item 16. Thumbs alone are 25 flexion, 29 abduction, 6 roll.

Palm joints: SHARPA's pinky CMC (flexion, 15 deg) and SVH's j5 (roll, 15 deg) are near a kind, while Shadow's LFJ5 (roll, 35 deg) and ARMS's CMC4 and CMC5 (roll, 45 and 37 deg) are not.

What this decides:
- **Fingers.** A fine tilt of +/-15 deg covers 96% of non-thumb finger joints, provided the importer slides joint origins along their axes (variant B). As imported, it covers 86%. The remaining 4% are Barrett's spread joints (36 deg: the spread axis is the palm normal, but the link it moves rises out of the palm), Tesollo's finger bases (23-36 deg) and two ARMS joints (17 deg).
- **Thumbs.** +/-15 deg covers only 45% of thumb joints, and 30 deg covers 85%. Every thumb joint in B is within 45 deg (max 40 deg, SHARPA's CMC), while in A one is beyond (Inspire's thumb yaw, 54 deg). The thumbs need the coarse tilt step: 30 deg steps up to +/-45 deg, with the fine +/-15 deg on top, reach every thumb joint in B.
- **Palm joints** need the same coarse step: 3 of 5 sit 35-45 deg from the nearest kind.
- **Where thumb tilts come from.** Most of the thumb deviation is one rotation of the whole thumb about its own direction (opposition). Allowing a single twist per finger about its own direction lifts the thumb joints within 15 deg from 45% to 72% (fingers: 96% to 97%). The grammar has no such twist (the mount has a facing and a tilt), so the coarse axis tilt is the locked grammar's way to express it.

## Method notes

- **Loading** is the same as in the cross-section study. Ours come through the viewer's loader with the manifest's annotations. Kushal's Dex3 and Wuji v2 were extracted from his branch into the scratchpad, and MIDAS uses the URDF on Vatsal's branch, for kinematics only. SVH uses its download's own URDF, and the duplicate Shadow export is left out.
- **Finger chains** come from the projection's `name_map`. MIDAS's three four-bar closing joints (`*_dip_linkage_joint`) are not in any chain and are left out. Palm joints are the movable joints whose child is a palm body: SHARPA `left_5_pinky_CMC`, Shadow `rh_LFJ5`, SVH `right_hand_j5`, ARMS `CMC4`/`CMC5`.
- **Fingertip point**: the centroid of the far cap of the last link's mesh (with its fixed attachments), i.e. vertices within 10% of the largest distance from the last joint. MIDAS and ARMS have no meshes, so their last link continues the previous one.
- **Closing direction**: the base-plane normal, oriented by the limit vote above. For links within 30 deg of that normal, the finger is taken to close toward the palm centre instead: DClaw's fingers point out of their plate and close toward each other, and the same applies to 4 Barrett, 3 Dex3 and 1 Tesollo links.
- **Variant B's centre line** runs along the raw base-to-fingertip direction, through the point that best fits (in the least-squares sense) every joint axis line and the fingertip. Axes within 30 deg of the line (roll-like) are not slid. The normal's tilt is removed using the mean axis of the non-thumb fingers' last joints, which barely moves DClaw (0.9 deg), whose last axes point three ways.
- **Thumbs** are identified as in the cross-section study. Barrett's opposing finger 3 counts as its thumb.

## Caveats

- **The base plane can be tilted against the palm.** It is fitted through the finger bases, thumb included, so it can be tilted relative to the palm itself: by 22 deg for Allegro and 24 deg for SVH against their palm meshes, and by 34 deg for Barrett, whose spread joints sit inside the palm while finger 3's first joint is on top. This changes thicknesses by at most 3 mm (Barrett aside). For the axis study, a tilt about the flexion axis is harmless, and variant B removes the tilt about the finger direction.
- **Thickness is an envelope.** It is the outer palm block, actuators included, not a structural plate. DClaw's 6 mm is a mounting plate, and Barrett's 86 mm is its motor housing.
- **Out-of-plane offsets are measured from the best-fit plane**, so they are the smallest error any flat plate could give. A plate placed some other way can only do worse.
- **Kind angles depend on zero-pose link directions.** Fingers that are curled at the zero pose (Barrett's distal links, most thumbs) put their x off the straight line, so thumb numbers carry more noise than finger numbers.
- **Scripts** (`palm_axis.py` and `palm_axis_report.py`, with `hands.py` and `measure.py` from the cross-section study) are in the session scratchpad and are not committed. They run in under 10 s on CPU.
