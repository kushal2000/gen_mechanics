# Link cross-section study: one rounded box for every link (2026-10-07)

GRAMMAR-LOCK item 6 makes every link of every hand a rounded box with one shared cross-section, taken from the commercial hands. This note measures that cross-section on 16 commercial hands and checks two things that depend on it: whether it can house the XM335-T323-T motor, and which finger-spacing rule the commercial hands satisfy (open item 10).

**Result: 19 mm wide x 18 mm high, corner radius 6 mm.** Width is along the link's flexion axis, height along the closing direction.

## Why rounded boxes

Simulator probe (Isaac Sim 5.1, RTX 4090, 4096 envs, mean ms per physics step):

| collider | ms / step | behaviour at edges |
|---|---|---|
| capsule | 5.1-5.3 | reference |
| sharp box | 6.3-6.5 | sharp edges |
| rounded box: 8-vertex convex hull of the core box, PhysX `restOffset = r` | 5.6 | exact rounded-edge normals, no gap |
| plain Cube collider with `restOffset = r` | 5.6 | broken: our box-shaped object sank 4-5 mm into the rounded shell with no force, then was pushed along the box's face normals |

The hull version costs about 6-10% more than a capsule and less than a sharp box (+19-27%).

Reasons to prefer it over capsules:
1. One parameterisation covers capsule-like fingers (Wuji, r = w/2), boxy ones (LEAP, Allegro, r near 0) and rounded ones (SHARPA).
2. Around the same motor it is slimmer than a capsule, which must circumscribe the motor's corners.
3. The policy's per-link token features already describe links as boxes.

Risks:
- The grammar's overlap and reach checks and the viewer must switch from capsule distance to rounded-box distance, so they agree with the simulator.
- The behaviour depends on PhysX's GPU convex-convex path, so add a regression test on edge contact normals.
- Keep the core box at least about 1 mm thick in every direction (w, h > 2r + 1 mm), or hull cooking falls back to the CPU. The chosen section has a 7 x 6 mm core.

Geometrically, the shape family matters less than the size (the same finding as E14, `cross-section-study.md`). Scored against every measured slice, the chosen rounded box has a mean IoU of 0.632, a 19 mm circle (a capsule's section) 0.628, and a sharp 19 x 18 mm box 0.618. The case for rounded boxes rests on the probe and on reasons 1-3, not on fit.

## The average cross-section

| statistic (hand-weighted) | w (mm) | h (mm) | r (mm) |
|---|---|---|---|
| median | 19.4 | 18.4 | 5.8 |
| IQR | 15.4-25.0 | 14.4-21.6 | 3.8-7.2 |
| best mean IoU over all slices | 19.5 | 18.25 | 7.5 |
| **chosen (medians rounded to 1 mm)** | **19** | **18** | **6** |

208 links on 16 hands, 564 slices. Each hand counts equally: a link's weight is 1 / (its hand's link count).

How it was chosen:
- The hand-weighted medians and the single rounded rectangle with the best mean IoU against all slices agree to 0.5 mm in w and h.
- The IoU optimum is flat. The 15 best shapes on the 1 mm grid all lie within w 18-20, h 17-19 and r 5-9 and score 0.630-0.632. The chosen 19 x 18, r 6 scores 0.632, tied with the optimum (19.5 x 18.25, r 7.5).
- The IoU barely constrains r, so r comes from the area-based median (5.8), rounded to 6.
- A mean IoU of 0.63 reflects the spread between hands, not the shape family. A rounded rectangle fitted to each slice on its own reaches a median IoU of 0.965.

Sensitivity:

| variant | w | h | r |
|---|---|---|---|
| keeping the 58 slices flagged as joint housings | 19.6 | 18.6 | 6.0 |
| without Dex1 (parallel-jaw gripper) | 18.7 | 18.6 | 5.9 |
| without the grippers and three-finger hands (Barrett, DClaw, Dex1, Dex3) | 17.0 | 18.5 | 6.1 |
| median of per-hand medians | 17.5 | 17.9 | 5.8 |

## Per hand

Median over each hand's links. IoU is the mean IoU of the chosen 19 x 18, r 6 section against that hand's slices. Spacing is the minimum centre-to-centre distance between neighbouring non-thumb finger bases (see below).

| hand | source | links | w | h | r | IoU | min spacing | thumb to nearest finger |
|---|---|---|---|---|---|---|---|---|
| Allegro | ours | 16 | 27.0 | 19.6 | 0.0 | 0.60 | 43.6 | 53.7 |
| LEAP | ours | 16 | 27.9 | 23.5 | 7.4 | 0.48 | 45.4 | 65.8 |
| Barrett | ours | 8 | 23.0 | 23.3 | 4.2 | 0.65 | 50.0 | 65.4 |
| Ability | ours | 10 | 14.9 | 20.5 | 7.0 | 0.63 | 20.1 | 65.1 |
| Inspire | ours | 12 | 13.4 | 17.9 | 6.1 | 0.52 | 19.2 | 70.6 |
| DClaw | ours | 9 | 34.0 | 28.0 | 0.9 | 0.47 | 101.7 | no thumb |
| Wuji v1 | ours | 16 | 15.0 | 15.0 | 7.5 | 0.55 | 22.0 | 65.1 |
| XHand | ours | 12 | 16.7 | 22.7 | 6.6 | 0.69 | 20.2 | 59.6 |
| Tesollo DG-5F | ours | 19 | 22.4 | 19.0 | 1.9 | 0.68 | 24.8 | 47.0 |
| Orca | ours | 15 | 18.0 | 14.5 | 3.8 | 0.77 | 23.4 | 41.7 |
| SHARPA | ours | 15 | 15.5 | 17.9 | 6.0 | 0.70 | 20.5 | 75.2 |
| Shadow | ours (.dae by file name, checked) | 15 | 17.0 | 15.1 | 5.3 | 0.80 | 22.4 | 66.6 |
| SVH | download's own URDF | 16 | 15.7 | 16.2 | 6.9 | 0.68 | 23.3 | 113.3 |
| Dex1 | ours (manifest-excluded) | 2 | 30.3 | 11.1 | 5.6 | 0.52 | jaws opposed | no thumb |
| Dex3 | Kushal | 7 | 23.3 | 15.9 | 0.0 | 0.72 | 57.0 | 59.5 |
| Wuji v2 | Kushal | 20 | 15.0 | 15.0 | 7.5 | 0.65 | 20.9 | 55.0 |
| MIDAS | Vatsal | no meshes | | | | | 31.0 | 58.4 |
| ARMS (human skeleton) | ours | no meshes | | | | | 15.8 | 64.2 |

The hands split into two groups. Allegro, LEAP, DClaw, Barrett, Tesollo and Dex3 have links 22-34 mm wide, mostly with small radii (0-7.4 mm, four of six under 5 mm). Ability, Inspire, Wuji v1/v2, SHARPA, Shadow, SVH, XHand and Orca have links 13-18 mm wide with r of 4-7.5 mm. The average sits between the two groups and fits the slim group better (mean IoU 0.67, range 0.52-0.80) than the large one (0.60, range 0.47-0.72).

## Proximal, middle, distal

| role | hands | links | w | h | r |
|---|---|---|---|---|---|
| base (before the first flexion joint: knuckle and thumb-rotation housings) | 13 | 30 | 19.4 (17.2-26.9) | 20.1 (18.5-28.5) | 5.8 (1.3-7.6) |
| proximal | 15 | 74 | 19.6 (15.7-25.8) | 19.6 (17.5-23.5) | 6.7 (1.1-7.5) |
| middle | 10 | 35 | 17.0 (15.0-22.4) | 17.9 (15.0-19.3) | 4.9 (1.1-6.6) |
| distal (fingertip link) | 16 | 69 | 18.1 (14.9-23.3) | 14.8 (13.2-19.2) | 6.0 (5.6-7.0) |

Hand-weighted median (IQR), mm. Widths agree within 2.6 mm across roles. The one systematic difference is height: distal links are 4.8 mm thinner than proximal ones (14.8 against 19.6 mm), and the distal median lies below the proximal IQR. The two IQRs still overlap (13.2-19.2 and 17.5-23.5), and the shared 18 mm height sits inside both. One cross-section for all roles costs fingertips about 3 mm of extra thickness and proximal links about 2 mm. If a second section is ever wanted, a thinner fingertip (h about 15 mm) is the first candidate.

## The XM335-T323-T motor (19 x 22 mm body)

The chosen section cannot house it in either orientation:
- **19 mm along the width:** the height is 4 mm short. The 6 mm corners also cut into the motor's edges, so the smallest r = 6 mm rounded box around it is 22.5 x 25.5 mm with zero wall. That is 3.5 mm wider and 7.5 mm taller than the average.
- **22 mm along the width:** the smallest box is 25.5 x 22.5 mm, 6.5 mm wider and 4.5 mm taller.
- **Sharp corners (r = 0):** the minimum is 19 x 22 mm, still 4 mm taller.
- **Upper quartile:** even 25.0 x 21.6 mm does not fit it in either orientation.
- **Real links:** across the hands, 15% of links can house the motor, weighting each hand equally: Allegro 9/16, DClaw 6/9, LEAP 8/16, Tesollo 4/19, Barrett 2/8, and one thumb link each in XHand, SHARPA and SVH. No link of Ability, Inspire, Wuji v1/v2, Orca, Shadow, Dex1 or Dex3 can.

Wall thickness is not included: a printed housing adds about 1.5-2 mm per side. A section sized for the motor (22.5 x 25.5, r 6) scores a mean IoU of 0.52 against the hands, against 0.63 for the average. Building with the XM335 therefore means links comparable to LEAP's (27.9 x 23.5) or Barrett's (23.0 x 23.3), about 4 mm wider and 8 mm taller than the commercial average before adding walls.

## Finger spacing

Spacing is the minimum centre-to-centre distance between the first joints of neighbouring non-thumb fingers at the zero pose. "Own w" is the median width of the hand's own non-thumb links. The candidate rules use the chosen w = 19 mm.

| hand | fingers (no thumb) | min spacing | own w | spacing - own w | >= w (19) | >= w + 5 (24) | >= 29 | >= 35 |
|---|---|---|---|---|---|---|---|---|
| Allegro | 3 | 43.6 | 27.0 | +16.6 | pass | pass | pass | pass |
| LEAP | 3 | 45.4 | 29.9 | +15.5 | pass | pass | pass | pass |
| Barrett | 2 | 50.0 | 23.0 | +27.0 | pass | pass | pass | pass |
| Ability | 4 | 20.1 | 14.8 | +5.4 | pass | fail (-3.9) | fail (-8.9) | fail (-14.9) |
| Inspire | 4 | 19.2 | 12.4 | +6.8 | pass (+0.2) | fail (-4.8) | fail (-9.8) | fail (-15.8) |
| DClaw | 3 (120 deg apart) | 101.7 | 34.0 | +67.6 | pass | pass | pass | pass |
| Wuji v1 | 4 | 22.0 | 15.0 | +7.0 | pass | fail (-2.0) | fail (-7.0) | fail (-13.0) |
| XHand | 4 | 20.2 | 16.1 | +4.1 | pass | fail (-3.8) | fail (-8.8) | fail (-14.8) |
| Tesollo DG-5F | 4 | 24.8 | 22.4 | +2.4 | pass | pass | fail (-4.2) | fail (-10.2) |
| Orca | 4 | 23.4 | 18.0 | +5.4 | pass | fail (-0.6) | fail (-5.6) | fail (-11.6) |
| SHARPA | 4 | 20.5 | 15.5 | +5.0 | pass | fail (-3.5) | fail (-8.5) | fail (-14.5) |
| Shadow | 4 | 22.4 | 17.0 | +5.4 | pass | fail (-1.6) | fail (-6.6) | fail (-12.6) |
| SVH | 4 | 23.3 | 15.5 | +7.8 | pass | fail (-0.7) | fail (-5.7) | fail (-11.7) |
| Dex3 | 2 | 57.0 | 24.5 | +32.5 | pass | pass | pass | pass |
| Wuji v2 | 4 | 20.9 | 15.0 | +5.9 | pass | fail (-3.1) | fail (-8.1) | fail (-14.1) |
| MIDAS | 3 | 31.0 | no meshes | | pass | pass | pass | fail (-4.0) |
| Dex1 | 2 jaws, opposed | | | | n/a | n/a | n/a | n/a |
| ARMS (human) | 4 | 15.8 | no meshes | | fail (-3.2) | fail (-8.2) | fail (-13.2) | fail (-19.2) |

Of the 16 commercial hands with neighbouring fingers:

| rule | rejects | which |
|---|---|---|
| spacing >= w (19 mm) | 0 | Inspire passes by 0.2 mm, ARMS (human) fails |
| spacing >= w + 1 (20 mm) | 1 | Inspire |
| spacing >= w + 2 (21 mm) | 5 | Ability, Inspire, XHand, SHARPA, Wuji v2 |
| spacing >= w + 5 (24 mm) | 9 | Ability, Inspire, Wuji v1, XHand, Orca, SHARPA, Shadow, SVH, Wuji v2 |
| spacing >= 29 mm | 10 | the above and Tesollo |
| spacing >= 35 mm | 11 | the above and MIDAS |

Martin's worry holds for w + 5: it rejects every dense five-finger hand except Tesollo. The nine rejected hands do keep 4.1-7.8 mm (median 5.4 mm) between their own fingers, but their fingers are 12.4-18.0 mm wide, narrower than the 19 mm average. Once their links take the shared cross-section, that clearance disappears. Of the candidate rules, only spacing >= w admits every commercial hand, and it leaves Inspire's links 0.2 mm apart when parallel. A rule on each hand's own width would fit them all (spacing >= own w + 2 mm admits every hand, with Tesollo the tightest at +2.4), so the conflict comes from sharing one width, not from the hands. Thumbs sit 42-113 mm from the nearest finger base, so no candidate rule rejects a thumb pair.

## Method

- **Hands.**
  - Ours: the grammar-bench manifest, through the viewer's loader (`gviewer/commercial.py`: `load_urdf` + `project_to_derivation` with the manifest's `hand_root`, `palm_joints` and `tip_frames`).
  - Kushal's: Dex3, Wuji v2 and, as a check on Shadow, his Shadow export. All three are `vendor_left` URDFs, extracted with `git archive` from `origin/2026-10-06-controlled_wuji_experiments` into the session scratchpad. Kushal's Shadow was projected with the manifest's Shadow annotations (`palm` root, `LFJ5` as a palm joint).
  - Vatsal's MIDAS: from `origin/2026-10-02_physical_grammar`.
  - Dex1 is `excluded` in the manifest (prismatic jaws), so it was loaded straight from the source tree.
  - Neither branch was checked out or modified.
- **Links.** The finger chains come from the projection's `name_map`, as in the viewer and E14. MIDAS's projection fails on its four-bar closing joints. For MIDAS, the chains follow the first non-linkage child from the palm, which is used for spacing only.
  - Every link between two joints and every fingertip link is measured. Links under 10 mm are co-located joints and are skipped. The palm is skipped.
  - A link's geometry is its own visual mesh plus those of bodies fixed to it (fingertip pads, Orca's `*_jointbody` carriers), posed at the zero pose.
  - Every hand had visual meshes, so no collision fallback was needed.
- **Frame.**
  - Axis: joint to next joint. For the fingertip link, joint to the mesh point farthest from it in the flexion plane.
  - Width: the link's flexion axis projected perpendicular to the axis. A joint counts as flexion if its axis is within 45 deg of the finger's last joint axis. A link on an abduction joint takes the nearest flexion joint along the finger.
  - Height: axis x width. Dex1's jaws take their slide direction as the height (the closing direction).
- **Slices.** Cuts at 30, 50 and 70% of the length, using the convex hull of each cut.
  - w and h are the hull's bounding box, A is its area, and r = sqrt((wh - A)/(4 - pi)), clamped to [0, min(w, h)/2].
  - Why the hull: the target shape is convex, and many visual meshes do not close. Where trimesh did close the loops (545 of 564 slices), the filled outline equals the hull at the median and is 0.84 of it at the lower quartile. Measuring concave cuts (U-brackets, motor plus horn) directly would read the concavity as corner rounding.
  - A slice whose area exceeds 1.35x the link's smallest slice is taken to hit a joint housing and dropped (58 of 622).
- **Corner-radius check.** A direct per-slice fit (the rounded rectangle on a 1 mm grid with the best IoU against that slice) gives the following, with w and h agreeing to 0.4 mm at the median:

  | slices | median r difference (fit - area) | median abs. difference | 90th percentile |
  |---|---|---|---|
  | all 564 | -0.5 mm | 0.56 mm | 2.6 mm |
  | SHARPA, 44 | -0.7 mm | 0.81 mm | 2.0 mm |

- **Aggregation.**
  - A link's value is the median over its kept slices, and a hand's is the median over its links.
  - Overall quantiles weight each link 1 / (its hand's link count), so every hand counts equally.
  - IoU: each hull is centred on its bounding-box centre, kept in the (w, h) frame, and rasterised at 0.4 mm. The mean IoU weights each hand equally. Search grid: 1 mm, then 0.25 mm around the best.
- **Spacing.** The first joint of each finger at the zero pose.
  - Thumbs: the digit named thumb, or, where none is named, Wuji v1's finger 1, Tesollo's finger 1 and Barrett's opposing finger 3.
  - DClaw has no thumb. Dex1's two jaws face each other, so they have no neighbours.

## Caveats

- **Shadow.** The viewer substitutes `.dae` files found by name ("alignment unverified"). On all 15 finger links they give the same slices (difference 0.0 mm) as Kushal's vendor OBJ export placed by its own URDF, so the finger geometry is verified.
- **SVH.** The viewer's substitution finds meshes for only 10 of the 16 finger links: the manifest URDF's `finger_tip` and `f22_f32` files do not exist in the download. The download's own `SVH/urdf/svh_right_hand.urdf` has the same link names and vendor `.dae` paths, and it gives identical slices on the 10 shared links, so it was used for all 16.
- **Skipped for cross-sections.**
  - MIDAS: only the URDF is vendored on Vatsal's branch, and no copy of its meshes exists on this machine.
  - ARMS: a skeleton without meshes.
  - Both are in the spacing table.
- **Dex1** is a parallel-jaw gripper, and its jaws are plates (30 x 11 mm). It is included because it is one of our hands. Without it, the median is 18.7 x 18.6, r 5.9.
- **Visual meshes include cosmetic covers.** In LEAP, DClaw and Allegro the measured links are mostly motor housings and brackets, while in Wuji they are capsule-like covers.
- **Not measured.** Dex5, XHand Lite and Tesollo DG-5F-S are in the source tree but neither in the manifest nor in this study's hand list.
- **Scripts** (`hands.py`, `measure.py`, `aggregate.py`, `motor_spacing.py`) were kept in the session scratchpad and are not committed. Everything above can be rerun from the method description, using the trimesh and shapely in `.venv_viewer`; the run takes under 10 s on CPU.
