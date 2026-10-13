**Feetech HL-3915-C001: when consecutive joints fit**

The recommended default is **41 mm between transverse joints** (bend or spread), with every servo body held by its parent link, every body pointing back along the finger, and no axial offsets. A slightly tighter rule distinguishes spread from bend; the formula below also handles every oblique axis and twist. Its largest required integer length is **51 mm**. These are continuous-motion guarantees for the specified motor boxes.

There is no universal motor-derived minimum that depends only on the angle between the axes. With different attachment choices, centred transverse motors can have a **19 mm** pair spacing. With unrestricted axial offsets, **every axis pair has a box-feasible minimum of 0 mm**, including parallel pairs. Those alternatives have different brackets, footprints, and consequences for the next joint. They should be explicit assembly templates, not automatic exceptions based on joint kind.

All dimensions below are millimetres. Run `python3 verify.py` in this directory to reproduce the tables and checks. [TABLES.md](TABLES.md) contains the full numerical output; [results.json](results.json) contains unrounded results. No network sources or commercial CAD were used. The motor dimensions and commercial spacings are the supplied measurements. Repository reading only confirmed the axis conventions, kind tie-breaking, limits, and coupling semantics.

**What is being checked.** Let `a` and `b` be the signed unit axes of two consecutive joints, with origins `0` and `L*x` in the straight finger. A motor placement needs three choices that are absent from the present genotype:

- Body attached to the parent (`P`) or child (`C`) of its joint.
- An axial offset `s`: the physical output centre is `p + s*a`.
- A clock angle: a unit back direction `v` perpendicular to `a`; `w = a × v`.

The motor box is `p + s*a + u*a + vcoord*v + wcoord*w`, where `u` is in `[-14.5,14.5]`, `vcoord` in `[-9.5,24.5]`, and `wcoord` in `[-10,10]`. Its geometric centre is **7.5 mm behind its axis**, not on the axis. The calculations depend on the stated assumption that the shaft is centred across the 20 mm width, and on the supplied symmetric 29 mm axial envelope already including horn and idler. Neither is added a second time.

For zero-pose boxes `K_a`, `K_b` relative to their joint origins, the poses are

```
motor j:     R_a(qj)^cj K_a
motor j+1:   R_a(qj) [ L*x + R_b(qj+1)^ck K_b ]
```

Here `c=0` for `P`, `c=1` for `C`; each `K` includes its offset. Consequently:

| Bodies | Relative motion that must be checked | Body gap |
| --- | --- | --- |
| P, P | All positions of joint j | 2 |
| C, C | All positions of joint j+1 | 2 |
| P, C | All combinations of both joint positions | 2 |
| C, P | None: both bodies are on the intervening link | 0; touching allowed |

For moving bodies I require their **Euclidean box distance** to be at least 2. For bodies on the same link I require non-overlapping interiors, using the separating-axis test (SAT); distance zero alone cannot distinguish touching from penetration. A positive SAT margin is a lower bound on distance, not generally the exact distance. The script also evaluates exact vertex/face and edge/edge distances.

These are motor-envelope results. A 2 mm empty gap reserves room for one printed wall under Martin's model. It does not specify horn bolt circles, screws, cable exits, bearings, wall stiffness, or the entire U-bracket sweep. Two independently enclosing 2 mm shells need **at least 4 mm body separation** wherever both walls intervene, before any extra moving clearance. Substitute that gap in the formulas if that is the intended construction. The very compact results below are exact for the stated box model, not complete bracket designs.

`L` is the grammar's distance between chosen joint origins along the finger; it is not the shortest distance between the two infinite axis lines. For nonparallel axes that line distance is

```
d_lines = L * abs(x dot (a cross b)) / norm(a cross b).
```

For parallel axes it is `L*sqrt(1-(a dot x)^2)`. A bend axis and a twist axis intersect even when `L > 0`. Intersecting lines do not imply that their motors occupy the intersection.

**Centred pairs: two different minimum questions.** The first numerical column below allows arbitrary clocking and either body attachment. The optimum uses `C,P`: the motors share a link and point away from one another, with their fronts facing. The second column fixes the repeatable `P,P` assembly: no slides, both backs proximal. For a pure twist the default back is `-y` (dorsal); otherwise it is the projection of `-x` perpendicular to the shaft. The moving gap is 2.

| Pair, proximal to distal | Exact free centred pair minimum, C/P | Exact centred proximal P/P minimum | Integer P/P floor |
| --- | --- | --- | --- |
| bend → bend | 19.000 | 40.293 | 41 |
| bend → spread | 19.000 | 40.293 | 41 |
| bend → twist | 24.000 | 30.293 | 31 |
| spread → bend | 19.000 | 39.727 | 40 |
| spread → spread | 19.000 | 39.727 | 40 |
| spread → twist | 24.000 | 29.727 | 30 |
| twist → bend | 24.000 | 41.000 | 41 |
| twist → spread | 24.000 | 41.000 | 41 |
| twist → twist | 29.000 | 31.000 | 31 |

For all these pairs the unrestricted-offset minimum is **0**, as constructed below.

For the first column, two transverse motors with fronts opposed use `9.5+9.5=19`. One twist substitutes its axial half-length 14.5, giving 24; two twists give 29. These are also lower bounds: every possible clocking contains the respective centred line segment along `x`. The `C,P` construction attains them throughout both joint ranges because the bodies do not move relative to one another.

Those minima cannot be assigned independently to every edge of a chain. To make pair `(j,j+1)` share a link requires servo `j+1` to be parent-mounted; to make `(j+1,j+2)` share its link requires the same servo to be child-mounted. One physical motor cannot do both. Compact brackets and opposing clocks also affect neighbouring modules.

For Martin's repeated bend assembly, view the previous motor from the moving intervening link. Its greatest forward reach is

```
max[-30 <= q <= 90] (9.5*cos(q) + 10*abs(sin(q)))
    = sqrt(9.5^2 + 10^2) = 13.793114.
```

The critical rotation is about **46.47°**, not an endpoint of the range. The next motor extends 24.5 back from its own axis, so the exact threshold is **24.5 + 13.793114 + 2 = 40.293114**. At 40 mm the worst gap is only **1.706886 mm**. Martin's approximately 40 mm estimate is well calibrated; round upward to 41 for a strict integer rule.

A spread only reaches ±30°, giving `9.5*cos(30°)+10*sin(30°)=13.227241`, hence 39.727241. The transverse downstream motor's different axial thickness does not change this end-face contact. This is why the order of a bend/spread pair matters slightly even before offsets are introduced.

**Oblique axes: an exact free-pair answer and a cheap repeatable answer.** There is an exact low-dimensional solution for the first column's generalization. Every centred box contains a coaxial cylinder with radius 9.5 and axial half-length 14.5. For a proposed separating normal `n`, the smallest achievable support over the clock angle is

```
h_a(n) = 14.5*abs(a dot n) + 9.5*norm(a cross n).
```

Clock the first motor's back away from `n`, and the second's toward `n`; both attain these opposing supports. With shared-link mounting this yields the global centred **pair** optimum

```
L_free(a,b) = min over ny,nz of [ h_a(n) + h_b(n) ],  n=(1,ny,nz).
```

This is a convex minimization in two variables: sums of absolute values and norms. It is exact for arbitrary axes, including noncoplanar ones. Searching `ny,nz` in `[-29/19,29/19]` suffices: the objective is at least `19*norm(n)`, and its value at `n=x` is at most `sqrt(29^2+19^2)=34.670`. The NumPy solver uses nested scalar convex minimization and returns a witnessing normal and motor clocks. For parallel axes with `t=abs(a dot x)`, the result simplifies to

```
L_free(a,a) = min(29/t, 19/sqrt(1-t*t)),
```

with division by zero interpreted as infinity. It ranges from 19 for transverse shafts to 29 for shafts along the finger, and can reach 34.670 at an intermediate inclination. Thus even **parallel** axes do not have one minimum.

For the repeatable `P,P` assembly, define `ta=abs(a dot x)`, `tb=abs(b dot x)`, and `ra=sqrt(1-ta^2)`, `rb=sqrt(1-tb^2)`. Let `F(a)` be 13.227241 for nearest-kind spread and 13.793114 for bend or twist. The recommended sufficient floor is

```
L_plane(a,b) = 2 + 14.5*(ta+tb) + F(a)*ra + 24.5*rb
L_allowed_min = ceil(L_plane(a,b))
```

This separates the boxes by a plane perpendicular to the current intervening link. In that link's frame, the previous motor never reaches farther forward than `14.5*ta + F(a)*ra`; the next motor never reaches farther backward than `14.5*tb + 24.5*rb`. Their sum plus 2 is therefore sufficient **for the entire continuous range**. The derivation uses the actual asymmetric body, not a sphere. Reversing the sign convention of an axis does not change this bound, because the sweep expression uses `cos(q)` and `abs(sin(q))` over the supplied ranges.

It is exact for all on-kind pairs in the table. For oblique pairs a tilted separating plane or an edge/edge gap can permit a shorter link. Here are measured examples; `tilted z` means `(sin e,0,cos e)`. The P/P intervals enclose the fixed-mount minimum using a 0.25° sweep plus a continuous-motion bound; endpoints are displayed rounded to 0.001 mm.

| Axes | e | Exact free C/P minimum | P/P minimum interval | P/P plane bound | Integer rule |
| --- | --- | --- | --- | --- | --- |
| z → tilted z | 0° | 19.000 | 40.293–40.293 | 40.293 | 41 |
| z → tilted z | 15° | 22.429 | 43.211–43.211 | 43.211 | 44 |
| z → tilted z | 30° | 24.977 | 44.261–44.261 | 44.261 | 45 |
| z → tilted z | 45° | 26.471 | 43.370–43.370 | 43.370 | 44 |
| z → tilted z | 60° | 26.807 | 40.600–40.600 | 40.600 | 41 |
| z → tilted z | 75° | 25.965 | 34.760–34.820 | 36.140 | 37 |
| z → tilted z | 90° | 24.000 | 30.293–30.293 | 30.293 | 31 |
| parallel tilted z | 0° | 19.000 | 40.293–40.293 | 40.293 | 41 |
| parallel tilted z | 15° | 19.670 | 41.715–41.774 | 46.494 | 47 |
| parallel tilted z | 30° | 21.939 | 46.526–46.593 | 49.663 | 50 |
| parallel tilted z | 45° | 26.870 | 43.841–43.922 | 49.583 | 50 |
| parallel tilted z | 60° | 33.486 | 35.796–35.862 | 46.261 | 47 |
| parallel tilted z | 75° | 30.023 | 32.094–32.153 | 39.923 | 40 |
| parallel tilted z | 90° | 29.000 | 31.000–31.000 | 31.000 | 31 |

An angle-only lookup would confuse bend→spread, bend→twist, and twist→bend: all have 90° between axes, but their repeated-mount minima are 40.293, 30.293, and 41.000. It would also miss the entire parallel-inclination series. Exact oblique results additionally depend on clocking, offsets, attachment, joint ranges, and potentially signed rotation ranges. A detailed lookup must fix those choices. The proposed bound needs only the two absolute longitudinal components and the upstream kind.

The maximum of the bound over all axes is

```
2 + hypot(14.5,13.793114) + hypot(14.5,24.5) = 50.481778.
```

Thus **51 mm is a conservative single floor for every axis pair in this assembly**, if simplicity is more important than compactness. Using 41 for all transverse pairs is the simpler version of the on-kind table. Applying 41 indiscriminately to oblique axes is unsafe.

**Sliding, intersecting axes, and bulk.** Without sliding, `L=0` is impossible for every pair: both solid boxes contain an open neighbourhood of the common joint origin, whatever their clock angles or attachments. Perpendicular shafts alone do not create a gimbal.

With sliding there is a simple construction for *any* pair, using `P,P`. Keep the downstream box at its joint origin and slide the upstream body by `-S*a`. Let its downstream neighbour's zero-offset support in direction `-a` be `h_b(-a)`, now using the actual chosen box clock. Choose

```
S = max(0, 14.5 + 2 + h_b(-a) - L*(a dot x))
h_b(n) = 7.5*(v_b dot n)
         + 14.5*abs(b dot n) + 17*abs(v_b dot n) + 10*abs(w_b dot n).
```

The upper face of the shifted upstream body lies 2 mm below the neighbour's lowest projection along `a`. Rotation about `a` preserves that projection, so the construction works throughout the upstream range; the downstream body is unaffected by its own joint angle. Setting `L=0` proves the unrestricted-offset minimum is zero. There is consequently no useful length-only optimum unless axial offsets or physical footprint are bounded.

The following are explicit zero-length placements with the default proximal clocks. They are convenient sufficient offset values, not a claim to minimize all possible footprints.

| Pair | Upstream slide magnitude | Far upstream face from joint origin along the slide | Direction at zero pose |
| --- | --- | --- | --- |
| bend → bend | 31.0 | 45.5 | sideways, −z |
| bend → spread | 26.5 | 41.0 | sideways, −z |
| bend → twist | 26.5 | 41.0 | sideways, −z |
| spread → bend | 26.5 | 41.0 | dorsal, −y |
| spread → spread | 31.0 | 45.5 | dorsal, −y |
| spread → twist | 41.0 | 55.5 | dorsal, −y |
| twist → bend | 41.0 | 55.5 | proximal, −x |
| twist → spread | 41.0 | 55.5 | proximal, −x |
| twist → twist | 31.0 | 45.5 | proximal, −x |

A slide along a bend shaft widens the finger. A negative spread slide puts the motor behind the hand; the positive alternative occupies the grasp side. A pure twist slide stays **on** the centre line and creates a proximal or distal overhang. For an oblique shaft, the output centre's transverse displacement is `abs(s)*sqrt(1-(a dot x)^2)`, resolved into `s*a_y` toward/away from the grasp side and `s*a_z` sideways. These directions are relative to that link at zero pose; preceding joints rotate them during motion.

For two transverse axes with acute angle `theta`, both backs proximal, the same zero-length construction becomes

```
S = 16.5 + 14.5*abs(cos(theta)) + 10*sin(theta).
```

| Axis angle | S | Far upstream face along −a | Total body extent along a |
| --- | --- | --- | --- |
| 0° | 31.000 | 45.500 | 60.000 |
| 15° | 33.094 | 47.594 | 64.188 |
| 30° | 34.057 | 48.557 | 66.115 |
| 45° | 33.824 | 48.324 | 65.648 |
| 60° | 32.410 | 46.910 | 62.821 |
| 75° | 29.912 | 44.412 | 57.824 |
| 90° | 26.500 | 41.000 | 51.000 |

In particular, a zero-length spread→bend module can put the spread body at `s=-26.5*y`: its y interval is `[-41,-12]`, while the bend body occupies `[-10,10]`. The gap stays exactly 2 for every spread position. At zero pose the combined bare envelope is **34 along x × 51 along y × 29 along z**. Over the ±30° spread range, its swept envelope grows to **43.945 × 51 × 49.615**, before brackets: the x and z extrema occur at the range endpoints. With shared-link `C,P` mounting the slide can be 24.5 instead, with touching bodies and a 49 mm y extent. That saves 2 mm but changes the neighbouring attachment requirements.

Changing the downstream clock to put its front toward the displaced upstream motor reduces the moving orthogonal one-sided slide to **26.0**, but turns its 24.5 mm back in the opposite transverse direction. For spread→bend that can put 24.5 mm of body into the grasp side. Minimizing the offset alone need not minimize a useful hand footprint.

For parallel axes, split the required 31 mm differential slide as ±15.5 if symmetric bulk is acceptable: the bare stack is still 60 mm wide but extends only 30 mm each way from the centre line. Coincident parallel joints are kinematically redundant rotations about the same line, not a two-axis knuckle.

These arrangements are possible away from the base, but are usually unattractive there: they occupy neighbouring-finger or object space and add mass and long brackets. A base module has more room behind or under the palm. **The 37 mm palm does not conceal the zero-length example**: the dorsal body reaches −41 from a mid-plane at zero, while the palm stops at −18.5. It protrudes 22.5 mm below the bare plate; one 2 mm outer wall increases that to 24.5. “Inside or under the palm” must explicitly allow the *under* part. Larger separate shells require a larger interbody gap as discussed above.

**What the commercial spacings demonstrate.** LEAP's fitted joint lines do not encode its motor centres, body/link attachment, clocks, clearances, or casing shape. Those cannot be recovered uniquely from the supplied skeleton. Nevertheless, a concrete assembly using even these Feetech boxes demonstrates that the whole 38/15/36 pattern is possible without rejecting the grammar's ranges:

- All four bodies are parent-mounted.
- The first bend, spread, and second bend bodies point proximally. Slide only the spread body by −26.5 along its y shaft.
- Point the final bend body's back distally. The 48 mm terminal link has room beyond it.

| Consecutive pair in this hypothetical assembly | L | Minimum sampled body gap |
| --- | --- | --- |
| bend → spread | 38 | 4.269342 |
| spread → bend | 15 | 2.000000 |
| bend → bend | 36 | 12.706888 |

The middle pair would work at 0 too; at 15 its y separation is unchanged. The last pair puts two fronts toward one another rather than putting the next motor's back toward the previous front. Its moving threshold is only `13.793114+9.5+2 = 25.293114`. This use of the distal link's volume explains one way to get 36, while retaining the same motor size. Shared-link mounts or substantial axial staggering are other possibilities.

This witness was also checked for all non-adjacent body pairs. A 5° grid over the three angles that affect these bodies, plus conservative motion bounds between samples, gives these continuous clearance lower bounds:

| Non-adjacent motor indices | Certified distance lower bound |
| --- | --- |
| 0 / 2 | 3.134 approximately |
| 0 / 3 | 12.359 approximately |
| 1 / 3 | 13.293 approximately |

The precise values are printed by the script. This is a motor-body feasibility witness, **not an identification of LEAP's actual construction**, and it does not include its palm or brackets.

The XC330's 26 rather than 29 mm axial dimension by itself does **not** explain 36: axial thickness does not enter the centred transverse end-to-end formula. With a 34×20 transverse rectangle, 2 mm clearance and the bend range, that convention requires `34-f+sqrt(f^2+10^2)+2 > 36` for any front offset `f` within the body. Changing the offset alone cannot reach 36 under those assumptions. A different mounting convention, axial stacking, clocking, smaller effective casing envelope, or different motion/clearance assumptions is needed.

DClaw's 68 and 69 mm spacings are unsurprising for the larger XM430. Its shaft offset is not provided, so an exact calibration would be invented. Assuming its 46.5 length and 28.5 width are the transverse rectangle, with a width-centred shaft at front offset `f`, the same conservative end-to-end bend expression is

```
46.5-f + sqrt(f^2+14.25^2) + 2 <= 62.75,  for 0 <= f <= 46.5.
```

Both observed spacings exceed that bound. They leave room for structure and other design choices; their larger values are not minimum-size measurements. The 84 mm terminal link is also ample under this model.

**The construction rule for the generator.** Store or deterministically derive a physical assembly, not just joint kinds. For the first version:

1. Use motorized revolute joints; use the `P,P`, zero-slide, proximal-body convention throughout ordinary fingers. For pure twist use the stated −y clock. A different first-servo palm clock must be an explicit base template with its own first-pair clearance.
2. Draw axes on the existing 5° grid. Compute each intervening minimum with `ceil(L_plane)`, then draw its integer length from that floor to 90. An even simpler transverse-only variant uses 41 everywhere. A single all-axis constant can use 51.
3. Compute the terminal minimum below. Allocate lengths jointly so their sum stays at most 250. With five joints, even the conservative bounds need at most `4*51+23=227` minimum total length, so the existing count and total-length limits remain compatible for every axis choice.
4. Draw a total between the sum of the minima and `min(250,90*n)`, then distribute the integer slack subject to each link's capacity `90-minimum`. A sequential bounded composition draws the next slack in `[max(0, remaining_slack-sum(later_capacities)), min(this_capacity, remaining_slack)]`. This makes every draw valid without rejecting a hand.
5. When an axis changes, recompute the two affected pair minima and, for the final joint, the terminal minimum. Raise lengths to their new floors and reclaim excess total length from available slack. Topology mutations recompute the same floors. Never retain an old compact-mount lookup value after changing its attachment, clock, or offset.
6. Disable generic zero links. If compact bases are wanted, add named two-axis base modules containing their attachment choices, clocks, offsets, bracket geometry and occupied volume. Initially allow them only at the base, with no consecutive zero modules. A kind switch by itself must not grant a 0 or 15 mm exception.

The rule deliberately gives up compact shared-link assemblies, offset packing, and some oblique edge/edge clearances. In exchange it needs a few arithmetic operations, has an explainable proof, and composes consistently for **adjacent motor bodies**.

A local length rule cannot make the current arbitrary whole-hand geometry automatically collision-free. Two explicit failures are checked in the script:

- Three co-located parallel joints with body offsets −31, 0, −31: both adjacent pairs have a 2 mm gap for their entire ranges, but motors 0 and 2 occupy exactly the same volume at zero pose.
- Five ordinary bends with four 41 mm links: setting the first four angles to +90° makes the chain close a square. Joint 4 and its body coincide with joint 0 and its body, although every adjacent pair obeys the rule.

Nonzero short modules can also have non-adjacent collisions; the issue is not confined to exact zeros. For a literal “every hand is buildable throughout its independent ranges” guarantee, use certified whole-finger and palm layout templates, or retain whole-hand checks for non-adjacent motors, brackets, palm and other fingers. Pair lengths should still be drawn by construction; global checks address genuinely different constraints. If the intended promise allows controller-enforced self-collision limits, document that separately. The existing 19×18 simulated links cannot stand in for these motor and bracket volumes.

**Other dimensions set by the motor.** For the default parent-mounted proximal final servo, a sufficient terminal **point** length with 2 mm clearance is

```
T_min(a) = 2 + 14.5*abs(a dot x) + F(a)*sqrt(1-(a dot x)^2).
```

This gives **15.793 for bend, 15.227 for spread, and 16.500 for twist**: safe integer floors 16, 16, 17. The all-axis bound is **22.012496**, hence a simple universal terminal floor of 23. This locates the fingertip beyond the swept motor; it does not fill the line from shaft to tip with solid plastic. The link must route around the servo using the U-bracket, and a finite fingertip pad needs its own rear extent added. The grammar's 10 mm terminal minimum is insufficient for this assembly.

For a point tip, the bend and twist figures are exact in this mounting convention. A spread can be shorter: throughout ±30° the tip remains opposite the motor's front face, giving the exact minimum `(9.5+2)/cos(30°) = 13.279056`, or 14 on the integer grid. The plane rule deliberately uses 16 for simplicity. There is no assembly-independent fingertip minimum if the motor may be offset away from the tip or clocked differently.

A distal-pointing motor parent-mounted at the last bend instead needs `hypot(24.5,10)+2 = 28.462` for the same terminal-point convention. If that body and terminal link are on the same child link, the static back-plus-wall length is 26.5. These differing values are why terminal length also needs an assembly convention.

A simple 2 mm rectangular enclosure around the supplied motor is **33 along u × 38 along v × 24 along w**. At zero pose the default shells occupy:

| Kind | x interval | y interval | z interval |
| --- | --- | --- | --- |
| bend | [−26.5,11.5] | [−12,12] | [−16.5,16.5] |
| spread | [−26.5,11.5] | [−16.5,16.5] | [−12,12] |
| twist | [−16.5,16.5] | [−26.5,11.5] | [−12,12] |

Thus a bend needs a **33 mm width × 24 mm height** cross-section, and a spread needs **24 × 33**, before screws or extra clearance. A twist with this clock has a 24×38 cross-section offset dorsally by 7.5; forcing that enclosure to be symmetric about the centre line would need 53 mm height. Oblique motors require their actual oriented envelopes.

The simulation's radius-6 rounded box also needs attention: rounding a nominal 33×24 shell cuts into the available wall near a rectangular motor's corners. To keep at least 2 mm wall with an outer corner radius of 6, equal padding must be at least `6-4/sqrt(2)=3.172` per side. A bend's rounded cross-section then needs approximately **35.35×26.35**, or use a different corner design. These are geometric wall envelopes, not a strength calculation.

For aligned, flat fingers whose first bodies are fixed in the same palm, take local y along the palm normal and the knuckle row along z. Base pitch is then a packing problem:

| First kind | Bare-body pitch, touching | One shared 2 mm partition | Separate 2 mm shells | Shell thickness normal to a flat palm |
| --- | --- | --- | --- | --- |
| bend | 29 | 31 | 33 | 24 |
| spread | 20 | 22 | 24 | 33 |

These pitches apply across an aligned knuckle row with the specified body clocks; moving U-brackets can demand more. The old 19 mm base spacing is too small for either example. Facing, tilt, mixed axes and axial offsets change the footprints, so there is no universal tight base-to-base distance. For centred, unshifted motors on a fixed palm, a deliberately loose orientation-independent bound is **60.349** between bare axis origins, or **66.888** for separate rectangular shells: twice their largest radius about the joint origin. Using 67 mm would solve static shell packing, but wastes substantial space and still says nothing about downstream fingers.

The 37 mm plate can house the flat, centred examples: 24 mm for a bend or 33 for a spread. It does not automatically fit all orientations. For palm normal `n`, the rectangular-shell fit at a mid-plane joint origin is exactly the slab condition

```
abs((s*a + 7.5*v) dot n)
  + 16.5*abs(a dot n) + 19*abs(v dot n) + 12*abs(w dot n)
  <= 18.5.
```

Even with the best clock and no slide, the required half-thickness is `16.5*t+12*sqrt(1-t^2)`, where `t=abs(a dot n)`. Its maximum is 20.402, requiring **40.804 mm**, so some shaft inclinations cannot fit a 37 mm closed rectangular pocket. For the prescribed bend clock, tilting the preceding aligned example by `tau` in its x/y plane gives half-thickness `26.5*abs(sin(tau))+12*abs(cos(tau))`: about 18.450 at 15°, already 23.642 at 30°. The grammar's full tilt range therefore needs exposed base mounts or separate palm templates. A pure twist in that flat layout with its back dorsal reaches 26.5 below the mid-plane; clocking it sideways can improve palm fit but changes the default assembly. For other grasp-directed frames use the general slab formula. Planform room behind the rim, fasteners, and wiring must also be reserved.

Palm joints are possible in principle: one plate holds the body and the moving plate holds the horn/idler bracket. A hinge in a solid 37 mm plate does not supply that design automatically. It needs a split palm, a motor pocket, a bracket, and checks against every carried finger over ±30°. Recommend **no palm joints in the initial rule**, then add a separately certified palm-hinge module. Motor torque and the load of several fingers also need evaluation; 14.2 kgf·cm converts to about **1.39 N·m**, not a continuous-duty load rating established by this study.

The specified **motorless coupled joints cannot remain unrestricted**. A passive hinge does not enforce the exact 1.1 ratio; it needs a designed transmission. Either exclude them initially, add a certified 11:10 coupling mechanism with its own geometry, or put a motor on each joint and impose coupling in control, which changes the stated motorless convention. Prismatic joints likewise require a conversion mechanism; axial placement offsets in this report are not actuated sliding joints.

**Verification and remaining decisions.** [feetech_geometry.py](feetech_geometry.py) implements placements, rotations, all 15 SAT directions, Euclidean box distance, the support rules, the free-centred convex optimization, and swept queries. [verify.py](verify.py) prints the tables and asserts the quoted witnesses and counterexamples. It uses only NumPy; normal invocation is:

```
python3 verify.py
```

The default P/P sweep is 0.25°. In the intervening link frame only the upstream motor rotates, with radial extent `hypot(24.5,10)=26.462237`. Every angle lies within 0.125° of a sample, so sampled distance can overestimate the continuous minimum by at most **0.057732 mm**. The numerical oblique intervals use that bound; the upper endpoint is the better of a padded sampled certificate and the analytical plane rule. Binary-search width is under 0.00002 mm, and endpoints in the report are rounded. The closed-form on-kind values and the plane and slide constructions require no sampling assumption.

For the four-motor LEAP-spacing witness, the non-adjacent certificate sums the relevant downstream-length-plus-body-radius motion bounds for each varying angle. `verify.py` prints both the sampled SAT margins and the subtracted bounds. No sweep over the fourth joint is necessary for motor-body checks because its body is parent-mounted; its bracket and tip still move. The checker accepts both joint-angle arrays for assemblies such as `P,C` where both matter.

The geometry kernel is also cross-checked against an independent convex closest-points optimization on 50 randomly oriented box pairs in `python3 check_distance_scipy.py`; that optional check uses the already available SciPy. Its maximum discrepancy in this run was below 0.000000001 mm. Assertions also check distance symmetry, rigid-transform invariance, known face/edge/vertex gaps, and arbitrary oblique axes on the grammar grid. The optional `python3 verify.py --step 0.1` tightens sampled intervals if needed; analytical guarantees are unchanged.

Open decisions for Martin:

- Is the first version the centred serial assembly, or should it include a specified dorsal two-axis base module? What sideways and below-palm protrusion is acceptable?
- Does “2 mm structure” mean one bracket wall between bodies, or independent 2 mm enclosing shells? The latter changes the gap and therefore every minimum.
- Confirm the horn/idler mounting geometry, cable exits, width-centred shaft assumption, and attainable actuator travel against a drawing or specimen. Those keep-outs are not in the supplied box.
- Should every independent joint combination be collision-free, or will the controller exclude self-colliding combinations? Choose whole-hand templates/checks accordingly.
- Obtain actual commercial motor-body transforms and brackets before attributing a specific offset or assembly convention to LEAP or DClaw. The fitted skeleton alone does not establish them.
- Decide whether palm hinges and motorless 1.1 couplings justify additional certified mechanical modules in the first buildable rule set.
