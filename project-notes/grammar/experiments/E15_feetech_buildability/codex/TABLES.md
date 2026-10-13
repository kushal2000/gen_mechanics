## On-kind: exact pair optimum versus repeatable PP assembly

| Pair | Free centred CP minimum | Centred proximal PP minimum | Integer rule | Sampled gap at PP minimum |
| --- | --- | --- | --- | --- |
| B -> B | 19.000 | 40.293 | 41 | 2.000002 |
| B -> S | 19.000 | 40.293 | 41 | 2.000002 |
| B -> T | 24.000 | 30.293 | 31 | 2.000002 |
| S -> B | 19.000 | 39.727 | 40 | 2.000000 |
| S -> S | 19.000 | 39.727 | 40 | 2.000000 |
| S -> T | 24.000 | 29.727 | 30 | 2.000000 |
| T -> B | 24.000 | 41.000 | 41 | 2.000000 |
| T -> S | 24.000 | 41.000 | 41 | 2.000000 |
| T -> T | 29.000 | 31.000 | 31 | 2.000000 |

## Oblique examples (degrees; all lengths mm)

| Axes | Elevation | Kinds | Free CP minimum | PP threshold bracket | PP plane bound | Integer rule | Critical q sample |
| --- | --- | --- | --- | --- | --- | --- | --- |
| z -> tilted z | 0 | BB | 19.000 | 40.293--40.293 | 40.293 | 41 | 46.50 |
| z -> tilted z | 15 | BB | 22.429 | 43.211--43.211 | 43.211 | 44 | 46.50 |
| z -> tilted z | 30 | BB | 24.977 | 44.261--44.261 | 44.261 | 45 | 46.50 |
| z -> tilted z | 45 | BB | 26.471 | 43.370--43.370 | 43.370 | 44 | 46.50 |
| z -> tilted z | 60 | BT | 26.807 | 40.600--40.600 | 40.600 | 41 | 46.50 |
| z -> tilted z | 75 | BT | 25.965 | 34.760--34.820 | 36.140 | 37 | 46.50 |
| z -> tilted z | 90 | BT | 24.000 | 30.293--30.293 | 30.293 | 31 | 46.50 |
| parallel tilted z | 0 | BB | 19.000 | 40.293--40.293 | 40.293 | 41 | 46.50 |
| parallel tilted z | 15 | BB | 19.670 | 41.714--41.774 | 46.494 | 47 | 46.50 |
| parallel tilted z | 30 | BB | 21.939 | 46.526--46.593 | 49.663 | 50 | 46.50 |
| parallel tilted z | 45 | BB | 26.870 | 43.841--43.922 | 49.583 | 50 | -6.25 |
| parallel tilted z | 60 | TT | 33.486 | 35.796--35.862 | 46.261 | 47 | -5.00 |
| parallel tilted z | 75 | TT | 30.023 | 32.094--32.153 | 39.923 | 40 | -87.00 |
| parallel tilted z | 90 | TT | 29.000 | 31.000--31.000 | 31.000 | 31 | -90.00 |

## Zero-length transverse pairs, PP, default clocks; upstream slide -S*a

| Axis angle | S | Upstream far extent along -a | Total extent along a | Minimum sampled gap |
| --- | --- | --- | --- | --- |
| 0 | 31.000 | 45.500 | 60.000 | 2.000000 |
| 15 | 33.094 | 47.594 | 64.188 | 2.000000 |
| 30 | 34.057 | 48.557 | 66.115 | 2.000000 |
| 45 | 33.824 | 48.324 | 65.648 | 2.000000 |
| 60 | 32.410 | 46.910 | 62.821 | 2.000000 |
| 75 | 29.912 | 44.412 | 57.824 | 2.000000 |
| 90 | 26.500 | 41.000 | 51.000 | 2.000000 |

## Explicit offset witnesses

| Pair | L | Mount | Upstream slide | Minimum gap |
| --- | --- | --- | --- | --- |
| SB | 15.0 | PP | 0.0 | 0.000000 |
| SB | 15.0 | PP | -26.5 | 2.000000 |
| SB | 0.0 | PP | -26.5 | 2.000000 |
| BS | 0.0 | PP | -26.5 | 2.000000 |
| SB | 0.0 | CP | -24.5 | 0.000000 |
| BB | 0.0 | PP | -31.0 | 2.000000 |
| TT | 0.0 | PP | -31.0 | 2.000000 |

## Zero-length on-kind PP witnesses, default clocks

| Pair | Slide magnitude | Far face from origin along slide | Direction |
| --- | --- | --- | --- |
| BB | 31.000 | 45.500 | sideways (-z) |
| BS | 26.500 | 41.000 | sideways (-z) |
| BT | 26.500 | 41.000 | sideways (-z) |
| SB | 26.500 | 41.000 | dorsal (-y) |
| SS | 31.000 | 45.500 | dorsal (-y) |
| ST | 41.000 | 55.500 | dorsal (-y) |
| TB | 41.000 | 55.500 | proximal (-x) |
| TS | 41.000 | 55.500 | proximal (-x) |
| TT | 31.000 | 45.500 | proximal (-x) |

## 36 mm parallel examples (Feetech boxes)

| Assembly | Minimum gap |
| --- | --- |
| PP, both backs proximal | 0.000000 |
| PP, upstream back proximal / downstream back distal | 12.706888 |
| PP, backs proximal, axial offset difference 31 | 2.000000 |

## LEAP-spacing adjacent witnesses (hypothetical Feetech assembly)

| Pair | L | Minimum sampled gap | Critical q |
| --- | --- | --- | --- |
| B -> S | 38.0 | 4.269342 | -4.75 |
| S -> B | 15.0 | 2.000000 | -30.00 |
| B -> B | 36.0 | 12.706888 | 46.50 |

## Non-adjacent motors in the hypothetical LEAP-spacing assembly

| Pair | Minimum sampled SAT margin | Between-sample motion bound | Certified distance lower bound |
| --- | --- | --- | --- |
| 0/2 | 8.734176 | 5.600281 | 3.133895 |
| 0/3 | 23.987972 | 11.629281 | 12.358691 |
| 1/3 | 19.722432 | 6.429312 | 13.293120 |

## Other dimensions and conservative bounds

| Quantity | mm |
| --- | --- |
| Front corner radius | 13.793114 |
| Rear corner radius (rotation Lipschitz bound) | 26.462237 |
| Largest box radius about the joint origin | 30.174493 |
| All-axis PP plane-rule upper bound | 50.481778 |
| Bend terminal point bound (2 mm gap) | 15.793114 |
| Spread terminal point bound (2 mm gap) | 15.227241 |
| Twist terminal point bound (2 mm gap) | 16.500000 |
| All-axis terminal point bound | 22.012496 |
| Universal centred bare base spacing (same fixed palm) | 60.348985 |
| Universal centred rectangular-shell base spacing | 66.887966 |
| Maximum minimum palm thickness over axis inclinations, rectangular shell | 40.804412 |
| Rounded radius-6 shell, padding per side for >=2 mm wall | 3.171573 |

## On-kind terminal point minima versus the simple terminal plane rule

| Kind | Exact point minimum | Integer point floor | Plane rule | Integer plane floor |
| --- | --- | --- | --- | --- |
| B | 15.793114 | 16 | 15.793114 | 16 |
| S | 13.279056 | 14 | 15.227241 | 16 |
| T | 16.500000 | 17 | 16.500000 | 17 |

## Straight rectangular shells, 2 mm walls, default clocks

| Kind | Extent along x | Extent along y | Extent along z |
| --- | --- | --- | --- |
| B | [-26.5, 11.5] | [-12, 12] | [-16.5, 16.5] |
| S | [-26.5, 11.5] | [-16.5, 16.5] | [-12, 12] |
| T | [-16.5, 16.5] | [-26.5, 11.5] | [-12, 12] |

## Aligned first-servos on one palm, axes centred at the bases

| First kind | Bare body pitch | One shared 2 mm partition | Separate 2 mm shells | Shell thickness at flat tilt |
| --- | --- | --- | --- | --- |
| B | 29 | 31 | 33 | 24 |
| S | 20 | 22 | 24 | 33 |

All checks passed. Step 0.25 deg; continuous gap error bound for sampled PP distances: 0.057732 mm. Analytic rules do not depend on this discretization.
