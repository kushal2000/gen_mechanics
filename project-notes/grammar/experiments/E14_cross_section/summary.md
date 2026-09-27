# E14: cross-section study (capsule vs. rounded rectangle)

11/11 target hands available; 146 finger links sectioned, 413 sections total (25/50/75% of joint-to-joint length, links >= 10 mm).

## Per-hand availability

| hand | availability | links | sections | reason |
|---|---|---|---|---|
| sharpa_left_on_iiwa14 | available | 15 | 45 | - |
| allegro_right | available | 16 | 44 | - |
| leap_right | available | 16 | 43 | - |
| inspire_right | available | 12 | 36 | - |
| barrett_bh | available | 5 | 11 | - |
| dclaw | available | 9 | 24 | - |
| ability_right | available | 10 | 30 | - |
| xhand_right | available | 12 | 36 | - |
| wuji_right | available | 16 | 39 | - |
| tesollo_dg5f_right | available | 20 | 60 | - |
| orca_right | available | 15 | 45 | - |

## Template search

Best single template: h/w = 1.00, r/h = 0.45, pooled 80th-percentile max boundary error = 6.43 mm (one size per link, 146 links, 413 sections, 11 hands).

| fit | within 2 mm | within 3 mm |
|---|---|---|
| (a) best single template | 0.15 | 0.41 |
| (b) per-link-radius capsule | 0.15 | 0.39 |
| (c) one radius per hand capsule | 0.10 | 0.23 |
| (d) one radius for all hands | 0.05 | 0.17 |

Global single capsule radius (all hands): 9.39 mm.
