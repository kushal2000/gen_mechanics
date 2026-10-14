"""Base layouts: sites, longest contiguous row, and how often two neighbouring
flat first motors (outward facings, on-kind first axes, 69/27/4) collide.
Uses Codex's motor_at / sat_generic and E15's geometry (2 mm enclosure).

    python3 project-notes/grammar/experiments/E16_grammar_rebuild/ours/compare_layouts.py
"""
import sys, math
from itertools import combinations, product
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "codex"))
import review_calculations as R
import feetech_geometry as g

W = {'B': .69, 'S': .27, 'T': .04}

def longest_row(points, pitch):
    S = set(points); best = 0
    for (x, z) in points:
        for dx, dz in ((pitch, 0), (0, pitch)):
            if (x - dx, z - dz) in S:
                continue
            n = 1
            while (x + n*dx, z + n*dz) in S:
                n += 1
            best = max(best, n)
    return best

def pair_stats(points, pitch):
    vals_bare, vals_enc, ws = [], [], []
    for p, q in combinations(points, 2):
        if abs(math.dist(p, q) - pitch) > 1e-8:
            continue
        fp = [f for f in range(0, 360, 30) if np.dot(p, [math.cos(math.radians(f)), math.sin(math.radians(f))]) >= -1e-8]
        fq = [f for f in range(0, 360, 30) if np.dot(q, [math.cos(math.radians(f)), math.sin(math.radians(f))]) >= -1e-8]
        for f, ff, k, kk in product(fp, fq, 'BST', 'BST'):
            c, A = R.motor_at(p, f, 0, k); d, B = R.motor_at(q, ff, 0, kk)
            vals_bare.append(float(g.sat_margin(c, A, d, B)))
            vals_enc.append(float(R.sat_generic(c, A, g.H + 1, d, B, g.H + 1)))   # 1 mm each = one 2 mm wall
            ws.append(W[k] * W[kk] / len(fp) / len(fq))
    ws = np.array(ws)
    return (np.average(np.array(vals_bare) < -1e-8, weights=ws), np.average(np.array(vals_enc) < -1e-8, weights=ws))

for pitch, rin, rout in ((35, 35, 90), (40, 40, 90), (40, 55, 90), (35, 55, 90)):
    pts = R.grid(pitch, rin, rout)
    bare, enc = pair_stats(pts, pitch)
    radii = sorted({round(math.hypot(*p), 1) for p in pts})
    print(f"pitch {pitch}, {rin}-{rout} mm: {len(pts)} spots at radii {radii}; longest row {longest_row(pts, pitch)}; "
          f"neighbouring first motors collide: {bare*100:.0f}% (bodies), {enc*100:.0f}% (with a 2 mm wall)")
