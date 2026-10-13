import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from hub_model import *
from hub_pairs import pair_gaps, ok

def reach(a):
    c = abs(unit(a) @ X); s = np.sqrt(max(0.0, 1 - c * c))
    return HUB_R * s + HUB_H * c

def direction(az, el):
    az, el = np.radians(az), np.radians(el)
    return np.array([np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)])

# 1) hubs only (servos on the finger line), any body direction: shortest L vs the closed form
print("1) Shortest link with both hubs on the finger line, best body direction, vs reach(a1)+reach(a2)")
rng = np.random.default_rng(0)
grid = [(az, el) for az in range(0, 180, 30) for el in (-60, -30, 0, 30, 60)] + [(0, 90)]
worst = 0.0
rows = []
for trial in range(14):
    i, j = rng.integers(len(grid), size=2)
    a1, a2 = direction(*grid[i]), direction(*grid[j])
    kind = ["bend", "spread", "twist"][int(np.argmax([abs(a1 @ Z), abs(a1 @ Y), abs(a1 @ X)]))]
    r1 = {"bend": (-30, 90), "spread": (-30, 30), "twist": (-90, 90)}[kind]
    pred = reach(a1) + reach(a2)
    best = None
    for vdir in [-X, -Y, Z, -Z, Y, X]:
        if abs(unit(vdir) @ unit(a2)) > 0.95 or abs(unit(-X) @ unit(a1)) > 0.95 and False:
            continue
        v1 = -X if abs(unit(a1) @ X) < 0.95 else -Y
        try:
            for L in np.arange(np.floor(pred) - 1, pred + 12, 0.5):
                if ok(pair_gaps(a1, a2, L, v1, vdir, rng1=r1, step=1.0)):
                    best = L if best is None else min(best, L); break
        except ValueError:
            pass
    rows.append((grid[i], grid[j], kind, round(pred, 1), best))
    print(f"  {str(grid[i]):10s} -> {str(grid[j]):10s} ({kind:6s}) closed form {pred:5.1f}  sweep {best}", flush=True)

# 2) crossing axes (L = 0): slide of hub 1 along its axis needed to clear hub 2, vs angle
print("2) Two axes crossing at the joint point: how far one servo must move along its axis")
for th in (90, 75, 60, 45, 30, 15):
    a1 = Z; a2 = np.array([0.0, np.sin(np.radians(th)), np.cos(np.radians(th))])
    need = None
    for s in np.arange(0, 60, 0.5):
        if gap(Hub(np.zeros(3), a1, s), Hub(np.zeros(3), a2, 0.0), 0.75) >= 0:
            need = s; break
    print(f"  angle {th:2d} deg: slide {need} mm (closed form along a1: {HUB_H + HUB_H*abs(np.cos(np.radians(th))) + HUB_R*np.sin(np.radians(th)):.1f})")
