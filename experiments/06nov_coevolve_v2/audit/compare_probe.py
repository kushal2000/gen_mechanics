"""Compare two scene_dump.py probe files field by field and slot by slot."""
import json, sys
import numpy as np
A, B = (np.load(p, allow_pickle=False) for p in sys.argv[1:3])
la, lb = (a.split(":")[0] for a in sys.argv[1:3])
sl = json.loads(str(A["slices"]))
assert sl == json.loads(str(B["slices"]))
oa, ob = A["obs0"], B["obs0"]
print("zero-pose obs, max |diff| per field:")
for f, (s, e) in sl.items():
    d = np.abs(oa[s:e] - ob[s:e])
    w = int(d.argmax())
    print(f"  {f:22s} [{s}:{e}] max {d.max():.3g} at +{w} ({oa[s+w]:.4g} vs {ob[s+w]:.4g})")
print("mu0 max |diff|", np.abs(A["mu0"] - B["mu0"]).max(), " v0", float(A["v0"]), float(B["v0"]))
print("mu0 A", np.round(A["mu0"], 3).tolist())
print("mu0 B", np.round(B["mu0"], 3).tolist())
print("limits/home equal:", np.allclose(A["joint_lower"], B["joint_lower"], atol=1e-6),
      np.allclose(A["joint_upper"], B["joint_upper"], atol=1e-6), np.abs(A["joint_home"] - B["joint_home"]).max())
print("slot->joint equal:", json.loads(str(A["slot_joint"])) == json.loads(str(B["slot_joint"])))
na, nb = list(A["link_names"]), list(B["link_names"])
common = [l for l in na if l in nb]
ia, ib = [na.index(l) for l in common], [nb.index(l) for l in common]
print(f"links: {len(na)} vs {len(nb)}, common {len(common)}; only A {set(na)-set(nb)} only B {set(nb)-set(na)}")
print("zero-pose link pos (palm frame) max |diff| mm:", 1000 * np.abs(A["links0"][ia] - B["links0"][ib]).max())
print("tips0 max |diff| mm:", 1000 * np.abs(A["tips0"] - B["tips0"]).max())
# per-slot probe: which links moved, and how far, in each scene
LA, LB = A["links_end"][:, ia], B["links_end"][:, ib]
worst = 0.0
for k in range(31):
    da = np.linalg.norm(LA[1 + k] - LA[0], axis=-1)
    db = np.linalg.norm(LB[1 + k] - LB[0], axis=-1)
    diff = np.abs(LA[1 + k] - LB[1 + k]).max() * 1000
    worst = max(worst, diff)
    mv = [common[i] for i in np.nonzero(da > 1e-3)[0]]
    mvb = [common[i] for i in np.nonzero(db > 1e-3)[0]]
    if mv or mvb or diff > 0.5:
        print(f"  slot {k:2d}: moved A {mv[:1]}..({len(mv)}) B {mvb[:1]}..({len(mvb)}) same={mv==mvb}  "
              f"max link |A-B| {diff:.3f} mm  dq A {A['q_end'][1+k,6+k]:.3f} B {B['q_end'][1+k,6+k]:.3f}")
print(f"per-slot probe: worst link position difference {worst:.3f} mm")
d = np.abs(A["traj_links"][:, ia] - B["traj_links"][:, ib]).max(axis=(1, 2)) * 1000
print("sinusoid trajectory link |A-B| mm per step:", np.round(d[[0, 4, 9, 19, 29, 39]], 3).tolist())
do = np.abs(A["traj_obs"] - B["traj_obs"])
print("sinusoid trajectory obs max |A-B| per field (over 40 steps):")
for f, (s, e) in sl.items():
    print(f"  {f:22s} {do[:, s:e].max():.3g}")
