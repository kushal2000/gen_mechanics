"""Per-hand progress of the commercial-hands sanity run (16 Nov 2026): goals / episode, pen lifted, from the run's
per-episode logs (design_stats), over the last --dumps dumps.
    .venv_isaacsim/bin/python experiments/16nov_v6/sanity_progress.py [--dumps 3]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("--dumps", type=int, default=3); a = ap.parse_args()
d = sorted(glob.glob(str(REPO / "debug_outputs/train_logs/sanity/commercial_v5_interface/*/design_stats")))[-1]
hands = json.loads((Path(d) / "hands.json").read_text())
hoe = np.load(Path(d) / "meta.npz")["hand_of_env"].astype(int)
f = sorted(glob.glob(f"{d}/step_*.npz"))
z = [np.load(x) for x in f[-a.dumps:]]
c = lambda k: np.concatenate([q[k] for q in z])
env, g = c("env").astype(int), c("goals")
step = int(f[-1].split("step_")[1].split(".")[0]) if f else 0
print(f"{len(f)} dumps, last at env step {step} (~{step * 24576 / 1e9:.2f}B samples)")
for k, h in enumerate(hands):
    m = hoe[env] == k
    print(f"  {Path(h).name.split('_left')[0]:8s} goals/ep {g[m].mean():6.3f}  (max {g[m].max():3.0f}, >=1 goal {100 * (g[m] >= 1).mean():3.0f}%, {m.sum()} ep)")
print(f"  {'ALL':8s} goals/ep {g.mean():6.3f}")
