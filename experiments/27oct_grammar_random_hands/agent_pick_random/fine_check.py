"""Stage 3 again for the leaders on a finer set of pen axis directions (240 on the sphere -> ~120 on the hemisphere,
4x stage 3's 30), at each hand's 3 best stage-3 centres (clean.json and clean_seed17_8c.json), fresh samples.

    .venv_isaacsim/bin/python experiments/27oct_grammar_random_hands/agent_pick_random/fine_check.py 025,081,075 --jobs 6

Writes fine_check.json: per hand and centre, CH2 / CH3 / ROTC over the fine direction set.
"""
from __future__ import annotations

import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clean as CL  # noqa: E402
import screen as S  # noqa: E402
from screen import gvb, finger_section  # noqa: E402


def run(name, centres, seed=29):
    t0 = time.time()
    H = S.load(name)
    nF = len(H.fingers)
    fp = gvb.footprint_mm(H)
    moving = [finger_section(H, i) == 0 for i in range(nF)]
    levels = S.HINGE_LEVELS if H.palm_joints else np.array([0.0])
    fds = {}
    for li, h in enumerate(levels):
        for i in range(nF):
            if li > 0 and not moving[i]:
                fds[(li, i)] = fds[(0, i)]
                continue
            fds[(li, i)] = CL.finger_data(H, i, 40000, np.random.default_rng(seed + 11 * li + i), h, fp)
    frames = S.pen_frames(n_dirs=240)
    out = []
    for c in centres:
        h2, h3, rot = CL.at_centre(fds, levels, nF, np.array(c), frames)
        out.append({"centre": c, "CH2": h2, "CH3": h3, "ROTC": rot})
    return {"name": name, "n_dirs": int(frames.shape[0]), "per_centre": out,
            "CH3_best": max(o["CH3"] for o in out), "ROTC_best": max(o["ROTC"] for o in out),
            "best": max(out, key=lambda o: o["CH3"] + o["ROTC"]), "secs": time.time() - t0}


def main():
    names = [x if x.startswith("coarse") else f"coarse_s0_{x}" for x in sys.argv[1].split(",")]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
    cents = {}
    for f in ("clean.json", "clean_seed17_8c.json"):
        for r in json.load(open(HERE / f)):
            pc = sorted(zip(r["per_centre"], r["centres"]), key=lambda t: -(t[0][1] + t[0][2]))
            cents.setdefault(r["name"], [])
            for _, c in pc[:2]:
                if c not in cents[r["name"]]:
                    cents[r["name"]].append(c)
    from multiprocessing import Pool
    rows = []
    with Pool(jobs) as pool:
        res = [pool.apply_async(run, (nm, cents[nm][:3])) for nm in names]
        for r in res:
            r = r.get()
            rows.append(r)
            print(f"{r['name']} dirs={r['n_dirs']} CH3={r['CH3_best']:.2f} ROTC={r['ROTC_best']:.2f} "
                  + " ".join(f"[{o['centre']}: {o['CH2']:.2f}/{o['CH3']:.2f}/{o['ROTC']:.2f}]" for o in r["per_centre"])
                  + f" ({r['secs']:.0f}s)", flush=True)
    json.dump(rows, open(HERE / "fine_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
