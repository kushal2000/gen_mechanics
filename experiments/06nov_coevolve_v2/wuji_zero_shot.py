"""Zero-shot transfer to a commercial hand: Wuji v2 (left) fitted onto the hand grammar and scored by the last v1
k5_none controller (gen 7, 8B steps) on the training task, next to the evolved designs it was trained on.

The fit is the grammar's own (`hand_sampler/grammar_bench/conformed_hands.json`, `conform.conform_buildable`): 5
fingers x 4 joints, within the fit target (worst joint 7.9 mm, axis 16 deg, tip 3.1 mm) at real size, but not
buildable from Feetech servos there (links 10-40 mm; the evolved hands' links are 30-90 mm). So it is scored at real
size and scaled up (positions and link lengths x s, on the 10 mm grid) into the evolved designs' size range. Every
variant is built with the campaign's exporter (FEETECH_COARSE_NOCUT_RULES), so its joint tokens and palm extents come
from the same pipeline as the training hands.

References: from the controller's own training archive (gen 7), the best design by training label in Wuji's niche
(f5_m20-21) and in its neighbours, plus that niche's median design.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/wuji_zero_shot.py build       # hands + manifest + submit
    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/wuji_zero_shot.py report      # table
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
OUT = REPO / "debug_outputs/coevolve/studies/wuji_zero_shot"
RUN = REPO / "debug_outputs/coevolve/v1/k5_none"
GEN = 7
SCALES = (1.0, 1.5, 2.0, 2.5)
NICHES = ("f5_m20-21", "f5_m18-19", "f5_m22-23", "f4_m16-17", "f6_m24-25")
EVAL_SUB = REPO / "experiments/05nov_noevolution/studies/study_beam_eval.sub"


def scaled(design: dict, s: float) -> dict:
    def r10(v: float, floor: int | None = None) -> int:           # floor only for link lengths (6 Nov fix:
        r = int(round(v / 10.0)) * 10                                # base positions are signed)
        return r if floor is None else max(floor, r)
    out = json.loads(json.dumps(design))
    for f in out["fingers"]:
        f["y"], f["z"] = r10(f["y"] * s), r10(f["z"] * s)
        for j in f["joints"]:
            j["length"] = r10(j["length"] * s, 10 if j["length"] > 0 else 0)
    return out


def build() -> None:
    import study_beam as sb
    from coevolution.evolution.descriptors import niche32
    from hand_sampler.grammar import hand as gh
    conf = json.loads((REPO / "hand_sampler/grammar_bench/conformed_hands.json").read_text())["hands"]
    wuji = next(h for h in conf if h["id"] == "wuji2_left")["hand"]
    OUT.mkdir(parents=True, exist_ok=True)
    hands, info = [], {}
    for s in SCALES:
        hid = f"wuji2_left_x{s:g}".replace(".", "p")
        d = scaled(wuji, s)
        viol = gh.check(gh.hand_from_dict(d), gh.mutation_rules(gh.FEETECH_COARSE_NOCUT_RULES))
        _, spec, fits = sb._build((hid, d, str(OUT / "hands")))
        hands.append({"id": hid, "spec": spec})
        info[hid] = {"scale": s, "niche": niche32(d), "violations": viol, "fits_grid": bool(fits), "design": d,
                     "finger_mm": [sum(j["length"] for j in f["joints"]) for f in d["fingers"]]}
        print(f"[wuji] {hid}: niche {info[hid]['niche']}, fingers {info[hid]['finger_mm']} mm, "
              f"{len(viol)} mutation-rule violations{': ' + viol[0] if viol else ''}")
    # references from the controller's own training archive
    man = json.loads((RUN / f"gen_{GEN:04d}/manifest.json").read_text())["hands"]
    lab = json.loads((RUN / f"gen_{GEN:04d}/labels.json").read_text())
    by = {}
    for h in man:
        by.setdefault(h["niche"], []).append(h)
    for n in NICHES:
        ranked = sorted(by.get(n, []), key=lambda h: -lab.get(h["id"], [0])[0])
        picks = [("best", ranked[0])] + ([("median", ranked[len(ranked) // 2])] if n == NICHES[0] else [])
        for tag, h in picks:
            hid = f"ref_{n}_{tag}"
            hands.append({"id": hid, "spec": h["spec"]})
            info[hid] = {"niche": n, "source": h["id"], "train_label": lab.get(h["id"], [None])[0]}
    (OUT / "info.json").write_text(json.dumps(info, indent=1))
    (OUT / "manifest.json").write_text(json.dumps({"generation": GEN, "grid": None, "hands": hands}))
    if os.environ.get("NO_SUBMIT"):
        print("[wuji] built; not submitted (NO_SUBMIT)")
        return
    S = json.loads((RUN / "state.json").read_text())
    tr = S["train_runs"][str(GEN)]
    pol = (RUN / f"gen_{GEN:04d}/policy.pth").resolve()
    (OUT / "jobs.txt").write_text(f"{OUT / 'manifest.json'} 100 {OUT / 'fitness.json'} 7100 {pol} {tr} 0.02\n")
    out = subprocess.run(["sbatch", "--array=0", "--job-name=WUJI_zero_shot", "--partition=portal",
                          "--exclude=portal-compute-01", "--export", f"ALL,JOBS={OUT / 'jobs.txt'}", str(EVAL_SUB)],
                         capture_output=True, text=True)
    print(f"[wuji] {len(hands)} hands -> {out.stdout.strip()} {out.stderr.strip()}")


def report() -> None:
    info = json.loads((OUT / "info.json").read_text())
    man = json.loads((OUT / "manifest.json").read_text())["hands"]
    fit = json.loads((OUT / "fitness.json").read_text())["hands"]
    print(f"{'hand':28s} {'niche':10s} {'goals/ep (100 envs)':>20s}  notes")
    for h in man:
        v = fit[h["spec"]]["goals_per_episode"]
        i = info[h["id"]]
        note = (f"scale x{i['scale']:g}, fingers {i['finger_mm']} mm" if "scale" in i else
                f"{i['source']}, training label {i['train_label']:.1f}")
        print(f"{h['id']:28s} {i['niche'] or '-':10s} {v:20.2f}  {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("build", "report"))
    args = ap.parse_args()
    build() if args.phase == "build" else report()


if __name__ == "__main__":
    main()
