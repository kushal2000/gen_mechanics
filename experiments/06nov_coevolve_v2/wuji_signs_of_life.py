"""Signs of life for a Wuji v2 zero-shot: Wuji-like grammar hands with only the policy's three joint kinds, scored
locally by several v1 controllers (100 envs each, eval_values.py on this node's GPU).

The four fingers are the grammar fit of wuji2_left (conformed_hands.json: flexion / abduction / flexion / flexion,
in the plate); the fit's thumb has oblique axes no training hand has, so it is replaced by a three-kind thumb that
keeps Wuji's base and facing and mimics opposition the grammar's way -- abduction at the base, then three flexions,
the finger tilted out of the plate toward the grasp side. Variants: that thumb at tilt 30 / 60, and no thumb, each
scaled x1.5 / x2 / x2.5 (positions and links, 10 mm grid), plus the controller's best design in Wuji's niche.

    .venv_isaacsim/bin/python experiments/06nov_coevolve_v2/wuji_signs_of_life.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = REPO / "debug_outputs/coevolve/studies/wuji_signs_of_life"
CONTROLLERS = [("k5_none", 7), ("k1_none", 8)]
SCALES = (1.5, 2.0, 2.5)


def variants() -> dict[str, dict]:
    import wuji_zero_shot as W
    conf = next(h for h in json.loads((REPO / "hand_sampler/grammar_bench/conformed_hands.json").read_text())["hands"]
                if h["id"] == "wuji2_left")["hand"]
    fingers, thumb = conf["fingers"][:4], conf["fingers"][4]
    lens = [j["length"] for j in thumb["joints"]]
    out = {}
    for tilt in (30, 60):
        t = {"y": thumb["y"], "z": thumb["z"], "facing": thumb["facing"], "tilt": tilt,
             "joints": [{"type": "hinge", "axis": [90, 0], "length": lens[0]}] +
                       [{"type": "hinge", "axis": [0, 0], "length": L} for L in lens[1:]]}
        d = {"grammar": conf.get("grammar", "hand_grammar/2.0"), "fingers": fingers + [t]}
        for s in SCALES:
            out[f"wuji3k_tilt{tilt}_x{s:g}".replace(".", "p")] = W.scaled(d, s)
    for s in SCALES:
        out[f"wuji_nothumb_x{s:g}".replace(".", "p")] = W.scaled({"grammar": "hand_grammar/2.0", "fingers": fingers}, s)
    return out


def main() -> None:
    import study_beam as sb
    from coevolution.evolution.descriptors import niche32
    OUT.mkdir(parents=True, exist_ok=True)
    hands = []
    for hid, d in variants().items():
        _, spec, _ = sb._build((hid, d, str(OUT / "hands")))
        hands.append({"id": hid, "spec": spec, "niche": niche32(d)})
    ref = next(h for h in json.loads((REPO / "debug_outputs/coevolve/studies/wuji_zero_shot/manifest.json").read_text())
               ["hands"] if h["id"] == "ref_f5_m20-21_best")
    hands.append({"id": "ref_best_f5_m20-21", "spec": ref["spec"], "niche": "f5_m20-21"})
    man = OUT / "manifest.json"
    man.write_text(json.dumps({"generation": 0, "grid": {"n_rigid": 6, "n_moving": 0, "n_slots": 5}, "hands": hands}))
    results = {}
    for run, g in CONTROLLERS:
        rd = REPO / "debug_outputs/coevolve/v1" / run
        tr = json.loads((rd / "state.json").read_text())["train_runs"][str(g)]
        fo = OUT / f"fitness_{run}_g{g}.json"
        if not fo.exists():
            print(f"[wuji-sol] {run} gen {g}: scoring {len(hands)} hands x 100 envs ...", flush=True)
            subprocess.run([str(REPO / ".venv_isaacsim/bin/python"), str(REPO / "experiments/05nov_noevolution/studies/eval_values.py"),
                            "--manifest", str(man), "--checkpoint", str((rd / f"gen_{g:04d}/policy.pth").resolve()),
                            "--train-run", tr, "--out", str(fo), "--envs-per-hand", "100", "--seed", "7300",
                            "--table-clearance", "0.02"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results[f"{run} g{g}"] = json.loads(fo.read_text())["hands"]
    cols = list(results)
    print(f"{'hand':24s} {'niche':10s} " + " ".join(f"{c:>22s}" for c in cols))
    for h in hands:
        vals = [results[c][h["spec"]] for c in cols]
        print(f"{h['id']:24s} {str(h['niche']):10s} " + " ".join(
            f"{v['goals_per_episode']:7.2f} g/ep {v['episode_steps']:5.0f} st" for v in vals))


if __name__ == "__main__":
    main()
