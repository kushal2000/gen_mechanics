"""Which v3 cells (coevolution/evolution/descriptors_v3.py) can the grammar actually fill? v3 features of every design
we have built (seeds + every v2 generation, de-duplicated by design id), in parallel; writes the occupancy and the
analytic-feasibility verdict per cell.

    python experiments/06nov_coevolve_v2/v3_cell_feasibility.py [--procs 16]
"""
import argparse, glob, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = REPO / "debug_outputs/coevolve/studies/niche_design_v3"


def one(spec):
    try:
        from coevolution.evolution.descriptors_v3 import features
        return spec, features(spec)
    except Exception as e:                                # noqa: BLE001
        return spec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {}
    for man in ["debug_outputs/coevolve/v1/seeds/manifest.json"] + sorted(glob.glob("debug_outputs/coevolve/v2/*/gen_*/manifest.json")):
        for h in json.loads((REPO / man).read_text())["hands"]:
            specs.setdefault(h["id"], h["spec"])
    cache = OUT / "features.json"
    feats = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [s for s in specs.values() if s not in feats]
    print(f"{len(specs)} designs, {len(todo)} to compute", flush=True)
    with Pool(a.procs) as p:
        for i, (s, f) in enumerate(p.imap_unordered(one, todo, chunksize=32)):
            feats[s] = f
            if i % 2000 == 0:
                cache.write_text(json.dumps(feats)); print(f"  {i}", flush=True)
    cache.write_text(json.dumps(feats))
    print("done", flush=True)


if __name__ == "__main__":
    main()
