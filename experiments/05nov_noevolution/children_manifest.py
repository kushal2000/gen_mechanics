"""Every valid one-step child of one design, built and listed for viser_population.py, grouped by operator.

    .venv_isaacsim/bin/python experiments/05nov_noevolution/children_manifest.py --design f4_m12-13_126 [--iteration 3]
    .venv_isaacsim/bin/python experiments/05nov_noevolution/viser_population.py \
        --manifest debug_outputs/noevolution/children/<design>/manifest.json --port 8089

The manifest's "niche" field is the group the viewer's dropdown shows: "parent", then one per operator; each child
also carries its edit (shown under the design) and its own niche.
"""
import argparse, json, sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/05nov_noevolution/studies"))
sys.path.insert(0, str(REPO / "experiments/04nov_audit_grammar"))
import study_beam as sb  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--iteration", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    from coevolution.evolution.descriptors import niche32
    g = REPO / f"debug_outputs/evolution/noevolution_4096/gen_{args.iteration:04d}"
    man = json.loads((g / "manifest.json").read_text())
    h = next(h for h in man["hands"] if h["id"] == args.design)
    spec = json.loads((REPO / h["spec"]).read_text())
    d = spec["design"] if isinstance(spec["design"], dict) else json.loads(spec["design"])
    _, kids = sb._children((args.design, d))      # exactly what the grammar yields and deploys (canonical form)
    seen, rows = {sb._key(d)}, []
    for key, cd, op, edit in kids:
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"{op}_{sum(r['op'] == op for r in rows):03d}", "op": op, "edit": edit, "design": cd,
                     "cell": niche32(cd) or "none"})
    out = REPO / "debug_outputs/noevolution/children" / args.design
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        built = dict((i, (s, ok)) for i, s, ok in pool.map(sb._build, [(r["id"], r["design"], str(out)) for r in rows], chunksize=4))
    hands = [{"id": args.design, "spec": h["spec"], "niche": "parent", "edit": "the parent", "cell": h["niche"]}]
    hands += [{"id": r["id"], "spec": built[r["id"]][0], "niche": r["op"], "edit": r["edit"], "cell": r["cell"],
               "fits_grid": built[r["id"]][1]} for r in rows]
    (out / "manifest.json").write_text(json.dumps({"parent": args.design, "grid": man["grid"], "hands": hands}, indent=1))
    print(f"[children] {args.design}: {len(rows)} distinct children {dict(Counter(r['op'] for r in rows))}; "
          f"{sum(not built[r['id']][1] for r in rows)} do not fit the slot grid -> {out / 'manifest.json'}")

if __name__ == "__main__":
    main()
