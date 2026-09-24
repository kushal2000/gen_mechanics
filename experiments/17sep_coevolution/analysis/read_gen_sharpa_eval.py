"""Where did gen-SHARPA land? Reads the frozen evaluations and reports its rank.

    .venv_isaacsim/bin/python experiments/17sep_coevolution/analysis/read_gen_sharpa_eval.py [label]
"""
import glob, json, pathlib, sys
import numpy as np
from coevolution.design_rewards import merge_rank_files

label = sys.argv[1] if len(sys.argv) > 1 else "coevolution_v2_gen2k"
M = pathlib.Path("debug_outputs/17sep_coevo_analysis/gen_sharpa_eval")
rows = []
print(f"{'gen':>4} {'tol':>7} {'sharpa return':>14} {'rank':>10} {'survives':>9} {'pop p50':>8} {'pop p90':>8} {'episodes':>9}")
for meta in sorted(M.glob(f"{label}_gen*.meta.json")):
    m = json.load(open(meta)); g, slot = m["gen"], m["slot"]
    stem = meta.with_suffix("").with_suffix("").as_posix()
    tables = sorted(glob.glob(f"{stem}.rank*.table.json"))
    if not tables:
        print(f"{g:4d} {m['tolerance']:7.4f}   (not run yet)"); continue
    rew = merge_rank_files(tables)
    v = np.array([rew.get(i, {}).get("return_mean", np.nan) for i in range(1024)])
    s = v[slot]; rank = int((v > s).sum()) + 1
    ep = rew.get(slot, {}).get("episodes", 0)
    rows.append(dict(gen=g, tol=m["tolerance"], ret=float(s), rank=rank, episodes=int(ep),
                     p50=float(np.nanmedian(v)), p90=float(np.nanpercentile(v, 90))))
    print(f"{g:4d} {m['tolerance']:7.4f} {s:14.0f} {rank:6d}/1024 {'yes' if rank <= 512 else 'NO':>9} "
          f"{np.nanmedian(v):8.0f} {np.nanpercentile(v, 90):8.0f} {ep:9d}"
          + ("" if len(tables) == 2 else f"   [{len(tables)}/2 ranks]"))
if rows:
    out = M / f"ranks_{label}.json"; out.write_text(json.dumps(rows, indent=1)); print("wrote", out)
