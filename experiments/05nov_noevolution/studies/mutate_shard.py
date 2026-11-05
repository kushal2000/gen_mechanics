"""One shard of a search round's mutations, for a SLURM CPU array task (study_vf_archive.py --shards N).

    .venv_isaacsim/bin/python experiments/05nov_noevolution/studies/mutate_shard.py --jobs <round dir> --shard k

Reads <round dir>/shard_<k>_in.json ([[parent id, design, seed], ...]), applies study_vf_archive._mutate_all (one random
valid application of each of the 9 mutation types) to every parent with all the task's CPUs, writes
<round dir>/shard_<k>_out.json ([[parent id, [[operator, child design, edit], ...]], ...]).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--shard", type=int, required=True)
    args = ap.parse_args()
    import study_vf_archive as sva
    d = Path(args.jobs)
    jobs = json.loads((d / f"shard_{args.shard:03d}_in.json").read_text())
    n = len(os.sched_getaffinity(0))
    t = time.time()
    with Pool(n) as pool:
        res = pool.map(sva._mutate_all, [(design, seed) for _, design, seed in jobs], chunksize=4)
    out = d / f"shard_{args.shard:03d}_out.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps([[pid, kids] for (pid, _, _), kids in zip(jobs, res)]))
    tmp.rename(out)                                   # complete or absent: the driver never reads half a file
    print(f"[shard {args.shard}] {len(jobs)} parents, {sum(len(k) for k in res)} children on {n} CPUs in "
          f"{time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
