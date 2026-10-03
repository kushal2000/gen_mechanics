#!/usr/bin/env python3
"""Grammar-solo comparison, step 2: split a population.json into one
single-design population file per design, for solo training (one design
per job).

Pure stdlib (json + hashlib only) -- deliberately does not import
``isaacsimenvs``/``hand_sampler``: it only re-slices an already-validated
population file, recomputing ``population_sha256`` for each 1-design slice
exactly as ``population_file.write_population``/``load_population`` define
it (``sha256_hex(canonical_json([entry["sha256"]]))``), so each output file
still round-trips through ``population_file.load_population`` unchanged.
Run with plain ``python3`` (no venv needed); no GPU, no Kit.

    python3 experiments/grammar_solo/split_population.py \\
        RUN_DIR/viable/G_V1/population.json --out-dir RUN_DIR/designs/G_V1

Writes ``<out-dir>/<index>/population.json`` (index 0..N-1, in the source
file's order) and ``<out-dir>/manifest.csv`` (index, source, role, sha256).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def split(population_path: Path, out_dir: Path) -> list[dict]:
    doc = json.loads(population_path.read_text())
    designs = doc["designs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, design in enumerate(designs):
        one_doc = dict(doc)
        one_doc["designs"] = [design]
        one_doc["population_sha256"] = sha256_hex(canonical_json_bytes([design["sha256"]]))
        design_dir = out_dir / str(i)
        design_dir.mkdir(parents=True, exist_ok=True)
        (design_dir / "population.json").write_text(json.dumps(one_doc, sort_keys=True, indent=2))
        role = None
        # viable.py/build_viable_population.py don't store role in the population file itself
        # (it lives in viability.json's "designs" list, same order); carried through if present.
        rows.append({"index": i, "source": design["source"], "sha256": design["sha256"], "role": role})
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "source", "sha256", "role"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("population", help="path to a population.json (multi-design)")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    rows = split(Path(a.population), Path(a.out_dir))
    print(f"[split_population] {a.population}: {len(rows)} design(s) -> {a.out_dir}/<index>/population.json")
    for r in rows:
        print(f"  {r['index']}: {r['source']} ({r['sha256'][:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
