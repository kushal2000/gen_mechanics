r"""Pick each hand's expert checkpoint and apply the keep rule, from deterministic
evaluations in the population env (``run.py --mode eval --policies expert``
once per checkpoint kind, and ``--policies zero`` for the floor). Kit-free.

- Checkpoint: per design, the kind (e.g. ``best`` reward or ``last``) whose
  expert turns the object more per episode (ties: longer holding).
- Keep rule, after GET-Zero's filter (an expert must complete a full turn
  within 30 s): rad/s while holding >= 2 pi / 30, and rotations per episode
  at least twice the zero-action level.

    python -m isaacsimenvs.inhand_reorient.distill.select_experts --evals best=EVAL_BEST last=EVAL_LAST \
        --floors FLOORS --maps best=BEST.json last=LAST.json --population TRAIN.json --out DIR

Writes ``DIR/experts_sel.json`` (kept designs: source -> checkpoint),
``DIR/selection.json`` (every design's numbers and decision) and
``DIR/population.json`` (the kept designs of ``--population``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MIN_RAD_PER_S = 2.0 * math.pi / 30.0
FLOOR_FACTOR = 2.0


def keep(expert: dict, zero: dict | None) -> bool:
    """GET-Zero's full turn within 30 s, and clearly more rotation than zero actions."""
    z = max(float((zero or {}).get("rotations_mean") or 0.0), 1e-3)
    return float(expert["rad_per_s"]) >= MIN_RAD_PER_S and float(expert["rotations_mean"]) >= FLOOR_FACTOR * z


def select(evals: dict, zero: dict, maps: dict) -> dict:
    """``evals[kind][source]`` and ``zero[source]``: ``dagger.summarise`` rows;
    ``maps[kind][source]``: checkpoint paths. Returns ``source -> row``."""
    rows = {}
    sources = []
    for e in evals.values():
        sources += [s for s in e if s not in sources]
    for s in sources:
        cands = [(k, e.get(s) or {}) for k, e in evals.items()]
        cands = [(k, m) for k, m in cands if m.get("episodes")]
        if not cands:
            continue
        k, m = max(cands, key=lambda km: (km[1]["rotations_mean"], km[1]["ttt_mean_s"]))
        rows[s] = {"checkpoint": k, "path": maps[k][s], "expert": m, "zero": zero.get(s) or {},
                   "kept": keep(m, zero.get(s)), "other": {kk: mm for kk, mm in cands if kk != k}}
    return rows


def _pairs(items):
    return dict(i.split("=", 1) for i in items)


def main(argv=None) -> int:
    import argparse

    from ..scene import population_file as pf

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evals", nargs="+", required=True, help="kind=EVAL_DIR (run.py --mode eval output)")
    ap.add_argument("--floors", required=True, help="EVAL_DIR with the zero-action evaluation")
    ap.add_argument("--maps", nargs="+", required=True, help="kind=EXPERTS.json")
    ap.add_argument("--population", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    evals = {k: json.loads((Path(d) / "eval.json").read_text())["expert"] for k, d in _pairs(a.evals).items()}
    zero = json.loads((Path(a.floors) / "eval.json").read_text())["zero"]
    maps = {k: json.loads(Path(p).read_text()) for k, p in _pairs(a.maps).items()}
    rows = select(evals, zero, maps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "experts_sel.json").write_text(json.dumps({s: r["path"] for s, r in rows.items() if r["kept"]}, indent=1))
    (out / "selection.json").write_text(json.dumps(rows, indent=1))
    doc = json.loads(Path(a.population).read_text())
    designs = [d for d in doc["designs"] if d["source"] in rows and rows[d["source"]]["kept"]]
    doc = dict(doc, designs=designs,
               population_sha256=pf._sha256_hex(pf._canonical_json_bytes([d["sha256"] for d in designs])))
    (out / "population.json").write_text(json.dumps(doc, sort_keys=True, indent=2))
    pf.load_population(out / "population.json")
    for s, r in rows.items():
        m, z = r["expert"], r["zero"]
        print(f"{s:34s} {r['checkpoint']:5s} ttt {m['ttt_mean_s']:5.2f} rot {m['rotations_mean']:.3f} "
              f"rad/s {m['rad_per_s']:.2f} | zero ttt {z.get('ttt_mean_s', 0):5.2f} rot {z.get('rotations_mean', 0):.3f}"
              f" | {'KEEP' if r['kept'] else 'DROP'}")
    print(f"kept {sum(r['kept'] for r in rows.values())} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
