"""Audit the 29 Oct evolution runs' bookkeeping: scores, populations and the MAP-Elites map.

    .venv_isaacsim/bin/python experiments/04nov_audit_grammar/audit_bookkeeping.py

Kit-free. Per run and generation g:
  scores      every archive fitness entry of generation g (trained, not screen) equals gen_g/fitness.json's value for
              that hand's spec (goals_per_episode + 0.1 x reward_per_step), and every hand of gen_g/manifest.json got one
  population  every hand gen_g/fitness.json scores is in gen_g/manifest.json (the reverse is the scores check)
  map_elites  the map is replayed from the seeds with the recorded scores (generation.py's rule: a challenger trained
              at g takes its niche if empty or it scores strictly higher, challengers in descending score order) and
              compared with state.json's per-generation won/lost counts and its final map
Prints the summary; writes experiments/04nov_audit_grammar/results/bookkeeping.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "results"
SHAPING = 0.1


def main() -> None:
    from coevolution.evolution import descriptors
    report = {}
    for run in ("map_elites", "baseline"):
        rd = REPO / "debug_outputs/evolution" / run
        recs = {r["id"]: r for r in json.loads((rd / "archive.json").read_text())["hands"]}
        state = json.loads((rd / "state.json").read_text())
        by_spec = {r["spec"]: r for r in recs.values()}
        gens = sorted(int(p.name[4:]) for p in rd.glob("gen_*") if (p / "fitness.json").exists())
        score_bad, missing, pop_bad = [], [], []
        trained_at = {}                     # (id, g) -> value
        for g in gens:
            fit = json.loads((rd / f"gen_{g:04d}" / "fitness.json").read_text())["hands"]
            man = [h["spec"] for h in json.loads((rd / f"gen_{g:04d}" / "manifest.json").read_text())["hands"]]
            for spec in man:
                r = by_spec.get(spec)
                e = [x for x in (r["fitness"] if r else []) if x["generation"] == g and not x.get("screen")]
                if r is None or not e:
                    missing.append((g, spec))
                    continue
                v = float(fit[spec]["goals_per_episode"]) + SHAPING * float(fit[spec].get("reward_per_step", 0.0))
                if abs(e[-1]["value"] - v) > 1e-9:
                    score_bad.append((g, r["id"], e[-1]["value"], v))
                trained_at[(r["id"], g)] = e[-1]["value"]
            extra = set(fit) - set(man)
            if extra:
                pop_bad.append((g, "scored but not in manifest", sorted(extra)[:3]))
        rep = {"generations": len(gens), "score_mismatch": score_bad[:20], "n_score_mismatch": len(score_bad),
               "missing_score": missing[:20], "n_missing": len(missing), "population": pop_bad}
        if run == "map_elites":
            seed_man = json.loads((REPO / "assets/urdf/29oct_niche_seed/manifest.json").read_text())["hands"]
            emap = {}
            for h in seed_man:
                rid = h["id"]
                emap.setdefault(descriptors.niche(recs[rid]["design"]), rid)
            replay = []
            for g in gens:
                chall = [r for r in recs.values() if r["operator"] != "seed" and r.get("admitted") == g
                         and (r["id"], g) in trained_at and r["id"] not in emap.values()]
                won = lost = 0
                for c in sorted(chall, key=lambda r: -trained_at[(r["id"], g)]):
                    n = descriptors.niche(c["design"])
                    inc = emap.get(n)
                    if inc is None or trained_at[(c["id"], g)] > trained_at[(inc, g)]:
                        emap[n] = c["id"]
                        won += 1
                    else:
                        lost += 1
                replay.append((g, won, lost))
            hist = {h["generation"]: h for h in state["history"]}
            # state's history row g carries the challengers settled in propose(g), i.e. those trained at g
            count_bad = [(g, w, l, hist.get(g, {}).get("won"), hist.get(g, {}).get("lost")) for g, w, l in replay
                         if g in hist and (hist[g].get("won"), hist[g].get("lost")) != (w, l)]
            final_bad = {n: (emap.get(n), state["map"].get(n)) for n in set(emap) | set(state["map"])
                         if emap.get(n) != state["map"].get(n)}
            rep.update({"replay_count_mismatch": count_bad[:20], "n_replay_count_mismatch": len(count_bad),
                        "final_map_mismatch": final_bad})
        report[run] = rep
        print(f"[bookkeeping] {run}: {len(gens)} generations; score mismatches {len(score_bad)}, missing scores "
              f"{len(missing)}, population issues {len(pop_bad)}"
              + (f"; map replay: won/lost mismatches {rep['n_replay_count_mismatch']}, final map mismatches "
                 f"{len(rep['final_map_mismatch'])}" if run == "map_elites" else ""), flush=True)
        for k in ("score_mismatch", "missing_score", "replay_count_mismatch"):
            if rep.get(k):
                print(f"    {k}: {rep[k][:4]}", flush=True)
        if rep.get("final_map_mismatch"):
            print(f"    final_map_mismatch: {list(rep['final_map_mismatch'].items())[:4]}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bookkeeping.json").write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
