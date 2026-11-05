"""The finger-removal test (30 Oct, the user's call): do MAP-Elites' 3-finger elites lose anything when one finger goes?

    .venv_isaacsim/bin/python experiments/oct_2026_experiments/30oct_evolution_evaluation/removal_test/build_removals.py

Takes every 3-finger elite of MAP-Elites' current map and builds each of its single-finger removals with the evolution
runs' own builder. Most plain removals break the grammar's mount rule (the palm is built round the finger bases and the
wrist mount must sit within half a base step of its centre; dropping a finger moves the centre 23-53 mm), so each
removal also gets a re-centred variant (``_rm<i>c``): the remaining fingers shifted together by the fewest whole base
steps that put the palm back on the mount and pass every check -- the same hand, re-mounted. Both use (hand_sampler.grammar.operators remove_finger, the grammar's checks, urdf_export.build, the 5 + 2 x 5
slot-grid check), into debug_outputs/evolution/removal_test/hands/<parent>_rm<i>/. Writes
debug_outputs/evolution/removal_test/manifest.json: parents and children in one hand set, so both are scored in the same
pass (eval_hands.py; zero-shot with MAP-Elites' latest policy, then after removal_finetune.sub).
"""
from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import replace
from pathlib import Path

from coevolution.evolution.archive import Archive
from coevolution.evolution.config import EvolutionConfig
from coevolution.evolution.descriptors import niche
from coevolution.evolution.variation import fits_grid
from hand_sampler.grammar import hand as gh
from hand_sampler.grammar import operators as gops
from hand_sampler.grammar.urdf_export import build

RUN = Path("debug_outputs/evolution/map_elites")


def recentred(hand, rules):
    """The fewest whole base steps (in y, z) moving every finger so the hand passes every check; None if none within 3."""
    pitch = rules.base_pitch_mm
    for dy, dz in sorted(itertools.product(range(-3, 4), repeat=2), key=lambda d: (abs(d[0]) + abs(d[1]), d)):
        if (dy, dz) == (0, 0):
            continue
        h = replace(hand, fingers=tuple(replace(f, y=f.y + dy * pitch, z=f.z + dz * pitch) for f in hand.fingers))
        if not gh.check(h, rules):
            return h, (dy * pitch, dz * pitch)
    return None
OUT = Path("debug_outputs/evolution/removal_test")


def main() -> None:
    cfg = EvolutionConfig.load(RUN / "config.json")
    rules = cfg.rule_set()
    st = json.loads((RUN / "state.json").read_text())
    arch = Archive.load(RUN / "archive.json")
    g = st["history"][-1]["generation"]
    hands, rows = [], []
    for n in sorted(k for k in st["map"] if k.startswith("f3_")):
        p = arch[st["map"][n]]
        hands.append({"id": p.id, "spec": p.spec, "role": "parent", "niche": n})
        parent = gh.hand_from_dict(p.design)
        for i, f in enumerate(parent.fingers):
            child = gops._remove_finger(parent, (i,), rules)
            variants = [(f"{p.id}_rm{i}", child, None)]
            rc = recentred(child, rules)
            if rc is not None:
                variants.append((f"{p.id}_rm{i}c", rc[0], rc[1]))
            for name, hand, shift in variants:
                why = "; ".join(gh.check(hand, rules))
                if why:
                    rows.append((name, f"not admissible: {why}"))
                    continue
                row = build(hand, name, OUT / "hands" / name)
                if not fits_grid(row["spec"], cfg.grid):
                    shutil.rmtree(OUT / "hands" / name, ignore_errors=True)
                    rows.append((name, "does not fit the slot grid"))
                    continue
                d = gh.hand_to_dict(hand)
                hands.append({"id": name, "spec": row["spec"], "role": "child", "parent": p.id, "removed": i,
                              "removed_joints": len(f.joints), "recentred_mm": shift, "niche": niche(d)})
                rows.append((name, f"built: {niche(d)}, removed finger {i} ({len(f.joints)} joints)"
                             + (f", fingers shifted {shift} mm" if shift else "")))
    (OUT / "manifest.json").write_text(json.dumps({"generation": g, "grid": cfg.grid, "hands": hands}, indent=1))
    for r in rows:
        print(*r)
    print(f"{sum(h['role'] == 'parent' for h in hands)} parents, {sum(h['role'] == 'child' for h in hands)} children "
          f"-> {OUT / 'manifest.json'} (MAP-Elites map at iteration {g})")


if __name__ == "__main__":
    main()
