#!/usr/bin/env python3
"""Grammar-solo comparison: build the (non-grammar) control population --
the projected commercial hands allegro_right and sharpa_left_on_iiwa14 --
as a population.json with the same schema/provenance as a grammar
population (``population_file.projected_entry`` + ``write_population``,
both existing, unmodified). No grasp search here: run
``grasp_cache_gen --population <out>/population.json`` separately (same
command used for the grammar variants), then
``split_population.py`` to get one single-design population per control
for solo training.

CPU-only (no Kit, no GPU): pure grammar/kinematics admission checks.

    .venv_isaacsim/bin/python3 experiments/grammar_solo/build_controls_population.py \\
        --hands allegro_right,sharpa_left_on_iiwa14 --out RUN_DIR/viable/controls/population.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaacsimenvs.inhand_reorient.scene import population_file as pf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", default="allegro_right,sharpa_left_on_iiwa14")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    entries = []
    for hand_id in a.hands.split(","):
        entry, status, reason = pf.projected_entry(hand_id)
        if status != "admitted":
            raise RuntimeError(f"control hand {hand_id!r} is not admitted ({status}): {reason}")
        entries.append(entry)
    doc = pf.write_population(Path(a.out), entries)
    print(f"[build_controls_population] {len(entries)} control(s) -> {a.out} "
          f"(population_sha256={doc['population_sha256'][:12]}...)")
    for e, hand_id in zip(entries, a.hands.split(",")):
        print(f"  - {hand_id}: {e.source} ({e.sha256[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
