"""The two runs' shared starting population: one random grammar hand per niche (coevolution/evolution/seeding.py).

    .venv_isaacsim/bin/python experiments/29oct_evolution_from_scratch/make_seed.py

-> assets/urdf/29oct_niche_seed/<seed_<niche>>/ and manifest.json (the slot grid pinned). Both runs' .sub point at
that manifest, so archive and MAP-Elites start from identical hands.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coevolution.evolution.seeding import one_per_niche  # noqa: E402
from hand_sampler.grammar import hand as gh  # noqa: E402

OUT = REPO / "assets/urdf/29oct_niche_seed"
GRID = {"n_rigid": 5, "n_moving": 2, "n_slots": 5}

if __name__ == "__main__":
    if (OUT / "manifest.json").exists():
        sys.exit(f"{OUT} already holds a seed set; delete it to redraw")
    m = one_per_niche(OUT, gh.FEETECH_COARSE_RULES, GRID, seed=0, max_draws=20000)
    print(f"{len(m['hands'])} seed hands -> {OUT}; niches never reached: {m['missing']}")
