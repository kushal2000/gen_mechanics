"""Generate a hand's printable kit from its genotype.

    python -m hand_sampler.cad --hand hand.json [--out outputs/designs]
    python -m hand_sampler.cad --commercial xhand_right
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hand_sampler.grammar.hand import hand_from_json

from .package import DESIGNS_DIR, generate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hand", type=Path, help="a hand_grammar genotype (JSON)")
    src.add_argument("--commercial", help="a conformed commercial hand's id (grammar_bench/conformed_hands.json)")
    ap.add_argument("--out", type=Path, default=DESIGNS_DIR)
    a = ap.parse_args(argv)
    if a.hand is not None:
        hand = hand_from_json(a.hand.read_text())
    else:
        from hand_sampler.grammar import build_conformed as BC

        hand = BC.load()[a.commercial]["hand"]
    kit = generate(hand, a.out, progress=lambda s: print(s, flush=True))
    print(kit.report.summary())
    print(f"kit: {kit.folder}  ({kit.zip_path}, {kit.seconds:.0f} s)")
    return 0 if kit.report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
