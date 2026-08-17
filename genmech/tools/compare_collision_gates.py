"""Does the analytic self-collision gate agree with the mesh gate?

The analytic gate is ~1000x faster, which is worth nothing if it accepts hands
the real check rejects. A faster gate that changes WHICH hands are accepted does
not speed up population building; it silently builds a different population.

So this runs both gates over the same random candidates and reports:

  * ACCEPT/REJECT agreement -- the decision that actually matters, since the
    sampler only uses the boolean
  * per-pair agreement, and any pair one gate finds that the other does not
  * the depth each reports where both agree, since the two measure differently:
    the analytic gate computes exact penetration (r_a + r_b - segment distance),
    while the mesh gate samples vertices and takes a signed-distance maximum,
    which UNDER-estimates. Expect the analytic depths to be >= the mesh ones.

Disagreement near the EPS_M threshold is expected and benign -- a pair grazing
at 1e-5 m is a coin flip between the two. Disagreement on deep overlaps is not.

    .venv_isaacsim/bin/python -m genmech.tools.compare_collision_gates \\
        --candidates 200
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--max_disagree_frac", type=float, default=0.02,
                    help="accept/reject disagreement tolerated, as a fraction")
    args = ap.parse_args()

    from genmech.robots.generated import params as P
    from genmech.tools.capsule_collision import analytic_hand_hits
    from genmech.tools.check_self_collision import generated_hand_hits

    rng = random.Random(args.seed)
    work = Path(tempfile.mkdtemp(prefix="genmech_gatecmp_"))

    n_drawn = agree = 0
    disagree: list[tuple[str, bool, bool, float, float]] = []
    t_analytic = t_mesh = 0.0
    both_free = both_hit = 0

    while n_drawn < args.candidates:
        try:
            hand = P.sample(rng, name=f"cand_{n_drawn:04d}")
        except P.InvalidHand:
            continue      # rejected by the cheap proxies; neither gate sees it
        n_drawn += 1

        t0 = time.perf_counter()
        a_hits = analytic_hand_hits(hand)
        t_analytic += time.perf_counter() - t0

        t0 = time.perf_counter()
        m_hits = generated_hand_hits(hand, work, name=f"cand_{n_drawn:04d}")
        t_mesh += time.perf_counter() - t0

        a_free, m_free = not a_hits, not m_hits
        if a_free == m_free:
            agree += 1
            both_free += int(a_free)
            both_hit += int(not a_free)
        else:
            disagree.append((
                hand.name, a_free, m_free,
                a_hits[0][2] if a_hits else 0.0,
                m_hits[0][2] if m_hits else 0.0,
            ))

    frac = len(disagree) / max(n_drawn, 1)
    print(f"[gates] {n_drawn} candidates that passed params.validate()")
    print(f"[gates] accept/reject agreement : {agree}/{n_drawn} "
          f"({100 * agree / max(n_drawn,1):.1f}%)")
    print(f"[gates]   both collision-free   : {both_free}")
    print(f"[gates]   both collide          : {both_hit}")
    print(f"[gates]   disagreed             : {len(disagree)}")
    for name, a_free, m_free, a_d, m_d in disagree[:12]:
        who = "analytic says FREE, mesh says HIT" if a_free else \
              "analytic says HIT, mesh says FREE"
        print(f"[gates]     {name}: {who} "
              f"(analytic depth {a_d * 1000:.4f} mm, mesh {m_d * 1000:.4f} mm)")

    print(f"\n[gates] time: analytic {t_analytic:.3f}s, mesh {t_mesh:.1f}s "
          f"-> {t_mesh / max(t_analytic, 1e-9):.0f}x faster")
    print(f"[gates] per candidate: analytic {t_analytic / n_drawn * 1e6:.0f} us, "
          f"mesh {t_mesh / n_drawn * 1000:.0f} ms")

    ok = frac <= args.max_disagree_frac
    print(f"\n[gates] {'PASS' if ok else 'FAIL'}: disagreement {100 * frac:.1f}% "
          f"(tolerance {100 * args.max_disagree_frac:.1f}%)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
