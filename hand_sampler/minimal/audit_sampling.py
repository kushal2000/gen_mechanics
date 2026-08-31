"""What the sampler actually favours, dimension by dimension.

    python audit_sampling.py --count 4000

For each design dimension it prints the sampler's empirical marginal next to the
"uniform over enumerated designs" marginal, and the ratio between them. A ratio
of 1.0 means the sampler explores that dimension in proportion to how much of
the design space it occupies.

Proportional is NOT automatically the goal. Uniform-over-designs is dominated by
whatever dimension is most combinatorially rich, which is usually not what you
want to spend rollouts on -- 3-finger hands outnumber 2-finger ones by ~3,000x,
but nobody wants 99.97% of a population to have 3 fingers. The point of the table
is to make each such choice visible and deliberate.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter

from hand_sampler.minimal import sampler as S


def pct(c: Counter) -> dict:
    n = sum(c.values())
    return {k: 100.0 * v / n for k, v in sorted(c.items())}


def table(title: str, emp: Counter, space: Counter | None, note: str = "") -> None:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    e, s = pct(emp), pct(space) if space else None
    keys = sorted(set(e) | (set(s) if s else set()))
    head = f"    {'value':<22}{'sampled':>10}"
    if s:
        head += f"{'of space':>10}{'ratio':>9}"
    print(head)
    for k in keys:
        row = f"    {str(k):<22}{e.get(k, 0.0):>9.1f}%"
        if s:
            sv = s.get(k, 0.0)
            r = (e.get(k, 0.0) / sv) if sv > 1e-9 else float("inf")
            row += f"{sv:>9.1f}%{r:>9.2f}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hands = S.sample_population(args.seed, args.count)
    fingers = [f for h in hands for f in h.fingers]

    # ---- the enumerated design space, for comparison ----------------------
    kin = S.enumerate_fingers()
    space_links, space_joints, space_couplings = Counter(), Counter(), Counter()
    for lens, dofs in kin:
        n_c = len(S.couplings_of(dofs))
        space_links[len(lens)] += n_c
        space_joints[sum(len(d) for d in dofs)] += n_c
        for c in S.couplings_of(dofs):
            space_couplings[("passive", len(c.passive))] += 1
            space_couplings[("rigid", len(c.rigid))] += 1

    print(f"{len(hands)} hands / {len(fingers)} fingers, seed {args.seed}")

    table("n_fingers per hand", Counter(len(h.fingers) for h in hands), None,
          "sampler: uniform over (2, 3). Uniform-over-designs would be ~99.97% "
          "3-finger,\n  since 3-finger hands outnumber 2-finger by ~3,000x. "
          "Balance is the deliberate choice here.")

    table("faces used", Counter(tuple(sorted(f.face for f in h.fingers)) for h in hands),
          None, "should be uniform over the 3 pairs when n=2, and the single triple when n=3")

    table("n_links per finger", Counter(len(f.link_lengths) for f in fingers),
          space_links,
          "sampler: rng.randint(1, 3), i.e. uniform over COMPLEXITY LEVEL")

    table("n_joints per finger", Counter(len(S.finger_joints(f)) for f in fingers),
          space_joints)

    table("n_actuators per finger", Counter(S.n_actuators(f) for f in fingers), None)

    emp_p = Counter(("passive", len(f.coupling.passive)) for f in fingers)
    emp_r = Counter(("rigid", len(f.coupling.rigid)) for f in fingers)
    table("passive joints per finger",
          Counter(k[1] for k in emp_p.elements()),
          Counter(k[1] for k in space_couplings.elements() if k[0] == "passive"),
          "sampler draws the COUNT uniformly (capped at 3); the space is "
          "Binomial-shaped")
    table("rigid pairs per finger",
          Counter(k[1] for k in emp_r.elements()),
          Counter(k[1] for k in space_couplings.elements() if k[0] == "rigid"),
          "sampler: uniform over matchings")

    roles = Counter(S.joint_role(f, j) for f in fingers for j in S.finger_joints(f))
    table("joint role mix", roles, None)

    # splay should be flat; bucket it
    sb = Counter(int(math.floor(math.degrees(f.splay) / 15.0) * 15) for f in fingers)
    table("splay, 15-degree buckets", sb, None, "uniform on [-45, 45) by construction")

    def mean(c: Counter) -> float:
        n = sum(c.values())
        return sum(k * v for k, v in c.items()) / n

    emp_j = Counter(len(S.finger_joints(f)) for f in fingers)
    print("\nSUMMARY")
    print(f"    mean joints per finger    sampled {mean(emp_j):.2f}   "
          f"of space {mean(space_joints):.2f}")
    print(f"    mean actuators per finger sampled "
          f"{mean(Counter(S.n_actuators(f) for f in fingers)):.2f}")
    print("    The sampler leans SIMPLE on every axis at once: fewer joints,")
    print("    fewer passive, fewer rigid pairs. Those biases compound.")


if __name__ == "__main__":
    main()
