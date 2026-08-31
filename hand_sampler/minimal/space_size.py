"""Where the design-space size actually comes from, and what would shrink it.

    python space_size.py

The space is now fully discrete, so every number here is exact.
"""
from math import comb, log2

from hand_sampler.minimal import sampler as S

kin = S.enumerate_fingers()
n_kin = len(kin)
n_coup = sum(len(S.couplings_of(d)) for l, d in kin)
coup_per = n_coup / n_kin
n_splay = len(S.SPLAY_CHOICES)
per_finger = n_coup * n_splay

def total(pf: int) -> int:
    return comb(3, 2) * pf ** 2 + pf ** 3

print("PER-FINGER FACTORS")
print(f"    kinematics (lengths x dofs)   {n_kin:>10,}   {log2(n_kin):5.1f} bits")
print(f"    x couplings (avg per kin)     {coup_per:>10.1f}   {log2(coup_per):5.1f} bits")
print(f"    x splay values                {n_splay:>10,}   {log2(n_splay):5.1f} bits")
print(f"    = distinct finger designs     {per_finger:>10,}   {log2(per_finger):5.1f} bits")
print()
print(f"TOTAL HANDS  (3 choose 2)*N^2 + N^3 = {total(per_finger):,}")
print(f"    2-finger  {comb(3,2)*per_finger**2:>22,}")
print(f"    3-finger  {per_finger**3:>22,}   <- dominates")
print()
print("The exponent is the story: independent fingers cube the per-finger count.")
print("Shrinking a per-finger factor by k shrinks the total by ~k^3.")
print()

base = total(per_finger)
print(f"{'OPTION':<46}{'total hands':>20}{'vs now':>10}")
print(f"{'(current)':<46}{base:>20,}{'1.0x':>10}")

# Inter-finger constraints (identical fingers, thumb+fingers) are deliberately
# NOT listed: fingers are to stay independently sampled.
opts = [
    ("splay 30 deg steps (3 values)", total(n_coup * 3)),
    ("splay fixed at 0 (1 value)", total(n_coup * 1)),
    ("max 2 actuators per finger", None),
    ("at most 1 rigid pair per finger", None),
    ("no cross-DOF rigid pairs (FE-FE, AA-AA only)", None),
    ("max 2 links per finger", None),
]

# recompute the two coupling restrictions exactly
def coup_count(pred) -> int:
    n = 0
    for l, d in kin:
        n += sum(1 for c in S.couplings_of(d) if pred(d, c))
    return n

c1 = coup_count(lambda d, c: len(c.rigid) <= 1)
c2 = coup_count(lambda d, c: all(a[1] == b[1] for a, b in c.rigid))

def coup_with_act_cap(cap: int) -> int:
    n = 0
    for l, d in kin:
        nj = len(S.joints_of(d))
        n += sum(1 for c in S.couplings_of(d)
                 if nj - len(c.rigid) - len(c.passive) <= cap)
    return n

two_link = sum(len(S.couplings_of(d)) for l, d in kin if len(l) <= 2)

for name, val in opts:
    if name.startswith("max 2 actuators"):
        val = total(coup_with_act_cap(2) * n_splay)
    elif name.startswith("at most 1 rigid"):
        val = total(c1 * n_splay)
    elif name.startswith("no cross-DOF"):
        val = total(c2 * n_splay)
    elif name.startswith("max 2 links"):
        val = total(two_link * n_splay)
    print(f"{name:<46}{val:>20,}{base / val:>9.1f}x")
