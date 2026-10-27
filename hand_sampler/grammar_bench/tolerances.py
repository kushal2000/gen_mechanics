"""Frozen tolerances for the hand-kinematics grammar benchmark.

Every test takes its tolerance from here. Changing a value is a benchmark commit with a
written reason in project-notes/grammar/LOG.md. Justification: hands are <= 0.3 m across
and have <= 30 joints, so double-precision forward kinematics differs between two correct
implementations by about 1e-14 m; the values below leave a >= 100x margin above that and
still fail float32 contamination (about 1e-8 m) and any convention error (>= 1e-4).
"""

# Position / orientation (geodesic angle) error vs. a hand-derived closed form evaluated
# with sympy/mpmath at >= 30 digits and rounded to float.
ANALYTIC_POS_M = 1e-12
ANALYTIC_ROT_RAD = 1e-12

# Error vs. an independent float64 implementation (Pinocchio, yourdfpy, MuJoCo mj_kinematics)
# reading the SAME source file.
ORACLE_POS_M = 1e-10
ORACLE_ROT_RAD = 1e-10

# Loop-closure residual after our own solve, and re-evaluated by an oracle (M2).
CLOSURE_OWN = 1e-12
CLOSURE_ORACLE_M = 1e-10
FREUDENSTEIN_RAD = 1e-9          # valid where cond(J_dependent) <= 1e4

# Rank decisions for mobility: singular values above RANK_RTOL * sigma_max count as nonzero
# and the gap to the next singular value must be >= RANK_GAP, else the configuration is
# marked near-singular and excluded (at most NEAR_SINGULAR_FRACTION of samples).
RANK_RTOL = 1e-8
RANK_GAP = 1e4
NEAR_SINGULAR_FRACTION = 0.05

# Negative control: perturbing one dependent coordinate by NEG_CONTROL_STEP must raise a
# residual of at least NEG_CONTROL_MIN, otherwise the residual is zero by construction.
NEG_CONTROL_STEP = 1e-3
NEG_CONTROL_MIN = 1e-6

# Exact comparisons (limits, coupling tuples, JSON and URDF round trips) use ==.

# Configuration sampling per hand: N_RANDOM seeded samples inside the admissible box shrunk
# by BOX_SHRINK_REL relative, plus the extremal configurations (zero, all-lower, all-upper,
# and 4 mixed corners) = 71 configurations for N_RANDOM = 64.
N_RANDOM = 64
BOX_SHRINK_REL = 1e-9
N_EXTREMAL = 7
