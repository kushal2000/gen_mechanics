"""Hand-picked object pools, for runs where every design should meet the same
few objects (``assets.object_assignment: design_cycle``).

A sampled pool of 24 is two random draws from each of the twelve size
distributions -- which lands wherever it lands. These are chosen instead: for
each distribution one object at the small, light end of its ranges and one at
the large, heavy end, with the dimensions that matter for a grasp -- handle
cross-section, overall length, where the mass sits -- pushed apart on purpose.
Every value lies inside its distribution's sampling range, so the set is
in-distribution for the 1200-object pool and a policy can be evaluated on
either.

Conventions are ``author_objects``': a 3-tuple is a box (lx, ly, lz) with lx
along the handle axis; a 2-tuple is a cylinder (length, diameter) along the
same axis; the head sits on the +x end. Densities in kg/m^3: handles 300-600,
hammer and screwdriver heads 800-2000.
"""

from __future__ import annotations

import math

from hand_sampler.design_space import compute_mass_and_inertia

# (type, handle_scale, head_scale, handle_density, head_density)
#   type              handle                 head                      dens
DIVERSE_24: list[tuple] = [
    # hammer, box handle: the classic; a small tack hammer and a heavy framing hammer
    ("hammer",      (0.16, 0.022, 0.018),  (0.03, 0.06, 0.03),        400, 1000),
    ("hammer",      (0.28, 0.038, 0.028),  (0.055, 0.11, 0.055),      550, 1800),
    # hammer, round handle: the longest lever and the heaviest object in the set
    ("hammer",      (0.17, 0.018),         (0.025, 0.055, 0.025),     350,  900),
    ("hammer",      (0.29, 0.030),         (0.06, 0.12, 0.06),        600, 2000),
    # screwdriver, box handle: stubby with a short shaft; slim with a long one
    ("screwdriver", (0.08, 0.035, 0.035),  (0.08, 0.012, 0.012),      450, 1200),
    ("screwdriver", (0.115, 0.027, 0.027), (0.145, 0.010, 0.010),     500, 1800),
    # screwdriver, round handle
    ("screwdriver", (0.075, 0.038),        (0.075, 0.015, 0.015),     350,  900),
    ("screwdriver", (0.12, 0.026),         (0.15, 0.011, 0.011),      550, 1900),
    # marker: a thin pen -- the smallest cross-section here -- and a fat marker
    ("marker",      (0.08, 0.016),         (0.012, 0.006, 0.006),     300,  300),
    ("marker",      (0.15, 0.030),         (0.03, 0.010, 0.010),      600,  600),
    # spatula, box handle: a thin flexible-looking one and a big broad one
    ("spatula",     (0.11, 0.014, 0.007),  (0.06, 0.035, 0.012),      350,  350),
    ("spatula",     (0.20, 0.025, 0.025),  (0.15, 0.07, 0.03),        600,  600),
    # spatula, round handle
    ("spatula",     (0.12, 0.014),         (0.07, 0.04, 0.015),       400,  400),
    ("spatula",     (0.19, 0.025),         (0.14, 0.065, 0.028),      550,  550),
    # eraser: no head; a small stick and the bulkiest block in the set
    ("eraser",      (0.075, 0.025, 0.020), None,                      300, None),
    ("eraser",      (0.14, 0.065, 0.065),  None,                      600, None),
    # brush, box handle, tall head (v1)
    ("brush",       (0.06, 0.012, 0.012),  (0.06, 0.035, 0.035),      300,  300),
    ("brush",       (0.19, 0.038, 0.028),  (0.115, 0.05, 0.075),      600,  600),
    # brush, round handle, tall head (v1)
    ("brush",       (0.08, 0.012),         (0.055, 0.03, 0.04),       350,  350),
    ("brush",       (0.20, 0.030),         (0.12, 0.05, 0.08),        550,  550),
    # brush, box handle, wide flat head (v2)
    ("brush",       (0.10, 0.020, 0.015),  (0.07, 0.07, 0.025),       400,  400),
    ("brush",       (0.18, 0.035, 0.030),  (0.12, 0.12, 0.04),        600,  600),
    # brush, round handle, wide flat head (v2)
    ("brush",       (0.07, 0.015),         (0.06, 0.06, 0.02),        300,  300),
    ("brush",       (0.16, 0.025),         (0.11, 0.11, 0.035),       500,  500),
]

CURATED_POOLS: dict[str, list[tuple]] = {"diverse24": DIVERSE_24}


def object_mass(handle_scale, head_scale, handle_density, head_density) -> float:
    m = compute_mass_and_inertia(handle_scale, handle_density)[0]
    if head_scale is not None:
        m += compute_mass_and_inertia(head_scale, head_density)[0]
    return float(m)


def curated_entries(name: str, object_base_size: float, density_scale: float = 1.0) -> list[dict]:
    """The pool as ``sample_pool_params`` entries, in the listed order."""
    from .generate_objects import _scale_to_3d
    from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS

    if name not in CURATED_POOLS:
        raise KeyError(f"no curated pool {name!r}; have {sorted(CURATED_POOLS)}")
    entries = []
    for k, (typ, handle, head, hd, headd) in enumerate(CURATED_POOLS[name]):
        dist = _matching_distribution(typ, handle, head)
        _check_in_range(dist, handle, head, hd, headd, f"{name}[{k}] {typ}")
        x, y, z = _scale_to_3d(handle)
        entries.append({
            "type": typ, "shape": dist.shape,
            "distribution_index": OBJECT_SIZE_DISTRIBUTIONS.index(dist), "sample_index": k,
            "handle_scale": tuple(float(v) for v in handle),
            "head_scale": None if head is None else tuple(float(v) for v in head),
            "handle_density": float(hd) * density_scale,
            "head_density": None if headd is None else float(headd) * density_scale,
            "scale_normalized": (x / object_base_size, y / object_base_size, z / object_base_size),
        })
    return entries


def _matching_distribution(typ, handle, head):
    from .object_size_distributions import OBJECT_SIZE_DISTRIBUTIONS
    for d in OBJECT_SIZE_DISTRIBUTIONS:
        if d.type != typ or len(d.handle_min_lengths) != len(handle):
            continue
        if (d.head_min_lengths is None) != (head is None):
            continue
        if head is not None and not all(lo - 1e-9 <= v <= hi + 1e-9 for v, lo, hi in
                                        zip(head, d.head_min_lengths, d.head_max_lengths)):
            continue   # brush v1 vs v2 differ only by head range
        return d
    raise ValueError(f"no size distribution for {typ} handle {handle} head {head}")


def _check_in_range(d, handle, head, hd, headd, who):
    def within(v, lo, hi):
        return all(l - 1e-9 <= x <= h + 1e-9 for x, l, h in zip(v, lo, hi))
    if not within(handle, d.handle_min_lengths, d.handle_max_lengths):
        raise ValueError(f"{who}: handle {handle} outside {d.handle_min_lengths}..{d.handle_max_lengths}")
    if head is not None and not within(head, d.head_min_lengths, d.head_max_lengths):
        raise ValueError(f"{who}: head {head} outside {d.head_min_lengths}..{d.head_max_lengths}")
    if not d.handle_min_density - 1e-9 <= hd <= d.handle_max_density + 1e-9:
        raise ValueError(f"{who}: handle density {hd} outside {d.handle_min_density}..{d.handle_max_density}")
    if head is not None and not d.head_min_density - 1e-9 <= headd <= d.head_max_density + 1e-9:
        raise ValueError(f"{who}: head density {headd} outside {d.head_min_density}..{d.head_max_density}")


def describe(name: str = "diverse24") -> str:
    """One line per object: shape, overall length, handle cross-section, mass."""
    rows = []
    for k, (typ, handle, head, hd, headd) in enumerate(CURATED_POOLS[name]):
        length = handle[0] + (0.0 if head is None else (head[0] if len(head) == 3 else head[1]))
        thick = max(handle[1:]) if len(handle) == 3 else handle[1]
        rows.append(f"{k:2d} {typ:11s} {'box' if len(handle) == 3 else 'cyl'} handle  "
                    f"length {length*100:5.1f} cm  handle {thick*1000:4.0f} mm  "
                    f"mass {object_mass(handle, head, hd, headd)*1000:6.0f} g")
    return "\n".join(rows)


__all__ = ["CURATED_POOLS", "DIVERSE_24", "curated_entries", "describe", "object_mass"]
