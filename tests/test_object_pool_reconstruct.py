"""Does the reconstructed object pool match the one a run actually builds?

``genmech/eval/embodiments/object_pool.py`` recovers a past run's object pool
from its config, because nothing durable records it -- the pool is built into a
temp dir that dies with the process. Everything downstream (which object each
design trained against, the viewer's object labels, the object-set eval) trusts
that reconstruction.

The failure mode is silent, which is why this test exists. If the reconstruction
drifts from ``generate_handle_head_urdfs`` -- a reordered draw, a shuffle applied
before normalization, a changed default -- it does not raise. It returns a pool
of exactly the right size, full of plausible objects, describing a different run.
Every consumer then reports confident and wrong answers.

So compare against the real generator's output rather than a stored golden: the
generator is the definition, and a golden file would freeze whatever was true
the day it was captured.

No Isaac Sim needed.

    .venv_isaacsim/bin/python tests/test_object_pool_reconstruct.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genmech.eval.embodiments.object_pool import (  # noqa: E402
    ALL_CATEGORIES,
    expected_pool_size,
    pool_index_for_env,
    reconstruct_pool,
    type_counts,
)
from genmech.tasks.pose_reach.utils.generate_objects import (  # noqa: E402
    generate_handle_head_urdfs,
)

# Each case is a config a run could plausibly have used. The seed-42 all-six
# case is the one the population runs actually use; the others cover the knobs
# an eval condition moves (single category, alternate seed, density scale) and
# the shuffle-off debug path.
CASES = [
    dict(handle_head_types=ALL_CATEGORIES, num_assets_per_type=4, object_seed=42),
    dict(handle_head_types=ALL_CATEGORIES, num_assets_per_type=4, object_seed=20260811),
    dict(handle_head_types=ALL_CATEGORIES, num_assets_per_type=3, object_seed=42,
         shuffle=False),
    dict(handle_head_types=("hammer",), num_assets_per_type=5, object_seed=42),
    dict(handle_head_types=("eraser",), num_assets_per_type=5, object_seed=7),
    dict(handle_head_types=("brush", "marker"), num_assets_per_type=2, object_seed=1,
         density_scale=2.0),
]


def check_case(case: dict) -> None:
    pool = reconstruct_pool(**case)

    with tempfile.TemporaryDirectory(prefix="genmech_pooltest_") as tmp:
        paths, scales, params = generate_handle_head_urdfs(
            handle_head_types=tuple(case["handle_head_types"]),
            num_per_type=case["num_assets_per_type"],
            out_dir=tmp,
            seed=case["object_seed"],
            shuffle=case.get("shuffle", True),
            density_scale=case.get("density_scale", 1.0),
        )
        real_names = [Path(p).name for p in paths]

    label = (f"{'+'.join(case['handle_head_types'])} n={case['num_assets_per_type']} "
             f"seed={case['object_seed']}")

    assert len(pool) == len(paths), (
        f"{label}: reconstructed {len(pool)} entries, generator made {len(paths)}")

    # Size must be predictable WITHOUT sampling -- callers assert a pool is big
    # enough for their env count before booting anything.
    predicted = expected_pool_size(case["handle_head_types"], case["num_assets_per_type"])
    assert predicted == len(paths), (
        f"{label}: expected_pool_size said {predicted}, generator made {len(paths)}")

    # The filename encodes sample index, type, both scales and both densities, so
    # matching it in order is a strong check on the whole draw sequence AND the
    # shuffle permutation.
    assert [e.urdf_filename for e in pool] == real_names, (
        f"{label}: URDF filenames differ; first divergence at index "
        f"{next(i for i, (a, b) in enumerate(zip([e.urdf_filename for e in pool], real_names)) if a != b)}")

    # The normalized scale is what the reward and the object_scales observation
    # read, so it has to survive the permutation in lockstep with the paths.
    assert [list(e.scale_normalized) for e in pool] == [list(s) for s in scales], (
        f"{label}: normalized scales differ")

    for i, entry in enumerate(pool):
        h_scale, head_scale, h_density, head_density = params[i]
        assert list(entry.handle_scale) == list(h_scale), f"{label}: handle_scale[{i}]"
        assert (entry.head_scale is None) == (head_scale is None), f"{label}: head presence[{i}]"
        if head_scale is not None:
            assert list(entry.head_scale) == list(head_scale), f"{label}: head_scale[{i}]"
        assert entry.handle_density == h_density, f"{label}: handle_density[{i}]"
        assert entry.head_density == head_density, f"{label}: head_density[{i}]"
        assert entry.index == i, f"{label}: index[{i}] mislabelled"

    print(f"  OK  {label:44s} {len(pool):4d} entries  {type_counts(pool)}")


def check_pool_size_arithmetic() -> None:
    """Six types are twelve distributions -- the mistake this guards against."""
    n = 10
    assert expected_pool_size(ALL_CATEGORIES, n) == 12 * n, (
        "all six handle-head types should map to 12 distributions")
    assert expected_pool_size(("eraser",), n) == n, "eraser has one distribution"
    assert expected_pool_size(("brush",), n) == 4 * n, "brush has four distributions"
    print(f"  OK  pool-size arithmetic: 6 types -> {expected_pool_size(ALL_CATEGORIES, n)} "
          f"entries at n={n}, not {6 * n}")


def check_env_mapping() -> None:
    """env i holds pool entry i % pool_size, and wraps."""
    assert pool_index_for_env(0, 1200) == 0
    assert pool_index_for_env(1199, 1200) == 1199
    assert pool_index_for_env(1200, 1200) == 0
    assert pool_index_for_env(24575, 1200) == 575
    try:
        pool_index_for_env(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("pool_index_for_env should reject pool_size=0")
    print("  OK  env->pool mapping wraps at pool_size")


def main() -> int:
    print("Reconstructed pool vs. the real generator:")
    for case in CASES:
        check_case(case)
    check_pool_size_arithmetic()
    check_env_mapping()
    print("\nobject pool reconstruction test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
