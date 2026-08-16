"""Is the k*n cost just an unbatched change-notification loop?

``spawn_multi_asset`` has two loops and only one is batched:

    for asset_cfg in cfg.assets_cfg:        # k template spawns -- NO ChangeBlock
        asset_cfg.func(...)
    with Sdf.ChangeBlock():                 # n copies -- batched
        for prim_path in prim_paths:
            Sdf.CopySpec(...)

``Sdf.ChangeBlock`` batches USD change notifications so subscribers (the PhysX
parser, Fabric, Hydra) process them once instead of after every edit. And with
replicate_physics=False, InteractiveScene clones the env Xforms BEFORE spawning
assets, so by the time the template loop runs the stage already holds n env
subtrees. If each unbatched spawn triggers stage-wide change processing, then k
spawns against an n-sized stage costs O(k*n) from a loop that reads as O(k).

This tests that directly and in isolation -- no PhysX, no scene, no reset:

  * build a stage holding n empty env Xforms
  * spawn the same USD k times into a template scope
  * do it with and without a surrounding ChangeBlock
  * repeat for several n

The prediction, if the hypothesis holds: UNBATCHED time grows with n while
BATCHED time stays roughly flat in n. If both grow, the cost is inside the USD
reference resolution itself and batching will not help.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_changeblock --k 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=32, help="template spawns per trial")
    parser.add_argument("--env_counts", type=int, nargs="+",
                        default=[0, 1024, 4096, 16384])
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import tempfile
    from pathlib import Path

    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom

    from genmech.robots.generated.population import load_population
    from genmech.robots.generated.synth_spec import synth_spec
    from genmech.tools.multi_embodiment_demo import _prepare_robot_usd

    work = Path(tempfile.mkdtemp(prefix="genmech_cb_"))
    (work / "usd").mkdir(parents=True, exist_ok=True)
    spec = synth_spec(load_population(args.seed)[0])
    usd = _prepare_robot_usd(spec, work, "probe")
    print(f"[cb] using {usd}")

    def trial(n_envs: int, batched: bool) -> float:
        """Time k spawns into a fresh stage that already holds n_envs Xforms."""
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")

        # The env Xforms exist BEFORE any asset is spawned, which is the order
        # InteractiveScene uses when replicate_physics=False.
        if n_envs:
            with Sdf.ChangeBlock():
                for i in range(n_envs):
                    Sdf.CreatePrimInLayer(stage.GetRootLayer(),
                                          Sdf.Path(f"/World/envs/env_{i}"))
        UsdGeom.Scope.Define(stage, "/World/Template")

        cfg = sim_utils.UsdFileCfg(usd_path=usd)
        t0 = time.perf_counter()
        if batched:
            # NOT POSSIBLE: the spawner uses Usd-level APIs (UsdStage.DefinePrim),
            # which raise inside an Sdf.ChangeBlock -- a change block defers the
            # composition that Usd-level calls read back. This is exactly why
            # spawn_multi_asset batches only its Sdf.CopySpec loop and leaves the
            # template-spawn loop unbatched. The asymmetry is required, not an
            # oversight, so "wrap it in a ChangeBlock" is not an available fix.
            return float("nan")
        for i in range(args.k):
            cfg.func(f"/World/Template/Asset_{i:04d}", cfg)
        return time.perf_counter() - t0

    print(f"\n[cb] k={args.k} spawns per trial")
    print(f"[cb] {'n_envs':>8} {'unbatched':>12} {'batched':>12} {'speedup':>9}")
    rows = []
    for n in args.env_counts:
        u = trial(n, batched=False)
        b = trial(n, batched=True)
        rows.append((n, u, b))
        print(f"[cb] {n:>8} {u:>11.2f}s {b:>11.2f}s {u / max(b, 1e-9):>8.2f}x")

    # Does the UNBATCHED cost grow with n? That is the whole question.
    base_u = rows[0][1]
    base_b = rows[0][2]
    print(f"\n[cb] growth from n=0 to n={rows[-1][0]}:")
    print(f"[cb]   unbatched {base_u:.2f}s -> {rows[-1][1]:.2f}s "
          f"({rows[-1][1] / max(base_u, 1e-9):.2f}x)")
    print(f"[cb]   batched   {base_b:.2f}s -> {rows[-1][2]:.2f}s "
          f"({rows[-1][2] / max(base_b, 1e-9):.2f}x)")
    print("[cb] hypothesis holds if unbatched grows with n and batched does not")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
