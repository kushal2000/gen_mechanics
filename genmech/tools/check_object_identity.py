"""Does each env hold the object its OBSERVATION says it holds?

Every check so far asked "is the authored object the same as the converted
object" -- mass, inertia, 36 physics attributes, and now composed collider
geometry in the object frame, all exact. So the assets are right, and the
authored objects still score 3.00 goals against the converted 5.07.

That leaves a different question, which no asset comparison can answer: the
policy does not observe the object directly. It observes ``object_scales``,
built by ``_build_object_scale_tensor`` from the LEXICOGRAPHIC position of each
Object prim (env_0, env_1, env_10, env_100, ...), not the numeric env id. The
authored path assigns pool entries by ``sorted(env.scene.env_prim_paths)`` and
a comment asserts the two orders agree. A comment is not a measurement, and if
they disagree the policy is told the wrong object size in most envs -- which
looks exactly like a physics regression and is not one.

So read the ground truth back out of PhysX per env and cross-check it two ways:

  1. WITHIN a run: actual object mass vs the ``object_scales`` observation the
     policy receives. Same pool, so mass and scale must agree one-to-one --
     every env sharing a scale must share a mass. Scatter means the observation
     is describing a different object than the one in the scene.

  2. BETWEEN runs: env i's mass under the authored path vs under the converted
     path. Equal everywhere means both paths put the same object in the same
     env; a permutation means the assignment diverged.

Run once per path, then compare:

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.check_object_identity --num_envs 512 --out conv.npz
    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.check_object_identity --num_envs 512 --author \\
        --out auth.npz --compare conv.npz
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_assets_per_type", type=int, default=100)
    parser.add_argument("--author", action="store_true",
                        help="author object USDs instead of converting")
    parser.add_argument("--robot_spec", default="sharpa_iiwa14")
    parser.add_argument("--out", required=True, help="npz to write")
    parser.add_argument("--compare", default=None,
                        help="npz from the other path, to diff against")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    return args


def main() -> None:
    args = _parse_args()

    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import gymnasium as gym
    import numpy as np

    import genmech.tasks  # noqa: F401  registers the env
    from isaaclab.sim.utils import find_matching_prim_paths
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.author_object_usds = args.author
    cfg.assets.robot_spec = args.robot_spec

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    tag = "authored" if args.author else "converted"

    # --- ground truth: what PhysX actually holds, per env -------------------
    masses = inner.object.root_physx_view.get_masses().cpu().numpy().reshape(-1)
    inertias = inner.object.root_physx_view.get_inertias().cpu().numpy()
    inertias = inertias.reshape(args.num_envs, -1)
    scales = inner._object_scale_per_env.cpu().numpy()
    asset_idx = inner._object_asset_index_per_env.cpu().numpy()

    # --- ordering assumption, measured rather than asserted ----------------
    found = list(find_matching_prim_paths("/World/envs/env_.*/Object"))
    sorted_envs = sorted(inner.scene.env_prim_paths)
    found_envs = [p.rsplit("/", 1)[0] for p in found]
    order_match = found_envs == sorted_envs
    print(f"[ident:{tag}] find_matching_prim_paths order == sorted(env_prim_paths): "
          f"{order_match}")
    if not order_match:
        n_bad = sum(a != b for a, b in zip(found_envs, sorted_envs))
        print(f"[ident:{tag}]   MISMATCH in {n_bad}/{len(found_envs)} positions")
        for i, (a, b) in enumerate(zip(found_envs, sorted_envs)):
            if a != b:
                print(f"[ident:{tag}]   first at position {i}: found {a}, sorted {b}")
                break

    # --- check 1: does the observation describe the object in the scene? ---
    # Same pool both ways, so the map asset_index -> mass must be a function.
    # If one asset index shows several masses, some env holds an object its
    # object_scales observation does not describe.
    bad = []
    for idx in np.unique(asset_idx):
        sel = masses[asset_idx == idx]
        spread = float(sel.max() - sel.min())
        if spread > 1e-9:
            bad.append((int(idx), spread, int(sel.size)))
    # This test can only fail when some pool entry is shared by two or more
    # envs. With 600 pool entries and 512 envs every env holds a unique asset,
    # so it passes no matter how wrong the assignment is -- say so rather than
    # let a green line read as evidence.
    reuse = len(asset_idx) - len(np.unique(asset_idx))
    if reuse == 0:
        print(f"[ident:{tag}] NOTE: pool larger than num_envs, every env holds "
              f"a unique asset -- the mass-consistency test below is vacuous. "
              f"The ordering check above is the informative one.")
    print(f"[ident:{tag}] asset_index -> mass is single-valued for "
          f"{len(np.unique(asset_idx)) - len(bad)}/{len(np.unique(asset_idx))} "
          f"pool entries")
    for idx, spread, n in bad[:10]:
        print(f"[ident:{tag}]   asset {idx}: {n} envs span {spread:.6e} kg "
              f"<-- observation does not match the object")

    # Same test through the scale the policy actually reads.
    keys = {}
    for e in range(args.num_envs):
        keys.setdefault(tuple(np.round(scales[e], 9)), []).append(masses[e])
    inconsistent = sum(1 for v in keys.values()
                       if max(v) - min(v) > 1e-9)
    print(f"[ident:{tag}] distinct object_scales values: {len(keys)}; "
          f"{inconsistent} map to more than one mass")

    print(f"[ident:{tag}] mass min/mean/max = {masses.min():.6f} / "
          f"{masses.mean():.6f} / {masses.max():.6f} kg")

    np.savez(args.out, masses=masses, inertias=inertias, scales=scales,
             asset_idx=asset_idx, tag=tag)
    print(f"[ident:{tag}] wrote {args.out}")

    # --- check 2: same object in the same env as the other path? -----------
    ok = not bad and inconsistent == 0
    if args.compare and os.path.exists(args.compare):
        other = np.load(args.compare)
        om, oi, osc = other["masses"], other["inertias"], other["scales"]
        other_tag = str(other["tag"])
        dm = np.abs(masses - om)
        ds = np.abs(scales - osc).max(axis=1)
        di = np.abs(inertias - oi).max(axis=1)
        n_mass = int((dm > 1e-9).sum())
        n_scale = int((ds > 1e-9).sum())
        n_inert = int((di > 1e-12).sum())
        print(f"\n[ident] vs {other_tag}: per-env mass differs in "
              f"{n_mass}/{args.num_envs} envs, scale obs in {n_scale}, "
              f"inertia in {n_inert}")
        if n_mass:
            # Is it a permutation of the same pool, or genuinely different
            # objects? A permutation means the assignment diverged; different
            # values mean the assets differ.
            same_multiset = np.allclose(np.sort(masses), np.sort(om), atol=1e-9)
            print(f"[ident]   same multiset of masses (i.e. a PERMUTATION): "
                  f"{same_multiset}")
            for e in range(args.num_envs):
                if dm[e] > 1e-9:
                    print(f"[ident]   first at env {e}: {tag} {masses[e]:.6f} kg, "
                          f"{other_tag} {om[e]:.6f} kg")
                    break
            ok = False
        else:
            print("[ident]   every env holds an identically-massed object")

    print(f"\n[ident] {'PASS' if ok else 'FAIL'}")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
