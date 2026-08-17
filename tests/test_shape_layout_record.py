"""Does the authoring-recorded collider map match what PhysX actually loads?

The env's friction pass needs each link's shape-index range so it can paint
fingertip friction onto the fingertip shapes. Asking PhysX for that -- one
create_rigid_body_view per link per design -- is correct and costs ~96 min at
24,576 designs. The fast path instead uses the map recorded while the designs
were authored, which is free.

Being wrong here is SILENT. Every value involved is a plausible friction, so a
mis-assigned range does not raise; it just gives some links the wrong contact
properties and changes grasp behaviour. Two earlier attempts at this
optimisation are why this test exists:

  1. A layout cache keyed on a GUESSED signature (per-slot active mask plus
     self-collision adjacency). Segment lengths also decide whether a link gets
     a capsule, so designs sharing a mask could differ. Design 5120 of an 8,192
     population took another design's layout. A 2,048-design run passed clean.
  2. Counting the shapes in USD instead. Correct, but slower than the PhysX
     calls it replaced.

So this checks EVERY design in a population, not a sample -- the failure above
was invisible at 2,048 designs and to a 16-design spot check.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        tests/test_shape_layout_record.py --population_count 256
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args():
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population_count", type=int, default=256)
    p.add_argument("--population_seed", type=int, default=2)
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.headless = True
    return a


def main() -> None:
    args = _parse_args()
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app

    import gymnasium as gym

    import genmech.tasks  # noqa: F401
    from genmech.tasks.pose_reach.env_multi_cfg import PoseReachMultiEnvCfg

    n = int(args.population_count)
    cfg = PoseReachMultiEnvCfg()
    # One env per design: every design must be checked, and env i holds design
    # i % k, so k = n is the only setting where all of them are present.
    cfg.scene.num_envs = n
    cfg.assets.robot_population_count = n
    cfg.assets.robot_population_seed = int(args.population_seed)
    cfg.assets.author_robot_usds = True
    cfg.assets.num_assets_per_type = 2

    env = gym.make("GenMech-PoseReachMulti-Direct-v0", cfg=cfg).unwrapped

    recorded = getattr(env, "_robot_collider_links", None)
    assert recorded, "authoring recorded no collider map"
    assert len(recorded) == n, f"recorded {len(recorded)} designs, expected {n}"

    view = env.robot.root_physx_view
    link_names = list(view.shared_metatype.link_names)
    design_idx = env._robot_design_index_per_env.detach().cpu().numpy()

    # The arm is identical across designs and referenced from one USD, so its
    # per-link counts are measured once -- the same thing the fast path does.
    def measure(env_id):
        out, start = [], 0
        for nm, path in zip(link_names, view.link_paths[env_id]):
            lv = env.robot._physics_sim_view.create_rigid_body_view(path)
            out.append((nm, start, start + lv.max_shapes))
            start += lv.max_shapes
        return out

    # Use the PRODUCTION helpers, not a copy of their arithmetic. The first
    # version of this test reimplemented the layout maths, inherited the same
    # off-by-one the fast path had (the merged palm counted once instead of
    # arm + palm), and so agreed with the code instead of checking it. What is
    # being checked is the code's output against PHYSX, which is the only
    # independent source of truth here.
    from genmech.tasks.pose_reach.utils.scene_utils import (
        arm_counts_from,
        shape_layouts_from_record,
    )

    ref_design = int(design_idx[0])
    arm_counts = arm_counts_from(measure(0), recorded[ref_design])
    layouts = shape_layouts_from_record(link_names, recorded, arm_counts)

    bad = []
    for env_id in range(env.num_envs):
        design = int(design_idx[env_id])
        expected = layouts[design][0]
        actual = measure(env_id)
        if actual != expected:
            diff = [(a, b) for a, b in zip(actual, expected) if a != b]
            bad.append((design, diff[:3]))

    if bad:
        print(f"[layout] {len(bad)}/{env.num_envs} designs MISMATCH")
        for design, diff in bad[:5]:
            print(f"[layout]   design {design}: {diff}")
        raise SystemExit(1)

    # A test that only ever compares two empty things passes vacuously, which is
    # a failure mode this repo has already shipped once (compare_authored_robot
    # agreed that two arms had the same colliders when both had none).
    total = sum(sum(v.values()) for v in recorded.values())
    assert total > 0, "recorded map is empty for every design"
    print(f"[layout] {env.num_envs} envs / {n} designs verified against PhysX, "
          f"{total} recorded collision shapes, {len(arm_counts)} arm links")
    print("shape layout record test OK")

    del app
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
