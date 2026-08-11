"""The pose viewer must be consistent with the robot it is rendering.

Two viewer bugs reached a training run in a row, both in a path no test touched:

  1. A stale relative import crashed --capture_viewer 22 minutes into a run,
     after the full 24576-env scene build.
  2. The robot URDF URL was pinned to SHARPA, so the Allegro run published 23
     joint names against a 29-joint URDF and the browser reported
     'Joint "index_joint_0" not found in URDF'.

Neither is exotic; both are caught by simply building the viewer's HTML for
every registered robot and checking it against that robot's URDF. That is what
this does, without booting Kit -- it drives the pure functions in
genmech.viewer.pose_viewer with synthetic frames rather than a live env.

    .venv_isaacsim/bin/python tests/test_pose_viewer.py
"""

from __future__ import annotations

import sys
from urllib.parse import urlsplit

import yourdfpy

from genmech.robots import REGISTRY, get_robot_spec
from genmech.utils.paths import resolve as resolve_repo_path
from genmech.viewer.pose_viewer import (
    DEFAULT_ROBOT_URDF_RELATIVE_PATH,
    GITHUB_RAW_BASE_MAIN,
    build_pose_viewer_html,
)


def _fake_frame(spec) -> dict:
    """One frame in the exact shape capture_frame() emits, for this robot."""
    import numpy as np

    zero_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # xyz + xyzw
    return {
        "env_id": 0,
        "robot_joint_names": list(spec.joint_names_canonical),
        "robot_joint_pos": np.zeros(spec.num_joints),
        "robot_base_pose": zero_pose,
        "object_pose": zero_pose,
        "goal_pose": zero_pose,
        "table_pose": zero_pose,
    }


def main() -> None:
    minimal_urdf = (
        '<robot name="x"><link name="base"/></robot>'
    )
    failures: list[str] = []

    for name in sorted(REGISTRY):
        spec = get_robot_spec(name)
        print(f"\n=== {name} ===")

        # 1. The spec's URDF exists and declares exactly the spec's joints. This
        #    is the invariant the Allegro failure violated.
        urdf_path = resolve_repo_path(spec.urdf_path)
        if not urdf_path.exists():
            failures.append(f"{name}: URDF missing at {urdf_path}")
            continue
        urdf = yourdfpy.URDF.load(str(urdf_path))
        actuated = {
            j for j, jt in urdf.joint_map.items()
            if jt.type in ("revolute", "continuous", "prismatic")
        }
        missing = sorted(set(spec.joint_names_canonical) - actuated)
        extra = sorted(actuated - set(spec.joint_names_canonical))
        if missing or extra:
            failures.append(
                f"{name}: spec/URDF joint mismatch (missing={missing}, extra={extra})"
            )
        else:
            print(f"  [1] {len(actuated)} URDF joints match the spec exactly")

        # 2. The HTML actually references THIS robot's URDF, not another's.
        html = build_pose_viewer_html(
            frames=[_fake_frame(spec)],
            object_urdf_text=minimal_urdf,
            table_urdf_text=minimal_urdf,
            robot_urdf_relpath=spec.urdf_path,
            url_check="skip",
        )
        if spec.urdf_path not in html:
            failures.append(f"{name}: built HTML does not reference {spec.urdf_path}")
        else:
            print(f"  [2] HTML references {spec.urdf_path}")

        # 3. No OTHER registered robot's URDF leaks in. A hardcoded default
        #    would show up here even when the right path is also present.
        for other in REGISTRY.values():
            if other.name != name and other.urdf_path in html:
                failures.append(
                    f"{name}: HTML also references {other.name}'s URDF "
                    f"({other.urdf_path}) -- a hardcoded path is leaking through"
                )
        print("  [3] no other robot's URDF referenced")

        # 4. Every joint the frame names is one the URDF can animate.
        for jname in _fake_frame(spec)["robot_joint_names"]:
            if jname not in actuated:
                failures.append(f"{name}: frame names joint {jname!r}, absent from URDF")
        print(f"  [4] all {spec.num_joints} frame joint names exist in the URDF")

    # 5. The default raw base must serve this repo, since the generated Allegro
    #    URDF and its mirrored meshes exist nowhere upstream.
    host_path = urlsplit(GITHUB_RAW_BASE_MAIN).path
    if "gen_mechanics" not in host_path:
        failures.append(
            f"GITHUB_RAW_BASE_MAIN points at {GITHUB_RAW_BASE_MAIN!r}, which will "
            f"not serve this repo's generated assets"
        )
    else:
        print(f"\n[5] default raw base serves this repo: {GITHUB_RAW_BASE_MAIN}")
    print(f"[6] fallback URDF path still resolves: "
          f"{resolve_repo_path(DEFAULT_ROBOT_URDF_RELATIVE_PATH).exists()}")

    if failures:
        for f in failures:
            print(f"\nFAIL: {f}")
        raise AssertionError(f"{len(failures)} pose-viewer inconsistency(ies)")

    print("\n[viewer] pose viewer test OK")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
