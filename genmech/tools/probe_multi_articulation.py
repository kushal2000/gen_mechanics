"""Can one Isaac Lab scene hold articulations with DIFFERENT topologies?

Follow-up to probe_heterogeneous_envs.py, which established that a single
``Articulation`` view requires uniform topology: ``num_joints`` reads
``root_physx_view.shared_metatype.dof_count``, one scalar for the whole view.
Geometry may vary per env, joint count may not.

That blocks variable finger count, which is the headline variation in the
morphology literature (LocoFormer trains across 100k robots spanning bipeds,
quadrupeds and wheeled variants -- joint counts that cannot share a view).
IsaacGym exposes DOF state as a flat tensor with per-actor offsets, so
heterogeneity is natural there; Isaac Lab's wrapper is stricter. The question is
whether the restriction is per-VIEW (workable: use several views) or
per-SCENE (a real blocker for this stack).

Two configurations are probed:

  A. Disjoint env subsets -- articulation X owns envs 0..k, Y owns k+1..N.
     This is what a morphology-conditioned rollout actually wants: env i holds
     morphology m_i, one batched step for all of them.
  B. Co-resident -- both articulations present in every env, side by side.
     Weaker, but if A fails and B works, morphology can still be batched by
     masking whichever robot is inactive.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_multi_articulation
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    from genmech.robots import get_robot_spec

    sharpa = get_robot_spec("sharpa_iiwa14")
    allegro = get_robot_spec("allegro_iiwa14")
    print(f"[probe] SHARPA  {sharpa.num_joints} joints")
    print(f"[probe] Allegro {allegro.num_joints} joints  "
          f"-> topologies differ by {sharpa.num_joints - allegro.num_joints}")

    app = AppLauncher(args).app

    import tempfile

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.from_files import UsdFileCfg
    from isaaclab.utils import configclass

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )
    from genmech.utils.paths import resolve as resolve_repo_path

    work = Path(tempfile.mkdtemp(prefix="genmech_multiart_"))
    usd_dir = work / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)

    usds = {}
    for spec in (sharpa, allegro):
        raw = _convert_urdf_to_usd(
            str(resolve_repo_path(spec.urdf_path)), usd_dir,
            fix_base=True, self_collision=False,
            joint_drive=_robot_joint_drive_cfg(),
        )
        usds[spec.name] = _bake_usd(
            raw, work / "baked", spec.name,
            props=dict(disable_gravity=True, max_depenetration_velocity=1000.0,
                       enabled_self_collisions=False,
                       solver_position_iterations=8, solver_velocity_iterations=0),
            apply_physx_articulation=True,
        )
    print(f"[probe] baked USDs for both robots")

    def art_cfg(spec, prim_path: str) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=UsdFileCfg(usd_path=usds[spec.name]),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=spec.base_pos, rot=spec.base_rot,
                joint_pos={**spec.arm_default_joint_pos,
                           **spec.hand_default_joint_pos},
                joint_vel={".*": 0.0},
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=list(spec.arm_joint_names),
                    stiffness=dict(spec.arm_stiffness),
                    damping=dict(spec.arm_damping)),
                "hand": ImplicitActuatorCfg(
                    joint_names_expr=list(spec.hand_joint_names),
                    stiffness=dict(spec.hand_stiffness),
                    damping=dict(spec.hand_damping),
                    armature=dict(spec.hand_armature)),
            },
        )

    n = args.num_envs
    half = n // 2
    results = {}

    def try_config(label: str, path_a: str, path_b: str) -> None:
        print(f"\n{'=' * 62}\n[probe] CONFIG {label}\n"
              f"[probe]   sharpa  -> {path_a}\n[probe]   allegro -> {path_b}")
        try:
            sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cuda:0"))

            @configclass
            class Cfg(InteractiveSceneCfg):
                sharpa_robot: ArticulationCfg = art_cfg(sharpa, path_a)
                allegro_robot: ArticulationCfg = art_cfg(allegro, path_b)

            scene = InteractiveScene(
                Cfg(num_envs=n, env_spacing=3.0,
                    replicate_physics=False, clone_in_fabric=False)
            )
            sim.reset()
            a: Articulation = scene["sharpa_robot"]
            b: Articulation = scene["allegro_robot"]
            for _ in range(3):
                sim.step()
                scene.update(1 / 120.0)
            print(f"[probe]   sharpa view : {a.num_instances} instances x "
                  f"{a.num_joints} joints")
            print(f"[probe]   allegro view: {b.num_instances} instances x "
                  f"{b.num_joints} joints")
            ok = (a.num_joints == sharpa.num_joints
                  and b.num_joints == allegro.num_joints)
            print(f"[probe]   RESULT: {'WORKS' if ok else 'joint counts wrong'}")
            results[label] = "works" if ok else "wrong-joint-counts"
            sim.clear_all_callbacks()
            sim.clear_instance()
        except Exception as exc:
            print(f"[probe]   RESULT: FAILED -- {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
            results[label] = f"failed: {type(exc).__name__}"
            try:
                SimulationContext.clear_instance()
            except Exception:
                pass

    # A: disjoint env subsets -- what a morphology-conditioned rollout wants.
    envs_a = "|".join(str(i) for i in range(half))
    envs_b = "|".join(str(i) for i in range(half, n))
    try_config("A: disjoint env subsets",
               f"/World/envs/env_({envs_a})/Robot",
               f"/World/envs/env_({envs_b})/Robot")

    print(f"\n{'=' * 62}")
    for label, outcome in results.items():
        print(f"[probe] {label:28s} -> {outcome}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
