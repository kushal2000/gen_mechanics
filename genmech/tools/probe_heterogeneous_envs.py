"""Can parallel envs hold DIFFERENT robots? Probe the actual constraint.

Motivation: a morphology-conditioned policy needs many morphologies in one
batched rollout. Retraining per candidate is hopeless -- one hand costs ~6
GPU-days here -- so the question is whether Isaac Lab can put distinct robots in
distinct envs of a single vectorised scene.

The constraint is in PhysX, not Isaac Lab: ``Articulation.num_joints`` reads
``root_physx_view.shared_metatype.dof_count``, one scalar for the whole view,
and likewise link_count / dof_names / link_names. Every buffer is
``(num_instances, num_joints)``. So one ArticulationView cannot span robots with
different joint counts or names.

That leaves a case worth testing, and it is the one that matters for parametric
morphology: **same topology, different geometry**. Identical joint names and
count, but different link lengths, masses, inertias, gains. If that works, a
parametric hand (finger lengths, joint axes, gains) can be randomised per env
exactly the way object geometry already is.

    OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python \\
        -m genmech.tools.probe_heterogeneous_envs
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher


def make_scaled_variant(src: Path, dst: Path, finger_scale: float) -> Path:
    """Same joints, same names, longer finger links.

    Scales the translation of every joint origin below the palm, which changes
    link geometry while leaving the kinematic topology -- and therefore
    shared_metatype -- untouched.
    """
    tree = ET.parse(src)
    root = tree.getroot()
    touched = 0
    for joint in root.findall("joint"):
        name = joint.get("name", "")
        if not any(f in name for f in ("index", "middle", "ring", "thumb")):
            continue
        origin = joint.find("origin")
        if origin is None:
            continue
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
        origin.set("xyz", " ".join(f"{v * finger_scale:.9g}" for v in xyz))
        touched += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"[probe] wrote {dst.name}: scaled {touched} finger joint origins "
          f"by {finger_scale}")
    return dst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--robot_spec", default="allegro_iiwa14")
    parser.add_argument("--finger_scale", type=float, default=1.35)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    from genmech.robots import get_robot_spec
    from genmech.utils.paths import resolve as resolve_repo_path

    spec = get_robot_spec(args.robot_spec)
    src = resolve_repo_path(spec.urdf_path)

    app = AppLauncher(args).app

    import tempfile

    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg

    from genmech.tasks.pose_reach.utils.scene_utils import (
        _bake_usd,
        _convert_urdf_to_usd,
        _robot_joint_drive_cfg,
    )

    work = Path(tempfile.mkdtemp(prefix="genmech_hetero_"))
    # The variant must sit BESIDE the original: its mesh filenames are relative
    # (../kuka_sharpa_description/..., allegro_meshes_mirrored/...) and resolve
    # against the URDF's own directory, so a temp-dir copy finds nothing.
    variants = [src, make_scaled_variant(src, src.parent / "_probe_variant_long.urdf",
                                         args.finger_scale)]

    usd_dir = work / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    usd_paths = []
    for i, urdf in enumerate(variants):
        raw = _convert_urdf_to_usd(str(urdf), usd_dir, fix_base=True,
                                   self_collision=True,
                                   joint_drive=_robot_joint_drive_cfg())
        baked = _bake_usd(raw, work / "baked", f"robot{i}",
                          props=dict(disable_gravity=True,
                                     max_depenetration_velocity=1000.0,
                                     enabled_self_collisions=False,
                                     solver_position_iterations=8,
                                     solver_velocity_iterations=0),
                          apply_physx_articulation=True)
        usd_paths.append(baked)
    print(f"[probe] baked {len(usd_paths)} distinct robot USDs")

    sim = SimulationContext(SimulationCfg(dt=1 / 120.0, device="cuda:0"))

    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass

    @configclass
    class ProbeSceneCfg(InteractiveSceneCfg):
        # Per-env distinct USDs need both flags off; same requirement the object
        # pool already relies on.
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=MultiUsdFileCfg(usd_path=usd_paths, random_choice=False),
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

    scene_cfg = ProbeSceneCfg(num_envs=args.num_envs, env_spacing=2.0,
                              replicate_physics=False, clone_in_fabric=False)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    print(f"\n[probe] view initialised")
    print(f"[probe]   num_instances {robot.num_instances}   "
          f"num_joints {robot.num_joints}   num_bodies {len(robot.data.body_names)}")

    # The decisive check: do the envs actually differ geometrically, or did the
    # cloner collapse them onto one template?
    for _ in range(3):
        sim.step()
        scene.update(1 / 120.0)

    tip = spec.fingertip_body_names[0]
    bid = robot.find_bodies(tip)[0][0]
    palm_id = robot.find_bodies(spec.palm_body_name)[0][0]
    pos = robot.data.body_pos_w[:, bid, :] - robot.data.body_pos_w[:, palm_id, :]
    reach = torch.linalg.norm(pos, dim=-1)
    print(f"\n[probe] palm->{tip} distance per env (m):")
    for i, r in enumerate(reach.tolist()):
        print(f"[probe]   env {i}: {r:.4f}   (variant {i % len(usd_paths)})")

    spread = float(reach.max() - reach.min())
    print(f"\n[probe] spread across envs: {spread * 100:.2f} cm")
    if spread > 1e-3:
        print("[probe] RESULT: envs hold GEOMETRICALLY DISTINCT robots in one "
              "ArticulationView.")
        print("[probe] Same topology (identical joint count and names) is the "
              "requirement; link geometry, mass and gains may vary per env.")
    else:
        print("[probe] RESULT: envs are identical -- the variants collapsed onto "
              "one template, so per-env morphology did NOT take effect.")

    for v in variants[1:]:
        v.unlink(missing_ok=True)   # generated beside the real assets; clean up

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
