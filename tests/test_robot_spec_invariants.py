"""Structural invariants every registered robot must satisfy.

Replaces simtoolreal's ``test_transfer_invariants.py``, five of whose seven
checks referenced ``isaacgymenvs`` (not ported) and were SHARPA-specific by
construction. These checks are stated against the spec instead, so they apply
unchanged to any hand added later.

This matters most for a hand with no pretrained checkpoint to fall back on:
SHARPA's correctness is pinned by ``test_sharpa_parity.py``, but a new hand has
no such anchor, and every failure mode here is otherwise silent — wrong gains,
a mis-ordered permutation, unmasked self-collisions, fingertip friction that
quietly reverts to the robot default.

    .venv_isaacsim/bin/python tests/test_robot_spec_invariants.py
    .venv_isaacsim/bin/python tests/test_robot_spec_invariants.py --robot_spec allegro_iiwa14
"""

from __future__ import annotations

import argparse
import os
import sys


def check_registry_offline() -> list[str]:
    """Spec-only checks that need no simulator.

    The arm-equality check is the mechanical enforcement of "the arm is held
    constant across hands" — the premise the whole comparison rests on
    (docs/methodology.md §1). A hand that brought its own arm gains or home pose
    would confound every result.
    """
    from genmech.robots import REGISTRY

    msgs = []
    assert REGISTRY, "robot registry is empty"

    by_arm: dict[str, list] = {}
    for spec in REGISTRY.values():
        spec.validate()  # cheap, but re-run so a mutated spec cannot slip through
        by_arm.setdefault(spec.arm_name, []).append(spec)

    for arm_name, specs in by_arm.items():
        ref = specs[0]
        for other in specs[1:]:
            for field in ("arm_joint_names", "arm_stiffness", "arm_damping",
                          "arm_default_joint_pos", "start_arm_higher_deltas",
                          "base_pos", "base_rot"):
                a, b = getattr(ref, field), getattr(other, field)
                assert a == b, (
                    f"arm {arm_name!r} differs between {ref.name!r} and "
                    f"{other.name!r} in {field}: {a} vs {b}. The arm must be "
                    f"identical across hands (docs/methodology.md §1)."
                )
        msgs.append(f"arm {arm_name!r}: identical across {[s.name for s in specs]}")

    return msgs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default=None,
                        help="Robot to check in sim. Default: every registered robot.")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--num_assets_per_type", type=int, default=1)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True

    # Offline checks first — they need no GPU, so a spec typo fails in seconds
    # rather than after a two-minute Kit boot.
    from genmech.robots import REGISTRY, get_robot_spec
    for msg in check_registry_offline():
        print(f"[spec] {msg}")

    names = [args.robot_spec] if args.robot_spec else sorted(REGISTRY)
    for n in names:
        get_robot_spec(n)  # fail early on a bad --robot_spec

    # ONE ENV PER PROCESS. Building a second env in this process wedges: the
    # first spec passes all its checks, then the run stops producing output
    # while still burning CPU, and never finishes. Reproduced in isolation on a
    # dedicated GPU, so it is neither contention nor a specific robot -- each
    # spec passes on its own (sharpa OK, allegro OK), and only the pair hangs.
    # It is the same constraint run_all.sh is built around: "Kit cannot be torn
    # down and re-created in-process."
    #
    # So when several specs are requested, re-exec this script once per spec
    # instead of looping in-process. Doing it here rather than in run_all.sh
    # keeps the runner generic and fixes every caller.
    if len(names) > 1:
        import subprocess

        passthrough = [a for a in sys.argv[1:] if a != "--robot_spec"]
        failures = []
        for n in names:
            print(f"\n[spec] === subprocess for {n} ===", flush=True)
            rc = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--robot_spec", n, *passthrough],
            ).returncode
            if rc != 0:
                failures.append((n, rc))
        if failures:
            raise AssertionError(f"spec checks failed: {failures}")
        print(f"\n[spec] all {len(names)} specs checked in separate processes")
        print("robot spec invariants OK")
        return

    print(f"[spec] checking in sim: {names}")

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import genmech.tasks  # noqa: F401  registers GenMech-PoseReach-Direct-v0
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg
    from genmech.tasks.pose_reach.utils.obs_utils import compute_obs_dim

    for name in names:
        spec = get_robot_spec(name)
        print(f"\n=== {name} ===")

        cfg = PoseReachEnvCfg()
        cfg.scene.num_envs = args.num_envs
        cfg.assets.num_assets_per_type = args.num_assets_per_type
        cfg.assets.robot_spec = name
        # A noise-free reset, so the home-pose check compares against the spec.
        cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        cfg.reset.reset_dof_vel_random_interval = 0.0
        # Observation DR off: the quaternion-convention check reads the object
        # rotation straight out of the obs vector and compares it to the live
        # articulation, and object-state DR adds 5 deg of rotation noise plus a
        # 10-step delay, which is far larger than the difference being tested.
        cfg.domain_randomization.use_obs_delay = False
        cfg.domain_randomization.use_object_state_delay_noise = False
        cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0

        env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        inner._replay_target_lab_order = None

        # 1. The URDF's joints are exactly the spec's joints.
        lab_names = list(inner.robot.data.joint_names)
        assert set(lab_names) == set(spec.joint_names_canonical), (
            f"{name}: URDF joints != spec joints"
        )
        print(f"  [1] joint set matches ({len(lab_names)} joints)")

        # 2. Canonical<->Lab permutations are mutually inverse bijections.
        c2l = inner._perm_canon_to_lab.cpu()
        l2c = inner._perm_lab_to_canon.cpu()
        n = spec.num_joints
        assert sorted(c2l.tolist()) == list(range(n)), f"{name}: c2l not a permutation"
        assert sorted(l2c.tolist()) == list(range(n)), f"{name}: l2c not a permutation"
        assert torch.equal(c2l[l2c], torch.arange(n)), f"{name}: perms not inverse"
        assert [lab_names[i] for i in l2c.tolist()] == list(spec.joint_names_canonical), (
            f"{name}: l2c does not map Lab order onto canonical order"
        )
        print("  [2] permutations are mutually inverse bijections")

        # 3. Live PhysX gains equal the spec tables. Isaac Lab silently falls
        #    back to defaults for any joint an actuator group misses.
        stiffness = inner.robot.data.joint_stiffness[0]
        damping = inner.robot.data.joint_damping[0]
        armature = inner.robot.data.joint_armature[0]
        want_k = {**spec.arm_stiffness, **spec.hand_stiffness}
        want_d = {**spec.arm_damping, **spec.hand_damping}
        checked = 0
        for i, jname in enumerate(lab_names):
            assert abs(float(stiffness[i]) - want_k[jname]) < 1e-4, (
                f"{name}: {jname} stiffness {float(stiffness[i])} != {want_k[jname]}"
            )
            assert abs(float(damping[i]) - want_d[jname]) < 1e-4, (
                f"{name}: {jname} damping {float(damping[i])} != {want_d[jname]}"
            )
            if jname in spec.hand_armature:
                assert abs(float(armature[i]) - spec.hand_armature[jname]) < 1e-6, (
                    f"{name}: {jname} armature {float(armature[i])} != "
                    f"{spec.hand_armature[jname]}"
                )
            checked += 1
        assert checked == spec.num_joints
        print(f"  [3] live gains/armature match the spec for all {checked} joints")

        # 4. Arm and hand ids partition the joint vector, ascending.
        arm_ids, hand_ids = inner._arm_joint_ids, inner._hand_joint_ids
        assert sorted(arm_ids + hand_ids) == list(range(n)), (
            f"{name}: arm+hand ids do not partition range({n})"
        )
        assert arm_ids == sorted(arm_ids) and hand_ids == sorted(hand_ids), (
            f"{name}: joint ids must be ascending so reward summation order is "
            f"independent of how the spec lists joints"
        )
        print(f"  [4] {len(arm_ids)} arm + {len(hand_ids)} hand ids partition the joints")

        # 5. Fingertip bodies: right count, and in spec order (index i must be
        #    the same finger as fingertip_offsets[i] and obs column i).
        #    The ids address SLOTS, not active fingers: a generated hand pads to
        #    the template's 5 finger slots and marks the ghosted ones invalid, so
        #    designs with different finger counts share one observation layout.
        #    For a fixed hand the two lists are identical and this is the same
        #    assertion as before.
        body_names = list(inner.robot.data.body_names)
        got_tips = [body_names[i] for i in inner._fingertip_body_ids]
        assert got_tips == list(spec.fingertip_slots), (
            f"{name}: fingertip bodies {got_tips} != spec slot order "
            f"{list(spec.fingertip_slots)}"
        )
        # ...and the mask must select exactly the ACTIVE fingertips, in order.
        mask = inner._fingertip_mask[0].tolist()
        active = [b for b, ok in zip(got_tips, mask) if ok]
        assert active == list(spec.fingertip_body_names), (
            f"{name}: mask selects {active}, spec declares "
            f"{list(spec.fingertip_body_names)}"
        )
        print(f"  [5] {len(got_tips)} fingertip slots in spec order, "
              f"{len(active)} active")

        # 6. Derived dims.
        assert int(inner.cfg.action_space) == spec.num_joints
        assert inner.cfg.observation_space == compute_obs_dim(cfg.obs.obs_list, spec)
        assert inner.cfg.state_space == compute_obs_dim(cfg.obs.state_list, spec)
        print(f"  [6] dims derived: act={int(inner.cfg.action_space)} "
              f"obs={inner.cfg.observation_space} state={inner.cfg.state_space}")

        # 7. Home pose after a noise-free reset.
        env.reset()
        joint_pos = inner.robot.data.joint_pos[0]
        want_arm = spec.arm_default_joint_pos_resolved(
            start_arm_higher=getattr(cfg.reset, "start_arm_higher", False)
        )
        for i, jname in enumerate(lab_names):
            want = want_arm.get(jname, spec.hand_default_joint_pos.get(jname))
            assert want is not None, f"{name}: {jname} has no default pose"
            assert abs(float(joint_pos[i]) - want) < 1e-3, (
                f"{name}: {jname} reset to {float(joint_pos[i])}, spec says {want}"
            )
        print("  [7] noise-free reset lands on the spec's home pose")

        # 8. object_rot is xyzw. Isaac Lab is wxyz everywhere else; the policy
        #    was trained against the legacy xyzw convention, and swapping them
        #    is silent (both are unit 4-vectors).
        obs, _ = env.reset()
        idx = list(cfg.obs.obs_list).index("object_rot")
        offset = compute_obs_dim(list(cfg.obs.obs_list)[:idx], spec)
        quat = obs["policy"][0, offset:offset + 4].tolist()
        w, x, y, z = inner.object.data.root_quat_w[0].tolist()  # Isaac Lab is wxyz
        err_xyzw = max(abs(a - b) for a, b in zip(quat, (x, y, z, w)))
        err_wxyz = max(abs(a - b) for a, b in zip(quat, (w, x, y, z)))
        assert err_xyzw < err_wxyz and err_xyzw < 1e-4, (
            f"{name}: object_rot is not xyzw (err_xyzw={err_xyzw:.2e}, "
            f"err_wxyz={err_wxyz:.2e}). Isaac Lab is wxyz everywhere else, but "
            f"the policy was trained against the legacy xyzw convention and the "
            f"two are indistinguishable at runtime -- both are unit 4-vectors."
        )
        print(f"  [8] object_rot observation is xyzw (err {err_xyzw:.1e} vs "
              f"wxyz {err_wxyz:.1e})")

        env.close()

    print("\n[spec] robot spec invariants OK")
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
