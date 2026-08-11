"""Can this hand's PD gains hold a tool against gravity?

**STATUS: this test does not currently produce a grasp, and its hold rate must
not be quoted.** It is kept for the contact gate below and as a record of what
did not work; the question it was written to answer was settled analytically
instead -- see the gain comparison in genmech/robots/allegro_iiwa14.py.

Three attempts, none of which got the fingers onto the object:

  1. Object placed at ``palm_center_offset``. That is the observation frame's
     origin, not a grasp point -- it sits behind the knuckles, so the fingers
     closed on air and the object stayed put by wedging against the palm. This
     produced a confident "94% vs 73%" hold-rate difference between the hands
     that was pure artifact; at 32 envs both hands gave 91%.
  2. Object moved to the fingertip centroid, re-seated every 20 steps. It was in
     free fall between re-seats and had dropped ~18 cm by the time the fingers
     arrived. 16% contact.
  3. Object pinned every step at the OPEN-hand fingertip centroid. As the
     fingers curl inward the tips move away from that point, so they close
     behind it. 0% contact.

The lesson worth keeping is the contact gate: it refuses to report a hold rate
unless fingertips are demonstrably near the object. Without it, attempts 2 and 3
would each have emitted another plausible-looking number. Any future version
needs the object where the fingers *end up*, not where they start -- e.g. record
the fingertip centroid at the closed pose first, then place the object there.

A hand whose gains are too soft drops objects, and a training run on such a hand
would look like weak *hardware* while actually reporting a tuning artifact. That
is exactly the confound docs/methodology.md is built to avoid, so this runs
before any hand's training is trusted.

The check is deliberately policy-free: the object is teleported into the grasp
region at the palm centre, the fingers are driven closed, and the object's fall
is measured over a fixed horizon. No learning, no reward -- just whether the
closed hand holds.

Run it for **both** hands. SHARPA has a working pretrained policy, so its number
is the bar; Allegro is only suspect relative to it.

Motivating concern for Allegro: its gains (stiffness 3.0, damping 0.1) come from
Isaac Lab's KUKA_ALLEGRO_CFG, which pairs them with
``solver_position_iteration_count=32``. This task runs 8.

    .venv_isaacsim/bin/python tests/test_grasp_hold.py --robot_spec sharpa_iiwa14
    .venv_isaacsim/bin/python tests/test_grasp_hold.py --robot_spec allegro_iiwa14
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_spec", default="sharpa_iiwa14")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_assets_per_type", type=int, default=3)
    parser.add_argument("--settle_steps", type=int, default=60,
                        help="Steps spent closing the fingers before measuring.")
    parser.add_argument("--hold_steps", type=int, default=240,
                        help="Steps over which the drop is measured (4 s at 60 Hz).")
    parser.add_argument("--max_drop_m", type=float, default=0.02,
                        help="Fail threshold on median drop.")
    parser.add_argument("--close_frac", type=float, default=0.6,
                        help="How far toward each hand joint's closing limit to drive.")
    parser.add_argument("--video", default="",
                        help="Directory to write an mp4 of the grasp. Uses the "
                             "Lab-canonical render_mode + RecordVideo path, and "
                             "forces --enable_cameras.")
    parser.add_argument("--cam_offset", type=float, nargs=3, default=(0.32, -0.26, 0.14),
                        help="Camera position relative to the palm centre. The "
                             "target is computed from the spec by FK, so the shot "
                             "frames the hand rather than the whole scene.")
    parser.add_argument("--contact_m", type=float, default=0.05,
                        help="Fingertip-to-object distance counted as contact.")
    parser.add_argument("--min_contacts", type=int, default=2,
                        help="Fingertips that must be in contact for the hold rate "
                             "to mean anything.")
    parser.add_argument("--solver_iters", type=int, default=0,
                        help="Override PhysX position-iteration count (0 = task "
                             "default, 8). Isaac Lab ships Allegro's gains with 32, "
                             "so this separates 'soft gains' from 'under-solved "
                             "contact'.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video:
        # Rendering needs cameras even when running headless.
        args.enable_cameras = True

    from genmech.robots import get_robot_spec
    spec = get_robot_spec(args.robot_spec)

    app = AppLauncher(args).app

    import gymnasium as gym
    import torch

    import genmech.tasks  # noqa: F401  registers GenMech-PoseReach-Direct-v0
    from genmech.tasks.pose_reach.env_cfg import PoseReachEnvCfg

    cfg = PoseReachEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type
    cfg.assets.robot_spec = args.robot_spec
    # Deterministic: the question is about gains, not about initial-state spread.
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    if args.solver_iters:
        cfg.sim.physx.min_position_iteration_count = args.solver_iters
        cfg.sim.physx.max_position_iteration_count = args.solver_iters

    if args.video:
        # Aim at the palm centre, computed by FK from the spec, so the hand fills
        # the frame. The scene default frames all envs at once, which at this
        # scale makes the hand a few pixels across and the grasp invisible.
        import numpy as np
        import yourdfpy

        from genmech.utils.paths import resolve as _resolve

        _u = yourdfpy.URDF.load(str(_resolve(spec.urdf_path)))
        _u.update_cfg({**spec.arm_default_joint_pos, **spec.hand_default_joint_pos})
        _T = _u.get_transform(spec.palm_body_name, _u.base_link)
        palm_w = (np.array(spec.base_pos) + _T[:3, 3]
                  + _T[:3, :3] @ np.array(spec.palm_center_offset))
        cfg.viewer.lookat = tuple(float(v) for v in palm_w)
        cfg.viewer.eye = tuple(float(v) for v in (palm_w + np.array(args.cam_offset)))
        print(f"[grasp] camera eye {np.round(cfg.viewer.eye,3).tolist()} "
              f"-> palm {np.round(palm_w,3).tolist()}")

    env = gym.make("GenMech-PoseReach-Direct-v0", cfg=cfg,
                   render_mode="rgb_array" if args.video else None)
    if args.video:
        from pathlib import Path

        Path(args.video).mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video,
            step_trigger=lambda step: step == 0,
            video_length=args.settle_steps + args.hold_steps,
            name_prefix=f"grasp_{args.robot_spec}",
            disable_logger=True,
        )
        print(f"[grasp] recording {args.settle_steps + args.hold_steps} frames "
              f"-> {args.video}")
    inner = env.unwrapped
    inner._replay_target_lab_order = None
    env.reset()

    n_act = int(inner.cfg.action_space)
    dev = inner.device
    N = args.num_envs

    # Actions are canonical order; hand joints follow the arm ones. Hand actions
    # are an absolute rescale of [-1, 1] onto the joint range, so a constant
    # positive value drives the fingers toward their closing limit while the arm
    # (a velocity-delta accumulator) is left at zero and holds its pose.
    action = torch.zeros((N, n_act), device=dev)
    action[:, spec.num_arm_joints:] = 2.0 * args.close_frac - 1.0

    mass = float(inner._object_mass.mean().item())
    iters = args.solver_iters or cfg.sim.physx.max_position_iteration_count
    print(f"[grasp] {spec.name}: {spec.num_hand_joints} hand DoF, "
          f"{spec.num_fingertips} fingertips, mean object mass {mass * 1000:.0f} g, "
          f"solver position iters {iters}")

    def grasp_centre() -> torch.Tensor:
        """Centroid of the fingertip pads, i.e. where the fingers converge.

        The first version of this test used palm_center_offset instead. That is
        the observation frame's origin, not a grasp point: it sits at z = 0.16 in
        the palm frame while the knuckles are at 0.182 and the fingertips at
        ~0.25, so the object was being placed BEHIND the knuckles, inside the
        palm. The fingers then closed on air and the object stayed put only
        because it was wedged -- which produced a confident-looking hold rate
        that measured nothing about grasping.
        """
        from isaaclab.utils.math import quat_apply

        ft = inner.robot.data.body_state_w[:, inner._fingertip_body_ids, :]
        offs = inner._fingertip_offsets.unsqueeze(0).expand(N, -1, 3)
        pads = ft[:, :, 0:3] + quat_apply(
            ft[:, :, 3:7].reshape(-1, 4), offs.reshape(-1, 3)
        ).reshape(N, -1, 3)
        return pads.mean(dim=1)

    def place_object_in_hand() -> None:
        """Teleport the object into the finger enclosure with zero velocity."""
        root = torch.zeros((N, 13), device=dev)
        root[:, 0:3] = grasp_centre()
        root[:, 3] = 1.0  # identity wxyz
        inner.object.write_root_state_to_sim(root)

    # Hold the object effectively kinematic while the fingers close: pin it to
    # ONE fixed point, re-written every step, so the fingers converge on a
    # stationary target and then release it.
    #
    # Re-seating only every 20 steps (the previous version) left the object in
    # free fall between re-seats, so it had dropped ~18 cm by the time the
    # fingers arrived and they closed on empty space. Pinning every step is the
    # difference between "the hand closed around an object" and "the hand closed".
    target = grasp_centre().clone()
    pinned = torch.zeros((N, 13), device=dev)
    pinned[:, 0:3] = target
    pinned[:, 3] = 1.0  # identity wxyz
    for _ in range(args.settle_steps):
        inner.object.write_root_state_to_sim(pinned)
        env.step(action)

    # Diagnostic: is the object actually being gripped, or merely resting?
    # inner._curr_fingertip_distances is fingertip-to-object-centre distance, so
    # a real grasp shows several fingertips within roughly the object's own
    # half-extent. Large values everywhere mean the fingers closed on air and any
    # "hold" is the object wedged against the palm -- which would make the whole
    # comparison meaningless.
    d = inner._curr_fingertip_distances
    per_tip = d.mean(dim=0).cpu()
    print(f"[grasp]   fingertip-to-object distance at end of close (cm): "
          f"{[round(float(v) * 100, 1) for v in per_tip]}")
    n_contact = (d < args.contact_m).float().sum(dim=-1)
    print(f"[grasp]   closest fingertip {float(d.min(dim=-1).values.mean()) * 100:.1f} cm, "
          f"tips within {args.contact_m * 100:.0f} cm: {float(n_contact.mean()):.1f}"
          f" of {spec.num_fingertips}")

    # Gate the result on contact. Without this the test happily reports a hold
    # rate for a hand that never touched the object -- which is exactly what it
    # did before, giving a 94%-vs-73% "hardware difference" that was pure
    # artifact. A hold rate is only meaningful if something is being held.
    grasping = (n_contact >= args.min_contacts).float().mean()
    print(f"[grasp]   envs with >= {args.min_contacts} fingertips in contact: "
          f"{float(grasping):.0%}")
    if float(grasping) < 0.5:
        raise AssertionError(
            f"{spec.name}: only {float(grasping):.0%} of envs have "
            f">= {args.min_contacts} fingertips within {args.contact_m * 100:.0f} cm "
            f"of the object. The fingers are closing on air, so any drop measured "
            f"here reflects the object resting or wedged, not a grasp. Fix the "
            f"placement or closure before reading anything into the hold rate."
        )

    z_start = (inner.object.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]).clone()
    for _ in range(args.hold_steps):
        env.step(action)
    z_end = inner.object.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]

    drop = (z_start - z_end).cpu()
    median = float(drop.median())
    held = float((drop < args.max_drop_m).float().mean())

    print(f"[grasp] over {args.hold_steps} steps ({args.hold_steps / 60:.1f} s):")
    print(f"[grasp]   median drop {median * 100:.2f} cm   "
          f"mean {float(drop.mean()) * 100:.2f} cm   max {float(drop.max()) * 100:.2f} cm")
    print(f"[grasp]   held (< {args.max_drop_m * 100:.0f} cm): {held:.0%} of {N} envs")

    if median > args.max_drop_m:
        raise AssertionError(
            f"{spec.name}: median drop {median * 100:.2f} cm exceeds "
            f"{args.max_drop_m * 100:.0f} cm. These gains cannot hold the object; "
            f"a training run on this hand would report a tuning artifact as weak "
            f"hardware. Compare against the other hand before changing anything."
        )

    print("[grasp] grasp hold test OK")

    # RecordVideo writes the mp4 in close(). The os._exit(0) below skips normal
    # interpreter teardown -- it exists because Kit hangs on shutdown -- so
    # without an explicit close here the file is never flushed and the run
    # completes leaving an empty directory.
    if args.video:
        env.close()
        from pathlib import Path

        made = sorted(Path(args.video).glob("*.mp4"))
        print(f"[grasp] wrote {len(made)} video(s): "
              f"{[str(m) for m in made]}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
