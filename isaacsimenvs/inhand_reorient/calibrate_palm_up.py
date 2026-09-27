r"""Palm-up base-rotation calibration for a hand-only spec (Phase 1b gap 2).

The env currently mounts every hand at ``HandOnlySpec.base_rot`` = identity,
which is only "palm up" by accident. This script finds a better one, per
hand, as DATA (``hand_calibration.json``), not per-hand code: it tries a
fixed set of candidate base rotations (which LOCAL axis of the hand-only
URDF's root/palm link ends up pointing along world +z, each with a roll),
holds the hand at its default joint targets with zero action, drops a cube
just above the palm's own geometric extent, and scores each candidate by how
long/close the cube stays near the palm under gravity.

One Kit process, one scene: every candidate (x ``--repeats`` copies, for a
little robustness to the position jitter each copy gets) is a DIFFERENT
group of envs in the SAME InteractiveScene, so this never needs more than one
Kit boot regardless of how many candidates are tried.

Mechanism: the articulation is spawned once (identity pose, like any other
InHandReorient boot); each candidate group's root pose is then overwritten
with ``Articulation.write_root_pose_to_sim`` (Isaac Lab's documented way to
reposition a FIXED-base robot -- e.g. base-pose domain randomization for
table-mounted arms) rather than re-authoring the scene per candidate.

Run (single Kit process; ~1-2 min boot + a few seconds of physics):
    .venv_isaacsim/bin/python isaacsimenvs/inhand_reorient/calibrate_palm_up.py \
        --hand sharpa [--full] [--repeats 8] [--seconds 2.0]

Writes/merges into ``hand_calibration.json`` next to this file.
"""

from __future__ import annotations

import argparse
import datetime as _dt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hand", required=True, help="manifest hand id, e.g. sharpa, dclaw, allegro_right")
    p.add_argument("--full", action="store_true", help="24 candidates (6 axes x 4 rolls) instead of 6")
    p.add_argument("--repeats", type=int, default=8, help="envs per candidate")
    p.add_argument("--seconds", type=float, default=2.0, help="sim seconds to hold each candidate")
    p.add_argument("--settle-seconds", type=float, default=0.15,
                   help="physics settle after the root-pose overwrite, before dropping the cube")
    p.add_argument("--threshold-m", type=float, default=0.08,
                   help="object-to-palm-origin distance under which the object counts as 'on the hand'")
    p.add_argument("--score-window-frac", type=float, default=0.5,
                   help="score over the trailing fraction of steps (lets it settle first)")
    p.add_argument("--object-size-m", type=float, default=0.055)
    p.add_argument("--spawn-margin-m", type=float, default=0.02,
                   help="clearance between the hand's highest body origin and the cube surface")
    p.add_argument("--position-jitter-m", type=float, default=0.01)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # isaaclab.* is only safe to import once Kit is up (see the package's own
    # tests/test_config_sanity.py docstring for how this was confirmed).
    import torch

    from isaacsimenvs.inhand_reorient import palm_calibration as pc
    from isaacsimenvs.inhand_reorient.env import InHandReorientEnv
    from isaacsimenvs.inhand_reorient.env_cfg import InHandReorientEnvCfg

    candidates = pc.candidate_rotations(full=args.full)
    n_groups, repeats = len(candidates), args.repeats
    n_envs = n_groups * repeats
    print(f"[calibrate_palm_up] hand={args.hand} {n_groups} candidates x {repeats} repeats "
          f"= {n_envs} envs", flush=True)

    cfg = InHandReorientEnvCfg()
    cfg.assets.hand_id = args.hand
    cfg.assets.object_size_m = args.object_size_m
    cfg.scene.num_envs = n_envs
    cfg.sim.device = args.device

    env = InHandReorientEnv(cfg=cfg)
    env.reset()
    device = env.device
    dt = env.sim.get_physics_dt()

    group_of_env = torch.arange(n_envs, device=device) // repeats
    quat_table = torch.tensor([c["quat_wxyz"] for c in candidates], device=device, dtype=torch.float32)
    quat_per_env = quat_table[group_of_env]  # (n_envs, 4)

    base_pos_local = torch.tensor(env.hand_spec.base_pos, device=device, dtype=torch.float32)
    world_pos = env.scene.env_origins + base_pos_local.expand(n_envs, 3)
    root_pose = torch.cat([world_pos, quat_per_env], dim=-1)
    env.robot.write_root_pose_to_sim(root_pose)
    default_pos = env.robot.data.default_joint_pos.clone()
    env.robot.write_joint_state_to_sim(default_pos, torch.zeros_like(default_pos))

    def _hold_step() -> None:
        env.robot.set_joint_position_target(default_pos)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=dt)

    n_settle = max(1, int(round(args.settle_seconds / dt)))
    for _ in range(n_settle):
        _hold_step()

    palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx].clone()
    moved = (palm_pos_w - world_pos).norm(dim=-1)
    if moved.max().item() > 1e-3:
        print(f"[calibrate_palm_up] WARNING: palm did not land at the written root pose for "
              f"every env (max offset {moved.max().item():.4f} m) -- write_root_pose_to_sim may "
              f"not be repositioning this fixed-base articulation as expected.", flush=True)

    # Analytic-ish spawn point: above the hand's own highest body ORIGIN
    # (a cheap proxy for the palm's collision/visual bbox top -- exact per
    # the plan would trace real geometry, but body origins already track the
    # hand's spatial envelope well enough to place a cube "above the palm"
    # without needing an extra USD bbox query), directly over the palm's own
    # (x, y), offset along WORLD +z by the object's half-size + clearance.
    max_body_z = env.robot.data.body_pos_w[:, :, 2].max(dim=1).values
    offset_z = (max_body_z - palm_pos_w[:, 2]) + args.spawn_margin_m + args.object_size_m / 2.0
    spawn_xy = palm_pos_w[:, :2]
    spawn_pos = torch.cat([spawn_xy, (palm_pos_w[:, 2] + offset_z).unsqueeze(-1)], dim=-1)

    jitter = (torch.rand(n_envs, 3, device=device) * 2.0 - 1.0) * args.position_jitter_m
    obj_pos = spawn_pos + jitter
    obj_quat = torch.zeros(n_envs, 4, device=device)
    obj_quat[:, 0] = 1.0
    obj_state = torch.cat([obj_pos, obj_quat, torch.zeros(n_envs, 6, device=device)], dim=-1)
    env.object.write_root_state_to_sim(obj_state)
    for _ in range(2):  # let the write land before the scoring window starts
        env.scene.update(dt=dt)

    n_steps = max(1, int(round(args.seconds / dt)))
    n_score = max(1, int(round(n_steps * args.score_window_frac)))
    on_hand_counts = torch.zeros(n_envs, device=device)
    final_dist = None
    for step in range(n_steps):
        _hold_step()
        dist = (env.object.data.root_pos_w - palm_pos_w).norm(dim=-1)
        if step >= n_steps - n_score:
            on_hand_counts += (dist < args.threshold_m).float()
        final_dist = dist

    score_per_env = on_hand_counts / n_score
    score_per_group = score_per_env.view(n_groups, repeats).mean(dim=1)
    final_dist_per_group = final_dist.view(n_groups, repeats).mean(dim=1)

    score_table = []
    for gi, cand in enumerate(candidates):
        score_table.append({
            "axis": cand["axis"], "roll_deg": cand["roll_deg"],
            "score": float(score_per_group[gi]),
            "final_dist_m": float(final_dist_per_group[gi]),
        })
    score_table.sort(key=lambda e: -e["score"])
    for e in score_table:
        print(f"[calibrate_palm_up]   axis={e['axis']:>2} roll={e['roll_deg']:>5.0f}deg  "
              f"score={e['score']:.3f}  final_dist={e['final_dist_m']:.4f}m", flush=True)

    winner_gi = int(score_per_group.argmax())
    winner = candidates[winner_gi]
    winner_quat = torch.tensor(winner["quat_wxyz"], dtype=torch.float32)

    # Fold the (analytic, pre-jitter) world spawn offset back into the
    # palm's own local frame under the WINNING rotation, so scene_utils can
    # hand it straight to env.cfg.reset.object_spawn_offset unchanged.
    import numpy as np

    world_offset = np.array([0.0, 0.0, float(offset_z.view(n_groups, repeats)[winner_gi].mean())])
    local_offset = pc.quat_apply(pc.quat_inv(np.array(winner["quat_wxyz"])), world_offset)

    entry = {
        "base_rot": [float(v) for v in winner["quat_wxyz"]],
        "axis": winner["axis"],
        "roll_deg": winner["roll_deg"],
        "spawn_offset_local": [float(v) for v in local_offset],
        "score": float(score_per_group[winner_gi]),
        "final_dist_m": float(final_dist_per_group[winner_gi]),
        "method": (f"calibrate_palm_up.py: {n_groups} candidates (full={args.full}), "
                   f"repeats={repeats}, held {args.seconds:.1f}s @ {1.0 / dt:.0f} Hz zero-action "
                   f"default-joint-target, scored over the trailing "
                   f"{args.score_window_frac:.0%} of steps at threshold={args.threshold_m} m"),
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": pc.git_sha(),
        "score_table": score_table,
    }
    pc.save_calibration({args.hand: entry})
    print(f"[calibrate_palm_up] hand={args.hand} WINNER axis={winner['axis']} "
          f"roll={winner['roll_deg']:.0f}deg score={entry['score']:.3f} "
          f"base_rot={entry['base_rot']} spawn_offset_local={entry['spawn_offset_local']} "
          f"-> written to {pc.CALIB_PATH}", flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
