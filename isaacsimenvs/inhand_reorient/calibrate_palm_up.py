r"""Palm-up base-rotation calibration for a hand-only spec (Phase 1b gap 2).

The env currently mounts every hand at ``HandOnlySpec.base_rot`` = identity,
which is only "palm up" by accident. This script finds a better one, per
hand, as DATA (``hand_calibration.json``), not per-hand code: it tries a
fixed set of candidate base rotations (which LOCAL axis of the hand-only
URDF's root/palm link ends up pointing along world +z, each with a roll),
holds the hand at its default joint targets with zero action, drops a cube
onto the FINGERTIPS' OWN centroid (not just "above the palm's own geometric
extent" -- see below), and scores each candidate by how long/close the cube
stays there AND whether at least 2 fingertips stay within reach of it.

Both stability and reach are scored (as of I24's priority check) because
distance-to-the-PALM-ORIGIN alone cannot tell a cupped hold apart from the
cube resting stably on some OTHER flat part of the hand the fingers can't
reach (SHARPA's first calibration did exactly that: winning candidate scored
stability 1.0 while sitting 0.09-0.19 m from every fingertip -- confirmed
unreachable in a 300-config random joint sweep with the cube frozen in
place). Spawning at the fingertip centroid rather than above the tallest
body origin also matters on its own: the tallest body is whatever part of
the hand happens to be highest in WORLD z under a given candidate rotation,
which is not necessarily anywhere near the fingers (e.g. it was the wrist/
base for SHARPA's "-z up" candidate, whose fingers pointed away from it).

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
                   help="object-to-spawn-point distance under which the object counts as stable")
    p.add_argument("--reach-threshold-m", type=float, default=0.05,
                   help="object-center-to-fingertip distance under which a fingertip counts as "
                        "'in reach'; a candidate needs >=2 in reach to score, not just stability "
                        "(I24 priority check: proximity to the palm ORIGIN alone can't tell a "
                        "cupped hold apart from the cube resting on an unrelated flat face)")
    p.add_argument("--score-window-frac", type=float, default=0.5,
                   help="score over the trailing fraction of steps (lets it settle first)")
    p.add_argument("--object-size-m", type=float, default=0.055)
    p.add_argument("--spawn-margin-m", type=float, default=0.02,
                   help="clearance between the hand's highest body origin and the cube surface")
    p.add_argument("--position-jitter-m", type=float, default=0.01)
    p.add_argument("--curl-fracs", default="0.0,0.35,0.6",
                   help="comma list of joint-curl fractions to also grid-search (I24 priority "
                        "check): held pose per joint = lower + frac*(upper-lower), same frac for "
                        "every joint. 0.0 = the articulation's own default (uncurled/flat for "
                        "every hand in this repo so far); the 0 lower bound of a typical flexion "
                        "joint here means positive angle = curling INTO the hand, so a moderate "
                        "positive frac searches for an actual cupped hold instead of assuming the "
                        "flat default touches the object at all. Ignored when --curl-profiles "
                        "is given.")
    p.add_argument("--curl-profiles", default="",
                   help="I25 dclaw retry: comma list of candidate PROFILES, each a '/'-separated "
                        "list of per-position curl fractions tiled cyclically across the joint "
                        "declaration order -- e.g. '0.0/0.7/0.9' on dclaw's regular "
                        "joint_f{1,2,3}_{0,1,2} order curls each finger's base/roll joint (the 0 "
                        "position) at 0.0 and its distal joint (the 2 position) hardest, instead "
                        "of one uniform fraction for the whole hand. Overrides --curl-fracs when "
                        "non-empty; each profile is one more candidate group, same as one more "
                        "curl_frac value.")
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
    if args.curl_profiles:
        # list[list[float]]; each a per-position profile tiled cyclically
        # over the joint list below (I25).
        curl_specs: list = [
            [float(v) for v in prof.split("/")] for prof in args.curl_profiles.split(",")
        ]
    else:
        curl_specs = [float(v) for v in args.curl_fracs.split(",")]  # list[float], uniform
    # Every (axis candidate, curl spec) pair is its own group of envs -- same
    # one-Kit-process multiplexing trick as the axis search alone used.
    combos = [(cand, cf) for cand in candidates for cf in curl_specs]
    n_groups, repeats = len(combos), args.repeats
    n_envs = n_groups * repeats
    print(f"[calibrate_palm_up] hand={args.hand} {len(candidates)} axis candidates x "
          f"{len(curl_specs)} curl specs x {repeats} repeats = {n_groups} groups, "
          f"{n_envs} envs", flush=True)

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
    quat_table = torch.tensor([c["quat_wxyz"] for c, _cf in combos], device=device, dtype=torch.float32)
    quat_per_env = quat_table[group_of_env]  # (n_envs, 4)

    base_pos_local = torch.tensor(env.hand_spec.base_pos, device=device, dtype=torch.float32)
    world_pos = env.scene.env_origins + base_pos_local.expand(n_envs, 3)
    root_pose = torch.cat([world_pos, quat_per_env], dim=-1)
    env.robot.write_root_pose_to_sim(root_pose)

    lower0 = env.robot.data.soft_joint_pos_limits[0, :, 0]
    upper0 = env.robot.data.soft_joint_pos_limits[0, :, 1]

    def _curl_target(cf) -> torch.Tensor:
        """Per-joint held target = lower + frac*(upper-lower). ``cf`` is
        either one float (uniform, the original behaviour) or a per-position
        profile (I25) tiled cyclically across the joint declaration order --
        see --curl-profiles' help above."""
        n = lower0.shape[0]
        if isinstance(cf, (int, float)):
            frac = torch.full((n,), float(cf), device=lower0.device, dtype=lower0.dtype)
        else:
            reps = -(-n // len(cf))  # ceil div
            frac = torch.tensor((list(cf) * reps)[:n], device=lower0.device, dtype=lower0.dtype)
        return lower0 + frac * (upper0 - lower0)

    curl_table = torch.stack([_curl_target(cf) for _cand, cf in combos])  # (n_groups, j)
    default_pos = curl_table[group_of_env]  # (n_envs, j) -- per-env held target
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

    # Spawn point: over the FINGERTIPS' OWN centroid (pushed a little further
    # outward along the palm->fingertip-centroid direction), NOT "above the
    # hand's highest body origin" -- the latter can be any part of the hand
    # (e.g. the wrist/base), and the old scoring below (distance to the PALM
    # ORIGIN, not to the spawn point) could not tell "resting stably under
    # the fingers" apart from "resting stably on some unrelated flat face
    # that happens to be near the palm origin too". Confirmed on SHARPA
    # (I24 priority check): the previous winning "-z up" candidate scored
    # 1.0 while every fingertip stayed 0.09-0.19 m from the cube at rest, and
    # a 300-config random joint sweep with the cube frozen at that spawn
    # point never got any fingertip within 3 cm of it. Reachability is now
    # scored directly (``reach_score`` below), not just assumed from stability.
    tip_pos_w = env.robot.data.body_pos_w[:, env.fingertip_body_idx, :]  # (n_envs, k, 3)
    tip_centroid_w = tip_pos_w.mean(dim=1)
    direction = torch.nn.functional.normalize(tip_centroid_w - palm_pos_w, dim=-1)
    spawn_pos = tip_centroid_w + direction * (args.spawn_margin_m + args.object_size_m / 2.0)

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
    stable_counts = torch.zeros(n_envs, device=device)
    reach_counts = torch.zeros(n_envs, device=device)
    final_dist = None
    for step in range(n_steps):
        _hold_step()
        dist_to_spawn = (env.object.data.root_pos_w - spawn_pos).norm(dim=-1)
        tip_pos_w = env.robot.data.body_pos_w[:, env.fingertip_body_idx, :]
        dist_to_tips = (tip_pos_w - env.object.data.root_pos_w.unsqueeze(1)).norm(dim=-1)
        n_close_tips = (dist_to_tips < args.reach_threshold_m).sum(dim=-1)
        if step >= n_steps - n_score:
            stable_counts += (dist_to_spawn < args.threshold_m).float()
            reach_counts += (n_close_tips >= 2).float()
        final_dist = dist_to_spawn

    stability_score = stable_counts / n_score
    reach_score = reach_counts / n_score
    # Both required: a candidate that is stable but unreachable (or briefly
    # in reach but unstable) should not win over one that is both.
    score_per_env = stability_score * reach_score

    score_per_group = score_per_env.view(n_groups, repeats).mean(dim=1)
    stability_per_group = stability_score.view(n_groups, repeats).mean(dim=1)
    reach_per_group = reach_score.view(n_groups, repeats).mean(dim=1)
    final_dist_per_group = final_dist.view(n_groups, repeats).mean(dim=1)

    def _curl_field(cf) -> dict:
        """``{"curl_frac": cf}`` for the original uniform-fraction schema, or
        ``{"curl_profile": [...]}`` for an I25 per-position profile -- kept as
        two distinct, self-describing keys rather than overloading one, so a
        reader of hand_calibration.json (or apply_palm_calibration, which
        never reads either -- only hand_default_joint_pos) is never handed a
        float where it might expect a list."""
        return {"curl_frac": cf} if isinstance(cf, (int, float)) else {"curl_profile": list(cf)}

    def _curl_str(cf) -> str:
        return f"{cf:.2f}" if isinstance(cf, (int, float)) else "/".join(f"{v:.2f}" for v in cf)

    score_table = []
    for gi, (cand, cf) in enumerate(combos):
        score_table.append({
            "axis": cand["axis"], "roll_deg": cand["roll_deg"], **_curl_field(cf),
            "score": float(score_per_group[gi]),
            "stability_score": float(stability_per_group[gi]),
            "reach_score": float(reach_per_group[gi]),
            "final_dist_m": float(final_dist_per_group[gi]),
        })
    score_table.sort(key=lambda e: -e["score"])
    for e in score_table:
        cf_str = (f"{e['curl_frac']:.2f}" if "curl_frac" in e
                  else "/".join(f"{v:.2f}" for v in e["curl_profile"]))
        print(f"[calibrate_palm_up]   axis={e['axis']:>2} roll={e['roll_deg']:>5.0f}deg "
              f"curl={cf_str}  score={e['score']:.3f} "
              f"(stability={e['stability_score']:.3f} reach={e['reach_score']:.3f})  "
              f"final_dist={e['final_dist_m']:.4f}m", flush=True)

    winner_gi = int(score_per_group.argmax())
    winner, winner_curl_frac = combos[winner_gi]
    winner_quat = torch.tensor(winner["quat_wxyz"], dtype=torch.float32)
    # env.robot.data.joint_names, NOT env.hand_spec.hand_joint_names: see
    # diagnostics.py's matching comment -- curl_table is indexed by the
    # ARTICULATION VIEW's own joint order (soft_joint_pos_limits), which is
    # not guaranteed to match the spec's URDF-declaration order.
    winner_joint_pos = {
        name: float(v) for name, v in zip(env.robot.data.joint_names, curl_table[winner_gi])
    }

    # Fold the (analytic, pre-jitter) world spawn point back into the palm's
    # own local frame under the WINNING rotation, so scene_utils can hand it
    # straight to env.cfg.reset.object_spawn_offset unchanged. A full 3-vector
    # now (not just a z-offset): the fingertip-centroid spawn point is not
    # necessarily directly above the palm origin's own (x, y).
    import numpy as np

    winner_spawn_w = spawn_pos.view(n_groups, repeats, 3)[winner_gi].mean(dim=0)
    winner_palm_w = palm_pos_w.view(n_groups, repeats, 3)[winner_gi].mean(dim=0)
    world_offset = (winner_spawn_w - winner_palm_w).cpu().numpy().astype(float)
    local_offset = pc.quat_apply(pc.quat_inv(np.array(winner["quat_wxyz"])), world_offset)

    entry = {
        "base_rot": [float(v) for v in winner["quat_wxyz"]],
        "axis": winner["axis"],
        "roll_deg": winner["roll_deg"],
        "spawn_offset_local": [float(v) for v in local_offset],
        **_curl_field(winner_curl_frac),
        "hand_default_joint_pos": winner_joint_pos,
        "score": float(score_per_group[winner_gi]),
        "stability_score": float(stability_per_group[winner_gi]),
        "reach_score": float(reach_per_group[winner_gi]),
        "final_dist_m": float(final_dist_per_group[winner_gi]),
        "method": (f"calibrate_palm_up.py: {len(candidates)} axis candidates x {len(curl_specs)} "
                   f"curl specs (full={args.full}, profiles={bool(args.curl_profiles)}), "
                   f"repeats={repeats}, held {args.seconds:.1f}s @ {1.0 / dt:.0f} Hz zero-action "
                   f"holding a curled joint target (lower + frac*(upper-lower) per joint, frac "
                   f"either one uniform value or a per-position profile tiled cyclically over the "
                   f"joint order -- I25), scored over the trailing {args.score_window_frac:.0%} "
                   f"of steps as stability(dist to the fingertip-centroid spawn point < "
                   f"{args.threshold_m} m) x reach(>=2 fingertips within {args.reach_threshold_m} "
                   f"m of the object)"),
        "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": pc.git_sha(),
        "score_table": score_table,
    }
    pc.save_calibration({args.hand: entry})
    print(f"[calibrate_palm_up] hand={args.hand} WINNER axis={winner['axis']} "
          f"roll={winner['roll_deg']:.0f}deg curl={_curl_str(winner_curl_frac)} "
          f"score={entry['score']:.3f} (stability={entry['stability_score']:.3f} "
          f"reach={entry['reach_score']:.3f}) base_rot={entry['base_rot']} "
          f"spawn_offset_local={entry['spawn_offset_local']} -> written to {pc.CALIB_PATH}",
          flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
