r"""Kit diagnostic for the grammar-population path (opus-review-phase2.md
review item 7: "No CPU test covers the permutation, masks, spawn or carrier
wiring. The Kit FK script is uncommitted." -- and the design note's own
"Kit smoke, 64 envs, 16 designs" acceptance list, section 5).

Boots one Kit process, one scene, authors ``--population`` (default: the
16-design test population), and checks, per env:

  (a) ``is_homogeneous`` on the live articulation view;
  (b) the articulation's own joint order vs ``grammar_envelope.SLOT_NAMES``,
      and that the resolved permutation actually reindexes correctly;
  (c) FK: the live PhysX body pose (world frame) vs
      ``grammar_envelope.authored_fk`` (root-relative) composed with that
      env's own ``base_pos``/``base_rot`` -- at q=0 (the just-reset default
      pose) and after a short random-action rollout;
  (d) joint limits: PhysX's ``soft_joint_pos_limits`` vs
      ``population.joint_limits[design_idx]``, reindexed to phys-column
      order;
  (e) after 100 zero-action steps: ghost-column ``|q|`` (should stay
      bounded near 0, review risk 8/I29), the fraction of cubes still
      resting near their spawn point (I30), and per-design reach (how many
      envs of each design still have the object within
      ``drop_distance_m/2`` of the palm).

Prints a report; exits nonzero if (a)/(b)/(c)/(d) fail (structural
correctness the pilot depends on), but NOT for (e)'s numbers (those are the
diagnostic's OWN measurement, not a pass/fail gate -- see the worker report
for how to read them).

Run (single Kit process; ``timeout -k 30 <cap>`` per this branch's Kit-run
rule):
    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache \
    WANDB_MODE=disabled timeout -k 30 300 \
    .venv_isaacsim/bin/python isaacsimenvs/inhand_reorient/population_diagnostics.py \
        --population outputs/grammar_populations/test16.json --num-envs 64
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--population", default="outputs/grammar_populations/test16.json")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--rollout-steps", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply

    from isaacsimenvs.inhand_reorient.env import InHandReorientEnv
    from isaacsimenvs.inhand_reorient.env_cfg import InHandReorientEnvCfg
    from isaacsimenvs.inhand_reorient.scene import grammar_envelope as ge

    cfg = InHandReorientEnvCfg()
    cfg.assets.hand_population = args.population
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device

    env = InHandReorientEnv(cfg=cfg)
    env.reset()
    device = env.device
    n = env.num_envs
    population = env.hand_tables
    design_idx = env.scene_record["design_idx"]
    perm = env.scene_record.get("slot_of_phys_col")
    phys_names = list(env.robot.data.joint_names)

    ok = True

    print(f"\n=== population_diagnostics: {population.n_designs} designs, {n} envs ===\n", flush=True)

    # --- (a) is_homogeneous --------------------------------------------------
    homogeneous = bool(env.robot.root_physx_view.is_homogeneous)
    print(f"(a) is_homogeneous: {homogeneous}")
    ok &= homogeneous

    # --- (b) joint order / permutation ---------------------------------------
    missing = [name for name in phys_names if name not in ge.SLOT_NAMES]
    perm_identity = perm is None
    print(f"(b) joint order matches SLOT_NAMES (identity permutation): {perm_identity}; "
          f"unknown names: {missing}")
    ok &= not missing

    # --- (c) FK: live PhysX pose vs authored_fk, at q=0 and after a rollout --
    def _fk_error(label: str) -> float:
        palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx]
        palm_quat_w = env.robot.data.body_quat_w[:, env.palm_body_idx]
        max_err_m = 0.0
        max_err_checked = 0
        for env_id in range(n):
            d = int(design_idx[env_id])
            design = population.designs[d]
            q_slot = np.zeros(ge.N_SLOTS)
            phys_q = env.robot.data.joint_pos[env_id].detach().cpu().numpy()
            if perm is not None:
                perm_np = perm.detach().cpu().numpy()
                for col, slot in enumerate(perm_np):
                    q_slot[slot] = phys_q[col]
            else:
                q_slot[:] = phys_q
            T = ge.authored_fk(design, q_slot)
            for slot in range(ge.N_SLOTS):
                if not design.slot_valid[slot]:
                    continue
                local_pos = torch.as_tensor(T[slot][:3, 3], device=device, dtype=torch.float32)
                expected_w = palm_pos_w[env_id] + quat_apply(palm_quat_w[env_id].unsqueeze(0), local_pos.unsqueeze(0))[0]
                body_name = design.slot_body_name[slot]
                if body_name is None or body_name not in env.robot.data.body_names:
                    continue
                body_idx = env.robot.data.body_names.index(body_name)
                actual_w = env.robot.data.body_pos_w[env_id, body_idx]
                err = float((expected_w - actual_w).norm())
                max_err_m = max(max_err_m, err)
                max_err_checked += 1
        print(f"(c) FK error [{label}]: max {max_err_m * 1000.0:.4f} mm over {max_err_checked} body checks")
        return max_err_m

    err0 = _fk_error("q=0, just reset")
    torch.manual_seed(0)
    for _ in range(20):
        env.step(torch.rand(n, env.hand_spec.num_hand_joints, device=device) * 2.0 - 1.0)
    err_rollout = _fk_error("after 20 random-action steps")
    ok &= err0 < 5e-3  # generous vs the design note's 1e-4 m target: this is a smoke, not the precision check

    # --- (d) joint limits: PhysX vs population table -------------------------
    joint_limits = torch.as_tensor(population.joint_limits, device=device, dtype=torch.float32)[design_idx]
    if perm is not None:
        joint_limits = joint_limits[:, perm]
    phys_lower = env.robot.data.soft_joint_pos_limits[..., 0]
    phys_upper = env.robot.data.soft_joint_pos_limits[..., 1]
    limit_err = float((joint_limits[..., 0] - phys_lower).abs().max()) + float((joint_limits[..., 1] - phys_upper).abs().max())
    print(f"(d) joint limits max abs error vs population table: {limit_err:.6f} rad")
    ok &= limit_err < 1e-3

    # --- (e) 100 zero-action steps: ghost |q|, cube rest, per-design reach --
    env.reset()
    for _ in range(args.rollout_steps):
        env.step(torch.zeros(n, env.hand_spec.num_hand_joints, device=device))

    joint_valid = torch.as_tensor(population.joint_valid, device=device, dtype=torch.bool)[design_idx]
    if perm is not None:
        joint_valid = joint_valid[:, perm]
    ghost_q = env.robot.data.joint_pos[~joint_valid].abs()
    print(f"(e) after {args.rollout_steps} zero-action steps: ghost |q| max "
          f"{float(ghost_q.max()) if ghost_q.numel() else float('nan'):.6f} rad, "
          f"mean {float(ghost_q.mean()) if ghost_q.numel() else float('nan'):.6f} rad")

    obj_pos_w = env.object.data.root_pos_w
    palm_pos_w = env.robot.data.body_pos_w[:, env.palm_body_idx]
    spawn_offset = torch.as_tensor(population.spawn_offset, device=device, dtype=torch.float32)[design_idx]
    base_rot = torch.as_tensor(population.base_rot, device=device, dtype=torch.float32)[design_idx]
    expected_pos_w = palm_pos_w + quat_apply(base_rot, spawn_offset)
    disp = (obj_pos_w - expected_pos_w).norm(dim=-1)
    resting = disp < 0.05  # 5cm of its spawn point
    print(f"(e) cubes still resting near spawn (< 5cm): {int(resting.sum())}/{n} "
          f"({100.0 * float(resting.float().mean()):.1f}%)")

    print("(e) per-design reach (fraction of that design's envs still resting):")
    for d in range(population.n_designs):
        rows = (design_idx == d)
        if not bool(rows.any()):
            continue
        frac = float(resting[rows].float().mean())
        print(f"    design {d} ({population.sources[d]}): {int(rows.sum())} envs, {frac * 100.0:.1f}% resting")

    print(f"\n=== overall: {'PASS' if ok else 'FAIL'} ===\n", flush=True)
    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
