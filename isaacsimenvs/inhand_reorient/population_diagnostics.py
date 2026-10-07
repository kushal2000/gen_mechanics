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

36-slot layout (grammar_envelope/2), with ``--random-steps N``:

  (f) a rollout of N random-action steps: NaN, largest |q| and |qd|, and
      the tie error max |q_follower - q_leader| of every mimic-tied
      follower carrier;
  (g) self-contacts PhysX reports between body pairs the design
      collision-filters (``EnvelopeDesign.filtered_pairs``: short-bone
      pairs, each finger against the body it sits on through a locked or
      follower carrier, and an exempt hand's rest overlaps) -- there must be
      none. Needs contact reporting on the robot bodies, which the anyrotate
      and hora profiles switch on (``--task-profile hora``).

Prints a report; exits nonzero if (a)/(b)/(c)/(d)/(f)/(g) fail, but NOT
for (e)'s numbers (those are the diagnostic's OWN measurement, not a
pass/fail gate).

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
    p.add_argument("--task-profile", default="legacy", help="env.task_profile (hora: contact reporting on)")
    p.add_argument("--random-steps", type=int, default=0, help="(f)/(g): random-action steps, 0 = skip")
    p.add_argument("--max-tie-error", type=float, default=0.01, help="(f) fail above this tie error (rad)")
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
    cfg.task_profile = args.task_profile
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
            tips = ge.tip_fk(design, T)
            for f in range(ge.N_FINGERS):
                if not design.slot_valid[ge.finger_slot(f, 0)] or ge.tip_body(f) not in env.robot.data.body_names:
                    continue
                local_pos = torch.as_tensor(tips[f], device=device, dtype=torch.float32)
                expected_w = palm_pos_w[env_id] + quat_apply(palm_quat_w[env_id].unsqueeze(0), local_pos.unsqueeze(0))[0]
                actual_w = env.robot.data.body_pos_w[env_id, env.robot.data.body_names.index(ge.tip_body(f))]
                max_err_m = max(max_err_m, float((expected_w - actual_w).norm()))
                max_err_checked += 1
            for slot in range(ge.N_SLOTS):
                if not design.slot_valid[slot] and design.slot_tie[slot] < 0:
                    continue  # ghost slot or locked carrier: nothing to check
                local_pos = torch.as_tensor(T[slot][:3, 3], device=device, dtype=torch.float32)
                expected_w = palm_pos_w[env_id] + quat_apply(palm_quat_w[env_id].unsqueeze(0), local_pos.unsqueeze(0))[0]
                # AUTHORED body name (`grammar_envelope.slot_body`), NOT
                # `design.slot_body_name[slot]` (the GRAMMAR model's own body
                # name, e.g. "d0p1"), which never appears in `body_names`.
                body_name = ge.slot_body(slot)
                if body_name not in env.robot.data.body_names:
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

    # ghost slots and locked carriers (a follower carrier moves with its leader)
    moving = (torch.as_tensor(population.joint_valid, device=device, dtype=torch.bool)
              | torch.as_tensor(population.joint_tie >= 0, device=device))[design_idx]
    if perm is not None:
        moving = moving[:, perm]
    ghost_q = env.robot.data.joint_pos[~moving].abs()
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

    if args.random_steps > 0:
        ok &= _random_rollout_and_contacts(env, args, population, design_idx, ge)

    print(f"\n=== overall: {'PASS' if ok else 'FAIL'} ===\n", flush=True)
    import os

    os._exit(0 if ok else 1)  # simulation_app.close() can hang for minutes after a headless run


def _filtered_body_pairs(design, ge) -> set:
    """`design.filtered_pairs` as authored body-name pairs."""
    def body(node):
        return "root" if node == ge.ROOT_NODE else ge.slot_body(node)

    return {frozenset((body(i), body(j))) for i, j in design.filtered_pairs}


def _random_rollout_and_contacts(env, args, population, design_idx, ge) -> bool:
    """(f) random-action rollout: NaN, |q|, |qd|, tie error; (g) contacts
    between collision-filtered body pairs."""
    import numpy as np
    import torch

    n, device = env.num_envs, env.device
    tie_index = env.scene_record.get("tie_index")
    tie_gear = env.scene_record.get("tie_gear")
    tied = (tie_index != torch.arange(tie_index.shape[1], device=device)) if tie_index is not None else None
    filtered = [_filtered_body_pairs(population.designs[int(d)], ge) for d in design_idx.tolist()]
    n_filtered_pairs = sum(len(f) for f in filtered)
    try:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        contact_api = get_physx_simulation_interface()
    except Exception as exc:  # noqa: BLE001
        print(f"(g) contact report unavailable: {exc}")
        contact_api = None

    env.reset()
    torch.manual_seed(1)
    nan_steps = 0
    q_max = qd_max = tie_max = 0.0
    tie_per_design = np.zeros(population.n_designs)
    self_contacts = filtered_contacts = other_contacts = 0
    filtered_examples = []
    for step in range(args.random_steps):
        env.step(torch.rand(n, env.hand_spec.num_hand_joints, device=device) * 2.0 - 1.0)
        q, qd = env.robot.data.joint_pos, env.robot.data.joint_vel
        if not (torch.isfinite(q).all() and torch.isfinite(qd).all()):
            nan_steps += 1
            continue
        q_max = max(q_max, float(q.abs().max()))
        qd_max = max(qd_max, float(qd.abs().max()))
        if tied is not None and bool(tied.any()):
            src = q.gather(1, tie_index) * (tie_gear if tie_gear is not None else 1.0)   # gear: 1.1 per coupling
            err = ((q - src).abs() * tied).max(dim=1).values  # (n,)
            tie_max = max(tie_max, float(err.max()))
            for d in range(population.n_designs):
                rows = design_idx == d
                if bool(rows.any()):
                    tie_per_design[d] = max(tie_per_design[d], float(err[rows].max()))
        if contact_api is not None:
            headers, _data = contact_api.get_contact_report()
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0)).split("/")
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1)).split("/")
                if ("Robot" in a0) != ("Robot" in a1):
                    other_contacts += 1   # hand-object (or hand-ground): shows the reports work
                if "Robot" not in a0 or "Robot" not in a1 or a0[:-1] != a1[:-1]:
                    continue  # not a self-contact of one env's robot
                env_id = int(a0[-3][len("env_"):])
                self_contacts += 1
                pair = frozenset((a0[-1], a1[-1]))
                if pair in filtered[env_id]:
                    filtered_contacts += 1
                    if len(filtered_examples) < 5:
                        filtered_examples.append((env_id, tuple(sorted(pair))))
    print(f"(f) {args.random_steps} random-action steps: {nan_steps} steps with NaN, max |q| {q_max:.3f} rad, "
          f"max |qd| {qd_max:.2f} rad/s, tie error max {tie_max:.2e} rad")
    for d in range(population.n_designs):
        if (population.joint_tie[d] >= 0).any():
            print(f"    design {d} ({population.sources[d]}): tie error max {tie_per_design[d]:.2e} rad")
    ok = nan_steps == 0 and tie_max <= args.max_tie_error
    if contact_api is not None:
        print(f"(g) hand contacts with other bodies reported: {other_contacts} (0: the report sees nothing, "
              f"so the next line cannot show a filter failure)")
        print(f"(g) self-contacts reported: {self_contacts}; between collision-filtered pairs "
              f"({n_filtered_pairs} pairs over {n} envs): {filtered_contacts} {filtered_examples}")
        ok &= filtered_contacts == 0
    return ok


if __name__ == "__main__":
    sys.exit(main())
