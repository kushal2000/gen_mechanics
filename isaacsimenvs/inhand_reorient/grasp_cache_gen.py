r"""Stable-grasp generation in a live ``anyrotate`` env (Kit side), and its
Kit entry point.

``ensure_grasp_table(env)`` is called from ``anyrotate_hooks.allocate_buffers``
when ``env.anyrotate.grasp_cache`` names a file: it loads the cache, runs
``generate`` for the designs it lacks (if ``grasp_cache_generate``), saves
the merged file and returns the reset-time ``GraspTable``. ``generate``
follows HORA's grasp generation (``grasp_cache.py``'s docstring): every env
holds one candidate per round; all envs of all designs run in parallel, so a
population costs about as much as one hand.

One round:
1. PD targets q~ per env (``grasp_cache.sample_joint_candidates``); joints
   written to q~ at rest; the object parked on the ground under the hand.
2. ``grasp_presettle_steps`` control steps: the hand settles at q~ under
   gravity.
3. The object is placed at the hand's spawn point or at the centroid of
   its valid fingertips (palm frame, + U(-noise, noise)), in a uniformly
   random orientation, at rest.
4. ``grasp_hold_s`` of physics with q~ held (zero actions) and gravity on
   (optionally turned through +-x, +-y, +-z as in AnyRotate App. C).
5. ``grasp_cache.stable_mask``; the survivors' settled joint positions, q~
   and the object pose in the palm frame are kept.

Rounds continue until every design has ``grasp_per_design`` grasps, has none
after ``grasp_gen_rounds_without_grasp`` rounds, or ``grasp_gen_max_rounds``
/ ``grasp_gen_max_minutes`` run out.

Kit entry point (generates and saves, then exits)::

    OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache WANDB_MODE=disabled \
    timeout -k 30 1800 .venv_isaacsim/bin/python -m isaacsimenvs.inhand_reorient.grasp_cache_gen \
        --population outputs/stability/pop_v3s_32.json --num-envs 4096 \
        [--out PATH] [env.anyrotate.grasp_per_design=1000 ...]

``--hand-id allegro_right`` instead of ``--population`` for a single hand
(default output ``outputs/grasp_cache/<hand>.grasps.npz``; a population's
default is its sidecar ``<stem>.grasps.npz``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import grasp_cache as gc
from . import repose_profile as rp

__all__ = ["env_design_keys", "canonical_grasp_pose", "generate", "ensure_grasp_table", "main"]

GRAVITY = 9.81
# AnyRotate App. C: gravity "sequentially" along the hand's 6 principal axes.
GRAVITY_SWEEP = ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                 (0.0, -1.0, 0.0))


# --------------------------------------------------------------------------
# Per-env identity and canonical pose
# --------------------------------------------------------------------------


def env_design_keys(env) -> Tuple[List[str], List[str], torch.Tensor]:
    """``(keys, sources, design_idx)``: one cache key and source label per
    design, and each env's design index."""
    tables = getattr(env, "hand_tables", None)
    if tables is not None:
        keys = [gc.design_key(d.sha256) for d in tables.designs]
        return keys, list(tables.sources), env.scene_record["design_idx"].to(torch.long)
    from .palm_calibration import load_calibration, load_repose_hand_poses, resolve_repose_hand_pose_path

    a = env.cfg.anyrotate
    hand_id = env.cfg.assets.hand_id
    pose = load_repose_hand_poses(resolve_repose_hand_pose_path(a.hand_pose_file)).get(hand_id)
    cfv = bool(a.collision_from_visuals) or bool((pose or {}).get("collision_from_visuals"))
    sha = gc.hand_calibration_sha(hand_id, load_calibration().get(hand_id), pose, cfv)
    return [gc.hand_key(hand_id, sha)], [hand_id], torch.zeros(env.num_envs, dtype=torch.long, device=env.device)


def canonical_grasp_pose(env) -> torch.Tensor:
    """``(num_envs, J)``: HORA's grasp pose for a hand that has one
    (``grasp_cache.CANONICAL_GRASP_POSES``), else each env's calibrated
    default pose (a grammar design's ``palm_up`` curl)."""
    from .reset_utils import _population_default_joint_pos

    ids = torch.arange(env.num_envs, device=env.device)
    q = _population_default_joint_pos(env, ids).clone()
    if getattr(env, "hand_tables", None) is None:
        named = gc.CANONICAL_GRASP_POSES.get(env.cfg.assets.hand_id, {})
        names = list(env.robot.data.joint_names)
        for jn, v in named.items():
            if jn in names:
                q[:, names.index(jn)] = float(v)
    limits = env.robot.data.soft_joint_pos_limits
    return torch.max(torch.min(q, limits[..., 1]), limits[..., 0])


# --------------------------------------------------------------------------
# Physics helpers (bypass env.step: no rewards, terminations or resets)
# --------------------------------------------------------------------------


def _set_gravity(direction, magnitude: float = GRAVITY) -> None:
    import carb
    import isaaclab.sim as sim_utils

    g = tuple(float(c) * magnitude for c in direction)
    sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(carb.Float3(*g))


def _control_step(env, targets: torch.Tensor) -> None:
    for _ in range(int(env.cfg.decimation)):
        env.robot.set_joint_position_target(targets)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)


def _contact_magnitudes(env) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(tip (n, k), nontip (n, m))`` contact force magnitudes, through
    ``anyrotate_hooks.contact_forces`` (population tips: last real link)."""
    from .anyrotate_hooks import contact_forces

    tip_f, _pos, nontip = contact_forces(env)
    return tip_f.norm(dim=-1), nontip


def _thresholds(a) -> gc.StabilityThresholds:
    return gc.StabilityThresholds(
        max_disp_m=float(a.grasp_max_disp_m), max_lin_speed=float(a.grasp_max_lin_speed),
        max_ang_speed=float(a.grasp_max_ang_speed), min_tip_contacts=int(a.grasp_min_tip_contacts),
        max_nontip_contacts=int(a.grasp_max_nontip_contacts), max_tip_dist_m=float(a.grasp_max_tip_dist_m),
        max_mean_tip_dist_m=float(a.grasp_max_mean_tip_dist_m), max_joint_speed=float(a.grasp_max_joint_speed))


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate(env, wanted: List[int], keys: List[str], sources: List[str], design_idx: torch.Tensor,
             seed: int = 0) -> Tuple[Dict[str, gc.GraspSet], dict]:
    """Grasps for the design indices ``wanted``; returns ``(sets, report)``."""
    from .obs_utils import _fingertip_valid_mask, _joint_valid_mask
    from .reset_utils import _object_spawn_offset

    a = env.cfg.anyrotate
    t0 = time.perf_counter()
    n, dev = env.num_envs, env.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    th = _thresholds(a)
    k_target = int(a.grasp_per_design)
    joint_names = list(env.robot.data.joint_names)
    n_designs = len(keys)

    wanted_mask = torch.zeros(n_designs, dtype=torch.bool, device=dev)
    wanted_mask[torch.as_tensor(wanted, dtype=torch.long, device=dev)] = True
    canonical = canonical_grasp_pose(env)
    limits = env.robot.data.soft_joint_pos_limits
    lower, upper = limits[..., 0], limits[..., 1]
    jvalid = _joint_valid_mask(env)
    tvalid = _fingertip_valid_mask(env)
    k_tips = len(env.fingertip_body_idx)
    if tvalid is None:
        tvalid = torch.ones(n, k_tips, dtype=torch.bool, device=dev)
    tvalid_f = tvalid.float()
    n_valid_tips = tvalid_f.sum(dim=-1).clamp(min=1.0)
    ids = torch.arange(n, device=dev)
    spawn = _object_spawn_offset(env, ids)
    origins = env.scene.env_origins
    park = torch.cat([origins + torch.tensor([0.0, 0.0, 0.04], device=dev),
                      torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4),
                      torch.zeros(n, 6, device=dev)], dim=-1)
    hold_steps = max(1, int(round(float(a.grasp_hold_s) / float(env.step_dt))))
    contact_window = min(3, hold_steps)
    mass = getattr(env, "_ar_object_mass", torch.zeros(n, device=dev))

    found: Dict[int, List[dict]] = {d: [] for d in wanted}
    n_found = torch.zeros(n_designs, dtype=torch.long)
    candidates = torch.zeros(n_designs, dtype=torch.long)
    modes = ("canonical_spawn", "canonical_tips", "curl_spawn", "curl_tips")
    cand_mode = torch.zeros(n_designs, 4, dtype=torch.long)
    pass_mode = torch.zeros(n_designs, 4, dtype=torch.long)
    fails = {name: torch.zeros(n_designs, dtype=torch.long) for name in
             ("disp", "speed", "tip_contacts", "tip_dist", "nontip", "mean_tip_dist", "joint_speed", "nonfinite")}
    jspeed_stable = {d: [] for d in wanted}  # peak joint speed of each stable grasp (before the joint test)
    rounds = 0
    gravity_now = getattr(env, "_ar_gravity", torch.tensor([0.0, 0.0, -GRAVITY], device=dev))
    _set_gravity((0.0, 0.0, -1.0))
    print(f"[grasp_cache] generating up to {k_target} grasps for {len(wanted)}/{n_designs} design(s) on "
          f"{n} envs: hold {hold_steps} steps ({hold_steps * float(env.step_dt):.2f} s), "
          f"thresholds {th}", flush=True)
    while True:
        active_d = wanted_mask & (n_found.to(dev) < k_target)
        stale = (n_found.to(dev) == 0) & (rounds >= int(a.grasp_gen_rounds_without_grasp))
        active_d = active_d & ~stale
        elapsed_min = (time.perf_counter() - t0) / 60.0
        if (not bool(active_d.any()) or rounds >= int(a.grasp_gen_max_rounds)
                or elapsed_min >= float(a.grasp_gen_max_minutes)):
            break
        rounds += 1
        active = active_d[design_idx]

        # 1-2. Candidate targets; the hand settles there with the object parked.
        q_t, curl = gc.sample_joint_candidates(
            canonical, lower, upper, jvalid, noise=float(a.grasp_joint_sample_noise),
            curl_frac=float(a.grasp_curl_frac), generator=gen, return_curl=True)
        env.robot.write_joint_state_to_sim(q_t, torch.zeros_like(q_t))
        env.object.write_root_state_to_sim(park)
        for _ in range(int(a.grasp_presettle_steps)):
            _control_step(env, q_t)

        # 3. Object at the spawn point or the fingertip centroid, random orientation.
        palm_p = env.robot.data.body_pos_w[:, env.palm_body_idx]
        palm_q = env.robot.data.body_quat_w[:, env.palm_body_idx]
        tips_w = env.robot.data.body_pos_w[:, env.fingertip_body_idx]
        centroid_w = (tips_w * tvalid_f.unsqueeze(-1)).sum(dim=1) / n_valid_tips.unsqueeze(-1)
        centroid = rp.quat_apply_inverse(palm_q, centroid_w - palm_p)
        at_tips = torch.rand(n, 1, device=dev, generator=gen) < float(a.grasp_tip_place_frac)
        noise = (torch.rand(n, 3, device=dev, generator=gen) * 2.0 - 1.0) * float(a.grasp_obj_pos_noise)
        p0 = torch.where(at_tips, centroid, spawn) + noise
        q0_palm = gc.random_quats(n, device=dev, generator=gen)
        state = torch.cat([palm_p + rp.quat_apply(palm_q, p0), rp.quat_mul(palm_q, q0_palm),
                           torch.zeros(n, 6, device=dev)], dim=-1)
        env.object.write_root_state_to_sim(state)

        # 4. Hold.
        max_disp = torch.zeros(n, device=dev)
        max_tip_dist = torch.zeros(n, device=dev)
        max_jspeed = torch.zeros(n, device=dev)
        finite = torch.ones(n, dtype=torch.bool, device=dev)
        tip_f = torch.zeros(n, k_tips, device=dev)
        nontip_f = None
        for step in range(hold_steps):
            if a.grasp_gravity_cycle:
                seg = min(len(GRAVITY_SWEEP) - 1, step * len(GRAVITY_SWEEP) // hold_steps)
                if step == 0 or seg != min(len(GRAVITY_SWEEP) - 1, (step - 1) * len(GRAVITY_SWEEP) // hold_steps):
                    _set_gravity(GRAVITY_SWEEP[seg])
            _control_step(env, q_t)
            obj_w = env.object.data.root_pos_w
            p = rp.quat_apply_inverse(palm_q, obj_w - palm_p)
            finite = finite & torch.isfinite(p).all(dim=-1)
            p = torch.nan_to_num(p, nan=1e3, posinf=1e3, neginf=-1e3)
            max_disp = torch.maximum(max_disp, (p - p0).norm(dim=-1))
            d_tip = (env.robot.data.body_pos_w[:, env.fingertip_body_idx] - obj_w.unsqueeze(1)).norm(dim=-1)
            d_tip = torch.nan_to_num(d_tip, nan=1e3) * tvalid_f
            max_tip_dist = torch.maximum(max_tip_dist, d_tip.max(dim=-1).values)
            qd = torch.nan_to_num(env.robot.data.joint_vel.abs(), nan=1e3)
            if jvalid is not None:
                qd = qd * jvalid
            max_jspeed = torch.maximum(max_jspeed, qd.max(dim=-1).values)
            if step >= hold_steps - contact_window:
                tf, nf = _contact_magnitudes(env)
                tip_f = tip_f + tf / contact_window
                nf = nf / contact_window
                nontip_f = nf if nontip_f is None else nontip_f + nf
        if a.grasp_gravity_cycle:
            _set_gravity((0.0, 0.0, -1.0))

        # 5. Acceptance.
        thr = float(a.contact_threshold)
        tip_c = ((tip_f > thr) & tvalid).sum(dim=-1)
        nontip_c = (nontip_f > thr).sum(dim=-1) if nontip_f is not None else torch.zeros_like(tip_c)
        lin = torch.nan_to_num(env.object.data.root_lin_vel_w, nan=1e3).norm(dim=-1)
        ang = torch.nan_to_num(env.object.data.root_ang_vel_w, nan=1e3).norm(dim=-1)
        obj_w = env.object.data.root_pos_w
        d_end = (env.robot.data.body_pos_w[:, env.fingertip_body_idx] - obj_w.unsqueeze(1)).norm(dim=-1)
        mean_tip = torch.nan_to_num((d_end * tvalid_f).sum(dim=-1) / n_valid_tips, nan=1e3)
        ok = gc.stable_mask(max_disp=max_disp, lin_speed=lin, ang_speed=ang, tip_contacts=tip_c,
                            nontip_contacts=nontip_c, max_tip_dist=max_tip_dist, mean_tip_dist=mean_tip,
                            finite=finite, joint_speed=max_jspeed, th=th) & active
        no_joint = gc.StabilityThresholds(**{**vars(th), "max_joint_speed": -1.0})
        ok_object = gc.stable_mask(max_disp=max_disp, lin_speed=lin, ang_speed=ang, tip_contacts=tip_c,
                                   nontip_contacts=nontip_c, max_tip_dist=max_tip_dist, mean_tip_dist=mean_tip,
                                   finite=finite, th=no_joint) & active
        for d_i, js in zip(design_idx[ok_object].tolist(), max_jspeed[ok_object].tolist()):
            if d_i in jspeed_stable:
                jspeed_stable[d_i].append(js)

        d_cpu = design_idx.cpu()
        act_cpu = active.cpu()
        candidates += torch.bincount(d_cpu[act_cpu], minlength=n_designs)
        mode = (curl.long() * 2 + at_tips.squeeze(-1).long()).cpu()
        flat = d_cpu * 4 + mode
        cand_mode += torch.bincount(flat[act_cpu], minlength=n_designs * 4).reshape(n_designs, 4)
        pass_mode += torch.bincount(flat[ok.cpu()], minlength=n_designs * 4).reshape(n_designs, 4)
        for name, bad in (("disp", max_disp > th.max_disp_m),
                          ("speed", (lin > th.max_lin_speed) | (ang > th.max_ang_speed)),
                          ("tip_contacts", tip_c < th.min_tip_contacts),
                          ("tip_dist", max_tip_dist > th.max_tip_dist_m),
                          ("nontip", (nontip_c > th.max_nontip_contacts) if th.max_nontip_contacts >= 0
                           else torch.zeros_like(finite)),
                          ("mean_tip_dist", (mean_tip > th.max_mean_tip_dist_m) if th.max_mean_tip_dist_m > 0
                           else torch.zeros_like(finite)),
                          ("joint_speed", (max_jspeed > th.max_joint_speed) if th.max_joint_speed > 0
                           else torch.zeros_like(finite)),
                          ("nonfinite", ~finite)):
            fails[name] += torch.bincount(d_cpu[(bad & active).cpu()], minlength=n_designs)

        hit = ok.nonzero(as_tuple=False).squeeze(-1)
        if hit.numel() > 0:
            obj_q_palm = rp.quat_mul(rp.quat_conjugate(palm_q), env.object.data.root_quat_w)
            obj_p_palm = rp.quat_apply_inverse(palm_q, obj_w - palm_p)
            rows = {
                "d": design_idx[hit].cpu().numpy(),
                "q": env.robot.data.joint_pos[hit].cpu().numpy(),
                "qt": q_t[hit].cpu().numpy(),
                "obj": torch.cat([obj_p_palm[hit], obj_q_palm[hit]], dim=-1).cpu().numpy(),
                "info": torch.stack([tip_c[hit].float(), nontip_c[hit].float(), mass[hit],
                                     max_disp[hit]], dim=-1).cpu().numpy(),
            }
            for d in np.unique(rows["d"]):
                sel = rows["d"] == d
                found[int(d)].append({k: v[sel] for k, v in rows.items() if k != "d"})
                n_found[int(d)] += int(sel.sum())
        print(f"[grasp_cache] round {rounds}: {int(act_cpu.sum())} candidates, {int(ok.sum())} stable; designs "
              f"with >= 1 grasp {int((n_found[wanted] > 0).sum())}/{len(wanted)}, done "
              f"{int((n_found[wanted] >= k_target).sum())}/{len(wanted)} ({time.perf_counter() - t0:.0f} s)",
              flush=True)

    _set_gravity(tuple(float(c) / GRAVITY for c in gravity_now.tolist()), GRAVITY)
    gen_s = time.perf_counter() - t0
    sets: Dict[str, gc.GraspSet] = {}
    for d in wanted:
        stats = {"candidates": int(candidates[d]), "stable": int(n_found[d]), "rounds": rounds,
                 "pass_rate": float(n_found[d]) / max(int(candidates[d]), 1),
                 "fails": {name: int(v[d]) for name, v in fails.items()},
                 "candidates_by_mode": {m: int(cand_mode[d, i]) for i, m in enumerate(modes)},
                 "joint_speed_of_object_stable": (
                     {"n": len(jspeed_stable[d]), "median": float(np.median(jspeed_stable[d])),
                      "p90": float(np.percentile(jspeed_stable[d], 90)), "max": float(np.max(jspeed_stable[d]))}
                     if jspeed_stable[d] else None),
                 "stable_by_mode": {m: int(pass_mode[d, i]) for i, m in enumerate(modes)},
                 "gen_s_shared": round(gen_s, 1), "created": gc.now_iso()}
        if found[d]:
            cat = {k: np.concatenate([r[k] for r in found[d]], 0) for k in ("q", "qt", "obj", "info")}
            s = gc.GraspSet(cat["q"], cat["qt"], cat["obj"], cat["info"], joint_names, sources[d], stats)
            sets[keys[d]] = s.truncated(k_target)
        else:
            sets[keys[d]] = gc.GraspSet.empty(joint_names, sources[d], stats)
    report = {
        "gen_s": round(gen_s, 1), "rounds": rounds, "n_envs": n, "designs_generated": len(wanted),
        "s_per_design": round(gen_s / max(len(wanted), 1), 2), "hold_steps": hold_steps,
        "thresholds": vars(th), "viable": int(sum(1 for d in wanted if n_found[d] > 0)),
        "per_design": {sources[d]: {"key": keys[d], **sets[keys[d]].stats, "kept": sets[keys[d]].n}
                       for d in wanted},
    }
    print(f"[grasp_cache] generated in {gen_s:.1f} s ({rounds} rounds, {report['s_per_design']} s per design): "
          f"{report['viable']}/{len(wanted)} design(s) with a stable grasp", flush=True)
    return sets, report


def _write_report(env, report: dict, cache_path: Path) -> None:
    out_dir = None
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # noqa: BLE001 -- no Hydra context
        out_dir = None
    path = (out_dir / "grasp_cache_report.json") if out_dir is not None else cache_path.with_suffix(".report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"[grasp_cache] report -> {path}", flush=True)


def ensure_grasp_table(env) -> Tuple[gc.GraspTable, torch.Tensor, List[str], dict]:
    """Load ``env.anyrotate.grasp_cache``, generate what it lacks (if
    ``grasp_cache_generate``), save, and return ``(table, design_idx,
    sources, report)``."""
    a = env.cfg.anyrotate
    path = Path(a.grasp_cache)
    keys, sources, design_idx = env_design_keys(env)
    signature = gc.object_signature(a)
    sets: Dict[str, gc.GraspSet] = {}
    doc: dict = {}
    if path.exists():
        sets, doc = gc.load_cache(path)
        bad = gc.signature_mismatches(doc.get("object_signature", {}), signature)
        if bad:
            raise ValueError(f"grasp cache {path} was made for different physics: {bad}")
    missing = [i for i, k in enumerate(keys) if k not in sets]
    report = {"cache": str(path), "designs": len(keys), "reused": len(keys) - len(missing), "generated": 0,
              "gen_s": 0.0}
    runs = list(doc.get("runs", []))
    if missing:
        if not a.grasp_cache_generate:
            raise FileNotFoundError(
                f"grasp cache {path} has no grasps for {[sources[i] for i in missing]}; generate them "
                f"(python -m isaacsimenvs.inhand_reorient.grasp_cache_gen) or set "
                f"env.anyrotate.grasp_cache_generate=true")
        new, gen_report = generate(env, missing, keys, sources, design_idx, seed=int(a.grasp_gen_seed))
        sets = gc.merge_sets(sets, new)
        runs.append({k: v for k, v in gen_report.items() if k != "per_design"})
        report.update(generated=len(missing), gen_s=gen_report["gen_s"], generation=gen_report)
    stale = [k for k in sets if k not in keys]
    if a.grasp_cache_prune and stale:
        sets = gc.prune_sets(sets, keys)
        report["pruned"] = len(stale)
    if missing or (a.grasp_cache_prune and stale):
        gc.save_cache(path, sets, {"object_signature": signature, "runs": runs, "updated": gc.now_iso()})
        print(f"[grasp_cache] saved {len(sets)} entr(ies) -> {path}", flush=True)
    per = [sets[k] for k in keys]
    table = gc.build_table(per, list(env.robot.data.joint_names), device=env.device,
                           max_per_design=int(a.grasp_per_design))
    counts = table.counts.cpu().tolist()
    report["grasps_per_design"] = {src: int(c) for src, c in zip(sources, counts)}
    report["viable"] = int(sum(1 for c in counts if c > 0))
    report["non_viable"] = [src for src, c in zip(sources, counts) if c == 0]
    _write_report(env, report, path)
    print(f"[grasp_cache] {path}: {report['viable']}/{len(keys)} design(s) with stable grasps "
          f"(median {float(np.median([c for c in counts if c > 0] or [0])):.0f} per viable design); "
          f"non-viable: {report['non_viable']}", flush=True)
    return table, design_idx, sources, report


# --------------------------------------------------------------------------
# Kit entry point
# --------------------------------------------------------------------------


def main() -> None:
    import argparse
    import os
    import sys

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Generate the anyrotate stable-grasp cache.")
    parser.add_argument("--task", default="GenMech-InHandReorient-Direct-v0")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--population", help="grammar population JSON (cache: its sidecar by default)")
    group.add_argument("--hand-id", help="single manifest hand, e.g. allegro_right")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--task-profile", default="anyrotate", choices=("anyrotate", "hora"),
                        help="the profile whose physics (timing, friction) the grasps are made under")
    parser.add_argument("--out", default=None, help="cache path (.npz)")
    parser.add_argument("--force", action="store_true", help="regenerate every key (ignore an existing file)")
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    sys.argv = [sys.argv[0]] + hydra_args
    app = AppLauncher(args).app

    import gymnasium as gym

    import isaacsimenvs  # noqa: F401  registers the task
    from coevolution.utils.hydra_utils import hydra_task_config_with_yaml

    repo = Path(__file__).resolve().parents[2]
    if args.out:
        out = Path(args.out)
    elif args.population:
        out = gc.sidecar_path(args.population)
    else:
        out = repo / "outputs" / "grasp_cache" / f"{args.hand_id}.grasps.npz"
    if args.force and out.exists():
        out.rename(out.with_name(out.name + ".bak"))

    @hydra_task_config_with_yaml(args.task, "rl_games_anyrotate_ppo_cfg_entry_point")
    def run(env_cfg, agent_cfg) -> None:
        env_cfg.task_profile = args.task_profile
        env_cfg.scene.num_envs = int(args.num_envs)
        if args.population:
            env_cfg.assets.hand_population = str(args.population)
        else:
            env_cfg.assets.hand_id = args.hand_id
        env_cfg.anyrotate.grasp_cache = str(out)
        env_cfg.anyrotate.grasp_cache_generate = True
        env_cfg.anyrotate.gravity_curriculum = False
        t0 = time.perf_counter()
        env = gym.make(args.task, cfg=env_cfg)
        inner = env.unwrapped
        report = getattr(inner, "_ar_grasp_report", {})
        print(f"[grasp_cache_gen] done in {time.perf_counter() - t0:.1f} s (boot + generation): "
              f"{json.dumps({k: v for k, v in report.items() if k != 'generation'})}", flush=True)

    run()
    sys.stdout.flush()
    sys.stderr.flush()
    del app
    os._exit(0)


if __name__ == "__main__":
    main()
