"""PhysX material properties, applied once the simulator has started.

Friction is set per shape through the PhysX tensor views, so the fingertip
shape indices of every design have to be known.
"""

from __future__ import annotations

import time

import torch

from ..common_utils.physx import _log_scene_step


def shape_layouts_from_record(link_names, recorded: dict, arm_counts: dict):
    """Per-design ``([(link, start, end)], total)`` from the authoring record.
    The merged palm has both an arm mesh and an authored box, so the two sum."""
    out = {}
    for design, per_link in recorded.items():
        start, layout = 0, []
        for name in link_names:
            n = arm_counts.get(name, 0) + per_link.get(name, 0)
            layout.append((name, start, start + n))
            start += n
        out[design] = (layout, start)
    return out


def arm_counts_from(measured_layout, ref_record: dict) -> dict:
    """The arm's own shape counts: one measured env minus its design's record."""
    return {nm: (e - s) - ref_record.get(nm, 0)
            for nm, s, e in measured_layout if nm.startswith("iiwa14_link")}


def _bucketed(lo: float, hi: float, base: float, n_buckets: int, n: int) -> torch.Tensor:
    """Per-env friction on ``n_buckets`` values, which caps the PhysX material count."""
    values = torch.linspace(lo, hi, n_buckets) * base
    return values[torch.randint(0, n_buckets, (n,))]


def apply_physx_material_properties(env) -> None:
    """Set contact materials through the PhysX tensor views (no post-spawn USD
    authoring, no per-clone material prims). Needs the started simulator."""
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return
    t0 = time.perf_counter()
    dr = env.cfg.domain_randomization
    n_buckets = int(dr.friction_n_buckets)
    ft_lo, ft_hi = (float(v) for v in dr.fingertip_friction_scale_range)
    obj_lo, obj_hi = (float(v) for v in dr.object_friction_scale_range)
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    def material(friction: float) -> torch.Tensor:
        return torch.tensor([friction, friction, 0.0], dtype=torch.float32, device="cpu")

    default = material(float(assets_cfg.robot_friction))
    fingertip = material(float(assets_cfg.finger_tip_friction))

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default
    link_names = list(robot_view.shared_metatype.link_names)

    def measure_layout(rep_env: int):
        """``([(link, start, end)], total)`` from PhysX, one view per link."""
        out, start = [], 0
        for link_name, link_path in zip(link_names, robot_view.link_paths[rep_env]):
            end = start + env.robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
            out.append((link_name, start, end))
            start = end
        return out, start

    # Layouts are per design: a ghosted finger keeps its links but no shapes.
    population = env._robot_population
    if population is None:
        groups = {0: env_ids}
        tips_of = {0: set(env.robot_spec.fingertip_body_names)}
        layouts = {0: measure_layout(0)}
    else:
        design_idx = env._robot_design_index_per_env.detach().cpu()
        groups = {int(d): (design_idx == int(d)).nonzero(as_tuple=True)[0]
                  for d in design_idx.unique()}
        tips_of = {d: set(population.specs[d].fingertip_body_names) for d in groups}
        # From the authoring record, with the shared arm measured once;
        # measuring every design is ~96 min at 24,576.
        recorded = env._robot_collider_links
        ref = next(iter(groups))
        arm_layout, _ = measure_layout(int(groups[ref][0]))
        layouts = shape_layouts_from_record(
            link_names, {d: recorded[d] for d in groups},
            arm_counts_from(arm_layout, recorded[ref]))
        _log_scene_step(t0, f"shape layouts for {len(groups)} designs from the authoring record")
        # Spot-check against PhysX: a wrong layout is silent.
        for design in sorted(groups)[::max(1, len(groups) // 8)][:8]:
            physx, _ = measure_layout(int(groups[design][0]))
            diff = [(a, b) for a, b in zip(physx, layouts[design][0]) if a != b]
            if diff:
                raise RuntimeError(
                    f"recorded shape layout disagrees with PhysX for design {design}:\n"
                    + "\n".join(f"  {a[0]}: physx={a[1:]} recorded={b[1:]}"
                                for a, b in diff[:6]))

    ft_shape_mask = torch.zeros(
        (env.num_envs, robot_view.max_shapes), dtype=torch.bool, device="cpu")
    for design, group_env_ids in groups.items():
        layout, n_shapes = layouts[design]
        for link_name, s0, s1 in layout:
            if link_name in tips_of[design]:
                robot_materials[group_env_ids, s0:s1] = fingertip
                ft_shape_mask[group_env_ids, s0:s1] = True
        # Ghosted fingers leave a design short of the view's maximum; overrunning is a bug.
        if n_shapes > robot_view.max_shapes or (
                population is None and n_shapes != robot_view.max_shapes):
            raise RuntimeError(
                f"design {design}: computed {n_shapes} shapes, view reports "
                f"{robot_view.max_shapes}")

    # No fingertip shape: a name mismatch (fatal) or a distal phalanx too
    # short for a capsule (harmless, it cannot touch anything).
    unmatched = [d for d in groups if not bool(ft_shape_mask[groups[d]].any())]
    misnamed = [d for d in unmatched if not (tips_of[d] & set(link_names))]
    if misnamed:
        who = (env.robot_spec.name if population is None
               else f"designs {sorted(misnamed)[:5]}")
        raise RuntimeError(
            f"{who}: fingertip_body_names={sorted(tips_of[misnamed[0]])} are not "
            f"links of this robot: {sorted(link_names)}")
    if unmatched:
        print(f"[scene] {len(unmatched)} design(s) have no fingertip collision "
              f"geometry (first: {sorted(unmatched)[:5]})", flush=True)

    if (ft_lo, ft_hi) != (1.0, 1.0):
        per_env = _bucketed(ft_lo, ft_hi, float(assets_cfg.finger_tip_friction),
                            n_buckets, env.num_envs)
        scaled = per_env.unsqueeze(-1).expand(-1, robot_view.max_shapes)
        for channel in (0, 1):
            robot_materials[..., channel] = torch.where(
                ft_shape_mask, scaled, robot_materials[..., channel])
    robot_view.set_material_properties(robot_materials, env_ids)

    for name, friction in (("table", assets_cfg.table_friction),
                           ("object", assets_cfg.object_friction),
                           ("goal_viz", assets_cfg.robot_friction)):
        view = getattr(env, name).root_physx_view
        materials = view.get_material_properties()
        materials[:] = material(float(friction))
        if name == "object":
            if (obj_lo, obj_hi) != (1.0, 1.0):
                per_env = _bucketed(obj_lo, obj_hi, float(assets_cfg.object_friction),
                                    n_buckets, env.num_envs)
                materials[:, :, 0] = per_env.unsqueeze(-1)
                materials[:, :, 1] = per_env.unsqueeze(-1)
            materials[:, :, 2] = float(assets_cfg.object_restitution)
        view.set_material_properties(materials, env_ids)

    _log_scene_step(t0, "applied PhysX material properties")


__all__ = ["apply_physx_material_properties", "arm_counts_from", "shape_layouts_from_record"]
