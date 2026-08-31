"""PhysX material properties, applied to the built stage.

Friction and restitution are bound per shape after the scene exists, which
means reconciling the shape layout PhysX reports against the one recorded
for the robot spec.
"""

from __future__ import annotations

import time

import torch


from isaacsimenvs.pose_reaching_6d.utils.physx import _log_scene_step


def shape_layouts_from_record(link_names, recorded: dict, arm_counts: dict):
    """Per-design [(link, start, end)] from the authoring record + arm counts.

    ONE implementation, used by the friction pass and by
    tests/test_shape_layout_record.py. An earlier version of that test carried
    its own copy of this arithmetic, inherited the same off-by-one the fast path
    had, and so agreed with the code instead of checking it.
    """
    out = {}
    for design, per_link in recorded.items():
        start, layout = 0, []
        for name in link_names:
            # SUM both sources. A hand link has only a record, an arm link only
            # a measurement, and the merged palm (iiwa14_link_7) has BOTH: the
            # arm's own collision mesh plus the palm box authored onto it.
            n = arm_counts.get(name, 0) + per_link.get(name, 0)
            layout.append((name, start, start + n))
            start += n
        out[design] = (layout, start)
    return out


def arm_counts_from(measured_layout, ref_record: dict) -> dict:
    """The ARM's own per-link shape counts, isolated from a measured env.

    A measured layout comes from a composed robot, so iiwa14_link_7 already
    includes the authored palm box. Subtracting the record for the design that
    was measured leaves what the referenced arm USD contributes on its own;
    adding the record back per design is then exact rather than double-counted.
    """
    return {nm: (e - s) - ref_record.get(nm, 0)
            for nm, s, e in measured_layout if nm.startswith("iiwa14_link")}


def apply_physx_material_properties(env) -> None:
    """Set contact materials through PhysX tensor views.

    Follows Isaac Lab's large-scale randomization path: avoid post-spawn USD
    relationship authoring and per-clone material prims. Must run after
    ``DirectRLEnv`` starts the simulator and ``root_physx_view`` exists.
    """
    assets_cfg = env.cfg.assets
    if not assets_cfg.modify_asset_frictions:
        return

    t0 = time.perf_counter()
    default = torch.tensor(
        [float(assets_cfg.robot_friction), float(assets_cfg.robot_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    fingertip = torch.tensor(
        [float(assets_cfg.finger_tip_friction), float(assets_cfg.finger_tip_friction), 0.0],
        dtype=torch.float32, device="cpu",
    )
    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device="cpu")

    dr = env.cfg.domain_randomization
    n_buckets = int(dr.friction_n_buckets)
    ft_lo, ft_hi = float(dr.fingertip_friction_scale_range[0]), float(dr.fingertip_friction_scale_range[1])
    obj_lo, obj_hi = float(dr.object_friction_scale_range[0]), float(dr.object_friction_scale_range[1])
    ft_active = (ft_lo, ft_hi) != (1.0, 1.0)
    obj_active = (obj_lo, obj_hi) != (1.0, 1.0)

    robot_view = env.robot.root_physx_view
    robot_materials = robot_view.get_material_properties()
    robot_materials[:] = default

    fingertip_link_names = set(env.robot_spec.fingertip_body_names)

    # The shape layout is derived PER DESIGN, not once from env 0.
    #
    # A ghosted finger keeps its links but carries no collision geometry, so two
    # designs in one population have different shape counts and different shape
    # ORDERING within the flattened per-env shape array. Deriving the fingertip
    # slice from env 0 and applying it to every env would paint fingertip
    # friction onto whatever shape happened to occupy that index in another
    # design -- silently, since the values are all plausible frictions.
    #
    # Grouping by design keeps this O(k x links) rather than O(n x links). With
    # a single robot there is one group represented by env 0, which is exactly
    # the previous behaviour on exactly the same values.
    pop_specs = getattr(env, "_robot_population_specs", None)
    if pop_specs is None:
        groups = {0: torch.arange(env.num_envs, dtype=torch.int64)}
        tips_of = {0: fingertip_link_names}
    else:
        design_idx = env._robot_design_index_per_env.detach().cpu()
        groups = {int(d): (design_idx == int(d)).nonzero(as_tuple=True)[0]
                  for d in design_idx.unique()}
        tips_of = {d: set(pop_specs[d].fingertip_body_names) for d in groups}

    # (N, max_shapes): which shapes are fingertips, for THIS env's design.
    ft_shape_mask = torch.zeros(
        (env.num_envs, robot_view.max_shapes), dtype=torch.bool, device="cpu")

    def _measure_layout(rep_env: int):
        """(link_name, shape_start, shape_end) per link, and the total.

        One create_rigid_body_view per link, which is the expensive part: at
        ~1.2 ms a call it is the whole cost of this function.
        """
        out, shape_start = [], 0
        for link_name, link_path in zip(robot_view.shared_metatype.link_names,
                                        robot_view.link_paths[rep_env]):
            link_view = env.robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_end = shape_start + link_view.max_shapes
            out.append((link_name, shape_start, shape_end))
            shape_start = shape_end
        return out, shape_start

    # Per DESIGN, measured from PhysX. Correct, and slow: 24,576 designs x 37
    # links is 909,312 create_rigid_body_view calls, ~71 min at 24,576 envs.
    #
    # Two attempts to make this faster have been reverted from here, and both
    # are worth remembering:
    #
    #   1. Cache the layout under a GUESSED signature (per-slot active mask plus
    #      self-collision adjacency). Fast -- 171 s -> 6.9 s at 2,048 -- and
    #      WRONG: segment lengths also decide which links carry geometry, so two
    #      designs with the same active mask can have different layouts. Design
    #      5120 of an 8,192 population took another design's layout and its
    #      fingertip shapes landed at the wrong indices. Only the guard below
    #      caught it; a 2,048-design test passed it clean.
    #
    #   2. Count the shapes in USD instead of inferring them. Correct, but
    #      Usd.PrimRange with TraverseInstanceProxies per link is no cheaper
    #      than the PhysX call it replaced -- measured SLOWER at 8,192 designs
    #      than the code here.
    #
    # The real fix is to record shape counts while AUTHORING the designs, where
    # they are already known, rather than rediscovering them per link afterwards.
    # Until that exists, this stays: being right is not negotiable, and this is
    # a one-time setup cost.
    # Progress, because this loop runs for over an hour at 24,576 designs and
    # used to print nothing at all: a run sitting here was indistinguishable
    # from a run that had hung, which is exactly how two bad "fixes" for it got
    # diagnosed by stopwatch instead of by evidence.
    _n_designs = len(groups)
    _t_layout = time.perf_counter()

    # FAST PATH: use the collider map recorded while the designs were authored.
    #
    # The loop below is correct and costs ~96 min at 24,576 designs. It is slow
    # for one reason: it asks PhysX, per link per design, a question the
    # authoring already answered. When the robots were authored in-process, they
    # told us exactly which links got colliders, so the only thing left to
    # measure is the ARM -- identical across designs, referenced from one USD --
    # which is a single _measure_layout call.
    recorded = getattr(env, "_robot_collider_links", None)
    fast_layouts = None
    if recorded and len(recorded) >= len(groups):
        # Measure one env, then SUBTRACT what the authoring contributed to it.
        #
        # _measure_layout reads a real composed robot, so its iiwa14_link_7
        # already includes the palm box this pipeline authored onto that link.
        # Adding the recorded palm on top counted it twice and pushed the total
        # one over the view's max_shapes. What is wanted is the ARM'S OWN
        # contribution, which is the measurement minus the record for the very
        # design that was measured.
        _ref_design = next(iter(groups))
        arm_layout, _ = _measure_layout(int(groups[_ref_design][0]))
        arm_counts = arm_counts_from(arm_layout, recorded[_ref_design])
        fast_layouts = shape_layouts_from_record(
            list(robot_view.shared_metatype.link_names),
            {d: recorded[d] for d in groups}, arm_counts)
        _log_scene_step(_t_layout,
                        f"shape layouts for {len(groups)} designs from the "
                        f"authoring record")

    for _i, (design, group_env_ids) in enumerate(groups.items()):
        if _n_designs > 1000 and _i and _i % 2000 == 0:
            _el = time.perf_counter() - _t_layout
            print(f"[scene]   shape layouts {_i}/{_n_designs} "
                  f"({_el:.0f}s elapsed, ~{_el / _i * (_n_designs - _i):.0f}s left)",
                  flush=True)
        layout, shape_start = (fast_layouts[design] if fast_layouts is not None
                               else _measure_layout(int(group_env_ids[0])))
        for link_name, s0, s1 in layout:
            if link_name in tips_of[design]:
                robot_materials[group_env_ids, s0:s1] = fingertip
                ft_shape_mask[group_env_ids, s0:s1] = True
        # Designs with ghosted fingers legitimately carry FEWER shapes than the
        # view's maximum; only overrunning it is a bug.
        if shape_start > robot_view.max_shapes:
            raise RuntimeError(
                f"Robot shape count mismatch while assigning materials for "
                f"design {design}: computed {shape_start}, view reports "
                f"{robot_view.max_shapes}."
            )
        if pop_specs is None and shape_start != robot_view.max_shapes:
            raise RuntimeError(
                f"Robot shape count mismatch while assigning materials: "
                f"computed {shape_start}, view reports {robot_view.max_shapes}."
            )

    # VERIFY the fast path against PhysX on a sample. A wrong layout paints
    # fingertip friction onto the wrong shapes with every value still plausible,
    # which is exactly how an earlier version of this optimisation shipped a
    # silent bug. tests/test_shape_layout_record.py checks a WHOLE population;
    # this is the belt-and-braces check that runs in every real scene build.
    if fast_layouts is not None:
        for design in sorted(groups)[::max(1, len(groups) // 8)][:8]:
            physx, _ = _measure_layout(int(groups[design][0]))
            rec = fast_layouts[design][0]
            if physx != rec:
                # Show the first ENTRIES THAT DIFFER, not the first entries.
                # Printing [:6] of each showed six identical arm links and hid
                # the actual disagreement further down the list.
                diff = [(a, b) for a, b in zip(physx, rec) if a != b]
                raise RuntimeError(
                    f"authoring-recorded shape layout disagrees with PhysX for "
                    f"design {design}: {len(diff)} link(s) differ\n"
                    + "\n".join(f"  {a[0]}: physx={a[1:]} recorded={b[1:]}"
                                 for a, b in diff[:6]))

    # A name mismatch here is silent and consequential: every fingertip would
    # quietly fall back to the robot's base friction, changing grasp behavior
    # with no error.
    unmatched = [d for d in groups if not bool(ft_shape_mask[groups[d]].any())]
    # Two very different things produce an unmatched design, and only one is a
    # bug:
    #
    #   NAME MISMATCH -- the spec's fingertip_body_names are not links of this
    #   robot at all. Silent and catastrophic: every fingertip falls back to
    #   base friction and grasping changes with no error. Still fatal.
    #
    #   DEGENERATE FINGERTIP -- the names ARE links, but the design's distal
    #   phalanx is shorter than its own diameter, so has_collision_geometry
    #   deliberately gives it no capsule. That is a legitimate (if useless) hand
    #   the sampler can produce: exactly 1 design in 24,576 of population seed 2,
    #   index 5120, whose distal is 16.4 mm long at 12 mm radius. Its fingertips
    #   cannot touch anything, so there is no fingertip friction to assign and
    #   nothing to fix. Failing the whole run for it blocks every population of
    #   more than 5,120 designs.
    known_links = set(robot_view.shared_metatype.link_names)
    misnamed = [d for d in unmatched if not (set(tips_of[d]) & known_links)]
    if misnamed:
        who = (env.robot_spec.name if pop_specs is None
               else f"designs {sorted(misnamed)[:5]}")
        raise RuntimeError(
            f"{who}: fingertip_body_names={sorted(tips_of[misnamed[0]])} are not "
            f"links of this robot. Robot links: "
            f"{sorted(robot_view.shared_metatype.link_names)}"
        )
    if unmatched:
        print(f"[scene] {len(unmatched)} design(s) have no fingertip collision "
              f"geometry and get no fingertip friction "
              f"(first: {sorted(unmatched)[:5]}); their distal phalanges are "
              f"shorter than their own diameter", flush=True)

    # Per-env bucketed fingertip friction (init-only). Quantizing to
    # `n_buckets` distinct values caps the PhysX material count regardless
    # of n_envs.
    if ft_active:
        ft_base = float(assets_cfg.finger_tip_friction)
        bucket_vals = torch.linspace(ft_lo, ft_hi, n_buckets) * ft_base  # (B,)
        bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
        per_env_ft = bucket_vals[bucket_idx]  # (N_envs,)
        # Applied through the per-env fingertip mask so each design's own
        # fingertip shapes are the ones scaled. A shared index list would be the
        # union across designs, which for a heterogeneous population scales
        # shapes that are not fingertips in most envs.
        scaled = per_env_ft.unsqueeze(-1).expand(-1, robot_view.max_shapes)
        for channel in (0, 1):
            robot_materials[..., channel] = torch.where(
                ft_shape_mask, scaled, robot_materials[..., channel])

    robot_view.set_material_properties(robot_materials, env_ids)

    for name in ("table", "object", "goal_viz", "hole"):
        if not hasattr(env, name):
            continue
        view = getattr(env, name).root_physx_view
        materials = view.get_material_properties()
        materials[:] = default
        if name == "object" and obj_active:
            obj_base = float(assets_cfg.object_friction)
            bucket_vals = torch.linspace(obj_lo, obj_hi, n_buckets) * obj_base
            bucket_idx = torch.randint(0, n_buckets, (env.num_envs,))
            per_env_obj = bucket_vals[bucket_idx]  # (N_envs,)
            materials[:, :, 0] = per_env_obj.unsqueeze(-1)
            materials[:, :, 1] = per_env_obj.unsqueeze(-1)
        # Column 2 is restitution. Training uses 0.0 (fully inelastic), so the
        # default leaves this exactly as before; the object-physics eval axis
        # raises it to probe bouncier contacts.
        if name == "object":
            materials[:, :, 2] = float(assets_cfg.object_restitution)
        view.set_material_properties(materials, env_ids)

    _log_scene_step(t0, "applied PhysX material properties")


# ----------------------------------------------------------------------------
# Scene assembly
# ----------------------------------------------------------------------------
