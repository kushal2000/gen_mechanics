# Commercial hands rebuilt in the hand grammar

One replica per vendor hand that `hand_sampler.commercial.fit` handles: the closest LEGAL grammar design to the
vendor's kinematics (straight capsule links, coordinate-direction joint axes, mounts on the 5 mm grid, links at
least 20 mm, knuckles at least 35 mm apart), with the uniform actuator. The counterpart of
`../unified_dynamics_commercial_hands/`, which keeps the vendors' own links and meshes under the same actuator,
so a replica vs its vendor hand isolates what the grammar's geometry costs.

    .venv_isaacsim/bin/python assets/urdf/unified_dynamics_commercial_hands_replicas/make_replicas.py   # all four
    .venv_isaacsim/bin/python -m hand_sampler.overlay_fits --out assets/urdf/unified_dynamics_commercial_hands_replicas/overlay.png

| replica | from | fingers / joints | palm (mm) | base offset mean / worst | worst shape error |
|---|---|---|---|---|---|
| `leap/leap_left_replica` | `leap_left.urdf` | 4 / 16 | 25 x 100 x 85 | 1.8 / 2.0 mm | 8.0 mm |
| `allegro/allegro_left_replica` | `allegro_left.urdf` | 4 / 16 | 25 x 100 x 85 | 2.2 / 3.0 mm | 27.4 mm |
| `wuji2/wuji2_left_replica` | `wuji2_left.urdf` | 5 / 20 | 25 x 100 x 75 | 10.4 / 21.0 mm | 17.5 mm |
| `midas/midas_replica` | `midas.urdf` | 4 / 16 | 25 x 80 x 70 | 4.3 / 6.0 mm | 27.8 mm |

(Fit quality from `hand_sampler/DESIGN.md` A10.) Read the per-hand `*.notes.md` before trusting one, and
`overlay.png` (replica in colour over the vendor in grey) for where:

* **LEAP and Allegro are close copies by base placement** (within 3 mm). Allegro's 27.4 mm worst shape error
  is unexplained in DESIGN.md; check its thumb in the overlay before reading much into an Allegro comparison.
* **Wuji v2 is the loosest.** Its knuckles are 22 mm apart against the 35 mm floor, so the row is spread 13 mm
  wider, and it clears the floor by exactly 0. The middle finger really does not match (DESIGN.md: it needs
  +80/-86 deg to come onto the vendor's). Treat it as "a Wuji-like grammar hand", not Wuji.
* **MIDAS has no trainable vendor counterpart yet**: only its raw URDF is in `../unified_dynamics_commercial_hands/midas`,
  not onboarded (no `*_uniform_handonly` spec). Its thumb's 27.8 mm is a rest-pose disagreement, not a wrong mechanism.
* SHARPA is not here: `commercial.fit` dropped it as too distorted; `hand_sampler.sharpa_capsule` is the hand-built
  stand-in the earlier runs used.

## Files, per hand

| file | what | used by |
|---|---|---|
| `<stem>_replica.json` | one-design population (`population_io`), provenance includes the fit notes and actuator | **training**: `ROBOT_SPEC=handonly:<absolute path>` |
| `<stem>_replica.urdf` + `_palm.obj` | `build.urdf_for_viewing`: the full 5x4 slot envelope, ghost slots empty | viewers only, NOT simulation |
| `<stem>_replica.notes.md` | every way the replica is not the vendor's | reading |

## Dynamics

Generated joints take `robot_param_constants.gen_joint_drive()`: since 2026-10-07 the same uniform actuator as
`make_uniform.py` (0.5 N.m, 5 rad/s, kp 3.0, kd 0.0775, armature 0.00058). Masses are NOT the vendor's:
links are solid capsules at the grammar's density. Runs have also pinned the actuator with `env.physics.hand_*`
overrides, as 01oct_uniform_dynamics did for gen-SHARPA.

Regenerate after any change to `hand_sampler/commercial.py`, `design_space.py` or the constants: the JSON is a
record of what the fitter produced at the commit in its provenance, and `load_population` re-validates it.
