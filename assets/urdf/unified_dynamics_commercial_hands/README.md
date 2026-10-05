# Commercial hands with uniform dynamics

Same kinematics and geometry as `../unified_commercial_hands/` (joint origins, axes, ranges, meshes --
meshes are referenced there, not copied); every dynamic property replaced by one shared value, so a
comparison across hands compares embodiment, not each vendor's actuator and mass guesses.

Regenerate with `.venv_isaacsim/bin/python assets/urdf/unified_dynamics_commercial_hands/make_uniform.py`.
Registered as `<hand>_left_uniform_handonly` (`isaacsimenvs/.../robots/unified_dynamics_hands.py`); run
files via `experiments/old_experiments/30sep_ten_hands/make_runs.py --uniform`.

| property | value | source |
|---|---|---|
| torque limit | 0.5 N.m, every joint | near real continuous ratings (gen-SHARPA uses 1.0, a stall-like figure) |
| speed limit | 10 rad/s, every joint | gen-SHARPA (`GEN_JOINT_VELOCITY_RAD_S`) |
| stiffness / damping / armature | 3.0 / 0.078 / 0.00058 | stiffness: Isaac Lab's Allegro setting at 0.5 N.m and what the converged runs used (gen-SHARPA's limit/1 rad would be 0.5, a 2-step tracking lag); armature = 0.00116 x limit (gen-SHARPA / SHARPA hardware); damping critical, ratio 0.93 (SHARPA) -- spec, not URDF |
| joint friction | 0 | as before |
| finger-link mass, COM, inertia | collision hulls at 1750 kg/m^3; 1 g floor | median of the vendors' own mass / hull volume (IQR 1190-2510) |
| palm inertial | vendor value kept | the base is fixed, the palm never moves |
| contact friction | 0.5 on every hand link, fingertips and the cube | run config (`env.assets.*_friction`), not the URDF |

What vendors had, for contrast: torque limits 0.2-10 N.m (Allegro 10 vs a real 0.7; Tesollo 7.5 vs a real
2 N.m stall), speed limits 3.14-15 rad/s, stiffness 3.0 (onboarded) or SHARPA's per-joint 0.9-13.2,
fingertip friction 1.5, degenerate inertias on Shadow / XHAND / SHARPA.

Hands: SHARPA, Allegro, LEAP, Shadow, Unitree Dex3, Tesollo DG-5F, Wuji v2, XHAND (all left).
`masses.json` lists every link's new mass.
