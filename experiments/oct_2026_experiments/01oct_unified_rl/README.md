# One policy across all commercial hands (1 Oct 2026)

Scene: `ROBOT_SPEC=multi:uniform` (or `multi:<spec>+<spec>+...`) -- the 8 uniform-dynamics commercial hands
(assets/urdf/unified_dynamics_commercial_hands) in ONE scene, envs dealt round-robin (env i -> hand i mod 8,
so every SAPG block holds every hand), each hand its own articulation with its real topology
(`scene_utils/multi_articulation.py`, `robots/multi_hand.py`). Env budget stays 12288 (1536 per hand).
gen-SHARPA is not in the scene yet (generated hands are authored by a different path).

Checks:
- `check_multi_equivalence.py`: same open-loop actions, hand alone vs inside a 2-hand scene. Allegro: mean
  joint positions within 0.0004 / 0.0008 / 0.007 rad at steps 20 / 100 / 200 (env spread up to 0.12 rad),
  cube distance within 0.6 mm, drops 5.9% vs 7.0%. Dex3: within 0.004 rad, cube within 2.4 mm, drops 89.7%
  vs 88.9%. Results in debug_outputs/inhand_debug/multi_equiv/.
- 8-hand training smoke (local, 40 epochs): ~42k fps, in line with the hands' own single-hand speeds.

Logging: per-hand episode metrics `per_hand/<hand>_{successes,done_fall,done_timeout}` -- their own wandb
tab ("per_hand"); each hand keeps its own rolling window of finished episodes.
