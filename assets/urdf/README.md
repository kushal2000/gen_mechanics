# assets/urdf

| Folder | What | Built from |
|---|---|---|
| `unified_commercial_hands/` | Vendor hand models (Allegro, LEAP, Shadow, Dex3, Tesollo, Wuji v2, XHAND, SHARPA, …) with their meshes, re-exported into one convention (left hand, palm frame, joint signs) as `<hand>_left.urdf` + `.spec.json` | the vendors' files (`vendor_left/`), by `onboard.py` / `import_vendor_left.py` |
| `unified_dynamics_commercial_hands/` | The same hands with **one shared set of dynamics** (torque, 5 rad/s speed, gains, density), so comparing hands compares embodiment. `<hand>_left_uniform_handonly` — the hands every in-hand run uses (e.g. Wuji in the olympics) | `unified_commercial_hands/` (meshes referenced there, not copied), by `make_uniform.py` |
| `unified_dynamics_commercial_hands_replicas/` | Each commercial hand re-expressed in the old generated-hand design space (07oct experiment) | `make_replicas.py` (uses `hand_sampler/commercial.py`) |
| `kuka_sharpa_description/` | The iiwa14 arm + SHARPA hand for the pose-reaching task | vendor |
| `generated/` | Generated hands: `14oct_design_search/` primitive hands (`experiments/14oct_design_search/primitive_hands.py`), old populations. Ignored by git; rebuild from their scripts | |
| `table_narrow.urdf` | The pose-reaching table | |

The chain matters: deleting or moving `unified_commercial_hands/` breaks every uniform-dynamics hand, whose URDFs point
at its meshes. Each folder's own README has the details.
