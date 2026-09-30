# Twelve hands, left and right: one RL policy per hand model

**Goal (30 Sep):** train in-hand reorientation policies for 11 commercial hands plus our gen-SHARPA, each as a **left and a
right** model, with the default configuration (joint transformer, 4 layers, no ghost mask; γ 0.998,
adaptive lr from 5e-4, 5 mini-epochs; trimmed observation; no fall penalty; 5° tolerance).

## Hands (user's list)

| # | hand | joints (actuated) | left/right | status | integration notes |
|---|---|---|---|---|---|
| 1 | SHARPA Wave | 22 (22) | L/R | **left integrated** (`sharpa_handonly`) | need the right-hand model |
| 2 | Wuji Hand v2 | ~20 (20) | L/R | not started | source the URDF (vendor?) |
| 3 | Tesollo DG-5F | 20 (20) | L/R | not started | fully actuated |
| 4 | Allegro | 16 (16) | L/R | **right integrated** (`allegro_handonly`) | need the left-hand model |
| 5 | LEAP Hand | 16 (16) | L/R | not started | fully actuated |
| 6 | XHAND1 | 12 (12) | L/R | not started | fully actuated |
| 7 | Shadow Dexterous Hand | 24 (20) | L/R | not started | distal joints coupled on hardware; usually independent in sim |
| 8 | Unitree Dex3-1 | 7 (7) | L/R | not started | 3 fingers |
| 9 | PSYONIC Ability Hand | ~10 (6) | L/R | not started | **mimic joints** — needs mimic support |
| 10 | Inspire RH56 | ~12 (6) | L/R | not started | **mimic joints** — needs mimic support |
| 11 | BarrettHand | 8 (4) | **symmetric** | not started | left = right; mimic/coupled finger joints |
| 12 | gen-SHARPA (design grammar) | 30 slots: 21 real + 9 ghost | L (R?) | **left integrated** (`handonly:assets/populations/sharpa_capsule.json`) | capsule rebuild of left SHARPA; right needs a mirrored design — check the grammar supports mirroring |

Open decision: Barrett has no left/right distinction (train once, or twice as two seeds).

## Per-hand integration (what "integrated" means — see assets/urdf/unified_commercial_hands/README.md)

1. Source left and right URDFs + meshes; keep provenance in a `unify_from_vendor.py`.
2. Unify: root at the palm, repair colliders/inertias/limits, declare the palm frame (3x3 to our convention).
3. `RobotSpec` module (joint names, fingertips, gains, default pose, palm keypoints) + registry entry.
4. Self-collision adjacency map; verify filtered-pair count in a built scene.
5. Smoke: scene builds, reset geometry sane (no pad inside the cube), null-policy hold rate.
6. Train (2000-epoch budget unless decided otherwise).

Mimic-joint hands (Ability, Inspire, Barrett) additionally need mimic support in the env (policy acts on
motors only; coupled joints follow).
