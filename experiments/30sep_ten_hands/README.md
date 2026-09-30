# Twelve hands: one RL policy per hand model (left hands first)

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


## Scope now (user, 30 Sep)

**Train a LEFT-hand policy for all 12 hands**, default configuration, **5000 epochs** (`make_runs.py`).
Mimic/coupled hands (Inspire, Ability, Barrett) are deferred. wandb project `gen_mechanics_ten_hands`.

## Tracker — left hands

| hand | left source (commit in vendor_left/SOURCE.md) | joints | 1 vendor | 2 unify+spec | 3 checks | 4 smoke | 5 train (5000 ep) |
|---|---|---|---|---|---|---|---|
| SHARPA | integrated (`sharpa_handonly`) | 22 | ✅ | ✅ | ✅ | ✅ | **696814** |
| gen-SHARPA | population `sharpa_capsule.json` | 21 (+9 ghost) | ✅ | ✅ | ✅ | ✅ | **696815** |
| Allegro | dex-urdf `allegro_hand_left.urdf` | 16 | ✅ | ✅ | ✅ | ✅ | **696962** |
| LEAP | dex-urdf `leap_hand_left.urdf` | 16 | ✅ | ✅ | ✅ | ✅ | **696963** |
| Shadow | dex-urdf `shadow_hand_left.urdf` | 24 incl. 2 wrist → 22 | ✅ | ✅ | ✅ | ✅ | **696964** |
| Unitree Dex3-1 | unitree_ros `dex3_1_l.urdf` (official) | 7 | ✅ | ✅ | ✅ | ✅ | **696965** |
| Tesollo DG-5F | dg5f_ros2 `dg5f_left.urdf` (official) | 20 | ✅ | ✅ | ✅ | ✅ | **696966** |
| Wuji Hand 2 | wuji-description `hand2/hand2_beta2/.../left.urdf` (official; Beta 2 = what wuji-mjlab deploys) | 20 | ✅ | ✅ | ✅ | ✅ | **696967** |
| XHAND1 | spider `xhand_left.urdf` (Robot Era SolidWorks export; licence NOASSERTION) | 12 | ✅ | ✅ | ✅ | ✅ | **696968** |
| Inspire / Ability / Barrett | dex-urdf (deferred: mimic joints) | | | | | | |

Step 2 = `assets/urdf/unified_commercial_hands/onboard.py` (re-root at the palm, measure joints / home pose /
palm frame / palm box / pads / self-collision pairs → `<hand>/<hand>_left.spec.json` + `.report.md`) and
`robots/unified_hands.py` (JSON → RobotSpec `<hand>_left_handonly`). Validated by reproducing the hand-written
Allegro right spec field for field (frame, palm box, mount pose, palm keypoints; pads within 1 cm).
Step 3 = the report's checks (all PASS; Tesollo's 64 mm palm reviewed and waived: one continuous collider)
and `isaacsimenvs/inhand_reorient/tests/test_unified_hands.py` (9 tests).

Step 1 = `assets/urdf/unified_commercial_hands/import_vendor_left.py`: left URDF + only its meshes, pinned
commit, mesh paths made local; each loads with every mesh resolved (checked with yourdfpy).
Sourcing notes: the older `wuji-hand-description` is Wuji Hand **1**; Hand 2 (left and right) is in
`wuji-technology/wuji-description`. XHAND's vendor (Robot Era, org `roboterax`) publishes no hand URDF;
the left model is from Meta's `spider` (also in `EmptyBlueBox/DexLatent`).

## Per-hand integration (what "integrated" means — see assets/urdf/unified_commercial_hands/README.md)

1. Source left and right URDFs + meshes; keep provenance in a `unify_from_vendor.py`.
2. Unify: root at the palm, repair colliders/inertias/limits, declare the palm frame (3x3 to our convention).
3. `RobotSpec` module (joint names, fingertips, gains, default pose, palm keypoints) + registry entry.
4. Self-collision adjacency map; verify filtered-pair count in a built scene.
5. Smoke: scene builds, reset geometry sane (no pad inside the cube), null-policy hold rate.
6. Train (2000-epoch budget unless decided otherwise).

Mimic-joint hands (Ability, Inspire, Barrett) additionally need mimic support in the env (policy acts on
motors only; coupled joints follow).

**Smoke (step 4), 30 Sep:** all seven build, apply their self-collision filters and train 3 epochs. Bodies =
joints + palm for each (Allegro/LEAP 17, Shadow 23, Tesollo/Wuji2 21, XHAND 13, Dex3 8). Allegro and LEAP
first failed: Isaac's URDF importer renames names that are not valid USD identifiers (Allegro's
`link_0.0` → `link_0_0`), and LEAP's bare-number joint names (`0`, `1`, ...) collapsed the whole hand into
one rigid body. `onboard.py` now renames every link and joint to a valid identifier (recorded in the URDF).
Not yet checked per hand: cube spawn geometry and drive gains (uniform, Allegro's) — watch early curves
and viewer pages.
