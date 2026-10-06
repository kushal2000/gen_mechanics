# 05 Oct: embodiment generalization

Does training one policy on 8 hands (`experiments/01oct_unified_rl`) make it robust to changes in the HAND?
The object-side perturbations (push force, mass, friction) live in `experiments/02oct_ood_evals`;
there the Wuji-only policy won everywhere. This folder perturbs the embodiment itself.

## Missing fingers (`results/missing_finger/`, `plots/missing_finger_*.png`)

Wuji v2 with one finger removed -- the finger's subtree deleted from the URDF, its joints / fingertip / collision
filters dropped from the spec, everything else identical (`assets/urdf/unified_dynamics_commercial_hands/
make_missing_fingers.py`, registered as `wuji2_left_uniform_handonly_no_<finger>`). Both policies drive the
16-joint hand zero-shot: the joint transformer has one token per joint, so it is rebuilt for the hand's joints
with the same weights. `run_embodiment_sweep.sub AXIS=finger`.

Unified 40k vs Wuji-only ~32k, goals per episode (50-goal max, 100 first episodes), drops/min in brackets:

| missing | none | thumb | index | middle | ring | pinky |
|---|---|---|---|---|---|---|
| Unified 40k | 42.8 (0.3) | 0.0 (120) | 0.2 (82) | 4.1 (11) | **31.0 (0.6)** | **36.6 (0.8)** |
| Wuji-only ~32k | 49.4 (0.1) | 0.1 (94) | 2.1 (35) | 1.9 (35) | 3.1 (22) | 1.3 (35) |

Without the ring or pinky finger the unified policy keeps 72-85% of its full-hand score; the Wuji-only policy
collapses. Thumb and index are needed by both. Single training run per policy.

### Other hands (unified 40k vs each hand's own palm-frame policy, ~5k epochs; not matched compute)

| hand | policy | none | thumb | index | middle | ring | pinky |
|---|---|---|---|---|---|---|---|
| Shadow | Unified 40k | 48.1 | 0.0 | 0.8 | 1.1 | 1.9 | 3.0 |
| Shadow | Shadow only | 43.1 | 0.0 | 0.1 | 0.5 | 3.8 | 0.4 |
| Tesollo | Unified 40k | 40.3 | 0.0 | 0.0 | 2.3 | 32.3 | **39.9** |
| Tesollo | Tesollo only | 49.9 | 0.0 | **20.9** | **8.7** | 40.2 | 0.1 |
| XHAND | Unified 40k | 35.4 | 0.0 | 0.1 | 7.3 | **18.7** | **19.0** |
| XHAND | XHAND only | 35.8 | 0.0 | 0.0 | 10.9 | 5.0 | 6.1 |
| Allegro (4 fingers) | Unified 40k | 45.2 | 0.0 | **23.3** | 0.0 | 0.0 | -- |
| Allegro (4 fingers) | Allegro only | 49.3 | 0.0 | 0.5 | 0.0 | 0.2 | -- |

Thumb: every policy collapses on every hand. Where a policy survives losing a finger, it is more often the
unified one (Wuji ring/pinky, Tesollo pinky, XHAND ring/pinky, Allegro index); Shadow collapses for both; the
single-hand policy wins Tesollo index/middle. One run per policy.

## Action noise (`results/action_noise/`, `plots/action_noise_*.png`)

Gaussian noise of std 0-1.0 added to every action, every step, in the normalized action space ([-1, 1] over each
joint's range), then clipped (`eval_niches.py --action-noise`; `run_embodiment_sweep.sub AXIS=actnoise`): the
hand's actuation made noisy. No action smoothing in these runs, so the noise reaches the joint targets directly.

Protocol: as `experiments/02oct_ood_evals/README.md` (goals per episode as in training, each env's first
episode, 240 s, greedy actions, uniform dynamics, palm-frame observations).
