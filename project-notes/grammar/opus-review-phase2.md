# Opus review: Phase 2 grammar-to-simulator adapter (51be731..f81c3ca)

Date: 2026-09-27. Read-only; CPU checks on a `git archive` of f81c3ca plus `outputs/grammar_populations/test16.json`. Scratch scripts: session scratchpad check1-4.py.

## Correct
- **Permutation direction.** `perm[c] = SLOT_NAMES.index(phys_names[c])`, then `table[:, perm]` (scene_utils.py:324, 343; reward_utils.py:103).
- **Tensor order.** Actions, targets, soft limits, joint pos/vel and prev_actions are all in PhysX column order.
- **Ghost actions** are clamped to (0, 1e-8).
- **Single-hand path** is semantically unchanged.
- **Spawn gather** by `design_idx[env_ids]` is correct across resets.

## Confirmed bugs
1. **High: fingertip bodies are one link short.** `f{f}_link5` sits at the base of the last real link, because ghosts keep an identity slot origin.
   - Effect: fingertip observations are off by 15-84 mm on every finger.
   - Fix: translate the first ghost by (0,0,L_last) and mask fingertip observations by `fingertip_valid` (as in the old sampler, build.py `link_frames`).
2. **High: drop check along palm-local z.** The below-palm drop check uses palm-local z, while `palm_up` spawns along the design normal.
   - Effect: some designs drop-terminate on step 1. 9-10% of filtered designs sit near the threshold.
   - Fix: measure along world up or the design normal.
3. **Medium-high: per-design rewards cannot be recorded.**
   - `DesignRewardWrapper` reads `scene_record.robot_design_index` by attribute, but `scene_record` is a dict.
   - `extras["successes"]` is stale by one episode.
   - Fix: key per-design results by `scene_record["design_idx"]` and record `_successes` at done time.
4. **Medium: the rest-overlap filter is not enforced and not faithful.**
   - `load_population` does not re-check it. `--variant sampled_only` skips it. Projected hands are never filtered (SHARPA overlaps by 20 mm).
   - It models capsules as [0, L] with caps, but the authored capsule spans [0, L].
   - True rejection is 59-64%, not 75%.
   - It ignores the root capsule and pairs that meet across ghosts.
5. **Medium: provenance does not pin the derived hand.**
   - Grammar version and envelope are not checked at load.
   - No digest covers the derived tables, so changes to derive, canonicalize or `palm_up` alter populations under the same hash.

## Plausible risks
6. There is no `is_homogeneous` assert.
7. No CPU test covers the permutation, masks, spawn or carrier wiring. The Kit FK script is uncommitted.
8. Carrier gains apply to real carriers too.
   - SHARPA and carrier designs get a palm DOF 128x stiffer than their fingers.
   - maxForce of 1 Nm saturates the drive, which explains the identical 6x stiffness result.
9. Ghost prev_actions are observed but unpenalized, giving the policy a free memory channel. Mask actions in `pre_physics_step`.
10. The env-0 rotation can silently drop a design; it is unnecessary.
11. Other gaps:
    - Only the root capsule is authored; palm cells get no geometry. This bears on I30's cube support.
    - Robot colliders have no contact offsets or friction.
    - `population.drive` is unused and inconsistent.
    - Per-env defaults would revert to the template if the articulation re-initializes.

## Test adequacy
Add a fake-env CPU test with:
- shuffled joint names;
- 3 designs across 7 envs;
- asserts on masks, defaults and spawn rows;
- zero penalty on ghost columns;
- fingertip FK equal to the real tip;
- no spawn-triggered termination.

Commit the Kit FK check.

## For E-R1 per-design success
- Key results by `design_idx`.
- Record `_successes` at done time.
- Report the per-design mean with its episode count, aggregated as a mean over designs.
- Normalise per unit time, or use a fixed-horizon eval.
- Log the curriculum state with each window.
- Expect unequal env counts per design.
- Log reach, spawn z and carrier-ness as covariates.
- Fix items 1, 2 and 8 before comparing grammars.
