# Methodology: what makes this a hardware comparison

The claim this repo is built to support is narrow and easy to break:

> Holding the task, the arm, the reward, the training budget, and the evaluation
> protocol fixed, hand hardware A retains more of its nominal performance than
> hand hardware B when the world shifts away from the training distribution.

Everything below exists to keep some part of that sentence true. Each section
names the confound it removes and the mechanism that removes it.

## 1. The arm is held constant

Both robots are a KUKA iiwa14 with byte-identical link geometry, inertias, joint
limits, PD gains, and home pose. The Allegro URDF is authored by splicing the
Allegro hand onto the *same* iiwa14 chain used by SHARPA, rather than using the
stock `kuka_allegro.urdf` (which ships on an iiwa7 with different link lengths
and limits).

Mechanism: the arm tables live in one module shared by every `RobotSpec`, and a
test asserts every spec's arm fields are equal. A hand cannot quietly bring its
own arm.

## 2. The reward is byte-identical across hands

The reward function is ported from simtoolreal unchanged. There is no
per-hand tuning and no DoF-count normalization.

Two terms therefore scale with hardware:

- the pre-lift fingertip-approach term sums over fingertips, so a 4-fingered
  hand collects ~20% less of it than a 5-fingered one;
- the hand action penalty sums over hand DoFs, so a 16-DoF hand is penalized
  ~27% less than a 22-DoF one.

These are treated as properties of the hardware under a fixed objective, not as
bugs to correct. Normalizing them would require choosing a normalization, and
every choice is itself a modeling decision that favors some morphology. The
reward is also globally tuned for SHARPA (keypoint scale, lifting threshold,
penalty ratios) — a bias that cannot be removed without per-hand tuning, which
would be strictly worse.

The consequences are handled at the *reporting* layer, not the reward layer
(§4).

## 3. Equal compute, not equal walltime

Allegro (16 hand DoF, 4 fingers, fewer collision shapes) simulates measurably
faster than SHARPA (22 DoF, 5 fingers). A walltime-bounded training job would
therefore hand Allegro more gradient steps — a confound that looks exactly like
"Allegro is better hardware."

Mechanism: study runs are bounded by a fixed `agent.params.config.max_epochs`,
never by SLURM walltime. Wall-clock and steps/sec are recorded as secondary
observations (they are a real, reportable property of the hardware — just not
one that should leak into the training budget).

Everything else is matched too: identical `num_envs`, identical DR profile,
identical algorithm and hyperparameters, ≥2 seeds per hand. The *only*
difference between two training configs is `assets.robot_spec`.

## 4. Report retention, never raw reward

Reward magnitude depends on the shaping constants in §2 and on how far the
success-tolerance curriculum advanced, both of which differ across runs. It is a
debugging signal, not a result.

Headline metrics are hardware-agnostic outcomes:

- `goal_pct` — goals reached / goals in the trajectory
- success rate at a **pinned** tolerance (`termination.eval_success_tolerance`),
  so the curriculum's endpoint cannot inflate one hand
- time-to-first-goal, lift rate, drop rate
- termination breakdown (fall / hand_far / timeout / max_successes)

The cross-hand comparison is **retention**:

```
retention(condition) = goal_pct(condition) / goal_pct(nominal)
gap(condition)       = goal_pct(condition) - goal_pct(nominal)
```

Each hand is scored against its own nominal baseline. A hand that is uniformly
weaker under the SHARPA-tuned reward is not thereby judged to generalize worse —
which is precisely the failure mode §2 would otherwise introduce.

## 5. Held-out sets are frozen, not resampled

Every held-out condition resolves to concrete assets and explicit seeds
committed to the repo, and the *same* set is run for every hand. No condition
may contain a hand-dependent field.

The four axes:

| axis | held-out set |
|---|---|
| object physics | density (baked into the generated URDF), friction, restitution |
| object geometry | unseen pool seed, held-out category, shifted size distributions, real DexToolBench meshes |
| DR settings | `off` / `train` / `hard` profiles |
| goals | unseen trajectory slice, larger delta distance/rotation |

Note on object mass: runtime mass randomization is blocked in this Isaac
Lab/PhysX build (`set_masses` raises), but mass is baked from **density** at URDF
generation time, so held-out mass needs no runtime API at all.

## 6. Statistics

The unit of analysis for a hand-vs-hand claim is the **policy seed**, not the
episode — episodes within a seed are correlated. Compute per-seed means over the
shared condition set, then compare across seeds, paired by condition.

CUDA reductions are non-deterministic, so repeated runs of the same
`(condition, seed)` are not bit-reproducible. Hence ≥10 episodes per condition
and SEM reported everywhere.

## 7. Known residual confounds

Stated here so they end up in the writeup rather than being discovered by a
reviewer:

- **Reward shaping is SHARPA-tuned** (§2). Mitigated by retention reporting, not
  eliminated.
- **Workspace reachability.** The two hands place the palm at different distances
  and orientations from the flange. A hand that cannot reach part of the goal
  volume looks worse at generalizing for a purely kinematic reason. The goal
  volume and table height are held identical across hands; if a hand cannot
  reach, the fix is its mounting transform or arm home pose (mounting choices),
  never the workspace. Reachability is signed off visually before training via
  `genmech/tools/reachability_viewer.py`.
- **Torque authority differs.** Allegro's URDF effort limits and the gains
  sourced from Isaac Lab differ from SHARPA's. This is a legitimate part of "the
  hardware" — but it should be reported, not silently absorbed.
- **Observation dimension differs** (fewer fingertips → smaller obs). This
  changes the input layer by well under 1% of parameters. Documented; not padded.
- **No pretrained parity anchor for Allegro.** SHARPA's port correctness is
  gated by a golden-file replay against simtoolreal. Allegro has no equivalent;
  its correctness rests on the structural spec invariants, a static-hold grasp
  test, and visual inspection.
