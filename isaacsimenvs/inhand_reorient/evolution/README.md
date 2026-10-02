# Evolution pilot: MAP-Elites over grammar hands

Implements the evolution pilot (E-R2') from `project-notes/grammar/plan-rl-grammar-tuning.md`'s
"Revision, 2026-09-27": a MAP-Elites archive over grammar-derived hands, driven by one shared
RL controller whose weights carry across generations. Lessons applied from
`project-notes/grammar/coevolution-reuse-survey.md` (the old co-evolution loop): fitness comes
from graded per-design scoring accumulated during training (`design_scoring.py`), not pooled
episode returns; the archive is real MAP-Elites (cells + elites), not truncation selection; and
the whole run is **one process, one SLURM job** -- no self-resubmitting chains.

## CLI

```
python -m isaacsimenvs.inhand_reorient.evolution.driver \
    --variant G_V1 --seed 0 --generations 20 --designs 64 \
    --probes allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right \
    --num-envs 4096 --epochs-per-gen 200 --fitness train_tail \
    --run-dir outputs/evolution_pilot/run0
```

Run under the isaacsim venv (`.venv_isaacsim/bin/python3 -m isaacsimenvs.inhand_reorient.evolution.driver ...`)
with `PYTHONPATH` clear of anything else (e.g. `env -u PYTHONPATH ...` if a ROS/other Python
environment is sourced in your shell) -- the driver itself never boots Kit (no `isaaclab` import
anywhere in this package), it only shells out to `coevolution/train.py`, which does.

Key flags (see `driver.py`'s `parse_args` for the full list and defaults):

- `--variant NAME`: resolved through `hand_sampler.grammar.variants.NAMED_DISTRIBUTIONS` and
  `G0_SCREEN_VARIANTS` (in that order) -- **never hard-code a variant list here**; a new G0
  variant a concurrent worker adds to either registry is picked up automatically by name.
- `--designs N` (default 64): total population size per generation, including probes.
- `--probes`: comma-separated manifest hand ids, fixed identity every generation, logged but
  never entered into the archive. Default: `allegro_right,dclaw,sharpa_left_on_iiwa14,leap_right`
  (the plan's "allegro_right, dclaw, sharpa ... and one more admissible commercial hand" --
  `leap_right` is the fourth: verified admitted via `population_file.projected_entry` alongside
  the other three, no Kit needed).
- `--fitness {train_tail,eval}`: see "Fitness" below. `eval` currently falls back to
  `train_tail` with a printed warning (see "eval fitness" below for why and what would be
  needed to wire it in for real).
- `--gen-timeout-s`: the `<cap>` in `timeout -k 30 <cap>` wrapping every generation's
  `coevolution/train.py` subprocess.
- `--target-windows` (default 15): `GENMECH_DESIGN_SCORE_WRITE_EVERY` is derived from this so
  at least 10 (the plan's minimum) design-scoring write windows fall inside one generation,
  given `--epochs-per-gen` and `--horizon-length` (16, the yaml's own `horizon_length`).

## Task profile (`--task-profile`, 2026-10-01)

The env has two task specs, selected by `env.task_profile`:

- `isaaclab_repose` (selectable; not adopted, 2026-10-01: the team chose a newer spec): NVIDIA's
  `Isaac-Repose-Cube-Allegro-Direct-v0` spec (Isaac Lab 2.3.2) ported to our hands
  (`isaacsimenvs/inhand_reorient/repose_profile.py`, numbers under `repose:` in
  `coevolution/cfg/task/InHandReorient.yaml`): 30 Hz policy, 10 s episodes, friction 1.0, a
  7.2 cm, 0.216 kg cube, NVIDIA's reward, observation, joint-limit control and reset noise,
  random goals, success at 0.2 rad with no curricula, hand gravity off. A single hand is also
  driven with NVIDIA's Allegro actuator (`repose.hand_*`) and, if it has an entry in
  `repose_hand_poses.json` (allegro_right, sharpa), placed palm-up as NVIDIA's Allegro is; a
  population's designs keep their own gains and palm-up calibration. Its agent configs are plain PPO with NVIDIA's
  hyperparameters: `InHandReposeIsaacLabPPO.yaml` (single hand) and
  `InHandReposeIsaacLabPopPPO.yaml` (the same plus the I41 log-std bound, for populations).
- `legacy`, the env's default: the spec every run before 2026-10-01 used (SAPG-era reward,
  tolerance and palm-normal-axis goal curricula).

The driver defaults to the legacy spec and the SAPG agent, so runs started before the
profile existed resume unchanged; it always passes `env.task_profile=...` explicitly, so a
later change of the env's own default cannot change a resumed run. To evolve under the new spec:

```
python -m isaacsimenvs.inhand_reorient.evolution.driver ... \
    --task-profile isaaclab_repose \
    --agent-entry-point rl_games_repose_pop_ppo_cfg_entry_point
```

With a non-SAPG agent the driver leaves out the SAPG-only overrides
(`expl_coef_block_size`, `central_value_config.minibatch_size`) and caps
`minibatch_size` at NVIDIA's 32768. `--task-profile` enters the resolved config (and its hash)
only when it is not `legacy`. `evaluate_population.py` has no such flag: pass
`env.task_profile=legacy` to it when evaluating a legacy checkpoint.

## AnyRotate profile (`--task-profile anyrotate`, 2026-10-02)

`env.task_profile=anyrotate` ports multi-axis in-hand rotation from M. Yang et al., "AnyRotate:
Gravity-Invariant In-Hand Object Rotation with Sim-to-Real Touch" (CoRL 2024, arXiv 2405.07391v3).
No code is released; `anyrotate_profile.py` (CPU-tested arithmetic), `anyrotate_hooks.py` (env
hooks) and the `anyrotate:` block of `coevolution/cfg/task/InHandReorient.yaml` cite the paper
for every element. Agents: `rl_games_anyrotate_ppo_cfg_entry_point` (single hand,
`InHandAnyRotatePPO.yaml`) and `rl_games_anyrotate_pop_ppo_cfg_entry_point` (population,
`InHandAnyRotatePopPPO.yaml`, plus the I41 log-std bound). With the population agent the driver
uses its 8-step rollout for the minibatch and the scoring cadence.

Fitness under this profile (`design_scoring.anyrotate_fitness_components`) is AnyRotate's own
evaluation: rotations about the commanded axis per episode (Rot) + 0.25 x the fraction of the
episode before termination (TTT / 30 s). Goals reached are banked but not scored. Rot counts
from the end of the 10-step settle phase (from commit 02c0637 on; runs before it also
counted the tumbling of objects that landed or fell in the first 0.5 s).

| Paper element | Paper | Port |
|---|---|---|
| Task | rotate about commanded axis k (Sec. 3.1) | as paper; k in the palm frame, an observation |
| Axis sampling | "arbitrary" axes; distribution not given | uniform on S^2 (`axis_sampling`; `principal`, `z`); optional z-first stage (`axis_curriculum_z_first`). "z" is the palm body's +z by default (`z_axis_frame: palm`); for allegro_right under NVIDIA's pose that runs along the fingers (horizontal), so its z-first runs before 2026-10-02 rotated about a horizontal axis. `z_axis_frame: world_up` uses the palm normal of the palm-up hand, the paper's z |
| Auxiliary goal | current orientation rotated about k at regular intervals; new goal when reached | as paper, 30 deg increment (Table 9); goal position = object position when the goal is made |
| Goal tolerance d_tol | 0.15 teacher, 0.25 student (Table 5), on keypoint distance | 0.15 read as rotation distance in rad: 0.15 m exceeds the paper's own drop threshold (kp_dist > 0.1 m) and a 30 deg goal moves the keypoints 0.017-0.026 m; `goal_tol_metric=kp_dist_m` applies it literally |
| Keypoints | 6, 5 cm along the principal axes; kp_dist mean distance (App. B.1) | as paper |
| r_kp | d_kp / (e^{ax} + b + e^{-ax}), a 50, b 2 (Eq. 3) | as paper; d_kp undefined, `kp_scale` 1.0 |
| r_rot | clip(Delta Theta . k, +-0.025) (Eq. 4) | as paper, per 20 Hz step |
| r_goal | 1 if goal reached (Eq. 5) | as paper |
| r_gc | 1 if >= 2 tip contacts (Eq. 6) | as paper; tip contact = App. F binary contact |
| r_bc | 1 if non-tip contacts ">= 0" (Eq. 7), a penalty | `> 0` (the printed `>= 0` always holds) |
| r_omega | -min(\|w\| - 0.6, 0) (Eq. 8), "penalises exceeding the maximum" | -max(\|w\| - 0.6, 0), the stated intent |
| r_pose | -\|q - q0\| (Eq. 9) | as paper; q0 = the hand's canonical (calibrated) pose; with the grasp cache, the episode's initial grasp pose (HORA's `init_pose_buf`) |
| r_work | -tau^T q-bar (Eq. 10), "controller work done" | -\|tau^T Delta q\| over the step |
| r_torque | -\|tau\| (Eq. 11) | as paper (Isaac Lab's applied torque) |
| r_terminate | -1 x 50 on drop or off-axis (Eq. 12) | as paper |
| Weights | 1, 5, 10, 0.1, 0.2, 0.5, 0.5, 0.1, 0.05, 50 | as paper |
| lambda_rew curriculum | linear in goals/episode over [1, 2] (Eq. 15) | as paper; g_eval = running mean over finished episodes (EMA weight 0.01 per episode, not given) |
| Drop termination | kp_dist > 0.1 (Eq. 12) | as paper |
| Off-axis termination | object rotation axis > 45 deg from k (Eq. 12); k_o undefined | tilt of the object's body axis that lay along k at the episode start or last goal; off for a 10-step settle phase |
| Episode | 600 steps, 30 s, dt 1/60, 20 Hz (Sec. 4) | as paper |
| Action | relative targets, Delta theta in [-0.026, 0.026], EMA eta (Sec. 3.1) | as paper; eta not given, 0.5 |
| Observation o_t | q, f^p, f^o, a_{t-1}, q-bar, c, P, F, k (Table 2; 95 for Allegro) | as paper, palm frame; per-joint fields over the hand's joints, per-tip over its fingertips (masked for populations) |
| Privileged x_t | object pos, orientation, ang. vel., dims, CoM, mass, gravity, goal pos/orientation (Table 3) | as paper, palm frame; gravity / 9.81 |
| Simulated touch | threshold 0.25 N, force EMA 0.5, clip/scale 5 N x 0.6, pose +-0.53 rad x 0.6, masked (App. F) | as paper; one contact sensor per hand body filtered to the object; contact pose = tilt angles of the contact point about the fingertip's x and y axes. Population: a finger's tip contact is read on its last real link, because the envelope's fingertip markers `f*_link5` are colliderless ghosts (`anyrotate_profile.tip_sources`). Before commit d457dc8 population tip contacts were always 0 (r_gc never paid) and real tip contacts counted as non-tip (r_bc) |
| Teacher network | privileged MLP [256, 128, 8] ReLU + policy MLP [512, 256, 128] ELU (Table 5) | one MLP [512, 256, 128] ELU over [o_t, x_t] (no student distillation here) |
| PPO | 8192 envs, rollout 8, minibatch 32768, 5 epochs, lr 5e-3, gamma 0.99, tau 0.95, clip 0.2, KL 0.02, grad norm 1 (Table 5) | as paper; unspecified (normalisation, reward scale 0.01, critic 4, bounds loss) as our other PPO configs |
| Student, TCN, d_tol 0.25 | Sec. 3.2, Table 5 | not ported (teacher only) |
| Objects | capsules and boxes, sizes and mass/CoM ranges (Table 4) | one shape per run, default a 5.25 cm box: without the grasp cache a capsule rolls off the open palm; mass U(0.025, 0.2), CoM U(+-0.01) per env |
| Friction | object and hand 10.0 (Table 4) | as paper (default physics material) |
| Grasp initialisation | cache of stable grasps (App. C) | `anyrotate.grasp_cache` (off by default; see "Grasp cache" below): HORA's generation and reset, AnyRotate's U(+-0.3) joint sampling; without it the object is dropped onto the palm-up hand, joints at the canonical pose + U(+-0.1), 10-step settle phase |
| Hand orientation | random per episode, gravity invariance | stationary palm-up hand (team decision); `hand_orientation_randomization` raises |
| Hand | Allegro, system-identified (App. D), gravity on | each hand's palm-up placement (`repose_hand_poses.json`: NVIDIA's for allegro_right), NVIDIA's Allegro actuator, gravity on |
| Observation noise | joint 0.03, tip pos 0.005, tip orientation 0.01, contact pose 0.0174, force 0.1 (Table 4) | as paper (Gaussian std) |
| PD and disturbance randomisation | x U(0.9, 1.1) gains; disturbance scale 2, p 0.25, decay 0.99 (Table 4) | not ported |
| Gravity curriculum | (not in AnyRotate) | optional: Isaac Lab Dexsuite's ADR gravity (difficulty 0..10, +-1 per episode at >= 1 goal, gravity = fraction x 9.81, 0 below 0.1); `anyrotate.gravity_curriculum` |

## Making the anyrotate profile rotate (A/B, 2026-10-02)

The paper on the goal update: "Targets are generated by rotating the current object orientation
about the desired rotation axis in regular intervals. When a target is reached, a new one is
generated about the rotation axis until the episode ends." The "regular intervals" are angular:
Table 9 ablates the "goal increment intervals" theta = 30/40/50 deg. Nothing mentions a timer.
d_kp (Eq. 3) is not defined, and kp_dist is not normalised by an initial distance. lambda_rew
(App. B.3) "increases with successive goals reached per episode" over [g_min, g_max] = [1, 2]. The
port follows all three. `goal_advance: timer` (A/B option) instead moves a pending goal on by the
increment every `goal_timer_s` (1.5 s) from where it was, so a stalled object falls behind.

allegro_right, 8192 envs, grasp cache, `z_axis_frame: world_up`, z axis only, one factor changed at
a time, 15 min of training (`outputs/isaaclab_repose/ab/`). Reward columns are per-episode sums of
the weighted terms.

| Run | Change | Rot/ep | Goals/ep | TTT (s) | Off-axis | rad/s | r_kp | r_rot | r_goal | contact (gc + bc) | stability (w, pose, work, torque) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (`ar_allegro_zup_gc_8192`) | none | 0.131 | 1.31 | 29.4 | 3.4% | 0.029 | 125.7 | 2.6 | 13.1 | -11.2 | -115.7 |
| kp0 | w_kp 0 | 0.134 | 1.25 | 29.3 | 4.1% | 0.027 | 0 | 2.7 | 12.5 | -9.7 | -99.4 |
| rot20 | w_rot 5 -> 20 | 0.152 | 1.43 | 29.2 | 4.2% | 0.032 | 122.5 | 12.7 | 14.3 | -16.6 | -171.6 |
| act2x | action scale 0.026 -> 0.052 | 0.131 | 1.26 | 29.3 | 3.9% | 0.028 | 122.2 | 1.9 | 12.6 | -11.5 | -110.9 |
| timer | goal_advance timer (1.5 s) | 0.270 | 0.98 | 23.3 | 35.2% | 0.071 | 40.6 | 5.8 | 9.8 | -1.6 | -18.1 |
| lam0 | contact and stability weights 0 | 0.359 | 5.00 | 27.6 | 15.4% | 0.082 | 104.8 | 6.7 | 50.0 | 0 | 0 |
| combo60 | timer + rot20 + lam0 | 0.349 | 1.42 | 22.3 | 41.0% | 0.100 | 39.3 | 30.8 | 14.2 | 0 | 0 |
| combo60 at 30 min | | 0.384 | 2.19 | 24.5 | 29.4% | 0.099 | 44.0 | 34.8 | 21.8 | 0 | 0 |

Removing the keypoint term or doubling the action scale changes nothing. What holds rotation
back is the lambda_rew curriculum: a still policy reaches about 1.2 goals per episode by drift, so
lambda_rew turns on at about 0.2-0.4 before any rotation is learned, and the stability penalties
(mostly r_pose and r_omega, about -100 per episode) then outweigh the rotation reward (2.6). With
those weights at 0, goals per episode rise from 1.3 to 5.0 and rotation from 0.13 to 0.36. A
timer goal raises rotation further, but tumbling ends 30-40% of episodes off-axis. The best
combination plateaus at 0.38 rotations per episode (0.10 rad/s) from 20 min on and misses the 1.0
target (stopped at 30 min). The friction (10) and fingertip-only-grasp hypotheses were not tested.

## HORA profile (`--task-profile hora`, 2026-10-02)

`env.task_profile=hora` ports HORA's training task from its released code (H. Qi et al., CoRL
2022; `hora/tasks/allegro_hand_hora.py`, `configs/task/AllegroHandHora.yaml`,
`configs/train/AllegroHandHora.yaml`). It belongs to the anyrotate family
(`anyrotate_profile.is_anyrotate`): scene, object, palm-up hand placement and actuator, contact
sensors, grasp cache and design scoring are the anyrotate ones, while `hora_profile.py` (CPU-tested)
and the `hora` branches of `anyrotate_hooks.py` replace the reward, termination, observation,
control and timing. Numbers are in the `hora:` block of `InHandReorient.yaml`.

| HORA element | HORA code | Port |
|---|---|---|
| Rotation reward | clip(omega . k, -0.5, 0.5) x 1.0, omega from the quaternion change over the control step | as HORA; k the palm normal of the palm-up hand (`z_axis_frame: world_up`; HORA: world -z) |
| Penalties | object velocity L1 x -0.3, (q - q_init)^2 x -0.3, tau^2 x -0.1, (tau . qdot)^2 x -2.0 | as HORA (q_init = the episode's cached grasp; qdot Isaac Lab's joint velocity) |
| Termination | object z < 0.645 m (start 0.65-0.66 m) or 400 steps | a fall of `drop_dz` (1.5 cm) below the episode's start height, or 400 steps (20 s) |
| Control | targets += a / 24, clamped, 20 Hz (1/120 s x 6); PD torque 3 / 0.1, clip 0.5 N m | as HORA; Isaac Lab implicit PD with the same gains and effort limit |
| Observation | last 3 of [unscaled q + U(+-0.02), targets] (96) + 9 privileged values through an 8-d MLP embedding | the same 96 + 9 values into one MLP |
| Reset | cached grasp (`allegro_hand_grasp.py`), no noise | the anyrotate grasp cache made under this profile's physics (`grasp_cache_gen --task-profile hora`) |
| PPO | lr 5e-3, KL 0.02, horizon 8, minibatch 32768, 5 epochs, critic 4, bounds 1e-4, [512, 256, 128] | `InHandAnyRotatePPO.yaml` (identical numbers) |
| Not ported | object set and scales, friction U(0.3, 3), PD-gain randomisation, disturbance forces, stage 2 | one 5.25 cm box, friction 1.0 |

Results, allegro_right, 8192 envs, grasp cache made under this profile's physics (10,000 grasps in
84 s), z (palm normal), 60 min (`outputs/isaaclab_repose/hora_allegro_8192`, about 100k fps):

| Minute | Rot/ep (20 s episodes) | TTT (s) | Drop | rad/s | r_rot | r_linvel | r_pose | r_torque | r_work | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 0.35 | 3.8 | 100% | 0.57 | 30.1 | -0.8 | -11.6 | -1.0 | -2.8 | 13.8 |
| 30 | 1.64 | 15.6 | 29% | 0.66 | 123.8 | -4.4 | -52.6 | -4.3 | -15.5 | 46.9 |
| 45 | 2.09 | 17.5 | 15% | 0.74 | 147.1 | -5.1 | -53.3 | -5.3 | -20.0 | 63.4 |
| 60 | 2.14 | 17.8 | 14% | 0.75 | 150.8 | -4.8 | -55.9 | -5.5 | -20.0 | 64.5 |

HORA's own simulation numbers (Table 1, within the training distribution): expert TTF 0.85 of a
20 s episode and RotR 233.7 (the per-episode sum of the unclipped omega . k). Here at 60 min TTF is
0.89 and RotR about 0.755 rad/s x 17.8 s x 20 Hz = 268, so the port matches HORA's expert in about
0.36 B agent steps (HORA trains 1.5 B). Rot counts every step, including any turning while the
object falls (at most 1.5 cm before the episode ends).

Population (`pop_v3s_32`, 4096 envs, 30 min, `rl_games_anyrotate_pop_ppo_cfg_entry_point`). The
cache made under this profile's physics (`outputs/stability/pop_v3s_32_hora.grasps.npz`, 254 s,
96,000 candidates, 1551 grasps) has 6 viable designs: the allegro_right, sharpa and leap projections
(1000, 145, 243 grasps) and founders 4, 10 and 22 (19, 11, 121). Last scoring window:

| Design | TTT (s) | Rot/ep | Fitness | with the gravity curriculum: TTT (s) | Rot/ep | Fitness |
|---|---|---|---|---|---|---|
| projected:allegro_right | 0.8 | 0.095 | 0.105 | 7.8 | 0.380 | 0.477 |
| projected:sharpa_left_on_iiwa14 | 1.9 | 0.050 | 0.074 | 6.6 | 0.865 | 0.947 |
| projected:leap_right | 0.4 | 0.031 | 0.036 | 2.9 | 0.055 | 0.092 |
| founder 4 | 0.4 | 0.179 | 0.184 | 14.8 | 0.446 | 0.631 |
| founder 10 | 0.2 | 0.011 | 0.013 | 16.9 | 0.322 | 0.533 |
| founder 22 | 0.5 | 0.013 | 0.019 | 6.8 | 0.130 | 0.216 |
| all 32 (mean, std, max) | | | 0.013, 0.038, 0.184 | | | 0.091, 0.224, 0.947 |

Without the curriculum the shared policy does not learn the population in 30 min: viable designs
hold for 0.2-1.9 s. The 26 non-viable designs keep the drop reset, and under HORA's 1.5 cm drop rule
each of their episodes lasts 0.13 s, so they fill the batch with episodes that carry no signal;
training only on viable designs is the obvious next step. With the curriculum (promotion on >= 0.25
rotations per episode, `gravity_promote_metric: rotations`), the mean difficulty stalled at 0.9-1.0
of 10, so gravity stayed at 0 except for a brief 0.98 m/s^2. The objects float, the hold times and
rotations above are for (near) zero gravity, and the non-viable designs also hold for 10 s (scored
0). The curriculum does not change which designs are viable, since the cache is made under full
gravity.

## Viable-only populations (`--viable-only`, 2026-10-02)

`--viable-only` (needs `--grasp-cache`; off by default) trains only designs with at least one stable
grasp (`evolution/viable.py`, `tests/test_evolution_viable.py`). Per generation:

1. Elites are kept (viable by construction; their grasps stay in the run cache). Probes are
   searched once; one without a stable grasp (e.g. dclaw under HORA physics) is dropped from
   training and listed under `probes_dropped`.
2. New designs are drawn as before (founders in generation 0, else offspring of archive elites or
   an immigrant when a mutation fails) and pre-filtered on the CPU (0.04 s per design):
   `viability_report` admitted, at least `min_tip_contacts` (2) digits, at least one reachable
   fingertip. One finger gives at most one tip contact, so a design failing this can never pass
   the search; requiring two reachable tips would have dropped a viable founder.
3. Survivors are grasp-searched in batches, one `grasp_cache_gen` Kit launch each
   (`--viable-batch-size` 64 designs over `--viable-search-envs` 4096 envs, `--viable-search-rounds`
   40, `--viable-grasps-per-design` 64, at most `--viable-max-batches` 4 launches), until `--designs`
   viable designs are filled. The search cannot run inside the training launch: a design is authored
   into the scene at boot, and the candidates that fail are exactly the ones training must not see.
4. Designs searched before are not searched again (`known_viability`, sha256 -> grasps, in
   state.json). A design without a stable grasp is recorded non-viable (fitness 0, never in the
   archive).

`generations.jsonl` gets a `viability` entry per generation (rates per role, candidates drawn /
pre-filter rejected / searched, grasp-search seconds and launches, non-viable designs with reasons,
dropped probes) and `timings.grasp_search_s`; `generations.csv` adds `offspring_viability_rate`,
`immigrant_viability_rate`, `founder_viability_rate`, `viability_rate`, `candidates_drawn`,
`candidates_searched`, `prefilter_rejected`, `grasp_search_s`, `search_batches`, `n_non_viable`,
`n_unused_viable`, `n_designs_trained`, `short_by` and `probes_dropped` (legacy runs keep their
columns). The same assembly builds a single population without training:
`python -m isaacsimenvs.inhand_reorient.evolution.viable --variants G_V3S,G_V1 --designs 32 --out-dir DIR`
(founders round-robin over the variants).

A 32-design mixed population (`outputs/viable/mixed32`, 4090): 489 founders drawn, 297 rejected on
the CPU (228 single-digit, 69 without a reachable tip), 192 searched in 3 launches (850 s), 29 viable
plus the allegro, sharpa and leap probes. Founder viability: G_V3S 8.6% of drawn (16.4% of
searched), G_V1 3.3% (12.5%).

## Grasp cache (`anyrotate.grasp_cache`, 2026-10-02)

Episodes start from cached stable grasps, as in AnyRotate (App. C) and the HORA code it builds on
(H. Qi et al., CoRL 2022; `hora/tasks/allegro_hand_grasp.py`, `scripts/gen_grasp.sh`).
`grasp_cache.py` (Kit-free: sampling, acceptance, storage, keys; `tests/test_grasp_cache.py`) and
`grasp_cache_gen.py` (generation in a live env, and its Kit entry point).

What HORA does: 20000 envs, canonical grasp pose + 0.25 x U(-1, 1) rad per joint, the object at a
fixed point in the hand, PD targets held (zero actions) for 50 control steps at 15 Hz with gravity
on. A candidate is reset as soon as a fingertip is more than 0.1 m from the object, fewer than 2
fingertips touch it, or it falls more than 1.5 cm; the joint positions and object pose of the
survivors at the end of the hold are saved, 50k per object scale. At reset an episode copies one
saved state with no noise (joint positions and PD targets both set to the saved joint positions);
the pose penalty uses that state as its reference. AnyRotate App. C: object 13 cm above the hand
base in a random orientation, canonical pose + U(-0.3, 0.3) rad, 120 steps (6 s) while gravity
turns through the hand's +-x, +-y, +-z; kept when > 2 tip contacts, no non-tip contact, total
fingertip-to-object distance < 0.2 and the object stays stable; 10000 grasps per object.

How it is adapted here (all numbers are `anyrotate.grasp_*` fields):

| Element | HORA / AnyRotate | Port |
|---|---|---|
| Canonical pose | HORA's Allegro grasp pose | allegro_right: HORA's pose mapped onto the drake joint names (`CANONICAL_GRASP_POSES`); other hands and grammar designs: their calibrated default pose (`palm_up` curl) |
| Joint sampling | + 0.25 U(-1, 1) (HORA), U(-0.3, 0.3) (AnyRotate) | U(-0.3, 0.3); a `grasp_curl_frac` share instead curls every joint to one random fraction U(0.15, 0.85) of its range (+- 0.1), because a design has no hand-made grasp pose |
| Object placement | fixed point in the hand; random orientation (AnyRotate) | the hand's spawn point or, for a `grasp_tip_place_frac` share, the centroid of its valid fingertips after the hand has settled 5 steps at its targets (+- 1 cm); uniformly random orientation |
| Hold | 3.3 s at fixed gravity (HORA); 6 s with gravity along 6 axes (AnyRotate) | 3 s (60 steps at 20 Hz), gravity down (the hand never turns); `grasp_gravity_cycle` turns it through +-x, +-y, +-z |
| Acceptance | see above | HORA's: displacement <= 2 cm and every valid fingertip within 0.1 m at every step, >= 2 tip contacts (App. F force > 0.25 N, mean of the last 3 steps); plus speed <= 0.05 m/s and 0.5 rad/s at the end. AnyRotate's no-non-tip-contact and mean-distance tests are options (`grasp_max_nontip_contacts`, `grasp_max_mean_tip_dist_m`): with a 5.25 cm cube the mean-distance test (0.05 m) rejected 6088 of 6144 allegro candidates |
| Stored | joint positions, object pose (world) | per design: settled joint positions, the PD targets that held the grasp, object pose in the palm frame, tip/non-tip contact counts, mass, displacement |
| Reset | copy, no noise; targets = settled joints | copy, no noise by default (`grasp_reset_*_noise`); targets = the held targets (HORA's choice drops the squeeze that held the object); no settle phase (`grasp_settle_steps` 0) |
| Count | 50k (HORA), 10k (AnyRotate) | `grasp_per_design` (default 1000; 10000 for the single allegro run) |

Keys: a population design by its entry sha256 (`design:<sha256>`), a single hand by its id and a
sha256 over its palm-up calibration, pose-file entry and collider source (`hand:<id>:<sha16>`). A
cache also records the physics it was made with (`grasp_cache.object_signature`: object shape and
size, friction, dt, decimation, hand gains); loading it into an env with other physics raises. A
population's default file is its sidecar `<stem>.grasps.npz`.

A design with no stable grasp (none after `grasp_gen_rounds_without_grasp` rounds) is non-viable:
its envs keep the drop reset (shapes stay fixed) but `design_scoring` scores its episodes 0 and
reports `has_stable_grasp: false` (per design, with `grasps_cached`; snapshot-level
`grasp_cache_viable` and `grasp_cache_non_viable`).

Generate (one Kit launch; all envs of all designs run candidates in parallel):

```
OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_CACHE_PATH=/tmp/$USER/ov_cache WANDB_MODE=disabled timeout -k 30 1800 \
  .venv_isaacsim/bin/python -m isaacsimenvs.inhand_reorient.grasp_cache_gen \
  --population outputs/stability/pop_v3s_32.json --num-envs 4096          # or --hand-id allegro_right
```

and train with `env.anyrotate.grasp_cache=<file>` (add `env.anyrotate.grasp_cache_generate=true` to
generate missing designs at env start instead). The evolution driver's `--grasp-cache` does the
latter in every generation's training launch, on `<run-dir>/grasp_cache.npz` pruned to the current
population (`grasp_cache_prune`), so elites and probes reuse their grasps and only new designs cost
generation time; `generations.jsonl` gets a `grasp_cache` entry per generation.

### Grasp cache results (2026-10-02, RTX 4090)

Generation (HORA's acceptance tests, 3 s hold):

| Hand(s) | Envs | Candidates | Stable | Viable | Time |
|---|---|---|---|---|---|
| allegro_right (HORA pose, object at the fingertips) | 8192 | 278,528 (34 rounds) | 3.7% (10,000 kept) | 1/1 | 99 s (111 s with boot) |
| pop_v3s_32, before the population touch fix | 4096 | 24,576 (6 rounds) | 0 | 0/32 | 12 s |
| pop_v3s_32, first defaults (50% curls, 50% at the fingertips, 6 empty rounds) | 4096 | 43,008 (30 rounds) | 366 | 6/32 | 64 s (196 s with boot) |
| pop_v3s_32, current defaults | 4096 | 101,248 (60 rounds) | 1780 | 7/32 | 136 s (264 s with boot) |

Viable under the current defaults: the allegro_right, sharpa and leap projections (1000, 186 and 445
grasps) and 4 of 28 G_V3S founders (11-76 grasps); the dclaw projection and 24 founders have none.
Placing the object at the fingertip centroid of a randomly curled hand produced 1704 of the 1780
grasps (2.5% of such candidates); placing it at the spawn point produced 1 in 10,051. Of the
allegro grasps, 75% have 2 tip contacts, 22% 3 and 3% 4, and 10% touch no other hand body
(AnyRotate's condition).

Training, 8192 envs (allegro) or 4096 envs (population), z-first, PPO as Table 5. Rot counts after
the settle phase from commit 02c0637 on; the "before 02c0637" runs also counted the landing tumble.

| Run | z axis | Cache | Rot/episode | Goals/episode | TTT (s) | Drop | Off-axis | rad/s |
|---|---|---|---|---|---|---|---|---|
| allegro, 60 min, before 02c0637 (`ar_allegro_zfirst_8192`) | palm +z | no | 0.129 | 1.24 | 24.9 | 16.8% | 4.4% | 0.033 |
| allegro, 20 min, current code (`ar_allegro_zfirst_nocache_ctrl_8192`) | palm +z | no | 0.126 | 1.17 | 25.6 | 13.4% | 4.2% | 0.028 |
| allegro, 60 min (`ar_allegro_zfirst_gc_8192`) | palm +z | yes | 0.100 | 1.21 | 29.4 | 0.2% | 2.8% | 0.021 |
| allegro, 60 min (`ar_allegro_zup_gc_8192`) | world up | yes | 0.126 | 1.39 | 29.6 | 0.0% | 2.4% | 0.026 |
| AnyRotate (paper) | | | 1.8-2.2 | | | | | |

At 19 min the three cached or current-code runs read 0.126 (no cache), 0.103 (cache) and 0.134
(cache, world up) rotations per episode. The cache removes the drops and keeps the object for the
whole episode, but rotation does not learn in 60 min with or without it: every run is flat from 5 min on, so the z stage never ended
and the S^2 stage never started. The per-episode reward is dominated by the keypoint term
(about 111 of a possible 150 for holding the object near its first goal) against 2.5 for rotation
and 12 for goals; `kp_scale` (the paper's undefined d_kp, 1.0 here) is the first suspect.

Population, `pop_v3s_32`, 30 min, z-first (palm +z), current code (`ar_pop32_zfirst_tipfix_4096`
without the cache, `ar_pop32_zfirst_gc2_4096` with the first-defaults cache). The 6 designs viable
under that cache, last scoring window (200 steps):

| | TTT (s) | Rot/episode | Goals/episode | Fitness, all 32 designs (mean, std, max) |
|---|---|---|---|---|
| no cache | 3.9 | 0.011 | 0.01 | 0.013, 0.030, 0.126 |
| cache, first defaults | 9.7 | 0.044 | 0.02 | 0.023, 0.052, 0.189 (26 non-viable designs at 0) |
| cache, current defaults (`ar_pop32_zfirst_gc3_4096`; 7 viable: 9.1 s, 0.058, 0.03) | 9.4 | 0.052 | 0.04 | 0.029, 0.058, 0.198 (25 at 0) |

Aggregate drop rates stay at 98-99% in both, because the non-viable designs keep the drop reset
and finish most episodes.

Evolution driver (`--grasp-cache`, 32 designs, 4096 envs, `outputs/evolution_pilot/gc_timing_*`):
a generation's train.py boot rose from 138 s to 267 s; grasp generation took 134 s in generation 0
(32 designs, 7 viable) and 133 s in generation 1 (18 new designs, 14 reused by sha256, 10 viable).
The time is set by the round cap, not by how many designs are new (every env runs candidates in
parallel); `grasp_gen_max_rounds` trades it against grasps per low-yield design. The pruned run
cache is 395 KB.

## File layout

```
isaacsimenvs/inhand_reorient/evolution/
    __init__.py
    archive.py     # MAP-Elites archive: descriptor, Candidate/Elite, insertion, reporting,
                   # JSON (de)serialization. Pure python + numpy (no isaaclab/torch import) --
                   # CPU-testable under plain pytest, no Kit boot.
    driver.py      # The generation loop: population assembly (hand_sampler + this package's
                   # own scene.grammar_envelope/scene.population_file, also CPU-only), shells
                   # out to coevolution/train.py per generation, polls design_scoring's
                   # per-design snapshot file while training runs, computes fitness, updates
                   # the archive, writes generations.jsonl/.csv and state.json.
    README.md      # this file
isaacsimenvs/inhand_reorient/tests/
    test_evolution_archive.py   # 40 tests: descriptors, insertion/replacement, founder
                                 # tracking, sample_parent, state round-trip/resume.
    test_evolution_driver.py    # 29 tests: variant resolution, population assembly (gen-0,
                                 # offspring, immigrant), train_tail fitness weighting,
                                 # checkpoint discovery, build_train_cmd argument
                                 # construction, state.json round-trip.
experiments/evolution_pilot/
    submit_template.sh          # the exact `cluster submit` command for a cluster run,
                                 # not yet submitted (see its own header for recommended
                                 # --designs/--epochs-per-gen/--generations/--time).
```

Per run (`--run-dir DIR`):

```
DIR/
    state.json             # archive + driver RNG state + generation index + last checkpoint +
                            # config hash -- atomic write, read on startup to resume.
    generations.jsonl       # one JSON object per completed generation (see "Per-generation log").
    generations.csv          # the same, flattened to scalar columns only, for quick plotting.
    gen_<k>/
        population.json     # this generation's population file (population_file schema 0.2).
        train.log            # coevolution/train.py's stdout+stderr for this generation.
        train/               # hydra.run.dir for this generation's train.py -- rl_games nests
                              # its OWN experiment_dir one level below this
                              # (train/<config_name>/{nn,last,best}/...; see find_last_checkpoint's
                              # own docstring), and per_design_scores_rank0.json lives directly
                              # under train/ (design_scoring.py's own convention).
```

## Archive design

`archive.py`'s descriptor is `(digit_bin, joint_bin)`: digit_count clamped to 1-5 (5 bins),
joint_count (an admitted design's total valid envelope joint slots, 0-32) bucketed into
1-5/6-10/11-15/16-20/21-25/26-32 (6 bins) -- 30 cells total, matching the plan exactly.

Each generation, `driver.build_generation_population` assembles the population as: every
current archive elite (re-derived from its stored derivation, to be RE-EVALUATED this
generation under the shared controller's new weights) + offspring/immigrants filling the rest
of `--designs` minus the probe count + the fixed probes. Offspring: a parent is drawn uniformly
from the archive's elites (`Archive.sample_parent`), mutated via
`hand_sampler.grammar.derive.vary(..., operators=EVOLUTION_OPERATORS)`, and accepted only if
`grammar_envelope.admit` admits the resulting model; up to `--max-offspring-retries` (default
16, per the plan) attempts, each `vary` call itself retrying internally up to 32 times. If none
of those land on an admitted design, an immigrant is drawn instead (a fresh
`sample_derivation` + `admit` retry loop, exactly generation 0's own sampling, with a new
founder identity). Generation 0 has no elites yet, so every non-probe slot is a fresh founder.

After training, `Archive.update_generation` folds this generation's candidates (every
re-evaluated elite plus every offspring/immigrant) into the archive with a **per-generation
batch argmax per cell**: whichever candidate submitted THIS generation for a given cell has the
highest fitness becomes (or remains) that cell's elite. This is deliberately not a running max
against a stale previous-generation value -- see `archive.py`'s own module docstring for the
corollary this implies (an elite that is not resubmitted in a generation's batch offers no
protection at all; the real driver never hits this, since every elite is always part of every
generation's population).

## Fitness

**`train_tail`** (default): `driver.compute_train_tail_fitness` reads every design-scoring
write-window snapshot the driver captured while training ran this generation (polling
`per_design_scores_rank0.json`'s mtime -- design_scoring.py itself is out of this branch's edit
scope beyond the new `evolution/` subpackage, so it can only be read from outside, not made to
retain history on disk itself), takes the last `--tail-frac` (default 30%) of those windows, and
computes each design's mean `graded_fitness` weighted by that window's own episode count. A
design with fewer than `--min-episodes-per-design` (default 10) total tail-window episodes is
flagged `low_confidence` (still reported, never dropped -- a low-confidence 0 is informative:
either the design never got a chance to run, or it is genuinely uncontrollable).

Boot/train timings, fps and window count are all derived from the SAME poll loop
(`run_training_subprocess`): `boot_s` is wall time until the first window appears (an
approximation -- design-scoring has no other externally observable "Kit is up" signal),
`train_s` is the remainder until the subprocess exits, and `fps` is
`total_envs * cumulative_steps / cumulative_elapsed_s` read off the LAST window (design_scoring's
own `_score_total_steps`/`elapsed_s` are cumulative since env init, not per-window).

### `eval` fitness (I38)

`evaluate_population.py`'s frozen-policy evaluation was blocked (I38, commit 03d4e06) by two
bugs in how it drove rl_games' vendored player against a `coef_cond`/`mixed_expl_learn_param`
SAPG network (see that file's own updated docstring and the fix at the top of its `run`
function): `BasePlayer.has_batch_dimension` was never set (the script's own manual step loop
never calls `BasePlayer.run()`'s `get_batch_size`), and the network's declared input width is
genuinely one column wider than the env's own observation (a per-env SAPG exploration-block id
the env never produces). Both are now fixed, confined entirely to `evaluate_population.py` --
verified against a real 15-epoch SAPG checkpoint (see the worker report for the exact repro).

`--fitness eval` is accepted by the CLI but **falls back to `train_tail`** with a printed
warning: wiring a full per-generation frozen-policy eval into the archive-selection loop (an
extra Kit-booting subprocess per generation, on top of the training one) was out of this pass's
time budget beyond the fix itself and the one-generation correlation check below. The
`compute_train_tail_fitness`-shaped hook is `run_generation`'s own `fitness_by_source` local; a
follow-up `--fitness eval` implementation would run `evaluate_population.py` here instead (or
in addition), on the SAME `population.json` and the checkpoint just produced.

**train_tail vs eval correlation**: see the worker report for the measured Spearman correlation
on one generation's designs (computed from a real checkpoint + population, both fitness methods
against the identical set of designs).

## State and resume

`state.json` is written atomically (temp file + rename) after every COMPLETED generation:
the archive (`Archive.to_dict()`), the driver's own `numpy.random.Generator` bit-generator
state (so parent selection / mutation / immigrant sampling resume the EXACT same random
stream), the last completed generation index, the last checkpoint path, the last success
tolerance, the id-minter's counter (so ids minted after a resume never collide with ids minted
before it), and a sha256 of the resolved CLI config (mismatches are warned about, not fatal --
see `main`'s own comment). Restarting the driver with the same `--run-dir` loads this file and
continues from `generation_completed + 1`; a generation that was interrupted mid-flight is
simply redone from scratch (no partial-generation state is ever persisted), which the tiny
end-to-end verification run's own kill/resume test exercises directly (see the worker report).

The driver never self-resubmits to SLURM (per this branch's own rule): a cluster job runs it
once, with an honest `--time` limit; the run either finishes within that limit or is killed and
picked up later by rerunning the identical command with the same `--run-dir`.

## Training stability and failure handling (I41)

The first pilot (`outputs/evolution_pilot/pilot/G_V2S_s0`) trained the population with the
single-hand config, and its policy std grew from 1 to about 250 over seven generations, after
which training stopped updating (NaN losses) and generation 7 crashed. Root cause and fix, in
short (details in the I41 worker report):

- `expl_reward_type: entropy` with ONE SAPG block (`expl_coef_block_size == num_envs`) is an
  entropy bonus of 0.5 x `expl_reward_coef_scale` = 0.001 on every sample, whatever
  `entropy_coef` says. Where the PPO gradient on the log-std is noise-dominated (ghost joint
  slots, a population not yet learned), Adam turns that into a steady climb; a larger std shrinks
  the KL per update, so the adaptive learning rate climbs to its 1e-2 cap; weights then grow
  until fp16 activations overflow in the mixed-precision update (NaN loss, GradScaler scale 0).
- `coevolution/cfg/train/InHandReorientPopSAPG.yaml` (`--agent-entry-point
  rl_games_sapg_pop_cfg_entry_point`) is the population config: `expl_reward_type: none` and the
  `inhand_actor_critic` network (`isaacsimenvs/inhand_reorient/policy_network.py`), which projects
  the log-std parameter to `<= logstd_max` (0.0, i.e. std <= 1) before every forward. Its
  checkpoints are interchangeable with the single-hand config's.

Driver behaviour:

- A generation counts as completed only if `train.py` exits 0, at least one scoring window was
  written, and the checkpoint it leaves is finite with a live GradScaler. Otherwise the attempt's
  `train/` and `train.log` are renamed `train_failed_<k>` and the generation is retried
  (`--train-retries`, default 1) from the same population and carried checkpoint, with
  `agent.params.seed` bumped. If every attempt fails the driver exits with status 2 and leaves
  `state.json` at the last completed generation; nothing of the failed generation is scored or
  archived. Rerunning the same command retries it.
- The checkpoint carried into generation k is written to `gen_<k>/carry_checkpoint.pth`, after a
  finiteness check (`NonFiniteCheckpoint` stops the driver), with the GradScaler reset
  (`--no-reset-grad-scaler` to keep it), the log-std handled per `--sigma-on-carry
  {keep,reset,clamp}` (`--sigma-reset-value`, `--sigma-clamp-max`), and optionally the running
  observation/value normalizer counts capped (`--norm-count-cap`, so a new generation's designs
  reshape the normalizer within a few epochs).
- `generations.jsonl` rows add `attempts`, `sigma_carried_in` / `sigma_trained` (finite flag,
  log-std and std mean/min/max, grad scale), `train_metrics` (TensorBoard head/tail means of
  `successes`, `rot_error_mean`, episode length, drop rate, entropy, lr, ...), `window_metrics`
  (episode-weighted goals per episode, time held and graded fitness over all designs, first and
  last `--tail-frac` of windows) and `nonfinite_by_design`. `generations.csv` carries the scalar
  versions.
- The env's non-finite guard (`isaacsimenvs/inhand_reorient/nan_guard.py`) terminates and resets
  any env whose physics state goes NaN/Inf (reason `nonfinite`, logged as
  `episode_final/done_nonfinite`), gives it the drop penalty, sanitises observations, and
  `design_scoring` counts these per design (`nonfinite_resets`, `nonfinite_resets_total`).
- `--train-override KEY=VALUE` (repeatable) passes extra Hydra overrides to every generation.

Recommended pilot settings after I41 (validated locally for 3 generations, see the worker
report): `--designs 32` (28 archive slots + 4 probes; at 64 designs per 4096 envs the shared
controller learned 2-3x more slowly per epoch, and at 16 designs the re-evaluated elites would
soon fill every archive slot and leave no room for offspring), `--epochs-per-gen 900` (the
first pilot's per-generation sample budget), `--agent-entry-point
rl_games_sapg_pop_cfg_entry_point --sigma-on-carry clamp --sigma-clamp-max 0.0
--norm-count-cap 1e6`. With the bounded network the clamp is a no-op safety net; use
`--sigma-on-carry reset` only when resuming from a checkpoint trained with the old config.
