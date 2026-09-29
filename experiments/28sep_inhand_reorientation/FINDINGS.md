# In-hand reorientation: why our env never learned to hold the cube, and the fix

Branch `2026-09-28-commercial_embodiments_ihr`. Chronological log of every experiment below the
summary. Run index: `README.md` in this folder.

## Summary

1. **Our env learned to hold the cube and then unlearned it, because dropping paid more.** On the
   Allegro control (585759) the policy holds by epoch 50 (483/600 steps) and abandons it by ~200, for
   good. A dropped episode earned 130–150, a held one ~60. Every reset issues a fresh start + goal = a
   fresh keypoint-PROGRESS budget; a tumbling cube harvests it fast and, at 20°, collects accidental
   1000-point goals. Per step: ~0.9 dropping vs ~0.12 holding.
2. **Fix: `fall_penalty` 200 on the drop step.** Holds for thousands of epochs where the control sits
   at ~150 / 0.98. Bonus 250, a dense proximity term, γ 0.998 and the reference's spawn placement
   each only DELAYED the collapse. (Not established as *necessary*: IsaacLab's env with our progress
   reward learned without one — see the discussion of bonus ratio, tolerance and horizon in the log.)
3. **Reproduced inside IsaacLab's env** (`ourdefault`: their env with everything of ours except the
   observation and the network — timing, hand, frictions, masses, spawn, reset, goals, keypoint
   success test, per-goal timer, reward, learner): same rise-and-collapse (337 at e50 → 53 at e100).
   So the observation and the joint transformer are not needed for the failure.
4. **With the reward fixed, the MLP saturates the task.** Fall penalty 200 + γ 0.998 + MLP + lr 5e-4
   / 5 mini_epochs: ≥45 goals/episode by epoch 1856 at 20° (623419) and by 4624 at "5°" (624497,
   ~1 goal/s; our 5° keypoint threshold admits 5–8.7° by axis). Our lr 1e-4 / 2 mini_epochs also gets
   there, ~5× slower (620496).
5. **The joint transformer does not learn to reorient under any setting tried** (global skip into the
   per-joint head, d_model 128 / 4 heads, both). Diagnosis in progress: under rl_games' adaptive LR
   schedule its lr sits at the 1e-2 cap (20× configured) with KL spikes to 0.2–0.35 and RISING entropy
   (22.7 → 25.7–27.6), whereas the MLP settles at 2e-4–1.7e-3 with falling entropy. Constant-lr runs
   are going.
6. **Cleared as causes** (each a single change on IsaacLab's env, which kept learning): SAPG, γ
   0.99/0.98, 45 mm cube, random start orientation, our PPO hyperparameters, 60 Hz / 10 s, our physical
   Allegro (0.35 vs 0.50 N·m cap, 5× lighter fingertips), our spawn, our progress shape, our keypoint
   metric.

## Code added (all opt-in; defaults reproduce the old behaviour)

- our env: `RewardCfg.fall_penalty`, `RewardCfg.orientation_proximity_scale/_eps`,
  `ResetCfg.in_hand_placement="palm_body_offset"` + `in_hand_palm_offset`
- `JointTransformerNet` `hand_global_skip`
- `run_rank.sh`: `N_HEADS`, `FF_MULT`, `HAND_GLOBAL_SKIP`, `LR_SCHEDULE` env vars
- `common.sh`: `USER_HYDRA` pass-through; `ARCH` in the run name and on the chain; no resubmit after
  a crash (an OOM had chained itself to c10)
- `isaacsimenvs/inhand_isaaclab/`: per-episode observer + in-run viewer, `--sapg`, `--gamma`,
  `--our-hparams`, `--mixed-precision`, and the rungs (see README)
- `pose_viewer.build_pose_viewer_html(fps=)` (default 60, unchanged)

---

## Chronological log



Question: IsaacLab's Allegro in-hand env learns to hold the cube (episode length 19 → 546/600 by
epoch 50) with our own rl_games learner; our env never does (≈130–235/600, fall 0.8–0.9, flat for
8–10k epochs). Method: single-delta "rungs" on their env (break it toward ours) and, later, on ours
(fix it toward theirs). Metric that matters: EPISODE LENGTH / drop rate at matched epochs, not
goals/episode (a policy that throws the cube scores goals early and plateaus).

## Established before tonight (reference env, 8192 envs, matched epochs)

| rung (one delta from reference) | possession learned? | notes |
|---|---|---|
| baseline | yes, by epoch 50 | 1.85 goals/ep @3633 |
| SAPG learner (ours) | yes | ~same as PPO |
| gamma 0.99 | yes | 8.4 goals/ep; trades possession (drop 0.20) |
| gamma 0.98 (= our horizon in seconds) | yes | 3.0 goals/ep; drop 0.68 |
| cube 45 mm (ours, same density) | yes | |
| random object start orientation (ours) | yes | faster goals |
| all four above at once | yes | 1.08 goals/ep @597 |
| our progress SHAPE on their angle | yes, ~15x slower | 1.76 goals/ep @912 |
| our full reward (progress on keypoint residual) | yes, slower still | 282/600 @309, climbing |
| our MLP vs joint transformer on OUR env | neither | both stuck |

Null-policy probe (same Allegro, hold own reset pose, 512 envs): both envs keep the cube ≥98% for
10 s. Spawn differs: reference cube sits over the fingertip centroid on the finger pads
(palm frame 102, 9, 57 mm); ours 32 mm back toward the palm centre and ~60 mm above the tips
(65, −14, 85 mm), i.e. on the palm.

## Tonight

### ~00:30 — placement, tested from both directions
- **Break side:** `isaaclab_ourspawn` (603997): their env, cube moved to our palm-frame spawn
  (65, −14, 82 mm; bottom 51 mm above palm). Probe: lands at (65.1, −13.6, 81.6), 100% held
  by a do-nothing hand. Everything else rung 0.
- **Fix side:** `allegro_tol20_refspawn` (604812): OUR env + OUR reward + our Allegro, identical
  to 585759 except placement, via a new `in_hand_placement="palm_body_offset"` mode
  (`reset.py`; offset (0.102, 0.009, 0.025) = their (x, y), lowest point 25 mm above palm_link,
  per-env support lift). Probe: lands at (102.1, 9.0, 62.1), 98.8% held at 10 s.
- New plumbing: `USER_HYDRA` in `common.sh` (appended when EXTRA_HYDRA is composed fresh, so safe
  on chained links).
- Side observation from the probe: at reset their fingertip centroid sits 9 mm higher than ours
  (10.7 vs 1.4 mm along palm +z). Their reset samples U[0.2·lo + d, 0.2·hi + d]; with asymmetric
  flexion limits the MEAN is d + 0.1(lo+hi), so their fingers start curled up into a cradle.
  Ours lerps 10% toward U(lo, hi), half the offset. Candidate next rung if placement alone
  doesn't close the gap.

### ~00:50 — placement does NOT break the reference
`isaaclab_ourspawn` (their cube at our spawn) learns possession exactly like the baseline:
ep_len/drop e30 504/0.55 (baseline 461/0.62), e50 567/0.20 (546/0.23), e125 566/0.13 (562/0.14).
Eighth single delta on their env that fails to reproduce our failure.

GPUs were fully allocated (24/24); stopped the six probes whose question is answered to make room:
SAPG 592046, gamma.99 593430, gamma.98 595871, cube45 593431, objinit 593432, ourspawn 603997
(logs + checkpoints kept). Kept: baseline 590311, progressrew, keypointrew, combined, and every
run on our env.

Untested so far, i.e. where the answer must be: (1) OUR learner hyperparameters on their env (the
SAPG rung used THEIR lr 5e-4 / 5 mini_epochs / horizon 24); (2) control timing (decimation 2,
10 s episode, no action EMA); (3) the hand model itself (our Allegro URDF vs their USD);
(4) observation; (5) success bonus / keypoint success test; (6) sim params.
- ~01:10: launched allours 606666, ourhparams 606679, timing 606680; stopped combined 599616 (answered) to make room.

### ~01:40 — the two Allegros are not the same hand
Read live from PhysX (env 0) in both envs:

| per finger joint | reference | ours |
|---|---|---|
| torque cap | 0.50 N·m | 0.35 N·m (URDF effort; our actuator group sets no effort_limit_sim) |
| stiffness / damping | 3.0 / 0.1 (+ their gain DR) | 3.0 / 0.1 |
| armature | 0 | 0.001 |
| fingertip link mass idx/mid/ring/thumb | 0.114/0.099/0.101/0.103 kg | 0.021/0.021/0.021/0.100 kg |
| hand total | 2.14 kg | 1.67 kg |
| contact friction | 0.77–1.30 (DR) | 0.5 palm – 1.5 tips |

Joint limits identical. Launched `isaaclab_ourhand` (606808-ish, see squeue): OUR converted USD
(copied to debug_outputs/inhand_debug/our_allegro_usd/) in their env at their palm pose, our
actuator settings. Probe: loads with 0.35 cap / 0.022 kg tips; cube spawns where rung 0 has it;
99.6% held.

Side finding: with THEIR reset pose, OUR hand's fingertip centroid sits at 10.9 mm along palm +z —
same as theirs. So the 9 mm lower tips in our env are OUR RESET JOINT POSE (fingers less curled),
not hand geometry.

### ~02:15 — THE MAIN FINDING: our env learns possession, then correctly unlearns it
Looking at the Allegro control (585759) at low epochs rather than its latest value changed the
question. It is not "never learns to hold the cube":

| epoch | ep_len | fall | goals/ep | keypoint rew | bonus | TOTAL / episode |
|---|---|---|---|---|---|---|
| 30 (random-ish) | 54 | 1.00 | 0.068 | 61.4 | 68.2 | **128.8** |
| 50 | 416 | 0.32 | 0.028 | 38.7 | 28.3 | **61.5** |
| 75 | 483 | 0.27 | 0.027 | 38.6 | 27.0 | **59.6** |
| 200 | 217 | 0.90 | 0.055 | 63.5 | 55.3 | 116.1 |
| 500 | 158 | 0.98 | 0.073 | 69.0 | 73.3 | 140.0 |
| 3000 | 165 | 0.96 | 0.078 | 76.5 | 78.3 | 152.5 |

**A policy that drops the cube earns 2× the return of one that holds it.** PPO learns to hold by
epoch 50 and then — correctly, for this reward — abandons it around epoch 150–230 and never comes
back (best-ever episode length across 11,669 epochs: 497). The transformer and MLP arms show the
same rise-then-fall. Both reward terms rise when the cube is dropped: a tumbling cube banks keypoint
PROGRESS on the way out, and with a 20° tolerance it passes near the goal often enough to collect
frequent 1000-point bonuses. One goal = 7.5× a whole holding episode's return.

Why placement matters (it does, but through the reward): our spawn drops the cube ~60 mm onto the
palm, so an untrained hand makes it tumble — random policy 0.068 goals/ep, return 128.8. With the
reference placement (604812) the cube starts cradled on the fingers, tumbles less: random policy
0.026 goals/ep, return 88 vs ~72–82 for holding. The pull toward dropping is much weaker, and
604812 has not collapsed through epoch 229 (ep_len 436, fall 0.35) where the control had fallen to
165 / 0.97. Whether it merely delays the collapse is still open.

Why the reference env doesn't show this: its dense term 1/(θ+0.1) pays every step the cube is in
the hand, so holding always out-earns dropping; and its bonus is 250×dt = 8.3 per goal, 29% of an
episode, not 750%. Even our full reward on their env (keypointrew) keeps climbing — its bonus stayed
theirs, and its success test is their 5.7°, so accidental goals during tumbles are ~40× rarer.

**Fix tests launched on OUR env** (identical to 585759 except one reward knob):
- `allegro_tol20_bonus250` 607320 — reach_goal_bonus 1000 → 250
- `allegro_tol20_dense` 607321 — new dense term `orientation_proximity_scale` 0.5 /(x + 0.1)
  (new RewardCfg fields, default 0 = old behaviour; logged as episode_cumulative/proximity_rew)
Prediction: no collapse after epoch 150. Also stopped (answered): progressrew, timing (their env with
decimation 2 + 10 s learns fine: 592/0.21 at e50), MLP arm.
(Cosmetic: the two new STUDY_IDs came out with a doubled suffix, e.g. `..._dense_dense`.)
- ~02:25: ourhparams (519/0.17 @50) and ourhand (583/0.18 @50) both learn like the baseline -> our PPO hyperparameters and our physical Allegro are exonerated; stopped both. allours is broken (72/0.99 @150). Launched kpspawn (their env: our reward + our spawn only). refspawn (our env) still holding at e334: 460/0.27.

### ~03:20 — the three fixes only DELAY the collapse; bonus is not the driver
Our env, Allegro, 20° — ep_len / fall at matched epochs:

| epoch | control 585759 | ref placement 604812 | dense 0.5 607321 | bonus 250 607511 |
|---|---|---|---|---|
| 75 | 483 / 0.27 | 485 / 0.33 | 495 / 0.23 | 497 / 0.23 |
| 200 | 217 / 0.90 | 431 / 0.33 | 377 / 0.55 | 355 / 0.61 |
| 300 | 176 / 0.95 | 460 / 0.25 | 354 / 0.60 | 246 / 0.91 |
| 700 | 171 / 0.95 | 280 / 0.85 | — | — |

- Reference placement: collapse delayed ~400 epochs, then happens (213/0.96 at 776).
- Bonus 250: collapses like the control while the bonus is only 18 of an 88 return → the jackpot
  is NOT the driver; keypoint PROGRESS rises 38 → 73/episode as the policy starts dropping.
- Dense proximity 0.5: slows the decline, doesn't stop it.

Per-STEP, not per-episode: every reset issues a fresh start + goal = a fresh progress budget, and a
short dropped episode harvests it fast (control: ~0.9 reward/step dropping vs ~0.12 holding). Only
FUTURE reward can make holding worth it, and γ=0.99 at 60 Hz sees ~1.7 s.

Their env: `kpspawn` (our reward + our spawn) tracks `keypointrew` exactly (197 vs 192 at e200) →
spawn doesn't interact with the reward there. `allours` (everything, incl. γ 0.98) broke. Hypothesis:
**our progress reward × our short horizon**. Two-sided test launched:
- `isaaclab_kpgamma98` 610829 — their env, our reward + γ 0.98. Prediction: stalls.
- `allegro_tol20_gamma998` 610830 — our env, γ 0.99 → 0.998 only. Prediction: no collapse.
Stopped (answered): allours, kpspawn, bonus250, refspawn.

### ~04:30 — horizon: decisive on their env, not sufficient on ours
- **Their env, our reward + γ 0.98** (`kpgamma98`): STALLS — 67/0.99 from e150 to e300, where our
  reward with their γ 0.998 climbs to 274/0.77. So on their env, progress reward × short horizon
  reproduces our failure.
- **Our env, γ 0.998 only** (610830): still collapses, just later (421 @50 → 281/0.85 @400).
- Terminations are correct in our env (fall → `terminated`, not a timeout; value_bootstrap only
  applies to timeouts), so the critic does not bootstrap through a drop.

Per-step economics on our env explain why each single fix only delays:

| run | holding (e75) reward/step | dropping (late) reward/step |
|---|---|---|
| control | 0.12 | 0.60 |
| dense 0.5 | 0.43 | 0.99 |
| γ 0.998 | 0.14 | 0.43 |

The dense term pays ~equally per step in both regimes (0.30 vs 0.31: similar orientation error), so
it can only protect holding via what a drop FORFEITS — small at γ 0.99. γ 0.998 has the horizon but
nothing much per step to forfeit. Hence three new fix tests on our env (otherwise = 585759):
- `allegro_dense_g998` 612716 — dense 0.5 + γ 0.998 together
- `allegro_fallpen200` 612717 — new `fall_penalty` 200 charged on the drop step (γ 0.99)
- `allegro_fallpen200_g998` 612718 — fall penalty 200 + γ 0.998
New RewardCfg field `fall_penalty` (default 0; logged as episode_cumulative/fall_penalty). Verified
DirectRLEnv.step calls _get_dones before _get_rewards, so the penalty lands on the drop step.
Stopped (answered): dense 607321, gamma998 610830, kpgamma98 610829.

### ~05:20 — FIXED: all three stop the collapse
Our env, Allegro, 20° — ep_len / fall / goals-per-episode at matched epochs:

| epoch | control 585759 | dense 0.5 + γ .998 | fall penalty 200 | fall penalty + γ .998 |
|---|---|---|---|---|
| 75 | 483 / .27 / .027 | 494 / .21 / .024 | 509 / .17 / .025 | 500 / .18 / .026 |
| 150 | 402 / .54 / .038 | 464 / .30 / .035 | 490 / .22 / .025 | 498 / .20 / .032 |
| 200 | 217 / .90 / .055 | 453 / .30 / .027 | 461 / .28 / .033 | 448 / .48 / .040 |
| 300 | 176 / .95 / .063 | 461 / .25 / .031 | 459 / .25 / .026 | 441 / .47 / .052 |
| 500 | 158 / .98 / .073 | — | 461 / .30 / .034 | — |

The control has collapsed by epoch 200; all three fixes hold at ~450–500 / 0.3 through epoch ~500.
A fall penalty of 200 ALONE is sufficient, even at γ 0.99. Cause confirmed from both directions.

Still open: whether they go on to REORIENT. Goals/ep are 0.03–0.045 — the reference was at a similar
level at this stage (0.047 at e100, 0.12 at e400), so this needs a few more hours to read.

### ~06:30 — holding solved; reorientation is the next gap
At ~2100 epochs (control: 118–158 / 0.98 the whole way):

| run | ep_len | fall | goals/ep |
|---|---|---|---|
| dense 0.5 + γ .998 | 506 | 0.21 | 0.044 (flat) |
| fall penalty 200 (γ .99) | 460 | 0.44 | 0.067 (eroding since ~e1300) |
| **fall penalty 200 + γ .998** | **553** | **0.14** | **0.089 (climbing from 0.040)** |

Best fix = fall penalty + γ 0.998. But reorientation is slow: 0.09 goals/ep at 20°, while OUR reward on
THEIR env (keypointrew) reached 1.26 goals/ep at 5.7° by e1084. So something else in our env slows
reorientation. Launched three single deltas on the best base (fall penalty 200 + γ .998):
- `allegro_fpg_theirhp` — their PPO step: lr 5e-4, 5 mini_epochs
- `allegro_fpg_mlp` — MLP instead of joint transformer
- `allegro_fpg_curl` — their curled-finger reset (reset_dof_pos_random_interval_fingers 0.1 → 0.2)
Stopped (answered): reference baseline 590311, keypointrew 601929, fallpen200 612717.

### ~08:30 — with the reward fixed, the JOINT TRANSFORMER is what slows reorientation
Single deltas on the fixed base (fall penalty 200 + γ .998), ep_len / fall / goals-per-episode:

| epoch | base | + their lr/mini_epochs | **+ MLP** | + curled reset |
|---|---|---|---|---|
| 300 | 441/.47/.052 | 491/.40/.045 | 583/.01/.057 | 489/.24/.032 |
| 600 | 510/.34/.055 | 543/.15/.046 | 619/.05/.154 | 510/.18/.024 |
| 1000 | 537/.17/.049 | 556/.13/.047 | **619/.02/.166** | 542/.11/.021 |
| 1500 | 560/.16/.080 | — | **626/.05/.223** | 554/.10/.026 |

MLP: near-perfect holding and 3–4× the goal rate at matched epochs; also ~2× the epochs per hour.
This REVERSES the earlier "network is not the cause": that comparison was made under the broken reward,
where both networks collapsed. Their PPO step size and the curled reset don't help.
Remaining gap: MLP ~0.2 goals/ep at 20° vs our reward on their env 1.26 at 5.7°.
Launched on base+MLP: `allegro_fpgm_theirhp` (their lr/mini_epochs), `allegro_fpgm_dense` (dense 0.5).
Stopped (answered): base 612718, curl 620497, dense+g998 612716.

### ~09:40 — reorientation cracked: MLP + their PPO step size
On the fixed reward (fall penalty 200 + γ .998), ep_len / fall / goals-per-episode:

| epoch | MLP (620496) | **MLP + lr 5e-4 / 5 mini_epochs (623419)** | MLP + dense 0.5 (623420) |
|---|---|---|---|
| 200 | 526/.31/.095 | 582/.09/.166 | 575/.06/.046 |
| 300 | 583/.01/.057 | 645/.13/.463 | 609/.06/.141 |
| 500 | 616/.01/.098 | 813/.15/1.23 | 598/.02/.081 |
| 700 | 636/.03/.207 | 1145/.33/3.58 | 592/.03/.049 |
| 1000 | 619/.02/.166 | 2078/.66/13.3 | 593/.02/.048 |
| 1155 | — | **3175/.56/30.1** | — |

Episode length exceeds 600 because each goal resets the 600-step budget. 30 independent uniformly random
goals in 3175 steps (53 s) cannot come from holding still — it is real reorientation. Transformer + their
step size (620495) stays at 0.056 goals/ep after 1898 epochs, so the transformer is a hard bottleneck.
The dense term makes the MLP content to hold (drop 0.007) without reorienting.
Launched the recipe at 5° (624497). Stopped: 623420, 620495.

### ~10:40 — recipe confirmed at both tolerances
| epoch | 20° recipe (623419) ep_len/drop/goals | 5° recipe (624497) ep_len/drop/goals |
|---|---|---|
| 500 | 813/.15/1.23 | 633/.06/0.194 |
| 1000 | 2078/.66/13.3 | 676/.08/0.345 |
| 1300 | 3348/.40/38.6 | 684/.06/0.342 |
| 2000 | 2807/.15/45.9 | — |
| latest | e2641: 2466/.14/**46.0** | e1449: 701/.07/**0.406** |

The MLP with OUR step size (620496) also took off eventually: 2.37 goals/ep at e5000, 2.88 at e6013 —
the step size speeds it up ~4×, it isn't required. The transformer never did under any setting tried.

Jobs of mine still running at the end of the night: 623419 (20° recipe), 624497 (5° recipe),
620496 (MLP, our step size), 620432 (the control 585759, auto-chained to link 2), and the
long-running user runs from before (577945, 581377, 583967). Nothing committed.

### Next day — making the joint transformer work
Diagnosis: hand-only => no arm head, so every action is `mu_head(joint_token)`; the goal
(`keypoints_rel_goal`) and object pose/velocity are GLOBAL-only fields (layout.py), reachable by a joint
only through one attention head at d_model 64. The MLP sees them directly.
Added `hand_global_skip` to JointTransformerNet (default off; existing tests pass): the shared per-joint
head reads [joint token, global token, raw global] — 199 inputs instead of 64 for the Allegro.
run_rank.sh now takes N_HEADS / FF_MULT / HAND_GLOBAL_SKIP env vars (defaults 1 / 2 / false = unchanged).
Launched on the recipe (fall penalty 200 + γ .998 + lr 5e-4 / 5 mini_epochs), Allegro 20°:
T1 `allegro_tf_skip`, T2 `allegro_tf_big` (d128, 4 heads, ff 4), T3 `allegro_tf_skip_big`.
Control: 620495 (transformer, same recipe) 0.056 goals/ep at e1898. Stopped saturated MLP runs 623419, 620496.

### `ourdefault` — their env moved all the way to our default (except obs + network)
New `isaacsimenvs/inhand_isaaclab/ourdefault.py` (command subclass with keypoint success test,
Haar-uniform goals and per-goal timer; our reward terms at our magnitudes; drop-from-palm; our reset
events) + `InHandIsaacLabOurDefaultCfg` + `--mixed-precision`. Probe-verified identical to our env:
5.00° palm, spawn (65.0,−13.8,84.8) mm, reset fingertip centroid (97.3,−20.1,1.3), 98.0% null hold,
0.35 N·m / 0.021 kg tips / armature 0.001, friction 0.5/1.5/1.0, cube 0.0456 kg, DR off.
768-env smoke at e400 already looks like our collapse: ep_len 132, drop 0.94, keypoint 53 + bonus 60.
Launched full 12288-env run.
- T2 (d128, 4 heads) and T3 (+skip) ran out of GPU memory at minibatch 114688 (T3 at epoch 1, T2 at
  epoch 33). Relaunched with GLOBAL_MINIBATCH 57344 (divides the SAPG-augmented 229376 4×): 631581,
  631582 — note 4 gradient steps/epoch instead of 2, so not a pure capacity comparison.
- T1 (skip, d64) at e410: 317 / drop 0.69 / 0.043 goals — possession ERODING despite the fall penalty,
  while the transformer without the skip (620495) held (568 / 0.09) but never scored. The skip made it
  try to move the cube without learning to succeed.

### `ourdefault` reproduces our collapse; chain bug fixed
- `ourdefault` (their env + their obs + their MLP, everything else ours): 337/0.50 at e50 → 53/1.00 at
  e100 → 150/0.99 at e1286. Same rise-and-collapse as our env, faster. So our observation and the
  joint transformer are NOT needed to produce the collapse; env/reward/learner settings are enough.
- Runaway chain: T2's first submission OOM'd at epoch 33 after writing a checkpoint; `run_and_chain`
  chains after any exit that left a checkpoint, so it resubmitted c02..c08, each OOMing ~9 min in, with
  duplicate c06 and c07. Cancelled 632547 and 633301. `common.sh` now refuses to chain after a nonzero
  exit that hasn't already chained via the USR1 time-limit trap.
- Transformer on the fixed recipe: no variant scoring yet (T1 skip e1070 0.036 goals/ep; T2 d128/4 heads
  e292 0.022; T3 e274 0.045) vs MLP 0.17 at e200 / 1.2 at e500.

### The transformer's real problem: the adaptive LR runs away
Logged lr / KL / entropy on the fixed recipe:
| run | lr during training | KL per update | entropy e1 → latest |
|---|---|---|---|
| MLP 623419 | 2e-4 – 1.7e-3 | 0.012–0.024 | 22.7 → 21.4 (sharpening) |
| transformer 620495 | pinned at the 1e-2 CAP | spikes 0.19–0.35 | 22.7 → 25.7 (noisier) |
| T2 d128/4h 631581 | 1e-2 from e50 on | spikes 0.14 | 22.7 → 27.6 |
The transformer's early updates barely move KL, so rl_games' AdaptiveScheduler (×1.5 whenever
KL < threshold/2, cap 1e-2; schedulers.py:19-32) drives it to 20× the configured lr, and the policy
grows steadily noisier — it can hold the cube (fall penalty) but not coordinate reorientation.
Added `LR_SCHEDULE` env var to run_rank.sh (default adaptive). Launched constant-lr transformer runs:
`tfc3e4` (3e-4), `tfc1e4` (1e-4), `tfc3e4_skip` (3e-4 + hand global skip).
Stopped: ourdefault (answered), T1 skip, T3 (both confounded by the lr runaway).
