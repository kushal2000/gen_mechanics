# Isaac validation of the locked grammar (2026-10-07)

Local RTX 4090, Isaac Sim 5.1 / Isaac Lab, headless, GPU pipeline (`cuda:0`). Code at 878a3d4-688e9bd (the rim rule of 82fae71 does not change any of these hands' simulation) plus the diagnostic's contact-count line. Population (`make_grammar_population --variant validation`, 11 designs, 64 envs, about 6 per design):

| design | fingers / joints / palm joints | coupled (mimic) | palm followers (mimic) | 0 mm links | filtered pairs |
|---|---|---|---|---|---|
| commercial: Allegro | 4 / 16 / 0 | 0 | 0 | 1 | 7 |
| commercial: SVH | 5 / 19 / 1 | 8 | 1 | 4 | 10 |
| commercial: Wuji v2 (left) | 5 / 20 / 0 | 0 | 0 | 0 | 18 |
| commercial: Shadow | 5 / 21 / 1 | 0 | 0 | 6 | 16 |
| commercial: Inspire | 5 / 12 / 0 | 6 | 0 | 0 | 7 |
| 3+2+1 palm-joint split | 6 / 18 / 2 | 5 | 1 | 1 | 5 |
| 0 mm links | 4 / 19 / 0 | 0 | 0 | 8 | 12 |
| oblique axes | 4 / 12 / 0 | 0 | 0 | 0 | 4 |
| random (3 hands) | 5-6 / 11-18 / 0-1 | 0 | 0 | 1 each | 6-7 |

The three random hands were drawn before the uniform prior (878a3d4) replaced the hand-shaped one.

## Results

`population_diagnostics.py --task-profile hora --random-steps 3000` (overall PASS):

1. **Authored frames match the grammar's FK.** Every body's PhysX pose against `grammar_envelope.authored_fk` (which equals the grammar's FK to 1e-9 on the CPU, `test_grammar_envelope.py`): largest error 0.0005 mm just after reset and 0.028 mm after 20 random-action steps, over 1413 body checks. Joint limits match the population table exactly; ghost joints stay within 3e-6 rad.
2. **Mimic ties hold.** Over 3000 random-action steps the largest tie error was 2.4e-3 rad (SVH: its palm follower and 8 coupled joints at gear 1.1; Inspire 1.9e-3 rad; the 3+2+1 split 1.4e-3 rad). A separate 300-step run gave 1.3e-3 rad.
3. **Rounded-box links and palm plates cook for the GPU.** The Kit log has no PhysX cooking, convex-hull or CPU-fallback warning (the only warnings are the usual extension and GPU-foundation notices). Every link collider is an 8-vertex core hull; every palm plate has at most 64 vertices and 34 faces (checked on the CPU for every design).
4. **No NaN or blow-up.** 3000 random-action steps of the HORA env on 64 envs: no step with NaN, largest |q| 1.89 rad, largest |qd| 5.08 rad/s (the 5 rad/s velocity limit). After 100 zero-action steps all 64 objects still rest at their start point.
5. **Collision filters.** 576 filtered pairs over 64 envs, including pairs that overlap deeply at rest by construction (the 0 mm "pucks" against both neighbours, 6-12 mm; commercial hands' shared-cross-section overlaps, 6 mm). An unfiltered overlap of that depth made PhysX push links apart at over 100 rad/s within one or two steps (the reason for C1); here every joint stayed at or below the 5 rad/s limit and the FK error at reset is 0.0005 mm, so the filters act. The direct check, PhysX's contact report, returned no contacts at all in this env (not even hand-object contacts, with `PhysxContactReportAPI` on every body), so its "0 contacts between filtered pairs" proves nothing; the diagnostic now prints the hand-object count next to it.
6. **Training smoke (joint-token transformer, HORA token config).** `coevolution/train.py --agent rl_games_hora_token_ppo_cfg_entry_point env.hora.token_obs=true env.task_profile=hora`, 1024 envs, 150 epochs: see below.

It ran end to end: Kit boot and scene 38 s for 1024 envs, then 150 epochs in 176 s (1.22 M frames, about 15,000 env steps/s for physics, 9,000 with the policy update), checkpoint saved, clean exit ("MAX EPOCHS NUM"). No NaN; actor loss 0.11 -> 0.01, critic loss 0.66 -> 0.11, KL 0.03 at the end; mean episode length grew from 6 to 275 steps (the object is dropped less often). Rewards are not a learning result at this length.

## Dex1 (sliding jaws)

Sliding joints author as prismatic slots. Dex1 alone (4 envs, 200 random steps) works: FK error 0.0001 mm, limits exact, no NaN, the jaws stay in their -20..+25 mm range. Mixed with revolute hands (Dex1, Dex3, MIDAS, Tesollo in one scene), Isaac Lab cannot create the articulation view (`root_physx_view` is None at `is_fixed_base`): PhysX's tensor API needs every articulation in a view to have the same joint types slot by slot, and Dex1's first finger joints are prismatic where the others are revolute. So a population can hold sliding hands only on their own. Sliding joints are outside the Evolution Rules, so this affects only Dex1 as a probe or held-out hand.

## Not checked here

- The physics grasp test (C5) and any training beyond a smoke.
- Whether coupled joints should be visible to the policy: they are not policy-controlled, so their tokens are disabled like palm followers' (a question for Martin).
