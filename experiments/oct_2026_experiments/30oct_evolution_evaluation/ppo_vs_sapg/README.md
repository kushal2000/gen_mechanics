# SAPG vs PPO (31 Oct)

The user's question. Every run so far uses SAPG (rl_games fork: 6 policy blocks with learned exploration coefficients,
leader-follower off-policy data, per-block action noise `fixed_sigma: coef_cond`). Plain PPO here = the same network,
learner and task with `expl_type=none use_others_experience=none` and one action noise (`fixed_sigma=fixed`).

The SAPG condition is `../lr_schedule_test/lr_adapt_*` (adaptive lr capped at 1e-3, 3,180 epochs); the PPO runs copy those
files and add only the three overrides above:
- one hand (MAP-Elites' best `g0018_c00`, alone), seeds 100-102: `ppo_g0018_s10{0,1,2}.sub`
- the 30 seed hands under one shared policy, seed 100: `ppo_seed30_s100.sub`

Compare training goals/episode curves, epochs to 1 goal/episode, and a final evaluation (`../eval_hands_cap.py`, cap 200).
