# Is the adaptive learning rate needed? (31 Oct)

The user's question after the adaptive-lr runaway (rl_games' AdaptiveScheduler climbed to its 1e-2 ceiling and went
NaN in the long RL-only run and MAP-Elites; every train yaml now caps it at max_lr 1e-3). Same recipe as every 29 Oct run;
only the lr schedule changes. 3,180 epochs each (half the solo budget; the solo curves separate by ~2,000).

- **A, one hand** (MAP-Elites' best `g0018_c00`, alone), seeds 100-102: fixed 1e-4, fixed 3e-4, adaptive capped at 1e-3.
  Reference: its solo runs (`../solo_retrain/`), adaptive with rl_games' uncapped 1e-2 ceiling (lr peaked 1.1e-3-3.8e-3).
- **B, 30 hands** (the seed population under one shared policy, as the RL-only control), seed 100: fixed 1e-4 vs adaptive
  capped at 1e-3.

A fixed lr is `lr_schedule=constant` (anything but adaptive/linear gives rl_games' IdentityScheduler); the
SchedulerBoundsObserver ignores max_lr for it. Compare training goals/episode curves, epochs to 1 goal/episode, and a
final evaluation (`../eval_hands_cap.py`, cap 200).
