# Embodiment niches, phase 1 (1 Oct 2026)

Hypothesis (user): different embodiments have different niches. Goals/episode is saturated
(40-48 of a 50-goal cap for all 9 left hands, `experiments/30sep_ten_hands`), so measure the
same final policies on other axes and look for rank reversals and non-dominated hands.

`eval_niches.py` -- one hand x one condition per Kit process, 1024 envs x 60 s sim, greedy
policy, no 50-goal cap. `run_hand.sub` runs every condition for one hand; jobs in `.jobs`;
JSON in `debug_outputs/embodiment_niches/`.

Metrics: goals/min, drops/min, goals per drop, time per goal, path efficiency (goal angle /
cube rotation travelled), work per goal and power (PhysX joint forces; the PD-law estimate is
kept as pd_*, ~6x higher because joints sit at the speed cap), action rate, joint acceleration,
fraction of time a joint is at its speed cap, cube speed / spin / acceleration, cube distance
from the palm centre, participation ratio of joint velocities.

Conditions: nominal, cube 40/55/65 mm (density fixed, so mass scales), cube mass x0.5 / x2,
cube friction x0.5, random pushes ~2 g and ~5 g.

Caveats: each policy was trained for goals only, so "efficient" here means efficient by
accident; joint speed caps differ (10 rad/s for allegro/leap/shadow/tesollo, vendor 11.5-15
otherwise), which confounds speed and effort comparisons.

Smoke (allegro, 64 envs x 5 s): ~110 goals/min, a joint at the 10 rad/s cap 98% of the time,
cube rotates ~3.7x the required angle -- the policy is near bang-bang.

## Results (81/81 runs, 1 Oct)

Full tables: `analysis/aggregate.py` -> `debug_outputs/embodiment_niches/summary.json`.

- **Throughput: Allegro wins every condition in absolute goals/min** (113 nominal; best under
  every size, mass, friction and push condition). No robustness niche in absolute terms.
- **Secondary axes do split.** Pareto front goals/min vs work/goal: sharpa, allegro, leap,
  dex3, wuji2 (5 of 9). Work/goal 1.5 J wuji2 .. 14.9 J shadow (10x); path efficiency best
  dex3 (0.31); gentlest cube gen-SHARPA / xhand / wuji2; most joints used leap (PR/J 0.80),
  fewest shadow (0.27). Speed vs reliability: allegro and leap dominate everyone.
- **Cube size is a cliff for all**: 40 mm keeps 74-99% (dex3 99%), 55 mm 28-50%, 65 mm ~0-5%,
  failing by timeouts (held, not turned), not drops. Trained on 45 mm only.
- Hard pushes (~5 g) separate reliability: drops/min gen-SHARPA 17, wuji2 10, dex3 8.4 vs
  leap 0.8, allegro 1.6.
- Confounds: torque limits differ ~50x across vendors (wuji2 0.2-0.3 N.m, allegro 10) and speed
  caps differ; energy winners are also the weakest hands. Policies were trained for goals only.

Bar charts (one per metric, one bar per hand; darker = best): `plots/details/`, made by `analysis/plot_bars.py`.

Main figures: `plots/niche_map.png` (rank of every hand on every metric), `plots/speed_vs_energy.png`
(Pareto front), `plots/robustness.png` (throughput kept per perturbation) -- `analysis/plot_main.py`.
Per-metric bar charts: `plots/details/`.
