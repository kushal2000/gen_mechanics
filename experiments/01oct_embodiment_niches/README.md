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
