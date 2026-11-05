# Does a looser first goal fix runs that never find the pickup? (15 Oct 2026)

**Setting:** `env.goal.first_goal_tolerance`. It applies to an episode's first goal only, the pickup; every later
goal keeps the 1 cm `success_tolerance`.

**Arms:** 1 cm (control, the setting off) against 3 cm (treatment). Seeds 101-104 in each arm; everything else is
identical. Stagger is on (the default).

## Cases (both known to get stuck before the first goal)

| Case | Known failure | Recipe | Budget |
|---|---|---|---|
| A: primitive hand 0/65/150/270, pen flip | seed 100: no goal in 5,000 epochs | MLP | 1,500 epochs |
| B: full Wuji hand, pen lift | 2 of 3 seeds never escaped (257624/5) | joint transformer, its own learner settings | 3,000 epochs |

## Metrics

- **Escape:** the leader block's 20-epoch mean reaches 1 goal/episode within the budget.
- **Epochs to the first goal.**
- **Final performance, at the strict tolerance:** `analysis/eval_tasks.py` scores at 1 cm whatever a run trained
  with. Training curves count the loose first goal, so they are not compared across arms.

## Decision rule (written before any result)

- **The 3 cm first goal fixes it** if, in BOTH cases, the 3 cm arm escapes on at least 2 more seeds out of 4 than the
  1 cm arm, and its strict-tolerance eval among escaped runs is not lower.
- **It helps but doesn't fix it** if that holds in one case only, or the gain is 1 seed.
- **No evidence** otherwise.

**Caveat:** a loose first goal also pays the goal bonus at 3 cm, a denser reward near the pickup. A gain can come
from that as much as from the easier success test; both are fixes for discovery.

## Runs

| arm | seed | A: primitive flip job | B: Wuji JT lift job |
|---|---|---|---|
| ft1cm | 101 | 287917 | 287918 |
| ft1cm | 102 | 287919 | 287920 |
| ft1cm | 103 | 287921 | 287922 |
| ft1cm | 104 | 287923 | 287924 |
| ft3cm | 101 | 287925 | 287926 |
| ft3cm | 102 | 287927 | 287928 |
| ft3cm | 103 | 287929 | 287930 |
| ft3cm | 104 | 287931 | 287932 |

## Result so far (15 Oct)

**Case A (finished):** all 4 seeds escaped in both arms, so there was no failure for 3 cm to fix, and the rule can't
show "fixes it" here. The flip pick's seed-100 failure was bad luck: the hand learns flip at 1 cm on 4/4 new seeds.
3 cm found the first goal sooner on every seed (epoch 38-181 vs 56-375) and reached 1 goal/episode sooner on 3 of 4
(225-247 vs 271-325); seed 104 was about the same in both (876 vs 814).

**Case B: cancelled early (15 Oct, at the user's request), so it's inconclusive.** State when stopped:

| arm | seed 101 | seed 102 | seed 103 | seed 104 |
|---|---|---|---|---|
| 1 cm | s101: epoch 709, first goal -, >=1 at - | s102: epoch 812, first goal -, >=1 at - | s103: epoch 718, first goal -, >=1 at - | s104: epoch 755, first goal 503, >=1 at - |
| 3 cm | s101: epoch 744, first goal 736, >=1 at - | s102: epoch 799, first goal -, >=1 at - | s103: epoch 414, first goal -, >=1 at - | s104: epoch 409, first goal -, >=1 at - |
