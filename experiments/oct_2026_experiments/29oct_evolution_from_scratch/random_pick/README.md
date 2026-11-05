# A random design (30 Oct)

The user's baseline: one of the 30 seed hands drawn uniformly at random (`numpy.random.default_rng(20261030)`, index 22)
-> **`seed_f5_cut_j1`** (5 fingers, palm cut, 3/3/3/2/2 joints). Trained alone exactly like the Claude designer's picks
(`../claude_pick/`): `../long_random_pick.sub`, one continuous 8B-step run, checkpoints at the ends of generations 0, 3, 8, 15, 24 and 31 (rl_games keeps
epoch/636 = perfect squares, plus the last).
One draw is one sample of "a random design"; the RL-only control (`../rl_only.sub`, `../long_rl_only.sub`) covers all 30.
