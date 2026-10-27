# 16 Oct: viewing every fixed-palm policy

## `viser_play_all.py`

```bash
.venv_isaacsim/bin/python experiments/16oct_viser_evals/viser_play_all.py [--port 8097] [--envs 16] [--policy 292508]
```

Plays every run trained with the palm's rotation locked (its saved config's `action.wrist_max_speed_ang == 0`) that has
a checkpoint, from `debug_outputs/train_logs/{15oct_pen_repose,15oct_spin_flip_ft,15oct_pen_twist}`: the v2 repose
hands and Wuji, the spin / flip fine-tunes, the locked-wrist twist runs.

- **Sims:** one Kit per kind of scene, booted together at the start (about a minute): all primitive hands in one
  multi-hand scene, each other hand on its own, each twist pen on its own. Picking a policy loads its weights onto its
  hand's envs instantly; only the sim shown runs.
- **Task:** the one-hot (the task the policy is told) and the goals (whose goal generator runs) are chosen separately,
  linked by default. Also live: repose goal sampling and first goal, spin / flip success rules, the pen's start yaw,
  pushes, the twist cap's friction torque. Any change resets the envs and the statistics.
- **Stats:** goals per episode for every hand in the sim shown, since the last change.

Actions are sampled (the SAPG leader block), as in every eval since 12 Oct.

## Design gallery

`plots/designs/`: the designs compared on 15–16 Oct (Claude-chosen cross, its runner-up, three random layouts, the
bunched adversarial hand, Wuji). `00_overview.png` shows them side by side at one scale with their repose peaks;
`NN_<design>_{face,curl45}.png` each. Rebuild with `analysis/make_design_gallery.py`.
