"""Per-embodiment evaluation: what one design saw, and how it does elsewhere.

The multi-embodiment training env gives every environment a different robot, so
a training run scores 24,576 designs at once. What it does not do is record the
conditions each design was scored under, or let you look at one of them.

* :mod:`object_pool` / :mod:`dump_assignment` -- reconstruct and record which
  object each design was trained against. Nothing is written during training,
  but the pairing is deterministic, so it can be recovered exactly from a run's
  config.
"""
