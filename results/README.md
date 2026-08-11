# Eval results

Each `result.json` records the protocol it ran under. **Do not pool results
across protocols** — the suite's success criterion changed during development,
and a mixed table is silently wrong rather than obviously wrong.

Check before comparing:

- `overrides["termination.success_steps"]` — 10 is current. Results with 1
  predate the dwell requirement being restored and are not comparable.
- `cli_overrides` — non-empty means the run used a deliberately non-standard
  protocol (via `--override`) and must not be pooled with suite-standard runs.
- `goals_reached_mean` — the headline. Older results lack it; `aggregate.py`
  backfills exactly from `goal_pct_mean * n_goals`.

`smoke01/` is a 4-condition harness validation against simtoolreal's pretrained
SHARPA checkpoint, under the old `success_steps=1` protocol. Kept as a record of
the harness working end to end; its numbers are not a performance reference —
that checkpoint trained on a different goal volume (z 0.68-1.05 vs our
0.60-0.95).
