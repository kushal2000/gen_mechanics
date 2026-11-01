# Per-joint noise predicted from each joint's token (31 Oct)

The user's second noise variant, next to `../shared_log_std/` (one learned level for all actions) and the default
(one learned log-std per padded action slot). With `agent.params.network.predict_log_std=true`
(coevolution/networks/joint_transformer.py), each joint's log-std is a linear read-out of the same token that gives
its mean: finger tokens give their slots', and the arm head's input gives the arm's and the palm-cut joint's. Each SAPG
block adds one learned offset. The read-outs are zero-initialised (log-std 0 at the start, like the other variants).
The noise then follows the joint (its geometry, its finger, the state), not the template slot, and is state-dependent,
as in SAC.

Solo check: `predicted_g0018_s10{0,1,2}.sub`, MAP-Elites' best hand g0018_c00 alone, 3,180 epochs, identical to
`../lr_schedule_test/lr_adapt_g0018_s10*.sub` and `../shared_log_std/shared_g0018_s10*.sub` except for the noise.
No log-std cap (as in the other two arms): watch whether a predicted log-std runs away, which would be the
saturated-joint mechanism again.
