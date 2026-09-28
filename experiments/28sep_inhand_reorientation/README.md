# In-hand reorientation: is the success threshold the binding constraint?

Hand-only in-hand cube reorientation has been running since 24 Sep and has never
scored a goal. `successes` reads ~0.0001. The threshold is one candidate
explanation and the cheapest to test, so this folder tests only that: **one
design, two thresholds, nothing else different.**

| run | threshold | `success_tolerance` |
|---|---|---|
| `tolerance_5deg.sub` | 5° | 0.00339979 m |
| `tolerance_20deg.sub` | 20° | 0.01353454 m |

```bash
sbatch experiments/28sep_inhand_reorientation/tolerance_5deg.sub
sbatch experiments/28sep_inhand_reorientation/tolerance_20deg.sub
```

Both use **gen-SHARPA** (`sharpa_capsule.json`, hand-only), 15 000 epochs per link,
chaining up to 10 links. Co-evolution is deliberately not used: it would confound
the threshold with which designs got selected under it.

20° is looser than anything published for this task — OpenAI's 0.4 rad (22.9°) is
the nearest. That is the point. If goals still read zero at 20°, the threshold was
never the constraint and the problem is upstream.

## The threshold is set in degrees, not metres

`common.sh` takes `SUCCESS_TOLERANCE_DEG` and converts:

```
success_tolerance = edge · √3 · sin(θ/2)
```

Two things about this are easy to get wrong, and both are pinned by
`isaacsimenvs/inhand_reorient/tests/test_orientation_metric.py`:

**`keypoint_scale` cancels and is not a difficulty knob.** It multiplies the
tolerance (`observations.py:269`) *and* is already inside the keypoint radius
(`reset.py:233`), so the threshold angle depends only on the tolerance and the
cube edge. Verified at `keypoint_scale` 0.5, 1.0, 1.5 and 3.0 — all give 5.7362°
for the shipped 0.0039. Changing it moves nothing and looks like it should.

**The threshold is a range, not an angle.** `KEYPOINT_CORNERS` is four points
forming *two antipodal pairs, all coplanar* — only two independent constraints. So
the same rotation reads differently depending on its axis, by a factor of
`1/sin(35.2644°) = 1.732`. For the shipped 0.0039:

| rotation axis | threshold |
|---|---|
| most sensitive | 5.7362° |
| cube corner | 6.0845° |
| cube face | 7.0269° |
| cube edge = least sensitive | 9.9437° |

Bisected in sim over 32 envs, matching the closed form to 0.0000°, with under
0.0001° spread across envs — so it is a function of the axis *relative to the
cube*, and the goal orientation is irrelevant. The figures above are the **most
sensitive** axis; multiply by 1.732 for the loosest.

Consequence: the cube's six geometrically equivalent edge axes split into two
classes under this metric, 5.74° and 9.94°. **The metric breaks the cube's own
symmetry.** A non-degenerate keypoint set (all 8 corners, or a true tetrahedron)
would make it isotropic.

## What this experiment does not test

- **Cube symmetry.** A plain box has 24 orientations that look identical and the
  reward accepts exactly one, so the policy can be visually aligned and score
  nothing. IsaacLab's DexCube has distinguishable faces, so for them the
  orientation genuinely is unique; ours does not. This is a larger effect than the
  factors above and is untested.
- **Whether the object can be held at all.** With the joints locked at 0, 90.6% of
  envs never drop the cube over 6.7 s, and every drop happens in the first second
  — the cube settles between the palm and the open fingers and stays. Holding is
  free at the current 5° palm tilt; the task is entirely about turning it.
- **The reward shape.** `keypoint_reward` pays for new best-ever progress, so a
  near miss permanently exhausts the term for that goal.

## Inspecting it

`coevolution/eval/play_inhand.py` serves a viser page with a **goal probe**: place
the object a chosen angle from the goal about a named axis with physics frozen, and
read the residual, the true angle, and the angle to the nearest identical-looking
pose. That last number is where the symmetry cost shows up.

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m coevolution.eval.play_inhand \
    --checkpoint <run>/rank_0/<name>/nn/<ckpt>.pth --port 8082 --success-steps 1
```
