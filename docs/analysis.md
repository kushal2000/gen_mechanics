# What makes a generated hand good?

Analysis of `results/population_eval/mec_population24k_seed0_2026-08-17_15-13-28_tol3cm`:
all 24,576 seed-3 designs, 10 episodes each, 3 cm success tolerance, DR off,
scored by the multi-embodiment policy at epoch 16,600 (14% of its budget).

Every design held **one fixed object for its entire life** — env `i` gets design
`i` and object `i % 1200` — so per-design scores carry that object's difficulty.
That confound is the largest single effect here (§5) and bounds everything below.

Headline: **mean 1.893 / 10 goals**, median 1.700, sd across designs 1.267,
best 7.700, and 1,107 designs (4.5%) never scored at all.

---

## 1. The population is not uniform

| fingers | n | share | | joints | n | share |
|---|---|---|---|---|---|---|
| 2 | 20,964 | 85.3% | | 4–6 | 273 | 1.1% |
| 3 | 3,284 | 13.4% | | 7–9 | 4,324 | 17.6% |
| 4 | 306 | 1.2% | | 10–12 | 16,627 | 67.7% |
| 5 | 22 | 0.1% | | 13–30 | 3,352 | 13.6% |

Finger count and joint count are nearly the same variable at ~5.1 joints per
finger, and their ranges barely overlap: every 2-finger design has ≤12 joints,
every 5-finger design has ≥19.

## 2. Fingers: a cliff, not a gradient

| fingers | n | mean goals | sem | never scored |
|---|---|---|---|---|
| 2 | 20,964 | **1.938** | ±0.009 | 4.3% |
| 3 | 3,284 | **1.753** | ±0.020 | 3.4% |
| 4 | 306 | **0.487** | ±0.030 | 23.2% |
| 5 | 22 | **0.000** | ±0.000 | 100% |

2f→3f is a real but small gap (~10%). 4f and 5f collapse. Note 5f is n=22 and
0.09% of training, so "the policy cannot drive it" and "the policy never saw it"
are not separable from this data.

## 3. What matters, within fixed topology

Measured on the 7,771 two-finger 11-joint designs unless noted — identical
topology, so geometry is the only thing varying.

**Mount separation** (distance between the two finger mounts on the palm) —
inverted U, peak 4–5 cm, ~0.7 goals of swing. The object's nominal size is 4 cm.

```
 2-3 cm  1.627     6-7 cm  2.079     10-11 cm  1.646
 3-4 cm  1.881     7-8 cm  1.986     11-12 cm  1.590
 4-5 cm  2.145 <-  8-9 cm  1.893     12-13 cm  1.438
 5-6 cm  2.127     9-10    1.822
```

**Fingertip reach** (mc + proximal + middle + distal, fully extended) — inverted
U, peak 14.5–16 cm, ~0.7 goals of swing. Too short is bad; too long is worse.

```
10.0-11.5  1.353    16.0-17.5  2.043
11.5-13.0  1.590    17.5-19.0  1.980
13.0-14.5  1.908    19.0-20.5  1.856
14.5-16.0  2.056 <- 20.5-22.0  1.629
```

**Joint count** (all 2f designs) — monotone 5→12, ~1.1 goals, the largest swing
of the three. But see §4: joint count and joint *identity* are the same variable.

```
 5  0.973    8  1.654    11  2.042
 6  1.134    9  1.710    12  2.077
 7  1.535   10  1.969
```

Separation and reach are independent (r = −0.016), so these are two effects, not
one restated.

## 4. What does not matter: link size

Individually and in aggregate, **link dimensions do not predict performance**:

| quantity | spearman | range spanned |
|---|---|---|
| proximal volume | −0.006 | 7× |
| middle volume | −0.007 | 7× |
| distal volume | −0.007 | 7× |
| CMC (metacarpal) volume | −0.018 | 20× |
| whole-finger volume | −0.016 | 10× |
| `radius_scale` | −0.005 | 2× |

A 10× span in finger volume moves the decile means from 1.979 to 2.089 — inside
noise at ±0.046. The three link volumes are also ~0.83 correlated with each
other (one `radius_scale` knob scales all tiers), so they are close to one
measurement repeated.

**Layout matters; bulk does not.** For a pinch on a small object that is a
plausible mechanism: opposition geometry decides whether a grasp exists at all,
and thickness only matters once contact is made.

## 5. Confounds, in order of size

1. **The object.** 17.9% of variance across designs, larger than every geometry
   effect combined. Each design saw one object for the whole run, so a design's
   score is partly its object's. The only strictly like-for-like comparison is
   within a shared-object group (`shared_object_groups` in the assignment JSON:
   1,200 groups of ~20 designs holding an identical object).
2. **`ACTIVATION_ORDER`.** Joints are not sampled independently. Each finger
   draws `n_joints` and takes the first `n` rungs of a fixed ladder:
   `MCP_FE, PIP, MCP_AA, DIP, CMC_FE, CMC_AA`. So `MCP_FE` and `PIP` are never
   ghosted (0 of 41,928 fingers) and `CMC_AA` is ghosted on 57.1%. All 7,771
   two-finger 11-joint designs drop `CMC_AA` and nothing else — there is no
   "same joint count, different joint missing" contrast to measure. "12 joints
   is best" means "has `CMC_AA` on both fingers", not "has one more joint".
3. **Training maturity.** Epoch 16,600 of 120,000.

## 6. Is this learnability rather than mechanics?

Designs in dense regions of morphology space do score better — spearman +0.163
between local neighbour density and goals across 2-finger designs. The natural
worry is that the optima above are just "where the policy has the most examples".

They are not, and the argument needs no normalization: **both optima sit below
the mode.** Separation peaks at 4.75 cm against a modal 6.25 cm; reach peaks at
15.5 cm against a modal 18.5 cm. Rarer-than-typical designs beat typical ones,
which is the opposite of what pure learnability predicts.

Residualizing performance on local density is **not** a valid control here:
density in this space is a function of the same features being tested, so
subtracting it removes variance the feature itself created and biases every
estimate toward zero. Raw numbers are reported throughout.

## 7. Open

* Re-run the sweep at a later checkpoint. Everything here describes one policy
  at 14% of training, and the 4f/5f collapse in particular may not survive.
* Control for the object by comparing within `shared_object_groups`, which is
  the one confound larger than the effects being measured.
* The `ACTIVATION_ORDER` ladder is a prior, not a finding. Testing it needs a
  population sampled with independent per-slot ghosting; `params.enabled_for` is
  the single point to change.
* `05_population_24k_no_morphology.sub` asks whether the policy uses the
  morphology descriptor at all. If it does not, none of §3 is being read by the
  network and the effects are purely mechanical.

Reproduce: `genmech/eval/embodiments/run_population_eval.py`, figure from
`plot_best_worst.py`.
