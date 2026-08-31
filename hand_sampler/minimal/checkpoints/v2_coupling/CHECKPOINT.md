# v2_coupling — kinematics + joint coupling, before anything further

Frozen state of the sampler with the three coupling roles in place and the design
space pruned. Everything here was verified numerically; `space_size.txt`,
`audit_sampling.txt` and `parameters.txt` are the outputs at the time of freezing.

## Design space

```
kinematics (lengths x dofs)           86     6.4 bits
x couplings (avg per kin)           15.7     4.0 bits
x splay values                         3     1.6 bits
= distinct finger designs          4,047    12.0 bits

TOTAL HANDS                   66,331,746,450
```

Fully discrete — every number is exact, not "times a continuum".

## Fixed geometry

palm 25 x 60 x 60 mm, wrist on -z, fingers at the centres of +z / +y / -y.
Finger length 100 mm total, 30 mm minimum link, 10 mm quanta, 1-3 links.
Capsule radius 10 mm. FE limit [-10, 120], AA [-20, 20]. Splay in {-30, 0, +30}.

## Rules

* MCP FE always exists; it may be rigidly coupled but **never fully passive**,
  so every finger has at least one actuator.
* At most **3 actuators** per finger — extra joints must be rigidly coupled or
  passive. `n_actuators = n_joints - n_rigid_pairs - n_passive`.
* At most **3 passive** joints per finger.
* A joint **location** may not have both DOFs passive.
* Rigid pairs only between **adjacent locations**, any DOF combination, 1:1 in
  **normalised range** (so a coupled slider at 0 does NOT mean both joints at 0 —
  this is deliberate, see README).
* Passive joints rest at **mid-range**: FE +55 deg, AA 0.

## Known and deliberate

* The sampler leans simple on every axis: mean 2.84 joints per finger against
  4.44 for a proportional draw, 1-joint fingers 260x over-represented. Left as is;
  the narrow fix is written up in the README TODO.
* Fingers are sampled independently, so 3-finger hands are >99.9% of the space.
  Inter-finger constraints were considered and explicitly ruled out.
* The rest pose is flat in the palm plane only for fingers with no passive FE and
  no coupling that drives an FE partner off zero.

## Restore

    cp checkpoints/v2_coupling/*.py .
