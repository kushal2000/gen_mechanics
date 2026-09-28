# Unified commercial hands

Off-the-shelf hands expressed in this project's terms, so they can be trained on the
same tasks, with the same joint-transformer encoding, as SHARPA and as hands from the
design grammar.

One folder per hand. Each carries the vendor meshes, a **generated** URDF, and the
script that generates it — never a hand-edited URDF, because the repairs a vendor
description needs are the provenance of the asset and a hand-edit loses them.

    <hand>/
        unify_from_vendor.py   run this; its docstring is the provenance
        <hand>.urdf            generated, self-checked
        meshes/                copied from the vendor

## What "unified" means

A vendor URDF is not wrong, it is written for someone else's simulator and someone
else's conventions. Unifying one means:

- **Repairing what blocks simulation.** Vendor hands ship malformed trees, triangle
  colliders on dynamic bodies, placeholder inertias and limits that exclude the zero
  pose. `allegro/unify_from_vendor.py` documents four such repairs and two it
  deliberately leaves alone.
- **Rooting it at the palm**, so `RobotSpec.palm_body_name` is a real body rather than
  something that has to survive `merge_fixed_joints`.
- **Declaring its frame**, not changing it. Our palm convention is x = thickness and
  grasp normal, y = width, z = wrist to fingertip. A vendor's palm link has no reason
  to agree, so the spec carries the 3x3 that maps one to the other and the meshes are
  left exactly as shipped. Where the mapping comes out as a mirror — a right hand
  against a left-handed convention — flipping the width axis recovers a proper
  rotation, which is preferable to re-authoring the geometry that was the point of
  importing the hand.

## Hands

| hand | joints | fingertips | source |
|---|---|---|---|
| `allegro` | 16 | `{index,middle,ring,thumb}_link_3` | Wonik Allegro right hand, as shipped with Isaac Gym (`simtoolreal/isaacgym/assets/urdf/kuka_allegro_description`) |
