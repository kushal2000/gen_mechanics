"""Shared helpers, with no single owner in the task.

Anything here is used by more than one part of the task, or would work the same
in another one:

* ``urdf_to_usd`` -- prepare a URDF for Kit's converter, convert it, apply the
  SDF collision markers and self-collision filters it declares, and bake the
  result. The slow asset path, ~876 ms/hand, which is why direct Sdf authoring
  exists alongside it.
* ``physx`` -- the USD attribute table PhysX properties are written through,
  and scene-step timing.
"""
