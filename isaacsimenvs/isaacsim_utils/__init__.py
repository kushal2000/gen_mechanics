"""Isaac helpers with no task in them.

Anything here works the same whichever task is being built:

* ``urdf_to_usd`` -- prepare a URDF for Kit's converter, convert it, apply the
  SDF collision markers and self-collision filters it declares, and bake the
  result. The slow path, ~876 ms/hand, which is why direct Sdf authoring exists.
* ``physx`` -- contact/rest offsets, the USD attribute table PhysX properties
  are written through, and scene-step timing.
"""
