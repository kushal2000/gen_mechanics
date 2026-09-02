"""Shared helpers with no single owner in the task.

* ``urdf_to_usd`` -- prepare a URDF for Kit's converter, convert it, apply the
  SDF collision markers and self-collision filters it declares. Mesh assets
  only; generated hands and procedural objects are authored directly.
* ``physx`` -- scene-step timing.
"""
