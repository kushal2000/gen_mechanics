"""minimal: a small, fully-enumerable hand design space.

A deliberate rewrite of :mod:`hand_sampler.params`, stripped to a space where
every parameter is discrete and every rule is explicit: finger length fixed at
100 mm and partitioned on a 10 mm quantum, at most 3 links, 2 or 3 fingers,
splay in 30-degree steps. It adds two concepts the older grammar has no notion
of -- passive joints and couplings between them.

It is NOT a drop-in replacement for ``hand_sampler.params``, and nothing in the
simulator path reads it yet:

* it emits kinematics, joint roles and geometry, but no URDF and no
  ``RobotSpec``, so it cannot reach Isaac;
* ``isaacsimenvs.pose_reaching_6d.morphology`` derives DESCRIPTOR_DIM (143)
  from ``params.N_FINGER_SLOTS``, and that width is baked into the trained
  policy's 329-wide observation. A different grammar means a different
  descriptor and a retrained policy.

Pure numpy/trimesh/viser/matplotlib, so it imports without a simulator like the
rest of :mod:`hand_sampler`.
"""

__all__: list[str] = []
