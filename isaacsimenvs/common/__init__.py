"""Shared infrastructure for every task in isaacsimenvs (in-hand olympics, pose reaching, in-hand reorientation).

Never imports a task package. Importing this package does nothing; import the module you need.

Kit-free (plain Python, importable without booting Isaac Sim):
  robots/          the robot registry: every arm / hand RobotSpec, multi-hand sets (multi_hand.py)
  obs_layout.py    observation field sizes and the joint-token layout the transformer gathers from
  sdf.py           Sdf spec helpers for authoring prims straight into a USD layer
  author_objects.py  authoring task objects (handle + head shapes) and physics materials
Need Kit:
  urdf_to_usd.py   URDF -> USD conversion (Isaac Lab's converter) and the robot joint-drive defaults
  usd_flatten.py   flatten a converted robot USD into one layer
  obs_geometry.py  batched geometry for observations (palm frame, link boxes, keypoints relative to joints)
  sim_cfg.py       the default simulation config (120 Hz physics, 60 Hz policy)
"""
