# 06 Oct: minimal viable embodiment for in-hand reorientation

What is the smallest hand that can still do the task? Controlled reductions of Wuji v2 -- same palm, link
geometry, uniform dynamics, palm-frame observations and RL config -- each trained from scratch, so viability is
a property of the hardware, not of transfer.

- `viser_variants.py`: inspect the full Wuji v2 and every reduced variant (kinematics only, instant switching):
  joint sliders, pose buttons, collision meshes, side-by-side with the full hand, a summary of what was removed.
  `.venv_isaacsim/bin/python experiments/06oct_minimal_embodiment/viser_variants.py --port 8082`
- Variants are generated next to the full hand in `assets/urdf/unified_dynamics_commercial_hands/wuji2/`
  (missing fingers: `make_missing_fingers.py`).
