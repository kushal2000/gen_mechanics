# MIDAS hand

| | |
|---|---|
| source | https://github.com/midas-hand-org/midas_hand_mujoco |
| file | `assets/midas_description/midas_hand_urdf.urdf` |
| licence | MIT, copied here as `LICENSE` |
| project | https://midas-hand.com/ |

Only the URDF is vendored. The fitter reads joint origins and axes and nothing
else, so the meshes are left in the upstream repo.

**Handedness is not declared** by the vendor -- the model is just "the MIDAS
hand", with no left/right in the repo, the README or the mesh names. The other
hands here are all left. If the fitted MIDAS turns out mirrored against them,
that is why.

**The distal joint is a four-bar, and is treated here as fully actuated.** The
URDF carries no `<mimic>`, so it already exposes all four joints of a finger as
independent; the coupling lives only in the MuJoCo model, as an `equality/
connect` between `*_dip_pin` and `*_linkage_pin`. The three `*_dip_linkage_joint`
revolute joints that close those loops are not part of any finger's chain and
the fitter ignores them.

**The URDF has no fingertip frames**, so the chains end at the last joint. The
tip offsets the fitter uses come from the MuJoCo model's own contact geometry --
the `fingertip_contact` class for the fingers and `thumb_dip_contact` for the
thumb -- which is where the vendor says the finger touches things. They are
recorded in `commercial.TIP_FALLBACK` rather than patched into this file, so
what is vendored stays byte-identical to upstream.
