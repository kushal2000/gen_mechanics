"""z-scores of each hand's zero-pose joint tokens under the checkpoint's own token normaliser (pooled over the
training population's real joints): how far out of the training distribution each token field sits."""
import sys, json
import numpy as np, torch
from pathlib import Path
from omegaconf import OmegaConf
sys.path.insert(0, ".")
from isaacsimenvs.common.obs_layout import build_finger_layout
from isaacsimenvs.common.robots import get_finger_layout, get_robot_spec
from coevolution.networks.joint_transformer import _WithArmJoints
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sd = (ck[0] if 0 in ck else ck)["model"]
mean = sd["a2c_network.token_norm.running_mean"].numpy(); var = sd["a2c_network.token_norm.running_var"].numpy()
gm = sd["a2c_network.global_norm.running_mean"].numpy(); gv = sd["a2c_network.global_norm.running_var"].numpy()
full = OmegaConf.to_container(OmegaConf.load(next(Path(sys.argv[2]).rglob(".hydra/config.yaml"))), resolve=True)
net = full["agent"]["params"]["network"]
import os
ref = os.environ.get("TOKEN_Z_REF", "multi:finger:wuji2_left_uniform_handonly")
lay = build_finger_layout(_WithArmJoints(get_robot_spec(ref), 6), net["obs_list"], get_finger_layout(ref), net["extra_field_sizes"])
cols = np.asarray(lay["token_columns"])            # (n_hand, token_dim)
fs = lay["token_field_slices"]
gidx = np.array([i for s, e in lay["global_slices"] for i in range(s, e)])
for npz in sys.argv[3:]:
    P = np.load(npz); o = P["obs0"]
    slot_joint = {int(k): v for k, v in json.loads(str(P["slot_joint"])).items()}
    tok = o[cols]
    z = (tok - mean) / np.sqrt(var + 1e-5)
    real = sorted(slot_joint)
    print(f"== {npz}: |z| over real joint tokens, per field (median / max); >5 is clipped by the normaliser")
    for f, (s, e) in (fs.items() if isinstance(fs, dict) else fs):
        a = np.abs(z[real, s:e])
        print(f"   {f:22s} median {np.median(a):5.2f}  max {a.max():5.2f}  frac>3 {np.mean(a>3):.2f}")
    zg = (o[gidx] - gm) / np.sqrt(gv + 1e-5)
    print(f"   global: max |z| {np.abs(zg).max():.2f} at global col {int(np.abs(zg).argmax())}; "
          f"palm_extents z {np.round(zg[-11:-8],2).tolist() if len(zg) > 11 else ''}")
