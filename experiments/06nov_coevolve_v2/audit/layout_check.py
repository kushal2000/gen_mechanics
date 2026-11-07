"""Kit-free: the network's token layout and the hand set's slot tables for one grammar hand, built from a
multi:template: ref (one-hand manifest, grid 6+0x5) and from a multi:finger: ref. Any difference is a piping bug."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from isaacsimenvs.common.robots import get_finger_layout, get_robot_spec
from isaacsimenvs.common.robots.multi_hand import hand_set
from isaacsimenvs.common.obs_layout import build_finger_layout
from coevolution.networks.joint_transformer import _WithArmJoints
from omegaconf import OmegaConf

spec = sys.argv[1]
run = sys.argv[2]
full = OmegaConf.to_container(OmegaConf.load(next(Path(run).rglob(".hydra/config.yaml"))), resolve=True)
fields = full["agent"]["params"]["network"]["obs_list"]
extra = full["agent"]["params"]["network"]["extra_field_sizes"]
d = Path(tempfile.mkdtemp()); (d / "manifest.json").write_text(json.dumps(
    {"generation": 0, "grid": {"n_rigid": 6, "n_moving": 0, "n_slots": 5}, "hands": [{"id": "h", "spec": spec}]}))
refs = {"template": f"multi:template:{d/'manifest.json'}", "finger": f"multi:finger:{spec}"}
L, H = {}, {}
for k, ref in refs.items():
    s = _WithArmJoints(get_robot_spec(ref), 6)
    L[k] = build_finger_layout(s, fields, get_finger_layout(ref), extra)
    H[k] = hand_set(ref)
for key in L["template"]:
    a, b = L["template"][key], L["finger"].get(key)
    same = (np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object)) if not isinstance(a, dict) else a == b)
    print(f"layout[{key}]: {'same' if same else 'DIFFERENT'}")
for attr in ("n_joints", "n_tips", "joint_slots", "finger_slots", "joint_valid", "fingertip_valid", "fingertip_offsets",
             "palm_center_offset", "palm_keypoints", "hand_scale", "joint_link_boxes", "joint_geometry_valid", "joint_moving"):
    a, b = getattr(H["template"], attr), getattr(H["finger"], attr)
    print(f"hand_set.{attr}: {'same' if np.array_equal(np.asarray(a), np.asarray(b)) else 'DIFFERENT'}")
print("grid", H["template"].grid, H["finger"].grid)
for k in ("template", "finger"):
    t = H[k].template
    print(k, "template spec palm_center_offset", t.palm_center_offset, "hand_scale", t.hand_scale, "F", t.num_fingertips)
