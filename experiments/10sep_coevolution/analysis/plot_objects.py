"""Render a curated object pool as a grid of top/side silhouettes, to scale.

    .venv_isaacsim/bin/python experiments/10sep_coevolution/analysis/plot_objects.py [pool]
"""
import importlib, sys, types, pathlib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import Rectangle, FancyBboxPatch
name = sys.argv[1] if len(sys.argv) > 1 else "diverse24"
pkg = "isaacsimenvs.pose_reaching_6d.scene_utils.objects"
if pkg not in sys.modules:   # skip the Kit-bound scene_utils __init__
    m = types.ModuleType(pkg); m.__path__ = ["isaacsimenvs/pose_reaching_6d/scene_utils/objects"]; sys.modules[pkg] = m
c = importlib.import_module(pkg + ".curated_pools")
pool = c.CURATED_POOLS[name]
cols = 6; rows = (len(pool) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.4 * rows), dpi=150)
for ax, (k, (typ, handle, head, hd, headd)) in zip(axes.flat, enumerate(pool)):
    hl = handle[0]; hw = handle[1] if len(handle) == 2 else handle[1]; ht = handle[1] if len(handle) == 2 else handle[2]
    box = len(handle) == 3
    # top view (x along, y across) above; side view (x along, z up) below, both in cm
    def draw(ax, y0, across, head_across, label):
        ax.add_patch(FancyBboxPatch((0, y0 - across / 2), hl, across, boxstyle=("square,pad=0" if box else f"round,pad=0,rounding_size={across/2}"),
                                    fc="#b07a3a", ec="#5a3a15", lw=0.8))
        if head is not None:
            if len(head) == 3:
                x0 = hl; ax.add_patch(Rectangle((x0, y0 - head_across / 2), head[0], head_across, fc="#8c8c8c", ec="#444", lw=0.8))
            else:
                x0 = hl; ax.add_patch(Rectangle((x0, y0 - head[0] / 2), head[1], head[0], fc="#8c8c8c", ec="#444", lw=0.8))
        ax.text(-0.005, y0, label, ha="right", va="center", fontsize=6, color="#555")
    head_y = None if head is None else (head[1] if len(head) == 3 else head[0])
    head_z = None if head is None else (head[2] if len(head) == 3 else head[1])
    draw(ax, 0.10, hw, head_y or 0, "top")
    draw(ax, 0.00, ht, head_z or 0, "side")
    ax.set_xlim(-0.03, 0.40); ax.set_ylim(-0.07, 0.17); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{k}  {typ} ({'box' if box else 'cyl'})   {c.object_mass(handle, head, hd, headd)*1000:.0f} g", fontsize=8)
for ax in axes.flat[len(pool):]: ax.axis("off")
ax = axes.flat[0]; ax.plot([0.30, 0.40], [-0.05, -0.05], color="k", lw=1.2); ax.text(0.35, -0.062, "10 cm", ha="center", fontsize=6)
fig.suptitle(f"Curated object pool '{name}': {len(pool)} objects, drawn to a common scale (handle brown, head grey)", fontsize=10)
fig.tight_layout(); out = pathlib.Path(f"debug_outputs/10sep_coevo_analysis/objects_{name}.png"); fig.savefig(out, facecolor="white"); print("wrote", out)
