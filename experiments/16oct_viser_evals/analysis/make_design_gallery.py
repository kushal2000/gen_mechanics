"""The designs compared on 15-16 Oct (repose v2, spin / flip fine-tunes), in one folder: each design's pictures under
a plain name, and one overview sheet with every design's face view, its role, finger layout and peak repose score.

Primitive hands: rendered here by experiments/14oct_design_search/analysis/render_primitive.py -- face: the grasp
face, fingers straight, every hand on the same fixed +-17 cm frame (so equal palms look equal); curl45: a
three-quarter view, every joint at 45 deg (each fitted to its hand). Wuji: experiments/12oct_inhand_olympics/analysis/
render_hands.py (palm, joints at 0), at its own scale.

    .venv_isaacsim/bin/python experiments/16oct_viser_evals/analysis/make_design_gallery.py
    # -> experiments/16oct_viser_evals/plots/designs/
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[3]
WUJI = REPO / "experiments/12oct_inhand_olympics/plots/hands"
OUT = pathlib.Path(__file__).resolve().parent.parent / "plots/designs"
sys.path.insert(0, str(REPO / "experiments/08oct_repro_wuji_full/analysis"))

# (file stem, role, finger bearings, gaps tag or None for Wuji, peak repose v2 goals/episode)
DESIGNS = [
    ("01_claude_choice", "Claude-chosen", "0 / 90 / 180 / 270", "090_090_090_090", 31.4),
    ("02_claude_runner_up", "Claude runner-up", "0 / 95 / 180 / 275", "085_095_085_095", 27.2),
    ("03_random_a", "Random A", "0 / 30 / 85 / 140", "030_055_055_220", 29.0),
    ("04_random_b", "Random B", "0 / 35 / 100 / 295", "035_065_195_065", 26.8),
    ("05_random_c", "Random C", "0 / 50 / 115 / 170", "050_065_055_190", 32.7),
    ("06_adversarial", "Adversarial", "0 / 25 / 50 / 75", "025_025_025_285", 19.3),
    ("07_wuji", "Wuji (human-like)", "5 fingers", None, 26.3),
]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import paper_style as ps

    OUT.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp())
    faces = []
    for stem, role, bearings, gaps, peak in DESIGNS:
        if gaps:
            hand = f"prim4x4_r10l25_g{gaps}"
            subprocess.run([sys.executable, str(REPO / "experiments/14oct_design_search/analysis/render_primitive.py"),
                            "--hand", hand, "--extent", "0.17", "--out", str(tmp)], check=True, capture_output=True)
            src = {"face": tmp / f"{hand}_face.png", "curl45": tmp / f"{hand}_curl45.png"}
        else:
            src = {"palm": WUJI / "wuji_full_flat.png"}
        for view, p in src.items():
            shutil.copy(p, OUT / f"{stem}_{view}.png")
        faces.append((role, bearings, peak, src.get("face", src.get("palm"))))
    ps.configure()
    fig, axes = plt.subplots(1, len(faces), figsize=(ps.TEXT_WIDTH * 1.6, 2.6))
    for ax, (role, bearings, peak, p) in zip(axes, faces):
        ax.imshow(mpimg.imread(p))
        ax.set_axis_off()
        ax.set_title(f"{role}\n{bearings}\nrepose peak {peak:.1f}" + ("\n(own scale)" if "Wuji" in role else ""),
                     fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "00_overview.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
