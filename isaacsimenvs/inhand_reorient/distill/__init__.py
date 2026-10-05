"""GET-Zero-style distillation of per-hand MLP experts into one joint-token
transformer (A. Patel and S. Song, "GET-Zero: Graph Embodiment Transformer
for Zero-shot Embodiment Generalization", 2024).

``dagger.py`` is the Kit-free core (expert loading and routing, the DAgger
action mix and beta schedule, the aggregated dataset, the masked action
loss, first-episode evaluation and checkpoints), CPU-tested in
``tests/test_distill.py``. ``run.py`` is the Kit entry point (AppLauncher
first) that trains the student on a population env and evaluates students,
experts and zero actions.
"""

from __future__ import annotations
