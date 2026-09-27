"""Grammar-to-simulator adapter (Phase 2): a padded 5x6 + 2 palm-carrier
envelope that admits grammar-derived hands (and grammar-projected commercial
hands) into the same fixed articulation topology, so a population of
different designs can share one env (see
``project-notes/grammar/phase2-adapter-design.md``).

``grammar_envelope.py`` and ``population_file.py`` are numpy + ``hand_sampler``
only (no isaaclab/pxr, CPU-testable under plain pytest). ``author_grammar.py``
is the only module here that imports ``pxr``/``isaaclab`` (lazily, at call
time -- see its own module docstring for why), mirroring
``isaacsimenvs.pose_reaching_6d.scene_utils.assembly``'s per-env authoring.
"""

from __future__ import annotations
