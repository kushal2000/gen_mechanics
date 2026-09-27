"""MAP-Elites evolution pilot for grammar hands (plan-rl-grammar-tuning.md's
"Revision, 2026-09-27", pilot E-R2').

``archive.py`` is the CPU-testable MAP-Elites archive (pure python + numpy +
``hand_sampler``, no ``isaacsimenvs``/``isaaclab``/``torch`` import); it is
importable and unit-testable without booting Kit. ``driver.py`` is the
generation loop that shells out to ``coevolution/train.py`` per generation
and drives the archive with it.
"""

from __future__ import annotations
