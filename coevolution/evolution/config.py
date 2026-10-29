"""Every setting of an evolution run, written into its run directory at init (``config.json``)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

STRATEGIES = ("archive", "map_elites", "greedy", "random")
SCREENED = ("archive", "greedy", "random")      # strategies that admit children by a zero-shot screen


@dataclass
class EvolutionConfig:
    strategy: str = "archive"          # parent selection and replacement: see select.py, generation.py
    population: int = 32               # active hands (screened strategies); map_elites: elites + challengers float
    children: int = 16                 # proposed per generation (one mutation of one parent each)
    admitted: int = 8                  # children admitted per generation (screened strategies; map_elites: all)
    protect_generations: int = 1       # a newcomer cannot be removed for this many generations after admission
    # DGM parent weight: sigmoid(lambda * (fitness / best - midpoint)) / (1 + children); fitness relative to the
    # archive's best, so the rule calibrates itself (DGM: lambda 10, midpoint 0.5 on scores in [0, 1]).
    sigmoid_lambda: float = 10.0
    sigmoid_midpoint: float = 0.5
    # Children whose zero-shot fitness is below this fraction of their parent's are not admitted (0: none refused).
    screen_floor: float = 0.0
    novelty_k: int = 5                 # novelty: mean distance to the k nearest archive hands (tie-breaker)
    rules: str = "FEETECH_COARSE_RULES"   # hand_sampler.grammar.hand rule set for mutation and random hands
    grid: dict = field(default_factory=lambda: {"n_rigid": 5, "n_moving": 2, "n_slots": 5})
    mutation_tries: int = 8            # operators tried before a parent is skipped this generation
    seed: int = 0

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        if self.strategy in SCREENED:
            if not 0 < self.admitted <= self.children:
                raise ValueError(f"admitted ({self.admitted}) must be in 1..children ({self.children})")
            if self.admitted >= self.population:
                raise ValueError("admitted must be smaller than the population")

    @property
    def screened(self) -> bool:
        """Children admitted by a zero-shot screen (archive, greedy, random); map_elites admits every child as a
        challenger, judged after a generation of training."""
        return self.strategy in SCREENED

    @classmethod
    def load(cls, path) -> "EvolutionConfig":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=1))

    def rule_set(self):
        from hand_sampler.grammar import hand as gh
        return getattr(gh, self.rules)


__all__ = ["EvolutionConfig", "SCREENED", "STRATEGIES"]
