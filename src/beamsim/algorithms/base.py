"""Base class for MBP-selection algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

from beamsim.bplm import BPLMState


class Algorithm(ABC):
    """Picks the next (k, l) measurement-beam-pair given the BPLM state."""

    name: str = "base"

    def reset(self, state: BPLMState, context: dict) -> None:
        """Called at the start of each Monte Carlo trial."""
        pass

    @abstractmethod
    def select_next_mbp(self, state: BPLMState, m: int, context: dict) -> tuple[int, int]:
        ...
