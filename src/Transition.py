from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Transition:
    """Represents a (state, action, next_state, reward, done) tuple."""

    state: List[float]
    action: List[float]
    next_state: List[float]
    reward: float
    done: bool
