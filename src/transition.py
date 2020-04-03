from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class Transition:
    """Represents a (state, action, next_state, reward, done) tuple."""

    state: torch.Tensor
    action: List[float]
    next_state: torch.Tensor
    reward: float
    done: bool
