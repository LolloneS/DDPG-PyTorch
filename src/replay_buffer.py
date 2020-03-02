import random
from typing import List

from src.transition import Transition

random.seed(42)


class ReplayBuffer:
    def __init__(self, size: int, random: bool = False):
        self.size = size
        self.memory: List[Transition] = []
        self.random = random
        self.occupied = 0
        if not random:
            self.current = 0

    def store(self, transition: Transition):
        """
        Store a Transition in the buffer.
        If self.random, then the overwritten transition is casual, otherwise the
        buffer is circular.
        """
        if len(self.memory) < self.size:
            self.memory.append(transition)
            self.occupied += 1
        else:
            if self.random:
                self.memory[random.randrange(self.size)] = transition
            else:
                self.memory[self.current] = transition
                self.current = (self.current + 1) % self.size

    def get(self, amount: int = 1):
        """Get either 1 (default) or `amount` elements randomly from the buffer."""
        if amount == 1:
            return random.choice(self.memory)
        return random.sample(self.memory, amount)
