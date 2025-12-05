import random
from typing import Any, List


class ReservoirBuffer:
    """
    Reservoir sampling buffer used in Deep CFR.

    - Fixed maximum capacity N
    - Every incoming element has probability min(1, N / seen_so_far)
      of being included
    - Items sampled uniformly from stored buffer

    This implementation is CPU-safe and works with any Python objects
    (tensors, tuples, states, etc.)
    """

    def __init__(self, capacity: int, rng: random.Random = None):
        self.capacity = capacity
        self.rng = rng if rng is not None else random.Random()
        self.memory: List[Any] = []
        self.count_seen = 0  # total items processed
        # Log creation for debug / instrumentation
        try:
            import logging
            logging.getLogger(__name__).info(f"ReservoirBuffer(capacity={capacity}) created")
        except Exception:
            pass

    def add(self, item: Any):
        """
        Add new item via reservoir sampling.
        """
        self.count_seen += 1

        # If buffer not full → append directly
        if len(self.memory) < self.capacity:
            self.memory.append(item)
            return

        # If full, replace random item with probability capacity/count_seen
        replace_prob = self.capacity / float(self.count_seen)
        if self.rng.random() < replace_prob:
            idx = self.rng.randint(0, self.capacity - 1)
            self.memory[idx] = item

    def sample(self, batch_size: int) -> List[Any]:
        """
        Uniform random batch from the buffer.
        """
        if len(self.memory) == 0:
            return []
        batch_size = min(batch_size, len(self.memory))
        return self.rng.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def clear(self):
        """
        Optional helper for clearing memory.
        """
        self.memory.clear()
        self.count_seen = 0
