import random
from typing import Any, List, Tuple


class ReservoirBuffer:
    """
    Reservoir sampling buffer used in Deep CFR (LCFR-enabled).

    Now supports storing weighted samples from training iteration t:

        item = (x, target, mask, weight)

    Reservoir sampling still works the same: every incoming item has
    probability min(1, N / seen_so_far) of landing in the buffer.
    """

    def __init__(self, capacity: int, rng: random.Random = None):
        self.capacity = capacity
        self.rng = rng if rng is not None else random.Random()
        self.memory: List[Tuple[Any, ...]] = []
        self.count_seen = 0  # total items processed

        try:
            import logging
            logging.getLogger(__name__).info(f"ReservoirBuffer(capacity={capacity}) created")
        except Exception:
            pass

    # -----------------------------------------------------------
    # ADD ITEM (supports 4-tuple: x, y, mask, weight)
    # -----------------------------------------------------------
    def add(self, item: Tuple[Any, ...]):
        """
        Add new item via reservoir sampling.

        item is usually a tuple:
            (x_tensor, y_tensor, mask_tensor, iteration_weight)
        """
        self.count_seen += 1

        # if buffer not full → append directly
        if len(self.memory) < self.capacity:
            self.memory.append(item)
            return

        # reservoir replacement with prob = capacity / count_seen
        replace_prob = self.capacity / float(self.count_seen)
        if self.rng.random() < replace_prob:
            idx = self.rng.randint(0, self.capacity - 1)
            self.memory[idx] = item

    # -----------------------------------------------------------
    # SAMPLE
    # -----------------------------------------------------------
    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """
        Uniform random sample from buffer.
        """
        if not self.memory:
            return []

        batch_size = min(batch_size, len(self.memory))
        return self.rng.sample(self.memory, batch_size)

    # -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.memory)

    # -----------------------------------------------------------
    def clear(self):
        self.memory.clear()
        self.count_seen = 0
