# ---------------------------------------------------------------------------
# File overview:
#   replay_buffer.py implements a simple reservoir buffer for Deep CFR data.
#   Import the buffer class from trainers; no CLI entry point.
# ---------------------------------------------------------------------------
import random
import collections


class ReservoirBuffer:
    # Function metadata:
    #   Inputs: capacity (int), rng (random.Random)
    #   Sample:
    #       buf = ReservoirBuffer(capacity=1000, rng=random.Random(0))
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.data = []
        self.n_seen = 0
        self.rng = rng

    # Function metadata:
    #   Inputs: sample (Any)
    #   Sample:
    #       buf.add((state_tensor, target_tensor))  # dtype=NoneType
    def add(self, sample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            idx = self.rng.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.data[idx] = sample

    # Function metadata:
    #   Inputs: batch_size (int)
    #   Sample:
    #       batch = buf.sample(batch_size=32)  # dtype=List[Any]
    def sample(self, batch_size: int):
        return [self.rng.choice(self.data) for _ in range(batch_size)]

    # Function metadata:
    #   Inputs: None (magic method)
    #   Sample:
    #       current = len(buf)  # dtype=int
    def __len__(self):
        return len(self.data)

