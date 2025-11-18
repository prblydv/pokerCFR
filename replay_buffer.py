# replay_buffer.py
import random


class ReservoirBuffer:
    def __init__(self, capacity: int, rng: random.Random):
        self.capacity = capacity
        self.data = []
        self.n_seen = 0
        self.rng = rng

    def add(self, sample):
        self.n_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            idx = self.rng.randint(0, self.n_seen - 1)
            if idx < self.capacity:
                self.data[idx] = sample

    def sample(self, batch_size: int):
        return [self.rng.choice(self.data) for _ in range(batch_size)]

    def __len__(self):
        return len(self.data)
