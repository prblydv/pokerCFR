"""External-sampling Deep CFR for the fixed-action heads-up Hold'em engine.

The neural policy has exactly ``NUM_ACTIONS`` outputs.  It never predicts a raw
chip amount.  Each output selects one of the engine's canonical, state-dependent
action slots; arbitrary off-tree ``raise_to`` actions remain an engine/search
concern.  Checkpoints lock the engine, action, and encoder schemas so a policy
cannot silently run against different betting mathematics.

This is practical approximate Deep CFR:

* at a traverser's node every legal action is evaluated;
* at the opponent's node one action is sampled from regret matching;
* opponent information states encountered by that external sampling process
  train the reach-weighted average strategy;
* terminal utility is net chips divided by the big blind.

Because heads-up poker is two-player zero-sum, no multiplayer reach correction
or third-player approximation is needed.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


def _configure_parallel_runtime() -> None:
    """Avoid descriptor exhaustion when spawned workers exchange CPU tensors."""

    try:
        import resource

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(int(hard_limit), 65_536)
        if int(soft_limit) < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard_limit))
    except (ImportError, OSError, ValueError):
        pass

    try:
        strategies = torch.multiprocessing.get_all_sharing_strategies()
        if "file_system" in strategies:
            torch.multiprocessing.set_sharing_strategy("file_system")
    except (AttributeError, RuntimeError):
        pass


_configure_parallel_runtime()

from heads_up_engine import (
    ACTION_NAMES,
    ACTION_SCHEMA_VERSION,
    ENGINE_SCHEMA_VERSION,
    NUM_ACTIONS,
    NUM_PLAYERS,
)
from heads_up_models import (
    DEFAULT_MAX_HISTORY,
    POLICY_RANGE_AUX_ARCHITECTURE,
    build_action_descriptors,
    build_advantage_network,
    build_policy_network,
    encode_information_state,
    encoder_metadata,
    information_state_size,
    masked_softmax,
)
from heads_up_ranges import (
    NUM_OPPONENT_COMBOS,
    masked_range_probabilities,
    opponent_combo_index,
    valid_combo_mask_from_encoded,
)


CHECKPOINT_KIND = "heads_up_deep_cfr"
CHECKPOINT_VERSION = 4
NETWORK_ARCHITECTURE = "hu_deep_cfr_compact_v4"
POLICY_NETWORK_ARCHITECTURE = POLICY_RANGE_AUX_ARCHITECTURE
TRAINING_DEFAULT_MAX_HISTORY = DEFAULT_MAX_HISTORY
ROOT_STACK_DISTRIBUTION_FIXED = "fixed_environment_stack_v1"
ROOT_STACK_DISTRIBUTION_MIXED = "mixed_equal50_unequal50_v1"
DEFAULT_ROOT_STACK_DEPTHS_BB = (10, 15, 20, 25, 30, 40, 50, 60, 75, 100)


def _legal_mask(legal: Iterable[int]) -> torch.Tensor:
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for action in legal:
        action = int(action)
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"legal action {action} is outside the fixed action space")
        mask[action] = 1.0
    if float(mask.sum()) <= 0.0:
        raise ValueError("a decision state must contain at least one legal action")
    return mask


class _PackedReservoirView(Sequence[tuple[torch.Tensor, ...]]):
    """Lazy row view over contiguous field-major reservoir tensors."""

    def __init__(
        self,
        fields: Sequence[torch.Tensor],
        length: int | None = None,
        start: int = 0,
    ):
        self.fields = [field.detach().cpu() for field in fields]
        storage = int(self.fields[0].shape[0]) if self.fields else 0
        self.length = storage if length is None else int(length)
        self.start = int(start)
        if self.length < 0 or self.length > storage:
            raise ValueError("packed reservoir length is invalid")
        if self.start < 0 or (storage and self.start >= storage):
            raise ValueError("packed reservoir start is invalid")
        if not storage and self.start:
            raise ValueError("empty packed reservoir must start at zero")
        if any(int(field.shape[0]) != storage for field in self.fields):
            raise ValueError("checkpoint reservoir fields have inconsistent lengths")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(self.length))]
        row = int(index)
        if row < 0:
            row += self.length
        if not 0 <= row < self.length:
            raise IndexError(row)
        physical = (self.start + row) % self.storage_capacity
        return tuple(field[physical] for field in self.fields)

    @property
    def storage_capacity(self) -> int:
        return int(self.fields[0].shape[0]) if self.fields else 0

    def reserve(self, capacity: int) -> None:
        target = int(capacity)
        if target <= self.storage_capacity:
            return
        expanded: list[torch.Tensor] = []
        for field in self.fields:
            destination = torch.empty(
                (target, *field.shape[1:]),
                dtype=field.dtype,
                device="cpu",
            )
            if self.length:
                first = min(self.length, self.storage_capacity - self.start)
                destination[:first].copy_(
                    field[self.start : self.start + first]
                )
                if first < self.length:
                    destination[first : self.length].copy_(
                        field[: self.length - first]
                    )
            expanded.append(destination)
        self.fields = expanded
        self.start = 0

    def physical_indices(
        self,
        logical_indices: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        indices = (
            logical_indices.detach().cpu().to(torch.long)
            if torch.is_tensor(logical_indices)
            else torch.tensor(logical_indices, dtype=torch.long)
        )
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= self.length)):
            raise IndexError("packed reservoir logical index is out of range")
        return (indices + self.start) % self.storage_capacity

    def replace(self, index: int, item: Sequence[torch.Tensor]) -> None:
        if len(item) != len(self.fields):
            raise ValueError("reservoir item has an inconsistent width")
        for field, value in zip(self.fields, item):
            source = (
                value.detach().cpu()
                if torch.is_tensor(value)
                else torch.as_tensor(value).detach().cpu()
            )
            physical = (self.start + int(index)) % self.storage_capacity
            field[physical].copy_(source)

    def append(self, item: Sequence[torch.Tensor]) -> None:
        if self.length >= self.storage_capacity:
            raise RuntimeError("packed reservoir storage is full")
        self.replace(self.length, item)
        self.length += 1

    def append_fields(
        self,
        fields: Sequence[torch.Tensor],
        source_start: int,
        count: int,
    ) -> None:
        source_start = int(source_start)
        count = int(count)
        if count <= 0:
            return
        if len(fields) != len(self.fields):
            raise ValueError("reservoir fields have an inconsistent width")
        if self.length + count > self.storage_capacity:
            raise RuntimeError("packed reservoir storage cannot fit batch")
        destination_start = (
            self.start + self.length
        ) % self.storage_capacity
        first = min(count, self.storage_capacity - destination_start)
        for destination, source in zip(self.fields, fields):
            if int(source.shape[0]) < source_start + count:
                raise ValueError("source reservoir batch is too short")
            destination[
                destination_start : destination_start + first
            ].copy_(source[source_start : source_start + first])
            if first < count:
                destination[: count - first].copy_(
                    source[
                        source_start + first : source_start + count
                    ]
                )
        self.length += count

    def drop_oldest(self, count: int) -> None:
        removed = int(count)
        if removed <= 0 or removed > self.length:
            raise ValueError("invalid packed reservoir eviction count")
        self.start = (self.start + removed) % self.storage_capacity
        self.length -= removed


class ReservoirBuffer:
    """Chunked-recent replay memory with compact field-major storage."""

    _COMPACT_AT = 4_096

    def __init__(
        self,
        capacity: int,
        rng: random.Random,
        turnover_fraction: float = 0.18,
    ):
        if isinstance(capacity, bool) or int(capacity) <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.rng = rng
        self.turnover_fraction = float(turnover_fraction)
        if (
            not math.isfinite(self.turnover_fraction)
            or not 0.0 < self.turnover_fraction < 1.0
        ):
            raise ValueError("reservoir turnover_fraction must be in (0, 1)")
        self.turnover_count = max(
            1,
            math.ceil(self.capacity * self.turnover_fraction),
        )
        self.memory: Sequence[tuple[torch.Tensor, ...]] = []
        self.seen = 0
        self.turnover_events = 0
        self.evicted_samples = 0

    def __len__(self) -> int:
        return len(self.memory)

    @staticmethod
    def _cpu_row(item: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(
            value.detach().to(device="cpu").clone()
            if torch.is_tensor(value)
            else torch.as_tensor(value).detach().cpu().clone()
            for value in item
        )

    def _compact(self) -> None:
        if isinstance(self.memory, _PackedReservoirView) or not self.memory:
            return
        width = len(self.memory[0])
        fields = [
            torch.stack([row[field] for row in self.memory])
            for field in range(width)
        ]
        self.memory = _PackedReservoirView(fields)

    def _ensure_append_capacity(self, required: int | None = None) -> None:
        if not isinstance(self.memory, _PackedReservoirView):
            return
        required = len(self.memory) + 1 if required is None else int(required)
        if required <= self.memory.storage_capacity:
            return
        current = self.memory.storage_capacity
        target = min(
            self.capacity,
            max(required, current + 1, current * 2),
        )
        self.memory.reserve(target)

    def _evict_oldest_chunk_if_full(self) -> None:
        if len(self.memory) < self.capacity:
            return
        removed = min(self.turnover_count, len(self.memory))
        if isinstance(self.memory, _PackedReservoirView):
            self.memory.drop_oldest(removed)
        else:
            assert isinstance(self.memory, list)
            del self.memory[:removed]
        self.turnover_events += 1
        self.evicted_samples += removed

    def add(self, item: Sequence[torch.Tensor]) -> None:
        row = self._cpu_row(item)
        self.seen += 1
        self._evict_oldest_chunk_if_full()
        if isinstance(self.memory, _PackedReservoirView):
            self._ensure_append_capacity()
            self.memory.append(row)
        else:
            assert isinstance(self.memory, list)
            self.memory.append(row)
            if len(self.memory) >= min(self.capacity, self._COMPACT_AT):
                self._compact()

    def add_packed_row(
        self,
        fields: Sequence[torch.Tensor],
        row: int,
    ) -> None:
        self.seen += 1
        self._evict_oldest_chunk_if_full()
        item = tuple(field[int(row)].clone() for field in fields)
        if isinstance(self.memory, _PackedReservoirView):
            self._ensure_append_capacity()
            self.memory.append(item)
        else:
            assert isinstance(self.memory, list)
            self.memory.append(item)
            if len(self.memory) >= min(self.capacity, self._COMPACT_AT):
                self._compact()

    def add_packed_fields(self, fields: Sequence[torch.Tensor]) -> None:
        """Append a worker result in bulk with exact FIFO turnover semantics."""

        if not fields:
            return
        length = int(fields[0].shape[0])
        if any(int(field.shape[0]) != length for field in fields):
            raise ValueError("source reservoir fields have inconsistent lengths")
        if length <= 0:
            return
        source_start = 0
        self.seen += length
        while source_start < length:
            self._evict_oldest_chunk_if_full()
            if not isinstance(self.memory, _PackedReservoirView):
                assert isinstance(self.memory, list)
                compact_at = min(self.capacity, self._COMPACT_AT)
                take = min(
                    length - source_start,
                    compact_at - len(self.memory),
                )
                for row in range(source_start, source_start + take):
                    self.memory.append(
                        tuple(field[row].clone() for field in fields)
                    )
                source_start += take
                if len(self.memory) >= compact_at:
                    self._compact()
                continue
            take = min(
                length - source_start,
                self.capacity - len(self.memory),
            )
            self._ensure_append_capacity(len(self.memory) + take)
            self.memory.append_fields(fields, source_start, take)
            source_start += take

    def _fields_at_indices(
        self,
        indices: Sequence[int] | torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(self.memory, _PackedReservoirView):
            index = self.memory.physical_indices(indices)
            return tuple(
                field.index_select(0, index) for field in self.memory.fields
            )
        rows = [self.memory[int(index)] for index in indices]
        return tuple(torch.stack(field) for field in zip(*rows))

    def sample_fields(
        self,
        batch_size: int,
        *,
        rng: random.Random | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not self.memory:
            raise RuntimeError("cannot sample an empty reservoir")
        count = min(int(batch_size), len(self.memory))
        source_rng = self.rng if rng is None else rng
        indices = source_rng.sample(range(len(self.memory)), count)
        return self._fields_at_indices(indices)

    def shuffled_field_batches(
        self,
        batch_size: int,
        steps: int,
        *,
        rng: random.Random | None = None,
    ):
        """Yield shuffled batches, exhausting every row before repetition."""

        if batch_size <= 0 or steps <= 0:
            raise ValueError("batch_size and steps must be positive")
        if not self.memory:
            raise RuntimeError("cannot batch an empty reservoir")
        yielded = 0
        length = len(self.memory)
        generator = torch.Generator(device="cpu")
        source_rng = self.rng if rng is None else rng
        while yielded < int(steps):
            generator.manual_seed(source_rng.getrandbits(63))
            order = torch.randperm(length, generator=generator)
            for start in range(0, length, int(batch_size)):
                yield self._fields_at_indices(order[start : start + int(batch_size)])
                yielded += 1
                if yielded >= int(steps):
                    return

    def mean_weight(self) -> float:
        if not self.memory:
            return 1.0
        if isinstance(self.memory, _PackedReservoirView):
            first = min(
                len(self.memory),
                self.memory.storage_capacity - self.memory.start,
            )
            total = self.memory.fields[3][
                self.memory.start : self.memory.start + first
            ].sum(dtype=torch.float64)
            if first < len(self.memory):
                total += self.memory.fields[3][
                    : len(self.memory) - first
                ].sum(dtype=torch.float64)
            return max(
                1e-8,
                float((total / len(self.memory)).item()),
            )
        return max(
            1e-8,
            sum(float(row[3].item()) for row in self.memory) / len(self.memory),
        )

    def state_dict(self) -> dict[str, Any]:
        if isinstance(self.memory, _PackedReservoirView):
            preserve_physical_layout = self.memory.start != 0
            fields = (
                list(self.memory.fields)
                if preserve_physical_layout
                else [
                    field
                    if len(self.memory) == self.memory.storage_capacity
                    else field[: len(self.memory)].clone()
                    for field in self.memory.fields
                ]
            )
        elif self.memory:
            fields = [
                torch.stack(column).cpu() for column in zip(*self.memory)
            ]
        else:
            fields = []
        return {
            "capacity": self.capacity,
            "seen": self.seen,
            "format_version": 3,
            "fields": fields,
            "length": len(self.memory),
            "start": (
                self.memory.start
                if isinstance(self.memory, _PackedReservoirView)
                and preserve_physical_layout
                else 0
            ),
            "physical_layout": bool(
                isinstance(self.memory, _PackedReservoirView)
                and preserve_physical_layout
            ),
            "turnover_fraction": self.turnover_fraction,
            "turnover_events": self.turnover_events,
            "evicted_samples": self.evicted_samples,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        checkpoint_capacity = int(state.get("capacity", -1))
        if checkpoint_capacity <= 0:
            raise ValueError("checkpoint reservoir capacity is invalid")
        if checkpoint_capacity > self.capacity:
            raise ValueError(
                "checkpoint reservoir capacity exceeds configured capacity"
            )
        checkpoint_fraction = float(
            state.get("turnover_fraction", self.turnover_fraction)
        )
        if not math.isclose(
            checkpoint_fraction,
            self.turnover_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint reservoir turnover fraction differs")
        fields = [field.detach().cpu() for field in state.get("fields", [])]
        if fields:
            storage = int(fields[0].shape[0])
            if any(int(field.shape[0]) != storage for field in fields):
                raise ValueError("checkpoint reservoir fields have inconsistent lengths")
            length = int(state.get("length", storage))
            if length > self.capacity:
                raise ValueError("checkpoint reservoir exceeds configured capacity")
            start = int(state.get("start", 0))
            if not bool(state.get("physical_layout", False)) and start != 0:
                raise ValueError("checkpoint reservoir layout metadata is invalid")
            self.memory = _PackedReservoirView(fields, length=length, start=start)
        else:
            self.memory = []
        self.seen = int(state.get("seen", len(self.memory)))
        if self.seen < len(self.memory):
            raise ValueError("checkpoint reservoir seen count is invalid")
        self.turnover_events = int(state.get("turnover_events", 0))
        self.evicted_samples = int(state.get("evicted_samples", 0))
        if self.turnover_events < 0 or self.evicted_samples < 0:
            raise ValueError("checkpoint reservoir turnover counters are invalid")


def _pack_worker_buffers(
    buffers: Sequence[ReservoirBuffer],
) -> list[list[torch.Tensor]]:
    packed: list[list[torch.Tensor]] = []
    for buffer in buffers:
        if not buffer.memory:
            packed.append([])
        elif isinstance(buffer.memory, _PackedReservoirView):
            packed.append(
                list(buffer._fields_at_indices(range(len(buffer.memory))))
            )
        else:
            field_count = len(buffer.memory[0])
            packed.append(
                [
                    torch.stack([row[field] for row in buffer.memory])
                    for field in range(field_count)
                ]
            )
    return packed


def _parallel_traversal_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a deterministic group of HU traversal roots on one CPU process."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    env = payload["env_type"](**payload["env_kwargs"])
    config = dict(payload["trainer_config"])
    maximum_samples = max(
        1,
        len(payload["tasks"]) * int(config["max_nodes_per_traversal"]),
    )
    config["advantage_capacity"] = maximum_samples
    config["policy_capacity"] = maximum_samples
    worker = HeadsUpNeuralCFR(env, device="cpu", **config)
    for network, state in zip(
        worker.advantage_nets,
        payload["advantage_states"],
    ):
        network.load_state_dict(state)
        network.eval()
    worker.iteration = int(payload["iteration"])
    contexts = [
        {
            "state": state,
            "traverser": int(traverser),
            "rng": random.Random(int(seed)),
        }
        for state, traverser, seed in payload["tasks"]
    ]
    worker._run_batched_traversals(contexts)
    return {
        "advantage_samples": _pack_worker_buffers(worker.advantage_buffers),
        "policy_samples": _pack_worker_buffers(worker.policy_buffers),
        "nodes": worker._nodes_this_iteration,
        "rollouts": worker._rollouts_this_iteration,
        "depth_cutoffs": worker._depth_cutoffs,
        "node_cutoffs": worker._node_cutoffs,
        "regret_magnitudes": worker._regret_magnitudes,
    }


class HeadsUpNeuralCFR:
    """Two advantage networks and two average-policy networks."""

    def __init__(
        self,
        env,
        *,
        device: str | torch.device = "cpu",
        hidden: int = 256,
        blocks: int = 6,
        learning_rate: float = 3e-4,
        advantage_capacity: int = 25_000,
        policy_capacity: int = 25_000,
        range_capacity: int = 500_000,
        max_history: int = TRAINING_DEFAULT_MAX_HISTORY,
        max_nodes_per_traversal: int = 5_000,
        max_depth: int = 96,
        exploration: float = 0.0,
        reinitialize_advantage_each_iteration: bool = True,
        advantage_reinitialize_from_iteration: int = 1,
        advantage_reinitialize_cycle: int = 1,
        range_loss_weight: float = 0.01,
        reservoir_turnover_fraction: float = 0.18,
        seed: int = 42,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.network_architecture = NETWORK_ARCHITECTURE
        self.policy_network_architecture = POLICY_NETWORK_ARCHITECTURE
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.learning_rate = float(learning_rate)
        self.max_history = int(max_history)
        self.max_nodes_per_traversal = int(max_nodes_per_traversal)
        self.max_depth = int(max_depth)
        self.exploration = float(exploration)
        self.reinitialize_advantage_each_iteration = bool(
            reinitialize_advantage_each_iteration
        )
        self.advantage_reinitialize_from_iteration = int(
            advantage_reinitialize_from_iteration
        )
        self.advantage_reinitialize_cycle = int(
            advantage_reinitialize_cycle
        )
        self.range_loss_weight = float(range_loss_weight)
        self.reservoir_turnover_fraction = float(
            reservoir_turnover_fraction
        )
        self.seed = int(seed)

        if self.hidden <= 0 or self.blocks < 0:
            raise ValueError("hidden must be positive and blocks cannot be negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        if self.max_nodes_per_traversal <= 0 or self.max_depth <= 0:
            raise ValueError("traversal node/depth limits must be positive")
        if not 0.0 <= self.exploration < 1.0:
            raise ValueError("exploration must be in [0, 1)")
        if self.advantage_reinitialize_from_iteration <= 0:
            raise ValueError(
                "advantage_reinitialize_from_iteration must be positive"
            )
        if self.advantage_reinitialize_cycle <= 0:
            raise ValueError("advantage_reinitialize_cycle must be positive")
        if not math.isfinite(self.range_loss_weight) or self.range_loss_weight < 0.0:
            raise ValueError("range_loss_weight must be finite and nonnegative")
        if (
            not math.isfinite(self.reservoir_turnover_fraction)
            or not 0.0 < self.reservoir_turnover_fraction < 1.0
        ):
            raise ValueError(
                "reservoir_turnover_fraction must be in (0, 1)"
            )
        if NUM_PLAYERS != 2 or NUM_ACTIONS != 10:
            raise RuntimeError("the heads-up trainer requires 2 players and 10 actions")

        self.rng = random.Random(self.seed)
        self.eval_rng = random.Random(self.seed + 10_000)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.encoder = encoder_metadata(self.max_history)
        self.input_dim = information_state_size(self.max_history)
        if int(self.encoder["input_dim"]) != self.input_dim:
            raise RuntimeError("encoder metadata width is inconsistent")

        probe = self.env.new_hand(button=0)
        probe_legal = self.env.legal_actions(probe)
        probe_x = self.encode(probe, int(probe.to_act), probe_legal)
        if int(probe_x.numel()) != self.input_dim:
            raise RuntimeError("probe encoder width does not match declared schema")

        self.advantage_nets = [
            self._new_advantage_network() for _ in range(NUM_PLAYERS)
        ]
        self.policy_nets = [self._new_policy_network() for _ in range(NUM_PLAYERS)]
        self.advantage_optimizers = [
            self._new_optimizer(net) for net in self.advantage_nets
        ]
        self.policy_optimizers = [
            self._new_optimizer(net) for net in self.policy_nets
        ]
        for network in self.advantage_nets + self.policy_nets:
            network.eval()

        self.advantage_buffers = [
            ReservoirBuffer(
                advantage_capacity,
                self.rng,
                self.reservoir_turnover_fraction,
            )
            for _ in range(NUM_PLAYERS)
        ]
        self.policy_buffers = [
            ReservoirBuffer(
                policy_capacity,
                self.rng,
                self.reservoir_turnover_fraction,
            )
            for _ in range(NUM_PLAYERS)
        ]
        self.range_buffers = [
            ReservoirBuffer(
                range_capacity,
                self.rng,
                self.reservoir_turnover_fraction,
            )
            for _ in range(NUM_PLAYERS)
        ]
        self.range_last_collected_iteration = 0

        self.iteration = 0
        self.last_fitted_iteration = 0
        self.can_resume_training = True
        self.metrics: list[dict[str, float]] = []
        self._next_traverser = 0
        self._position_cycle = 0
        self.last_traverser_schedule: tuple[int, ...] = ()
        self.last_root_buttons: tuple[int, ...] = ()
        self.last_root_stacks: tuple[tuple[int, int], ...] = ()
        self._nodes_this_traversal = 0
        self._nodes_this_iteration = 0
        self._rollouts_this_iteration = 0
        self._depth_cutoffs = 0
        self._node_cutoffs = 0
        self._regret_magnitudes: list[float] = []

    def _new_advantage_network(self):
        return build_advantage_network(
            NETWORK_ARCHITECTURE,
            self.input_dim,
            self.hidden,
            self.blocks,
        ).to(self.device)

    def _new_policy_network(self):
        return build_policy_network(
            POLICY_NETWORK_ARCHITECTURE,
            self.input_dim,
            self.hidden,
            self.blocks,
        ).to(self.device)

    def _new_optimizer(self, network):
        return torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )

    def encode(
        self,
        state,
        player: int,
        legal: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if legal is None:
            legal = self.env.legal_actions(state)
        descriptors = build_action_descriptors(self.env, state)
        return encode_information_state(
            state,
            int(player),
            legal,
            self.env.bb,
            self.max_history,
            action_descriptors=descriptors,
        )

    @staticmethod
    def regret_matching(
        advantages: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if advantages.shape != legal_mask.shape:
            raise ValueError("advantages and legal_mask must have identical shapes")
        positive = torch.clamp(advantages, min=0.0) * legal_mask
        total = positive.sum()
        if float(total) > 1e-12:
            return positive / total
        # Standard regret matching permits any legal distribution here; uniform
        # keeps every finite action reachable before the approximator has signal.
        return legal_mask / legal_mask.sum().clamp(min=1.0)

    @staticmethod
    def regret_matching_batch(
        advantages: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> torch.Tensor:
        if advantages.ndim != 2 or legal_masks.shape != advantages.shape:
            raise ValueError(
                "advantages and legal masks must have matching [batch, actions] shapes"
            )
        positive = torch.clamp(advantages, min=0.0) * legal_masks
        totals = positive.sum(dim=1, keepdim=True)
        uniform = legal_masks / legal_masks.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.where(
            totals > 1e-12,
            positive / totals.clamp(min=1e-12),
            uniform,
        )

    @torch.inference_mode()
    def current_strategy(
        self,
        state,
        player: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.terminal or state.to_act is None:
            raise ValueError("current_strategy requires a live decision state")
        actor = int(state.to_act)
        if player is None:
            player = actor
        if int(player) != actor:
            raise ValueError("player must be the acting seat")
        legal = self.env.legal_actions(state)
        x = self.encode(state, actor, legal)
        mask = _legal_mask(legal)
        predicted = self.advantage_nets[actor](
            x.to(self.device).unsqueeze(0)
        )[0].cpu()
        strategy = self.regret_matching(predicted, mask)
        if self.exploration > 0.0:
            uniform = mask / mask.sum()
            strategy = (
                (1.0 - self.exploration) * strategy
                + self.exploration * uniform
            )
        strategy *= mask
        strategy /= strategy.sum()
        return x, strategy, mask

    @torch.inference_mode()
    def average_policy(self, state, player: int | None = None) -> torch.Tensor:
        if state.terminal or state.to_act is None:
            raise ValueError("average_policy requires a live decision state")
        actor = int(state.to_act)
        if player is None:
            player = actor
        if int(player) != actor:
            raise ValueError("player must be the acting seat")
        legal = self.env.legal_actions(state)
        x = self.encode(state, actor, legal).to(self.device)
        mask = _legal_mask(legal).to(self.device)
        logits = self.policy_nets[actor](x.unsqueeze(0))[0]
        probabilities = masked_softmax(logits, mask)
        return probabilities.cpu()

    @torch.inference_mode()
    def opponent_range(
        self,
        state,
        player: int | None = None,
    ) -> torch.Tensor:
        """Predict the acting player's blocker-valid opponent combination range."""

        if state.terminal or state.to_act is None:
            raise ValueError("opponent_range requires a live decision state")
        actor = int(state.to_act)
        if player is None:
            player = actor
        if int(player) != actor:
            raise ValueError("player must be the acting seat")
        legal = self.env.legal_actions(state)
        x = self.encode(state, actor, legal).to(self.device).unsqueeze(0)
        network = self.policy_nets[actor]
        if not hasattr(network, "forward_with_range"):
            raise ValueError("policy network does not contain an opponent-range head")
        _, range_logits = network.forward_with_range(x)
        valid = valid_combo_mask_from_encoded(x)
        return masked_range_probabilities(range_logits, valid)[0].cpu()

    def add_range_training_samples(
        self,
        information_states: torch.Tensor,
        opponent_combos: torch.Tensor,
        players: torch.Tensor,
        hand_ids: torch.Tensor,
        hand_weights: torch.Tensor,
    ) -> int:
        """Add independent sampled-trajectory targets to the range reservoirs."""

        count = int(information_states.shape[0])
        values = (opponent_combos, players, hand_ids, hand_weights)
        if information_states.ndim != 2 or int(information_states.shape[1]) != self.input_dim:
            raise ValueError("range information states have an invalid shape")
        if any(int(value.shape[0]) != count for value in values):
            raise ValueError("range sample tensors have inconsistent lengths")
        if count <= 0:
            return 0
        xs = information_states.detach().cpu().to(torch.float16)
        combos = opponent_combos.detach().cpu().to(torch.int16)
        seats = players.detach().cpu().to(torch.long)
        ids = hand_ids.detach().cpu().to(torch.int64)
        weights = hand_weights.detach().cpu().to(torch.float32)
        if bool(((seats < 0) | (seats >= NUM_PLAYERS)).any()):
            raise ValueError("range samples contain an invalid player")
        if bool(((combos < 0) | (combos >= 1_326)).any()):
            raise ValueError("range samples contain an invalid opponent combination")
        if bool((~torch.isfinite(weights) | (weights <= 0)).any()):
            raise ValueError("range sample weights must be finite and positive")
        valid = valid_combo_mask_from_encoded(xs.to(torch.float32))
        if not bool(
            valid.gather(1, combos.to(torch.long).unsqueeze(1)).all()
        ):
            raise ValueError("range samples contain blocker-invalid targets")
        for player in range(NUM_PLAYERS):
            selected = torch.nonzero(seats == player, as_tuple=False).flatten()
            if not len(selected):
                continue
            self.range_buffers[player].add_packed_fields(
                (
                    xs.index_select(0, selected),
                    combos.index_select(0, selected),
                    ids.index_select(0, selected),
                    weights.index_select(0, selected),
                )
            )
        return count

    @torch.inference_mode()
    def average_policy_batch(
        self,
        states: Sequence[Any],
        *,
        policy_nets: Sequence[Any] | None = None,
        batch_size: int = 4_096,
    ) -> list[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        networks = self.policy_nets if policy_nets is None else list(policy_nets)
        if len(networks) != NUM_PLAYERS:
            raise ValueError("policy_nets must contain exactly two networks")
        outputs: list[torch.Tensor | None] = [None] * len(states)
        grouped: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(NUM_PLAYERS)
        ]
        for index, state in enumerate(states):
            if state.terminal or state.to_act is None:
                raise ValueError("every batch item must be a live decision state")
            player = int(state.to_act)
            legal = self.env.legal_actions(state)
            grouped[player].append(
                (index, self.encode(state, player, legal), _legal_mask(legal))
            )
        for player, items in enumerate(grouped):
            network = networks[player]
            network.eval()
            network_device = next(network.parameters()).device
            for start in range(0, len(items), int(batch_size)):
                chunk = items[start : start + int(batch_size)]
                xs = torch.stack([item[1] for item in chunk]).to(
                    network_device,
                    non_blocking=True,
                )
                masks = torch.stack([item[2] for item in chunk]).to(
                    network_device,
                    non_blocking=True,
                )
                probabilities = masked_softmax(network(xs), masks).cpu()
                for (index, _, _), probability in zip(chunk, probabilities):
                    outputs[index] = probability
        if any(output is None for output in outputs):
            raise RuntimeError("failed to evaluate one or more policy states")
        return [output for output in outputs if output is not None]

    @staticmethod
    def _draw_action(probabilities: torch.Tensor, rng: random.Random) -> int:
        draw = rng.random()
        cumulative = 0.0
        fallback = -1
        for action, probability in enumerate(probabilities.tolist()):
            if probability <= 0.0:
                continue
            fallback = action
            cumulative += float(probability)
            if draw < cumulative:
                return action
        if fallback < 0:
            raise RuntimeError("strategy contains no positive action probability")
        return fallback

    def _rollout_value(self, state, traverser: int) -> float:
        self._rollouts_this_iteration += 1
        actions = 0
        while not state.terminal:
            _, probabilities, _ = self.current_strategy(state)
            action = self._draw_action(probabilities, self.rng)
            state = self.env.step(state, action)
            actions += 1
            if actions > 512:
                raise RuntimeError("rollout exceeded 512 actions; engine did not terminate")
        return float(state.payoffs[traverser]) / float(self.env.bb)

    def _traverse(self, state, traverser: int, depth: int = 0) -> float:
        if state.terminal:
            return float(state.payoffs[traverser]) / float(self.env.bb)
        if depth >= self.max_depth:
            self._depth_cutoffs += 1
            return self._rollout_value(state, traverser)
        if self._nodes_this_traversal >= self.max_nodes_per_traversal:
            self._node_cutoffs += 1
            return self._rollout_value(state, traverser)

        self._nodes_this_traversal += 1
        self._nodes_this_iteration += 1
        actor = int(state.to_act)
        legal = self.env.legal_actions(state)
        x, strategy, mask = self.current_strategy(state, actor)

        if actor == traverser:
            action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            action_order = list(legal)
            self.rng.shuffle(action_order)
            for action in action_order:
                action_values[action] = self._traverse(
                    self.env.step(state, action),
                    traverser,
                    depth + 1,
                )
            node_value = float((action_values * strategy).sum().item())
            regrets = (action_values - node_value) * mask
            legal_mean = regrets.abs().sum() / mask.sum().clamp(min=1.0)
            self._regret_magnitudes.append(float(legal_mean))
            self.advantage_buffers[actor].add(
                (
                    x.to(torch.float16),
                    regrets,
                    mask,
                    torch.tensor(float(self.iteration), dtype=torch.float32),
                )
            )
            return node_value

        # With two players, sampled visitation already supplies the acting
        # player's reach weighting required for the average strategy.
        self.policy_buffers[actor].add(
            (
                x.to(torch.float16),
                strategy.to(torch.float32),
                mask,
                torch.tensor(float(self.iteration), dtype=torch.float32),
                torch.tensor(
                    opponent_combo_index(state.hole[1 - actor]),
                    dtype=torch.int16,
                ),
            )
        )
        action = self._draw_action(strategy, self.rng)
        return self._traverse(
            self.env.step(state, action),
            traverser,
            depth + 1,
        )

    @torch.inference_mode()
    def _batched_current_strategies(
        self,
        requests: Sequence[tuple[Any, int]],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(requests)
        grouped: list[list[tuple[int, torch.Tensor, Sequence[int]]]] = [
            [] for _ in range(NUM_PLAYERS)
        ]
        for index, (state, player) in enumerate(requests):
            legal = self.env.legal_actions(state)
            grouped[int(player)].append(
                (index, self.encode(state, int(player), legal), legal)
            )
        for player, items in enumerate(grouped):
            if not items:
                continue
            xs = torch.stack([item[1] for item in items]).to(self.device)
            masks = torch.stack([_legal_mask(item[2]) for item in items])
            values = self.advantage_nets[player](xs).cpu()
            probabilities = self.regret_matching_batch(values, masks)
            if self.exploration > 0.0:
                uniform = masks / masks.sum(dim=1, keepdim=True).clamp(min=1.0)
                probabilities = (
                    (1.0 - self.exploration) * probabilities
                    + self.exploration * uniform
                )
            for (index, x, _), strategy, mask in zip(
                items,
                probabilities,
                masks,
            ):
                outputs[index] = (x, strategy, mask)
        if any(output is None for output in outputs):
            raise RuntimeError("failed to evaluate a traversal frontier")
        return [output for output in outputs if output is not None]

    def _rollout_coroutine(
        self,
        state,
        traverser: int,
        rng: random.Random,
    ):
        self._rollouts_this_iteration += 1
        actions = 0
        while not state.terminal:
            actor = int(state.to_act)
            _, probabilities, _ = yield (state, actor)
            action = self._draw_action(probabilities, rng)
            state = self.env.step(state, action)
            actions += 1
            if actions > 512:
                raise RuntimeError("rollout exceeded 512 actions; engine did not terminate")
        return float(state.payoffs[traverser]) / float(self.env.bb)

    def _traverse_coroutine(
        self,
        state,
        traverser: int,
        depth: int,
        rng: random.Random,
        node_counter: list[int],
    ):
        if state.terminal:
            return float(state.payoffs[traverser]) / float(self.env.bb)
        if depth >= self.max_depth:
            self._depth_cutoffs += 1
            return (yield from self._rollout_coroutine(state, traverser, rng))
        if node_counter[0] >= self.max_nodes_per_traversal:
            self._node_cutoffs += 1
            return (yield from self._rollout_coroutine(state, traverser, rng))

        node_counter[0] += 1
        self._nodes_this_iteration += 1
        actor = int(state.to_act)
        legal = self.env.legal_actions(state)
        x, strategy, mask = yield (state, actor)

        if actor == traverser:
            action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            action_order = list(legal)
            rng.shuffle(action_order)
            for action in action_order:
                action_values[action] = yield from self._traverse_coroutine(
                    self.env.step(state, action),
                    traverser,
                    depth + 1,
                    rng,
                    node_counter,
                )
            node_value = float((action_values * strategy).sum().item())
            regrets = (action_values - node_value) * mask
            self._regret_magnitudes.append(
                float(regrets.abs().sum() / mask.sum().clamp(min=1.0))
            )
            self.advantage_buffers[actor].add(
                (
                    x.to(torch.float16),
                    regrets,
                    mask,
                    torch.tensor(float(self.iteration), dtype=torch.float32),
                )
            )
            return node_value

        self.policy_buffers[actor].add(
            (
                x.to(torch.float16),
                strategy.to(torch.float32),
                mask,
                torch.tensor(float(self.iteration), dtype=torch.float32),
                torch.tensor(
                    opponent_combo_index(state.hole[1 - actor]),
                    dtype=torch.int16,
                ),
            )
        )
        action = self._draw_action(strategy, rng)
        return (
            yield from self._traverse_coroutine(
                self.env.step(state, action),
                traverser,
                depth + 1,
                rng,
                node_counter,
            )
        )

    def _run_batched_traversals(
        self,
        contexts: Sequence[dict[str, Any]],
    ) -> None:
        active: list[tuple[Any, tuple[Any, int]]] = []
        for context in contexts:
            generator = self._traverse_coroutine(
                context["state"],
                int(context["traverser"]),
                0,
                context["rng"],
                [0],
            )
            try:
                active.append((generator, next(generator)))
            except StopIteration:
                continue
        while active:
            responses = self._batched_current_strategies(
                [request for _, request in active]
            )
            next_active: list[tuple[Any, tuple[Any, int]]] = []
            for (generator, _), response in zip(active, responses):
                try:
                    next_active.append((generator, generator.send(response)))
                except StopIteration:
                    pass
            active = next_active

    def _parallel_trainer_config(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "blocks": self.blocks,
            "learning_rate": self.learning_rate,
            "advantage_capacity": 1,
            "policy_capacity": 1,
            "max_history": self.max_history,
            "max_nodes_per_traversal": self.max_nodes_per_traversal,
            "max_depth": self.max_depth,
            "exploration": self.exploration,
            "reinitialize_advantage_each_iteration": (
                self.reinitialize_advantage_each_iteration
            ),
            "advantage_reinitialize_from_iteration": (
                self.advantage_reinitialize_from_iteration
            ),
            "range_loss_weight": self.range_loss_weight,
            "reservoir_turnover_fraction": self.reservoir_turnover_fraction,
            "seed": self.seed,
        }

    @staticmethod
    def _merge_packed_samples(
        packed: Sequence[Sequence[torch.Tensor]],
        buffers: Sequence[ReservoirBuffer],
    ) -> None:
        for fields, buffer in zip(packed, buffers):
            if not fields:
                continue
            length = int(fields[0].shape[0])
            if any(int(field.shape[0]) != length for field in fields):
                raise RuntimeError("parallel traversal returned inconsistent samples")
            buffer.add_packed_fields(fields)

    def _balanced_root_assignments(
        self,
        traversals_per_player: int,
    ) -> list[tuple[int, int]]:
        """Pair every traverser with both positions within each iteration.

        For an even traversal count, every ``(traverser, button)`` pair occurs
        exactly the same number of times.  For an odd count, both traversers
        and both buttons still occur equally often overall, while
        ``_position_cycle`` alternates which traverser receives the extra root
        in each position across consecutive iterations.
        """

        assignments: list[tuple[int, int]] = []
        starting_traverser = int(self._next_traverser)
        for local_index in range(int(traversals_per_player)):
            for offset in range(NUM_PLAYERS):
                traverser = (starting_traverser + offset) % NUM_PLAYERS
                button = (
                    self._position_cycle + local_index + traverser
                ) % NUM_PLAYERS
                assignments.append((traverser, button))
        self._next_traverser = (
            starting_traverser + len(assignments)
        ) % NUM_PLAYERS
        self._position_cycle = 1 - self._position_cycle
        return assignments

    def _root_stack_schedule(
        self,
        root_count: int,
        distribution: str,
        depths_bb: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Sample the observable root-stack chance event deterministically."""

        if int(root_count) <= 0:
            raise ValueError("root_count must be positive")
        if distribution == ROOT_STACK_DISTRIBUTION_FIXED:
            fixed = int(self.env.starting_stack)
            return [(fixed, fixed)] * int(root_count)
        if distribution != ROOT_STACK_DISTRIBUTION_MIXED:
            raise ValueError(f"unknown root stack distribution {distribution!r}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in depths_bb
        ):
            raise ValueError("root stack depths must be positive integers")
        depths = tuple(depths_bb)
        if len(depths) < 2 or len(set(depths)) != len(depths):
            raise ValueError(
                "mixed root stacks require at least two unique BB depths"
            )
        if any(value <= 0 for value in depths):
            raise ValueError("root stack depths must be positive integers")
        equal_count = int(root_count) // 2
        unequal_count = int(root_count) - equal_count
        stacks_bb: list[tuple[int, int]] = [
            (depth, depth)
            for depth in (
                self.rng.choice(depths) for _ in range(equal_count)
            )
        ]
        orientation = self.rng.randrange(2)
        for index in range(unequal_count):
            first, second = sorted(self.rng.sample(depths, 2))
            if (orientation + index) % 2:
                first, second = second, first
            stacks_bb.append((first, second))
        self.rng.shuffle(stacks_bb)
        big_blind = int(self.env.bb)
        return [
            (first * big_blind, second * big_blind)
            for first, second in stacks_bb
        ]

    def _collect_parallel_traversals(
        self,
        traversals_per_player: int,
        traversal_workers: int,
        root_stack_distribution: str,
        root_stack_depths_bb: Sequence[int],
    ) -> int:
        tasks: list[tuple[Any, int, int]] = []
        schedule: list[int] = []
        buttons: list[int] = []
        assignments = self._balanced_root_assignments(traversals_per_player)
        root_stacks = self._root_stack_schedule(
            len(assignments),
            root_stack_distribution,
            root_stack_depths_bb,
        )
        for (traverser, button), stacks in zip(assignments, root_stacks):
            state = self.env.new_hand(button=button, stacks=stacks)
            schedule.append(traverser)
            buttons.append(button)
            tasks.append((state, traverser, self.rng.getrandbits(63)))
        self.last_traverser_schedule = tuple(schedule)
        self.last_root_buttons = tuple(buttons)
        self.last_root_stacks = tuple(root_stacks)

        worker_count = min(int(traversal_workers), len(tasks))
        chunk_size, extra = divmod(len(tasks), worker_count)
        chunks: list[list[tuple[Any, int, int]]] = []
        start = 0
        for worker_index in range(worker_count):
            stop = start + chunk_size + int(worker_index < extra)
            chunks.append(tasks[start:stop])
            start = stop
        advantage_states = [
            {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }
            for network in self.advantage_nets
        ]
        common = {
            "env_type": type(self.env),
            "env_kwargs": {
                "starting_stack": self.env.starting_stack,
                "small_blind": self.env.small_blind,
                "big_blind": self.env.big_blind,
                "seed": self.seed + 70_000 + self.iteration,
            },
            "trainer_config": self._parallel_trainer_config(),
            "advantage_states": advantage_states,
            "iteration": self.iteration,
        }
        payloads = [{**common, "tasks": chunk} for chunk in chunks]
        context = mp.get_context("spawn")
        with context.Pool(processes=worker_count) as pool:
            results = pool.map(_parallel_traversal_worker, payloads, chunksize=1)
        for result in results:
            self._merge_packed_samples(
                result["advantage_samples"],
                self.advantage_buffers,
            )
            self._merge_packed_samples(
                result["policy_samples"],
                self.policy_buffers,
            )
            self._nodes_this_iteration += int(result["nodes"])
            self._rollouts_this_iteration += int(result["rollouts"])
            self._depth_cutoffs += int(result["depth_cutoffs"])
            self._node_cutoffs += int(result["node_cutoffs"])
            self._regret_magnitudes.extend(result["regret_magnitudes"])
        return worker_count

    def _collect_traversals(
        self,
        traversals_per_player: int,
        traversal_workers: int = 1,
        root_stack_distribution: str = ROOT_STACK_DISTRIBUTION_FIXED,
        root_stack_depths_bb: Sequence[int] = DEFAULT_ROOT_STACK_DEPTHS_BB,
    ) -> int:
        if isinstance(traversals_per_player, bool) or int(traversals_per_player) <= 0:
            raise ValueError("traversals_per_player must be positive")
        if isinstance(traversal_workers, bool) or int(traversal_workers) <= 0:
            raise ValueError("traversal_workers must be positive")
        if int(traversal_workers) > 1:
            return self._collect_parallel_traversals(
                int(traversals_per_player),
                int(traversal_workers),
                root_stack_distribution,
                root_stack_depths_bb,
            )
        schedule: list[int] = []
        buttons: list[int] = []
        assignments = self._balanced_root_assignments(traversals_per_player)
        root_stacks = self._root_stack_schedule(
            len(assignments),
            root_stack_distribution,
            root_stack_depths_bb,
        )
        for (traverser, button), stacks in zip(assignments, root_stacks):
            state = self.env.new_hand(button=button, stacks=stacks)
            schedule.append(traverser)
            buttons.append(button)
            self._nodes_this_traversal = 0
            self._traverse(state, traverser, 0)
        self.last_traverser_schedule = tuple(schedule)
        self.last_root_buttons = tuple(buttons)
        self.last_root_stacks = tuple(root_stacks)
        return 1

    @staticmethod
    def _scaled_weights(weights: torch.Tensor, fixed_mean: float) -> torch.Tensor:
        return weights.clamp(min=1.0) / max(float(fixed_mean), 1e-8)

    def _should_reinitialize_advantage(self) -> bool:
        """Return whether this iteration starts a fresh advantage-fit cycle."""

        return (
            self.reinitialize_advantage_each_iteration
            and self.iteration >= self.advantage_reinitialize_from_iteration
            and (
                self.iteration - self.advantage_reinitialize_from_iteration
            )
            % self.advantage_reinitialize_cycle
            == 0
        )

    def _fit_advantage(
        self,
        player: int,
        steps: int,
        batch_size: int,
        *,
        batch_rng: random.Random | None = None,
        network_prepared: bool = False,
    ) -> float:
        buffer = self.advantage_buffers[player]
        if not buffer.memory or steps <= 0:
            return float("nan")
        fresh_fit = self._should_reinitialize_advantage()
        if fresh_fit and not network_prepared:
            self.advantage_nets[player] = self._new_advantage_network()
            self.advantage_optimizers[player] = self._new_optimizer(
                self.advantage_nets[player]
            )
        if fresh_fit:
            # A fresh approximator must see at least one complete reservoir
            # pass; otherwise "reinitialize" can discard more information than
            # the requested handful of SGD steps puts back.
            steps = max(
                int(steps),
                math.ceil(len(buffer) / max(1, int(batch_size))),
            )
        network = self.advantage_nets[player]
        optimizer = self.advantage_optimizers[player]
        network.train()
        total_loss = 0.0
        weight_mean = buffer.mean_weight()
        batches = (
            buffer.shuffled_field_batches(
                batch_size,
                steps,
                rng=batch_rng,
            )
            if fresh_fit
            else (
                buffer.sample_fields(batch_size, rng=batch_rng)
                for _ in range(int(steps))
            )
        )
        for xs, targets, masks, weights in batches:
            xs = xs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            weights = self._scaled_weights(
                weights.to(self.device, dtype=torch.float32),
                weight_mean,
            )
            prediction = network(xs)
            squared_error = ((prediction - targets) * masks).square()
            per_sample = squared_error.sum(dim=1) / masks.sum(dim=1).clamp(min=1.0)
            loss = (per_sample * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
        network.eval()
        return total_loss / int(steps)

    def _fit_policy(
        self,
        player: int,
        steps: int,
        batch_size: int,
        *,
        batch_rng: random.Random | None = None,
        range_batch_size: int = 2_048,
    ) -> tuple[float, float, float]:
        buffer = self.policy_buffers[player]
        if not buffer.memory or steps <= 0:
            missing = float("nan")
            return missing, missing, missing
        range_buffer = self.range_buffers[player]
        has_range_data = bool(range_buffer.memory)
        network = self.policy_nets[player]
        optimizer = self.policy_optimizers[player]
        network.train()
        total_loss = 0.0
        total_action_loss = 0.0
        total_range_loss = 0.0
        weight_mean = buffer.mean_weight()
        range_weight_mean = range_buffer.mean_weight() if has_range_data else 1.0
        steps = max(
            int(steps),
            math.ceil(len(buffer) / max(1, int(batch_size))),
        )
        policy_batches = buffer.shuffled_field_batches(
            batch_size,
            steps,
            rng=batch_rng,
        )
        range_batches = (
            range_buffer.shuffled_field_batches(
                max(1, int(range_batch_size)),
                steps,
                rng=batch_rng,
            )
            if has_range_data
            else (None for _ in range(steps))
        )
        for policy_batch, range_batch in zip(policy_batches, range_batches):
            xs, targets, masks, weights, _unused_combo = policy_batch
            xs = xs.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            weights = self._scaled_weights(
                weights.to(self.device, dtype=torch.float32),
                weight_mean,
            )
            if range_batch is None:
                logits = network(xs)
                range_logits = None
            else:
                range_xs, opponent_combos, _hand_ids, range_weights = range_batch
                range_xs = range_xs.to(self.device, dtype=torch.float32)
                opponent_combos = opponent_combos.to(
                    self.device,
                    dtype=torch.long,
                )
                range_weights = self._scaled_weights(
                    range_weights.to(self.device, dtype=torch.float32),
                    range_weight_mean,
                )
                combined = torch.cat((xs, range_xs), dim=0)
                representation = network.backbone(combined)
                action_representation = representation[: len(xs)]
                range_representation = representation[len(xs) :]
                logits = network._action_logits(action_representation, xs)
                range_logits = network.range_head(range_representation)
            logits = logits.masked_fill(masks <= 0, -1e9)
            log_probabilities = F.log_softmax(logits, dim=1)
            action_per_sample = -(targets * log_probabilities).sum(dim=1)
            action_loss = (action_per_sample * weights).mean()

            if range_logits is None:
                range_loss = None
                loss = action_loss
            else:
                range_mask = valid_combo_mask_from_encoded(range_xs)
                target_valid = range_mask.gather(
                    1,
                    opponent_combos.unsqueeze(1),
                ).squeeze(1)
                if not bool(torch.all(target_valid)):
                    raise RuntimeError(
                        "an opponent-range target is blocked by visible cards"
                    )
                range_per_sample = F.cross_entropy(
                    range_logits.masked_fill(~range_mask, -1e9),
                    opponent_combos,
                    reduction="none",
                )
                range_loss = (range_per_sample * range_weights).mean()
                loss = action_loss + self.range_loss_weight * range_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_action_loss += float(action_loss.detach().item())
            if range_loss is not None:
                total_range_loss += float(range_loss.detach().item())
        network.eval()
        return (
            total_loss / int(steps),
            total_action_loss / int(steps),
            (
                total_range_loss / int(steps)
                if has_range_data
                else float("nan")
            ),
        )

    def _fit_player_pair(
        self,
        fit_method,
        steps: int,
        batch_size: int,
        *,
        prepare_advantage: bool = False,
        fit_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Fit independent seat networks concurrently on two CUDA streams."""

        batch_rngs = [
            random.Random(self.rng.getrandbits(64))
            for _ in range(NUM_PLAYERS)
        ]
        if prepare_advantage:
            fresh_fit = self._should_reinitialize_advantage()
            if fresh_fit:
                for player in range(NUM_PLAYERS):
                    self.advantage_nets[player] = (
                        self._new_advantage_network()
                    )
                    self.advantage_optimizers[player] = (
                        self._new_optimizer(
                            self.advantage_nets[player]
                        )
                    )

        def call(player: int):
            kwargs: dict[str, Any] = {
                "batch_rng": batch_rngs[player],
            }
            if fit_kwargs:
                kwargs.update(fit_kwargs)
            if prepare_advantage:
                kwargs["network_prepared"] = True
            return fit_method(
                player,
                steps,
                batch_size,
                **kwargs,
            )

        if self.device.type != "cuda":
            return [call(player) for player in range(NUM_PLAYERS)]

        streams = [
            torch.cuda.Stream(device=self.device)
            for _ in range(NUM_PLAYERS)
        ]

        def run(player: int):
            with (
                torch.cuda.device(self.device),
                torch.cuda.stream(streams[player]),
            ):
                return call(player)

        with ThreadPoolExecutor(max_workers=NUM_PLAYERS) as executor:
            futures = [
                executor.submit(run, player)
                for player in range(NUM_PLAYERS)
            ]
            results = [future.result() for future in futures]
        torch.cuda.synchronize(self.device)
        return results

    def train_iteration(
        self,
        *,
        traversals_per_player: int = 1,
        advantage_steps: int = 16,
        policy_steps: int = 8,
        batch_size: int = 128,
        range_batch_size: int = 2_048,
        traversal_workers: int = 1,
        root_stack_distribution: str = ROOT_STACK_DISTRIBUTION_FIXED,
        root_stack_depths_bb: Sequence[int] = DEFAULT_ROOT_STACK_DEPTHS_BB,
    ) -> dict[str, float]:
        if not self.can_resume_training:
            raise RuntimeError(
                "this light checkpoint has no CFR reservoirs and is inference-only"
            )
        if (
            advantage_steps <= 0
            or policy_steps < 0
            or batch_size <= 0
            or range_batch_size <= 0
            or traversal_workers <= 0
        ):
            raise ValueError(
                "advantage_steps, batch_size, range_batch_size, and traversal_workers must be "
                "positive; policy_steps cannot be negative"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        self.iteration += 1
        self._nodes_this_iteration = 0
        self._rollouts_this_iteration = 0
        self._depth_cutoffs = 0
        self._node_cutoffs = 0
        self._regret_magnitudes = []

        workers_used = self._collect_traversals(
            traversals_per_player,
            traversal_workers,
            root_stack_distribution,
            root_stack_depths_bb,
        )
        traversal_finished = time.perf_counter()
        advantage_losses = self._fit_player_pair(
            self._fit_advantage,
            advantage_steps,
            batch_size,
            prepare_advantage=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        advantage_fit_finished = time.perf_counter()
        policy_fit_results = self._fit_player_pair(
            self._fit_policy,
            policy_steps,
            batch_size,
            fit_kwargs={"range_batch_size": int(range_batch_size)},
        )
        policy_losses = [result[0] for result in policy_fit_results]
        policy_action_losses = [result[1] for result in policy_fit_results]
        policy_range_losses = [result[2] for result in policy_fit_results]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        policy_fit_finished = time.perf_counter()
        elapsed = time.perf_counter() - started
        finite_regrets = self._regret_magnitudes
        finite_advantage_losses = [
            value for value in advantage_losses if math.isfinite(value)
        ]
        finite_policy_losses = [
            value for value in policy_losses if math.isfinite(value)
        ]
        finite_policy_action_losses = [
            value for value in policy_action_losses if math.isfinite(value)
        ]
        finite_policy_range_losses = [
            value for value in policy_range_losses if math.isfinite(value)
        ]
        row: dict[str, float] = {
            "iteration": float(self.iteration),
            "nodes": float(self._nodes_this_iteration),
            "rollouts": float(self._rollouts_this_iteration),
            "seconds": float(elapsed),
            "traversal_seconds": float(traversal_finished - started),
            "advantage_fit_seconds": float(
                advantage_fit_finished - traversal_finished
            ),
            "policy_fit_seconds": float(
                policy_fit_finished - advantage_fit_finished
            ),
            "fitting_seconds": float(
                elapsed - (traversal_finished - started)
            ),
            "traversal_workers": float(workers_used),
            "root_equal_stack_fraction": float(
                sum(first == second for first, second in self.last_root_stacks)
                / max(1, len(self.last_root_stacks))
            ),
            "root_min_effective_stack_bb": float(
                min(min(stacks) for stacks in self.last_root_stacks)
                / float(self.env.bb)
            ),
            "root_max_effective_stack_bb": float(
                max(min(stacks) for stacks in self.last_root_stacks)
                / float(self.env.bb)
            ),
            "traversal_nodes_per_second": float(
                self._nodes_this_iteration
                / max(traversal_finished - started, 1e-9)
            ),
            "gpu_peak_memory_mb": float(
                torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                if self.device.type == "cuda"
                else 0.0
            ),
            "gpu_memory_allocated_mb": float(
                torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                if self.device.type == "cuda"
                else 0.0
            ),
            "gpu_memory_reserved_mb": float(
                torch.cuda.memory_reserved(self.device) / (1024 * 1024)
                if self.device.type == "cuda"
                else 0.0
            ),
            "depth_cutoffs": float(self._depth_cutoffs),
            "node_cutoffs": float(self._node_cutoffs),
            "advantage_samples": float(
                sum(len(buffer) for buffer in self.advantage_buffers)
            ),
            "policy_samples": float(
                sum(len(buffer) for buffer in self.policy_buffers)
            ),
            "range_samples": float(
                sum(len(buffer) for buffer in self.range_buffers)
            ),
            "mean_abs_regret": (
                sum(finite_regrets) / len(finite_regrets)
                if finite_regrets
                else float("nan")
            ),
            "advantage_loss": (
                sum(finite_advantage_losses) / len(finite_advantage_losses)
                if finite_advantage_losses
                else float("nan")
            ),
            "policy_loss": (
                sum(finite_policy_losses) / len(finite_policy_losses)
                if finite_policy_losses
                else float("nan")
            ),
            "policy_action_loss": (
                sum(finite_policy_action_losses)
                / len(finite_policy_action_losses)
                if finite_policy_action_losses
                else float("nan")
            ),
            "policy_range_loss": (
                sum(finite_policy_range_losses)
                / len(finite_policy_range_losses)
                if finite_policy_range_losses
                else float("nan")
            ),
            "range_loss_weight": float(self.range_loss_weight),
            "parallel_player_fitting": float(self.device.type == "cuda"),
        }
        for player in range(NUM_PLAYERS):
            row[f"adv_loss_p{player}"] = float(advantage_losses[player])
            row[f"policy_loss_p{player}"] = float(policy_losses[player])
            row[f"policy_action_loss_p{player}"] = float(
                policy_action_losses[player]
            )
            row[f"policy_range_loss_p{player}"] = float(
                policy_range_losses[player]
            )
            row[f"adv_buffer_p{player}"] = float(
                len(self.advantage_buffers[player])
            )
            row[f"policy_buffer_p{player}"] = float(
                len(self.policy_buffers[player])
            )
            row[f"range_buffer_p{player}"] = float(
                len(self.range_buffers[player])
            )
            row[f"adv_turnover_events_p{player}"] = float(
                self.advantage_buffers[player].turnover_events
            )
            row[f"policy_turnover_events_p{player}"] = float(
                self.policy_buffers[player].turnover_events
            )
            row[f"adv_evicted_samples_p{player}"] = float(
                self.advantage_buffers[player].evicted_samples
            )
            row[f"policy_evicted_samples_p{player}"] = float(
                self.policy_buffers[player].evicted_samples
            )
            row[f"range_turnover_events_p{player}"] = float(
                self.range_buffers[player].turnover_events
            )
            row[f"range_evicted_samples_p{player}"] = float(
                self.range_buffers[player].evicted_samples
            )
        self.last_fitted_iteration = self.iteration
        self.metrics.append(dict(row))
        return row

    def recover_incomplete_fit(
        self,
        *,
        advantage_steps: int,
        policy_steps: int,
        batch_size: int,
        range_batch_size: int = 2_048,
    ) -> dict[str, float]:
        """Refit every network when a checkpoint captured a partial iteration."""
        if self.last_fitted_iteration >= self.iteration:
            return {"recovery_fit_seconds": 0.0}
        started = time.perf_counter()
        advantage_losses = self._fit_player_pair(
            self._fit_advantage,
            advantage_steps,
            batch_size,
            prepare_advantage=True,
        )
        policy_fit_results = self._fit_player_pair(
            self._fit_policy,
            policy_steps,
            batch_size,
            fit_kwargs={"range_batch_size": int(range_batch_size)},
        )
        policy_losses = [result[0] for result in policy_fit_results]
        policy_action_losses = [result[1] for result in policy_fit_results]
        policy_range_losses = [result[2] for result in policy_fit_results]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.last_fitted_iteration = self.iteration
        return {
            "iteration": float(self.iteration),
            "recovery_fit_seconds": float(time.perf_counter() - started),
            "advantage_loss": float(
                sum(advantage_losses) / max(1, len(advantage_losses))
            ),
            "policy_loss": float(
                sum(policy_losses) / max(1, len(policy_losses))
            ),
            "policy_action_loss": float(
                sum(policy_action_losses) / max(1, len(policy_action_losses))
            ),
            "policy_range_loss": float(
                sum(policy_range_losses) / max(1, len(policy_range_losses))
            ),
        }

    def _make_environment(self, seed: int):
        return type(self.env)(
            starting_stack=self.env.starting_stack,
            small_blind=self.env.small_blind,
            big_blind=self.env.big_blind,
            seed=seed,
        )

    @torch.inference_mode()
    def evaluate_vs_random(
        self,
        games_per_seat: int = 100,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        if isinstance(games_per_seat, bool) or int(games_per_seat) <= 0:
            raise ValueError("games_per_seat must be positive")
        eval_seed = self.seed + 20_000 if seed is None else int(seed)
        evaluation_env = self._make_environment(eval_seed)
        evaluation_rng = random.Random(eval_seed + 1)
        payoffs = [0.0, 0.0]
        action_counts = [0] * NUM_ACTIONS
        for hero in range(NUM_PLAYERS):
            for game_index in range(int(games_per_seat)):
                state = evaluation_env.new_hand(
                    button=(hero + game_index) % NUM_PLAYERS
                )
                actions = 0
                while not state.terminal:
                    actor = int(state.to_act)
                    legal = evaluation_env.legal_actions(state)
                    if actor == hero:
                        # The trainer and evaluation engines have the same
                        # locked schema, so the trainer encoder can consume this
                        # immutable state directly.
                        probabilities = self.average_policy(state, actor)
                        action = self._draw_action(probabilities, evaluation_rng)
                    else:
                        action = legal[evaluation_rng.randrange(len(legal))]
                    action_counts[action] += 1
                    state = evaluation_env.step(state, action)
                    actions += 1
                    if actions > 512:
                        raise RuntimeError("evaluation hand exceeded 512 actions")
                payoffs[hero] += float(state.payoffs[hero]) / float(self.env.bb)
        seat_ev = [
            payoff / int(games_per_seat) for payoff in payoffs
        ]
        return {
            "games_per_seat": int(games_per_seat),
            "seat_ev_bb": seat_ev,
            "mean_ev_bb": sum(seat_ev) / NUM_PLAYERS,
            "action_counts": action_counts,
        }

    def save(self, path: str | Path, include_buffers: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "kind": CHECKPOINT_KIND,
            "version": CHECKPOINT_VERSION,
            "network_architecture": self.network_architecture,
            "policy_network_architecture": POLICY_NETWORK_ARCHITECTURE,
            "range_schema_version": "exact_opponent_combos_v1_1326",
            "num_players": NUM_PLAYERS,
            "num_actions": NUM_ACTIONS,
            "action_names": tuple(ACTION_NAMES),
            "engine_schema_version": ENGINE_SCHEMA_VERSION,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "encoder": dict(self.encoder),
            "input_dim": self.input_dim,
            "environment": {
                "starting_stack": self.env.starting_stack,
                "small_blind": self.env.small_blind,
                "big_blind": self.env.big_blind,
            },
            "iteration": self.iteration,
            "last_fitted_iteration": self.last_fitted_iteration,
            "can_resume_training": bool(include_buffers),
            "next_traverser": self._next_traverser,
            "position_cycle": self._position_cycle,
            "metrics": self.metrics,
            "config": {
                "hidden": self.hidden,
                "blocks": self.blocks,
                "learning_rate": self.learning_rate,
                "advantage_capacity": self.advantage_buffers[0].capacity,
                "policy_capacity": self.policy_buffers[0].capacity,
                "range_capacity": self.range_buffers[0].capacity,
                "max_history": self.max_history,
                "max_nodes_per_traversal": self.max_nodes_per_traversal,
                "max_depth": self.max_depth,
                "exploration": self.exploration,
                "reinitialize_advantage_each_iteration": (
                    self.reinitialize_advantage_each_iteration
                ),
                "advantage_reinitialize_from_iteration": (
                    self.advantage_reinitialize_from_iteration
                ),
                "advantage_reinitialize_cycle": (
                    self.advantage_reinitialize_cycle
                ),
                "range_loss_weight": self.range_loss_weight,
                "reservoir_turnover_fraction": (
                    self.reservoir_turnover_fraction
                ),
                "seed": self.seed,
            },
            "advantage_nets": [
                network.state_dict() for network in self.advantage_nets
            ],
            "policy_nets": [network.state_dict() for network in self.policy_nets],
            "advantage_optimizers": [
                optimizer.state_dict() for optimizer in self.advantage_optimizers
            ],
            "policy_optimizers": [
                optimizer.state_dict() for optimizer in self.policy_optimizers
            ],
            "rng_state": self.rng.getstate(),
            "eval_rng_state": self.eval_rng.getstate(),
            "range_last_collected_iteration": self.range_last_collected_iteration,
            "torch_rng_state": torch.get_rng_state(),
        }
        if hasattr(self.env, "rng") and hasattr(self.env.rng, "getstate"):
            checkpoint["env_rng_state"] = self.env.rng.getstate()
        if hasattr(self.env, "_last_button"):
            checkpoint["env_last_button"] = int(self.env._last_button)
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if include_buffers:
            checkpoint["advantage_buffers"] = [
                buffer.state_dict() for buffer in self.advantage_buffers
            ]
            checkpoint["policy_buffers"] = [
                buffer.state_dict() for buffer in self.policy_buffers
            ]
            checkpoint["range_buffers"] = [
                buffer.state_dict() for buffer in self.range_buffers
            ]
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        env,
        *,
        device: str | torch.device = "cpu",
    ) -> "HeadsUpNeuralCFR":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("kind") != CHECKPOINT_KIND:
            raise ValueError("checkpoint is not a heads-up Deep CFR artifact")
        if int(checkpoint.get("version", -1)) != CHECKPOINT_VERSION:
            raise ValueError("unsupported heads-up checkpoint version")
        if checkpoint.get("network_architecture") != NETWORK_ARCHITECTURE:
            raise ValueError("checkpoint network architecture does not match current code")
        if (
            checkpoint.get("policy_network_architecture")
            != POLICY_NETWORK_ARCHITECTURE
        ):
            raise ValueError(
                "checkpoint policy architecture does not match current code"
            )
        if (
            checkpoint.get("range_schema_version")
            != "exact_opponent_combos_v1_1326"
        ):
            raise ValueError("checkpoint opponent-range schema does not match")
        if int(checkpoint.get("num_players", -1)) != NUM_PLAYERS:
            raise ValueError("checkpoint player count does not match heads-up")
        if int(checkpoint.get("num_actions", -1)) != NUM_ACTIONS:
            raise ValueError("checkpoint action count does not match the engine")
        if tuple(checkpoint.get("action_names", ())) != tuple(ACTION_NAMES):
            raise ValueError("checkpoint action order does not match the engine")
        if checkpoint.get("engine_schema_version") != ENGINE_SCHEMA_VERSION:
            raise ValueError("checkpoint engine schema does not match current code")
        if checkpoint.get("action_schema_version") != ACTION_SCHEMA_VERSION:
            raise ValueError("checkpoint action schema does not match current code")

        environment = checkpoint.get("environment", {})
        for name, current in (
            ("starting_stack", env.starting_stack),
            ("small_blind", env.small_blind),
            ("big_blind", env.big_blind),
        ):
            if int(environment.get(name, -1)) != int(current):
                raise ValueError(
                    f"checkpoint {name} does not match the supplied environment"
                )

        config = dict(checkpoint["config"])
        config.setdefault("range_capacity", 500_000)
        expected_encoder = encoder_metadata(int(config["max_history"]))
        stored_encoder = dict(checkpoint.get("encoder", {}))
        if stored_encoder != expected_encoder:
            raise ValueError("checkpoint encoder metadata does not match current code")
        trainer = cls(env, device=device, **config)
        if int(checkpoint.get("input_dim", -1)) != trainer.input_dim:
            raise ValueError("checkpoint input dimension does not match current code")

        for network, state in zip(
            trainer.advantage_nets, checkpoint["advantage_nets"]
        ):
            network.load_state_dict(state)
            network.eval()
        for network, state in zip(trainer.policy_nets, checkpoint["policy_nets"]):
            network.load_state_dict(state)
            network.eval()
        for optimizer, state in zip(
            trainer.advantage_optimizers,
            checkpoint["advantage_optimizers"],
        ):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(
            trainer.policy_optimizers,
            checkpoint["policy_optimizers"],
        ):
            optimizer.load_state_dict(state)
        for optimizer in trainer.advantage_optimizers + trainer.policy_optimizers:
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(trainer.device)

        has_independent_range_buffers = "range_buffers" in checkpoint
        if not has_independent_range_buffers:
            # Legacy auxiliary supervision came from correlated CFR branches.
            # Preserve the shared trunk/action head, but restart only the
            # unreliable range projection and its Adam moments.
            with torch.no_grad():
                for network in trainer.policy_nets:
                    network.range_head.weight.zero_()
                    network.range_head.bias.zero_()
            for network, optimizer in zip(
                trainer.policy_nets,
                trainer.policy_optimizers,
            ):
                for parameter in network.range_head.parameters():
                    optimizer.state.pop(parameter, None)

        trainer.iteration = int(checkpoint["iteration"])
        trainer.last_fitted_iteration = int(
            checkpoint.get("last_fitted_iteration", trainer.iteration)
        )
        trainer.can_resume_training = bool(
            checkpoint.get(
                "can_resume_training",
                "advantage_buffers" in checkpoint
                and "policy_buffers" in checkpoint,
            )
        )
        trainer._next_traverser = int(checkpoint.get("next_traverser", 0))
        trainer._position_cycle = int(checkpoint.get("position_cycle", 0))
        trainer.metrics = [dict(row) for row in checkpoint.get("metrics", [])]
        trainer.range_last_collected_iteration = int(
            checkpoint.get("range_last_collected_iteration", trainer.iteration)
        )
        trainer.rng.setstate(checkpoint["rng_state"])
        trainer.eval_rng.setstate(checkpoint["eval_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if (
            "torch_cuda_rng_state_all" in checkpoint
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
        if "env_rng_state" in checkpoint:
            trainer.env.rng.setstate(checkpoint["env_rng_state"])
        if "env_last_button" in checkpoint:
            trainer.env._last_button = int(checkpoint["env_last_button"])
        if "advantage_buffers" in checkpoint:
            for buffer, state in zip(
                trainer.advantage_buffers,
                checkpoint["advantage_buffers"],
            ):
                buffer.load_state_dict(state)
        if "policy_buffers" in checkpoint:
            for buffer, state in zip(
                trainer.policy_buffers,
                checkpoint["policy_buffers"],
            ):
                buffer.load_state_dict(state)
        if "range_buffers" in checkpoint:
            for buffer, state in zip(
                trainer.range_buffers,
                checkpoint["range_buffers"],
            ):
                buffer.load_state_dict(state)
        return trainer


__all__ = [
    "CHECKPOINT_KIND",
    "CHECKPOINT_VERSION",
    "HeadsUpNeuralCFR",
    "NETWORK_ARCHITECTURE",
    "POLICY_NETWORK_ARCHITECTURE",
    "ReservoirBuffer",
    "TRAINING_DEFAULT_MAX_HISTORY",
    "ROOT_STACK_DISTRIBUTION_FIXED",
    "ROOT_STACK_DISTRIBUTION_MIXED",
    "DEFAULT_ROOT_STACK_DEPTHS_BB",
]
