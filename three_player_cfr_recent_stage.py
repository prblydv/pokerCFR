"""Practical neural external-sampling CFR for the three-player engine.

This is an *approximate* multiplayer CFR trainer.  Three-player poker is not a
two-player zero-sum game, so falling loss curves or positive EV versus random
opponents are useful diagnostics, not a proof of Nash convergence.

Tournament mode trains a single stack-conditioned policy on a mixture of
three-handed and heads-up *hand roots*.  Optional self-play continuations feed
realistic resulting stack distributions back into that mixture.  Utilities are
still per-hand chip gains, not finish-position/ICM utility, and a traversal does
not recurse across hand boundaries.  It therefore teaches variable depth,
elimination awareness, and heads-up play without claiming to solve the complete
multi-hand tournament as one extensive-form game.
"""

from __future__ import annotations

import copy
import bisect
import math
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from three_player_engine import ACTION_NAMES, NUM_ACTIONS
from three_player_models import (
    DEFAULT_MAX_HISTORY,
    AdvantageNetwork,
    NETWORK_ARCHITECTURES,
    PolicyNetwork,
    build_advantage_network,
    build_policy_network,
    encode_information_state,
    information_state_size,
    masked_softmax,
)


class _PackedReservoirView(Sequence[tuple[torch.Tensor, ...]]):
    """Lazy row view over contiguous reservoir fields.

    A full production checkpoint contains 900,000 samples and four tensor
    fields per sample.  Materialising every row as a tuple of tensor views on
    load creates 3.6 million Python/Tensor objects.  This view creates only the
    rows that sampling or diagnostics actually touch.
    """

    def __init__(self, fields: Sequence[torch.Tensor]):
        self.fields = [field.detach().cpu() for field in fields]
        self.length = int(self.fields[0].shape[0]) if self.fields else 0
        if any(int(field.shape[0]) != self.length for field in self.fields):
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
        return tuple(field[row] for field in self.fields)

    def replace(self, index: int, item: tuple[torch.Tensor, ...]) -> None:
        if len(item) != len(self.fields):
            raise ValueError("reservoir item has an inconsistent width")
        for field, value in zip(self.fields, item):
            source = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)
            field[int(index)].copy_(source)

    @property
    def storage_capacity(self) -> int:
        return int(self.fields[0].shape[0]) if self.fields else 0

    def reserve(self, capacity: int) -> None:
        """Grow contiguous backing tensors without materialising row objects."""

        target = int(capacity)
        if target <= self.storage_capacity:
            return
        expanded = []
        for field in self.fields:
            destination = torch.empty(
                (target, *field.shape[1:]), dtype=field.dtype, device="cpu"
            )
            destination[: self.length].copy_(field[: self.length])
            expanded.append(destination)
        self.fields = expanded

    def append(self, item: tuple[torch.Tensor, ...]) -> None:
        if self.length >= self.storage_capacity:
            raise RuntimeError("packed reservoir storage is full")
        self.replace(self.length, item)
        self.length += 1

    def bootstrap_to_capacity(self, capacity: int, rng: random.Random) -> None:
        """Expand a historical sample by unbiased bootstrap resampling.

        A larger reservoir cannot reconstruct old observations that were
        discarded at the former capacity. Sampling the retained uniform
        reservoir with replacement preserves its historical distribution in
        expectation and avoids biasing every new slot toward recent data.
        """

        target = int(capacity)
        if target <= self.length:
            return
        if self.length <= 0:
            raise RuntimeError("cannot bootstrap an empty packed reservoir")
        original_length = self.length
        self.reserve(target)
        indices = torch.tensor(
            [rng.randrange(original_length) for _ in range(target - original_length)],
            dtype=torch.long,
        )
        for field in self.fields:
            field[original_length:target].copy_(
                field[:original_length].index_select(0, indices)
            )
        self.length = target


class ReservoirBuffer:
    """Uniform reservoir memory with deterministic sampling."""

    def __init__(self, capacity: int, rng: random.Random):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = rng
        self.memory: Sequence[tuple[torch.Tensor, ...]] = []
        self.seen = 0

    def _compact(self) -> None:
        if isinstance(self.memory, _PackedReservoirView) or not self.memory:
            return
        width = len(self.memory[0])
        if any(len(item) != width for item in self.memory):
            raise RuntimeError("reservoir items have inconsistent widths")
        fields = [
            torch.stack(
                [
                    value.detach().cpu()
                    if torch.is_tensor(value)
                    else torch.as_tensor(value)
                    for value in (item[field] for item in self.memory)
                ]
            )
            for field in range(width)
        ]
        self.memory = _PackedReservoirView(fields)

    def add(self, item: tuple[torch.Tensor, ...]) -> None:
        self.seen += 1
        if len(self.memory) < self.capacity:
            if isinstance(self.memory, _PackedReservoirView):
                self.memory.reserve(self.capacity)
                self.memory.append(item)
            else:
                assert isinstance(self.memory, list)
                self.memory.append(item)
            if len(self.memory) == self.capacity:
                self._compact()
            return
        index = self.rng.randrange(self.seen)
        if index < self.capacity:
            if isinstance(self.memory, _PackedReservoirView):
                self.memory.replace(index, item)
            else:
                self.memory[index] = item

    def add_packed_row(
        self, fields: Sequence[torch.Tensor], row: int
    ) -> bool:
        """Consider one packed candidate and copy it only when retained.

        Parallel traversal returns field-major tensors containing millions of
        candidate observations.  Once the cumulative reservoir is full, nearly
        all of those candidates are rejected by reservoir sampling.  Building
        and cloning a complete row before making that decision wastes most of
        the merge time and memory bandwidth.

        This method deliberately performs the same ``seen`` increment and the
        same RNG draw as :meth:`add`.  It changes only when tensor data is
        copied, so a buffer with the same state and RNG retains byte-identical
        rows after processing the same ordered candidate stream.
        """

        self.seen += 1
        append = len(self.memory) < self.capacity
        destination: int | None
        if append:
            destination = len(self.memory)
        else:
            candidate = self.rng.randrange(self.seen)
            destination = candidate if candidate < self.capacity else None
        if destination is None:
            return False

        item = tuple(field[int(row)].clone() for field in fields)
        if append:
            if isinstance(self.memory, _PackedReservoirView):
                self.memory.reserve(self.capacity)
                self.memory.append(item)
            else:
                assert isinstance(self.memory, list)
                self.memory.append(item)
            if len(self.memory) == self.capacity:
                self._compact()
        elif isinstance(self.memory, _PackedReservoirView):
            self.memory.replace(destination, item)
        else:
            self.memory[destination] = item
        return True

    def resize_capacity(self, capacity: int) -> None:
        """Increase capacity while retaining an approximate historical sample."""

        target = int(capacity)
        if target <= 0:
            raise ValueError("capacity must be positive")
        if target < len(self.memory):
            raise ValueError("cannot shrink a populated reservoir")
        if target == self.capacity:
            return
        if target < self.capacity:
            self.capacity = target
            return
        if self.memory:
            self._compact()
            assert isinstance(self.memory, _PackedReservoirView)
            self.memory.bootstrap_to_capacity(target, self.rng)
            # The bootstrapped rows are virtual historical observations. This
            # keeps the reservoir replacement probability internally coherent.
            self.seen = max(self.seen, target)
        self.capacity = target

    def sample(self, size: int) -> list[tuple[torch.Tensor, ...]]:
        if not self.memory:
            return []
        count = min(int(size), len(self.memory))
        indices = self.rng.sample(range(len(self.memory)), count)
        return [self.memory[index] for index in indices]

    def sample_fields(self, size: int) -> tuple[torch.Tensor, ...]:
        """Sample one tensor per field without materialising Python rows.

        Full checkpoints restore reservoirs as contiguous field tensors.  The
        fitting hot path used to turn every selected row into a tuple and then
        stack each tuple field back into a tensor.  Indexing the packed fields
        directly preserves the identical reservoir sample and order while
        avoiding thousands of short-lived Python and Tensor objects per step.
        """

        if not self.memory:
            return ()
        count = min(int(size), len(self.memory))
        indices = self.rng.sample(range(len(self.memory)), count)
        return self._fields_at_indices(indices)

    def _fields_at_indices(self, indices: Sequence[int]) -> tuple[torch.Tensor, ...]:
        if isinstance(self.memory, _PackedReservoirView):
            index = (
                indices.detach().cpu().to(torch.long)
                if torch.is_tensor(indices)
                else torch.tensor(indices, dtype=torch.long)
            )
            return tuple(field.index_select(0, index) for field in self.memory.fields)

        rows = [self.memory[index] for index in indices]
        return tuple(torch.stack(field) for field in zip(*rows))

    def shuffled_field_batches(
        self, size: int, steps: int
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        """Yield shuffled, non-repeating batches, reshuffling after each epoch.

        A fresh advantage approximator is required to make at least one pass
        over cumulative memory. Independent random minibatches repeat roughly
        37% of their work before covering the buffer. A random permutation
        retains uniform ordering while ensuring every entry is fitted once
        before any entry repeats. The final batch of an epoch may be smaller.
        """

        if size <= 0 or steps <= 0:
            raise ValueError("batch size and steps must be positive")
        if not self.memory:
            return

        yielded = 0
        length = len(self.memory)
        while yielded < int(steps):
            order = list(range(length))
            self.rng.shuffle(order)
            for start in range(0, length, int(size)):
                if yielded >= int(steps):
                    return
                yield self._fields_at_indices(order[start : start + int(size)])
                yielded += 1

    def street_balanced_field_batches(
        self, size: int, steps: int
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        """Yield equal-quota preflop/flop/turn/river minibatches.

        The first four observation values are the encoder's street one-hot.
        Each non-empty street is independently shuffled and exhausted before
        reshuffling, so a small preflop partition receives regular updates
        without repeatedly selecting the same few rows inside one pass.
        """

        if size <= 0 or steps <= 0:
            raise ValueError("batch size and steps must be positive")
        if not self.memory:
            return
        self._compact()
        assert isinstance(self.memory, _PackedReservoirView)
        observations = self.memory.fields[0][: len(self.memory)]
        if observations.ndim != 2 or observations.shape[1] < 4:
            raise ValueError("street-balanced sampling requires encoded observations")
        street = observations[:, :4].float().argmax(dim=1)
        pools = [
            torch.nonzero(street == value, as_tuple=False).flatten()
            for value in range(4)
        ]
        available = [value for value, pool in enumerate(pools) if pool.numel()]
        if not available:
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.rng.getrandbits(63))
        orders = [
            pool[torch.randperm(pool.numel(), generator=generator)]
            if pool.numel()
            else pool
            for pool in pools
        ]
        cursors = [0, 0, 0, 0]

        def take(street_index: int, count: int) -> torch.Tensor:
            chunks: list[torch.Tensor] = []
            remaining = int(count)
            while remaining > 0:
                order = orders[street_index]
                cursor = cursors[street_index]
                available_count = int(order.numel()) - cursor
                if available_count <= 0:
                    pool = pools[street_index]
                    order = pool[
                        torch.randperm(pool.numel(), generator=generator)
                    ]
                    orders[street_index] = order
                    cursors[street_index] = 0
                    cursor = 0
                    available_count = int(order.numel())
                chosen = min(remaining, available_count)
                chunks.append(order[cursor : cursor + chosen])
                cursors[street_index] += chosen
                remaining -= chosen
            return torch.cat(chunks)

        base, remainder = divmod(int(size), len(available))
        for step in range(int(steps)):
            indices = []
            for offset, street_index in enumerate(available):
                count = base + int(offset < remainder)
                if count:
                    indices.append(take(street_index, count))
            batch_indices = torch.cat(indices)
            permutation = torch.randperm(
                batch_indices.numel(), generator=generator
            )
            yield self._fields_at_indices(batch_indices[permutation])

    def __len__(self) -> int:
        return len(self.memory)

    def state_dict(self) -> dict[str, Any]:
        # Serialize each field as one contiguous tensor.  Saving millions of
        # tiny tensor objects makes large production checkpoints dramatically
        # slower and larger because pickle/storage metadata dominates payload
        # data.  Runtime storage remains a reservoir list for O(1) replacement.
        if isinstance(self.memory, _PackedReservoirView):
            fields = [
                field
                if self.memory.length == self.memory.storage_capacity
                else field[: self.memory.length].clone()
                for field in self.memory.fields
            ]
        elif self.memory:
            width = len(self.memory[0])
            if any(len(item) != width for item in self.memory):
                raise RuntimeError("reservoir items have inconsistent widths")
            fields = [
                torch.stack(
                    [
                        value.detach().cpu()
                        if torch.is_tensor(value)
                        else torch.as_tensor(value)
                        for value in (item[field] for item in self.memory)
                    ]
                )
                for field in range(width)
            ]
        else:
            fields = []
        return {
            "format_version": 2,
            "capacity": self.capacity,
            "seen": self.seen,
            "fields": fields,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.seen = int(state["seen"])
        if int(state.get("format_version", 1)) >= 2:
            fields = [value.detach().cpu() for value in state.get("fields", [])]
            if not fields:
                self.memory = []
                return
            self.memory = _PackedReservoirView(fields)
            return
        self.memory = [
            tuple(value.detach().cpu() if torch.is_tensor(value) else value for value in item)
            for item in state["memory"]
        ]


class RecentWindowBuffer(ReservoirBuffer):
    """Fixed-size RAM-only window with one uniform sample per CFR iteration.

    The cumulative :class:`ReservoirBuffer` remains the unbiased lifetime
    memory.  This companion keeps ``capacity / window_iterations`` uniformly
    selected candidates from each of the latest iterations in a circular,
    contiguous tensor store.  It is intentionally omitted from checkpoints:
    production checkpoints are already disk-bound and the window safely warms
    up again after a resume.
    """

    def __init__(
        self, capacity: int, window_iterations: int, rng: random.Random
    ) -> None:
        if capacity <= 0 or window_iterations <= 0:
            raise ValueError("recent capacity and window must be positive")
        if capacity % window_iterations:
            raise ValueError("recent capacity must be divisible by its iteration window")
        super().__init__(capacity, rng)
        self.window_iterations = int(window_iterations)
        self.per_iteration = self.capacity // self.window_iterations
        self._next_slot = 0
        self._completed_iterations = 0
        self._pending = ReservoirBuffer(self.per_iteration, self.rng)

    def consider(self, item: tuple[torch.Tensor, ...]) -> None:
        """Consider a scalar-path candidate for the current iteration."""

        self._pending.add(item)

    def _commit_fields(self, fields: Sequence[torch.Tensor]) -> None:
        if not fields:
            self._completed_iterations += 1
            return
        count = int(fields[0].shape[0])
        if count > self.per_iteration:
            raise RuntimeError("recent iteration exceeds its retention quota")
        if count < self.per_iteration:
            # A player's traversal stream can legitimately contain fewer
            # information sets than the configured quota (policy samples are
            # especially variable). Keep the fixed-size iteration blocks that
            # make circular rotation cheap by repeating the retained rows as
            # evenly as possible. Every candidate occurs either floor(q / n)
            # or ceil(q / n) times, with the extra copies chosen uniformly.
            # This preserves equal weighting between recent CFR iterations and
            # avoids leaving stale rows behind when a short block rotates.
            repeats, remainder = divmod(self.per_iteration, count)
            indices = list(range(count)) * repeats
            if remainder:
                indices.extend(self.rng.sample(range(count), remainder))
            index = torch.tensor(indices, dtype=torch.long)
            fields = [field.index_select(0, index) for field in fields]
            count = self.per_iteration

        if not self.memory:
            storage = [
                torch.empty((self.capacity, *field.shape[1:]), dtype=field.dtype)
                for field in fields
            ]
            self.memory = _PackedReservoirView(storage)
            self.memory.length = 0
        assert isinstance(self.memory, _PackedReservoirView)
        if len(self.memory) < self.capacity:
            start = len(self.memory)
            self.memory.length += count
        else:
            start = self._next_slot * self.per_iteration
        for destination, source in zip(self.memory.fields, fields):
            destination[start : start + count].copy_(source)
        self._next_slot = (self._next_slot + 1) % self.window_iterations
        self._completed_iterations += 1

    def add_packed_iteration(
        self, chunks: Sequence[Sequence[torch.Tensor]]
    ) -> None:
        """Uniformly retain one quota from packed worker candidate chunks."""

        usable = [chunk for chunk in chunks if chunk]
        lengths = [int(chunk[0].shape[0]) for chunk in usable]
        if any(
            any(int(field.shape[0]) != length for field in chunk)
            for chunk, length in zip(usable, lengths)
        ):
            raise RuntimeError("parallel traversal returned inconsistent recent samples")
        total = sum(lengths)
        count = min(self.per_iteration, total)
        if count <= 0:
            self._completed_iterations += 1
            return
        selected = self.rng.sample(range(total), count)
        ends: list[int] = []
        running = 0
        for length in lengths:
            running += length
            ends.append(running)
        output = [
            torch.empty((count, *field.shape[1:]), dtype=field.dtype)
            for field in usable[0]
        ]
        grouped: dict[int, tuple[list[int], list[int]]] = {}
        for output_row, global_row in enumerate(selected):
            chunk_index = bisect.bisect_right(ends, global_row)
            chunk_start = 0 if chunk_index == 0 else ends[chunk_index - 1]
            output_rows, source_rows = grouped.setdefault(chunk_index, ([], []))
            output_rows.append(output_row)
            source_rows.append(global_row - chunk_start)
        for chunk_index, (output_rows, source_rows) in grouped.items():
            destination_index = torch.tensor(output_rows, dtype=torch.long)
            source_index = torch.tensor(source_rows, dtype=torch.long)
            for destination, source in zip(output, usable[chunk_index]):
                destination.index_copy_(
                    0, destination_index, source.index_select(0, source_index)
                )
        self._commit_fields(output)

    def finish_scalar_iteration(self) -> None:
        """Commit candidates gathered by the non-parallel traversal path."""

        if self._pending.memory:
            self._pending._compact()
            assert isinstance(self._pending.memory, _PackedReservoirView)
            fields = [
                field[: len(self._pending)].clone()
                for field in self._pending.memory.fields
            ]
        else:
            fields = []
        self._commit_fields(fields)
        self._pending = ReservoirBuffer(self.per_iteration, self.rng)


def _legal_mask(legal: Iterable[int]) -> torch.Tensor:
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
    for action in legal:
        mask[int(action)] = 1.0
    return mask


def _pack_worker_buffers(buffers: Sequence["ReservoirBuffer"]) -> list[list[torch.Tensor]]:
    """Pack worker samples into a few tensors for efficient process transfer."""
    packed: list[list[torch.Tensor]] = []
    for buffer in buffers:
        if not buffer.memory:
            packed.append([])
            continue
        field_count = len(buffer.memory[0])
        packed.append(
            [
                torch.stack([item[field] for item in buffer.memory])
                for field in range(field_count)
            ]
        )
    return packed


def _parallel_traversal_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a deterministic group of roots in a spawned, CPU-only process."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work.
        pass

    env = payload["env_type"](**payload["env_kwargs"])
    config = dict(payload["trainer_config"])
    maximum_samples = max(
        1, len(payload["tasks"]) * int(config["max_nodes_per_traversal"])
    )
    config["advantage_capacity"] = max(
        int(config["advantage_capacity"]), maximum_samples
    )
    config["policy_capacity"] = max(
        int(config["policy_capacity"]), maximum_samples
    )
    # A traversal worker performs frozen advantage-network inference only.  In
    # particular it must not construct the three policy networks or any AdamW
    # optimizers.  Besides being unused, optimizer construction imports the
    # Torch compiler/SymPy stack in every spawned process and was enough to
    # exhaust a 16-GiB Windows host with six workers.
    worker = ThreePlayerNeuralCFR(
        env, device="cpu", _traversal_worker=True, **config
    )
    for network, state in zip(worker.advantage_nets, payload["advantage_states"]):
        network.load_state_dict(state)
        network.eval()
    worker.iteration = int(payload["iteration"])

    continuation_results: list[tuple[int, tuple[float, float, float]]] = []
    task_contexts = [
        {
            "root_index": int(root_index),
            "state": state,
            "traverser": int(traverser),
            "rng": random.Random(int(seed)),
            "derive_continuation": bool(derive_continuation),
        }
        for root_index, state, traverser, seed, derive_continuation in payload["tasks"]
    ]
    worker._run_batched_traversals(task_contexts)
    for context in task_contexts:
        if context["derive_continuation"]:
            worker.rng = context["rng"]
            state = context["state"]
            root_index = context["root_index"]
            previous_count = len(worker._continuation_stacks)
            worker._derive_continuation(state)
            if len(worker._continuation_stacks) > previous_count:
                continuation_results.append(
                    (int(root_index), worker._continuation_stacks[-1])
                )

    return {
        "advantage_samples": _pack_worker_buffers(worker.advantage_buffers),
        "policy_samples": _pack_worker_buffers(worker.policy_buffers),
        "nodes": worker._nodes_this_iteration,
        "rollouts": worker._rollouts_this_iteration,
        "regret_magnitudes": worker._regret_magnitudes,
        "strategy_weights": worker._strategy_weights,
        "policy_entropies": worker._policy_entropies,
        "raw_strategy_importances": worker._raw_strategy_importances,
        "strategy_cap_hits": worker._strategy_cap_hits,
        "depth_cutoffs_by_street": worker._depth_cutoffs_by_street,
        "node_cutoffs_by_street": worker._node_cutoffs_by_street,
        "advantage_samples_by_street": worker._advantage_samples_by_street,
        "policy_samples_by_street": worker._policy_samples_by_street,
        "continuation_hands": worker._continuation_hands_this_iteration,
        "continuation_results": continuation_results,
    }


class ThreePlayerNeuralCFR:
    """Three advantage nets plus three average-policy nets.

    Advantage networks stay alive between iterations: iteration ``t`` gathers
    data using the networks fitted through ``t-1``, fits on cumulative reservoir
    memories, and only then moves to ``t+1``.  This avoids the reset-before-use
    lifecycle bug in the original heads-up trainer.
    """

    def __init__(
        self,
        env,
        *,
        device: str | torch.device = "cpu",
        hidden: int = 128,
        blocks: int = 2,
        network_architecture: str = "residual_mlp",
        learning_rate: float = 3e-4,
        advantage_capacity: int = 25_000,
        policy_capacity: int = 25_000,
        recent_capacity: int = 0,
        recent_window_iterations: int = 100,
        recent_batch_fraction: float = 0.5,
        max_history: int = DEFAULT_MAX_HISTORY,
        max_nodes_per_traversal: int = 20_000,
        max_depth: int = 64,
        max_strategy_importance: float = 100.0,
        exploration: float = 0.01,
        reinitialize_advantage_each_iteration: bool = True,
        advantage_reinitialize_from_iteration: int = 1,
        advantage_fit_every: int = 1,
        include_tournament_features: bool = False,
        variable_stack_training: bool = False,
        tournament_total_chips: float | None = None,
        heads_up_root_fraction: float = 0.25,
        continuation_root_fraction: float = 0.25,
        minimum_live_stack: float | None = None,
        root_stack_concentration: float = 0.7,
        continuation_capacity: int = 2_048,
        seed: int = 42,
        _traversal_worker: bool = False,
    ):
        self.env = env
        self.device = torch.device(device)
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.network_architecture = str(network_architecture)
        if self.network_architecture not in NETWORK_ARCHITECTURES:
            raise ValueError(
                f"network_architecture must be one of {NETWORK_ARCHITECTURES}"
            )
        self.learning_rate = float(learning_rate)
        self.recent_capacity = int(recent_capacity)
        self.recent_window_iterations = int(recent_window_iterations)
        self.recent_batch_fraction = float(recent_batch_fraction)
        if self.recent_capacity < 0:
            raise ValueError("recent_capacity cannot be negative")
        if self.recent_window_iterations <= 0:
            raise ValueError("recent_window_iterations must be positive")
        if not 0.0 <= self.recent_batch_fraction <= 1.0:
            raise ValueError("recent_batch_fraction must be in [0, 1]")
        if self.recent_capacity and (
            self.recent_capacity % self.recent_window_iterations
        ):
            raise ValueError(
                "recent_capacity must be divisible by recent_window_iterations"
            )
        self.max_history = int(max_history)
        self.max_nodes_per_traversal = int(max_nodes_per_traversal)
        self.max_depth = int(max_depth)
        self.max_strategy_importance = float(max_strategy_importance)
        self.exploration = float(exploration)
        if self.max_history <= 0:
            raise ValueError("max_history must be positive")
        if self.max_nodes_per_traversal <= 0 or self.max_depth <= 0:
            raise ValueError("traversal node/depth limits must be positive")
        if self.max_strategy_importance <= 0:
            raise ValueError("max_strategy_importance must be positive")
        if not 0.0 <= self.exploration < 1.0:
            raise ValueError("exploration must be in [0, 1)")
        self.reinitialize_advantage_each_iteration = bool(
            reinitialize_advantage_each_iteration
        )
        self.advantage_reinitialize_from_iteration = int(
            advantage_reinitialize_from_iteration
        )
        if self.advantage_reinitialize_from_iteration <= 0:
            raise ValueError(
                "advantage_reinitialize_from_iteration must be positive"
            )
        self.advantage_fit_every = int(advantage_fit_every)
        if self.advantage_fit_every <= 0:
            raise ValueError("advantage_fit_every must be positive")
        self.include_tournament_features = bool(include_tournament_features)
        self.variable_stack_training = bool(variable_stack_training)
        self.tournament_total_chips = float(
            3.0 * self.env.stack_size
            if tournament_total_chips is None
            else tournament_total_chips
        )
        self.heads_up_root_fraction = float(heads_up_root_fraction)
        self.continuation_root_fraction = float(continuation_root_fraction)
        self.minimum_live_stack = float(
            self.env.sb if minimum_live_stack is None else minimum_live_stack
        )
        self.root_stack_concentration = float(root_stack_concentration)
        self.continuation_capacity = int(continuation_capacity)
        if self.tournament_total_chips <= 0:
            raise ValueError("tournament_total_chips must be positive")
        if not 0.0 <= self.heads_up_root_fraction <= 1.0:
            raise ValueError("heads_up_root_fraction must be in [0, 1]")
        if not 0.0 <= self.continuation_root_fraction <= 1.0:
            raise ValueError("continuation_root_fraction must be in [0, 1]")
        if self.minimum_live_stack <= 0:
            raise ValueError("minimum_live_stack must be positive")
        if self.root_stack_concentration <= 0:
            raise ValueError("root_stack_concentration must be positive")
        if self.continuation_capacity <= 0:
            raise ValueError("continuation_capacity must be positive")
        if self.tournament_total_chips + 1e-9 < 3.0 * self.minimum_live_stack:
            raise ValueError(
                "tournament_total_chips must fund three minimum live stacks"
            )
        self.seed = int(seed)
        self._traversal_worker = bool(_traversal_worker)

        self.rng = random.Random(self.seed)
        self.eval_rng = random.Random(self.seed + 10_000)
        self._sample_env = type(self.env)(
            stack_size=self.env.stack_size,
            sb=self.env.sb,
            bb=self.env.bb,
            seed=self.seed + 40_000,
        )
        torch.manual_seed(self.seed)
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        probe = self.env.new_hand(button=0)
        probe_legal = self.env.legal_actions(probe)
        self.input_dim = int(
            encode_information_state(
                probe,
                probe.to_act,
                probe_legal,
                self.env.stack_size,
                self.max_history,
                include_tournament_features=self.include_tournament_features,
                tournament_total_chips=self.tournament_total_chips,
            ).numel()
        )
        expected_input_dim = information_state_size(
            self.max_history,
            include_tournament_features=self.include_tournament_features,
        )
        if self.input_dim != expected_input_dim:
            raise RuntimeError("probe encoder width does not match declared schema")

        self.advantage_nets = [
            build_advantage_network(
                self.network_architecture, self.input_dim, self.hidden, self.blocks
            ).to(self.device)
            for _ in range(3)
        ]
        self.policy_nets = (
            []
            if self._traversal_worker
            else [
                build_policy_network(
                    self.network_architecture,
                    self.input_dim,
                    self.hidden,
                    self.blocks,
                ).to(self.device)
                for _ in range(3)
            ]
        )
        # Traversal/evaluation inference is the default lifecycle state. Fit
        # methods enter train mode only for their optimization phase and restore
        # eval mode afterward. This avoids per-node recursive ``eval()`` calls
        # without ever leaving attention dropout active during CFR collection.
        for network in self.advantage_nets + self.policy_nets:
            network.eval()
        self.advantage_optimizers = (
            []
            if self._traversal_worker
            else [
                torch.optim.AdamW(
                    net.parameters(),
                    lr=self.learning_rate,
                    weight_decay=1e-5,
                )
                for net in self.advantage_nets
            ]
        )
        self.policy_optimizers = (
            []
            if self._traversal_worker
            else [
                torch.optim.AdamW(
                    net.parameters(),
                    lr=self.learning_rate,
                    weight_decay=1e-5,
                )
                for net in self.policy_nets
            ]
        )

        self.advantage_buffers = [
            ReservoirBuffer(advantage_capacity, self.rng) for _ in range(3)
        ]
        self.policy_buffers = [
            ReservoirBuffer(policy_capacity, self.rng) for _ in range(3)
        ]
        # Recent memories use independent RNG streams. Their sampling and
        # retention therefore cannot perturb the lifetime-reservoir sequence.
        # Traversal workers only collect packed candidates; the parent process
        # owns the six large RAM windows.
        self.recent_advantage_buffers = (
            []
            if self._traversal_worker or not self.recent_capacity
            else [
                RecentWindowBuffer(
                    self.recent_capacity,
                    self.recent_window_iterations,
                    random.Random(self.seed + 110_000 + player),
                )
                for player in range(3)
            ]
        )
        self.recent_policy_buffers = (
            []
            if self._traversal_worker or not self.recent_capacity
            else [
                RecentWindowBuffer(
                    self.recent_capacity,
                    self.recent_window_iterations,
                    random.Random(self.seed + 120_000 + player),
                )
                for player in range(3)
            ]
        )

        self.iteration = 0
        self.last_fitted_iteration = 0
        self.can_resume_training = True
        self._position_cycle = 0
        self.metrics: list[dict[str, float]] = []
        self._nodes_this_traversal = 0
        self._rollouts_this_iteration = 0
        self._nodes_this_iteration = 0
        self._parallel_worker_transfer_seconds = 0.0
        self._reservoir_merge_seconds = 0.0
        self._regret_magnitudes: list[float] = []
        self._strategy_weights: list[float] = []
        self._policy_entropies: list[float] = []
        self._raw_strategy_importances: list[float] = []
        self._strategy_cap_hits = 0
        self._depth_cutoffs_by_street = [0, 0, 0, 0]
        self._node_cutoffs_by_street = [0, 0, 0, 0]
        self._advantage_samples_by_street = [0, 0, 0, 0]
        self._policy_samples_by_street = [0, 0, 0, 0]
        # These are *root stack* continuations, not full tournament CFR nodes.
        # A sampled self-play hand can seed a later hand, improving coverage of
        # realistic stack distributions while the utility remains per-hand chip
        # EV. See ``_remember_continuation`` and the module docstring caveat.
        self._continuation_stacks: list[tuple[float, float, float]] = []
        self._continuation_states_seen = 0
        self._continuation_hands_this_iteration = 0
        self._three_handed_roots_this_iteration = 0
        self._heads_up_roots_this_iteration = 0
        self._eliminated_traversals_skipped_this_iteration = 0

    def configure_recent_memory(
        self, capacity: int, window_iterations: int, batch_fraction: float
    ) -> None:
        """Configure or replace the six RAM-only recent-memory windows."""

        capacity = int(capacity)
        window_iterations = int(window_iterations)
        batch_fraction = float(batch_fraction)
        if capacity < 0:
            raise ValueError("recent_capacity cannot be negative")
        if window_iterations <= 0:
            raise ValueError("recent_window_iterations must be positive")
        if not 0.0 <= batch_fraction <= 1.0:
            raise ValueError("recent_batch_fraction must be in [0, 1]")
        if capacity and capacity % window_iterations:
            raise ValueError(
                "recent_capacity must be divisible by recent_window_iterations"
            )
        if (
            capacity == self.recent_capacity
            and window_iterations == self.recent_window_iterations
            and batch_fraction == self.recent_batch_fraction
            and (not capacity or self.recent_advantage_buffers)
        ):
            return
        self.recent_capacity = capacity
        self.recent_window_iterations = window_iterations
        self.recent_batch_fraction = batch_fraction
        if self._traversal_worker or not capacity:
            self.recent_advantage_buffers = []
            self.recent_policy_buffers = []
            return
        self.recent_advantage_buffers = [
            RecentWindowBuffer(
                capacity,
                window_iterations,
                random.Random(self.seed + 110_000 + player),
            )
            for player in range(3)
        ]
        self.recent_policy_buffers = [
            RecentWindowBuffer(
                capacity,
                window_iterations,
                random.Random(self.seed + 120_000 + player),
            )
            for player in range(3)
        ]
    # ------------------------------------------------------------------
    # Information states and strategies
    # ------------------------------------------------------------------
    def encode(self, state, player: int, legal: Sequence[int] | None = None) -> torch.Tensor:
        if legal is None:
            legal = self.env.legal_actions(state)
        return encode_information_state(
            state,
            player,
            legal,
            self.env.stack_size,
            self.max_history,
            include_tournament_features=self.include_tournament_features,
            tournament_total_chips=self.tournament_total_chips,
        )

    @staticmethod
    def regret_matching(advantages: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        positive = torch.clamp(advantages, min=0.0) * mask
        total = positive.sum()
        if float(total) <= 1e-12:
            # The Deep-CFR paper found that selecting the highest predicted
            # legal regret here substantially outperformed a uniform fallback.
            # Configured exploration is mixed in by the caller afterward, so
            # every legal action remains reachable during data collection.
            legal_values = advantages.masked_fill(mask <= 0, -torch.inf)
            strategy = torch.zeros_like(advantages)
            strategy[int(torch.argmax(legal_values))] = 1.0
            return strategy
        return positive / total

    @staticmethod
    def regret_matching_batch(
        advantages: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized, row-equivalent regret matching for traversal frontiers."""

        if advantages.ndim != 2 or masks.shape != advantages.shape:
            raise ValueError("advantages and masks must have matching [batch, actions] shapes")
        positive = torch.clamp(advantages, min=0.0) * masks
        totals = positive.sum(dim=1, keepdim=True)
        normalized = positive / totals.clamp(min=1e-12)
        legal_values = advantages.masked_fill(masks <= 0, -torch.inf)
        fallback = F.one_hot(
            legal_values.argmax(dim=1), num_classes=advantages.shape[1]
        ).to(dtype=advantages.dtype)
        return torch.where(totals > 1e-12, normalized, fallback)

    @torch.no_grad()
    def current_strategy(
        self, state, player: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.terminal or state.to_act is None:
            raise ValueError("current_strategy requires a nonterminal decision state")
        if player is None:
            player = int(state.to_act)
        if int(player) != int(state.to_act):
            raise ValueError("current_strategy requires the state's acting player")
        legal = self.env.legal_actions(state)
        if not legal:
            raise ValueError("a nonterminal decision state must have legal actions")
        x = self.encode(state, player, legal)
        mask = _legal_mask(legal)
        values = self.advantage_nets[player](x.to(self.device).unsqueeze(0))[0].cpu()
        probabilities = self.regret_matching(values, mask)
        # External-sampling opponents need persistent exploration so an early
        # approximation error cannot permanently remove whole response lines
        # from the sampled tree.  The previous inverse-square-root decay took a
        # configured 2% to 0.09% by iteration 500 and locked the strategy into
        # calling almost every bet.
        effective_exploration = self.exploration
        if effective_exploration > 0.0:
            uniform = mask / mask.sum().clamp(min=1.0)
            probabilities = (
                (1.0 - effective_exploration) * probabilities
                + effective_exploration * uniform
            )
        return x, probabilities, mask

    @torch.no_grad()
    def average_policy(self, state, player: int | None = None) -> torch.Tensor:
        if state.terminal or state.to_act is None:
            raise ValueError("average_policy requires a nonterminal decision state")
        if player is None:
            player = int(state.to_act)
        if int(player) != int(state.to_act):
            raise ValueError("average_policy requires the state's acting player")
        legal = self.env.legal_actions(state)
        x = self.encode(state, player, legal).to(self.device)
        mask = _legal_mask(legal).to(self.device)
        logits = self.policy_nets[player](x.unsqueeze(0))[0]
        return masked_softmax(logits, mask).cpu()

    @torch.no_grad()
    def average_policy_batch(
        self,
        states: Sequence[Any],
        *,
        policy_nets: Sequence[PolicyNetwork] | None = None,
        batch_size: int = 4096,
    ) -> list[torch.Tensor]:
        """Evaluate many decision states without one CUDA sync per state.

        States are grouped by their acting player because each absolute player
        owns a separate average-policy network.  ``policy_nets`` can be a frozen
        three-network snapshot, which makes the same API useful for strategy
        comparisons and historical-league analysis.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        networks = self.policy_nets if policy_nets is None else list(policy_nets)
        if len(networks) != 3:
            raise ValueError("policy_nets must contain exactly three networks")
        outputs: list[torch.Tensor | None] = [None] * len(states)
        grouped: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(3)
        ]
        for index, state in enumerate(states):
            if state.terminal or state.to_act is None:
                raise ValueError("every batch item must be a nonterminal decision state")
            player = int(state.to_act)
            legal = self.env.legal_actions(state)
            if not legal:
                raise ValueError("every batch item must contain a legal action")
            grouped[player].append(
                (index, self.encode(state, player, legal), _legal_mask(legal))
            )

        for player, items in enumerate(grouped):
            if not items:
                continue
            net = networks[player]
            net.eval()
            net_device = next(net.parameters()).device
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                xs = torch.stack([item[1] for item in chunk]).to(
                    net_device, non_blocking=True
                )
                masks = torch.stack([item[2] for item in chunk]).to(
                    net_device, non_blocking=True
                )
                probabilities = masked_softmax(net(xs), masks).cpu()
                for (index, _, _), probability in zip(chunk, probabilities):
                    outputs[index] = probability
        if any(output is None for output in outputs):
            raise RuntimeError("failed to evaluate one or more policy states")
        return [output for output in outputs if output is not None]

    @staticmethod
    def _draw_action(probabilities: torch.Tensor, rng: random.Random) -> int:
        r = rng.random()
        cumulative = 0.0
        last = int(torch.argmax(probabilities).item())
        for action, probability in enumerate(probabilities.tolist()):
            if probability <= 0.0:
                continue
            last = action
            cumulative += probability
            if r <= cumulative + 1e-12:
                return action
        return last

    # ------------------------------------------------------------------
    # Tournament root sampling and external-sampling traversal
    # ------------------------------------------------------------------
    @staticmethod
    def _state_alive(state) -> list[bool]:
        if hasattr(state, "alive"):
            values = list(state.alive)
        elif hasattr(state, "eliminated"):
            values = [not bool(value) for value in state.eliminated]
        else:
            values = [float(value) > 1e-9 for value in state.initial_stacks]
        if len(values) != 3:
            raise ValueError("tournament status must contain exactly three seats")
        return [bool(value) for value in values]

    def _sample_tournament_stacks(
        self, live_players: int | None = None
    ) -> tuple[float, float, float]:
        """Sample a chip-conserving synthetic tournament hand root.

        ``live_players`` is primarily useful for deterministic tests and
        analysis. Ordinarily it is selected from ``heads_up_root_fraction``.
        Positive live stacks are bounded below by ``minimum_live_stack``;
        eliminated seats are represented by exact zeros.
        """

        if live_players is None:
            live_players = 2 if self.rng.random() < self.heads_up_root_fraction else 3
        if live_players not in (2, 3):
            raise ValueError("live_players must be 2 or 3")
        minimum_total = live_players * self.minimum_live_stack
        if minimum_total > self.tournament_total_chips + 1e-9:
            raise ValueError("configured total cannot fund requested live seats")

        live_seats = sorted(self.rng.sample(range(3), live_players))
        weights = [
            self.rng.gammavariate(self.root_stack_concentration, 1.0)
            for _ in live_seats
        ]
        weight_sum = sum(weights)
        distributable = max(0.0, self.tournament_total_chips - minimum_total)
        stacks = [0.0, 0.0, 0.0]
        for seat, weight in zip(live_seats, weights):
            stacks[seat] = self.minimum_live_stack + distributable * weight / weight_sum
        # Make conservation exact despite floating-point summation. The last
        # live seat always has at least the configured minimum before this tiny
        # roundoff correction.
        stacks[live_seats[-1]] += self.tournament_total_chips - sum(stacks)
        return tuple(stacks)

    def _remember_continuation(self, stacks: Sequence[float]) -> bool:
        """Reservoir-sample a terminal hand's stacks for later root sampling."""

        if len(stacks) != 3:
            raise ValueError("continuation stacks must contain exactly three values")
        values = tuple(0.0 if abs(float(v)) <= 1e-9 else float(v) for v in stacks)
        if any(value < 0.0 for value in values):
            raise ValueError("continuation stacks cannot be negative")
        if abs(sum(values) - self.tournament_total_chips) > 1e-6:
            return False
        # Once only one seat remains the tournament is over; it is not a poker
        # hand root and must never be passed to ``env.new_hand``.
        if sum(value > 1e-9 for value in values) < 2:
            return False
        self._continuation_states_seen += 1
        if len(self._continuation_stacks) < self.continuation_capacity:
            self._continuation_stacks.append(values)
        else:
            index = self.rng.randrange(self._continuation_states_seen)
            if index < self.continuation_capacity:
                self._continuation_stacks[index] = values
        return True

    def _root_stacks(self) -> tuple[float, float, float] | None:
        if not self.variable_stack_training:
            return None
        if (
            self._continuation_stacks
            and self.rng.random() < self.continuation_root_fraction
        ):
            return self._continuation_stacks[
                self.rng.randrange(len(self._continuation_stacks))
            ]
        return self._sample_tournament_stacks()

    @staticmethod
    def _button_for_live_role(
        traverser: int, alive: Sequence[bool], role_offset: int
    ) -> int:
        live_seats = [seat for seat, value in enumerate(alive) if value]
        if traverser not in live_seats:
            raise ValueError("an eliminated traverser has no table position")
        traverser_index = live_seats.index(traverser)
        return live_seats[(traverser_index - role_offset) % len(live_seats)]

    def _derive_continuation(self, root_state) -> None:
        """Play one sampled path and retain its resulting tournament stacks."""

        state = root_state
        steps = 0
        while not state.terminal:
            _, probabilities, _ = self.current_strategy(state)
            action = self._draw_action(probabilities, self.rng)
            state = self.env.step(state, action)
            steps += 1
            if steps > 256:
                raise RuntimeError(
                    "continuation rollout exceeded 256 actions; engine did not terminate"
                )
        self._continuation_hands_this_iteration += 1
        self._remember_continuation(state.stacks)

    def _rollout_value(self, state, traverser: int) -> float:
        self._rollouts_this_iteration += 1
        steps = 0
        while not state.terminal:
            _, probabilities, _ = self.current_strategy(state)
            action = self._draw_action(probabilities, self.rng)
            state = self.env.step(state, action)
            steps += 1
            if steps > 256:
                raise RuntimeError("rollout exceeded 256 actions; engine did not terminate")
        return float(state.payoffs[traverser]) / float(self.env.bb)

    def _traverse(
        self,
        state,
        traverser: int,
        reach: list[float],
        depth: int,
    ) -> float:
        if depth == 0 and not self._state_alive(state)[traverser]:
            # Training normally filters this before dealing. Keeping the guard
            # here makes direct/custom traversal drivers safe as well.
            return 0.0
        if state.terminal:
            return float(state.payoffs[traverser]) / float(self.env.bb)

        if depth >= self.max_depth:
            self._depth_cutoffs_by_street[int(state.street)] += 1
            return self._rollout_value(state, traverser)
        if self._nodes_this_traversal >= self.max_nodes_per_traversal:
            self._node_cutoffs_by_street[int(state.street)] += 1
            return self._rollout_value(state, traverser)

        self._nodes_this_traversal += 1
        self._nodes_this_iteration += 1
        player = int(state.to_act)
        legal = self.env.legal_actions(state)
        x, probabilities, mask = self.current_strategy(state, player)

        if player == traverser:
            action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            # Shuffle only the evaluation order so budget fallbacks do not always
            # favor low-numbered action labels.
            action_order = list(legal)
            self.rng.shuffle(action_order)
            for action in action_order:
                next_reach = reach.copy()
                next_reach[player] *= float(probabilities[action])
                action_values[action] = self._traverse(
                    self.env.step(state, action), traverser, next_reach, depth + 1
                )
            node_value = float((action_values * probabilities).sum())
            regrets = (action_values - node_value) * mask
            self._regret_magnitudes.append(
                float(regrets.abs().sum() / mask.sum().clamp(min=1.0))
            )
            sample = (
                x.cpu().to(torch.float16),
                regrets.cpu(),
                mask.cpu(),
                torch.tensor(float(self.iteration)),
            )
            self.advantage_buffers[player].add(sample)
            if self.recent_advantage_buffers:
                self.recent_advantage_buffers[player].consider(sample)
            self._advantage_samples_by_street[int(state.street)] += 1
            return node_value

        # For three players, an external-sampling traversal samples two players.
        # A policy sample for one of them is visited in proportion to both sampled
        # players' reaches. Divide by the third player's reach so the expected
        # weight is proportional to the acting player's own reach. Capping keeps
        # variance practical but intentionally biases the average; metrics expose
        # the cap-hit rate and raw maximum so that tradeoff is visible.
        third_players = [p for p in range(3) if p not in (traverser, player)]
        raw_correction = 1.0
        if third_players:
            raw_correction = 1.0 / max(reach[third_players[0]], 1e-12)
        self._raw_strategy_importances.append(raw_correction)
        if raw_correction > self.max_strategy_importance:
            self._strategy_cap_hits += 1
        correction = min(raw_correction, self.max_strategy_importance)
        policy_weight = float(self.iteration) * correction
        self._strategy_weights.append(policy_weight)
        positive_probabilities = probabilities[probabilities > 0]
        self._policy_entropies.append(
            float(-(positive_probabilities * positive_probabilities.log()).sum())
        )
        sample = (
            x.cpu().to(torch.float16),
            probabilities.cpu(),
            mask.cpu(),
            torch.tensor(policy_weight),
        )
        self.policy_buffers[player].add(sample)
        if self.recent_policy_buffers:
            self.recent_policy_buffers[player].consider(sample)
        self._policy_samples_by_street[int(state.street)] += 1

        action = self._draw_action(probabilities, self.rng)
        next_reach = reach.copy()
        next_reach[player] *= float(probabilities[action])
        return self._traverse(
            self.env.step(state, action), traverser, next_reach, depth + 1
        )

    @torch.inference_mode()
    def _batched_current_strategies(
        self,
        requests: Sequence[tuple[Any, int]],
        *,
        use_vectorized: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Evaluate independent traversal frontiers in player-grouped batches."""
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [
            None
        ] * len(requests)
        grouped: list[list[tuple[int, torch.Tensor, Sequence[int]]]] = [
            [] for _ in range(3)
        ]
        for index, (state, player) in enumerate(requests):
            legal = self.env.legal_actions(state)
            x = self.encode(state, player, legal)
            grouped[player].append((index, x, legal))
        effective_exploration = self.exploration
        for player, items in enumerate(grouped):
            if not items:
                continue
            xs = torch.stack([item[1] for item in items]).to(self.device)
            if use_vectorized:
                masks = torch.zeros(
                    (len(items), NUM_ACTIONS), dtype=torch.float32
                )
                for row, (_, _, legal) in enumerate(items):
                    masks[row, list(legal)] = 1.0
            else:
                masks = torch.stack([_legal_mask(item[2]) for item in items])
            values = self.advantage_nets[player](xs).cpu()
            if use_vectorized:
                probabilities_batch = self.regret_matching_batch(values, masks)
                if effective_exploration > 0.0:
                    uniform = masks / masks.sum(dim=1, keepdim=True).clamp(min=1.0)
                    probabilities_batch = (
                        (1.0 - effective_exploration) * probabilities_batch
                        + effective_exploration * uniform
                    )
                for (index, x, _), probabilities, mask in zip(
                    items, probabilities_batch, masks
                ):
                    outputs[index] = (x, probabilities, mask)
            else:
                for (index, x, _), row, mask in zip(items, values, masks):
                    probabilities = self.regret_matching(row, mask)
                    if effective_exploration > 0.0:
                        uniform = mask / mask.sum().clamp(min=1.0)
                        probabilities = (
                            (1.0 - effective_exploration) * probabilities
                            + effective_exploration * uniform
                        )
                    outputs[index] = (x, probabilities, mask)
        if any(output is None for output in outputs):
            raise RuntimeError("failed to evaluate a traversal frontier")
        return [output for output in outputs if output is not None]

    def _rollout_coroutine(self, state, traverser: int, rng: random.Random):
        self._rollouts_this_iteration += 1
        steps = 0
        while not state.terminal:
            player = int(state.to_act)
            _, probabilities, _ = yield (state, player)
            action = self._draw_action(probabilities, rng)
            state = self.env.step(state, action)
            steps += 1
            if steps > 256:
                raise RuntimeError("rollout exceeded 256 actions; engine did not terminate")
        return float(state.payoffs[traverser]) / float(self.env.bb)

    def _traverse_coroutine(
        self,
        state,
        traverser: int,
        reach: list[float],
        depth: int,
        rng: random.Random,
        node_counter: list[int],
    ):
        """Generator form of traversal, yielding states for batched inference."""
        if depth == 0 and not self._state_alive(state)[traverser]:
            return 0.0
        if state.terminal:
            return float(state.payoffs[traverser]) / float(self.env.bb)
        if depth >= self.max_depth:
            self._depth_cutoffs_by_street[int(state.street)] += 1
            return (yield from self._rollout_coroutine(state, traverser, rng))
        if node_counter[0] >= self.max_nodes_per_traversal:
            self._node_cutoffs_by_street[int(state.street)] += 1
            return (yield from self._rollout_coroutine(state, traverser, rng))

        node_counter[0] += 1
        self._nodes_this_iteration += 1
        player = int(state.to_act)
        legal = self.env.legal_actions(state)
        x, probabilities, mask = yield (state, player)

        if player == traverser:
            action_values = torch.zeros(NUM_ACTIONS, dtype=torch.float32)
            action_order = list(legal)
            rng.shuffle(action_order)
            for action in action_order:
                next_reach = reach.copy()
                next_reach[player] *= float(probabilities[action])
                action_values[action] = yield from self._traverse_coroutine(
                    self.env.step(state, action),
                    traverser,
                    next_reach,
                    depth + 1,
                    rng,
                    node_counter,
                )
            node_value = float((action_values * probabilities).sum())
            regrets = (action_values - node_value) * mask
            self._regret_magnitudes.append(
                float(regrets.abs().sum() / mask.sum().clamp(min=1.0))
            )
            sample = (
                x.cpu().to(torch.float16),
                regrets.cpu(),
                mask.cpu(),
                torch.tensor(float(self.iteration)),
            )
            self.advantage_buffers[player].add(sample)
            if self.recent_advantage_buffers:
                self.recent_advantage_buffers[player].consider(sample)
            self._advantage_samples_by_street[int(state.street)] += 1
            return node_value

        third_players = [p for p in range(3) if p not in (traverser, player)]
        raw_correction = 1.0
        if third_players:
            raw_correction = 1.0 / max(reach[third_players[0]], 1e-12)
        self._raw_strategy_importances.append(raw_correction)
        if raw_correction > self.max_strategy_importance:
            self._strategy_cap_hits += 1
        correction = min(raw_correction, self.max_strategy_importance)
        policy_weight = float(self.iteration) * correction
        self._strategy_weights.append(policy_weight)
        positive_probabilities = probabilities[probabilities > 0]
        self._policy_entropies.append(
            float(-(positive_probabilities * positive_probabilities.log()).sum())
        )
        sample = (
            x.cpu().to(torch.float16),
            probabilities.cpu(),
            mask.cpu(),
            torch.tensor(policy_weight),
        )
        self.policy_buffers[player].add(sample)
        if self.recent_policy_buffers:
            self.recent_policy_buffers[player].consider(sample)
        self._policy_samples_by_street[int(state.street)] += 1
        action = self._draw_action(probabilities, rng)
        next_reach = reach.copy()
        next_reach[player] *= float(probabilities[action])
        return (
            yield from self._traverse_coroutine(
                self.env.step(state, action),
                traverser,
                next_reach,
                depth + 1,
                rng,
                node_counter,
            )
        )

    def _run_batched_traversals(self, contexts: Sequence[dict[str, Any]]) -> None:
        """Cooperatively advance roots and batch every ready neural decision."""
        active: list[tuple[Any, tuple[Any, int]]] = []
        for context in contexts:
            generator = self._traverse_coroutine(
                context["state"],
                context["traverser"],
                [1.0, 1.0, 1.0],
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
        """Return the CPU worker configuration for frozen traversal inference."""
        return {
            "hidden": self.hidden,
            "blocks": self.blocks,
            "network_architecture": self.network_architecture,
            "learning_rate": self.learning_rate,
            "advantage_capacity": self.advantage_buffers[0].capacity,
            "policy_capacity": self.policy_buffers[0].capacity,
            "max_history": self.max_history,
            "max_nodes_per_traversal": self.max_nodes_per_traversal,
            "max_depth": self.max_depth,
            "max_strategy_importance": self.max_strategy_importance,
            "exploration": self.exploration,
            "reinitialize_advantage_each_iteration": (
                self.reinitialize_advantage_each_iteration
            ),
            "advantage_reinitialize_from_iteration": (
                self.advantage_reinitialize_from_iteration
            ),
            "advantage_fit_every": self.advantage_fit_every,
            "include_tournament_features": self.include_tournament_features,
            "variable_stack_training": self.variable_stack_training,
            "tournament_total_chips": self.tournament_total_chips,
            "heads_up_root_fraction": self.heads_up_root_fraction,
            "continuation_root_fraction": self.continuation_root_fraction,
            "minimum_live_stack": self.minimum_live_stack,
            "root_stack_concentration": self.root_stack_concentration,
            "continuation_capacity": self.continuation_capacity,
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
            for index in range(length):
                # Decide reservoir retention before cloning a 742-value
                # observation. Rejected candidates must not consume memory
                # bandwidth merely to preserve the transfer tensor's lifetime.
                buffer.add_packed_row(fields, index)

    def _collect_parallel_traversals(
        self, traversals_per_player: int, traversal_workers: int
    ) -> int:
        """Collect roots in spawned CPU processes and merge in the main process."""
        parallel_started = time.perf_counter()
        tasks: list[tuple[int, Any, int, int, bool]] = []
        root_index = 0
        for traverser in range(3):
            for traversal_index in range(traversals_per_player):
                stacks = self._root_stacks()
                if stacks is None:
                    alive = [True, True, True]
                else:
                    alive = [value > 1e-9 for value in stacks]
                    if not alive[traverser]:
                        self._eliminated_traversals_skipped_this_iteration += 1
                        root_index += 1
                        continue
                role_offset = (
                    self._position_cycle + traversal_index
                ) % sum(alive)
                button = self._button_for_live_role(traverser, alive, role_offset)
                state = self.env.new_hand(button=button, stacks=stacks)
                if sum(alive) == 2:
                    self._heads_up_roots_this_iteration += 1
                else:
                    self._three_handed_roots_this_iteration += 1
                tasks.append(
                    (
                        root_index,
                        state,
                        traverser,
                        self.rng.getrandbits(63),
                        bool(
                            self.variable_stack_training
                            and self.continuation_root_fraction > 0
                        ),
                    )
                )
                root_index += 1

        if not tasks:
            return 0
        worker_count = min(int(traversal_workers), len(tasks))
        chunks = [tasks[index::worker_count] for index in range(worker_count)]
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
                "stack_size": self.env.stack_size,
                "sb": self.env.sb,
                "bb": self.env.bb,
                "seed": self.seed + 70_000 + self.iteration,
            },
            "trainer_config": self._parallel_trainer_config(),
            "advantage_states": advantage_states,
            "iteration": self.iteration,
        }
        payloads = [{**common, "tasks": chunk} for chunk in chunks]

        context = mp.get_context("spawn")
        pool = context.Pool(processes=worker_count)
        try:
            results = pool.map(_parallel_traversal_worker, payloads, chunksize=1)
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
            pool.close()
            pool.join()

        self._parallel_worker_transfer_seconds = (
            time.perf_counter() - parallel_started
        )
        merge_started = time.perf_counter()
        continuation_results: list[tuple[int, tuple[float, float, float]]] = []
        for result in results:
            self._merge_packed_samples(
                result["advantage_samples"], self.advantage_buffers
            )
            self._merge_packed_samples(result["policy_samples"], self.policy_buffers)
            self._nodes_this_iteration += int(result["nodes"])
            self._rollouts_this_iteration += int(result["rollouts"])
            self._regret_magnitudes.extend(result["regret_magnitudes"])
            self._strategy_weights.extend(result["strategy_weights"])
            self._policy_entropies.extend(result["policy_entropies"])
            self._raw_strategy_importances.extend(
                result["raw_strategy_importances"]
            )
            self._strategy_cap_hits += int(result["strategy_cap_hits"])
            for street in range(4):
                self._depth_cutoffs_by_street[street] += int(
                    result["depth_cutoffs_by_street"][street]
                )
                self._node_cutoffs_by_street[street] += int(
                    result["node_cutoffs_by_street"][street]
                )
                self._advantage_samples_by_street[street] += int(
                    result["advantage_samples_by_street"][street]
                )
                self._policy_samples_by_street[street] += int(
                    result["policy_samples_by_street"][street]
                )
            self._continuation_hands_this_iteration += int(
                result["continuation_hands"]
            )
            continuation_results.extend(result["continuation_results"])
        if self.recent_advantage_buffers:
            for player, buffer in enumerate(self.recent_advantage_buffers):
                buffer.add_packed_iteration(
                    [result["advantage_samples"][player] for result in results]
                )
            for player, buffer in enumerate(self.recent_policy_buffers):
                buffer.add_packed_iteration(
                    [result["policy_samples"][player] for result in results]
                )
        for _, stacks in sorted(continuation_results, key=lambda item: item[0]):
            self._remember_continuation(stacks)
        self._reservoir_merge_seconds = time.perf_counter() - merge_started
        return worker_count

    # ------------------------------------------------------------------
    # Network fitting
    # ------------------------------------------------------------------
    @staticmethod
    def _buffer_weight_mean(buffer: ReservoirBuffer) -> float:
        if not buffer.memory:
            return 1.0
        if isinstance(buffer.memory, _PackedReservoirView):
            # Accumulate in float64 to match the previous Python-float sum while
            # reducing 150,000 row/tuple lookups to one contiguous tensor pass.
            return max(
                1e-8,
                float(
                    buffer.memory.fields[3][: len(buffer)].mean(dtype=torch.float64).item()
                ),
            )
        return max(
            1e-8,
            sum(float(item[3]) for item in buffer.memory) / len(buffer.memory),
        )

    @staticmethod
    def _scaled_weights(weights: torch.Tensor, fixed_scale: float) -> torch.Tensor:
        # A fixed buffer-wide scale preserves the weighted objective. Dividing by
        # each minibatch's random mean would bias the stochastic gradient.
        return weights.clamp(min=1e-8) / max(float(fixed_scale), 1e-8)

    def _mixed_weight_mean(
        self, historical: ReservoirBuffer, recent: ReservoirBuffer | None
    ) -> float:
        if recent is None or not recent.memory or self.recent_batch_fraction <= 0:
            return self._buffer_weight_mean(historical)
        fraction = self.recent_batch_fraction
        return (
            (1.0 - fraction) * self._buffer_weight_mean(historical)
            + fraction * self._buffer_weight_mean(recent)
        )

    def _fit_field_batches(
        self,
        historical: ReservoirBuffer,
        recent: ReservoirBuffer | None,
        size: int,
        steps: int,
        *,
        shuffled_historical: bool,
    ) -> Iterable[tuple[torch.Tensor, ...]]:
        """Yield fixed-size batches mixed from lifetime and recent memories."""

        if recent is None or not recent.memory or self.recent_batch_fraction <= 0:
            if self.network_architecture == "deep_cfr_branch_v2":
                yield from historical.street_balanced_field_batches(size, steps)
            elif shuffled_historical:
                yield from historical.shuffled_field_batches(size, steps)
            else:
                for _ in range(steps):
                    yield historical.sample_fields(size)
            return

        recent_size = min(
            int(size) - 1,
            max(1, int(round(int(size) * self.recent_batch_fraction))),
        )
        historical_size = int(size) - recent_size
        if self.network_architecture == "deep_cfr_branch_v2":
            historical_batches = historical.street_balanced_field_batches(
                historical_size, steps
            )
            recent_batches = recent.street_balanced_field_batches(recent_size, steps)
        else:
            historical_batches = (
                historical.shuffled_field_batches(historical_size, steps)
                if shuffled_historical
                else (
                    historical.sample_fields(historical_size) for _ in range(steps)
                )
            )
            recent_batches = recent.shuffled_field_batches(recent_size, steps)
        for historical_fields, recent_fields in zip(
            historical_batches, recent_batches
        ):
            yield tuple(
                torch.cat((old, new), dim=0)
                for old, new in zip(historical_fields, recent_fields)
            )

    def _fit_advantage(self, player: int, steps: int, batch_size: int) -> float:
        if not self.advantage_buffers[player].memory or steps <= 0:
            return float("nan")
        reinitialize = self._should_reinitialize_advantage()
        if reinitialize:
            # Strict Deep-CFR ordering: collect with the previous iteration's
            # fitted network, then refit a fresh approximator on cumulative
            # memory. The freshly fitted network is used on the next iteration.
            self.advantage_nets[player] = build_advantage_network(
                self.network_architecture, self.input_dim, self.hidden, self.blocks
            ).to(self.device)
            self.advantage_optimizers[player] = torch.optim.AdamW(
                self.advantage_nets[player].parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5,
            )
        net = self.advantage_nets[player]
        optimizer = self.advantage_optimizers[player]
        recent = (
            self.recent_advantage_buffers[player]
            if self.recent_advantage_buffers
            else None
        )
        weight_scale = self._mixed_weight_mean(
            self.advantage_buffers[player], recent
        )
        if reinitialize:
            # At least one complete pass is required after throwing away the old
            # approximator; the caller's step count is treated as a minimum.
            steps = max(
                steps,
                math.ceil(len(self.advantage_buffers[player]) / max(1, batch_size)),
            )
        net.train()
        loss_sum = torch.zeros((), device=self.device)
        batches = self._fit_field_batches(
            self.advantage_buffers[player],
            recent,
            batch_size,
            steps,
            shuffled_historical=reinitialize,
        )
        for xs, targets, masks, weights in batches:
            xs_t = xs.to(self.device, dtype=torch.float32)
            targets_t = targets.to(self.device)
            masks_t = masks.to(self.device)
            # Counterfactual regrets can differ by hundreds of big blinds
            # between tournament stack states.  Raw MSE consequently let a few
            # large negative fold regrets swamp many smaller positive ones.
            # Dividing every legal action at one information set by the same
            # positive value preserves regret-matching probabilities exactly,
            # while making the approximation learn action preference rather
            # than pot size.
            target_scale = (targets_t.abs() * masks_t).amax(
                dim=1, keepdim=True
            ).clamp(min=1.0)
            targets_t = targets_t / target_scale
            weights_t = self._scaled_weights(
                weights.to(self.device), weight_scale
            )
            prediction = net(xs_t)
            per_sample = (((prediction - targets_t) * masks_t) ** 2).sum(dim=1)
            loss = (per_sample * weights_t).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach()
        net.eval()
        return float((loss_sum / steps).item())

    def _should_fit_advantage(self, iteration: int | None = None) -> bool:
        """Fit through the bootstrap window, then at periodic boundaries."""
        value = self.iteration if iteration is None else int(iteration)
        return value > 0 and (
            value <= self.advantage_fit_every
            or value % self.advantage_fit_every == 0
        )

    def _should_reinitialize_advantage(
        self, iteration: int | None = None
    ) -> bool:
        """Apply fresh Deep-CFR refits only after the configured bootstrap."""
        value = self.iteration if iteration is None else int(iteration)
        return (
            self.reinitialize_advantage_each_iteration
            and value >= self.advantage_reinitialize_from_iteration
        )

    def _fit_policy(self, player: int, steps: int, batch_size: int) -> float:
        if not self.policy_buffers[player].memory or steps <= 0:
            return float("nan")
        net = self.policy_nets[player]
        optimizer = self.policy_optimizers[player]
        recent = (
            self.recent_policy_buffers[player]
            if self.recent_policy_buffers
            else None
        )
        weight_scale = self._mixed_weight_mean(self.policy_buffers[player], recent)
        net.train()
        loss_sum = torch.zeros((), device=self.device)
        batches = self._fit_field_batches(
            self.policy_buffers[player],
            recent,
            batch_size,
            steps,
            shuffled_historical=True,
        )
        for xs, targets, masks, weights in batches:
            xs_t = xs.to(self.device, dtype=torch.float32)
            targets_t = targets.to(self.device)
            masks_t = masks.to(self.device)
            weights_t = self._scaled_weights(
                weights.to(self.device), weight_scale
            )
            logits = net(xs_t).masked_fill(masks_t <= 0, -1e9)
            log_probabilities = F.log_softmax(logits, dim=1)
            per_sample = -(targets_t * log_probabilities).sum(dim=1)
            loss = (per_sample * weights_t).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.detach()
        net.eval()
        return float((loss_sum / steps).item())

    def train_iteration(
        self,
        *,
        traversals_per_player: int = 2,
        advantage_steps: int = 32,
        policy_steps: int = 8,
        batch_size: int = 128,
        traversal_workers: int = 1,
    ) -> dict[str, float]:
        """Collect one CFR iteration, periodically fit advantages, and fit policy."""
        if not self.can_resume_training:
            raise RuntimeError(
                "this light checkpoint has no CFR reservoirs and is inference-only"
            )
        if (
            traversals_per_player <= 0
            or advantage_steps <= 0
            or batch_size <= 0
            or traversal_workers <= 0
        ):
            raise ValueError(
                "traversals, advantage steps, batch size, and traversal workers "
                "must be positive"
            )
        if policy_steps < 0:
            raise ValueError("policy_steps cannot be negative")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        self.iteration += 1
        self._nodes_this_iteration = 0
        self._rollouts_this_iteration = 0
        self._parallel_worker_transfer_seconds = 0.0
        self._reservoir_merge_seconds = 0.0
        self._regret_magnitudes = []
        self._strategy_weights = []
        self._policy_entropies = []
        self._raw_strategy_importances = []
        self._strategy_cap_hits = 0
        self._depth_cutoffs_by_street = [0, 0, 0, 0]
        self._node_cutoffs_by_street = [0, 0, 0, 0]
        self._advantage_samples_by_street = [0, 0, 0, 0]
        self._policy_samples_by_street = [0, 0, 0, 0]
        self._continuation_hands_this_iteration = 0
        self._three_handed_roots_this_iteration = 0
        self._heads_up_roots_this_iteration = 0
        self._eliminated_traversals_skipped_this_iteration = 0

        workers_used = 1
        if traversal_workers > 1:
            workers_used = self._collect_parallel_traversals(
                traversals_per_player, traversal_workers
            )
        else:
            for traverser in range(3):
                for traversal_index in range(traversals_per_player):
                    stacks = self._root_stacks()
                    if stacks is None:
                        alive = [True, True, True]
                    else:
                        alive = [value > 1e-9 for value in stacks]
                        if not alive[traverser]:
                            # A busted seat has no information set or utility in
                            # this hand. Do not deal it cards or create samples for
                            # its player-specific networks.
                            self._eliminated_traversals_skipped_this_iteration += 1
                            continue

                    # Decouple position from traverser identity. Three-handed
                    # roots cover BTN/SB/BB; heads-up roots cover BTN(SB)/BB and
                    # correctly skip the eliminated physical seat.
                    role_offset = (
                        self._position_cycle + traversal_index
                    ) % sum(alive)
                    button = self._button_for_live_role(traverser, alive, role_offset)
                    state = self.env.new_hand(button=button, stacks=stacks)
                    if sum(alive) == 2:
                        self._heads_up_roots_this_iteration += 1
                    else:
                        self._three_handed_roots_this_iteration += 1
                    self._nodes_this_traversal = 0
                    self._traverse(state, traverser, [1.0, 1.0, 1.0], 0)
                    if (
                        self.variable_stack_training
                        and self.continuation_root_fraction > 0
                    ):
                        self._derive_continuation(state)
            if self.recent_advantage_buffers:
                for buffer in self.recent_advantage_buffers:
                    buffer.finish_scalar_iteration()
                for buffer in self.recent_policy_buffers:
                    buffer.finish_scalar_iteration()
        self._position_cycle = (
            self._position_cycle + traversals_per_player
        ) % 3

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        traversal_finished = time.perf_counter()

        result: dict[str, float] = {
            "iteration": float(self.iteration),
            "traversal_workers": float(workers_used),
            "nodes": float(self._nodes_this_iteration),
            "rollouts": float(self._rollouts_this_iteration),
            "parallel_worker_transfer_seconds": float(
                self._parallel_worker_transfer_seconds
            ),
            "reservoir_merge_seconds": float(self._reservoir_merge_seconds),
            "three_handed_roots": float(self._three_handed_roots_this_iteration),
            "heads_up_roots": float(self._heads_up_roots_this_iteration),
            "eliminated_traversals_skipped": float(
                self._eliminated_traversals_skipped_this_iteration
            ),
            "continuation_hands": float(self._continuation_hands_this_iteration),
            "continuation_roots_stored": float(len(self._continuation_stacks)),
            "mean_abs_regret": (
                sum(self._regret_magnitudes) / len(self._regret_magnitudes)
                if self._regret_magnitudes
                else float("nan")
            ),
        }
        if self._strategy_weights:
            weight_sum = sum(self._strategy_weights)
            weight_square_sum = sum(w * w for w in self._strategy_weights)
            result["strategy_weight_mean"] = weight_sum / len(self._strategy_weights)
            result["strategy_weight_max"] = max(self._strategy_weights)
            result["strategy_weight_ess"] = (
                weight_sum * weight_sum / max(weight_square_sum, 1e-12)
            )
            result["strategy_weight_ess_fraction"] = (
                result["strategy_weight_ess"] / len(self._strategy_weights)
            )
        else:
            result["strategy_weight_mean"] = float("nan")
            result["strategy_weight_max"] = float("nan")
            result["strategy_weight_ess"] = 0.0
            result["strategy_weight_ess_fraction"] = 0.0
        result["raw_importance_max"] = (
            max(self._raw_strategy_importances)
            if self._raw_strategy_importances
            else float("nan")
        )
        result["importance_cap_fraction"] = (
            self._strategy_cap_hits / len(self._raw_strategy_importances)
            if self._raw_strategy_importances
            else 0.0
        )
        street_names = ("preflop", "flop", "turn", "river")
        for street, name in enumerate(street_names):
            result[f"depth_cutoffs_{name}"] = float(
                self._depth_cutoffs_by_street[street]
            )
            result[f"node_cutoffs_{name}"] = float(
                self._node_cutoffs_by_street[street]
            )
            result[f"advantage_samples_{name}"] = float(
                self._advantage_samples_by_street[street]
            )
            result[f"policy_samples_{name}"] = float(
                self._policy_samples_by_street[street]
            )
        result["exploration"] = self.exploration
        result["mean_policy_entropy"] = (
            sum(self._policy_entropies) / len(self._policy_entropies)
            if self._policy_entropies
            else float("nan")
        )
        advantage_fit_started = time.perf_counter()
        # A notebook interrupt can leave its thread-local grad mode disabled.
        # Training must be self-contained rather than inheriting that ambient
        # state.  Explicitly leave inference mode and enable autograd for every
        # network construction, forward pass, and backward pass.
        fit_advantage = self._should_fit_advantage()
        result["advantage_fit_performed"] = float(fit_advantage)
        result["advantage_fit_every"] = float(self.advantage_fit_every)
        with torch.inference_mode(False), torch.enable_grad():
            for player in range(3):
                result[f"adv_loss_p{player}"] = (
                    self._fit_advantage(player, advantage_steps, batch_size)
                    if fit_advantage
                    else float("nan")
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        advantage_fit_finished = time.perf_counter()

        with torch.inference_mode(False), torch.enable_grad():
            for player in range(3):
                result[f"policy_loss_p{player}"] = self._fit_policy(
                    player, policy_steps, batch_size
                )
                result[f"adv_buffer_p{player}"] = float(len(self.advantage_buffers[player]))
                result[f"policy_buffer_p{player}"] = float(len(self.policy_buffers[player]))
                result[f"recent_adv_buffer_p{player}"] = float(
                    len(self.recent_advantage_buffers[player])
                    if self.recent_advantage_buffers
                    else 0
                )
                result[f"recent_policy_buffer_p{player}"] = float(
                    len(self.recent_policy_buffers[player])
                    if self.recent_policy_buffers
                    else 0
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        result["traversal_seconds"] = traversal_finished - started
        result["advantage_fit_seconds"] = (
            advantage_fit_finished - advantage_fit_started
        )
        result["policy_fit_seconds"] = finished - advantage_fit_finished
        result["traversal_nodes_per_second"] = self._nodes_this_iteration / max(
            result["traversal_seconds"], 1e-9
        )
        result["seconds"] = finished - started
        result["gpu_memory_allocated_mb"] = (
            torch.cuda.memory_allocated(self.device) / (1024.0**2)
            if self.device.type == "cuda"
            else 0.0
        )
        result["gpu_memory_reserved_mb"] = (
            torch.cuda.memory_reserved(self.device) / (1024.0**2)
            if self.device.type == "cuda"
            else 0.0
        )
        result["gpu_peak_memory_mb"] = (
            torch.cuda.max_memory_allocated(self.device) / (1024.0**2)
            if self.device.type == "cuda"
            else 0.0
        )
        self.last_fitted_iteration = self.iteration
        self.metrics.append(result)
        return result

    def recover_incomplete_fit(
        self,
        *,
        advantage_steps: int,
        policy_steps: int,
        batch_size: int,
    ) -> dict[str, float]:
        """Finish scheduled network fitting after an interrupted iteration.

        Traversal samples are checkpointed even if fitting is interrupted.  A
        fresh cumulative refit is therefore enough to make those samples usable
        without replaying a multi-minute traversal or discarding the run.
        """
        if self.last_fitted_iteration >= self.iteration:
            return {}
        if not self.can_resume_training:
            raise RuntimeError("cannot recover fitting without replay reservoirs")
        if advantage_steps <= 0 or policy_steps < 0 or batch_size <= 0:
            raise ValueError("invalid recovery fit configuration")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        started = time.perf_counter()
        result: dict[str, float] = {
            "recovered_through_iteration": float(self.iteration),
            "previous_last_fitted_iteration": float(self.last_fitted_iteration),
        }
        fit_advantage = self._should_fit_advantage()
        result["recovery_advantage_fit_performed"] = float(fit_advantage)
        with torch.inference_mode(False), torch.enable_grad():
            for player in range(3):
                result[f"recovery_adv_loss_p{player}"] = (
                    self._fit_advantage(player, advantage_steps, batch_size)
                    if fit_advantage
                    else float("nan")
                )
            for player in range(3):
                result[f"recovery_policy_loss_p{player}"] = self._fit_policy(
                    player, policy_steps, batch_size
                )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        result["recovery_fit_seconds"] = time.perf_counter() - started
        self.last_fitted_iteration = self.iteration
        return result

    def train(self, iterations: int, **iteration_kwargs) -> list[dict[str, float]]:
        return [self.train_iteration(**iteration_kwargs) for _ in range(iterations)]

    # ------------------------------------------------------------------
    # Evaluation and inspection
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _policy_from_net(self, state, player: int, net: PolicyNetwork) -> torch.Tensor:
        legal = self.env.legal_actions(state)
        x = self.encode(state, player, legal)
        mask = _legal_mask(legal)
        net_device = next(net.parameters()).device
        net.eval()
        logits = net(x.to(net_device).unsqueeze(0))[0]
        return masked_softmax(logits, mask.to(net_device)).cpu()

    def policy_snapshot(self) -> list[PolicyNetwork]:
        snapshot = [copy.deepcopy(net).cpu().eval() for net in self.policy_nets]
        return snapshot

    def warm_start_policy(
        self, source: str | Path | dict[str, Any]
    ) -> dict[str, Any]:
        """Initialize average-policy nets from a compatible policy checkpoint.

        ``source`` may be either a production policy snapshot, a full CFR
        checkpoint, or an already-loaded payload.  When a legacy encoder is
        expanded by the 15-value tournament suffix, all shared parameters and
        legacy input columns are copied and the new first-layer columns are set
        to zero. Replay reservoirs and advantage nets are intentionally not
        imported: their information-state widths/objectives are incompatible.

        LayerNorm spans the full input width, so expanded-network predictions
        are a close initialization rather than guaranteed bit-identical legacy
        predictions. Exact legacy inference should keep
        ``include_tournament_features=False``.
        """

        if isinstance(source, (str, Path)):
            payload = torch.load(
                Path(source), map_location="cpu", weights_only=False
            )
            source_name = str(source)
        elif isinstance(source, dict):
            payload = source
            source_name = "<payload>"
        else:
            raise TypeError("source must be a path or checkpoint payload")
        if tuple(payload.get("action_names", ())) != tuple(ACTION_NAMES):
            raise ValueError("warm-start action space does not match this engine")

        if payload.get("kind") == "three_player_policy_snapshot":
            source_states = payload.get("policy_nets", [])
            source_input_dim = int(payload["input_dim"])
            source_hidden = int(payload["hidden"])
            source_blocks = int(payload["blocks"])
            source_architecture = str(
                payload.get("network_architecture", "residual_mlp")
            )
            source_max_history = int(
                payload.get("max_history", DEFAULT_MAX_HISTORY)
            )
        elif "policy_nets" in payload and "config" in payload:
            source_states = payload.get("policy_nets", [])
            source_config = dict(payload["config"])
            source_input_dim = int(payload["input_dim"])
            source_hidden = int(source_config["hidden"])
            source_blocks = int(source_config["blocks"])
            source_architecture = str(
                source_config.get("network_architecture", "residual_mlp")
            )
            source_max_history = int(
                source_config.get("max_history", DEFAULT_MAX_HISTORY)
            )
        else:
            raise ValueError("source is not a policy snapshot or CFR checkpoint")

        if len(source_states) != 3:
            raise ValueError("warm-start source must contain three policy networks")
        if source_architecture != self.network_architecture:
            raise ValueError(
                "warm-start network architecture must match the target"
            )
        if source_hidden != self.hidden or source_blocks != self.blocks:
            raise ValueError(
                "warm-start network hidden width/block count must match the target"
            )
        if source_max_history != self.max_history:
            raise ValueError("warm-start max_history must match the target")
        legacy_width = information_state_size(
            self.max_history, include_tournament_features=False
        )
        expanded_width = information_state_size(
            self.max_history, include_tournament_features=True
        )
        if source_input_dim == self.input_dim:
            expanded = False
        elif source_input_dim == legacy_width and self.input_dim == expanded_width:
            expanded = True
        else:
            raise ValueError(
                f"cannot warm-start input width {source_input_dim} into {self.input_dim}"
            )

        for network, source_state in zip(self.policy_nets, source_states):
            target_state = network.state_dict()
            for name, source_value in source_state.items():
                if name not in target_state:
                    raise ValueError(f"warm-start parameter is unknown: {name}")
                source_value = source_value.detach().cpu()
                target_value = target_state[name]
                if source_value.shape == target_value.shape:
                    target_state[name] = source_value.to(target_value.dtype).clone()
                    continue
                if expanded and name in ("input_norm.weight", "input_norm.bias"):
                    if (
                        source_value.ndim == 1
                        and target_value.ndim == 1
                        and source_value.numel() == source_input_dim
                    ):
                        copied = target_value.clone()
                        copied[:source_input_dim] = source_value.to(
                            device=copied.device, dtype=copied.dtype
                        )
                        target_state[name] = copied
                        continue
                if expanded and name == "input_layer.weight":
                    if (
                        source_value.ndim == 2
                        and target_value.ndim == 2
                        and source_value.shape[0] == target_value.shape[0]
                        and source_value.shape[1] == source_input_dim
                    ):
                        copied = target_value.clone()
                        copied[:, :source_input_dim] = source_value.to(
                            device=copied.device, dtype=copied.dtype
                        )
                        copied[:, source_input_dim:] = 0.0
                        target_state[name] = copied
                        continue
                raise ValueError(
                    f"warm-start shape mismatch for {name}: "
                    f"{tuple(source_value.shape)} -> {tuple(target_value.shape)}"
                )
            network.load_state_dict(target_state)
            network.eval()

        # Avoid retaining Adam moments for parameters that were just replaced.
        self.policy_optimizers = [
            torch.optim.AdamW(
                network.parameters(), lr=self.learning_rate, weight_decay=1e-5
            )
            for network in self.policy_nets
        ]
        return {
            "source": source_name,
            "source_input_dim": source_input_dim,
            "target_input_dim": self.input_dim,
            "expanded_legacy_input": expanded,
            "policy_networks_loaded": 3,
        }

    def _evaluate_hero(
        self,
        hero: int,
        games: int,
        opponent_snapshot: Sequence[PolicyNetwork] | None,
        evaluation_env,
        evaluation_rng: random.Random,
    ) -> tuple[list[float], list[int]]:
        payoffs: list[float] = []
        action_counts = [0] * NUM_ACTIONS
        for _ in range(games):
            state = evaluation_env.new_hand()
            while not state.terminal:
                player = int(state.to_act)
                legal = evaluation_env.legal_actions(state)
                if player == hero:
                    probabilities = self.average_policy(state, player)
                    action = self._draw_action(probabilities, evaluation_rng)
                    action_counts[action] += 1
                elif opponent_snapshot is None:
                    action = legal[evaluation_rng.randrange(len(legal))]
                else:
                    probabilities = self._policy_from_net(
                        state, player, opponent_snapshot[player]
                    )
                    action = self._draw_action(probabilities, evaluation_rng)
                state = evaluation_env.step(state, action)
            payoffs.append(float(state.payoffs[hero]) / float(self.env.bb))
        return payoffs, action_counts

    def evaluate_vs_random(self, games_per_player: int = 99) -> dict[str, Any]:
        """Seat each trained policy against two uniform-random opponents."""
        if games_per_player <= 0:
            raise ValueError("games_per_player must be positive")
        all_values: list[float] = []
        counts = [0] * NUM_ACTIONS
        output: dict[str, Any] = {}
        for hero in range(3):
            evaluation_env = type(self.env)(
                stack_size=self.env.stack_size,
                sb=self.env.sb,
                bb=self.env.bb,
                seed=self.seed + 20_000,
            )
            evaluation_rng = random.Random(self.seed + 30_000)
            values, hero_counts = self._evaluate_hero(
                hero, games_per_player, None, evaluation_env, evaluation_rng
            )
            output[f"ev_p{hero}_bb"] = sum(values) / len(values)
            all_values.extend(values)
            counts = [a + b for a, b in zip(counts, hero_counts)]
        mean = sum(all_values) / len(all_values)
        variance = sum((value - mean) ** 2 for value in all_values) / max(
            1, len(all_values) - 1
        )
        output["mean_ev_bb"] = mean
        output["stderr_bb"] = math.sqrt(variance / len(all_values))
        output["action_counts"] = counts
        total_actions = sum(counts)
        frequencies = [count / total_actions for count in counts if count > 0]
        output["action_entropy"] = -sum(p * math.log(p) for p in frequencies)
        return output

    def evaluate_vs_snapshot(
        self, snapshot: Sequence[PolicyNetwork], games_per_player: int = 99
    ) -> dict[str, Any]:
        """Seat one current policy at a time against two frozen policies."""
        if len(snapshot) != 3:
            raise ValueError("snapshot must contain three policy networks")
        all_values: list[float] = []
        output: dict[str, Any] = {}
        for hero in range(3):
            evaluation_env = type(self.env)(
                stack_size=self.env.stack_size,
                sb=self.env.sb,
                bb=self.env.bb,
                seed=self.seed + 20_000,
            )
            evaluation_rng = random.Random(self.seed + 30_000)
            values, _ = self._evaluate_hero(
                hero, games_per_player, snapshot, evaluation_env, evaluation_rng
            )
            output[f"ev_p{hero}_bb"] = sum(values) / len(values)
            all_values.extend(values)
        mean = sum(all_values) / len(all_values)
        variance = sum((value - mean) ** 2 for value in all_values) / max(
            1, len(all_values) - 1
        )
        output["mean_ev_bb"] = mean
        output["stderr_bb"] = math.sqrt(variance / len(all_values))
        return output

    def play_sample_hand(self, use_average_policy: bool = True) -> tuple[Any, list[dict[str, Any]]]:
        """Play one self-play hand and return the terminal state plus a readable log."""
        state = self._sample_env.new_hand()
        log: list[dict[str, Any]] = []
        while not state.terminal:
            player = int(state.to_act)
            legal = self._sample_env.legal_actions(state)
            if use_average_policy:
                probabilities = self.average_policy(state, player)
            else:
                _, probabilities, _ = self.current_strategy(state, player)
            action = self._draw_action(probabilities, self.eval_rng)
            log.append(
                {
                    "street": int(state.street),
                    "player": player,
                    "pot": float(state.pot),
                    "legal": list(legal),
                    "action": action,
                    "probability": float(probabilities[action]),
                }
            )
            state = self._sample_env.step(state, action)
        return state, log

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save(self, path: str | Path, include_buffers: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "version": 1,
            "action_names": tuple(ACTION_NAMES),
            "input_dim": self.input_dim,
            "environment": {
                "stack_size": self.env.stack_size,
                "sb": self.env.sb,
                "bb": self.env.bb,
            },
            "iteration": self.iteration,
            "last_fitted_iteration": self.last_fitted_iteration,
            "can_resume_training": bool(include_buffers),
            "position_cycle": self._position_cycle,
            "metrics": self.metrics,
            "config": {
                "hidden": self.hidden,
                "blocks": self.blocks,
                "network_architecture": self.network_architecture,
                "learning_rate": self.learning_rate,
                "max_history": self.max_history,
                "max_nodes_per_traversal": self.max_nodes_per_traversal,
                "max_depth": self.max_depth,
                "max_strategy_importance": self.max_strategy_importance,
                "exploration": self.exploration,
                "reinitialize_advantage_each_iteration": (
                    self.reinitialize_advantage_each_iteration
                ),
                "advantage_reinitialize_from_iteration": (
                    self.advantage_reinitialize_from_iteration
                ),
                "advantage_fit_every": self.advantage_fit_every,
                "include_tournament_features": self.include_tournament_features,
                "variable_stack_training": self.variable_stack_training,
                "tournament_total_chips": self.tournament_total_chips,
                "heads_up_root_fraction": self.heads_up_root_fraction,
                "continuation_root_fraction": self.continuation_root_fraction,
                "minimum_live_stack": self.minimum_live_stack,
                "root_stack_concentration": self.root_stack_concentration,
                "continuation_capacity": self.continuation_capacity,
                "seed": self.seed,
                "advantage_capacity": self.advantage_buffers[0].capacity,
                "policy_capacity": self.policy_buffers[0].capacity,
                "recent_capacity": self.recent_capacity,
                "recent_window_iterations": self.recent_window_iterations,
                "recent_batch_fraction": self.recent_batch_fraction,
            },
            "advantage_nets": [net.state_dict() for net in self.advantage_nets],
            "policy_nets": [net.state_dict() for net in self.policy_nets],
            "advantage_optimizers": [
                optimizer.state_dict() for optimizer in self.advantage_optimizers
            ],
            "policy_optimizers": [
                optimizer.state_dict() for optimizer in self.policy_optimizers
            ],
            "rng_state": self.rng.getstate(),
            "eval_rng_state": self.eval_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "env_rng_state": self.env.rng.getstate(),
            "env_last_button": self.env._last_button,
            "sample_env_rng_state": self._sample_env.rng.getstate(),
            "sample_env_last_button": self._sample_env._last_button,
            "continuation_stacks": list(self._continuation_stacks),
            "continuation_states_seen": self._continuation_states_seen,
        }
        if torch.cuda.is_available():
            checkpoint["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if include_buffers:
            checkpoint["advantage_buffers"] = [
                buffer.state_dict() for buffer in self.advantage_buffers
            ]
            checkpoint["policy_buffers"] = [
                buffer.state_dict() for buffer in self.policy_buffers
            ]
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
        return path

    @classmethod
    def load(
        cls, path: str | Path, env, *, device: str | torch.device = "cpu"
    ) -> "ThreePlayerNeuralCFR":
        # Always deserialize reservoirs and RNG state on CPU. Model parameters
        # and optimizer state are moved to the requested device below; keeping
        # replay samples on CPU also prevents mixed-device buffers after resume.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("version", -1)) != 1:
            raise ValueError("unsupported checkpoint version")
        if tuple(checkpoint.get("action_names", ())) != tuple(ACTION_NAMES):
            raise ValueError("checkpoint action space does not match this engine")
        environment = checkpoint.get("environment", {})
        for name, current in (
            ("stack_size", env.stack_size),
            ("sb", env.sb),
            ("bb", env.bb),
        ):
            if name in environment and abs(float(environment[name]) - float(current)) > 1e-9:
                raise ValueError(f"checkpoint {name} does not match the supplied environment")
        config = dict(checkpoint["config"])
        config.setdefault("network_architecture", "residual_mlp")
        # Version-1 checkpoints created before tournament support did not store
        # an encoder mode. Infer it from their declared width so old 727-wide
        # snapshots/full checkpoints continue to load exactly as before.
        if "include_tournament_features" not in config:
            max_history = int(config.get("max_history", DEFAULT_MAX_HISTORY))
            checkpoint_width = int(checkpoint.get("input_dim", -1))
            legacy_width = information_state_size(
                max_history, include_tournament_features=False
            )
            expanded_width = information_state_size(
                max_history, include_tournament_features=True
            )
            if checkpoint_width == expanded_width:
                config["include_tournament_features"] = True
            elif checkpoint_width == legacy_width:
                config["include_tournament_features"] = False
        trainer = cls(env, device=device, **config)
        if int(checkpoint.get("input_dim", trainer.input_dim)) != trainer.input_dim:
            raise ValueError("checkpoint encoder dimension does not match current code")
        for net, state in zip(trainer.advantage_nets, checkpoint["advantage_nets"]):
            net.load_state_dict(state)
        for net, state in zip(trainer.policy_nets, checkpoint["policy_nets"]):
            net.load_state_dict(state)
        for optimizer, state in zip(
            trainer.advantage_optimizers, checkpoint["advantage_optimizers"]
        ):
            optimizer.load_state_dict(state)
        for optimizer, state in zip(
            trainer.policy_optimizers, checkpoint["policy_optimizers"]
        ):
            optimizer.load_state_dict(state)
        for optimizer in trainer.advantage_optimizers + trainer.policy_optimizers:
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(trainer.device)
        trainer.iteration = int(checkpoint["iteration"])
        trainer.can_resume_training = bool(
            checkpoint.get("can_resume_training", "advantage_buffers" in checkpoint)
        )
        trainer._position_cycle = int(checkpoint.get("position_cycle", 0))
        trainer.metrics = list(checkpoint.get("metrics", []))
        if "last_fitted_iteration" in checkpoint:
            trainer.last_fitted_iteration = int(checkpoint["last_fitted_iteration"])
        else:
            completed = [
                int(float(row["iteration"]))
                for row in trainer.metrics
                if "iteration" in row
            ]
            trainer.last_fitted_iteration = max(completed, default=0)
        trainer.rng.setstate(checkpoint["rng_state"])
        trainer.eval_rng.setstate(checkpoint["eval_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "torch_cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["torch_cuda_rng_state_all"])
        if "env_rng_state" in checkpoint:
            trainer.env.rng.setstate(checkpoint["env_rng_state"])
            trainer.env._last_button = int(checkpoint["env_last_button"])
        if "sample_env_rng_state" in checkpoint:
            trainer._sample_env.rng.setstate(checkpoint["sample_env_rng_state"])
            trainer._sample_env._last_button = int(checkpoint["sample_env_last_button"])
        trainer._continuation_stacks = [
            tuple(float(value) for value in stacks)
            for stacks in checkpoint.get("continuation_stacks", [])
        ]
        trainer._continuation_states_seen = int(
            checkpoint.get(
                "continuation_states_seen", len(trainer._continuation_stacks)
            )
        )
        if "advantage_buffers" in checkpoint:
            for buffer, state in zip(
                trainer.advantage_buffers, checkpoint["advantage_buffers"]
            ):
                buffer.load_state_dict(state)
        if "policy_buffers" in checkpoint:
            for buffer, state in zip(trainer.policy_buffers, checkpoint["policy_buffers"]):
                buffer.load_state_dict(state)
        return trainer


__all__ = ["RecentWindowBuffer", "ReservoirBuffer", "ThreePlayerNeuralCFR"]
