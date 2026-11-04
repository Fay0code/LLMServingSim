"""Bounded SID execution descriptions and replaceable timing providers.

A request has exactly three SID prediction stages and two beam expansion
rounds. Stage 0 uses one query; stages 1 and 2 each use ``beam_width`` queries.
The prefix is shared across those queries, never multiplied into private KV.
Prefix acquisition is a separate operation, not a fourth SID stage. Stage 0
includes prefix prefill when its acquisition mode is ``recompute``; transferred
or cached prefixes must already be available before stage execution.

These descriptions are not executable Chakra traces. A real ASTRA integration
must lower them to supported, calibrated operations and run the resulting
Chakra graph through ASTRA-Sim. This module imports no simulator or GPU stack.
"""

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, runtime_checkable


SID_STAGES = (0, 1, 2)
BEAM_EXPANSION_ROUNDS = 2


def _integer(value: int, name: str, minimum: int = 0) -> int:
    # Reject bool and float, including integer-looking floats, NaN and infinity.
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _endpoint(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class StageGraph:
    """Immutable logical shape of one batched SID stage, not an ASTRA graph.

    Tuple entries describe requests, not individual beams. ``query_tokens``
    counts all queries for a request. ``prefix_tokens`` counts its shared
    prefix once. ``kv_mode`` records how that prefix is acquired by the caller.
    The caller must finish transfer/cache acquisition before the SID stage.
    Stage 0 includes a separate prefix-prefill operation for ``recompute``.
    """

    stage: int
    batch_size: int
    beam_width: int
    prefix_tokens: tuple[int, ...]
    query_tokens: tuple[int, ...]
    kv_mode: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(self.stage, "stage")
        if self.stage not in SID_STAGES:
            raise ValueError("stage must be 0, 1 or 2")
        _integer(self.batch_size, "batch_size", 1)
        _integer(self.beam_width, "beam_width", 1)
        for name in ("prefix_tokens", "query_tokens", "kv_mode"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != self.batch_size:
                raise ValueError(f"{name} must be a tuple of batch_size entries")
        expected_queries = 1 if self.stage == 0 else self.beam_width
        for prefix, queries, mode in zip(
            self.prefix_tokens, self.query_tokens, self.kv_mode
        ):
            _integer(prefix, "prefix_tokens entry")
            _integer(queries, "query_tokens entry", 1)
            if queries != expected_queries:
                raise ValueError("query_tokens do not match the fixed SID stage shape")
            _endpoint(mode, "kv_mode entry")

    @property
    def peak_branch_tokens(self) -> int:
        """Conservative private KV token reservation *per request* at this stage.

        Reserve ``beam_width * (stage + 1)`` slots: at most one newly processed
        query position per stage per retained beam. Stage 0 evaluates one query
        but reserves width slots for expansion. This is a capacity bound, not
        a claim that stage 0 computes width queries or materializes width KV
        copies. Shared prefix slots, temporary logits and candidate-scoring
        scratch space are excluded. Across all three stages the bound is 3B,
        not B raised to the number of stages.
        """
        return self.beam_width * (self.stage + 1)

    @property
    def prefix_prefill_tokens(self) -> tuple[int, ...]:
        """Prefix tokens recomputed once before the first SID query."""
        return tuple(
            prefix if self.stage == 0 and mode == "recompute" else 0
            for prefix, mode in zip(self.prefix_tokens, self.kv_mode)
        )

    @property
    def expands_beam(self) -> bool:
        return self.stage > 0

    def describe(self) -> dict:
        """Return serializable inspection metadata, never executable ASTRA input."""
        return {
            "format": "openonerec-logical-stage-v1",
            "executable_astra_graph": False,
            "stage": self.stage,
            "batch_size": self.batch_size,
            "beam_width": self.beam_width,
            "prefix_tokens": self.prefix_tokens,
            "query_tokens": self.query_tokens,
            "kv_mode": self.kv_mode,
            "prefix_prefill_tokens": self.prefix_prefill_tokens,
            "prerequisite": "finish_prefix_transfer_or_cache_acquisition",
            "operations": (
                (("shared_prefix_prefill",) if any(self.prefix_prefill_tokens) else ())
                + ("sid_forward",)
                + (("beam_expand_and_prune",) if self.expands_beam else ("first_sid_selection",))
                + (("final_rank",) if self.stage == 2 else ())
            ),
            "peak_branch_tokens_per_request": self.peak_branch_tokens,
            "total_sid_stages": len(SID_STAGES),
            "total_beam_expansion_rounds": BEAM_EXPANSION_ROUNDS,
        }


def build_stage_graph(
    stage: int,
    prefix_tokens: Sequence[int],
    beam_width: int,
    kv_modes: Sequence[str],
) -> StageGraph:
    """Build and validate one nonempty fixed-shape batch from request metadata."""
    _integer(stage, "stage")
    if stage not in SID_STAGES:
        raise ValueError("stage must be 0, 1 or 2")
    _integer(beam_width, "beam_width", 1)
    if isinstance(prefix_tokens, (str, bytes)) or isinstance(kv_modes, (str, bytes)):
        raise ValueError("prefix_tokens and kv_modes must be per-request sequences")
    prefixes = tuple(prefix_tokens)
    modes = tuple(kv_modes)
    queries = (1 if stage == 0 else beam_width,) * len(prefixes)
    return StageGraph(stage, len(prefixes), beam_width, prefixes, queries, modes)


@runtime_checkable
class Timing(Protocol):
    """Nanosecond SID timings including stage-0 recompute, and transport timings."""

    label: str

    def estimate_duration_ns(self, destination: str, graph: StageGraph) -> int:
        """Pure hypothetical estimate; never execute or mutate backend state."""
        ...

    def estimate_transfer_ns(self, source: str, destination: str, bytes_count: int) -> int:
        """Pure hypothetical estimate; never execute or mutate backend state."""
        ...

    def duration_ns(self, destination: str, graph: StageGraph) -> int:
        ...

    def transfer_ns(self, source: str, destination: str, bytes_count: int) -> int:
        ...


@dataclass(frozen=True)
class SyntheticTiming:
    """Uncalibrated deterministic arithmetic for tests/demo, not performance data.

    Prefix-attention work is charged for every query, although prefix KV is
    stored once. Stage-0 recompute is charged once per prefix token, separately from SID
    attention work. This deliberately simple formula is not a prefill model.
    Transport uses ceil(bytes / bytes_per_ns) plus a fixed setup cost.
    """

    stage_base_ns: int = 1000
    query_ns: int = 100
    prefix_token_ns: int = 10
    attention_token_ns: int = 1
    transfer_base_ns: int = 100
    bytes_per_ns: int = 16
    label: str = "synthetic-uncalibrated"

    def __post_init__(self) -> None:
        for name in ("stage_base_ns", "query_ns", "prefix_token_ns", "attention_token_ns", "transfer_base_ns"):
            _integer(getattr(self, name), name)
        _integer(self.bytes_per_ns, "bytes_per_ns", 1)
        if self.label != "synthetic-uncalibrated":
            raise ValueError("synthetic timing must retain its uncalibrated label")

    def duration_ns(self, destination: str, graph: StageGraph) -> int:
        _endpoint(destination, "destination")
        if not isinstance(graph, StageGraph):
            raise TypeError("graph must be a StageGraph")
        return (
            self.stage_base_ns
            + self.prefix_token_ns * sum(graph.prefix_prefill_tokens)
            + self.query_ns * sum(graph.query_tokens)
            + self.attention_token_ns * sum(
                queries * (prefix + graph.stage + 1)
                for prefix, queries in zip(graph.prefix_tokens, graph.query_tokens)
            )
        )

    def transfer_ns(self, source: str, destination: str, bytes_count: int) -> int:
        _endpoint(source, "source")
        _endpoint(destination, "destination")
        _integer(bytes_count, "bytes_count")
        if bytes_count == 0 or source == destination:
            return 0
        return self.transfer_base_ns + (bytes_count + self.bytes_per_ns - 1) // self.bytes_per_ns

    # Synthetic arithmetic is pure, so estimates and execution costs coincide.
    estimate_duration_ns = duration_ns
    estimate_transfer_ns = transfer_ns


class AstraTiming:
    """Isolated-operation callback adapter, not production ASTRA integration.

    ``estimate_stage`` and ``estimate_transfer`` are mandatory, separate pure
    callables used for hypothetical routing/admission. They must never execute
    backend operations or advance backend state. Execution methods invoke only
    the execution callbacks, once each per actual requested operation.

    This adapter returns isolated-operation durations. It does not coordinate
    concurrent ASTRA workloads, model shared-backend contention across calls,
    or couple the backend clock to the serving event clock.

    ``stage_execution(destination, graph)`` must lower the logical description
    into a valid Chakra graph, execute it using the intended ASTRA deployment,
    and return elapsed integer nanoseconds. ``transfer_execution`` must model
    the indicated transfer on that deployment and return integer nanoseconds.
    Neither callback may substitute raw graph-description JSON for executable
    Chakra input. The caller owns calibration, topology and time-unit conversion;
    this adapter validates results but does not certify those integration facts.
    Stage-0 execution must include shared-prefix prefill for requests whose
    mode is ``recompute`` (see ``graph.prefix_prefill_tokens``). Other prefix
    acquisition is completed separately before stage execution.
    """

    label = "astra-callback"

    def __init__(
        self,
        stage_execution: Callable[[str, StageGraph], int],
        transfer_execution: Callable[[str, str, int], int],
        *,
        estimate_stage: Callable[[str, StageGraph], int],
        estimate_transfer: Callable[[str, str, int], int],
    ) -> None:
        for name, callback in (
            ("stage_execution", stage_execution),
            ("transfer_execution", transfer_execution),
            ("estimate_stage", estimate_stage),
            ("estimate_transfer", estimate_transfer),
        ):
            if not callable(callback):
                raise TypeError(f"{name} must be callable")
        self._stage_execution = stage_execution
        self._transfer_execution = transfer_execution
        self._estimate_stage = estimate_stage
        self._estimate_transfer = estimate_transfer

    def estimate_duration_ns(self, destination: str, graph: StageGraph) -> int:
        _endpoint(destination, "destination")
        if not isinstance(graph, StageGraph):
            raise TypeError("graph must be a StageGraph")
        return _integer(self._estimate_stage(destination, graph), "estimated stage duration_ns")

    def estimate_transfer_ns(self, source: str, destination: str, bytes_count: int) -> int:
        _endpoint(source, "source")
        _endpoint(destination, "destination")
        _integer(bytes_count, "bytes_count")
        return _integer(
            self._estimate_transfer(source, destination, bytes_count),
            "estimated transfer duration_ns",
        )

    def duration_ns(self, destination: str, graph: StageGraph) -> int:
        _endpoint(destination, "destination")
        if not isinstance(graph, StageGraph):
            raise TypeError("graph must be a StageGraph")
        return _integer(self._stage_execution(destination, graph), "stage duration_ns")

    def transfer_ns(self, source: str, destination: str, bytes_count: int) -> int:
        _endpoint(source, "source")
        _endpoint(destination, "destination")
        _integer(bytes_count, "bytes_count")
        return _integer(
            self._transfer_execution(source, destination, bytes_count),
            "transfer duration_ns",
        )
