---
title: OpenOneRec policy simulation
---

## Scope

The opt-in `serving.openonerec` frontend models three SID prediction stages,
with one query at stage 0 and a fixed beam width in each of the two following
rounds. It is a policy/lifecycle simulator, not a calibrated OpenOneRec model.
The ordinary `python -m serving` path, its profile catalogs and its recorded
clock baselines are unchanged.

The implementation separates user state and routing/retention policy
(`state.py`, `simulator.py`), serving lifecycle/stage batching (`simulator.py`),
and logical stage descriptions/timing providers (`timing.py`). It does not
import AgentServeSim or claim to reuse that project's implementation. This
policy loop has its own conservative byte accounting instead of integrating
LLMServingSim's block pools or continuous scheduler. A separate serial replay
command reuses its Chakra converter and Controller parser, as described below.

## Run the synthetic example

From the repository root, with Python 3.10 or newer:

```bash
python -m serving.openonerec \
  --workload workloads/openonerec_synthetic.jsonl \
  --config configs/openonerec/synthetic.json \
  --synthetic --output outputs/openonerec_synthetic.json
```

This command needs only the Python standard library. `--synthetic` is required
so invented timing coefficients cannot be mistaken for measured profiles.
All output carries `timing_label: synthetic-uncalibrated`. The example includes
overlapping requests from one user, a cold user, a repeated compatible prefix,
and a changed prefix version. It is not benchmark evidence.

## Requests and arrivals

Each JSONL row requires `request_id`, `user_id`, `model_version`,
`prefix_version`, `prefix_token_ids` and `arrival_time_ns`. IDs and versions
are nonempty strings; tokens and timestamps are nonnegative integers.
`budget_ns` is optional: zero means no budget. Request IDs must be unique.

Ordinary requests enter on their external arrival timestamps. A same-user
request is not a causal successor and can overlap a preceding request.
Compatibility requires the user, model version, prefix version **and actual
prefix token content** to agree. This first implementation matches a whole
prefix exactly, not partial prefixes. Callers must identify all model,
tokenizer, positional-encoding and adapter changes in `model_version`; token
IDs alone do not certify compatibility across those changes.

Only arrivals already processed update each user's exponentially smoothed
arrival-rate estimate. The estimate decays with idle time according to
`half_life_ns`, allowing hot/cold transitions. It influences eviction, not
request priority. A first observation has no estimated arrival rate; future
trace rows are never read by the routing or retention policy.

## Routing and serving

The router jointly enumerates destination and acquisition mode: local valid
copy, another GPU's valid copy, CPU reload, or recomputation. Its score uses
estimated acquisition/service time and a simple queue-length penalty. This
is a configurable research baseline, not an optimal completion predictor.
The acquisition mode is re-evaluated at admission because queued copies may
be evicted or newly published. The destination remains fixed after routing.

Each GPU executes one stage batch at a time. Only ready requests of the same
SID stage can batch together. Allocation proceeds in arrival order; if an
older unallocated request cannot fit, newer unallocated requests cannot jump
it. Already allocated work may continue to free capacity. Ready-stage selection prioritizes estimated remaining-budget slack. Requests
whose age reaches `aging_threshold_ns` (default one second) take precedence in
arrival order, protecting cold or budgetless work from repeated deadline-driven
bypass. Batches remain stage-specific. Arrival rate never promotes a hot request ahead of a cold
one. This policy avoids priority starvation for finite, feasible workloads,
without claiming bounded latency under sustained overload.

`budget_ns` is a soft response budget. Batch events report remaining budget;
responses report deadline misses. No request is dropped. Urgency changes stage selection only; admission remains
FCFS to protect memory-starved older requests. Waiting time is measured from
external arrival to the first stage start, including acquisition and queueing.

## KV ownership and retention

GPU and CPU capacities are explicit bytes. One public prefix copy per key per
tier may be retained across requests. Branch KV is private to a request and
released on response. Public prefix bytes are counted once per retained copy,
not once per beam. Conservative branch reservation is
`3 * beam_width * kv_bytes_per_token`, plus `scratch_bytes` **per request**.
This reserves all three stage positions for every beam; it does not represent
exponential beam growth or measured allocator behavior. The GPU capacities
are the available KV/workspace budgets after weights and other persistent
allocations, not device-total memory. An oversized request fails explicitly.

Recomputed prefixes become public after stage 0 completes; copied prefixes
become public only after transfer completes. A request pins a valid source
while reading it. Multiple requests can hold the same public GPU prefix;
private recomputation workspaces are charged separately until publication.
When capacity is needed, only ready, unpinned copies can be evicted, ordered
by decayed user arrival rate and then recency. `ttl_ns` additionally invalidates
idle unpinned copies on lookup; zero disables TTL. Old versions may remain
until eviction but cannot satisfy a new version's request.

Responses and writebacks are separate events. An optional GPU-to-CPU writeback
reserves destination bytes, pins the source and publishes a valid CPU copy
only on completion. Pending copies are never hits and cannot be evicted.
The simulation drains outstanding writebacks after the last response.
`drained_time_ns` therefore need not equal the last response timestamp.

## Timing and ASTRA-Sim boundary

`StageGraph` is a logical description with prefix lengths, query counts,
acquisition modes, SID stage and beam width. Its JSON description is **not**
an executable Chakra graph. Stage 0 explicitly includes prefix prefill for
`recompute`; transferred/cached prefixes must be ready before SID execution.
The last stage includes final ranking in its operation description.

`SyntheticTiming` uses transparent arithmetic solely for functional tests.
`AstraTiming` accepts separate pure estimate callbacks and execution callbacks.
Routing invokes only estimates, never speculative backend execution. A caller
must lower each admitted stage to a real Chakra graph and execute/measure it,
and provide actual transfer timing through its callbacks. The adapter checks
integer nanosecond outputs but does not implement this lowering itself.

The current callback boundary treats operations as isolated duration queries.
It does not synchronize concurrently outstanding stage, transfer and writeback
events with an ASTRA resource timeline. Transfers can overlap in this policy
model without shared-link contention. No ASTRA build is bundled or invoked
by the example. Full concurrent ASTRA integration, topology-aware read/write
contention, calibrated history/batch/beam/top-k/reorder profiles, physical KV
layout validation, and capacity comparison against OpenOneRec remain required
before quantitative performance or speedup claims.

## Actual Chakra export and serial backend replay

A separate infrastructure check now reuses the existing `TraceData`,
`GraphGenerator`, repository Chakra converter, and `Controller` report parser.
With the standard simulator dependencies and the repository's Chakra fork
installed, export three executable rank-zero `.et` protobuf files:

```bash
python -m serving.openonerec.astra_replay --synthetic \
  --prefix-tokens 8 --beam-width 4 --output-dir outputs/openonerec_replay
```

Each graph contains one aggregate SID-stage compute node with an explicitly
synthetic duration. The three separate graphs retain serial stage order;
these are not calibrated per-kernel graphs. `replay.json` preserves stage shape,
synthetic duration and generated file locations. This export was validated by
decoding the actual Chakra protobuf nodes, not by treating JSON as a graph.

To execute the serial replay using a **previously built** analytical backend,
append these options to the same command:

```bash
--binary astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra \
--system astra-sim/inputs/system/system.json \
--network astra-sim/inputs/network/network.yml \
--memory astra-sim/inputs/memory/no_memory_expansion.json
```

The runner requires `npus_count: [1]`, duration replay with roofline disabled,
and three ordered rank-zero completion reports. It sends successive workload
paths and `exit` through ASTRA's existing stdin protocol, parses stdout through
`Controller`, and applies a wall-clock timeout. A failed build or missing binary
is an error, never a synthetic fallback. The exported costs use the repository
converter/backend's nanosecond convention despite the protobuf field's legacy
`duration_micros` name.

This is **serial infrastructure replay**, not integration of the policy event
loop with the existing continuous scheduler. It has no transfer nodes or
shared-link contention, and no measured OpenOneRec operation breakdown.
Export/converter and parser/IPC-contract tests are available; mocked IPC tests
do not validate execution of the C++ backend. C++ execution requires the normal
ASTRA build toolchain, including CMake, protoc and protobuf development headers.

The optional real-backend smoke test executes two three-stage configurations
against the built C++ executable and checks every stage duration against its
explicit synthetic input. Enable it with:

```bash
OPENONEREC_ASTRA_BINARY="$PWD/astra-sim/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra" \
  python -m unittest discover -s serving/openonerec/tests -v
```

The normal simulator build needs `cmake`, `protobuf-compiler` and
`libprotobuf-dev` (Debian package names), plus the recursive repository
submodules and Python dependencies described in the installation guide.
Successful synthetic replay proves the graph conversion, IPC, backend execution
and report parser work together; it does not establish measured OpenOneRec
latency or concurrent transfer behavior.

## Functional regression tests

The standard-library tests cover request compatibility, stage shapes, asynchronous
publication, source recency, pinning, memory bounds, urgency with aging, causal
arrival observations and estimator/execution separation. A deterministic
randomized test covers 5,000 requests across 100 small capacity-limited runs.
Run from the repository root:

```bash
python -m unittest discover -s serving/openonerec/tests -v
```

Three additional graph/parser integration-contract tests run when the simulator and Chakra
dependencies are installed; otherwise they are explicitly skipped. They decode
real generated graphs and test the actual Controller parser with mocked IPC.
The separate real-C++ smoke test is enabled only by the environment variable
above; without it, that test is explicitly skipped.

These tests validate policy invariants using synthetic timings. They do not
replace the ordinary simulator's ASTRA regression suite or validate measured
OpenOneRec latency.

## Output

The output JSON contains responses, route/acquire/publish/batch/writeback
events, drained time, per-tier peak allocated bytes and eviction count.
Responses include arrival, response and waiting times, selected destination
and actual admission mode, and deadline-miss status. Batch events include
stage and remaining response budget. These values make lifecycle invariants
inspectable; synthetic nanoseconds are not empirical latency predictions.
