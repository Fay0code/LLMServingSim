---
title: OpenOneRec policy search
---

# Offline routing and batching search

`serving.openonerec.search` adds exhaustive finite-grid selection on top of the
immutable policy environment and reproducible evaluator. This is an opt-in
research frontend, not a replacement for the ordinary serving scheduler and not
an assertion of calibrated OpenOneRec performance.

## Command line

Prepare two nonempty JSONL request traces in the existing `RecRequest` format.
Use distinct request IDs between tuning and holdout. Each split starts cold;
cache state does not transfer. The caller chooses the split, for example earlier
sessions for tuning and later sessions for holdout. The implementation rejects
ID overlap but cannot detect semantic duplicate samples or guarantee independent
users. Avoid inspecting holdout results when choosing a new grid.

```bash
python -m serving.openonerec.search \
  --synthetic --config configs/openonerec/search.json \
  --grid grid.json --tuning-workload tuning.jsonl \
  --holdout-workload holdout.jsonl --seeds 0 1 \
  --objective latency_p99_ns --output outputs/search.json
```

The config path above is a user-created input, not a bundled hardware profile.
An example config is:

```json
{"gpu_capacities":{"gpu0":1048576,"gpu1":1048576},"max_batch_size":8,"synthetic_timing":{}}
```

Example grid:

```json
{"queue_weight":[1,2],"locality_bonus":[0,4],"batch_size":[1,4,8],"acquisition":["reuse","recompute"]}
```

Every combination is evaluated. Queue/locality weights must be finite and
nonnegative; batch size must be positive. The CLI rejects sizes above the
configured request-count cap. Runtime batch construction also respects visible
row and token limits and includes the required oldest ready request. Stage and
fixed-step execution group equal stages; wave execution may group mixed stages.

## Select profile-backed timing

Replace `--synthetic` with `--timing-profile measurements.json` to use
[`CalibratedTiming`](./openonerec-profiles.md). The options are mutually exclusive;
remove `synthetic_timing` from the environment config for profile-backed runs.
Exact lookup is the default. Add `--interpolate` only when bounded interpolation
is appropriate; it is rejected with synthetic timing. Missing files, malformed
profiles or uncovered executed shapes fail the search without synthetic fallback.

The selected file is read once before candidate evaluation. Every rollout uses a
fresh timing provider from that snapshot. Each evaluation's timing provenance
records the raw file SHA256, profile schema version, interpolation choice,
producer provenance and declared `data_kind`. Measured declarations carry
`measured-profile`; fixtures carry `synthetic-profile-fixture`. These are source
declarations, not independently verified measurements. Whitespace changes alter
the raw file hash; moving unchanged bytes to another path does not.

## Policy and objective contract

The deterministic policy sees only current immutable observations and legal
actions. It routes the oldest currently legal request to minimize
`queue_weight * visible_queue_count - locality_bonus * compatible_local_hit`,
with device name breaking ties. A local hit requires exact prefix compatibility
and a valid, ready copy. It acquires prefixes before launching batches, then
forms ready batches up to the selected size. It never waits deliberately to
fill a batch and cannot see future arrivals. Acquisition preference options are
`reuse` (local, remote, CPU, recompute), `local_only` (local, recompute, then legal
fallbacks), and `recompute` (recompute, then legal reuse fallbacks). These are
preferences, not prohibitions: memory pressure can make a fallback necessary.

Supported objectives are `latency_p99_ns` (default), `latency_p50_ns`,
`slo_success_fraction`, and `goodput_requests_per_second`. Latency is minimized;
SLO success and goodput are maximized. A score is the arithmetic mean of the
per-seed metric, **not a percentile pooled across seeds**. Undefined objectives,
invalid actions, backend errors and step-limit failures abort the search rather
than silently dropping candidates. SLO/goodput require positive-budget requests.
The policies are deterministic, so changing seeds alone does not add statistical
variation. The evaluator still records seeds for reproducibility and future
stochastic policy extensions.

Selection uses tuning scores only, with canonical parameter JSON breaking ties.
Only the chosen policy runs on holdout. The result contains all candidate
parameters, scores, decision traces, metric definitions, workload/config/policy/
timing hashes, and the selected holdout evaluation. Input metadata is declared
provenance, not proof of code identity or calibration. A finite grid gives no
global-optimum guarantee; synthetic scores must not be reported as measured
hardware gains.

## Python API and verification

`grid_search(environment_factory, tuning_requests, holdout_requests, grid=...,
seeds=..., config=..., timing_metadata=...)` supports the existing replaceable
timing providers. Each factory call must return a fresh environment and fresh
backend state. No execution callback runs merely to rank a hypothetical route;
actual admitted rollout batches execute through the environment's timing API.
Use `parameter_grid` and `SearchPolicy` independently for policy experiments.

```bash
python -m unittest serving.openonerec.tests.test_search -v
```
