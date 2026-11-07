---
title: OpenOneRec timing profiles
---

# OpenOneRec timing profiles

`serving.openonerec.calibrated_timing.CalibratedTiming` implements the same timing
interface as `SyntheticTiming`, using supplied nanosecond measurements rather
than an arithmetic performance model. It is independent of GPU libraries and
ASTRA-Sim. Loading a profile does not acquire measurements or validate serving
accuracy. The producer must measure the intended model, hardware, software,
precision, parallelism and execution policy and record those facts in provenance.
Use a separate profile for each deployment contract.

```python
from serving.openonerec.calibrated_timing import CalibratedTiming

timing = CalibratedTiming.from_json("measurements.json", interpolate=True)
# Pass timing=timing to Simulator or PolicyEnvironment.
```

## Profile schema

This minimal example contains **fabricated fixture values**, not measurements.
All durations are integer nanoseconds. Each stage row describes the duration of
the entire batch, not a per-request or per-token cost. Prefix recomputation is
included in stage 0 when its mode is `recompute`. Stage durations also include
SID selection, beam expansion/pruning, and final ranking as applicable.

```json
{
  "schema_version": 1,
  "data_kind": "fixture",
  "provenance": {
    "model": "example-model; precision and parallelism must be recorded here",
    "hardware": "example-device; device topology must be recorded here",
    "software": "example-runtime; execution policy must be recorded here",
    "measurement_source": "fabricated documentation example, not measured"
  },
  "stages": [
    {
      "destination": "g0",
      "stage": 0,
      "batch_size": 1,
      "beam_width": 2,
      "prefix_tokens": [128],
      "kv_mode": ["recompute"],
      "duration_ns": 100000
    }
  ],
  "transfers": [
    {
      "source": "cpu",
      "destination": "g0",
      "bytes_count": 65536,
      "duration_ns": 10000
    }
  ]
}
```

Use `data_kind: "measured"` only for actual acquired measurements with an
identifiable source. The timing label is `measured-profile` for that declaration,
and `synthetic-profile-fixture` for fixtures. Neither label certifies end-to-end
accuracy. Do not relabel generated fixture values as measurements.

Stage index is 0, 1 or 2. Beam width and batch size are positive integers;
prefix lengths are nonnegative integers. The ordered prefix and mode arrays
must each have exactly `batch_size` entries. `ready` means prefix acquisition
has completed, whether from local reuse or a finished transfer. `recompute` is
valid only in stage 0. An optional `cached_prefix_tokens` array records the
number of prefix tokens already available for each request. Omit it for zero
cached tokens with `recompute`, or a completely cached prefix with `ready`.
For partial reuse use `ready` with explicit cached counts; the duration includes
prefilling the missing suffix in stage 0. Cached counts must be integral and
between zero and the corresponding prefix length. A `recompute` row must have
zero cached tokens. Exact lookup distinguishes partial and complete cache hits.
Transfer rows describe directed endpoint pairs: reverse
transfers require separate measurements. Zero bytes and same-endpoint transfers
cost zero without requiring a row.

Duplicate shapes, duplicate JSON fields, unknown fields, malformed types,
nonfinite numbers, negative durations and incomplete provenance are rejected.
No samples or endpoints are silently averaged, merged or substituted.

## Coverage and interpolation

Exact lookup is the default. Optional interpolation is bounded bilinear
interpolation over history length and batch size for homogeneous batches only.
All enclosing corners must exist. Beam width, stage, destination and acquisition
mode remain exact categories; no beam or stage interpolation occurs. Partial
cached-token count is also an exact category, so the reader never invents a
cache-saving factor. Completely ready prefixes may interpolate across history
lengths, with each corner's full prefix already cached. Transfer
interpolation is linear on bytes within a measured directed endpoint pair.
Fractional nanoseconds round up, using exact rational arithmetic.

Heterogeneous history or mode vectors require an exact ordered shape match.
There is no averaging of heterogeneous history lengths, extrapolation outside
measured coverage, nearest-neighbor substitution or synthetic fallback. Missing
coverage raises `MissingProfileError`, including during hypothetical policy
estimates. Prepare coverage for every candidate the policy may evaluate, not
only the actions ultimately selected.

Estimate and execution calls perform the same pure lookup. These measurements
do not model dynamic contention, overlap, shared ASTRA backend clocks or network
congestion. The caller must select a profile matching the simulation deployment;
loading provenance strings alone cannot verify deployment compatibility.
